"""
Standalone repro captured via capture_hook.
Label: XLNetLMHeadModel
Pattern hash: a41c29446935
Shape hash: 6c714bb3
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
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "f32[32000, 1024]", arg2_1: "f32[1024, 16, 64]", arg3_1: "f32[1024, 16, 64]", arg4_1: "f32[1024, 16, 64]", arg5_1: "f32[1024, 16, 64]", arg6_1: "f32[16, 64]", arg7_1: "f32[16, 64]", arg8_1: "f32[1024, 16, 64]", arg9_1: "f32[1024]", arg10_1: "f32[1024]", arg11_1: "f32[4096]", arg12_1: "f32[4096, 1024]", arg13_1: "f32[1024]", arg14_1: "f32[1024, 4096]", arg15_1: "f32[1024]", arg16_1: "f32[1024]", arg17_1: "f32[1024, 16, 64]", arg18_1: "f32[1024, 16, 64]", arg19_1: "f32[1024, 16, 64]", arg20_1: "f32[1024, 16, 64]", arg21_1: "f32[16, 64]", arg22_1: "f32[16, 64]", arg23_1: "f32[1024, 16, 64]", arg24_1: "f32[1024]", arg25_1: "f32[1024]", arg26_1: "f32[4096]", arg27_1: "f32[4096, 1024]", arg28_1: "f32[1024]", arg29_1: "f32[1024, 4096]", arg30_1: "f32[1024]", arg31_1: "f32[1024]", arg32_1: "f32[1024, 16, 64]", arg33_1: "f32[1024, 16, 64]", arg34_1: "f32[1024, 16, 64]", arg35_1: "f32[1024, 16, 64]", arg36_1: "f32[16, 64]", arg37_1: "f32[16, 64]", arg38_1: "f32[1024, 16, 64]", arg39_1: "f32[1024]", arg40_1: "f32[1024]", arg41_1: "f32[4096]", arg42_1: "f32[4096, 1024]", arg43_1: "f32[1024]", arg44_1: "f32[1024, 4096]", arg45_1: "f32[1024]", arg46_1: "f32[1024]", arg47_1: "f32[1024, 16, 64]", arg48_1: "f32[1024, 16, 64]", arg49_1: "f32[1024, 16, 64]", arg50_1: "f32[1024, 16, 64]", arg51_1: "f32[16, 64]", arg52_1: "f32[16, 64]", arg53_1: "f32[1024, 16, 64]", arg54_1: "f32[1024]", arg55_1: "f32[1024]", arg56_1: "f32[4096]", arg57_1: "f32[4096, 1024]", arg58_1: "f32[1024]", arg59_1: "f32[1024, 4096]", arg60_1: "f32[1024]", arg61_1: "f32[1024]", arg62_1: "f32[1024, 16, 64]", arg63_1: "f32[1024, 16, 64]", arg64_1: "f32[1024, 16, 64]", arg65_1: "f32[1024, 16, 64]", arg66_1: "f32[16, 64]", arg67_1: "f32[16, 64]", arg68_1: "f32[1024, 16, 64]", arg69_1: "f32[1024]", arg70_1: "f32[1024]", arg71_1: "f32[4096]", arg72_1: "f32[4096, 1024]", arg73_1: "f32[1024]", arg74_1: "f32[1024, 4096]", arg75_1: "f32[1024]", arg76_1: "f32[1024]", arg77_1: "f32[1024, 16, 64]", arg78_1: "f32[1024, 16, 64]", arg79_1: "f32[1024, 16, 64]", arg80_1: "f32[1024, 16, 64]", arg81_1: "f32[16, 64]", arg82_1: "f32[16, 64]", arg83_1: "f32[1024, 16, 64]", arg84_1: "f32[1024]", arg85_1: "f32[1024]", arg86_1: "f32[4096]", arg87_1: "f32[4096, 1024]", arg88_1: "f32[1024]", arg89_1: "f32[1024, 4096]", arg90_1: "f32[1024]", arg91_1: "f32[1024]", arg92_1: "f32[1024, 16, 64]", arg93_1: "f32[1024, 16, 64]", arg94_1: "f32[1024, 16, 64]", arg95_1: "f32[1024, 16, 64]", arg96_1: "f32[16, 64]", arg97_1: "f32[16, 64]", arg98_1: "f32[1024, 16, 64]", arg99_1: "f32[1024]", arg100_1: "f32[1024]", arg101_1: "f32[4096]", arg102_1: "f32[4096, 1024]", arg103_1: "f32[1024]", arg104_1: "f32[1024, 4096]", arg105_1: "f32[1024]", arg106_1: "f32[1024]", arg107_1: "f32[1024, 16, 64]", arg108_1: "f32[1024, 16, 64]", arg109_1: "f32[1024, 16, 64]", arg110_1: "f32[1024, 16, 64]", arg111_1: "f32[16, 64]", arg112_1: "f32[16, 64]", arg113_1: "f32[1024, 16, 64]", arg114_1: "f32[1024]", arg115_1: "f32[1024]", arg116_1: "f32[4096]", arg117_1: "f32[4096, 1024]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 4096]", arg120_1: "f32[1024]", arg121_1: "f32[1024]", arg122_1: "f32[1024, 16, 64]", arg123_1: "f32[1024, 16, 64]", arg124_1: "f32[1024, 16, 64]", arg125_1: "f32[1024, 16, 64]", arg126_1: "f32[16, 64]", arg127_1: "f32[16, 64]", arg128_1: "f32[1024, 16, 64]", arg129_1: "f32[1024]", arg130_1: "f32[1024]", arg131_1: "f32[4096]", arg132_1: "f32[4096, 1024]", arg133_1: "f32[1024]", arg134_1: "f32[1024, 4096]", arg135_1: "f32[1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 16, 64]", arg138_1: "f32[1024, 16, 64]", arg139_1: "f32[1024, 16, 64]", arg140_1: "f32[1024, 16, 64]", arg141_1: "f32[16, 64]", arg142_1: "f32[16, 64]", arg143_1: "f32[1024, 16, 64]", arg144_1: "f32[1024]", arg145_1: "f32[1024]", arg146_1: "f32[4096]", arg147_1: "f32[4096, 1024]", arg148_1: "f32[1024]", arg149_1: "f32[1024, 4096]", arg150_1: "f32[1024]", arg151_1: "f32[1024]", arg152_1: "f32[1024, 16, 64]", arg153_1: "f32[1024, 16, 64]", arg154_1: "f32[1024, 16, 64]", arg155_1: "f32[1024, 16, 64]", arg156_1: "f32[16, 64]", arg157_1: "f32[16, 64]", arg158_1: "f32[1024, 16, 64]", arg159_1: "f32[1024]", arg160_1: "f32[1024]", arg161_1: "f32[4096]", arg162_1: "f32[4096, 1024]", arg163_1: "f32[1024]", arg164_1: "f32[1024, 4096]", arg165_1: "f32[1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024, 16, 64]", arg168_1: "f32[1024, 16, 64]", arg169_1: "f32[1024, 16, 64]", arg170_1: "f32[1024, 16, 64]", arg171_1: "f32[16, 64]", arg172_1: "f32[16, 64]", arg173_1: "f32[1024, 16, 64]", arg174_1: "f32[1024]", arg175_1: "f32[1024]", arg176_1: "f32[4096]", arg177_1: "f32[4096, 1024]", arg178_1: "f32[1024]", arg179_1: "f32[1024, 4096]", arg180_1: "f32[1024]", arg181_1: "f32[1024]", arg182_1: "f32[1024, 16, 64]", arg183_1: "f32[1024, 16, 64]", arg184_1: "f32[1024, 16, 64]", arg185_1: "f32[1024, 16, 64]", arg186_1: "f32[16, 64]", arg187_1: "f32[16, 64]", arg188_1: "f32[1024, 16, 64]", arg189_1: "f32[1024]", arg190_1: "f32[1024]", arg191_1: "f32[4096]", arg192_1: "f32[4096, 1024]", arg193_1: "f32[1024]", arg194_1: "f32[1024, 4096]", arg195_1: "f32[1024]", arg196_1: "f32[1024]", arg197_1: "f32[1024, 16, 64]", arg198_1: "f32[1024, 16, 64]", arg199_1: "f32[1024, 16, 64]", arg200_1: "f32[1024, 16, 64]", arg201_1: "f32[16, 64]", arg202_1: "f32[16, 64]", arg203_1: "f32[1024, 16, 64]", arg204_1: "f32[1024]", arg205_1: "f32[1024]", arg206_1: "f32[4096]", arg207_1: "f32[4096, 1024]", arg208_1: "f32[1024]", arg209_1: "f32[1024, 4096]", arg210_1: "f32[1024]", arg211_1: "f32[1024]", arg212_1: "f32[1024, 16, 64]", arg213_1: "f32[1024, 16, 64]", arg214_1: "f32[1024, 16, 64]", arg215_1: "f32[1024, 16, 64]", arg216_1: "f32[16, 64]", arg217_1: "f32[16, 64]", arg218_1: "f32[1024, 16, 64]", arg219_1: "f32[1024]", arg220_1: "f32[1024]", arg221_1: "f32[4096]", arg222_1: "f32[4096, 1024]", arg223_1: "f32[1024]", arg224_1: "f32[1024, 4096]", arg225_1: "f32[1024]", arg226_1: "f32[1024]", arg227_1: "f32[1024, 16, 64]", arg228_1: "f32[1024, 16, 64]", arg229_1: "f32[1024, 16, 64]", arg230_1: "f32[1024, 16, 64]", arg231_1: "f32[16, 64]", arg232_1: "f32[16, 64]", arg233_1: "f32[1024, 16, 64]", arg234_1: "f32[1024]", arg235_1: "f32[1024]", arg236_1: "f32[4096]", arg237_1: "f32[4096, 1024]", arg238_1: "f32[1024]", arg239_1: "f32[1024, 4096]", arg240_1: "f32[1024]", arg241_1: "f32[1024]", arg242_1: "f32[1024, 16, 64]", arg243_1: "f32[1024, 16, 64]", arg244_1: "f32[1024, 16, 64]", arg245_1: "f32[1024, 16, 64]", arg246_1: "f32[16, 64]", arg247_1: "f32[16, 64]", arg248_1: "f32[1024, 16, 64]", arg249_1: "f32[1024]", arg250_1: "f32[1024]", arg251_1: "f32[4096]", arg252_1: "f32[4096, 1024]", arg253_1: "f32[1024]", arg254_1: "f32[1024, 4096]", arg255_1: "f32[1024]", arg256_1: "f32[1024]", arg257_1: "f32[1024, 16, 64]", arg258_1: "f32[1024, 16, 64]", arg259_1: "f32[1024, 16, 64]", arg260_1: "f32[1024, 16, 64]", arg261_1: "f32[16, 64]", arg262_1: "f32[16, 64]", arg263_1: "f32[1024, 16, 64]", arg264_1: "f32[1024]", arg265_1: "f32[1024]", arg266_1: "f32[4096]", arg267_1: "f32[4096, 1024]", arg268_1: "f32[1024]", arg269_1: "f32[1024, 4096]", arg270_1: "f32[1024]", arg271_1: "f32[1024]", arg272_1: "f32[1024, 16, 64]", arg273_1: "f32[1024, 16, 64]", arg274_1: "f32[1024, 16, 64]", arg275_1: "f32[1024, 16, 64]", arg276_1: "f32[16, 64]", arg277_1: "f32[16, 64]", arg278_1: "f32[1024, 16, 64]", arg279_1: "f32[1024]", arg280_1: "f32[1024]", arg281_1: "f32[4096]", arg282_1: "f32[4096, 1024]", arg283_1: "f32[1024]", arg284_1: "f32[1024, 4096]", arg285_1: "f32[1024]", arg286_1: "f32[1024]", arg287_1: "f32[1024, 16, 64]", arg288_1: "f32[1024, 16, 64]", arg289_1: "f32[1024, 16, 64]", arg290_1: "f32[1024, 16, 64]", arg291_1: "f32[16, 64]", arg292_1: "f32[16, 64]", arg293_1: "f32[1024, 16, 64]", arg294_1: "f32[1024]", arg295_1: "f32[1024]", arg296_1: "f32[4096]", arg297_1: "f32[4096, 1024]", arg298_1: "f32[1024]", arg299_1: "f32[1024, 4096]", arg300_1: "f32[1024]", arg301_1: "f32[1024]", arg302_1: "f32[1024, 16, 64]", arg303_1: "f32[1024, 16, 64]", arg304_1: "f32[1024, 16, 64]", arg305_1: "f32[1024, 16, 64]", arg306_1: "f32[16, 64]", arg307_1: "f32[16, 64]", arg308_1: "f32[1024, 16, 64]", arg309_1: "f32[1024]", arg310_1: "f32[1024]", arg311_1: "f32[4096]", arg312_1: "f32[4096, 1024]", arg313_1: "f32[1024]", arg314_1: "f32[1024, 4096]", arg315_1: "f32[1024]", arg316_1: "f32[1024]", arg317_1: "f32[1024, 16, 64]", arg318_1: "f32[1024, 16, 64]", arg319_1: "f32[1024, 16, 64]", arg320_1: "f32[1024, 16, 64]", arg321_1: "f32[16, 64]", arg322_1: "f32[16, 64]", arg323_1: "f32[1024, 16, 64]", arg324_1: "f32[1024]", arg325_1: "f32[1024]", arg326_1: "f32[4096]", arg327_1: "f32[4096, 1024]", arg328_1: "f32[1024]", arg329_1: "f32[1024, 4096]", arg330_1: "f32[1024]", arg331_1: "f32[1024]", arg332_1: "f32[1024, 16, 64]", arg333_1: "f32[1024, 16, 64]", arg334_1: "f32[1024, 16, 64]", arg335_1: "f32[1024, 16, 64]", arg336_1: "f32[16, 64]", arg337_1: "f32[16, 64]", arg338_1: "f32[1024, 16, 64]", arg339_1: "f32[1024]", arg340_1: "f32[1024]", arg341_1: "f32[4096]", arg342_1: "f32[4096, 1024]", arg343_1: "f32[1024]", arg344_1: "f32[1024, 4096]", arg345_1: "f32[1024]", arg346_1: "f32[1024]", arg347_1: "f32[1024, 16, 64]", arg348_1: "f32[1024, 16, 64]", arg349_1: "f32[1024, 16, 64]", arg350_1: "f32[1024, 16, 64]", arg351_1: "f32[16, 64]", arg352_1: "f32[16, 64]", arg353_1: "f32[1024, 16, 64]", arg354_1: "f32[1024]", arg355_1: "f32[1024]", arg356_1: "f32[4096]", arg357_1: "f32[4096, 1024]", arg358_1: "f32[1024]", arg359_1: "f32[1024, 4096]", arg360_1: "f32[1024]", arg361_1: "f32[1024]", arg362_1: "f32[32000]", arg363_1: "i64[16, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833, _shape_param_834, _shape_param_835, _shape_param_836, _shape_param_837, _shape_param_838, _shape_param_839, _shape_param_840, _shape_param_841, _shape_param_842, _shape_param_843, _shape_param_844, _shape_param_845, _shape_param_846, _shape_param_847, _shape_param_848, _shape_param_849, _shape_param_850, _shape_param_851, _shape_param_852, _shape_param_853, _shape_param_854, _shape_param_855, _shape_param_856, _shape_param_857, _shape_param_858, _shape_param_859, _shape_param_860, _shape_param_861, _shape_param_862, _shape_param_863, _shape_param_864, _shape_param_865, _shape_param_866, _shape_param_867, _shape_param_868, _shape_param_869, _shape_param_870, _shape_param_871, _shape_param_872, _shape_param_873, _shape_param_874, _shape_param_875, _shape_param_876, _shape_param_877, _shape_param_878, _shape_param_879, _shape_param_880, _shape_param_881, _shape_param_882, _shape_param_883, _shape_param_884, _shape_param_885, _shape_param_886, _shape_param_887, _shape_param_888, _shape_param_889, _shape_param_890, _shape_param_891, _shape_param_892, _shape_param_893, _shape_param_894, _shape_param_895, _shape_param_896, _shape_param_897, _shape_param_898, _shape_param_899, _shape_param_900, _shape_param_901, _shape_param_902, _shape_param_903, _shape_param_904, _shape_param_905, _shape_param_906, _shape_param_907, _shape_param_908, _shape_param_909, _shape_param_910, _shape_param_911, _shape_param_912, _shape_param_913, _shape_param_914, _shape_param_915, _shape_param_916, _shape_param_917, _shape_param_918, _shape_param_919, _shape_param_920, _shape_param_921, _shape_param_922, _shape_param_923, _shape_param_924, _shape_param_925, _shape_param_926, _shape_param_927, _shape_param_928, _shape_param_929, _shape_param_930, _shape_param_931, _shape_param_932, _shape_param_933, _shape_param_934, _shape_param_935, _shape_param_936, _shape_param_937, _shape_param_938, _shape_param_939, _shape_param_940, _shape_param_941, _shape_param_942, _shape_param_943):
        # No stacktrace found for following nodes
        permute: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding: "f32[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone)
        inductor_seeds: "i64[99]" = torch.ops.prims.inductor_seeds.default(99, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_1: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None
        iota: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[512]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        div: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type, 1024);  convert_element_type = None
        pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul_2: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        iota_1: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_1: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        convert_element_type_2: "bf16[1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.bfloat16);  convert_element_type_1 = None
        convert_element_type_3: "bf16[512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        unsqueeze: "bf16[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        permute_1: "bf16[1024, 1]" = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None
        unsqueeze_1: "bf16[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        permute_2: "bf16[1, 512]" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_3: "bf16[1024, 512]" = torch.ops.aten.mul.Tensor(permute_1, permute_2);  permute_1 = permute_2 = None
        sin: "bf16[1024, 512]" = torch.ops.aten.sin.default(mul_3)
        cos: "bf16[1024, 512]" = torch.ops.aten.cos.default(mul_3);  mul_3 = None
        cat: "bf16[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        unsqueeze_2: "bf16[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        expand: "bf16[1024, 16, 1024]" = torch.ops.aten.expand.default(unsqueeze_2, _shape_param_1);  unsqueeze_2 = _shape_param_1 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[1024, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_2, inductor_lookup_seed_1, 'rand');  _shape_param_2 = inductor_lookup_seed_1 = None
        convert_element_type_4: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[1024, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_4, 0.1);  convert_element_type_4 = None
        mul_4: "bf16[1024, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_1, expand);  gt_1 = expand = None
        mul_5: "bf16[1024, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None
        convert_element_type_5: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16)
        convert_element_type_6: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.bfloat16);  arg2_1 = None
        unsqueeze_3: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 3);  convert_element_type_5 = None
        unsqueeze_4: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
        unsqueeze_5: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, 3);  convert_element_type_6 = None
        unsqueeze_6: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
        view: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_3);  unsqueeze_4 = _shape_param_3 = None
        view_1: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_6, _shape_param_4);  unsqueeze_6 = _shape_param_4 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view, 0)
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_1, 0)
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze_1 = None
        unsqueeze_7: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_7, _shape_param_5);  unsqueeze_7 = _shape_param_5 = None
        permute_3: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_6);  permute_3 = _shape_param_6 = None
        convert_element_type_7: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        unsqueeze_8: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_7, 3);  convert_element_type_7 = None
        unsqueeze_9: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 4);  unsqueeze_8 = None
        view_4: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_9, _shape_param_7);  unsqueeze_9 = _shape_param_7 = None
        squeeze_2: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_4, 0)
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_2);  squeeze_2 = None
        unsqueeze_10: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_1, 0);  mm_1 = None
        view_5: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_10, _shape_param_8);  unsqueeze_10 = _shape_param_8 = None
        permute_4: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_5, [0, 1, 3, 4, 2]);  view_5 = None
        view_6: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_4, _shape_param_9);  permute_4 = _shape_param_9 = None
        convert_element_type_8: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        unsqueeze_11: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_8, 3);  convert_element_type_8 = None
        unsqueeze_12: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_11, 4);  unsqueeze_11 = None
        view_7: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_12, _shape_param_10);  unsqueeze_12 = _shape_param_10 = None
        squeeze_3: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_7, 0)
        mm_2: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_3);  squeeze = squeeze_3 = None
        unsqueeze_13: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_2, 0);  mm_2 = None
        view_8: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_13, _shape_param_11);  unsqueeze_13 = _shape_param_11 = None
        permute_5: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_8, [0, 1, 3, 4, 2]);  view_8 = None
        view_9: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_5, _shape_param_12);  permute_5 = _shape_param_12 = None
        convert_element_type_9: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        convert_element_type_10: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg5_1, torch.bfloat16);  arg5_1 = None
        unsqueeze_14: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_9, 3);  convert_element_type_9 = None
        unsqueeze_15: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, 4);  unsqueeze_14 = None
        unsqueeze_16: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_10, 3);  convert_element_type_10 = None
        unsqueeze_17: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 4);  unsqueeze_16 = None
        view_10: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_15, _shape_param_13);  unsqueeze_15 = _shape_param_13 = None
        view_11: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_17, _shape_param_14);  unsqueeze_17 = _shape_param_14 = None
        squeeze_4: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_10, 0)
        squeeze_5: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_11, 0);  view_11 = None
        mm_3: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_5);  squeeze_5 = None
        unsqueeze_18: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_3, 0);  mm_3 = None
        view_12: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_18, _shape_param_15);  unsqueeze_18 = _shape_param_15 = None
        permute_6: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_12, [0, 1, 3, 4, 2]);  view_12 = None
        view_13: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_6, _shape_param_16);  permute_6 = _shape_param_16 = None
        add: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg6_1);  arg6_1 = None
        convert_element_type_11: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        unsqueeze_19: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_11, 4);  convert_element_type_11 = None
        permute_7: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_19, [1, 2, 0, 4, 3]);  unsqueeze_19 = None
        unsqueeze_20: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_6, 4);  view_6 = None
        permute_8: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_20, [1, 2, 4, 0, 3]);  unsqueeze_20 = None
        permute_9: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_7, [0, 1, 2, 4, 3]);  permute_7 = None
        view_14: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_9, _shape_param_17);  permute_9 = _shape_param_17 = None
        permute_10: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_8, [0, 1, 4, 3, 2]);  permute_8 = None
        view_15: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_10, _shape_param_18);  permute_10 = _shape_param_18 = None
        bmm: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_14, view_15)
        view_16: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm, _shape_param_19);  bmm = _shape_param_19 = None
        permute_11: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_16, [0, 1, 2, 4, 3]);  view_16 = None
        view_17: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_11, _shape_param_20);  permute_11 = _shape_param_20 = None
        add_1: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg7_1);  view_3 = arg7_1 = None
        convert_element_type_12: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        unsqueeze_21: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, 4);  convert_element_type_12 = None
        permute_12: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_21, [1, 2, 0, 4, 3]);  unsqueeze_21 = None
        unsqueeze_22: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_13, 4);  view_13 = None
        permute_13: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_22, [1, 2, 4, 0, 3]);  unsqueeze_22 = None
        permute_14: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_12, [0, 1, 2, 4, 3]);  permute_12 = None
        view_18: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_14, _shape_param_21);  permute_14 = _shape_param_21 = None
        permute_15: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_13, [0, 1, 4, 3, 2]);  permute_13 = None
        view_19: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_15, _shape_param_22);  permute_15 = _shape_param_22 = None
        bmm_1: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_18, view_19)
        view_20: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_1, _shape_param_23);  bmm_1 = _shape_param_23 = None
        permute_16: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_20, [0, 1, 2, 4, 3]);  view_20 = None
        view_21: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_16, _shape_param_24);  permute_16 = _shape_param_24 = None
        view_22: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_21, _shape_param_25);  view_21 = _shape_param_25 = None
        slice_1: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_22, 2, 1, 9223372036854775807);  view_22 = None
        view_23: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_1, _shape_param_26);  slice_1 = _shape_param_26 = None
        iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_23, [None, None, None, iota_2]);  view_23 = None
        add_2: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_17, index);  view_17 = index = None
        add_3: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_2, 0);  add_2 = None
        mul_6: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_3, 0.125)
        convert_element_type_13: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_6, torch.float32);  mul_6 = None
        convert_element_type_14: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        mul_7: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1);  convert_element_type_14 = None
        amax: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_7, [3], True)
        sub: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_7, amax);  mul_7 = None
        mul_8: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub, 0.125);  sub = None
        amax_1: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_13, [3], True)
        sub_1: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax_1)
        abs_1: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_13)
        ne: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        eq: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_13, convert_element_type_13);  convert_element_type_13 = None
        mul_9: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_9);  mul_9 = None
        any_1: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not, [3], True);  logical_not = None
        logical_not_1: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        where: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_1, mul_8, sub_1);  mul_8 = sub_1 = None
        exp: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where);  where = None
        sum_1: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div_1: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2)
        inductor_random_2: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_27, inductor_lookup_seed_2, 'rand');  _shape_param_27 = inductor_lookup_seed_2 = None
        gt_2: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_2, 0.1);  inductor_random_2 = None
        mul_10: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_2, div_1);  div_1 = None
        mul_11: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_10, 1.1111111111111112);  mul_10 = None
        convert_element_type_15: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_11, torch.bfloat16);  mul_11 = None
        unsqueeze_23: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_15, 4);  convert_element_type_15 = None
        unsqueeze_24: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_9, 4);  view_9 = None
        permute_17: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_24, [4, 1, 2, 3, 0]);  unsqueeze_24 = None
        view_24: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_23, _shape_param_28);  unsqueeze_23 = _shape_param_28 = None
        permute_18: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_17, [1, 2, 4, 3, 0]);  permute_17 = None
        view_25: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_18, _shape_param_29);  permute_18 = _shape_param_29 = None
        bmm_2: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_24, view_25)
        view_26: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_2, _shape_param_30);  bmm_2 = _shape_param_30 = None
        permute_19: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_26, [2, 0, 1, 4, 3]);  view_26 = None
        view_27: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_19, _shape_param_31);  permute_19 = _shape_param_31 = None
        convert_element_type_16: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        unsqueeze_25: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_27, 4);  view_27 = None
        permute_20: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_25, [0, 1, 4, 3, 2]);  unsqueeze_25 = None
        unsqueeze_26: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_16, 3);  convert_element_type_16 = None
        unsqueeze_27: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 4);  unsqueeze_26 = None
        permute_21: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_27, [3, 4, 0, 2, 1]);  unsqueeze_27 = None
        permute_22: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_20, [0, 1, 3, 4, 2]);  permute_20 = None
        clone_1: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_28: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_1, _shape_param_32);  clone_1 = _shape_param_32 = None
        permute_23: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_21, [3, 4, 2, 0, 1]);  permute_21 = None
        clone_2: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_29: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_33);  clone_2 = _shape_param_33 = None
        squeeze_6: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_28, 0)
        squeeze_7: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_29, 0)
        mm_4: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_6, squeeze_7);  squeeze_6 = squeeze_7 = None
        unsqueeze_28: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_4, 0);  mm_4 = None
        view_30: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_28, _shape_param_34);  unsqueeze_28 = _shape_param_34 = None
        permute_24: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_30, [0, 1, 4, 2, 3]);  view_30 = None
        view_31: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_24, _shape_param_35);  permute_24 = _shape_param_35 = None
        inductor_lookup_seed_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
        inductor_random_3: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_36, inductor_lookup_seed_3, 'rand');  _shape_param_36 = inductor_lookup_seed_3 = None
        convert_element_type_17: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_3, torch.bfloat16);  inductor_random_3 = None
        gt_3: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_17, 0.1);  convert_element_type_17 = None
        mul_12: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_3, view_31);  view_31 = None
        mul_13: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_12, 1.1111111111111112);  mul_12 = None
        add_4: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_13, mul_1);  mul_13 = mul_1 = None
        var_mean = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean[0]
        getitem_1: "f32[512, 16, 1]" = var_mean[1];  var_mean = None
        add_5: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_2: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_4, getitem_1);  add_4 = getitem_1 = None
        mul_14: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_15: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_14, arg9_1);  arg9_1 = None
        add_6: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_15, arg10_1);  mul_15 = arg10_1 = None
        convert_element_type_18: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        convert_element_type_19: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        convert_element_type_20: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16)
        view_32: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_20, _shape_param_37);  convert_element_type_20 = _shape_param_37 = None
        permute_25: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_18, view_32, permute_25);  convert_element_type_18 = None
        view_33: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_38);  _shape_param_38 = None
        convert_element_type_21: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        mul_16: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.5)
        mul_17: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476);  convert_element_type_21 = None
        erf: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_7: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_18: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_16, add_7);  mul_16 = add_7 = None
        convert_element_type_22: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        inductor_lookup_seed_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 4)
        inductor_random_4: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_39, inductor_lookup_seed_4, 'rand');  _shape_param_39 = inductor_lookup_seed_4 = None
        convert_element_type_23: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_4, torch.bfloat16);  inductor_random_4 = None
        gt_4: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_23, 0.1);  convert_element_type_23 = None
        mul_19: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_4, convert_element_type_22);  convert_element_type_22 = None
        mul_20: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_19, 1.1111111111111112);  mul_19 = None
        convert_element_type_24: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_25: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        view_34: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_20, _shape_param_40);  mul_20 = _shape_param_40 = None
        permute_26: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_25, [1, 0]);  convert_element_type_25 = None
        addmm_1: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_24, view_34, permute_26);  convert_element_type_24 = None
        view_35: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_1, _shape_param_41);  addmm_1 = _shape_param_41 = None
        inductor_lookup_seed_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 5)
        inductor_random_5: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_42, inductor_lookup_seed_5, 'rand');  _shape_param_42 = inductor_lookup_seed_5 = None
        convert_element_type_26: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_5, torch.bfloat16);  inductor_random_5 = None
        gt_5: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_26, 0.1);  convert_element_type_26 = None
        mul_21: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_5, view_35);  view_35 = None
        mul_22: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_21, 1.1111111111111112);  mul_21 = None
        add_8: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_22, add_6);  mul_22 = add_6 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_2: "f32[512, 16, 1]" = var_mean_1[0]
        getitem_3: "f32[512, 16, 1]" = var_mean_1[1];  var_mean_1 = None
        add_9: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_3: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_8, getitem_3);  add_8 = getitem_3 = None
        mul_23: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_24: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_23, arg15_1);  arg15_1 = None
        add_10: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_24, arg16_1);  mul_24 = arg16_1 = None
        convert_element_type_27: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16)
        convert_element_type_28: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        unsqueeze_29: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_27, 3);  convert_element_type_27 = None
        unsqueeze_30: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_29, 4);  unsqueeze_29 = None
        unsqueeze_31: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_28, 3);  convert_element_type_28 = None
        unsqueeze_32: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_31, 4);  unsqueeze_31 = None
        view_36: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_30, _shape_param_43);  unsqueeze_30 = _shape_param_43 = None
        view_37: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_32, _shape_param_44);  unsqueeze_32 = _shape_param_44 = None
        squeeze_8: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_36, 0)
        squeeze_9: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_37, 0)
        mm_5: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_8, squeeze_9);  squeeze_9 = None
        unsqueeze_33: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_5, 0);  mm_5 = None
        view_38: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_33, _shape_param_45);  unsqueeze_33 = _shape_param_45 = None
        permute_27: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_38, [0, 1, 3, 4, 2]);  view_38 = None
        view_39: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_27, _shape_param_46);  permute_27 = _shape_param_46 = None
        convert_element_type_29: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        unsqueeze_34: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_29, 3);  convert_element_type_29 = None
        unsqueeze_35: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 4);  unsqueeze_34 = None
        view_40: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_35, _shape_param_47);  unsqueeze_35 = _shape_param_47 = None
        squeeze_10: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_40, 0)
        mm_6: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_8, squeeze_10);  squeeze_10 = None
        unsqueeze_36: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_6, 0);  mm_6 = None
        view_41: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_36, _shape_param_48);  unsqueeze_36 = _shape_param_48 = None
        permute_28: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_41, [0, 1, 3, 4, 2]);  view_41 = None
        view_42: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_28, _shape_param_49);  permute_28 = _shape_param_49 = None
        convert_element_type_30: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        unsqueeze_37: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, 3);  convert_element_type_30 = None
        unsqueeze_38: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_37, 4);  unsqueeze_37 = None
        view_43: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_38, _shape_param_50);  unsqueeze_38 = _shape_param_50 = None
        squeeze_11: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_43, 0)
        mm_7: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_8, squeeze_11);  squeeze_8 = squeeze_11 = None
        unsqueeze_39: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_7, 0);  mm_7 = None
        view_44: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_39, _shape_param_51);  unsqueeze_39 = _shape_param_51 = None
        permute_29: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_44, [0, 1, 3, 4, 2]);  view_44 = None
        view_45: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_29, _shape_param_52);  permute_29 = _shape_param_52 = None
        convert_element_type_31: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        unsqueeze_40: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_31, 3);  convert_element_type_31 = None
        unsqueeze_41: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, 4);  unsqueeze_40 = None
        view_46: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_41, _shape_param_53);  unsqueeze_41 = _shape_param_53 = None
        squeeze_12: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_46, 0);  view_46 = None
        mm_8: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_12);  squeeze_12 = None
        unsqueeze_42: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_8, 0);  mm_8 = None
        view_47: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_42, _shape_param_54);  unsqueeze_42 = _shape_param_54 = None
        permute_30: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_47, [0, 1, 3, 4, 2]);  view_47 = None
        view_48: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_30, _shape_param_55);  permute_30 = _shape_param_55 = None
        add_11: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_39, arg21_1);  arg21_1 = None
        convert_element_type_32: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        unsqueeze_43: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, 4);  convert_element_type_32 = None
        permute_31: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_43, [1, 2, 0, 4, 3]);  unsqueeze_43 = None
        unsqueeze_44: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_42, 4);  view_42 = None
        permute_32: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_44, [1, 2, 4, 0, 3]);  unsqueeze_44 = None
        permute_33: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_31, [0, 1, 2, 4, 3]);  permute_31 = None
        view_49: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_33, _shape_param_56);  permute_33 = _shape_param_56 = None
        permute_34: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_32, [0, 1, 4, 3, 2]);  permute_32 = None
        view_50: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_34, _shape_param_57);  permute_34 = _shape_param_57 = None
        bmm_3: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_49, view_50)
        view_51: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_3, _shape_param_58);  bmm_3 = _shape_param_58 = None
        permute_35: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_51, [0, 1, 2, 4, 3]);  view_51 = None
        view_52: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_35, _shape_param_59);  permute_35 = _shape_param_59 = None
        add_12: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_39, arg22_1);  view_39 = arg22_1 = None
        convert_element_type_33: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None
        unsqueeze_45: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_33, 4);  convert_element_type_33 = None
        permute_36: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_45, [1, 2, 0, 4, 3]);  unsqueeze_45 = None
        unsqueeze_46: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_48, 4);  view_48 = None
        permute_37: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_46, [1, 2, 4, 0, 3]);  unsqueeze_46 = None
        permute_38: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_36, [0, 1, 2, 4, 3]);  permute_36 = None
        view_53: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_38, _shape_param_60);  permute_38 = _shape_param_60 = None
        permute_39: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_37, [0, 1, 4, 3, 2]);  permute_37 = None
        view_54: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_39, _shape_param_61);  permute_39 = _shape_param_61 = None
        bmm_4: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_53, view_54)
        view_55: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_4, _shape_param_62);  bmm_4 = _shape_param_62 = None
        permute_40: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_55, [0, 1, 2, 4, 3]);  view_55 = None
        view_56: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_40, _shape_param_63);  permute_40 = _shape_param_63 = None
        view_57: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_56, _shape_param_64);  view_56 = _shape_param_64 = None
        slice_2: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_57, 2, 1, 9223372036854775807);  view_57 = None
        view_58: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_2, _shape_param_65);  slice_2 = _shape_param_65 = None
        index_1: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_58, [None, None, None, iota_2]);  view_58 = None
        add_13: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_52, index_1);  view_52 = index_1 = None
        add_14: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_13, 0);  add_13 = None
        mul_25: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_14, 0.125)
        convert_element_type_34: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_25, torch.float32);  mul_25 = None
        convert_element_type_35: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        mul_26: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1);  convert_element_type_35 = None
        amax_2: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_26, [3], True)
        sub_4: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_26, amax_2);  mul_26 = None
        mul_27: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_4, 0.125);  sub_4 = None
        amax_3: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_34, [3], True)
        sub_5: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_3)
        abs_2: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_34)
        ne_1: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_2, inf);  abs_2 = None
        eq_1: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_34, convert_element_type_34);  convert_element_type_34 = None
        mul_28: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_1, ne_1);  eq_1 = ne_1 = None
        logical_not_2: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_28);  mul_28 = None
        any_2: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_2, [3], True);  logical_not_2 = None
        logical_not_3: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_1: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_3, mul_27, sub_5);  mul_27 = sub_5 = None
        exp_1: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_1);  where_1 = None
        sum_2: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [3], True)
        div_2: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        inductor_lookup_seed_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 6)
        inductor_random_6: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_66, inductor_lookup_seed_6, 'rand');  _shape_param_66 = inductor_lookup_seed_6 = None
        gt_6: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_6, 0.1);  inductor_random_6 = None
        mul_29: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_6, div_2);  div_2 = None
        mul_30: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_29, 1.1111111111111112);  mul_29 = None
        convert_element_type_36: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        unsqueeze_47: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, 4);  convert_element_type_36 = None
        unsqueeze_48: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_45, 4);  view_45 = None
        permute_41: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_48, [4, 1, 2, 3, 0]);  unsqueeze_48 = None
        view_59: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_47, _shape_param_67);  unsqueeze_47 = _shape_param_67 = None
        permute_42: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_41, [1, 2, 4, 3, 0]);  permute_41 = None
        view_60: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_42, _shape_param_68);  permute_42 = _shape_param_68 = None
        bmm_5: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_59, view_60)
        view_61: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_69);  bmm_5 = _shape_param_69 = None
        permute_43: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_61, [2, 0, 1, 4, 3]);  view_61 = None
        view_62: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_43, _shape_param_70);  permute_43 = _shape_param_70 = None
        convert_element_type_37: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        unsqueeze_49: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_62, 4);  view_62 = None
        permute_44: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_49, [0, 1, 4, 3, 2]);  unsqueeze_49 = None
        unsqueeze_50: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_37, 3);  convert_element_type_37 = None
        unsqueeze_51: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, 4);  unsqueeze_50 = None
        permute_45: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_51, [3, 4, 0, 2, 1]);  unsqueeze_51 = None
        permute_46: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_44, [0, 1, 3, 4, 2]);  permute_44 = None
        clone_3: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_63: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_71);  clone_3 = _shape_param_71 = None
        permute_47: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_45, [3, 4, 2, 0, 1]);  permute_45 = None
        clone_4: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_64: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_4, _shape_param_72);  clone_4 = _shape_param_72 = None
        squeeze_13: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_63, 0)
        squeeze_14: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_64, 0)
        mm_9: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_13, squeeze_14);  squeeze_13 = squeeze_14 = None
        unsqueeze_52: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_9, 0);  mm_9 = None
        view_65: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_52, _shape_param_73);  unsqueeze_52 = _shape_param_73 = None
        permute_48: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_65, [0, 1, 4, 2, 3]);  view_65 = None
        view_66: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_48, _shape_param_74);  permute_48 = _shape_param_74 = None
        inductor_lookup_seed_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 7)
        inductor_random_7: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_75, inductor_lookup_seed_7, 'rand');  _shape_param_75 = inductor_lookup_seed_7 = None
        convert_element_type_38: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_7, torch.bfloat16);  inductor_random_7 = None
        gt_7: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_38, 0.1);  convert_element_type_38 = None
        mul_31: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_7, view_66);  view_66 = None
        mul_32: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_31, 1.1111111111111112);  mul_31 = None
        add_15: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_32, add_10);  mul_32 = add_10 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_4: "f32[512, 16, 1]" = var_mean_2[0]
        getitem_5: "f32[512, 16, 1]" = var_mean_2[1];  var_mean_2 = None
        add_16: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_6: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_15, getitem_5);  add_15 = getitem_5 = None
        mul_33: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = None
        mul_34: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg24_1);  arg24_1 = None
        add_17: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_34, arg25_1);  mul_34 = arg25_1 = None
        convert_element_type_39: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        convert_element_type_40: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        convert_element_type_41: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16)
        view_67: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_76);  convert_element_type_41 = _shape_param_76 = None
        permute_49: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_40, [1, 0]);  convert_element_type_40 = None
        addmm_2: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_39, view_67, permute_49);  convert_element_type_39 = None
        view_68: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_2, _shape_param_77);  _shape_param_77 = None
        convert_element_type_42: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        mul_35: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.5)
        mul_36: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.7071067811865476);  convert_element_type_42 = None
        erf_1: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_18: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_37: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_35, add_18);  mul_35 = add_18 = None
        convert_element_type_43: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        inductor_lookup_seed_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 8)
        inductor_random_8: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_78, inductor_lookup_seed_8, 'rand');  _shape_param_78 = inductor_lookup_seed_8 = None
        convert_element_type_44: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_8, torch.bfloat16);  inductor_random_8 = None
        gt_8: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_44, 0.1);  convert_element_type_44 = None
        mul_38: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_8, convert_element_type_43);  convert_element_type_43 = None
        mul_39: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_38, 1.1111111111111112);  mul_38 = None
        convert_element_type_45: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        convert_element_type_46: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        view_69: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_39, _shape_param_79);  mul_39 = _shape_param_79 = None
        permute_50: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_46, [1, 0]);  convert_element_type_46 = None
        addmm_3: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_45, view_69, permute_50);  convert_element_type_45 = None
        view_70: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_3, _shape_param_80);  addmm_3 = _shape_param_80 = None
        inductor_lookup_seed_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 9)
        inductor_random_9: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_81, inductor_lookup_seed_9, 'rand');  _shape_param_81 = inductor_lookup_seed_9 = None
        convert_element_type_47: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_9, torch.bfloat16);  inductor_random_9 = None
        gt_9: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_47, 0.1);  convert_element_type_47 = None
        mul_40: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_9, view_70);  view_70 = None
        mul_41: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None
        add_19: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_41, add_17);  mul_41 = add_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_6: "f32[512, 16, 1]" = var_mean_3[0]
        getitem_7: "f32[512, 16, 1]" = var_mean_3[1];  var_mean_3 = None
        add_20: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_7: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_19, getitem_7);  add_19 = getitem_7 = None
        mul_42: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_3);  sub_7 = None
        mul_43: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_42, arg30_1);  arg30_1 = None
        add_21: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_43, arg31_1);  mul_43 = arg31_1 = None
        convert_element_type_48: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16)
        convert_element_type_49: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        unsqueeze_53: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, 3);  convert_element_type_48 = None
        unsqueeze_54: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 4);  unsqueeze_53 = None
        unsqueeze_55: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_49, 3);  convert_element_type_49 = None
        unsqueeze_56: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
        view_71: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_54, _shape_param_82);  unsqueeze_54 = _shape_param_82 = None
        view_72: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_56, _shape_param_83);  unsqueeze_56 = _shape_param_83 = None
        squeeze_15: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_71, 0)
        squeeze_16: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_72, 0)
        mm_10: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_15, squeeze_16);  squeeze_16 = None
        unsqueeze_57: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_10, 0);  mm_10 = None
        view_73: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_57, _shape_param_84);  unsqueeze_57 = _shape_param_84 = None
        permute_51: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_73, [0, 1, 3, 4, 2]);  view_73 = None
        view_74: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_51, _shape_param_85);  permute_51 = _shape_param_85 = None
        convert_element_type_50: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.bfloat16);  arg33_1 = None
        unsqueeze_58: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_50, 3);  convert_element_type_50 = None
        unsqueeze_59: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, 4);  unsqueeze_58 = None
        view_75: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_59, _shape_param_86);  unsqueeze_59 = _shape_param_86 = None
        squeeze_17: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_75, 0)
        mm_11: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_15, squeeze_17);  squeeze_17 = None
        unsqueeze_60: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_11, 0);  mm_11 = None
        view_76: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_60, _shape_param_87);  unsqueeze_60 = _shape_param_87 = None
        permute_52: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_76, [0, 1, 3, 4, 2]);  view_76 = None
        view_77: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_52, _shape_param_88);  permute_52 = _shape_param_88 = None
        convert_element_type_51: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        unsqueeze_61: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_51, 3);  convert_element_type_51 = None
        unsqueeze_62: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 4);  unsqueeze_61 = None
        view_78: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_62, _shape_param_89);  unsqueeze_62 = _shape_param_89 = None
        squeeze_18: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_78, 0)
        mm_12: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_15, squeeze_18);  squeeze_15 = squeeze_18 = None
        unsqueeze_63: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_12, 0);  mm_12 = None
        view_79: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_63, _shape_param_90);  unsqueeze_63 = _shape_param_90 = None
        permute_53: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_79, [0, 1, 3, 4, 2]);  view_79 = None
        view_80: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_53, _shape_param_91);  permute_53 = _shape_param_91 = None
        convert_element_type_52: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg35_1, torch.bfloat16);  arg35_1 = None
        unsqueeze_64: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, 3);  convert_element_type_52 = None
        unsqueeze_65: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, 4);  unsqueeze_64 = None
        view_81: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_65, _shape_param_92);  unsqueeze_65 = _shape_param_92 = None
        squeeze_19: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_81, 0);  view_81 = None
        mm_13: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_19);  squeeze_19 = None
        unsqueeze_66: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_13, 0);  mm_13 = None
        view_82: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_66, _shape_param_93);  unsqueeze_66 = _shape_param_93 = None
        permute_54: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_82, [0, 1, 3, 4, 2]);  view_82 = None
        view_83: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_54, _shape_param_94);  permute_54 = _shape_param_94 = None
        add_22: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_74, arg36_1);  arg36_1 = None
        convert_element_type_53: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None
        unsqueeze_67: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_53, 4);  convert_element_type_53 = None
        permute_55: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_67, [1, 2, 0, 4, 3]);  unsqueeze_67 = None
        unsqueeze_68: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_77, 4);  view_77 = None
        permute_56: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_68, [1, 2, 4, 0, 3]);  unsqueeze_68 = None
        permute_57: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_55, [0, 1, 2, 4, 3]);  permute_55 = None
        view_84: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_57, _shape_param_95);  permute_57 = _shape_param_95 = None
        permute_58: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_56, [0, 1, 4, 3, 2]);  permute_56 = None
        view_85: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_58, _shape_param_96);  permute_58 = _shape_param_96 = None
        bmm_6: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_84, view_85)
        view_86: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_97);  bmm_6 = _shape_param_97 = None
        permute_59: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_86, [0, 1, 2, 4, 3]);  view_86 = None
        view_87: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_59, _shape_param_98);  permute_59 = _shape_param_98 = None
        add_23: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_74, arg37_1);  view_74 = arg37_1 = None
        convert_element_type_54: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        unsqueeze_69: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, 4);  convert_element_type_54 = None
        permute_60: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_69, [1, 2, 0, 4, 3]);  unsqueeze_69 = None
        unsqueeze_70: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_83, 4);  view_83 = None
        permute_61: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_70, [1, 2, 4, 0, 3]);  unsqueeze_70 = None
        permute_62: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_60, [0, 1, 2, 4, 3]);  permute_60 = None
        view_88: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_62, _shape_param_99);  permute_62 = _shape_param_99 = None
        permute_63: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_61, [0, 1, 4, 3, 2]);  permute_61 = None
        view_89: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_63, _shape_param_100);  permute_63 = _shape_param_100 = None
        bmm_7: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_88, view_89)
        view_90: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_7, _shape_param_101);  bmm_7 = _shape_param_101 = None
        permute_64: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_90, [0, 1, 2, 4, 3]);  view_90 = None
        view_91: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_64, _shape_param_102);  permute_64 = _shape_param_102 = None
        view_92: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_91, _shape_param_103);  view_91 = _shape_param_103 = None
        slice_3: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_92, 2, 1, 9223372036854775807);  view_92 = None
        view_93: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_3, _shape_param_104);  slice_3 = _shape_param_104 = None
        index_2: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_93, [None, None, None, iota_2]);  view_93 = None
        add_24: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_87, index_2);  view_87 = index_2 = None
        add_25: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_24, 0);  add_24 = None
        mul_44: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_25, 0.125)
        convert_element_type_55: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_44, torch.float32);  mul_44 = None
        convert_element_type_56: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        mul_45: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1);  convert_element_type_56 = None
        amax_4: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_45, [3], True)
        sub_8: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_45, amax_4);  mul_45 = None
        mul_46: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_8, 0.125);  sub_8 = None
        amax_5: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_55, [3], True)
        sub_9: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_55, amax_5)
        abs_3: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_55)
        ne_2: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_3, inf);  abs_3 = None
        eq_2: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_55, convert_element_type_55);  convert_element_type_55 = None
        mul_47: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_2, ne_2);  eq_2 = ne_2 = None
        logical_not_4: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_47);  mul_47 = None
        any_3: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_4, [3], True);  logical_not_4 = None
        logical_not_5: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_2: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_5, mul_46, sub_9);  mul_46 = sub_9 = None
        exp_2: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_2);  where_2 = None
        sum_3: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [3], True)
        div_3: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        inductor_lookup_seed_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 10)
        inductor_random_10: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_105, inductor_lookup_seed_10, 'rand');  _shape_param_105 = inductor_lookup_seed_10 = None
        gt_10: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_10, 0.1);  inductor_random_10 = None
        mul_48: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_10, div_3);  div_3 = None
        mul_49: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None
        convert_element_type_57: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_49, torch.bfloat16);  mul_49 = None
        unsqueeze_71: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_57, 4);  convert_element_type_57 = None
        unsqueeze_72: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_80, 4);  view_80 = None
        permute_65: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_72, [4, 1, 2, 3, 0]);  unsqueeze_72 = None
        view_94: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_71, _shape_param_106);  unsqueeze_71 = _shape_param_106 = None
        permute_66: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_65, [1, 2, 4, 3, 0]);  permute_65 = None
        view_95: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_66, _shape_param_107);  permute_66 = _shape_param_107 = None
        bmm_8: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_94, view_95)
        view_96: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_108);  bmm_8 = _shape_param_108 = None
        permute_67: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_96, [2, 0, 1, 4, 3]);  view_96 = None
        view_97: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_67, _shape_param_109);  permute_67 = _shape_param_109 = None
        convert_element_type_58: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.bfloat16);  arg38_1 = None
        unsqueeze_73: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_97, 4);  view_97 = None
        permute_68: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_73, [0, 1, 4, 3, 2]);  unsqueeze_73 = None
        unsqueeze_74: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_58, 3);  convert_element_type_58 = None
        unsqueeze_75: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, 4);  unsqueeze_74 = None
        permute_69: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_75, [3, 4, 0, 2, 1]);  unsqueeze_75 = None
        permute_70: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 4, 2]);  permute_68 = None
        clone_5: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_98: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_5, _shape_param_110);  clone_5 = _shape_param_110 = None
        permute_71: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_69, [3, 4, 2, 0, 1]);  permute_69 = None
        clone_6: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_99: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_6, _shape_param_111);  clone_6 = _shape_param_111 = None
        squeeze_20: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_98, 0)
        squeeze_21: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_99, 0)
        mm_14: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_20, squeeze_21);  squeeze_20 = squeeze_21 = None
        unsqueeze_76: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_14, 0);  mm_14 = None
        view_100: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_76, _shape_param_112);  unsqueeze_76 = _shape_param_112 = None
        permute_72: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_100, [0, 1, 4, 2, 3]);  view_100 = None
        view_101: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_72, _shape_param_113);  permute_72 = _shape_param_113 = None
        inductor_lookup_seed_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11)
        inductor_random_11: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_114, inductor_lookup_seed_11, 'rand');  _shape_param_114 = inductor_lookup_seed_11 = None
        convert_element_type_59: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_11, torch.bfloat16);  inductor_random_11 = None
        gt_11: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_59, 0.1);  convert_element_type_59 = None
        mul_50: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_11, view_101);  view_101 = None
        mul_51: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None
        add_26: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_51, add_21);  mul_51 = add_21 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_8: "f32[512, 16, 1]" = var_mean_4[0]
        getitem_9: "f32[512, 16, 1]" = var_mean_4[1];  var_mean_4 = None
        add_27: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_10: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_26, getitem_9);  add_26 = getitem_9 = None
        mul_52: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_4);  sub_10 = None
        mul_53: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg39_1);  arg39_1 = None
        add_28: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg40_1);  mul_53 = arg40_1 = None
        convert_element_type_60: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        convert_element_type_61: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        convert_element_type_62: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16)
        view_102: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_62, _shape_param_115);  convert_element_type_62 = _shape_param_115 = None
        permute_73: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        addmm_4: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_60, view_102, permute_73);  convert_element_type_60 = None
        view_103: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_4, _shape_param_116);  _shape_param_116 = None
        convert_element_type_63: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_103, torch.float32);  view_103 = None
        mul_54: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 0.5)
        mul_55: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 0.7071067811865476);  convert_element_type_63 = None
        erf_2: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_29: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_56: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_29);  mul_54 = add_29 = None
        convert_element_type_64: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None
        inductor_lookup_seed_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 12)
        inductor_random_12: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_117, inductor_lookup_seed_12, 'rand');  _shape_param_117 = inductor_lookup_seed_12 = None
        convert_element_type_65: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_12, torch.bfloat16);  inductor_random_12 = None
        gt_12: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_65, 0.1);  convert_element_type_65 = None
        mul_57: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_12, convert_element_type_64);  convert_element_type_64 = None
        mul_58: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_57, 1.1111111111111112);  mul_57 = None
        convert_element_type_66: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        convert_element_type_67: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg44_1, torch.bfloat16);  arg44_1 = None
        view_104: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_58, _shape_param_118);  mul_58 = _shape_param_118 = None
        permute_74: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_67, [1, 0]);  convert_element_type_67 = None
        addmm_5: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_66, view_104, permute_74);  convert_element_type_66 = None
        view_105: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_5, _shape_param_119);  addmm_5 = _shape_param_119 = None
        inductor_lookup_seed_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 13)
        inductor_random_13: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_120, inductor_lookup_seed_13, 'rand');  _shape_param_120 = inductor_lookup_seed_13 = None
        convert_element_type_68: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_13, torch.bfloat16);  inductor_random_13 = None
        gt_13: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_68, 0.1);  convert_element_type_68 = None
        mul_59: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_13, view_105);  view_105 = None
        mul_60: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_59, 1.1111111111111112);  mul_59 = None
        add_30: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_60, add_28);  mul_60 = add_28 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_10: "f32[512, 16, 1]" = var_mean_5[0]
        getitem_11: "f32[512, 16, 1]" = var_mean_5[1];  var_mean_5 = None
        add_31: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_11: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_30, getitem_11);  add_30 = getitem_11 = None
        mul_61: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = None
        mul_62: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_61, arg45_1);  arg45_1 = None
        add_32: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_62, arg46_1);  mul_62 = arg46_1 = None
        convert_element_type_69: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16)
        convert_element_type_70: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        unsqueeze_77: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_69, 3);  convert_element_type_69 = None
        unsqueeze_78: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 4);  unsqueeze_77 = None
        unsqueeze_79: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_70, 3);  convert_element_type_70 = None
        unsqueeze_80: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 4);  unsqueeze_79 = None
        view_106: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_78, _shape_param_121);  unsqueeze_78 = _shape_param_121 = None
        view_107: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_80, _shape_param_122);  unsqueeze_80 = _shape_param_122 = None
        squeeze_22: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_106, 0)
        squeeze_23: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_107, 0)
        mm_15: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_22, squeeze_23);  squeeze_23 = None
        unsqueeze_81: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_15, 0);  mm_15 = None
        view_108: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_81, _shape_param_123);  unsqueeze_81 = _shape_param_123 = None
        permute_75: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_108, [0, 1, 3, 4, 2]);  view_108 = None
        view_109: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_75, _shape_param_124);  permute_75 = _shape_param_124 = None
        convert_element_type_71: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.bfloat16);  arg48_1 = None
        unsqueeze_82: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_71, 3);  convert_element_type_71 = None
        unsqueeze_83: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 4);  unsqueeze_82 = None
        view_110: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_83, _shape_param_125);  unsqueeze_83 = _shape_param_125 = None
        squeeze_24: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_110, 0)
        mm_16: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_22, squeeze_24);  squeeze_24 = None
        unsqueeze_84: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_16, 0);  mm_16 = None
        view_111: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_84, _shape_param_126);  unsqueeze_84 = _shape_param_126 = None
        permute_76: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_111, [0, 1, 3, 4, 2]);  view_111 = None
        view_112: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_76, _shape_param_127);  permute_76 = _shape_param_127 = None
        convert_element_type_72: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.bfloat16);  arg49_1 = None
        unsqueeze_85: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, 3);  convert_element_type_72 = None
        unsqueeze_86: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 4);  unsqueeze_85 = None
        view_113: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_86, _shape_param_128);  unsqueeze_86 = _shape_param_128 = None
        squeeze_25: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_113, 0)
        mm_17: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_22, squeeze_25);  squeeze_22 = squeeze_25 = None
        unsqueeze_87: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_17, 0);  mm_17 = None
        view_114: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_87, _shape_param_129);  unsqueeze_87 = _shape_param_129 = None
        permute_77: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_114, [0, 1, 3, 4, 2]);  view_114 = None
        view_115: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_77, _shape_param_130);  permute_77 = _shape_param_130 = None
        convert_element_type_73: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        unsqueeze_88: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_73, 3);  convert_element_type_73 = None
        unsqueeze_89: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 4);  unsqueeze_88 = None
        view_116: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_89, _shape_param_131);  unsqueeze_89 = _shape_param_131 = None
        squeeze_26: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_116, 0);  view_116 = None
        mm_18: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_26);  squeeze_26 = None
        unsqueeze_90: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_18, 0);  mm_18 = None
        view_117: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_90, _shape_param_132);  unsqueeze_90 = _shape_param_132 = None
        permute_78: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_117, [0, 1, 3, 4, 2]);  view_117 = None
        view_118: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_78, _shape_param_133);  permute_78 = _shape_param_133 = None
        add_33: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_109, arg51_1);  arg51_1 = None
        convert_element_type_74: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        unsqueeze_91: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_74, 4);  convert_element_type_74 = None
        permute_79: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_91, [1, 2, 0, 4, 3]);  unsqueeze_91 = None
        unsqueeze_92: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_112, 4);  view_112 = None
        permute_80: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_92, [1, 2, 4, 0, 3]);  unsqueeze_92 = None
        permute_81: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_79, [0, 1, 2, 4, 3]);  permute_79 = None
        view_119: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_81, _shape_param_134);  permute_81 = _shape_param_134 = None
        permute_82: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_80, [0, 1, 4, 3, 2]);  permute_80 = None
        view_120: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_82, _shape_param_135);  permute_82 = _shape_param_135 = None
        bmm_9: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_119, view_120)
        view_121: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_9, _shape_param_136);  bmm_9 = _shape_param_136 = None
        permute_83: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_121, [0, 1, 2, 4, 3]);  view_121 = None
        view_122: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_83, _shape_param_137);  permute_83 = _shape_param_137 = None
        add_34: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_109, arg52_1);  view_109 = arg52_1 = None
        convert_element_type_75: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        unsqueeze_93: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_75, 4);  convert_element_type_75 = None
        permute_84: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_93, [1, 2, 0, 4, 3]);  unsqueeze_93 = None
        unsqueeze_94: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_118, 4);  view_118 = None
        permute_85: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_94, [1, 2, 4, 0, 3]);  unsqueeze_94 = None
        permute_86: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_84, [0, 1, 2, 4, 3]);  permute_84 = None
        view_123: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_86, _shape_param_138);  permute_86 = _shape_param_138 = None
        permute_87: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_85, [0, 1, 4, 3, 2]);  permute_85 = None
        view_124: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_87, _shape_param_139);  permute_87 = _shape_param_139 = None
        bmm_10: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_123, view_124)
        view_125: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_10, _shape_param_140);  bmm_10 = _shape_param_140 = None
        permute_88: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_125, [0, 1, 2, 4, 3]);  view_125 = None
        view_126: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_88, _shape_param_141);  permute_88 = _shape_param_141 = None
        view_127: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_126, _shape_param_142);  view_126 = _shape_param_142 = None
        slice_4: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_127, 2, 1, 9223372036854775807);  view_127 = None
        view_128: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_4, _shape_param_143);  slice_4 = _shape_param_143 = None
        index_3: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_128, [None, None, None, iota_2]);  view_128 = None
        add_35: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_122, index_3);  view_122 = index_3 = None
        add_36: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_35, 0);  add_35 = None
        mul_63: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_36, 0.125)
        convert_element_type_76: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_63, torch.float32);  mul_63 = None
        convert_element_type_77: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        mul_64: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_77, 1);  convert_element_type_77 = None
        amax_6: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_64, [3], True)
        sub_12: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_64, amax_6);  mul_64 = None
        mul_65: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_12, 0.125);  sub_12 = None
        amax_7: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_76, [3], True)
        sub_13: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_76, amax_7)
        abs_4: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_76)
        ne_3: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_4, inf);  abs_4 = None
        eq_3: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_76, convert_element_type_76);  convert_element_type_76 = None
        mul_66: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_3, ne_3);  eq_3 = ne_3 = None
        logical_not_6: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_66);  mul_66 = None
        any_4: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_6, [3], True);  logical_not_6 = None
        logical_not_7: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_3: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_7, mul_65, sub_13);  mul_65 = sub_13 = None
        exp_3: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_3);  where_3 = None
        sum_4: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [3], True)
        div_4: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        inductor_lookup_seed_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 14)
        inductor_random_14: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_144, inductor_lookup_seed_14, 'rand');  _shape_param_144 = inductor_lookup_seed_14 = None
        gt_14: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_14, 0.1);  inductor_random_14 = None
        mul_67: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_14, div_4);  div_4 = None
        mul_68: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_67, 1.1111111111111112);  mul_67 = None
        convert_element_type_78: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_68, torch.bfloat16);  mul_68 = None
        unsqueeze_95: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, 4);  convert_element_type_78 = None
        unsqueeze_96: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_115, 4);  view_115 = None
        permute_89: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_96, [4, 1, 2, 3, 0]);  unsqueeze_96 = None
        view_129: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_95, _shape_param_145);  unsqueeze_95 = _shape_param_145 = None
        permute_90: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_89, [1, 2, 4, 3, 0]);  permute_89 = None
        view_130: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_90, _shape_param_146);  permute_90 = _shape_param_146 = None
        bmm_11: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_129, view_130)
        view_131: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_147);  bmm_11 = _shape_param_147 = None
        permute_91: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_131, [2, 0, 1, 4, 3]);  view_131 = None
        view_132: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_91, _shape_param_148);  permute_91 = _shape_param_148 = None
        convert_element_type_79: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg53_1, torch.bfloat16);  arg53_1 = None
        unsqueeze_97: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_132, 4);  view_132 = None
        permute_92: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_97, [0, 1, 4, 3, 2]);  unsqueeze_97 = None
        unsqueeze_98: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_79, 3);  convert_element_type_79 = None
        unsqueeze_99: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, 4);  unsqueeze_98 = None
        permute_93: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_99, [3, 4, 0, 2, 1]);  unsqueeze_99 = None
        permute_94: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 4, 2]);  permute_92 = None
        clone_7: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_133: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_7, _shape_param_149);  clone_7 = _shape_param_149 = None
        permute_95: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_93, [3, 4, 2, 0, 1]);  permute_93 = None
        clone_8: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_134: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_8, _shape_param_150);  clone_8 = _shape_param_150 = None
        squeeze_27: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_133, 0)
        squeeze_28: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_134, 0)
        mm_19: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_27, squeeze_28);  squeeze_27 = squeeze_28 = None
        unsqueeze_100: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_19, 0);  mm_19 = None
        view_135: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_100, _shape_param_151);  unsqueeze_100 = _shape_param_151 = None
        permute_96: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_135, [0, 1, 4, 2, 3]);  view_135 = None
        view_136: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_96, _shape_param_152);  permute_96 = _shape_param_152 = None
        inductor_lookup_seed_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 15)
        inductor_random_15: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_153, inductor_lookup_seed_15, 'rand');  _shape_param_153 = inductor_lookup_seed_15 = None
        convert_element_type_80: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_15, torch.bfloat16);  inductor_random_15 = None
        gt_15: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_80, 0.1);  convert_element_type_80 = None
        mul_69: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_15, view_136);  view_136 = None
        mul_70: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_69, 1.1111111111111112);  mul_69 = None
        add_37: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_70, add_32);  mul_70 = add_32 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_12: "f32[512, 16, 1]" = var_mean_6[0]
        getitem_13: "f32[512, 16, 1]" = var_mean_6[1];  var_mean_6 = None
        add_38: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_14: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_37, getitem_13);  add_37 = getitem_13 = None
        mul_71: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_6);  sub_14 = None
        mul_72: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_71, arg54_1);  arg54_1 = None
        add_39: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_72, arg55_1);  mul_72 = arg55_1 = None
        convert_element_type_81: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        convert_element_type_82: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        convert_element_type_83: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16)
        view_137: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_83, _shape_param_154);  convert_element_type_83 = _shape_param_154 = None
        permute_97: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_82, [1, 0]);  convert_element_type_82 = None
        addmm_6: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_81, view_137, permute_97);  convert_element_type_81 = None
        view_138: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_6, _shape_param_155);  _shape_param_155 = None
        convert_element_type_84: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None
        mul_73: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 0.5)
        mul_74: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 0.7071067811865476);  convert_element_type_84 = None
        erf_3: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
        add_40: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_75: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_73, add_40);  mul_73 = add_40 = None
        convert_element_type_85: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_75, torch.bfloat16);  mul_75 = None
        inductor_lookup_seed_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 16)
        inductor_random_16: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_156, inductor_lookup_seed_16, 'rand');  _shape_param_156 = inductor_lookup_seed_16 = None
        convert_element_type_86: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_16, torch.bfloat16);  inductor_random_16 = None
        gt_16: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_86, 0.1);  convert_element_type_86 = None
        mul_76: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_16, convert_element_type_85);  convert_element_type_85 = None
        mul_77: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_76, 1.1111111111111112);  mul_76 = None
        convert_element_type_87: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.bfloat16);  arg58_1 = None
        convert_element_type_88: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.bfloat16);  arg59_1 = None
        view_139: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_77, _shape_param_157);  mul_77 = _shape_param_157 = None
        permute_98: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_88, [1, 0]);  convert_element_type_88 = None
        addmm_7: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_87, view_139, permute_98);  convert_element_type_87 = None
        view_140: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_7, _shape_param_158);  addmm_7 = _shape_param_158 = None
        inductor_lookup_seed_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 17)
        inductor_random_17: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_159, inductor_lookup_seed_17, 'rand');  _shape_param_159 = inductor_lookup_seed_17 = None
        convert_element_type_89: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_17, torch.bfloat16);  inductor_random_17 = None
        gt_17: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_89, 0.1);  convert_element_type_89 = None
        mul_78: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_17, view_140);  view_140 = None
        mul_79: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None
        add_41: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_79, add_39);  mul_79 = add_39 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_14: "f32[512, 16, 1]" = var_mean_7[0]
        getitem_15: "f32[512, 16, 1]" = var_mean_7[1];  var_mean_7 = None
        add_42: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_15: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_41, getitem_15);  add_41 = getitem_15 = None
        mul_80: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_7);  sub_15 = None
        mul_81: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_80, arg60_1);  arg60_1 = None
        add_43: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_81, arg61_1);  mul_81 = arg61_1 = None
        convert_element_type_90: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16)
        convert_element_type_91: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.bfloat16);  arg62_1 = None
        unsqueeze_101: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, 3);  convert_element_type_90 = None
        unsqueeze_102: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 4);  unsqueeze_101 = None
        unsqueeze_103: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_91, 3);  convert_element_type_91 = None
        unsqueeze_104: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 4);  unsqueeze_103 = None
        view_141: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_102, _shape_param_160);  unsqueeze_102 = _shape_param_160 = None
        view_142: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_104, _shape_param_161);  unsqueeze_104 = _shape_param_161 = None
        squeeze_29: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_141, 0)
        squeeze_30: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_142, 0)
        mm_20: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_29, squeeze_30);  squeeze_30 = None
        unsqueeze_105: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_20, 0);  mm_20 = None
        view_143: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_105, _shape_param_162);  unsqueeze_105 = _shape_param_162 = None
        permute_99: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_143, [0, 1, 3, 4, 2]);  view_143 = None
        view_144: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_99, _shape_param_163);  permute_99 = _shape_param_163 = None
        convert_element_type_92: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.bfloat16);  arg63_1 = None
        unsqueeze_106: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, 3);  convert_element_type_92 = None
        unsqueeze_107: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 4);  unsqueeze_106 = None
        view_145: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_107, _shape_param_164);  unsqueeze_107 = _shape_param_164 = None
        squeeze_31: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_145, 0)
        mm_21: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_29, squeeze_31);  squeeze_31 = None
        unsqueeze_108: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_21, 0);  mm_21 = None
        view_146: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_108, _shape_param_165);  unsqueeze_108 = _shape_param_165 = None
        permute_100: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_146, [0, 1, 3, 4, 2]);  view_146 = None
        view_147: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_100, _shape_param_166);  permute_100 = _shape_param_166 = None
        convert_element_type_93: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        unsqueeze_109: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_93, 3);  convert_element_type_93 = None
        unsqueeze_110: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 4);  unsqueeze_109 = None
        view_148: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_110, _shape_param_167);  unsqueeze_110 = _shape_param_167 = None
        squeeze_32: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_148, 0)
        mm_22: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_29, squeeze_32);  squeeze_29 = squeeze_32 = None
        unsqueeze_111: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_22, 0);  mm_22 = None
        view_149: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_111, _shape_param_168);  unsqueeze_111 = _shape_param_168 = None
        permute_101: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_149, [0, 1, 3, 4, 2]);  view_149 = None
        view_150: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_101, _shape_param_169);  permute_101 = _shape_param_169 = None
        convert_element_type_94: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg65_1, torch.bfloat16);  arg65_1 = None
        unsqueeze_112: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_94, 3);  convert_element_type_94 = None
        unsqueeze_113: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 4);  unsqueeze_112 = None
        view_151: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_113, _shape_param_170);  unsqueeze_113 = _shape_param_170 = None
        squeeze_33: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_151, 0);  view_151 = None
        mm_23: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_33);  squeeze_33 = None
        unsqueeze_114: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_23, 0);  mm_23 = None
        view_152: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_114, _shape_param_171);  unsqueeze_114 = _shape_param_171 = None
        permute_102: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_152, [0, 1, 3, 4, 2]);  view_152 = None
        view_153: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_102, _shape_param_172);  permute_102 = _shape_param_172 = None
        add_44: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_144, arg66_1);  arg66_1 = None
        convert_element_type_95: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None
        unsqueeze_115: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_95, 4);  convert_element_type_95 = None
        permute_103: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_115, [1, 2, 0, 4, 3]);  unsqueeze_115 = None
        unsqueeze_116: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_147, 4);  view_147 = None
        permute_104: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_116, [1, 2, 4, 0, 3]);  unsqueeze_116 = None
        permute_105: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_103, [0, 1, 2, 4, 3]);  permute_103 = None
        view_154: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_105, _shape_param_173);  permute_105 = _shape_param_173 = None
        permute_106: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_104, [0, 1, 4, 3, 2]);  permute_104 = None
        view_155: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_106, _shape_param_174);  permute_106 = _shape_param_174 = None
        bmm_12: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_154, view_155)
        view_156: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_12, _shape_param_175);  bmm_12 = _shape_param_175 = None
        permute_107: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_156, [0, 1, 2, 4, 3]);  view_156 = None
        view_157: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_107, _shape_param_176);  permute_107 = _shape_param_176 = None
        add_45: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_144, arg67_1);  view_144 = arg67_1 = None
        convert_element_type_96: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None
        unsqueeze_117: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, 4);  convert_element_type_96 = None
        permute_108: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_117, [1, 2, 0, 4, 3]);  unsqueeze_117 = None
        unsqueeze_118: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_153, 4);  view_153 = None
        permute_109: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_118, [1, 2, 4, 0, 3]);  unsqueeze_118 = None
        permute_110: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_108, [0, 1, 2, 4, 3]);  permute_108 = None
        view_158: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_110, _shape_param_177);  permute_110 = _shape_param_177 = None
        permute_111: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_109, [0, 1, 4, 3, 2]);  permute_109 = None
        view_159: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_111, _shape_param_178);  permute_111 = _shape_param_178 = None
        bmm_13: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_158, view_159)
        view_160: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_13, _shape_param_179);  bmm_13 = _shape_param_179 = None
        permute_112: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_160, [0, 1, 2, 4, 3]);  view_160 = None
        view_161: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_112, _shape_param_180);  permute_112 = _shape_param_180 = None
        view_162: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_161, _shape_param_181);  view_161 = _shape_param_181 = None
        slice_5: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_162, 2, 1, 9223372036854775807);  view_162 = None
        view_163: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_5, _shape_param_182);  slice_5 = _shape_param_182 = None
        index_4: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_163, [None, None, None, iota_2]);  view_163 = None
        add_46: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_157, index_4);  view_157 = index_4 = None
        add_47: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_46, 0);  add_46 = None
        mul_82: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_47, 0.125)
        convert_element_type_97: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_82, torch.float32);  mul_82 = None
        convert_element_type_98: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        mul_83: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 1);  convert_element_type_98 = None
        amax_8: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_83, [3], True)
        sub_16: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_83, amax_8);  mul_83 = None
        mul_84: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_16, 0.125);  sub_16 = None
        amax_9: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_97, [3], True)
        sub_17: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_97, amax_9)
        abs_5: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_97)
        ne_4: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_5, inf);  abs_5 = None
        eq_4: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_97, convert_element_type_97);  convert_element_type_97 = None
        mul_85: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_4, ne_4);  eq_4 = ne_4 = None
        logical_not_8: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_85);  mul_85 = None
        any_5: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_8, [3], True);  logical_not_8 = None
        logical_not_9: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_4: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_9, mul_84, sub_17);  mul_84 = sub_17 = None
        exp_4: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_4);  where_4 = None
        sum_5: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [3], True)
        div_5: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        inductor_lookup_seed_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 18)
        inductor_random_18: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_183, inductor_lookup_seed_18, 'rand');  _shape_param_183 = inductor_lookup_seed_18 = None
        gt_18: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_18, 0.1);  inductor_random_18 = None
        mul_86: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_18, div_5);  div_5 = None
        mul_87: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_86, 1.1111111111111112);  mul_86 = None
        convert_element_type_99: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None
        unsqueeze_119: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_99, 4);  convert_element_type_99 = None
        unsqueeze_120: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_150, 4);  view_150 = None
        permute_113: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_120, [4, 1, 2, 3, 0]);  unsqueeze_120 = None
        view_164: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_119, _shape_param_184);  unsqueeze_119 = _shape_param_184 = None
        permute_114: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_113, [1, 2, 4, 3, 0]);  permute_113 = None
        view_165: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_114, _shape_param_185);  permute_114 = _shape_param_185 = None
        bmm_14: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_164, view_165)
        view_166: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_14, _shape_param_186);  bmm_14 = _shape_param_186 = None
        permute_115: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_166, [2, 0, 1, 4, 3]);  view_166 = None
        view_167: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_115, _shape_param_187);  permute_115 = _shape_param_187 = None
        convert_element_type_100: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        unsqueeze_121: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_167, 4);  view_167 = None
        permute_116: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_121, [0, 1, 4, 3, 2]);  unsqueeze_121 = None
        unsqueeze_122: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_100, 3);  convert_element_type_100 = None
        unsqueeze_123: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 4);  unsqueeze_122 = None
        permute_117: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_123, [3, 4, 0, 2, 1]);  unsqueeze_123 = None
        permute_118: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_116, [0, 1, 3, 4, 2]);  permute_116 = None
        clone_9: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
        view_168: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_9, _shape_param_188);  clone_9 = _shape_param_188 = None
        permute_119: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_117, [3, 4, 2, 0, 1]);  permute_117 = None
        clone_10: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
        view_169: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_10, _shape_param_189);  clone_10 = _shape_param_189 = None
        squeeze_34: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_168, 0)
        squeeze_35: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_169, 0)
        mm_24: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_34, squeeze_35);  squeeze_34 = squeeze_35 = None
        unsqueeze_124: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_24, 0);  mm_24 = None
        view_170: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_124, _shape_param_190);  unsqueeze_124 = _shape_param_190 = None
        permute_120: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_170, [0, 1, 4, 2, 3]);  view_170 = None
        view_171: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_120, _shape_param_191);  permute_120 = _shape_param_191 = None
        inductor_lookup_seed_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 19)
        inductor_random_19: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_192, inductor_lookup_seed_19, 'rand');  _shape_param_192 = inductor_lookup_seed_19 = None
        convert_element_type_101: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_19, torch.bfloat16);  inductor_random_19 = None
        gt_19: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_101, 0.1);  convert_element_type_101 = None
        mul_88: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_19, view_171);  view_171 = None
        mul_89: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_88, 1.1111111111111112);  mul_88 = None
        add_48: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_89, add_43);  mul_89 = add_43 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_16: "f32[512, 16, 1]" = var_mean_8[0]
        getitem_17: "f32[512, 16, 1]" = var_mean_8[1];  var_mean_8 = None
        add_49: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_18: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_48, getitem_17);  add_48 = getitem_17 = None
        mul_90: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_8);  sub_18 = None
        mul_91: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_90, arg69_1);  arg69_1 = None
        add_50: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_91, arg70_1);  mul_91 = arg70_1 = None
        convert_element_type_102: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg71_1, torch.bfloat16);  arg71_1 = None
        convert_element_type_103: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        convert_element_type_104: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16)
        view_172: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_104, _shape_param_193);  convert_element_type_104 = _shape_param_193 = None
        permute_121: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_103, [1, 0]);  convert_element_type_103 = None
        addmm_8: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_102, view_172, permute_121);  convert_element_type_102 = None
        view_173: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_8, _shape_param_194);  _shape_param_194 = None
        convert_element_type_105: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_92: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_105, 0.5)
        mul_93: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_105, 0.7071067811865476);  convert_element_type_105 = None
        erf_4: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
        add_51: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_94: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_92, add_51);  mul_92 = add_51 = None
        convert_element_type_106: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_94, torch.bfloat16);  mul_94 = None
        inductor_lookup_seed_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 20)
        inductor_random_20: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_195, inductor_lookup_seed_20, 'rand');  _shape_param_195 = inductor_lookup_seed_20 = None
        convert_element_type_107: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_20, torch.bfloat16);  inductor_random_20 = None
        gt_20: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_107, 0.1);  convert_element_type_107 = None
        mul_95: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_20, convert_element_type_106);  convert_element_type_106 = None
        mul_96: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_95, 1.1111111111111112);  mul_95 = None
        convert_element_type_108: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        convert_element_type_109: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        view_174: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_96, _shape_param_196);  mul_96 = _shape_param_196 = None
        permute_122: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        addmm_9: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_108, view_174, permute_122);  convert_element_type_108 = None
        view_175: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_9, _shape_param_197);  addmm_9 = _shape_param_197 = None
        inductor_lookup_seed_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 21)
        inductor_random_21: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_198, inductor_lookup_seed_21, 'rand');  _shape_param_198 = inductor_lookup_seed_21 = None
        convert_element_type_110: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_21, torch.bfloat16);  inductor_random_21 = None
        gt_21: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_110, 0.1);  convert_element_type_110 = None
        mul_97: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_21, view_175);  view_175 = None
        mul_98: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_97, 1.1111111111111112);  mul_97 = None
        add_52: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_98, add_50);  mul_98 = add_50 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_18: "f32[512, 16, 1]" = var_mean_9[0]
        getitem_19: "f32[512, 16, 1]" = var_mean_9[1];  var_mean_9 = None
        add_53: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_19: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_52, getitem_19);  add_52 = getitem_19 = None
        mul_99: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_9);  sub_19 = None
        mul_100: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_99, arg75_1);  arg75_1 = None
        add_54: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_100, arg76_1);  mul_100 = arg76_1 = None
        convert_element_type_111: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16)
        convert_element_type_112: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        unsqueeze_125: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_111, 3);  convert_element_type_111 = None
        unsqueeze_126: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 4);  unsqueeze_125 = None
        unsqueeze_127: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_112, 3);  convert_element_type_112 = None
        unsqueeze_128: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 4);  unsqueeze_127 = None
        view_176: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_126, _shape_param_199);  unsqueeze_126 = _shape_param_199 = None
        view_177: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_128, _shape_param_200);  unsqueeze_128 = _shape_param_200 = None
        squeeze_36: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_176, 0)
        squeeze_37: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_177, 0)
        mm_25: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_36, squeeze_37);  squeeze_37 = None
        unsqueeze_129: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_25, 0);  mm_25 = None
        view_178: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_129, _shape_param_201);  unsqueeze_129 = _shape_param_201 = None
        permute_123: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_178, [0, 1, 3, 4, 2]);  view_178 = None
        view_179: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_123, _shape_param_202);  permute_123 = _shape_param_202 = None
        convert_element_type_113: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        unsqueeze_130: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_113, 3);  convert_element_type_113 = None
        unsqueeze_131: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 4);  unsqueeze_130 = None
        view_180: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_131, _shape_param_203);  unsqueeze_131 = _shape_param_203 = None
        squeeze_38: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_180, 0)
        mm_26: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_36, squeeze_38);  squeeze_38 = None
        unsqueeze_132: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_26, 0);  mm_26 = None
        view_181: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_132, _shape_param_204);  unsqueeze_132 = _shape_param_204 = None
        permute_124: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_181, [0, 1, 3, 4, 2]);  view_181 = None
        view_182: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_124, _shape_param_205);  permute_124 = _shape_param_205 = None
        convert_element_type_114: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg79_1, torch.bfloat16);  arg79_1 = None
        unsqueeze_133: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, 3);  convert_element_type_114 = None
        unsqueeze_134: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 4);  unsqueeze_133 = None
        view_183: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_134, _shape_param_206);  unsqueeze_134 = _shape_param_206 = None
        squeeze_39: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_183, 0)
        mm_27: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_36, squeeze_39);  squeeze_36 = squeeze_39 = None
        unsqueeze_135: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_27, 0);  mm_27 = None
        view_184: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_135, _shape_param_207);  unsqueeze_135 = _shape_param_207 = None
        permute_125: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_184, [0, 1, 3, 4, 2]);  view_184 = None
        view_185: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_125, _shape_param_208);  permute_125 = _shape_param_208 = None
        convert_element_type_115: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.bfloat16);  arg80_1 = None
        unsqueeze_136: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_115, 3);  convert_element_type_115 = None
        unsqueeze_137: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 4);  unsqueeze_136 = None
        view_186: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_137, _shape_param_209);  unsqueeze_137 = _shape_param_209 = None
        squeeze_40: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_186, 0);  view_186 = None
        mm_28: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_40);  squeeze_40 = None
        unsqueeze_138: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_28, 0);  mm_28 = None
        view_187: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_138, _shape_param_210);  unsqueeze_138 = _shape_param_210 = None
        permute_126: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_187, [0, 1, 3, 4, 2]);  view_187 = None
        view_188: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_126, _shape_param_211);  permute_126 = _shape_param_211 = None
        add_55: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_179, arg81_1);  arg81_1 = None
        convert_element_type_116: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        unsqueeze_139: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_116, 4);  convert_element_type_116 = None
        permute_127: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_139, [1, 2, 0, 4, 3]);  unsqueeze_139 = None
        unsqueeze_140: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_182, 4);  view_182 = None
        permute_128: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_140, [1, 2, 4, 0, 3]);  unsqueeze_140 = None
        permute_129: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_127, [0, 1, 2, 4, 3]);  permute_127 = None
        view_189: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_129, _shape_param_212);  permute_129 = _shape_param_212 = None
        permute_130: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_128, [0, 1, 4, 3, 2]);  permute_128 = None
        view_190: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_130, _shape_param_213);  permute_130 = _shape_param_213 = None
        bmm_15: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_189, view_190)
        view_191: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_15, _shape_param_214);  bmm_15 = _shape_param_214 = None
        permute_131: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_191, [0, 1, 2, 4, 3]);  view_191 = None
        view_192: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_131, _shape_param_215);  permute_131 = _shape_param_215 = None
        add_56: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_179, arg82_1);  view_179 = arg82_1 = None
        convert_element_type_117: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None
        unsqueeze_141: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_117, 4);  convert_element_type_117 = None
        permute_132: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_141, [1, 2, 0, 4, 3]);  unsqueeze_141 = None
        unsqueeze_142: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_188, 4);  view_188 = None
        permute_133: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_142, [1, 2, 4, 0, 3]);  unsqueeze_142 = None
        permute_134: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_132, [0, 1, 2, 4, 3]);  permute_132 = None
        view_193: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_134, _shape_param_216);  permute_134 = _shape_param_216 = None
        permute_135: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_133, [0, 1, 4, 3, 2]);  permute_133 = None
        view_194: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_135, _shape_param_217);  permute_135 = _shape_param_217 = None
        bmm_16: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_193, view_194)
        view_195: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_16, _shape_param_218);  bmm_16 = _shape_param_218 = None
        permute_136: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_195, [0, 1, 2, 4, 3]);  view_195 = None
        view_196: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_136, _shape_param_219);  permute_136 = _shape_param_219 = None
        view_197: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_196, _shape_param_220);  view_196 = _shape_param_220 = None
        slice_6: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_197, 2, 1, 9223372036854775807);  view_197 = None
        view_198: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_6, _shape_param_221);  slice_6 = _shape_param_221 = None
        index_5: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_198, [None, None, None, iota_2]);  view_198 = None
        add_57: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_192, index_5);  view_192 = index_5 = None
        add_58: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_57, 0);  add_57 = None
        mul_101: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_58, 0.125)
        convert_element_type_118: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_101, torch.float32);  mul_101 = None
        convert_element_type_119: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32)
        mul_102: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_119, 1);  convert_element_type_119 = None
        amax_10: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_102, [3], True)
        sub_20: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_102, amax_10);  mul_102 = None
        mul_103: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_20, 0.125);  sub_20 = None
        amax_11: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_118, [3], True)
        sub_21: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_118, amax_11)
        abs_6: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_118)
        ne_5: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_6, inf);  abs_6 = None
        eq_5: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_118, convert_element_type_118);  convert_element_type_118 = None
        mul_104: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_5, ne_5);  eq_5 = ne_5 = None
        logical_not_10: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_104);  mul_104 = None
        any_6: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_10, [3], True);  logical_not_10 = None
        logical_not_11: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_5: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_11, mul_103, sub_21);  mul_103 = sub_21 = None
        exp_5: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_5);  where_5 = None
        sum_6: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [3], True)
        div_6: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        inductor_lookup_seed_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22)
        inductor_random_22: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_222, inductor_lookup_seed_22, 'rand');  _shape_param_222 = inductor_lookup_seed_22 = None
        gt_22: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_22, 0.1);  inductor_random_22 = None
        mul_105: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_22, div_6);  div_6 = None
        mul_106: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None
        convert_element_type_120: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_106, torch.bfloat16);  mul_106 = None
        unsqueeze_143: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_120, 4);  convert_element_type_120 = None
        unsqueeze_144: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_185, 4);  view_185 = None
        permute_137: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_144, [4, 1, 2, 3, 0]);  unsqueeze_144 = None
        view_199: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_143, _shape_param_223);  unsqueeze_143 = _shape_param_223 = None
        permute_138: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_137, [1, 2, 4, 3, 0]);  permute_137 = None
        view_200: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_138, _shape_param_224);  permute_138 = _shape_param_224 = None
        bmm_17: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_199, view_200)
        view_201: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_225);  bmm_17 = _shape_param_225 = None
        permute_139: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_201, [2, 0, 1, 4, 3]);  view_201 = None
        view_202: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_139, _shape_param_226);  permute_139 = _shape_param_226 = None
        convert_element_type_121: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.bfloat16);  arg83_1 = None
        unsqueeze_145: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_202, 4);  view_202 = None
        permute_140: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_145, [0, 1, 4, 3, 2]);  unsqueeze_145 = None
        unsqueeze_146: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_121, 3);  convert_element_type_121 = None
        unsqueeze_147: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 4);  unsqueeze_146 = None
        permute_141: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_147, [3, 4, 0, 2, 1]);  unsqueeze_147 = None
        permute_142: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 4, 2]);  permute_140 = None
        clone_11: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_203: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_11, _shape_param_227);  clone_11 = _shape_param_227 = None
        permute_143: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_141, [3, 4, 2, 0, 1]);  permute_141 = None
        clone_12: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
        view_204: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_12, _shape_param_228);  clone_12 = _shape_param_228 = None
        squeeze_41: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_203, 0)
        squeeze_42: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_204, 0)
        mm_29: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_41, squeeze_42);  squeeze_41 = squeeze_42 = None
        unsqueeze_148: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_29, 0);  mm_29 = None
        view_205: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_148, _shape_param_229);  unsqueeze_148 = _shape_param_229 = None
        permute_144: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_205, [0, 1, 4, 2, 3]);  view_205 = None
        view_206: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_144, _shape_param_230);  permute_144 = _shape_param_230 = None
        inductor_lookup_seed_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 23)
        inductor_random_23: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_231, inductor_lookup_seed_23, 'rand');  _shape_param_231 = inductor_lookup_seed_23 = None
        convert_element_type_122: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_23, torch.bfloat16);  inductor_random_23 = None
        gt_23: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_122, 0.1);  convert_element_type_122 = None
        mul_107: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_23, view_206);  view_206 = None
        mul_108: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_107, 1.1111111111111112);  mul_107 = None
        add_59: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_108, add_54);  mul_108 = add_54 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_20: "f32[512, 16, 1]" = var_mean_10[0]
        getitem_21: "f32[512, 16, 1]" = var_mean_10[1];  var_mean_10 = None
        add_60: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_22: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_59, getitem_21);  add_59 = getitem_21 = None
        mul_109: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_10);  sub_22 = None
        mul_110: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_109, arg84_1);  arg84_1 = None
        add_61: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_110, arg85_1);  mul_110 = arg85_1 = None
        convert_element_type_123: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.bfloat16);  arg86_1 = None
        convert_element_type_124: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.bfloat16);  arg87_1 = None
        convert_element_type_125: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        view_207: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_125, _shape_param_232);  convert_element_type_125 = _shape_param_232 = None
        permute_145: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_124, [1, 0]);  convert_element_type_124 = None
        addmm_10: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_123, view_207, permute_145);  convert_element_type_123 = None
        view_208: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_10, _shape_param_233);  _shape_param_233 = None
        convert_element_type_126: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        mul_111: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.5)
        mul_112: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.7071067811865476);  convert_element_type_126 = None
        erf_5: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_112);  mul_112 = None
        add_62: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_113: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_111, add_62);  mul_111 = add_62 = None
        convert_element_type_127: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_113, torch.bfloat16);  mul_113 = None
        inductor_lookup_seed_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24)
        inductor_random_24: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_234, inductor_lookup_seed_24, 'rand');  _shape_param_234 = inductor_lookup_seed_24 = None
        convert_element_type_128: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_24, torch.bfloat16);  inductor_random_24 = None
        gt_24: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_128, 0.1);  convert_element_type_128 = None
        mul_114: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_24, convert_element_type_127);  convert_element_type_127 = None
        mul_115: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None
        convert_element_type_129: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.bfloat16);  arg88_1 = None
        convert_element_type_130: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg89_1, torch.bfloat16);  arg89_1 = None
        view_209: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_115, _shape_param_235);  mul_115 = _shape_param_235 = None
        permute_146: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        addmm_11: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_129, view_209, permute_146);  convert_element_type_129 = None
        view_210: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_11, _shape_param_236);  addmm_11 = _shape_param_236 = None
        inductor_lookup_seed_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 25)
        inductor_random_25: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_237, inductor_lookup_seed_25, 'rand');  _shape_param_237 = inductor_lookup_seed_25 = None
        convert_element_type_131: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_25, torch.bfloat16);  inductor_random_25 = None
        gt_25: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_131, 0.1);  convert_element_type_131 = None
        mul_116: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_25, view_210);  view_210 = None
        mul_117: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_116, 1.1111111111111112);  mul_116 = None
        add_63: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_117, add_61);  mul_117 = add_61 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_22: "f32[512, 16, 1]" = var_mean_11[0]
        getitem_23: "f32[512, 16, 1]" = var_mean_11[1];  var_mean_11 = None
        add_64: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_23: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_63, getitem_23);  add_63 = getitem_23 = None
        mul_118: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_11);  sub_23 = None
        mul_119: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_118, arg90_1);  arg90_1 = None
        add_65: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_119, arg91_1);  mul_119 = arg91_1 = None
        convert_element_type_132: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16)
        convert_element_type_133: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.bfloat16);  arg92_1 = None
        unsqueeze_149: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_132, 3);  convert_element_type_132 = None
        unsqueeze_150: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 4);  unsqueeze_149 = None
        unsqueeze_151: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_133, 3);  convert_element_type_133 = None
        unsqueeze_152: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 4);  unsqueeze_151 = None
        view_211: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_150, _shape_param_238);  unsqueeze_150 = _shape_param_238 = None
        view_212: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_152, _shape_param_239);  unsqueeze_152 = _shape_param_239 = None
        squeeze_43: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_211, 0)
        squeeze_44: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_212, 0)
        mm_30: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_43, squeeze_44);  squeeze_44 = None
        unsqueeze_153: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_30, 0);  mm_30 = None
        view_213: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_153, _shape_param_240);  unsqueeze_153 = _shape_param_240 = None
        permute_147: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_213, [0, 1, 3, 4, 2]);  view_213 = None
        view_214: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_147, _shape_param_241);  permute_147 = _shape_param_241 = None
        convert_element_type_134: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        unsqueeze_154: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_134, 3);  convert_element_type_134 = None
        unsqueeze_155: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 4);  unsqueeze_154 = None
        view_215: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_155, _shape_param_242);  unsqueeze_155 = _shape_param_242 = None
        squeeze_45: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_215, 0)
        mm_31: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_43, squeeze_45);  squeeze_45 = None
        unsqueeze_156: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_31, 0);  mm_31 = None
        view_216: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_156, _shape_param_243);  unsqueeze_156 = _shape_param_243 = None
        permute_148: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_216, [0, 1, 3, 4, 2]);  view_216 = None
        view_217: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_148, _shape_param_244);  permute_148 = _shape_param_244 = None
        convert_element_type_135: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg94_1, torch.bfloat16);  arg94_1 = None
        unsqueeze_157: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_135, 3);  convert_element_type_135 = None
        unsqueeze_158: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 4);  unsqueeze_157 = None
        view_218: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_158, _shape_param_245);  unsqueeze_158 = _shape_param_245 = None
        squeeze_46: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_218, 0)
        mm_32: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_43, squeeze_46);  squeeze_43 = squeeze_46 = None
        unsqueeze_159: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_32, 0);  mm_32 = None
        view_219: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_159, _shape_param_246);  unsqueeze_159 = _shape_param_246 = None
        permute_149: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_219, [0, 1, 3, 4, 2]);  view_219 = None
        view_220: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_149, _shape_param_247);  permute_149 = _shape_param_247 = None
        convert_element_type_136: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.bfloat16);  arg95_1 = None
        unsqueeze_160: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_136, 3);  convert_element_type_136 = None
        unsqueeze_161: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 4);  unsqueeze_160 = None
        view_221: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_161, _shape_param_248);  unsqueeze_161 = _shape_param_248 = None
        squeeze_47: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_221, 0);  view_221 = None
        mm_33: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_47);  squeeze_47 = None
        unsqueeze_162: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_33, 0);  mm_33 = None
        view_222: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_162, _shape_param_249);  unsqueeze_162 = _shape_param_249 = None
        permute_150: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_222, [0, 1, 3, 4, 2]);  view_222 = None
        view_223: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_150, _shape_param_250);  permute_150 = _shape_param_250 = None
        add_66: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_214, arg96_1);  arg96_1 = None
        convert_element_type_137: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        unsqueeze_163: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_137, 4);  convert_element_type_137 = None
        permute_151: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_163, [1, 2, 0, 4, 3]);  unsqueeze_163 = None
        unsqueeze_164: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_217, 4);  view_217 = None
        permute_152: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_164, [1, 2, 4, 0, 3]);  unsqueeze_164 = None
        permute_153: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_151, [0, 1, 2, 4, 3]);  permute_151 = None
        view_224: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_153, _shape_param_251);  permute_153 = _shape_param_251 = None
        permute_154: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_152, [0, 1, 4, 3, 2]);  permute_152 = None
        view_225: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_154, _shape_param_252);  permute_154 = _shape_param_252 = None
        bmm_18: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_224, view_225)
        view_226: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_253);  bmm_18 = _shape_param_253 = None
        permute_155: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_226, [0, 1, 2, 4, 3]);  view_226 = None
        view_227: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_155, _shape_param_254);  permute_155 = _shape_param_254 = None
        add_67: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_214, arg97_1);  view_214 = arg97_1 = None
        convert_element_type_138: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None
        unsqueeze_165: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_138, 4);  convert_element_type_138 = None
        permute_156: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_165, [1, 2, 0, 4, 3]);  unsqueeze_165 = None
        unsqueeze_166: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_223, 4);  view_223 = None
        permute_157: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_166, [1, 2, 4, 0, 3]);  unsqueeze_166 = None
        permute_158: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_156, [0, 1, 2, 4, 3]);  permute_156 = None
        view_228: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_158, _shape_param_255);  permute_158 = _shape_param_255 = None
        permute_159: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_157, [0, 1, 4, 3, 2]);  permute_157 = None
        view_229: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_159, _shape_param_256);  permute_159 = _shape_param_256 = None
        bmm_19: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_228, view_229)
        view_230: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_19, _shape_param_257);  bmm_19 = _shape_param_257 = None
        permute_160: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_230, [0, 1, 2, 4, 3]);  view_230 = None
        view_231: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_160, _shape_param_258);  permute_160 = _shape_param_258 = None
        view_232: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_231, _shape_param_259);  view_231 = _shape_param_259 = None
        slice_7: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_232, 2, 1, 9223372036854775807);  view_232 = None
        view_233: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_7, _shape_param_260);  slice_7 = _shape_param_260 = None
        index_6: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_233, [None, None, None, iota_2]);  view_233 = None
        add_68: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_227, index_6);  view_227 = index_6 = None
        add_69: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_68, 0);  add_68 = None
        mul_120: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_69, 0.125)
        convert_element_type_139: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_120, torch.float32);  mul_120 = None
        convert_element_type_140: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        mul_121: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_140, 1);  convert_element_type_140 = None
        amax_12: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_121, [3], True)
        sub_24: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_121, amax_12);  mul_121 = None
        mul_122: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_24, 0.125);  sub_24 = None
        amax_13: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_139, [3], True)
        sub_25: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_139, amax_13)
        abs_7: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_139)
        ne_6: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_7, inf);  abs_7 = None
        eq_6: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_139, convert_element_type_139);  convert_element_type_139 = None
        mul_123: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_6, ne_6);  eq_6 = ne_6 = None
        logical_not_12: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_123);  mul_123 = None
        any_7: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_12, [3], True);  logical_not_12 = None
        logical_not_13: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_6: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_13, mul_122, sub_25);  mul_122 = sub_25 = None
        exp_6: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_6);  where_6 = None
        sum_7: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [3], True)
        div_7: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        inductor_lookup_seed_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 26)
        inductor_random_26: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_261, inductor_lookup_seed_26, 'rand');  _shape_param_261 = inductor_lookup_seed_26 = None
        gt_26: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_26, 0.1);  inductor_random_26 = None
        mul_124: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_26, div_7);  div_7 = None
        mul_125: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_124, 1.1111111111111112);  mul_124 = None
        convert_element_type_141: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_125, torch.bfloat16);  mul_125 = None
        unsqueeze_167: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_141, 4);  convert_element_type_141 = None
        unsqueeze_168: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_220, 4);  view_220 = None
        permute_161: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_168, [4, 1, 2, 3, 0]);  unsqueeze_168 = None
        view_234: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_167, _shape_param_262);  unsqueeze_167 = _shape_param_262 = None
        permute_162: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_161, [1, 2, 4, 3, 0]);  permute_161 = None
        view_235: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_162, _shape_param_263);  permute_162 = _shape_param_263 = None
        bmm_20: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_234, view_235)
        view_236: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_264);  bmm_20 = _shape_param_264 = None
        permute_163: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_236, [2, 0, 1, 4, 3]);  view_236 = None
        view_237: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_163, _shape_param_265);  permute_163 = _shape_param_265 = None
        convert_element_type_142: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.bfloat16);  arg98_1 = None
        unsqueeze_169: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_237, 4);  view_237 = None
        permute_164: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_169, [0, 1, 4, 3, 2]);  unsqueeze_169 = None
        unsqueeze_170: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_142, 3);  convert_element_type_142 = None
        unsqueeze_171: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 4);  unsqueeze_170 = None
        permute_165: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_171, [3, 4, 0, 2, 1]);  unsqueeze_171 = None
        permute_166: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 4, 2]);  permute_164 = None
        clone_13: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_238: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_13, _shape_param_266);  clone_13 = _shape_param_266 = None
        permute_167: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_165, [3, 4, 2, 0, 1]);  permute_165 = None
        clone_14: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_239: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_14, _shape_param_267);  clone_14 = _shape_param_267 = None
        squeeze_48: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_238, 0)
        squeeze_49: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_239, 0)
        mm_34: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_48, squeeze_49);  squeeze_48 = squeeze_49 = None
        unsqueeze_172: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_34, 0);  mm_34 = None
        view_240: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_172, _shape_param_268);  unsqueeze_172 = _shape_param_268 = None
        permute_168: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_240, [0, 1, 4, 2, 3]);  view_240 = None
        view_241: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_168, _shape_param_269);  permute_168 = _shape_param_269 = None
        inductor_lookup_seed_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 27)
        inductor_random_27: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_270, inductor_lookup_seed_27, 'rand');  _shape_param_270 = inductor_lookup_seed_27 = None
        convert_element_type_143: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_27, torch.bfloat16);  inductor_random_27 = None
        gt_27: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_143, 0.1);  convert_element_type_143 = None
        mul_126: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_27, view_241);  view_241 = None
        mul_127: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_126, 1.1111111111111112);  mul_126 = None
        add_70: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_127, add_65);  mul_127 = add_65 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_24: "f32[512, 16, 1]" = var_mean_12[0]
        getitem_25: "f32[512, 16, 1]" = var_mean_12[1];  var_mean_12 = None
        add_71: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_26: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_70, getitem_25);  add_70 = getitem_25 = None
        mul_128: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_12);  sub_26 = None
        mul_129: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_128, arg99_1);  arg99_1 = None
        add_72: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_129, arg100_1);  mul_129 = arg100_1 = None
        convert_element_type_144: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.bfloat16);  arg101_1 = None
        convert_element_type_145: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        convert_element_type_146: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16)
        view_242: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_146, _shape_param_271);  convert_element_type_146 = _shape_param_271 = None
        permute_169: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_145, [1, 0]);  convert_element_type_145 = None
        addmm_12: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_144, view_242, permute_169);  convert_element_type_144 = None
        view_243: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_12, _shape_param_272);  _shape_param_272 = None
        convert_element_type_147: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_243, torch.float32);  view_243 = None
        mul_130: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_147, 0.5)
        mul_131: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_147, 0.7071067811865476);  convert_element_type_147 = None
        erf_6: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_73: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_132: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_130, add_73);  mul_130 = add_73 = None
        convert_element_type_148: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_132, torch.bfloat16);  mul_132 = None
        inductor_lookup_seed_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 28)
        inductor_random_28: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_273, inductor_lookup_seed_28, 'rand');  _shape_param_273 = inductor_lookup_seed_28 = None
        convert_element_type_149: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_28, torch.bfloat16);  inductor_random_28 = None
        gt_28: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_149, 0.1);  convert_element_type_149 = None
        mul_133: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_28, convert_element_type_148);  convert_element_type_148 = None
        mul_134: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_133, 1.1111111111111112);  mul_133 = None
        convert_element_type_150: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg103_1, torch.bfloat16);  arg103_1 = None
        convert_element_type_151: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg104_1, torch.bfloat16);  arg104_1 = None
        view_244: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_134, _shape_param_274);  mul_134 = _shape_param_274 = None
        permute_170: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        addmm_13: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_150, view_244, permute_170);  convert_element_type_150 = None
        view_245: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_13, _shape_param_275);  addmm_13 = _shape_param_275 = None
        inductor_lookup_seed_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 29)
        inductor_random_29: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_276, inductor_lookup_seed_29, 'rand');  _shape_param_276 = inductor_lookup_seed_29 = None
        convert_element_type_152: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_29, torch.bfloat16);  inductor_random_29 = None
        gt_29: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_152, 0.1);  convert_element_type_152 = None
        mul_135: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_29, view_245);  view_245 = None
        mul_136: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None
        add_74: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_136, add_72);  mul_136 = add_72 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_26: "f32[512, 16, 1]" = var_mean_13[0]
        getitem_27: "f32[512, 16, 1]" = var_mean_13[1];  var_mean_13 = None
        add_75: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_27: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_74, getitem_27);  add_74 = getitem_27 = None
        mul_137: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_13);  sub_27 = None
        mul_138: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_137, arg105_1);  arg105_1 = None
        add_76: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_138, arg106_1);  mul_138 = arg106_1 = None
        convert_element_type_153: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16)
        convert_element_type_154: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        unsqueeze_173: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_153, 3);  convert_element_type_153 = None
        unsqueeze_174: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 4);  unsqueeze_173 = None
        unsqueeze_175: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_154, 3);  convert_element_type_154 = None
        unsqueeze_176: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 4);  unsqueeze_175 = None
        view_246: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_174, _shape_param_277);  unsqueeze_174 = _shape_param_277 = None
        view_247: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_176, _shape_param_278);  unsqueeze_176 = _shape_param_278 = None
        squeeze_50: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_246, 0)
        squeeze_51: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_247, 0)
        mm_35: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_50, squeeze_51);  squeeze_51 = None
        unsqueeze_177: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_35, 0);  mm_35 = None
        view_248: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_177, _shape_param_279);  unsqueeze_177 = _shape_param_279 = None
        permute_171: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_248, [0, 1, 3, 4, 2]);  view_248 = None
        view_249: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_171, _shape_param_280);  permute_171 = _shape_param_280 = None
        convert_element_type_155: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.bfloat16);  arg108_1 = None
        unsqueeze_178: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_155, 3);  convert_element_type_155 = None
        unsqueeze_179: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 4);  unsqueeze_178 = None
        view_250: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_179, _shape_param_281);  unsqueeze_179 = _shape_param_281 = None
        squeeze_52: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_250, 0)
        mm_36: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_50, squeeze_52);  squeeze_52 = None
        unsqueeze_180: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_36, 0);  mm_36 = None
        view_251: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_180, _shape_param_282);  unsqueeze_180 = _shape_param_282 = None
        permute_172: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_251, [0, 1, 3, 4, 2]);  view_251 = None
        view_252: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_172, _shape_param_283);  permute_172 = _shape_param_283 = None
        convert_element_type_156: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        unsqueeze_181: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_156, 3);  convert_element_type_156 = None
        unsqueeze_182: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 4);  unsqueeze_181 = None
        view_253: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_182, _shape_param_284);  unsqueeze_182 = _shape_param_284 = None
        squeeze_53: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_253, 0)
        mm_37: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_50, squeeze_53);  squeeze_50 = squeeze_53 = None
        unsqueeze_183: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_37, 0);  mm_37 = None
        view_254: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_183, _shape_param_285);  unsqueeze_183 = _shape_param_285 = None
        permute_173: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_254, [0, 1, 3, 4, 2]);  view_254 = None
        view_255: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_173, _shape_param_286);  permute_173 = _shape_param_286 = None
        convert_element_type_157: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.bfloat16);  arg110_1 = None
        unsqueeze_184: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_157, 3);  convert_element_type_157 = None
        unsqueeze_185: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 4);  unsqueeze_184 = None
        view_256: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_185, _shape_param_287);  unsqueeze_185 = _shape_param_287 = None
        squeeze_54: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_256, 0);  view_256 = None
        mm_38: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_54);  squeeze_54 = None
        unsqueeze_186: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_38, 0);  mm_38 = None
        view_257: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_186, _shape_param_288);  unsqueeze_186 = _shape_param_288 = None
        permute_174: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_257, [0, 1, 3, 4, 2]);  view_257 = None
        view_258: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_174, _shape_param_289);  permute_174 = _shape_param_289 = None
        add_77: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_249, arg111_1);  arg111_1 = None
        convert_element_type_158: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None
        unsqueeze_187: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_158, 4);  convert_element_type_158 = None
        permute_175: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_187, [1, 2, 0, 4, 3]);  unsqueeze_187 = None
        unsqueeze_188: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_252, 4);  view_252 = None
        permute_176: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_188, [1, 2, 4, 0, 3]);  unsqueeze_188 = None
        permute_177: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_175, [0, 1, 2, 4, 3]);  permute_175 = None
        view_259: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_177, _shape_param_290);  permute_177 = _shape_param_290 = None
        permute_178: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_176, [0, 1, 4, 3, 2]);  permute_176 = None
        view_260: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_178, _shape_param_291);  permute_178 = _shape_param_291 = None
        bmm_21: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_259, view_260)
        view_261: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_21, _shape_param_292);  bmm_21 = _shape_param_292 = None
        permute_179: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_261, [0, 1, 2, 4, 3]);  view_261 = None
        view_262: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_179, _shape_param_293);  permute_179 = _shape_param_293 = None
        add_78: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_249, arg112_1);  view_249 = arg112_1 = None
        convert_element_type_159: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None
        unsqueeze_189: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_159, 4);  convert_element_type_159 = None
        permute_180: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_189, [1, 2, 0, 4, 3]);  unsqueeze_189 = None
        unsqueeze_190: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_258, 4);  view_258 = None
        permute_181: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_190, [1, 2, 4, 0, 3]);  unsqueeze_190 = None
        permute_182: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_180, [0, 1, 2, 4, 3]);  permute_180 = None
        view_263: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_182, _shape_param_294);  permute_182 = _shape_param_294 = None
        permute_183: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_181, [0, 1, 4, 3, 2]);  permute_181 = None
        view_264: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_183, _shape_param_295);  permute_183 = _shape_param_295 = None
        bmm_22: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_263, view_264)
        view_265: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_22, _shape_param_296);  bmm_22 = _shape_param_296 = None
        permute_184: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_265, [0, 1, 2, 4, 3]);  view_265 = None
        view_266: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_184, _shape_param_297);  permute_184 = _shape_param_297 = None
        view_267: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_266, _shape_param_298);  view_266 = _shape_param_298 = None
        slice_8: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_267, 2, 1, 9223372036854775807);  view_267 = None
        view_268: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_8, _shape_param_299);  slice_8 = _shape_param_299 = None
        index_7: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_268, [None, None, None, iota_2]);  view_268 = None
        add_79: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_262, index_7);  view_262 = index_7 = None
        add_80: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_79, 0);  add_79 = None
        mul_139: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_80, 0.125)
        convert_element_type_160: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_139, torch.float32);  mul_139 = None
        convert_element_type_161: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_80, torch.float32)
        mul_140: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_161, 1);  convert_element_type_161 = None
        amax_14: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_140, [3], True)
        sub_28: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_140, amax_14);  mul_140 = None
        mul_141: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_28, 0.125);  sub_28 = None
        amax_15: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_160, [3], True)
        sub_29: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_160, amax_15)
        abs_8: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_160)
        ne_7: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_8, inf);  abs_8 = None
        eq_7: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_160, convert_element_type_160);  convert_element_type_160 = None
        mul_142: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_7, ne_7);  eq_7 = ne_7 = None
        logical_not_14: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_142);  mul_142 = None
        any_8: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_14, [3], True);  logical_not_14 = None
        logical_not_15: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_7: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_15, mul_141, sub_29);  mul_141 = sub_29 = None
        exp_7: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_7);  where_7 = None
        sum_8: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [3], True)
        div_8: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
        inductor_lookup_seed_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 30)
        inductor_random_30: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_300, inductor_lookup_seed_30, 'rand');  _shape_param_300 = inductor_lookup_seed_30 = None
        gt_30: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_30, 0.1);  inductor_random_30 = None
        mul_143: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_30, div_8);  div_8 = None
        mul_144: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_143, 1.1111111111111112);  mul_143 = None
        convert_element_type_162: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        unsqueeze_191: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_162, 4);  convert_element_type_162 = None
        unsqueeze_192: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_255, 4);  view_255 = None
        permute_185: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_192, [4, 1, 2, 3, 0]);  unsqueeze_192 = None
        view_269: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_191, _shape_param_301);  unsqueeze_191 = _shape_param_301 = None
        permute_186: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_185, [1, 2, 4, 3, 0]);  permute_185 = None
        view_270: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_186, _shape_param_302);  permute_186 = _shape_param_302 = None
        bmm_23: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_269, view_270)
        view_271: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_303);  bmm_23 = _shape_param_303 = None
        permute_187: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_271, [2, 0, 1, 4, 3]);  view_271 = None
        view_272: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_187, _shape_param_304);  permute_187 = _shape_param_304 = None
        convert_element_type_163: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.bfloat16);  arg113_1 = None
        unsqueeze_193: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_272, 4);  view_272 = None
        permute_188: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_193, [0, 1, 4, 3, 2]);  unsqueeze_193 = None
        unsqueeze_194: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_163, 3);  convert_element_type_163 = None
        unsqueeze_195: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 4);  unsqueeze_194 = None
        permute_189: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_195, [3, 4, 0, 2, 1]);  unsqueeze_195 = None
        permute_190: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_188, [0, 1, 3, 4, 2]);  permute_188 = None
        clone_15: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_273: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_15, _shape_param_305);  clone_15 = _shape_param_305 = None
        permute_191: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_189, [3, 4, 2, 0, 1]);  permute_189 = None
        clone_16: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
        view_274: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_16, _shape_param_306);  clone_16 = _shape_param_306 = None
        squeeze_55: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_273, 0)
        squeeze_56: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_274, 0)
        mm_39: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_55, squeeze_56);  squeeze_55 = squeeze_56 = None
        unsqueeze_196: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_39, 0);  mm_39 = None
        view_275: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_196, _shape_param_307);  unsqueeze_196 = _shape_param_307 = None
        permute_192: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_275, [0, 1, 4, 2, 3]);  view_275 = None
        view_276: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_192, _shape_param_308);  permute_192 = _shape_param_308 = None
        inductor_lookup_seed_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 31)
        inductor_random_31: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_309, inductor_lookup_seed_31, 'rand');  _shape_param_309 = inductor_lookup_seed_31 = None
        convert_element_type_164: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_31, torch.bfloat16);  inductor_random_31 = None
        gt_31: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_164, 0.1);  convert_element_type_164 = None
        mul_145: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_31, view_276);  view_276 = None
        mul_146: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_145, 1.1111111111111112);  mul_145 = None
        add_81: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_146, add_76);  mul_146 = add_76 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_28: "f32[512, 16, 1]" = var_mean_14[0]
        getitem_29: "f32[512, 16, 1]" = var_mean_14[1];  var_mean_14 = None
        add_82: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_30: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_81, getitem_29);  add_81 = getitem_29 = None
        mul_147: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_14);  sub_30 = None
        mul_148: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_147, arg114_1);  arg114_1 = None
        add_83: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_148, arg115_1);  mul_148 = arg115_1 = None
        convert_element_type_165: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.bfloat16);  arg116_1 = None
        convert_element_type_166: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        convert_element_type_167: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16)
        view_277: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_167, _shape_param_310);  convert_element_type_167 = _shape_param_310 = None
        permute_193: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_166, [1, 0]);  convert_element_type_166 = None
        addmm_14: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_165, view_277, permute_193);  convert_element_type_165 = None
        view_278: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_14, _shape_param_311);  _shape_param_311 = None
        convert_element_type_168: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_278, torch.float32);  view_278 = None
        mul_149: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_168, 0.5)
        mul_150: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_168, 0.7071067811865476);  convert_element_type_168 = None
        erf_7: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_150);  mul_150 = None
        add_84: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_151: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_149, add_84);  mul_149 = add_84 = None
        convert_element_type_169: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_151, torch.bfloat16);  mul_151 = None
        inductor_lookup_seed_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 32)
        inductor_random_32: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_312, inductor_lookup_seed_32, 'rand');  _shape_param_312 = inductor_lookup_seed_32 = None
        convert_element_type_170: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_32, torch.bfloat16);  inductor_random_32 = None
        gt_32: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_170, 0.1);  convert_element_type_170 = None
        mul_152: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_32, convert_element_type_169);  convert_element_type_169 = None
        mul_153: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_152, 1.1111111111111112);  mul_152 = None
        convert_element_type_171: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.bfloat16);  arg118_1 = None
        convert_element_type_172: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg119_1, torch.bfloat16);  arg119_1 = None
        view_279: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_153, _shape_param_313);  mul_153 = _shape_param_313 = None
        permute_194: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        addmm_15: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_171, view_279, permute_194);  convert_element_type_171 = None
        view_280: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_15, _shape_param_314);  addmm_15 = _shape_param_314 = None
        inductor_lookup_seed_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 33)
        inductor_random_33: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_315, inductor_lookup_seed_33, 'rand');  _shape_param_315 = inductor_lookup_seed_33 = None
        convert_element_type_173: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_33, torch.bfloat16);  inductor_random_33 = None
        gt_33: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_173, 0.1);  convert_element_type_173 = None
        mul_154: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_33, view_280);  view_280 = None
        mul_155: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_154, 1.1111111111111112);  mul_154 = None
        add_85: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_155, add_83);  mul_155 = add_83 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_30: "f32[512, 16, 1]" = var_mean_15[0]
        getitem_31: "f32[512, 16, 1]" = var_mean_15[1];  var_mean_15 = None
        add_86: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_31: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_85, getitem_31);  add_85 = getitem_31 = None
        mul_156: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_15);  sub_31 = None
        mul_157: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_156, arg120_1);  arg120_1 = None
        add_87: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_157, arg121_1);  mul_157 = arg121_1 = None
        convert_element_type_174: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16)
        convert_element_type_175: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        unsqueeze_197: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_174, 3);  convert_element_type_174 = None
        unsqueeze_198: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 4);  unsqueeze_197 = None
        unsqueeze_199: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_175, 3);  convert_element_type_175 = None
        unsqueeze_200: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 4);  unsqueeze_199 = None
        view_281: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_198, _shape_param_316);  unsqueeze_198 = _shape_param_316 = None
        view_282: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_200, _shape_param_317);  unsqueeze_200 = _shape_param_317 = None
        squeeze_57: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_281, 0)
        squeeze_58: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_282, 0)
        mm_40: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_57, squeeze_58);  squeeze_58 = None
        unsqueeze_201: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_40, 0);  mm_40 = None
        view_283: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_201, _shape_param_318);  unsqueeze_201 = _shape_param_318 = None
        permute_195: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_283, [0, 1, 3, 4, 2]);  view_283 = None
        view_284: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_195, _shape_param_319);  permute_195 = _shape_param_319 = None
        convert_element_type_176: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.bfloat16);  arg123_1 = None
        unsqueeze_202: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_176, 3);  convert_element_type_176 = None
        unsqueeze_203: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 4);  unsqueeze_202 = None
        view_285: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_203, _shape_param_320);  unsqueeze_203 = _shape_param_320 = None
        squeeze_59: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_285, 0)
        mm_41: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_57, squeeze_59);  squeeze_59 = None
        unsqueeze_204: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_41, 0);  mm_41 = None
        view_286: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_204, _shape_param_321);  unsqueeze_204 = _shape_param_321 = None
        permute_196: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_286, [0, 1, 3, 4, 2]);  view_286 = None
        view_287: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_196, _shape_param_322);  permute_196 = _shape_param_322 = None
        convert_element_type_177: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.bfloat16);  arg124_1 = None
        unsqueeze_205: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_177, 3);  convert_element_type_177 = None
        unsqueeze_206: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 4);  unsqueeze_205 = None
        view_288: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_206, _shape_param_323);  unsqueeze_206 = _shape_param_323 = None
        squeeze_60: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_288, 0)
        mm_42: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_57, squeeze_60);  squeeze_57 = squeeze_60 = None
        unsqueeze_207: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_42, 0);  mm_42 = None
        view_289: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_207, _shape_param_324);  unsqueeze_207 = _shape_param_324 = None
        permute_197: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_289, [0, 1, 3, 4, 2]);  view_289 = None
        view_290: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_197, _shape_param_325);  permute_197 = _shape_param_325 = None
        convert_element_type_178: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.bfloat16);  arg125_1 = None
        unsqueeze_208: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_178, 3);  convert_element_type_178 = None
        unsqueeze_209: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 4);  unsqueeze_208 = None
        view_291: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_209, _shape_param_326);  unsqueeze_209 = _shape_param_326 = None
        squeeze_61: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_291, 0);  view_291 = None
        mm_43: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_61);  squeeze_61 = None
        unsqueeze_210: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_43, 0);  mm_43 = None
        view_292: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_210, _shape_param_327);  unsqueeze_210 = _shape_param_327 = None
        permute_198: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_292, [0, 1, 3, 4, 2]);  view_292 = None
        view_293: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_198, _shape_param_328);  permute_198 = _shape_param_328 = None
        add_88: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_284, arg126_1);  arg126_1 = None
        convert_element_type_179: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16);  add_88 = None
        unsqueeze_211: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_179, 4);  convert_element_type_179 = None
        permute_199: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_211, [1, 2, 0, 4, 3]);  unsqueeze_211 = None
        unsqueeze_212: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_287, 4);  view_287 = None
        permute_200: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_212, [1, 2, 4, 0, 3]);  unsqueeze_212 = None
        permute_201: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_199, [0, 1, 2, 4, 3]);  permute_199 = None
        view_294: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_201, _shape_param_329);  permute_201 = _shape_param_329 = None
        permute_202: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_200, [0, 1, 4, 3, 2]);  permute_200 = None
        view_295: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_202, _shape_param_330);  permute_202 = _shape_param_330 = None
        bmm_24: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_294, view_295)
        view_296: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_24, _shape_param_331);  bmm_24 = _shape_param_331 = None
        permute_203: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_296, [0, 1, 2, 4, 3]);  view_296 = None
        view_297: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_203, _shape_param_332);  permute_203 = _shape_param_332 = None
        add_89: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_284, arg127_1);  view_284 = arg127_1 = None
        convert_element_type_180: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None
        unsqueeze_213: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_180, 4);  convert_element_type_180 = None
        permute_204: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_213, [1, 2, 0, 4, 3]);  unsqueeze_213 = None
        unsqueeze_214: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_293, 4);  view_293 = None
        permute_205: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_214, [1, 2, 4, 0, 3]);  unsqueeze_214 = None
        permute_206: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_204, [0, 1, 2, 4, 3]);  permute_204 = None
        view_298: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_206, _shape_param_333);  permute_206 = _shape_param_333 = None
        permute_207: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_205, [0, 1, 4, 3, 2]);  permute_205 = None
        view_299: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_207, _shape_param_334);  permute_207 = _shape_param_334 = None
        bmm_25: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_298, view_299)
        view_300: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_25, _shape_param_335);  bmm_25 = _shape_param_335 = None
        permute_208: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_300, [0, 1, 2, 4, 3]);  view_300 = None
        view_301: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_208, _shape_param_336);  permute_208 = _shape_param_336 = None
        view_302: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_301, _shape_param_337);  view_301 = _shape_param_337 = None
        slice_9: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_302, 2, 1, 9223372036854775807);  view_302 = None
        view_303: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_9, _shape_param_338);  slice_9 = _shape_param_338 = None
        index_8: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_303, [None, None, None, iota_2]);  view_303 = None
        add_90: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_297, index_8);  view_297 = index_8 = None
        add_91: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_90, 0);  add_90 = None
        mul_158: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_91, 0.125)
        convert_element_type_181: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_158, torch.float32);  mul_158 = None
        convert_element_type_182: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        mul_159: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 1);  convert_element_type_182 = None
        amax_16: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_159, [3], True)
        sub_32: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_159, amax_16);  mul_159 = None
        mul_160: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_32, 0.125);  sub_32 = None
        amax_17: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_181, [3], True)
        sub_33: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_181, amax_17)
        abs_9: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_181)
        ne_8: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_9, inf);  abs_9 = None
        eq_8: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_181, convert_element_type_181);  convert_element_type_181 = None
        mul_161: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_8, ne_8);  eq_8 = ne_8 = None
        logical_not_16: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_161);  mul_161 = None
        any_9: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_16, [3], True);  logical_not_16 = None
        logical_not_17: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_8: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_17, mul_160, sub_33);  mul_160 = sub_33 = None
        exp_8: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_8);  where_8 = None
        sum_9: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [3], True)
        div_9: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None
        inductor_lookup_seed_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 34)
        inductor_random_34: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_339, inductor_lookup_seed_34, 'rand');  _shape_param_339 = inductor_lookup_seed_34 = None
        gt_34: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_34, 0.1);  inductor_random_34 = None
        mul_162: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_34, div_9);  div_9 = None
        mul_163: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_162, 1.1111111111111112);  mul_162 = None
        convert_element_type_183: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_163, torch.bfloat16);  mul_163 = None
        unsqueeze_215: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_183, 4);  convert_element_type_183 = None
        unsqueeze_216: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_290, 4);  view_290 = None
        permute_209: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_216, [4, 1, 2, 3, 0]);  unsqueeze_216 = None
        view_304: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_215, _shape_param_340);  unsqueeze_215 = _shape_param_340 = None
        permute_210: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_209, [1, 2, 4, 3, 0]);  permute_209 = None
        view_305: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_210, _shape_param_341);  permute_210 = _shape_param_341 = None
        bmm_26: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_304, view_305)
        view_306: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_26, _shape_param_342);  bmm_26 = _shape_param_342 = None
        permute_211: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_306, [2, 0, 1, 4, 3]);  view_306 = None
        view_307: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_211, _shape_param_343);  permute_211 = _shape_param_343 = None
        convert_element_type_184: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        unsqueeze_217: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_307, 4);  view_307 = None
        permute_212: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_217, [0, 1, 4, 3, 2]);  unsqueeze_217 = None
        unsqueeze_218: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_184, 3);  convert_element_type_184 = None
        unsqueeze_219: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 4);  unsqueeze_218 = None
        permute_213: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_219, [3, 4, 0, 2, 1]);  unsqueeze_219 = None
        permute_214: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_212, [0, 1, 3, 4, 2]);  permute_212 = None
        clone_17: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_308: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_17, _shape_param_344);  clone_17 = _shape_param_344 = None
        permute_215: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_213, [3, 4, 2, 0, 1]);  permute_213 = None
        clone_18: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_309: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_18, _shape_param_345);  clone_18 = _shape_param_345 = None
        squeeze_62: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_308, 0)
        squeeze_63: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_309, 0)
        mm_44: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_62, squeeze_63);  squeeze_62 = squeeze_63 = None
        unsqueeze_220: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_44, 0);  mm_44 = None
        view_310: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_220, _shape_param_346);  unsqueeze_220 = _shape_param_346 = None
        permute_216: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_310, [0, 1, 4, 2, 3]);  view_310 = None
        view_311: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_216, _shape_param_347);  permute_216 = _shape_param_347 = None
        inductor_lookup_seed_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 35)
        inductor_random_35: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_348, inductor_lookup_seed_35, 'rand');  _shape_param_348 = inductor_lookup_seed_35 = None
        convert_element_type_185: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_35, torch.bfloat16);  inductor_random_35 = None
        gt_35: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_185, 0.1);  convert_element_type_185 = None
        mul_164: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_35, view_311);  view_311 = None
        mul_165: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_164, 1.1111111111111112);  mul_164 = None
        add_92: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_165, add_87);  mul_165 = add_87 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_32: "f32[512, 16, 1]" = var_mean_16[0]
        getitem_33: "f32[512, 16, 1]" = var_mean_16[1];  var_mean_16 = None
        add_93: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_34: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_92, getitem_33);  add_92 = getitem_33 = None
        mul_166: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_16);  sub_34 = None
        mul_167: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_166, arg129_1);  arg129_1 = None
        add_94: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_167, arg130_1);  mul_167 = arg130_1 = None
        convert_element_type_186: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.bfloat16);  arg131_1 = None
        convert_element_type_187: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        convert_element_type_188: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16)
        view_312: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_188, _shape_param_349);  convert_element_type_188 = _shape_param_349 = None
        permute_217: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_187, [1, 0]);  convert_element_type_187 = None
        addmm_16: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_186, view_312, permute_217);  convert_element_type_186 = None
        view_313: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_16, _shape_param_350);  _shape_param_350 = None
        convert_element_type_189: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_313, torch.float32);  view_313 = None
        mul_168: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_189, 0.5)
        mul_169: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_189, 0.7071067811865476);  convert_element_type_189 = None
        erf_8: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_169);  mul_169 = None
        add_95: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_170: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_168, add_95);  mul_168 = add_95 = None
        convert_element_type_190: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_170, torch.bfloat16);  mul_170 = None
        inductor_lookup_seed_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 36)
        inductor_random_36: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_351, inductor_lookup_seed_36, 'rand');  _shape_param_351 = inductor_lookup_seed_36 = None
        convert_element_type_191: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_36, torch.bfloat16);  inductor_random_36 = None
        gt_36: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_191, 0.1);  convert_element_type_191 = None
        mul_171: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_36, convert_element_type_190);  convert_element_type_190 = None
        mul_172: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_171, 1.1111111111111112);  mul_171 = None
        convert_element_type_192: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.bfloat16);  arg133_1 = None
        convert_element_type_193: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.bfloat16);  arg134_1 = None
        view_314: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_172, _shape_param_352);  mul_172 = _shape_param_352 = None
        permute_218: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None
        addmm_17: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_192, view_314, permute_218);  convert_element_type_192 = None
        view_315: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_17, _shape_param_353);  addmm_17 = _shape_param_353 = None
        inductor_lookup_seed_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 37)
        inductor_random_37: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_354, inductor_lookup_seed_37, 'rand');  _shape_param_354 = inductor_lookup_seed_37 = None
        convert_element_type_194: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_37, torch.bfloat16);  inductor_random_37 = None
        gt_37: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_194, 0.1);  convert_element_type_194 = None
        mul_173: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_37, view_315);  view_315 = None
        mul_174: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_173, 1.1111111111111112);  mul_173 = None
        add_96: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_174, add_94);  mul_174 = add_94 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_34: "f32[512, 16, 1]" = var_mean_17[0]
        getitem_35: "f32[512, 16, 1]" = var_mean_17[1];  var_mean_17 = None
        add_97: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_35: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_96, getitem_35);  add_96 = getitem_35 = None
        mul_175: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_17);  sub_35 = None
        mul_176: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_175, arg135_1);  arg135_1 = None
        add_98: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_176, arg136_1);  mul_176 = arg136_1 = None
        convert_element_type_195: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16)
        convert_element_type_196: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        unsqueeze_221: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_195, 3);  convert_element_type_195 = None
        unsqueeze_222: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 4);  unsqueeze_221 = None
        unsqueeze_223: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_196, 3);  convert_element_type_196 = None
        unsqueeze_224: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 4);  unsqueeze_223 = None
        view_316: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_222, _shape_param_355);  unsqueeze_222 = _shape_param_355 = None
        view_317: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_224, _shape_param_356);  unsqueeze_224 = _shape_param_356 = None
        squeeze_64: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_316, 0)
        squeeze_65: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_317, 0)
        mm_45: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_64, squeeze_65);  squeeze_65 = None
        unsqueeze_225: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_45, 0);  mm_45 = None
        view_318: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_225, _shape_param_357);  unsqueeze_225 = _shape_param_357 = None
        permute_219: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_318, [0, 1, 3, 4, 2]);  view_318 = None
        view_319: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_219, _shape_param_358);  permute_219 = _shape_param_358 = None
        convert_element_type_197: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.bfloat16);  arg138_1 = None
        unsqueeze_226: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_197, 3);  convert_element_type_197 = None
        unsqueeze_227: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 4);  unsqueeze_226 = None
        view_320: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_227, _shape_param_359);  unsqueeze_227 = _shape_param_359 = None
        squeeze_66: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_320, 0)
        mm_46: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_64, squeeze_66);  squeeze_66 = None
        unsqueeze_228: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_46, 0);  mm_46 = None
        view_321: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_228, _shape_param_360);  unsqueeze_228 = _shape_param_360 = None
        permute_220: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_321, [0, 1, 3, 4, 2]);  view_321 = None
        view_322: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_220, _shape_param_361);  permute_220 = _shape_param_361 = None
        convert_element_type_198: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg139_1, torch.bfloat16);  arg139_1 = None
        unsqueeze_229: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_198, 3);  convert_element_type_198 = None
        unsqueeze_230: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 4);  unsqueeze_229 = None
        view_323: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_230, _shape_param_362);  unsqueeze_230 = _shape_param_362 = None
        squeeze_67: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_323, 0)
        mm_47: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_64, squeeze_67);  squeeze_64 = squeeze_67 = None
        unsqueeze_231: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_47, 0);  mm_47 = None
        view_324: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_231, _shape_param_363);  unsqueeze_231 = _shape_param_363 = None
        permute_221: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_324, [0, 1, 3, 4, 2]);  view_324 = None
        view_325: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_221, _shape_param_364);  permute_221 = _shape_param_364 = None
        convert_element_type_199: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.bfloat16);  arg140_1 = None
        unsqueeze_232: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_199, 3);  convert_element_type_199 = None
        unsqueeze_233: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 4);  unsqueeze_232 = None
        view_326: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_233, _shape_param_365);  unsqueeze_233 = _shape_param_365 = None
        squeeze_68: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_326, 0);  view_326 = None
        mm_48: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_68);  squeeze_68 = None
        unsqueeze_234: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_48, 0);  mm_48 = None
        view_327: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_234, _shape_param_366);  unsqueeze_234 = _shape_param_366 = None
        permute_222: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_327, [0, 1, 3, 4, 2]);  view_327 = None
        view_328: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_222, _shape_param_367);  permute_222 = _shape_param_367 = None
        add_99: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_319, arg141_1);  arg141_1 = None
        convert_element_type_200: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        unsqueeze_235: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_200, 4);  convert_element_type_200 = None
        permute_223: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_235, [1, 2, 0, 4, 3]);  unsqueeze_235 = None
        unsqueeze_236: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_322, 4);  view_322 = None
        permute_224: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_236, [1, 2, 4, 0, 3]);  unsqueeze_236 = None
        permute_225: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_223, [0, 1, 2, 4, 3]);  permute_223 = None
        view_329: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_225, _shape_param_368);  permute_225 = _shape_param_368 = None
        permute_226: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_224, [0, 1, 4, 3, 2]);  permute_224 = None
        view_330: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_226, _shape_param_369);  permute_226 = _shape_param_369 = None
        bmm_27: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_329, view_330)
        view_331: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_27, _shape_param_370);  bmm_27 = _shape_param_370 = None
        permute_227: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_331, [0, 1, 2, 4, 3]);  view_331 = None
        view_332: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_227, _shape_param_371);  permute_227 = _shape_param_371 = None
        add_100: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_319, arg142_1);  view_319 = arg142_1 = None
        convert_element_type_201: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None
        unsqueeze_237: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_201, 4);  convert_element_type_201 = None
        permute_228: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_237, [1, 2, 0, 4, 3]);  unsqueeze_237 = None
        unsqueeze_238: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_328, 4);  view_328 = None
        permute_229: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_238, [1, 2, 4, 0, 3]);  unsqueeze_238 = None
        permute_230: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_228, [0, 1, 2, 4, 3]);  permute_228 = None
        view_333: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_230, _shape_param_372);  permute_230 = _shape_param_372 = None
        permute_231: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_229, [0, 1, 4, 3, 2]);  permute_229 = None
        view_334: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_231, _shape_param_373);  permute_231 = _shape_param_373 = None
        bmm_28: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_333, view_334)
        view_335: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_28, _shape_param_374);  bmm_28 = _shape_param_374 = None
        permute_232: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_335, [0, 1, 2, 4, 3]);  view_335 = None
        view_336: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_232, _shape_param_375);  permute_232 = _shape_param_375 = None
        view_337: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_336, _shape_param_376);  view_336 = _shape_param_376 = None
        slice_10: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_337, 2, 1, 9223372036854775807);  view_337 = None
        view_338: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_10, _shape_param_377);  slice_10 = _shape_param_377 = None
        index_9: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_338, [None, None, None, iota_2]);  view_338 = None
        add_101: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_332, index_9);  view_332 = index_9 = None
        add_102: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_101, 0);  add_101 = None
        mul_177: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_102, 0.125)
        convert_element_type_202: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_177, torch.float32);  mul_177 = None
        convert_element_type_203: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_102, torch.float32)
        mul_178: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_203, 1);  convert_element_type_203 = None
        amax_18: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_178, [3], True)
        sub_36: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_178, amax_18);  mul_178 = None
        mul_179: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_36, 0.125);  sub_36 = None
        amax_19: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_202, [3], True)
        sub_37: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_202, amax_19)
        abs_10: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_202)
        ne_9: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_10, inf);  abs_10 = None
        eq_9: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_202, convert_element_type_202);  convert_element_type_202 = None
        mul_180: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_9, ne_9);  eq_9 = ne_9 = None
        logical_not_18: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_180);  mul_180 = None
        any_10: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_18, [3], True);  logical_not_18 = None
        logical_not_19: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_9: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_19, mul_179, sub_37);  mul_179 = sub_37 = None
        exp_9: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_9);  where_9 = None
        sum_10: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [3], True)
        div_10: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
        inductor_lookup_seed_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 38)
        inductor_random_38: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_378, inductor_lookup_seed_38, 'rand');  _shape_param_378 = inductor_lookup_seed_38 = None
        gt_38: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_38, 0.1);  inductor_random_38 = None
        mul_181: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_38, div_10);  div_10 = None
        mul_182: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_181, 1.1111111111111112);  mul_181 = None
        convert_element_type_204: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None
        unsqueeze_239: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_204, 4);  convert_element_type_204 = None
        unsqueeze_240: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_325, 4);  view_325 = None
        permute_233: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_240, [4, 1, 2, 3, 0]);  unsqueeze_240 = None
        view_339: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_239, _shape_param_379);  unsqueeze_239 = _shape_param_379 = None
        permute_234: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_233, [1, 2, 4, 3, 0]);  permute_233 = None
        view_340: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_234, _shape_param_380);  permute_234 = _shape_param_380 = None
        bmm_29: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_339, view_340)
        view_341: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_29, _shape_param_381);  bmm_29 = _shape_param_381 = None
        permute_235: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_341, [2, 0, 1, 4, 3]);  view_341 = None
        view_342: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_235, _shape_param_382);  permute_235 = _shape_param_382 = None
        convert_element_type_205: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.bfloat16);  arg143_1 = None
        unsqueeze_241: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_342, 4);  view_342 = None
        permute_236: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_241, [0, 1, 4, 3, 2]);  unsqueeze_241 = None
        unsqueeze_242: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_205, 3);  convert_element_type_205 = None
        unsqueeze_243: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 4);  unsqueeze_242 = None
        permute_237: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_243, [3, 4, 0, 2, 1]);  unsqueeze_243 = None
        permute_238: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 4, 2]);  permute_236 = None
        clone_19: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_343: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_19, _shape_param_383);  clone_19 = _shape_param_383 = None
        permute_239: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_237, [3, 4, 2, 0, 1]);  permute_237 = None
        clone_20: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_344: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_20, _shape_param_384);  clone_20 = _shape_param_384 = None
        squeeze_69: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_343, 0)
        squeeze_70: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_344, 0)
        mm_49: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_69, squeeze_70);  squeeze_69 = squeeze_70 = None
        unsqueeze_244: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_49, 0);  mm_49 = None
        view_345: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_244, _shape_param_385);  unsqueeze_244 = _shape_param_385 = None
        permute_240: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_345, [0, 1, 4, 2, 3]);  view_345 = None
        view_346: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_240, _shape_param_386);  permute_240 = _shape_param_386 = None
        inductor_lookup_seed_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 39)
        inductor_random_39: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_387, inductor_lookup_seed_39, 'rand');  _shape_param_387 = inductor_lookup_seed_39 = None
        convert_element_type_206: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_39, torch.bfloat16);  inductor_random_39 = None
        gt_39: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_206, 0.1);  convert_element_type_206 = None
        mul_183: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_39, view_346);  view_346 = None
        mul_184: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_183, 1.1111111111111112);  mul_183 = None
        add_103: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_184, add_98);  mul_184 = add_98 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_36: "f32[512, 16, 1]" = var_mean_18[0]
        getitem_37: "f32[512, 16, 1]" = var_mean_18[1];  var_mean_18 = None
        add_104: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_38: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_103, getitem_37);  add_103 = getitem_37 = None
        mul_185: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_18);  sub_38 = None
        mul_186: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_185, arg144_1);  arg144_1 = None
        add_105: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_186, arg145_1);  mul_186 = arg145_1 = None
        convert_element_type_207: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.bfloat16);  arg146_1 = None
        convert_element_type_208: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.bfloat16);  arg147_1 = None
        convert_element_type_209: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16)
        view_347: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_209, _shape_param_388);  convert_element_type_209 = _shape_param_388 = None
        permute_241: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_208, [1, 0]);  convert_element_type_208 = None
        addmm_18: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_207, view_347, permute_241);  convert_element_type_207 = None
        view_348: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_18, _shape_param_389);  _shape_param_389 = None
        convert_element_type_210: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_348, torch.float32);  view_348 = None
        mul_187: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.5)
        mul_188: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.7071067811865476);  convert_element_type_210 = None
        erf_9: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
        add_106: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_189: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_187, add_106);  mul_187 = add_106 = None
        convert_element_type_211: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        inductor_lookup_seed_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 40)
        inductor_random_40: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_390, inductor_lookup_seed_40, 'rand');  _shape_param_390 = inductor_lookup_seed_40 = None
        convert_element_type_212: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_40, torch.bfloat16);  inductor_random_40 = None
        gt_40: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_212, 0.1);  convert_element_type_212 = None
        mul_190: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_40, convert_element_type_211);  convert_element_type_211 = None
        mul_191: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_190, 1.1111111111111112);  mul_190 = None
        convert_element_type_213: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.bfloat16);  arg148_1 = None
        convert_element_type_214: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg149_1, torch.bfloat16);  arg149_1 = None
        view_349: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_191, _shape_param_391);  mul_191 = _shape_param_391 = None
        permute_242: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_214, [1, 0]);  convert_element_type_214 = None
        addmm_19: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_213, view_349, permute_242);  convert_element_type_213 = None
        view_350: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_19, _shape_param_392);  addmm_19 = _shape_param_392 = None
        inductor_lookup_seed_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 41)
        inductor_random_41: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_393, inductor_lookup_seed_41, 'rand');  _shape_param_393 = inductor_lookup_seed_41 = None
        convert_element_type_215: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_41, torch.bfloat16);  inductor_random_41 = None
        gt_41: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_215, 0.1);  convert_element_type_215 = None
        mul_192: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_41, view_350);  view_350 = None
        mul_193: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_192, 1.1111111111111112);  mul_192 = None
        add_107: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_193, add_105);  mul_193 = add_105 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
        getitem_38: "f32[512, 16, 1]" = var_mean_19[0]
        getitem_39: "f32[512, 16, 1]" = var_mean_19[1];  var_mean_19 = None
        add_108: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_39: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_107, getitem_39);  add_107 = getitem_39 = None
        mul_194: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_19);  sub_39 = None
        mul_195: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_194, arg150_1);  arg150_1 = None
        add_109: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_195, arg151_1);  mul_195 = arg151_1 = None
        convert_element_type_216: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16)
        convert_element_type_217: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.bfloat16);  arg152_1 = None
        unsqueeze_245: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_216, 3);  convert_element_type_216 = None
        unsqueeze_246: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 4);  unsqueeze_245 = None
        unsqueeze_247: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_217, 3);  convert_element_type_217 = None
        unsqueeze_248: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 4);  unsqueeze_247 = None
        view_351: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_246, _shape_param_394);  unsqueeze_246 = _shape_param_394 = None
        view_352: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_248, _shape_param_395);  unsqueeze_248 = _shape_param_395 = None
        squeeze_71: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_351, 0)
        squeeze_72: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_352, 0)
        mm_50: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_71, squeeze_72);  squeeze_72 = None
        unsqueeze_249: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_50, 0);  mm_50 = None
        view_353: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_249, _shape_param_396);  unsqueeze_249 = _shape_param_396 = None
        permute_243: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_353, [0, 1, 3, 4, 2]);  view_353 = None
        view_354: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_243, _shape_param_397);  permute_243 = _shape_param_397 = None
        convert_element_type_218: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.bfloat16);  arg153_1 = None
        unsqueeze_250: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_218, 3);  convert_element_type_218 = None
        unsqueeze_251: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 4);  unsqueeze_250 = None
        view_355: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_251, _shape_param_398);  unsqueeze_251 = _shape_param_398 = None
        squeeze_73: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_355, 0)
        mm_51: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_71, squeeze_73);  squeeze_73 = None
        unsqueeze_252: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_51, 0);  mm_51 = None
        view_356: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_252, _shape_param_399);  unsqueeze_252 = _shape_param_399 = None
        permute_244: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_356, [0, 1, 3, 4, 2]);  view_356 = None
        view_357: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_244, _shape_param_400);  permute_244 = _shape_param_400 = None
        convert_element_type_219: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.bfloat16);  arg154_1 = None
        unsqueeze_253: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_219, 3);  convert_element_type_219 = None
        unsqueeze_254: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 4);  unsqueeze_253 = None
        view_358: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_254, _shape_param_401);  unsqueeze_254 = _shape_param_401 = None
        squeeze_74: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_358, 0)
        mm_52: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_71, squeeze_74);  squeeze_71 = squeeze_74 = None
        unsqueeze_255: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_52, 0);  mm_52 = None
        view_359: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_255, _shape_param_402);  unsqueeze_255 = _shape_param_402 = None
        permute_245: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_359, [0, 1, 3, 4, 2]);  view_359 = None
        view_360: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_245, _shape_param_403);  permute_245 = _shape_param_403 = None
        convert_element_type_220: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.bfloat16);  arg155_1 = None
        unsqueeze_256: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_220, 3);  convert_element_type_220 = None
        unsqueeze_257: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 4);  unsqueeze_256 = None
        view_361: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_257, _shape_param_404);  unsqueeze_257 = _shape_param_404 = None
        squeeze_75: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_361, 0);  view_361 = None
        mm_53: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_75);  squeeze_75 = None
        unsqueeze_258: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_53, 0);  mm_53 = None
        view_362: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_258, _shape_param_405);  unsqueeze_258 = _shape_param_405 = None
        permute_246: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_362, [0, 1, 3, 4, 2]);  view_362 = None
        view_363: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_246, _shape_param_406);  permute_246 = _shape_param_406 = None
        add_110: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_354, arg156_1);  arg156_1 = None
        convert_element_type_221: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_110, torch.bfloat16);  add_110 = None
        unsqueeze_259: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_221, 4);  convert_element_type_221 = None
        permute_247: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_259, [1, 2, 0, 4, 3]);  unsqueeze_259 = None
        unsqueeze_260: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_357, 4);  view_357 = None
        permute_248: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_260, [1, 2, 4, 0, 3]);  unsqueeze_260 = None
        permute_249: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_247, [0, 1, 2, 4, 3]);  permute_247 = None
        view_364: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_249, _shape_param_407);  permute_249 = _shape_param_407 = None
        permute_250: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_248, [0, 1, 4, 3, 2]);  permute_248 = None
        view_365: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_250, _shape_param_408);  permute_250 = _shape_param_408 = None
        bmm_30: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_364, view_365)
        view_366: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_30, _shape_param_409);  bmm_30 = _shape_param_409 = None
        permute_251: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_366, [0, 1, 2, 4, 3]);  view_366 = None
        view_367: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_251, _shape_param_410);  permute_251 = _shape_param_410 = None
        add_111: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_354, arg157_1);  view_354 = arg157_1 = None
        convert_element_type_222: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None
        unsqueeze_261: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_222, 4);  convert_element_type_222 = None
        permute_252: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_261, [1, 2, 0, 4, 3]);  unsqueeze_261 = None
        unsqueeze_262: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_363, 4);  view_363 = None
        permute_253: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_262, [1, 2, 4, 0, 3]);  unsqueeze_262 = None
        permute_254: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_252, [0, 1, 2, 4, 3]);  permute_252 = None
        view_368: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_254, _shape_param_411);  permute_254 = _shape_param_411 = None
        permute_255: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_253, [0, 1, 4, 3, 2]);  permute_253 = None
        view_369: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_255, _shape_param_412);  permute_255 = _shape_param_412 = None
        bmm_31: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_368, view_369)
        view_370: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_31, _shape_param_413);  bmm_31 = _shape_param_413 = None
        permute_256: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_370, [0, 1, 2, 4, 3]);  view_370 = None
        view_371: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_256, _shape_param_414);  permute_256 = _shape_param_414 = None
        view_372: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_371, _shape_param_415);  view_371 = _shape_param_415 = None
        slice_11: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_372, 2, 1, 9223372036854775807);  view_372 = None
        view_373: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_11, _shape_param_416);  slice_11 = _shape_param_416 = None
        index_10: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_373, [None, None, None, iota_2]);  view_373 = None
        add_112: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_367, index_10);  view_367 = index_10 = None
        add_113: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_112, 0);  add_112 = None
        mul_196: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_113, 0.125)
        convert_element_type_223: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_196, torch.float32);  mul_196 = None
        convert_element_type_224: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_113, torch.float32)
        mul_197: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_224, 1);  convert_element_type_224 = None
        amax_20: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_197, [3], True)
        sub_40: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_197, amax_20);  mul_197 = None
        mul_198: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_40, 0.125);  sub_40 = None
        amax_21: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_223, [3], True)
        sub_41: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_223, amax_21)
        abs_11: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_223)
        ne_10: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_11, inf);  abs_11 = None
        eq_10: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_223, convert_element_type_223);  convert_element_type_223 = None
        mul_199: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_10, ne_10);  eq_10 = ne_10 = None
        logical_not_20: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_199);  mul_199 = None
        any_11: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_20, [3], True);  logical_not_20 = None
        logical_not_21: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_10: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_21, mul_198, sub_41);  mul_198 = sub_41 = None
        exp_10: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_10);  where_10 = None
        sum_11: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [3], True)
        div_11: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None
        inductor_lookup_seed_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 42)
        inductor_random_42: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_417, inductor_lookup_seed_42, 'rand');  _shape_param_417 = inductor_lookup_seed_42 = None
        gt_42: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_42, 0.1);  inductor_random_42 = None
        mul_200: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_42, div_11);  div_11 = None
        mul_201: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_200, 1.1111111111111112);  mul_200 = None
        convert_element_type_225: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None
        unsqueeze_263: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_225, 4);  convert_element_type_225 = None
        unsqueeze_264: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_360, 4);  view_360 = None
        permute_257: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_264, [4, 1, 2, 3, 0]);  unsqueeze_264 = None
        view_374: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_263, _shape_param_418);  unsqueeze_263 = _shape_param_418 = None
        permute_258: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_257, [1, 2, 4, 3, 0]);  permute_257 = None
        view_375: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_258, _shape_param_419);  permute_258 = _shape_param_419 = None
        bmm_32: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_374, view_375)
        view_376: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_32, _shape_param_420);  bmm_32 = _shape_param_420 = None
        permute_259: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_376, [2, 0, 1, 4, 3]);  view_376 = None
        view_377: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_259, _shape_param_421);  permute_259 = _shape_param_421 = None
        convert_element_type_226: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.bfloat16);  arg158_1 = None
        unsqueeze_265: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_377, 4);  view_377 = None
        permute_260: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_265, [0, 1, 4, 3, 2]);  unsqueeze_265 = None
        unsqueeze_266: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_226, 3);  convert_element_type_226 = None
        unsqueeze_267: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 4);  unsqueeze_266 = None
        permute_261: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_267, [3, 4, 0, 2, 1]);  unsqueeze_267 = None
        permute_262: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_260, [0, 1, 3, 4, 2]);  permute_260 = None
        clone_21: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_262, memory_format = torch.contiguous_format);  permute_262 = None
        view_378: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_21, _shape_param_422);  clone_21 = _shape_param_422 = None
        permute_263: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_261, [3, 4, 2, 0, 1]);  permute_261 = None
        clone_22: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        view_379: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_22, _shape_param_423);  clone_22 = _shape_param_423 = None
        squeeze_76: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_378, 0)
        squeeze_77: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_379, 0)
        mm_54: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_76, squeeze_77);  squeeze_76 = squeeze_77 = None
        unsqueeze_268: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_54, 0);  mm_54 = None
        view_380: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_268, _shape_param_424);  unsqueeze_268 = _shape_param_424 = None
        permute_264: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_380, [0, 1, 4, 2, 3]);  view_380 = None
        view_381: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_264, _shape_param_425);  permute_264 = _shape_param_425 = None
        inductor_lookup_seed_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 43)
        inductor_random_43: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_426, inductor_lookup_seed_43, 'rand');  _shape_param_426 = inductor_lookup_seed_43 = None
        convert_element_type_227: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_43, torch.bfloat16);  inductor_random_43 = None
        gt_43: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_227, 0.1);  convert_element_type_227 = None
        mul_202: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_43, view_381);  view_381 = None
        mul_203: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_202, 1.1111111111111112);  mul_202 = None
        add_114: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_203, add_109);  mul_203 = add_109 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
        getitem_40: "f32[512, 16, 1]" = var_mean_20[0]
        getitem_41: "f32[512, 16, 1]" = var_mean_20[1];  var_mean_20 = None
        add_115: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_42: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_114, getitem_41);  add_114 = getitem_41 = None
        mul_204: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_20);  sub_42 = None
        mul_205: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_204, arg159_1);  arg159_1 = None
        add_116: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_205, arg160_1);  mul_205 = arg160_1 = None
        convert_element_type_228: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg161_1, torch.bfloat16);  arg161_1 = None
        convert_element_type_229: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.bfloat16);  arg162_1 = None
        convert_element_type_230: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16)
        view_382: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_230, _shape_param_427);  convert_element_type_230 = _shape_param_427 = None
        permute_265: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_229, [1, 0]);  convert_element_type_229 = None
        addmm_20: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_228, view_382, permute_265);  convert_element_type_228 = None
        view_383: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_20, _shape_param_428);  _shape_param_428 = None
        convert_element_type_231: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_383, torch.float32);  view_383 = None
        mul_206: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_231, 0.5)
        mul_207: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_231, 0.7071067811865476);  convert_element_type_231 = None
        erf_10: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_117: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_208: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_206, add_117);  mul_206 = add_117 = None
        convert_element_type_232: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None
        inductor_lookup_seed_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 44)
        inductor_random_44: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_429, inductor_lookup_seed_44, 'rand');  _shape_param_429 = inductor_lookup_seed_44 = None
        convert_element_type_233: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_44, torch.bfloat16);  inductor_random_44 = None
        gt_44: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_233, 0.1);  convert_element_type_233 = None
        mul_209: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_44, convert_element_type_232);  convert_element_type_232 = None
        mul_210: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_209, 1.1111111111111112);  mul_209 = None
        convert_element_type_234: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.bfloat16);  arg163_1 = None
        convert_element_type_235: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg164_1, torch.bfloat16);  arg164_1 = None
        view_384: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_210, _shape_param_430);  mul_210 = _shape_param_430 = None
        permute_266: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_235, [1, 0]);  convert_element_type_235 = None
        addmm_21: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_234, view_384, permute_266);  convert_element_type_234 = None
        view_385: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_21, _shape_param_431);  addmm_21 = _shape_param_431 = None
        inductor_lookup_seed_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 45)
        inductor_random_45: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_432, inductor_lookup_seed_45, 'rand');  _shape_param_432 = inductor_lookup_seed_45 = None
        convert_element_type_236: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_45, torch.bfloat16);  inductor_random_45 = None
        gt_45: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_236, 0.1);  convert_element_type_236 = None
        mul_211: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_45, view_385);  view_385 = None
        mul_212: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_211, 1.1111111111111112);  mul_211 = None
        add_118: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_212, add_116);  mul_212 = add_116 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
        getitem_42: "f32[512, 16, 1]" = var_mean_21[0]
        getitem_43: "f32[512, 16, 1]" = var_mean_21[1];  var_mean_21 = None
        add_119: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_43: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_118, getitem_43);  add_118 = getitem_43 = None
        mul_213: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_21);  sub_43 = None
        mul_214: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_213, arg165_1);  arg165_1 = None
        add_120: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_214, arg166_1);  mul_214 = arg166_1 = None
        convert_element_type_237: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16)
        convert_element_type_238: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        unsqueeze_269: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_237, 3);  convert_element_type_237 = None
        unsqueeze_270: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 4);  unsqueeze_269 = None
        unsqueeze_271: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_238, 3);  convert_element_type_238 = None
        unsqueeze_272: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 4);  unsqueeze_271 = None
        view_386: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_270, _shape_param_433);  unsqueeze_270 = _shape_param_433 = None
        view_387: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_272, _shape_param_434);  unsqueeze_272 = _shape_param_434 = None
        squeeze_78: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_386, 0)
        squeeze_79: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_387, 0)
        mm_55: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_78, squeeze_79);  squeeze_79 = None
        unsqueeze_273: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_55, 0);  mm_55 = None
        view_388: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_273, _shape_param_435);  unsqueeze_273 = _shape_param_435 = None
        permute_267: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_388, [0, 1, 3, 4, 2]);  view_388 = None
        view_389: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_267, _shape_param_436);  permute_267 = _shape_param_436 = None
        convert_element_type_239: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.bfloat16);  arg168_1 = None
        unsqueeze_274: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_239, 3);  convert_element_type_239 = None
        unsqueeze_275: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 4);  unsqueeze_274 = None
        view_390: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_275, _shape_param_437);  unsqueeze_275 = _shape_param_437 = None
        squeeze_80: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_390, 0)
        mm_56: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_78, squeeze_80);  squeeze_80 = None
        unsqueeze_276: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_56, 0);  mm_56 = None
        view_391: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_276, _shape_param_438);  unsqueeze_276 = _shape_param_438 = None
        permute_268: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_391, [0, 1, 3, 4, 2]);  view_391 = None
        view_392: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_268, _shape_param_439);  permute_268 = _shape_param_439 = None
        convert_element_type_240: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg169_1, torch.bfloat16);  arg169_1 = None
        unsqueeze_277: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_240, 3);  convert_element_type_240 = None
        unsqueeze_278: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 4);  unsqueeze_277 = None
        view_393: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_278, _shape_param_440);  unsqueeze_278 = _shape_param_440 = None
        squeeze_81: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_393, 0)
        mm_57: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_78, squeeze_81);  squeeze_78 = squeeze_81 = None
        unsqueeze_279: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_57, 0);  mm_57 = None
        view_394: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_279, _shape_param_441);  unsqueeze_279 = _shape_param_441 = None
        permute_269: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_394, [0, 1, 3, 4, 2]);  view_394 = None
        view_395: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_269, _shape_param_442);  permute_269 = _shape_param_442 = None
        convert_element_type_241: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.bfloat16);  arg170_1 = None
        unsqueeze_280: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_241, 3);  convert_element_type_241 = None
        unsqueeze_281: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 4);  unsqueeze_280 = None
        view_396: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_281, _shape_param_443);  unsqueeze_281 = _shape_param_443 = None
        squeeze_82: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_396, 0);  view_396 = None
        mm_58: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_82);  squeeze_82 = None
        unsqueeze_282: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_58, 0);  mm_58 = None
        view_397: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_282, _shape_param_444);  unsqueeze_282 = _shape_param_444 = None
        permute_270: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_397, [0, 1, 3, 4, 2]);  view_397 = None
        view_398: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_270, _shape_param_445);  permute_270 = _shape_param_445 = None
        add_121: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_389, arg171_1);  arg171_1 = None
        convert_element_type_242: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None
        unsqueeze_283: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_242, 4);  convert_element_type_242 = None
        permute_271: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_283, [1, 2, 0, 4, 3]);  unsqueeze_283 = None
        unsqueeze_284: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_392, 4);  view_392 = None
        permute_272: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_284, [1, 2, 4, 0, 3]);  unsqueeze_284 = None
        permute_273: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_271, [0, 1, 2, 4, 3]);  permute_271 = None
        view_399: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_273, _shape_param_446);  permute_273 = _shape_param_446 = None
        permute_274: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_272, [0, 1, 4, 3, 2]);  permute_272 = None
        view_400: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_274, _shape_param_447);  permute_274 = _shape_param_447 = None
        bmm_33: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_399, view_400)
        view_401: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_33, _shape_param_448);  bmm_33 = _shape_param_448 = None
        permute_275: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_401, [0, 1, 2, 4, 3]);  view_401 = None
        view_402: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_275, _shape_param_449);  permute_275 = _shape_param_449 = None
        add_122: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_389, arg172_1);  view_389 = arg172_1 = None
        convert_element_type_243: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16);  add_122 = None
        unsqueeze_285: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_243, 4);  convert_element_type_243 = None
        permute_276: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_285, [1, 2, 0, 4, 3]);  unsqueeze_285 = None
        unsqueeze_286: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_398, 4);  view_398 = None
        permute_277: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_286, [1, 2, 4, 0, 3]);  unsqueeze_286 = None
        permute_278: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_276, [0, 1, 2, 4, 3]);  permute_276 = None
        view_403: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_278, _shape_param_450);  permute_278 = _shape_param_450 = None
        permute_279: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_277, [0, 1, 4, 3, 2]);  permute_277 = None
        view_404: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_279, _shape_param_451);  permute_279 = _shape_param_451 = None
        bmm_34: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_403, view_404)
        view_405: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_34, _shape_param_452);  bmm_34 = _shape_param_452 = None
        permute_280: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_405, [0, 1, 2, 4, 3]);  view_405 = None
        view_406: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_280, _shape_param_453);  permute_280 = _shape_param_453 = None
        view_407: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_406, _shape_param_454);  view_406 = _shape_param_454 = None
        slice_12: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_407, 2, 1, 9223372036854775807);  view_407 = None
        view_408: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_12, _shape_param_455);  slice_12 = _shape_param_455 = None
        index_11: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_408, [None, None, None, iota_2]);  view_408 = None
        add_123: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_402, index_11);  view_402 = index_11 = None
        add_124: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_123, 0);  add_123 = None
        mul_215: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_124, 0.125)
        convert_element_type_244: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_215, torch.float32);  mul_215 = None
        convert_element_type_245: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_124, torch.float32)
        mul_216: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_245, 1);  convert_element_type_245 = None
        amax_22: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_216, [3], True)
        sub_44: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_216, amax_22);  mul_216 = None
        mul_217: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_44, 0.125);  sub_44 = None
        amax_23: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_244, [3], True)
        sub_45: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_244, amax_23)
        abs_12: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_244)
        ne_11: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_12, inf);  abs_12 = None
        eq_11: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_244, convert_element_type_244);  convert_element_type_244 = None
        mul_218: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_11, ne_11);  eq_11 = ne_11 = None
        logical_not_22: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_218);  mul_218 = None
        any_12: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_22, [3], True);  logical_not_22 = None
        logical_not_23: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_11: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_23, mul_217, sub_45);  mul_217 = sub_45 = None
        exp_11: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_11);  where_11 = None
        sum_12: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [3], True)
        div_12: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
        inductor_lookup_seed_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 46)
        inductor_random_46: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_456, inductor_lookup_seed_46, 'rand');  _shape_param_456 = inductor_lookup_seed_46 = None
        gt_46: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_46, 0.1);  inductor_random_46 = None
        mul_219: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_46, div_12);  div_12 = None
        mul_220: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_219, 1.1111111111111112);  mul_219 = None
        convert_element_type_246: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_220, torch.bfloat16);  mul_220 = None
        unsqueeze_287: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_246, 4);  convert_element_type_246 = None
        unsqueeze_288: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_395, 4);  view_395 = None
        permute_281: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_288, [4, 1, 2, 3, 0]);  unsqueeze_288 = None
        view_409: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_287, _shape_param_457);  unsqueeze_287 = _shape_param_457 = None
        permute_282: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_281, [1, 2, 4, 3, 0]);  permute_281 = None
        view_410: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_282, _shape_param_458);  permute_282 = _shape_param_458 = None
        bmm_35: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_409, view_410)
        view_411: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_459);  bmm_35 = _shape_param_459 = None
        permute_283: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_411, [2, 0, 1, 4, 3]);  view_411 = None
        view_412: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_283, _shape_param_460);  permute_283 = _shape_param_460 = None
        convert_element_type_247: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.bfloat16);  arg173_1 = None
        unsqueeze_289: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_412, 4);  view_412 = None
        permute_284: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_289, [0, 1, 4, 3, 2]);  unsqueeze_289 = None
        unsqueeze_290: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_247, 3);  convert_element_type_247 = None
        unsqueeze_291: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 4);  unsqueeze_290 = None
        permute_285: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_291, [3, 4, 0, 2, 1]);  unsqueeze_291 = None
        permute_286: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_284, [0, 1, 3, 4, 2]);  permute_284 = None
        clone_23: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
        view_413: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_23, _shape_param_461);  clone_23 = _shape_param_461 = None
        permute_287: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_285, [3, 4, 2, 0, 1]);  permute_285 = None
        clone_24: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_414: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_24, _shape_param_462);  clone_24 = _shape_param_462 = None
        squeeze_83: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_413, 0)
        squeeze_84: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_414, 0)
        mm_59: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_83, squeeze_84);  squeeze_83 = squeeze_84 = None
        unsqueeze_292: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_59, 0);  mm_59 = None
        view_415: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_292, _shape_param_463);  unsqueeze_292 = _shape_param_463 = None
        permute_288: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_415, [0, 1, 4, 2, 3]);  view_415 = None
        view_416: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_288, _shape_param_464);  permute_288 = _shape_param_464 = None
        inductor_lookup_seed_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 47)
        inductor_random_47: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_465, inductor_lookup_seed_47, 'rand');  _shape_param_465 = inductor_lookup_seed_47 = None
        convert_element_type_248: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_47, torch.bfloat16);  inductor_random_47 = None
        gt_47: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_248, 0.1);  convert_element_type_248 = None
        mul_221: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_47, view_416);  view_416 = None
        mul_222: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_221, 1.1111111111111112);  mul_221 = None
        add_125: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_222, add_120);  mul_222 = add_120 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_44: "f32[512, 16, 1]" = var_mean_22[0]
        getitem_45: "f32[512, 16, 1]" = var_mean_22[1];  var_mean_22 = None
        add_126: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_46: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_125, getitem_45);  add_125 = getitem_45 = None
        mul_223: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_22);  sub_46 = None
        mul_224: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_223, arg174_1);  arg174_1 = None
        add_127: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_224, arg175_1);  mul_224 = arg175_1 = None
        convert_element_type_249: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.bfloat16);  arg176_1 = None
        convert_element_type_250: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        convert_element_type_251: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16)
        view_417: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_251, _shape_param_466);  convert_element_type_251 = _shape_param_466 = None
        permute_289: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_250, [1, 0]);  convert_element_type_250 = None
        addmm_22: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_249, view_417, permute_289);  convert_element_type_249 = None
        view_418: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_22, _shape_param_467);  _shape_param_467 = None
        convert_element_type_252: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_418, torch.float32);  view_418 = None
        mul_225: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_252, 0.5)
        mul_226: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_252, 0.7071067811865476);  convert_element_type_252 = None
        erf_11: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_226);  mul_226 = None
        add_128: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_227: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_225, add_128);  mul_225 = add_128 = None
        convert_element_type_253: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_227, torch.bfloat16);  mul_227 = None
        inductor_lookup_seed_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 48)
        inductor_random_48: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_468, inductor_lookup_seed_48, 'rand');  _shape_param_468 = inductor_lookup_seed_48 = None
        convert_element_type_254: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_48, torch.bfloat16);  inductor_random_48 = None
        gt_48: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_254, 0.1);  convert_element_type_254 = None
        mul_228: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_48, convert_element_type_253);  convert_element_type_253 = None
        mul_229: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_228, 1.1111111111111112);  mul_228 = None
        convert_element_type_255: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg178_1, torch.bfloat16);  arg178_1 = None
        convert_element_type_256: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.bfloat16);  arg179_1 = None
        view_419: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_229, _shape_param_469);  mul_229 = _shape_param_469 = None
        permute_290: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_256, [1, 0]);  convert_element_type_256 = None
        addmm_23: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_255, view_419, permute_290);  convert_element_type_255 = None
        view_420: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_23, _shape_param_470);  addmm_23 = _shape_param_470 = None
        inductor_lookup_seed_49: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 49)
        inductor_random_49: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_471, inductor_lookup_seed_49, 'rand');  _shape_param_471 = inductor_lookup_seed_49 = None
        convert_element_type_257: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_49, torch.bfloat16);  inductor_random_49 = None
        gt_49: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_257, 0.1);  convert_element_type_257 = None
        mul_230: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_49, view_420);  view_420 = None
        mul_231: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_230, 1.1111111111111112);  mul_230 = None
        add_129: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_231, add_127);  mul_231 = add_127 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_46: "f32[512, 16, 1]" = var_mean_23[0]
        getitem_47: "f32[512, 16, 1]" = var_mean_23[1];  var_mean_23 = None
        add_130: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_47: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_129, getitem_47);  add_129 = getitem_47 = None
        mul_232: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_23);  sub_47 = None
        mul_233: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_232, arg180_1);  arg180_1 = None
        add_131: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_233, arg181_1);  mul_233 = arg181_1 = None
        convert_element_type_258: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16)
        convert_element_type_259: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        unsqueeze_293: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_258, 3);  convert_element_type_258 = None
        unsqueeze_294: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 4);  unsqueeze_293 = None
        unsqueeze_295: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_259, 3);  convert_element_type_259 = None
        unsqueeze_296: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 4);  unsqueeze_295 = None
        view_421: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_294, _shape_param_472);  unsqueeze_294 = _shape_param_472 = None
        view_422: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_296, _shape_param_473);  unsqueeze_296 = _shape_param_473 = None
        squeeze_85: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_421, 0)
        squeeze_86: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_422, 0)
        mm_60: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_85, squeeze_86);  squeeze_86 = None
        unsqueeze_297: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_60, 0);  mm_60 = None
        view_423: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_297, _shape_param_474);  unsqueeze_297 = _shape_param_474 = None
        permute_291: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_423, [0, 1, 3, 4, 2]);  view_423 = None
        view_424: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_291, _shape_param_475);  permute_291 = _shape_param_475 = None
        convert_element_type_260: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.bfloat16);  arg183_1 = None
        unsqueeze_298: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_260, 3);  convert_element_type_260 = None
        unsqueeze_299: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 4);  unsqueeze_298 = None
        view_425: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_299, _shape_param_476);  unsqueeze_299 = _shape_param_476 = None
        squeeze_87: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_425, 0)
        mm_61: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_85, squeeze_87);  squeeze_87 = None
        unsqueeze_300: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_61, 0);  mm_61 = None
        view_426: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_300, _shape_param_477);  unsqueeze_300 = _shape_param_477 = None
        permute_292: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_426, [0, 1, 3, 4, 2]);  view_426 = None
        view_427: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_292, _shape_param_478);  permute_292 = _shape_param_478 = None
        convert_element_type_261: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.bfloat16);  arg184_1 = None
        unsqueeze_301: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_261, 3);  convert_element_type_261 = None
        unsqueeze_302: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 4);  unsqueeze_301 = None
        view_428: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_302, _shape_param_479);  unsqueeze_302 = _shape_param_479 = None
        squeeze_88: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_428, 0)
        mm_62: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_85, squeeze_88);  squeeze_85 = squeeze_88 = None
        unsqueeze_303: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_62, 0);  mm_62 = None
        view_429: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_303, _shape_param_480);  unsqueeze_303 = _shape_param_480 = None
        permute_293: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_429, [0, 1, 3, 4, 2]);  view_429 = None
        view_430: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_293, _shape_param_481);  permute_293 = _shape_param_481 = None
        convert_element_type_262: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.bfloat16);  arg185_1 = None
        unsqueeze_304: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_262, 3);  convert_element_type_262 = None
        unsqueeze_305: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 4);  unsqueeze_304 = None
        view_431: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_305, _shape_param_482);  unsqueeze_305 = _shape_param_482 = None
        squeeze_89: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_431, 0);  view_431 = None
        mm_63: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_89);  squeeze_89 = None
        unsqueeze_306: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_63, 0);  mm_63 = None
        view_432: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_306, _shape_param_483);  unsqueeze_306 = _shape_param_483 = None
        permute_294: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_432, [0, 1, 3, 4, 2]);  view_432 = None
        view_433: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_294, _shape_param_484);  permute_294 = _shape_param_484 = None
        add_132: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_424, arg186_1);  arg186_1 = None
        convert_element_type_263: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_132, torch.bfloat16);  add_132 = None
        unsqueeze_307: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_263, 4);  convert_element_type_263 = None
        permute_295: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_307, [1, 2, 0, 4, 3]);  unsqueeze_307 = None
        unsqueeze_308: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_427, 4);  view_427 = None
        permute_296: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_308, [1, 2, 4, 0, 3]);  unsqueeze_308 = None
        permute_297: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_295, [0, 1, 2, 4, 3]);  permute_295 = None
        view_434: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_297, _shape_param_485);  permute_297 = _shape_param_485 = None
        permute_298: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_296, [0, 1, 4, 3, 2]);  permute_296 = None
        view_435: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_298, _shape_param_486);  permute_298 = _shape_param_486 = None
        bmm_36: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_434, view_435)
        view_436: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_36, _shape_param_487);  bmm_36 = _shape_param_487 = None
        permute_299: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_436, [0, 1, 2, 4, 3]);  view_436 = None
        view_437: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_299, _shape_param_488);  permute_299 = _shape_param_488 = None
        add_133: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_424, arg187_1);  view_424 = arg187_1 = None
        convert_element_type_264: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None
        unsqueeze_309: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_264, 4);  convert_element_type_264 = None
        permute_300: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_309, [1, 2, 0, 4, 3]);  unsqueeze_309 = None
        unsqueeze_310: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_433, 4);  view_433 = None
        permute_301: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_310, [1, 2, 4, 0, 3]);  unsqueeze_310 = None
        permute_302: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_300, [0, 1, 2, 4, 3]);  permute_300 = None
        view_438: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_302, _shape_param_489);  permute_302 = _shape_param_489 = None
        permute_303: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_301, [0, 1, 4, 3, 2]);  permute_301 = None
        view_439: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_303, _shape_param_490);  permute_303 = _shape_param_490 = None
        bmm_37: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_438, view_439)
        view_440: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_37, _shape_param_491);  bmm_37 = _shape_param_491 = None
        permute_304: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_440, [0, 1, 2, 4, 3]);  view_440 = None
        view_441: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_304, _shape_param_492);  permute_304 = _shape_param_492 = None
        view_442: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_441, _shape_param_493);  view_441 = _shape_param_493 = None
        slice_13: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_442, 2, 1, 9223372036854775807);  view_442 = None
        view_443: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_13, _shape_param_494);  slice_13 = _shape_param_494 = None
        index_12: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_443, [None, None, None, iota_2]);  view_443 = None
        add_134: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_437, index_12);  view_437 = index_12 = None
        add_135: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_134, 0);  add_134 = None
        mul_234: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_135, 0.125)
        convert_element_type_265: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_234, torch.float32);  mul_234 = None
        convert_element_type_266: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_135, torch.float32)
        mul_235: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_266, 1);  convert_element_type_266 = None
        amax_24: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_235, [3], True)
        sub_48: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_235, amax_24);  mul_235 = None
        mul_236: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_48, 0.125);  sub_48 = None
        amax_25: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_265, [3], True)
        sub_49: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_265, amax_25)
        abs_13: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_265)
        ne_12: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_13, inf);  abs_13 = None
        eq_12: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_265, convert_element_type_265);  convert_element_type_265 = None
        mul_237: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_12, ne_12);  eq_12 = ne_12 = None
        logical_not_24: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_237);  mul_237 = None
        any_13: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_24, [3], True);  logical_not_24 = None
        logical_not_25: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        where_12: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_25, mul_236, sub_49);  mul_236 = sub_49 = None
        exp_12: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_12);  where_12 = None
        sum_13: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [3], True)
        div_13: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None
        inductor_lookup_seed_50: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 50)
        inductor_random_50: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_495, inductor_lookup_seed_50, 'rand');  _shape_param_495 = inductor_lookup_seed_50 = None
        gt_50: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_50, 0.1);  inductor_random_50 = None
        mul_238: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_50, div_13);  div_13 = None
        mul_239: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_238, 1.1111111111111112);  mul_238 = None
        convert_element_type_267: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_239, torch.bfloat16);  mul_239 = None
        unsqueeze_311: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_267, 4);  convert_element_type_267 = None
        unsqueeze_312: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_430, 4);  view_430 = None
        permute_305: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_312, [4, 1, 2, 3, 0]);  unsqueeze_312 = None
        view_444: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_311, _shape_param_496);  unsqueeze_311 = _shape_param_496 = None
        permute_306: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_305, [1, 2, 4, 3, 0]);  permute_305 = None
        view_445: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_306, _shape_param_497);  permute_306 = _shape_param_497 = None
        bmm_38: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_444, view_445)
        view_446: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_38, _shape_param_498);  bmm_38 = _shape_param_498 = None
        permute_307: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_446, [2, 0, 1, 4, 3]);  view_446 = None
        view_447: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_307, _shape_param_499);  permute_307 = _shape_param_499 = None
        convert_element_type_268: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.bfloat16);  arg188_1 = None
        unsqueeze_313: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_447, 4);  view_447 = None
        permute_308: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_313, [0, 1, 4, 3, 2]);  unsqueeze_313 = None
        unsqueeze_314: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_268, 3);  convert_element_type_268 = None
        unsqueeze_315: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 4);  unsqueeze_314 = None
        permute_309: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_315, [3, 4, 0, 2, 1]);  unsqueeze_315 = None
        permute_310: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_308, [0, 1, 3, 4, 2]);  permute_308 = None
        clone_25: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
        view_448: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_25, _shape_param_500);  clone_25 = _shape_param_500 = None
        permute_311: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_309, [3, 4, 2, 0, 1]);  permute_309 = None
        clone_26: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_311, memory_format = torch.contiguous_format);  permute_311 = None
        view_449: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_26, _shape_param_501);  clone_26 = _shape_param_501 = None
        squeeze_90: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_448, 0)
        squeeze_91: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_449, 0)
        mm_64: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_90, squeeze_91);  squeeze_90 = squeeze_91 = None
        unsqueeze_316: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_64, 0);  mm_64 = None
        view_450: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_316, _shape_param_502);  unsqueeze_316 = _shape_param_502 = None
        permute_312: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_450, [0, 1, 4, 2, 3]);  view_450 = None
        view_451: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_312, _shape_param_503);  permute_312 = _shape_param_503 = None
        inductor_lookup_seed_51: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 51)
        inductor_random_51: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_504, inductor_lookup_seed_51, 'rand');  _shape_param_504 = inductor_lookup_seed_51 = None
        convert_element_type_269: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_51, torch.bfloat16);  inductor_random_51 = None
        gt_51: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_269, 0.1);  convert_element_type_269 = None
        mul_240: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_51, view_451);  view_451 = None
        mul_241: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_240, 1.1111111111111112);  mul_240 = None
        add_136: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_241, add_131);  mul_241 = add_131 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
        getitem_48: "f32[512, 16, 1]" = var_mean_24[0]
        getitem_49: "f32[512, 16, 1]" = var_mean_24[1];  var_mean_24 = None
        add_137: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_50: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_136, getitem_49);  add_136 = getitem_49 = None
        mul_242: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_24);  sub_50 = None
        mul_243: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_242, arg189_1);  arg189_1 = None
        add_138: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_243, arg190_1);  mul_243 = arg190_1 = None
        convert_element_type_270: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg191_1, torch.bfloat16);  arg191_1 = None
        convert_element_type_271: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.bfloat16);  arg192_1 = None
        convert_element_type_272: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16)
        view_452: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_272, _shape_param_505);  convert_element_type_272 = _shape_param_505 = None
        permute_313: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_271, [1, 0]);  convert_element_type_271 = None
        addmm_24: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_270, view_452, permute_313);  convert_element_type_270 = None
        view_453: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_24, _shape_param_506);  _shape_param_506 = None
        convert_element_type_273: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_453, torch.float32);  view_453 = None
        mul_244: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_273, 0.5)
        mul_245: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_273, 0.7071067811865476);  convert_element_type_273 = None
        erf_12: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_245);  mul_245 = None
        add_139: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_246: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_244, add_139);  mul_244 = add_139 = None
        convert_element_type_274: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        inductor_lookup_seed_52: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 52)
        inductor_random_52: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_507, inductor_lookup_seed_52, 'rand');  _shape_param_507 = inductor_lookup_seed_52 = None
        convert_element_type_275: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_52, torch.bfloat16);  inductor_random_52 = None
        gt_52: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_275, 0.1);  convert_element_type_275 = None
        mul_247: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_52, convert_element_type_274);  convert_element_type_274 = None
        mul_248: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_247, 1.1111111111111112);  mul_247 = None
        convert_element_type_276: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.bfloat16);  arg193_1 = None
        convert_element_type_277: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.bfloat16);  arg194_1 = None
        view_454: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_248, _shape_param_508);  mul_248 = _shape_param_508 = None
        permute_314: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_277, [1, 0]);  convert_element_type_277 = None
        addmm_25: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_276, view_454, permute_314);  convert_element_type_276 = None
        view_455: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_25, _shape_param_509);  addmm_25 = _shape_param_509 = None
        inductor_lookup_seed_53: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 53)
        inductor_random_53: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_510, inductor_lookup_seed_53, 'rand');  _shape_param_510 = inductor_lookup_seed_53 = None
        convert_element_type_278: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_53, torch.bfloat16);  inductor_random_53 = None
        gt_53: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_278, 0.1);  convert_element_type_278 = None
        mul_249: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_53, view_455);  view_455 = None
        mul_250: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_249, 1.1111111111111112);  mul_249 = None
        add_140: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_250, add_138);  mul_250 = add_138 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
        getitem_50: "f32[512, 16, 1]" = var_mean_25[0]
        getitem_51: "f32[512, 16, 1]" = var_mean_25[1];  var_mean_25 = None
        add_141: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_51: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_140, getitem_51);  add_140 = getitem_51 = None
        mul_251: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_25);  sub_51 = None
        mul_252: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_251, arg195_1);  arg195_1 = None
        add_142: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_252, arg196_1);  mul_252 = arg196_1 = None
        convert_element_type_279: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16)
        convert_element_type_280: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.bfloat16);  arg197_1 = None
        unsqueeze_317: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_279, 3);  convert_element_type_279 = None
        unsqueeze_318: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 4);  unsqueeze_317 = None
        unsqueeze_319: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_280, 3);  convert_element_type_280 = None
        unsqueeze_320: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 4);  unsqueeze_319 = None
        view_456: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_318, _shape_param_511);  unsqueeze_318 = _shape_param_511 = None
        view_457: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_320, _shape_param_512);  unsqueeze_320 = _shape_param_512 = None
        squeeze_92: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_456, 0)
        squeeze_93: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_457, 0)
        mm_65: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_92, squeeze_93);  squeeze_93 = None
        unsqueeze_321: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_65, 0);  mm_65 = None
        view_458: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_321, _shape_param_513);  unsqueeze_321 = _shape_param_513 = None
        permute_315: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 4, 2]);  view_458 = None
        view_459: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_315, _shape_param_514);  permute_315 = _shape_param_514 = None
        convert_element_type_281: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg198_1, torch.bfloat16);  arg198_1 = None
        unsqueeze_322: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_281, 3);  convert_element_type_281 = None
        unsqueeze_323: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 4);  unsqueeze_322 = None
        view_460: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_323, _shape_param_515);  unsqueeze_323 = _shape_param_515 = None
        squeeze_94: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_460, 0)
        mm_66: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_92, squeeze_94);  squeeze_94 = None
        unsqueeze_324: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_66, 0);  mm_66 = None
        view_461: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_324, _shape_param_516);  unsqueeze_324 = _shape_param_516 = None
        permute_316: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_461, [0, 1, 3, 4, 2]);  view_461 = None
        view_462: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_316, _shape_param_517);  permute_316 = _shape_param_517 = None
        convert_element_type_282: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.bfloat16);  arg199_1 = None
        unsqueeze_325: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_282, 3);  convert_element_type_282 = None
        unsqueeze_326: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 4);  unsqueeze_325 = None
        view_463: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_326, _shape_param_518);  unsqueeze_326 = _shape_param_518 = None
        squeeze_95: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_463, 0)
        mm_67: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_92, squeeze_95);  squeeze_92 = squeeze_95 = None
        unsqueeze_327: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_67, 0);  mm_67 = None
        view_464: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_327, _shape_param_519);  unsqueeze_327 = _shape_param_519 = None
        permute_317: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_464, [0, 1, 3, 4, 2]);  view_464 = None
        view_465: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_317, _shape_param_520);  permute_317 = _shape_param_520 = None
        convert_element_type_283: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.bfloat16);  arg200_1 = None
        unsqueeze_328: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_283, 3);  convert_element_type_283 = None
        unsqueeze_329: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 4);  unsqueeze_328 = None
        view_466: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_329, _shape_param_521);  unsqueeze_329 = _shape_param_521 = None
        squeeze_96: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_466, 0);  view_466 = None
        mm_68: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_96);  squeeze_96 = None
        unsqueeze_330: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_68, 0);  mm_68 = None
        view_467: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_330, _shape_param_522);  unsqueeze_330 = _shape_param_522 = None
        permute_318: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_467, [0, 1, 3, 4, 2]);  view_467 = None
        view_468: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_318, _shape_param_523);  permute_318 = _shape_param_523 = None
        add_143: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_459, arg201_1);  arg201_1 = None
        convert_element_type_284: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        unsqueeze_331: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_284, 4);  convert_element_type_284 = None
        permute_319: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_331, [1, 2, 0, 4, 3]);  unsqueeze_331 = None
        unsqueeze_332: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_462, 4);  view_462 = None
        permute_320: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_332, [1, 2, 4, 0, 3]);  unsqueeze_332 = None
        permute_321: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_319, [0, 1, 2, 4, 3]);  permute_319 = None
        view_469: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_321, _shape_param_524);  permute_321 = _shape_param_524 = None
        permute_322: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_320, [0, 1, 4, 3, 2]);  permute_320 = None
        view_470: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_322, _shape_param_525);  permute_322 = _shape_param_525 = None
        bmm_39: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_469, view_470)
        view_471: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_39, _shape_param_526);  bmm_39 = _shape_param_526 = None
        permute_323: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_471, [0, 1, 2, 4, 3]);  view_471 = None
        view_472: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_323, _shape_param_527);  permute_323 = _shape_param_527 = None
        add_144: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_459, arg202_1);  view_459 = arg202_1 = None
        convert_element_type_285: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None
        unsqueeze_333: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_285, 4);  convert_element_type_285 = None
        permute_324: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_333, [1, 2, 0, 4, 3]);  unsqueeze_333 = None
        unsqueeze_334: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_468, 4);  view_468 = None
        permute_325: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_334, [1, 2, 4, 0, 3]);  unsqueeze_334 = None
        permute_326: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_324, [0, 1, 2, 4, 3]);  permute_324 = None
        view_473: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_326, _shape_param_528);  permute_326 = _shape_param_528 = None
        permute_327: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_325, [0, 1, 4, 3, 2]);  permute_325 = None
        view_474: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_327, _shape_param_529);  permute_327 = _shape_param_529 = None
        bmm_40: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_473, view_474)
        view_475: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_40, _shape_param_530);  bmm_40 = _shape_param_530 = None
        permute_328: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_475, [0, 1, 2, 4, 3]);  view_475 = None
        view_476: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_328, _shape_param_531);  permute_328 = _shape_param_531 = None
        view_477: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_476, _shape_param_532);  view_476 = _shape_param_532 = None
        slice_14: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_477, 2, 1, 9223372036854775807);  view_477 = None
        view_478: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_14, _shape_param_533);  slice_14 = _shape_param_533 = None
        index_13: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_478, [None, None, None, iota_2]);  view_478 = None
        add_145: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_472, index_13);  view_472 = index_13 = None
        add_146: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_145, 0);  add_145 = None
        mul_253: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_146, 0.125)
        convert_element_type_286: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_253, torch.float32);  mul_253 = None
        convert_element_type_287: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_146, torch.float32)
        mul_254: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_287, 1);  convert_element_type_287 = None
        amax_26: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_254, [3], True)
        sub_52: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_254, amax_26);  mul_254 = None
        mul_255: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_52, 0.125);  sub_52 = None
        amax_27: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_286, [3], True)
        sub_53: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_286, amax_27)
        abs_14: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_286)
        ne_13: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_14, inf);  abs_14 = None
        eq_13: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_286, convert_element_type_286);  convert_element_type_286 = None
        mul_256: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_13, ne_13);  eq_13 = ne_13 = None
        logical_not_26: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_256);  mul_256 = None
        any_14: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_26, [3], True);  logical_not_26 = None
        logical_not_27: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        where_13: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_27, mul_255, sub_53);  mul_255 = sub_53 = None
        exp_13: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_13);  where_13 = None
        sum_14: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [3], True)
        div_14: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None
        inductor_lookup_seed_54: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 54)
        inductor_random_54: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_534, inductor_lookup_seed_54, 'rand');  _shape_param_534 = inductor_lookup_seed_54 = None
        gt_54: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_54, 0.1);  inductor_random_54 = None
        mul_257: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_54, div_14);  div_14 = None
        mul_258: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_257, 1.1111111111111112);  mul_257 = None
        convert_element_type_288: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_258, torch.bfloat16);  mul_258 = None
        unsqueeze_335: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_288, 4);  convert_element_type_288 = None
        unsqueeze_336: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_465, 4);  view_465 = None
        permute_329: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_336, [4, 1, 2, 3, 0]);  unsqueeze_336 = None
        view_479: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_335, _shape_param_535);  unsqueeze_335 = _shape_param_535 = None
        permute_330: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_329, [1, 2, 4, 3, 0]);  permute_329 = None
        view_480: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_330, _shape_param_536);  permute_330 = _shape_param_536 = None
        bmm_41: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_479, view_480)
        view_481: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_41, _shape_param_537);  bmm_41 = _shape_param_537 = None
        permute_331: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_481, [2, 0, 1, 4, 3]);  view_481 = None
        view_482: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_331, _shape_param_538);  permute_331 = _shape_param_538 = None
        convert_element_type_289: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg203_1, torch.bfloat16);  arg203_1 = None
        unsqueeze_337: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_482, 4);  view_482 = None
        permute_332: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_337, [0, 1, 4, 3, 2]);  unsqueeze_337 = None
        unsqueeze_338: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_289, 3);  convert_element_type_289 = None
        unsqueeze_339: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 4);  unsqueeze_338 = None
        permute_333: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_339, [3, 4, 0, 2, 1]);  unsqueeze_339 = None
        permute_334: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_332, [0, 1, 3, 4, 2]);  permute_332 = None
        clone_27: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
        view_483: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_27, _shape_param_539);  clone_27 = _shape_param_539 = None
        permute_335: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_333, [3, 4, 2, 0, 1]);  permute_333 = None
        clone_28: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_484: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_28, _shape_param_540);  clone_28 = _shape_param_540 = None
        squeeze_97: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_483, 0)
        squeeze_98: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_484, 0)
        mm_69: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_97, squeeze_98);  squeeze_97 = squeeze_98 = None
        unsqueeze_340: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_69, 0);  mm_69 = None
        view_485: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_340, _shape_param_541);  unsqueeze_340 = _shape_param_541 = None
        permute_336: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_485, [0, 1, 4, 2, 3]);  view_485 = None
        view_486: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_336, _shape_param_542);  permute_336 = _shape_param_542 = None
        inductor_lookup_seed_55: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 55)
        inductor_random_55: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_543, inductor_lookup_seed_55, 'rand');  _shape_param_543 = inductor_lookup_seed_55 = None
        convert_element_type_290: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_55, torch.bfloat16);  inductor_random_55 = None
        gt_55: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_290, 0.1);  convert_element_type_290 = None
        mul_259: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_55, view_486);  view_486 = None
        mul_260: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_259, 1.1111111111111112);  mul_259 = None
        add_147: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_260, add_142);  mul_260 = add_142 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_52: "f32[512, 16, 1]" = var_mean_26[0]
        getitem_53: "f32[512, 16, 1]" = var_mean_26[1];  var_mean_26 = None
        add_148: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_54: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_147, getitem_53);  add_147 = getitem_53 = None
        mul_261: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_26);  sub_54 = None
        mul_262: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_261, arg204_1);  arg204_1 = None
        add_149: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_262, arg205_1);  mul_262 = arg205_1 = None
        convert_element_type_291: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg206_1, torch.bfloat16);  arg206_1 = None
        convert_element_type_292: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.bfloat16);  arg207_1 = None
        convert_element_type_293: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16)
        view_487: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_293, _shape_param_544);  convert_element_type_293 = _shape_param_544 = None
        permute_337: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_292, [1, 0]);  convert_element_type_292 = None
        addmm_26: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_291, view_487, permute_337);  convert_element_type_291 = None
        view_488: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_26, _shape_param_545);  _shape_param_545 = None
        convert_element_type_294: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_488, torch.float32);  view_488 = None
        mul_263: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_294, 0.5)
        mul_264: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_294, 0.7071067811865476);  convert_element_type_294 = None
        erf_13: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_264);  mul_264 = None
        add_150: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_265: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_263, add_150);  mul_263 = add_150 = None
        convert_element_type_295: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_265, torch.bfloat16);  mul_265 = None
        inductor_lookup_seed_56: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 56)
        inductor_random_56: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_546, inductor_lookup_seed_56, 'rand');  _shape_param_546 = inductor_lookup_seed_56 = None
        convert_element_type_296: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_56, torch.bfloat16);  inductor_random_56 = None
        gt_56: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_296, 0.1);  convert_element_type_296 = None
        mul_266: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_56, convert_element_type_295);  convert_element_type_295 = None
        mul_267: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_266, 1.1111111111111112);  mul_266 = None
        convert_element_type_297: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg208_1, torch.bfloat16);  arg208_1 = None
        convert_element_type_298: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.bfloat16);  arg209_1 = None
        view_489: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_267, _shape_param_547);  mul_267 = _shape_param_547 = None
        permute_338: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_298, [1, 0]);  convert_element_type_298 = None
        addmm_27: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_297, view_489, permute_338);  convert_element_type_297 = None
        view_490: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_27, _shape_param_548);  addmm_27 = _shape_param_548 = None
        inductor_lookup_seed_57: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 57)
        inductor_random_57: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_549, inductor_lookup_seed_57, 'rand');  _shape_param_549 = inductor_lookup_seed_57 = None
        convert_element_type_299: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_57, torch.bfloat16);  inductor_random_57 = None
        gt_57: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_299, 0.1);  convert_element_type_299 = None
        mul_268: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_57, view_490);  view_490 = None
        mul_269: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_268, 1.1111111111111112);  mul_268 = None
        add_151: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_269, add_149);  mul_269 = add_149 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
        getitem_54: "f32[512, 16, 1]" = var_mean_27[0]
        getitem_55: "f32[512, 16, 1]" = var_mean_27[1];  var_mean_27 = None
        add_152: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        sub_55: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_151, getitem_55);  add_151 = getitem_55 = None
        mul_270: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_27);  sub_55 = None
        mul_271: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_270, arg210_1);  arg210_1 = None
        add_153: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_271, arg211_1);  mul_271 = arg211_1 = None
        convert_element_type_300: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16)
        convert_element_type_301: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.bfloat16);  arg212_1 = None
        unsqueeze_341: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_300, 3);  convert_element_type_300 = None
        unsqueeze_342: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 4);  unsqueeze_341 = None
        unsqueeze_343: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_301, 3);  convert_element_type_301 = None
        unsqueeze_344: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 4);  unsqueeze_343 = None
        view_491: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_342, _shape_param_550);  unsqueeze_342 = _shape_param_550 = None
        view_492: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_344, _shape_param_551);  unsqueeze_344 = _shape_param_551 = None
        squeeze_99: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_491, 0)
        squeeze_100: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_492, 0)
        mm_70: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_99, squeeze_100);  squeeze_100 = None
        unsqueeze_345: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_70, 0);  mm_70 = None
        view_493: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_345, _shape_param_552);  unsqueeze_345 = _shape_param_552 = None
        permute_339: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_493, [0, 1, 3, 4, 2]);  view_493 = None
        view_494: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_339, _shape_param_553);  permute_339 = _shape_param_553 = None
        convert_element_type_302: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg213_1, torch.bfloat16);  arg213_1 = None
        unsqueeze_346: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_302, 3);  convert_element_type_302 = None
        unsqueeze_347: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 4);  unsqueeze_346 = None
        view_495: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_347, _shape_param_554);  unsqueeze_347 = _shape_param_554 = None
        squeeze_101: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_495, 0)
        mm_71: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_99, squeeze_101);  squeeze_101 = None
        unsqueeze_348: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_71, 0);  mm_71 = None
        view_496: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_348, _shape_param_555);  unsqueeze_348 = _shape_param_555 = None
        permute_340: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_496, [0, 1, 3, 4, 2]);  view_496 = None
        view_497: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_340, _shape_param_556);  permute_340 = _shape_param_556 = None
        convert_element_type_303: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg214_1, torch.bfloat16);  arg214_1 = None
        unsqueeze_349: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_303, 3);  convert_element_type_303 = None
        unsqueeze_350: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 4);  unsqueeze_349 = None
        view_498: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_350, _shape_param_557);  unsqueeze_350 = _shape_param_557 = None
        squeeze_102: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_498, 0)
        mm_72: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_99, squeeze_102);  squeeze_99 = squeeze_102 = None
        unsqueeze_351: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_72, 0);  mm_72 = None
        view_499: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_351, _shape_param_558);  unsqueeze_351 = _shape_param_558 = None
        permute_341: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_499, [0, 1, 3, 4, 2]);  view_499 = None
        view_500: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_341, _shape_param_559);  permute_341 = _shape_param_559 = None
        convert_element_type_304: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg215_1, torch.bfloat16);  arg215_1 = None
        unsqueeze_352: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_304, 3);  convert_element_type_304 = None
        unsqueeze_353: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 4);  unsqueeze_352 = None
        view_501: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_353, _shape_param_560);  unsqueeze_353 = _shape_param_560 = None
        squeeze_103: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_501, 0);  view_501 = None
        mm_73: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_103);  squeeze_103 = None
        unsqueeze_354: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_73, 0);  mm_73 = None
        view_502: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_354, _shape_param_561);  unsqueeze_354 = _shape_param_561 = None
        permute_342: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_502, [0, 1, 3, 4, 2]);  view_502 = None
        view_503: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_342, _shape_param_562);  permute_342 = _shape_param_562 = None
        add_154: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_494, arg216_1);  arg216_1 = None
        convert_element_type_305: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None
        unsqueeze_355: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_305, 4);  convert_element_type_305 = None
        permute_343: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_355, [1, 2, 0, 4, 3]);  unsqueeze_355 = None
        unsqueeze_356: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_497, 4);  view_497 = None
        permute_344: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_356, [1, 2, 4, 0, 3]);  unsqueeze_356 = None
        permute_345: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_343, [0, 1, 2, 4, 3]);  permute_343 = None
        view_504: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_345, _shape_param_563);  permute_345 = _shape_param_563 = None
        permute_346: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_344, [0, 1, 4, 3, 2]);  permute_344 = None
        view_505: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_346, _shape_param_564);  permute_346 = _shape_param_564 = None
        bmm_42: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_504, view_505)
        view_506: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_42, _shape_param_565);  bmm_42 = _shape_param_565 = None
        permute_347: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_506, [0, 1, 2, 4, 3]);  view_506 = None
        view_507: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_347, _shape_param_566);  permute_347 = _shape_param_566 = None
        add_155: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_494, arg217_1);  view_494 = arg217_1 = None
        convert_element_type_306: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16);  add_155 = None
        unsqueeze_357: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_306, 4);  convert_element_type_306 = None
        permute_348: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_357, [1, 2, 0, 4, 3]);  unsqueeze_357 = None
        unsqueeze_358: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_503, 4);  view_503 = None
        permute_349: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_358, [1, 2, 4, 0, 3]);  unsqueeze_358 = None
        permute_350: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_348, [0, 1, 2, 4, 3]);  permute_348 = None
        view_508: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_350, _shape_param_567);  permute_350 = _shape_param_567 = None
        permute_351: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_349, [0, 1, 4, 3, 2]);  permute_349 = None
        view_509: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_351, _shape_param_568);  permute_351 = _shape_param_568 = None
        bmm_43: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_508, view_509)
        view_510: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_43, _shape_param_569);  bmm_43 = _shape_param_569 = None
        permute_352: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_510, [0, 1, 2, 4, 3]);  view_510 = None
        view_511: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_352, _shape_param_570);  permute_352 = _shape_param_570 = None
        view_512: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_511, _shape_param_571);  view_511 = _shape_param_571 = None
        slice_15: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_512, 2, 1, 9223372036854775807);  view_512 = None
        view_513: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_15, _shape_param_572);  slice_15 = _shape_param_572 = None
        index_14: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_513, [None, None, None, iota_2]);  view_513 = None
        add_156: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_507, index_14);  view_507 = index_14 = None
        add_157: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_156, 0);  add_156 = None
        mul_272: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_157, 0.125)
        convert_element_type_307: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_272, torch.float32);  mul_272 = None
        convert_element_type_308: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_157, torch.float32)
        mul_273: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_308, 1);  convert_element_type_308 = None
        amax_28: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_273, [3], True)
        sub_56: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_273, amax_28);  mul_273 = None
        mul_274: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_56, 0.125);  sub_56 = None
        amax_29: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_307, [3], True)
        sub_57: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_307, amax_29)
        abs_15: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_307)
        ne_14: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_15, inf);  abs_15 = None
        eq_14: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_307, convert_element_type_307);  convert_element_type_307 = None
        mul_275: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_14, ne_14);  eq_14 = ne_14 = None
        logical_not_28: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_275);  mul_275 = None
        any_15: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_28, [3], True);  logical_not_28 = None
        logical_not_29: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        where_14: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_29, mul_274, sub_57);  mul_274 = sub_57 = None
        exp_14: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_14);  where_14 = None
        sum_15: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [3], True)
        div_15: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None
        inductor_lookup_seed_58: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 58)
        inductor_random_58: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_573, inductor_lookup_seed_58, 'rand');  _shape_param_573 = inductor_lookup_seed_58 = None
        gt_58: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_58, 0.1);  inductor_random_58 = None
        mul_276: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_58, div_15);  div_15 = None
        mul_277: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_276, 1.1111111111111112);  mul_276 = None
        convert_element_type_309: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_277, torch.bfloat16);  mul_277 = None
        unsqueeze_359: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_309, 4);  convert_element_type_309 = None
        unsqueeze_360: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_500, 4);  view_500 = None
        permute_353: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_360, [4, 1, 2, 3, 0]);  unsqueeze_360 = None
        view_514: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_359, _shape_param_574);  unsqueeze_359 = _shape_param_574 = None
        permute_354: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_353, [1, 2, 4, 3, 0]);  permute_353 = None
        view_515: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_354, _shape_param_575);  permute_354 = _shape_param_575 = None
        bmm_44: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_514, view_515)
        view_516: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_576);  bmm_44 = _shape_param_576 = None
        permute_355: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_516, [2, 0, 1, 4, 3]);  view_516 = None
        view_517: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_355, _shape_param_577);  permute_355 = _shape_param_577 = None
        convert_element_type_310: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg218_1, torch.bfloat16);  arg218_1 = None
        unsqueeze_361: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_517, 4);  view_517 = None
        permute_356: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_361, [0, 1, 4, 3, 2]);  unsqueeze_361 = None
        unsqueeze_362: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_310, 3);  convert_element_type_310 = None
        unsqueeze_363: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 4);  unsqueeze_362 = None
        permute_357: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_363, [3, 4, 0, 2, 1]);  unsqueeze_363 = None
        permute_358: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_356, [0, 1, 3, 4, 2]);  permute_356 = None
        clone_29: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
        view_518: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_29, _shape_param_578);  clone_29 = _shape_param_578 = None
        permute_359: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_357, [3, 4, 2, 0, 1]);  permute_357 = None
        clone_30: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_519: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_30, _shape_param_579);  clone_30 = _shape_param_579 = None
        squeeze_104: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_518, 0)
        squeeze_105: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_519, 0)
        mm_74: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_104, squeeze_105);  squeeze_104 = squeeze_105 = None
        unsqueeze_364: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_74, 0);  mm_74 = None
        view_520: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_364, _shape_param_580);  unsqueeze_364 = _shape_param_580 = None
        permute_360: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_520, [0, 1, 4, 2, 3]);  view_520 = None
        view_521: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_360, _shape_param_581);  permute_360 = _shape_param_581 = None
        inductor_lookup_seed_59: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 59)
        inductor_random_59: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_582, inductor_lookup_seed_59, 'rand');  _shape_param_582 = inductor_lookup_seed_59 = None
        convert_element_type_311: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_59, torch.bfloat16);  inductor_random_59 = None
        gt_59: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_311, 0.1);  convert_element_type_311 = None
        mul_278: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_59, view_521);  view_521 = None
        mul_279: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_278, 1.1111111111111112);  mul_278 = None
        add_158: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_279, add_153);  mul_279 = add_153 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_56: "f32[512, 16, 1]" = var_mean_28[0]
        getitem_57: "f32[512, 16, 1]" = var_mean_28[1];  var_mean_28 = None
        add_159: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_58: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_158, getitem_57);  add_158 = getitem_57 = None
        mul_280: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_28);  sub_58 = None
        mul_281: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_280, arg219_1);  arg219_1 = None
        add_160: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_281, arg220_1);  mul_281 = arg220_1 = None
        convert_element_type_312: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg221_1, torch.bfloat16);  arg221_1 = None
        convert_element_type_313: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.bfloat16);  arg222_1 = None
        convert_element_type_314: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16)
        view_522: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_314, _shape_param_583);  convert_element_type_314 = _shape_param_583 = None
        permute_361: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_313, [1, 0]);  convert_element_type_313 = None
        addmm_28: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_312, view_522, permute_361);  convert_element_type_312 = None
        view_523: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_28, _shape_param_584);  _shape_param_584 = None
        convert_element_type_315: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_523, torch.float32);  view_523 = None
        mul_282: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_315, 0.5)
        mul_283: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_315, 0.7071067811865476);  convert_element_type_315 = None
        erf_14: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_283);  mul_283 = None
        add_161: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_284: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_282, add_161);  mul_282 = add_161 = None
        convert_element_type_316: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_284, torch.bfloat16);  mul_284 = None
        inductor_lookup_seed_60: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 60)
        inductor_random_60: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_585, inductor_lookup_seed_60, 'rand');  _shape_param_585 = inductor_lookup_seed_60 = None
        convert_element_type_317: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_60, torch.bfloat16);  inductor_random_60 = None
        gt_60: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_317, 0.1);  convert_element_type_317 = None
        mul_285: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_60, convert_element_type_316);  convert_element_type_316 = None
        mul_286: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_285, 1.1111111111111112);  mul_285 = None
        convert_element_type_318: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg223_1, torch.bfloat16);  arg223_1 = None
        convert_element_type_319: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg224_1, torch.bfloat16);  arg224_1 = None
        view_524: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_286, _shape_param_586);  mul_286 = _shape_param_586 = None
        permute_362: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_319, [1, 0]);  convert_element_type_319 = None
        addmm_29: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_318, view_524, permute_362);  convert_element_type_318 = None
        view_525: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_29, _shape_param_587);  addmm_29 = _shape_param_587 = None
        inductor_lookup_seed_61: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 61)
        inductor_random_61: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_588, inductor_lookup_seed_61, 'rand');  _shape_param_588 = inductor_lookup_seed_61 = None
        convert_element_type_320: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_61, torch.bfloat16);  inductor_random_61 = None
        gt_61: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_320, 0.1);  convert_element_type_320 = None
        mul_287: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_61, view_525);  view_525 = None
        mul_288: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_287, 1.1111111111111112);  mul_287 = None
        add_162: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_288, add_160);  mul_288 = add_160 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_162, [2], correction = 0, keepdim = True)
        getitem_58: "f32[512, 16, 1]" = var_mean_29[0]
        getitem_59: "f32[512, 16, 1]" = var_mean_29[1];  var_mean_29 = None
        add_163: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        sub_59: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_162, getitem_59);  add_162 = getitem_59 = None
        mul_289: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_29);  sub_59 = None
        mul_290: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_289, arg225_1);  arg225_1 = None
        add_164: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_290, arg226_1);  mul_290 = arg226_1 = None
        convert_element_type_321: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16)
        convert_element_type_322: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.bfloat16);  arg227_1 = None
        unsqueeze_365: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_321, 3);  convert_element_type_321 = None
        unsqueeze_366: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 4);  unsqueeze_365 = None
        unsqueeze_367: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_322, 3);  convert_element_type_322 = None
        unsqueeze_368: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 4);  unsqueeze_367 = None
        view_526: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_366, _shape_param_589);  unsqueeze_366 = _shape_param_589 = None
        view_527: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_368, _shape_param_590);  unsqueeze_368 = _shape_param_590 = None
        squeeze_106: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_526, 0)
        squeeze_107: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_527, 0)
        mm_75: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_106, squeeze_107);  squeeze_107 = None
        unsqueeze_369: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_75, 0);  mm_75 = None
        view_528: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_369, _shape_param_591);  unsqueeze_369 = _shape_param_591 = None
        permute_363: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_528, [0, 1, 3, 4, 2]);  view_528 = None
        view_529: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_363, _shape_param_592);  permute_363 = _shape_param_592 = None
        convert_element_type_323: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.bfloat16);  arg228_1 = None
        unsqueeze_370: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_323, 3);  convert_element_type_323 = None
        unsqueeze_371: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 4);  unsqueeze_370 = None
        view_530: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_371, _shape_param_593);  unsqueeze_371 = _shape_param_593 = None
        squeeze_108: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_530, 0)
        mm_76: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_106, squeeze_108);  squeeze_108 = None
        unsqueeze_372: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_76, 0);  mm_76 = None
        view_531: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_372, _shape_param_594);  unsqueeze_372 = _shape_param_594 = None
        permute_364: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_531, [0, 1, 3, 4, 2]);  view_531 = None
        view_532: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_364, _shape_param_595);  permute_364 = _shape_param_595 = None
        convert_element_type_324: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg229_1, torch.bfloat16);  arg229_1 = None
        unsqueeze_373: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_324, 3);  convert_element_type_324 = None
        unsqueeze_374: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 4);  unsqueeze_373 = None
        view_533: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_374, _shape_param_596);  unsqueeze_374 = _shape_param_596 = None
        squeeze_109: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_533, 0)
        mm_77: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_106, squeeze_109);  squeeze_106 = squeeze_109 = None
        unsqueeze_375: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_77, 0);  mm_77 = None
        view_534: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_375, _shape_param_597);  unsqueeze_375 = _shape_param_597 = None
        permute_365: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_534, [0, 1, 3, 4, 2]);  view_534 = None
        view_535: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_365, _shape_param_598);  permute_365 = _shape_param_598 = None
        convert_element_type_325: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg230_1, torch.bfloat16);  arg230_1 = None
        unsqueeze_376: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_325, 3);  convert_element_type_325 = None
        unsqueeze_377: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 4);  unsqueeze_376 = None
        view_536: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_377, _shape_param_599);  unsqueeze_377 = _shape_param_599 = None
        squeeze_110: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_536, 0);  view_536 = None
        mm_78: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_110);  squeeze_110 = None
        unsqueeze_378: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_78, 0);  mm_78 = None
        view_537: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_378, _shape_param_600);  unsqueeze_378 = _shape_param_600 = None
        permute_366: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_537, [0, 1, 3, 4, 2]);  view_537 = None
        view_538: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_366, _shape_param_601);  permute_366 = _shape_param_601 = None
        add_165: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_529, arg231_1);  arg231_1 = None
        convert_element_type_326: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16);  add_165 = None
        unsqueeze_379: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_326, 4);  convert_element_type_326 = None
        permute_367: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_379, [1, 2, 0, 4, 3]);  unsqueeze_379 = None
        unsqueeze_380: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_532, 4);  view_532 = None
        permute_368: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_380, [1, 2, 4, 0, 3]);  unsqueeze_380 = None
        permute_369: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_367, [0, 1, 2, 4, 3]);  permute_367 = None
        view_539: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_369, _shape_param_602);  permute_369 = _shape_param_602 = None
        permute_370: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_368, [0, 1, 4, 3, 2]);  permute_368 = None
        view_540: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_370, _shape_param_603);  permute_370 = _shape_param_603 = None
        bmm_45: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_539, view_540)
        view_541: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_45, _shape_param_604);  bmm_45 = _shape_param_604 = None
        permute_371: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_541, [0, 1, 2, 4, 3]);  view_541 = None
        view_542: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_371, _shape_param_605);  permute_371 = _shape_param_605 = None
        add_166: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_529, arg232_1);  view_529 = arg232_1 = None
        convert_element_type_327: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_166, torch.bfloat16);  add_166 = None
        unsqueeze_381: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_327, 4);  convert_element_type_327 = None
        permute_372: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_381, [1, 2, 0, 4, 3]);  unsqueeze_381 = None
        unsqueeze_382: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_538, 4);  view_538 = None
        permute_373: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_382, [1, 2, 4, 0, 3]);  unsqueeze_382 = None
        permute_374: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_372, [0, 1, 2, 4, 3]);  permute_372 = None
        view_543: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_374, _shape_param_606);  permute_374 = _shape_param_606 = None
        permute_375: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_373, [0, 1, 4, 3, 2]);  permute_373 = None
        view_544: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_375, _shape_param_607);  permute_375 = _shape_param_607 = None
        bmm_46: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_543, view_544)
        view_545: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_46, _shape_param_608);  bmm_46 = _shape_param_608 = None
        permute_376: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_545, [0, 1, 2, 4, 3]);  view_545 = None
        view_546: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_376, _shape_param_609);  permute_376 = _shape_param_609 = None
        view_547: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_546, _shape_param_610);  view_546 = _shape_param_610 = None
        slice_16: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_547, 2, 1, 9223372036854775807);  view_547 = None
        view_548: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_16, _shape_param_611);  slice_16 = _shape_param_611 = None
        index_15: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_548, [None, None, None, iota_2]);  view_548 = None
        add_167: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_542, index_15);  view_542 = index_15 = None
        add_168: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_167, 0);  add_167 = None
        mul_291: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_168, 0.125)
        convert_element_type_328: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_291, torch.float32);  mul_291 = None
        convert_element_type_329: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_168, torch.float32)
        mul_292: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_329, 1);  convert_element_type_329 = None
        amax_30: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_292, [3], True)
        sub_60: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_292, amax_30);  mul_292 = None
        mul_293: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_60, 0.125);  sub_60 = None
        amax_31: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_328, [3], True)
        sub_61: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_328, amax_31)
        abs_16: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_328)
        ne_15: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_16, inf);  abs_16 = None
        eq_15: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_328, convert_element_type_328);  convert_element_type_328 = None
        mul_294: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_15, ne_15);  eq_15 = ne_15 = None
        logical_not_30: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_294);  mul_294 = None
        any_16: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_30, [3], True);  logical_not_30 = None
        logical_not_31: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        where_15: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_31, mul_293, sub_61);  mul_293 = sub_61 = None
        exp_15: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_15);  where_15 = None
        sum_16: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [3], True)
        div_16: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None
        inductor_lookup_seed_62: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 62)
        inductor_random_62: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_612, inductor_lookup_seed_62, 'rand');  _shape_param_612 = inductor_lookup_seed_62 = None
        gt_62: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_62, 0.1);  inductor_random_62 = None
        mul_295: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_62, div_16);  div_16 = None
        mul_296: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_295, 1.1111111111111112);  mul_295 = None
        convert_element_type_330: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None
        unsqueeze_383: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_330, 4);  convert_element_type_330 = None
        unsqueeze_384: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_535, 4);  view_535 = None
        permute_377: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_384, [4, 1, 2, 3, 0]);  unsqueeze_384 = None
        view_549: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_383, _shape_param_613);  unsqueeze_383 = _shape_param_613 = None
        permute_378: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_377, [1, 2, 4, 3, 0]);  permute_377 = None
        view_550: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_378, _shape_param_614);  permute_378 = _shape_param_614 = None
        bmm_47: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_549, view_550)
        view_551: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_615);  bmm_47 = _shape_param_615 = None
        permute_379: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_551, [2, 0, 1, 4, 3]);  view_551 = None
        view_552: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_379, _shape_param_616);  permute_379 = _shape_param_616 = None
        convert_element_type_331: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.bfloat16);  arg233_1 = None
        unsqueeze_385: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_552, 4);  view_552 = None
        permute_380: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_385, [0, 1, 4, 3, 2]);  unsqueeze_385 = None
        unsqueeze_386: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_331, 3);  convert_element_type_331 = None
        unsqueeze_387: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 4);  unsqueeze_386 = None
        permute_381: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_387, [3, 4, 0, 2, 1]);  unsqueeze_387 = None
        permute_382: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_380, [0, 1, 3, 4, 2]);  permute_380 = None
        clone_31: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_382, memory_format = torch.contiguous_format);  permute_382 = None
        view_553: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_31, _shape_param_617);  clone_31 = _shape_param_617 = None
        permute_383: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_381, [3, 4, 2, 0, 1]);  permute_381 = None
        clone_32: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
        view_554: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_32, _shape_param_618);  clone_32 = _shape_param_618 = None
        squeeze_111: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_553, 0)
        squeeze_112: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_554, 0)
        mm_79: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_111, squeeze_112);  squeeze_111 = squeeze_112 = None
        unsqueeze_388: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_79, 0);  mm_79 = None
        view_555: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_388, _shape_param_619);  unsqueeze_388 = _shape_param_619 = None
        permute_384: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_555, [0, 1, 4, 2, 3]);  view_555 = None
        view_556: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_384, _shape_param_620);  permute_384 = _shape_param_620 = None
        inductor_lookup_seed_63: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 63)
        inductor_random_63: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_621, inductor_lookup_seed_63, 'rand');  _shape_param_621 = inductor_lookup_seed_63 = None
        convert_element_type_332: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_63, torch.bfloat16);  inductor_random_63 = None
        gt_63: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_332, 0.1);  convert_element_type_332 = None
        mul_297: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_63, view_556);  view_556 = None
        mul_298: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_297, 1.1111111111111112);  mul_297 = None
        add_169: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_298, add_164);  mul_298 = add_164 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_60: "f32[512, 16, 1]" = var_mean_30[0]
        getitem_61: "f32[512, 16, 1]" = var_mean_30[1];  var_mean_30 = None
        add_170: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_62: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_169, getitem_61);  add_169 = getitem_61 = None
        mul_299: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_30);  sub_62 = None
        mul_300: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_299, arg234_1);  arg234_1 = None
        add_171: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_300, arg235_1);  mul_300 = arg235_1 = None
        convert_element_type_333: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg236_1, torch.bfloat16);  arg236_1 = None
        convert_element_type_334: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.bfloat16);  arg237_1 = None
        convert_element_type_335: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16)
        view_557: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_335, _shape_param_622);  convert_element_type_335 = _shape_param_622 = None
        permute_385: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_334, [1, 0]);  convert_element_type_334 = None
        addmm_30: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_333, view_557, permute_385);  convert_element_type_333 = None
        view_558: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_30, _shape_param_623);  _shape_param_623 = None
        convert_element_type_336: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_558, torch.float32);  view_558 = None
        mul_301: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_336, 0.5)
        mul_302: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_336, 0.7071067811865476);  convert_element_type_336 = None
        erf_15: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_302);  mul_302 = None
        add_172: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_303: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_301, add_172);  mul_301 = add_172 = None
        convert_element_type_337: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_303, torch.bfloat16);  mul_303 = None
        inductor_lookup_seed_64: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 64)
        inductor_random_64: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_624, inductor_lookup_seed_64, 'rand');  _shape_param_624 = inductor_lookup_seed_64 = None
        convert_element_type_338: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_64, torch.bfloat16);  inductor_random_64 = None
        gt_64: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_338, 0.1);  convert_element_type_338 = None
        mul_304: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_64, convert_element_type_337);  convert_element_type_337 = None
        mul_305: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_304, 1.1111111111111112);  mul_304 = None
        convert_element_type_339: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg238_1, torch.bfloat16);  arg238_1 = None
        convert_element_type_340: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg239_1, torch.bfloat16);  arg239_1 = None
        view_559: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_305, _shape_param_625);  mul_305 = _shape_param_625 = None
        permute_386: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_340, [1, 0]);  convert_element_type_340 = None
        addmm_31: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_339, view_559, permute_386);  convert_element_type_339 = None
        view_560: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_31, _shape_param_626);  addmm_31 = _shape_param_626 = None
        inductor_lookup_seed_65: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 65)
        inductor_random_65: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_627, inductor_lookup_seed_65, 'rand');  _shape_param_627 = inductor_lookup_seed_65 = None
        convert_element_type_341: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_65, torch.bfloat16);  inductor_random_65 = None
        gt_65: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_341, 0.1);  convert_element_type_341 = None
        mul_306: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_65, view_560);  view_560 = None
        mul_307: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_306, 1.1111111111111112);  mul_306 = None
        add_173: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_307, add_171);  mul_307 = add_171 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_62: "f32[512, 16, 1]" = var_mean_31[0]
        getitem_63: "f32[512, 16, 1]" = var_mean_31[1];  var_mean_31 = None
        add_174: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_63: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_173, getitem_63);  add_173 = getitem_63 = None
        mul_308: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_31);  sub_63 = None
        mul_309: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_308, arg240_1);  arg240_1 = None
        add_175: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_309, arg241_1);  mul_309 = arg241_1 = None
        convert_element_type_342: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16)
        convert_element_type_343: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.bfloat16);  arg242_1 = None
        unsqueeze_389: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_342, 3);  convert_element_type_342 = None
        unsqueeze_390: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 4);  unsqueeze_389 = None
        unsqueeze_391: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_343, 3);  convert_element_type_343 = None
        unsqueeze_392: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 4);  unsqueeze_391 = None
        view_561: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_390, _shape_param_628);  unsqueeze_390 = _shape_param_628 = None
        view_562: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_392, _shape_param_629);  unsqueeze_392 = _shape_param_629 = None
        squeeze_113: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_561, 0)
        squeeze_114: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_562, 0)
        mm_80: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_113, squeeze_114);  squeeze_114 = None
        unsqueeze_393: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_80, 0);  mm_80 = None
        view_563: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_393, _shape_param_630);  unsqueeze_393 = _shape_param_630 = None
        permute_387: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_563, [0, 1, 3, 4, 2]);  view_563 = None
        view_564: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_387, _shape_param_631);  permute_387 = _shape_param_631 = None
        convert_element_type_344: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.bfloat16);  arg243_1 = None
        unsqueeze_394: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_344, 3);  convert_element_type_344 = None
        unsqueeze_395: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 4);  unsqueeze_394 = None
        view_565: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_395, _shape_param_632);  unsqueeze_395 = _shape_param_632 = None
        squeeze_115: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_565, 0)
        mm_81: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_113, squeeze_115);  squeeze_115 = None
        unsqueeze_396: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_81, 0);  mm_81 = None
        view_566: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_396, _shape_param_633);  unsqueeze_396 = _shape_param_633 = None
        permute_388: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_566, [0, 1, 3, 4, 2]);  view_566 = None
        view_567: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_388, _shape_param_634);  permute_388 = _shape_param_634 = None
        convert_element_type_345: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg244_1, torch.bfloat16);  arg244_1 = None
        unsqueeze_397: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_345, 3);  convert_element_type_345 = None
        unsqueeze_398: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 4);  unsqueeze_397 = None
        view_568: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_398, _shape_param_635);  unsqueeze_398 = _shape_param_635 = None
        squeeze_116: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_568, 0)
        mm_82: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_113, squeeze_116);  squeeze_113 = squeeze_116 = None
        unsqueeze_399: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_82, 0);  mm_82 = None
        view_569: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_399, _shape_param_636);  unsqueeze_399 = _shape_param_636 = None
        permute_389: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_569, [0, 1, 3, 4, 2]);  view_569 = None
        view_570: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_389, _shape_param_637);  permute_389 = _shape_param_637 = None
        convert_element_type_346: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg245_1, torch.bfloat16);  arg245_1 = None
        unsqueeze_400: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_346, 3);  convert_element_type_346 = None
        unsqueeze_401: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 4);  unsqueeze_400 = None
        view_571: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_401, _shape_param_638);  unsqueeze_401 = _shape_param_638 = None
        squeeze_117: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_571, 0);  view_571 = None
        mm_83: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_117);  squeeze_117 = None
        unsqueeze_402: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_83, 0);  mm_83 = None
        view_572: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_402, _shape_param_639);  unsqueeze_402 = _shape_param_639 = None
        permute_390: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_572, [0, 1, 3, 4, 2]);  view_572 = None
        view_573: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_390, _shape_param_640);  permute_390 = _shape_param_640 = None
        add_176: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_564, arg246_1);  arg246_1 = None
        convert_element_type_347: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16);  add_176 = None
        unsqueeze_403: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_347, 4);  convert_element_type_347 = None
        permute_391: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_403, [1, 2, 0, 4, 3]);  unsqueeze_403 = None
        unsqueeze_404: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_567, 4);  view_567 = None
        permute_392: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_404, [1, 2, 4, 0, 3]);  unsqueeze_404 = None
        permute_393: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_391, [0, 1, 2, 4, 3]);  permute_391 = None
        view_574: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_393, _shape_param_641);  permute_393 = _shape_param_641 = None
        permute_394: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_392, [0, 1, 4, 3, 2]);  permute_392 = None
        view_575: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_394, _shape_param_642);  permute_394 = _shape_param_642 = None
        bmm_48: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_574, view_575)
        view_576: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_48, _shape_param_643);  bmm_48 = _shape_param_643 = None
        permute_395: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_576, [0, 1, 2, 4, 3]);  view_576 = None
        view_577: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_395, _shape_param_644);  permute_395 = _shape_param_644 = None
        add_177: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_564, arg247_1);  view_564 = arg247_1 = None
        convert_element_type_348: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_177, torch.bfloat16);  add_177 = None
        unsqueeze_405: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_348, 4);  convert_element_type_348 = None
        permute_396: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_405, [1, 2, 0, 4, 3]);  unsqueeze_405 = None
        unsqueeze_406: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_573, 4);  view_573 = None
        permute_397: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_406, [1, 2, 4, 0, 3]);  unsqueeze_406 = None
        permute_398: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_396, [0, 1, 2, 4, 3]);  permute_396 = None
        view_578: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_398, _shape_param_645);  permute_398 = _shape_param_645 = None
        permute_399: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_397, [0, 1, 4, 3, 2]);  permute_397 = None
        view_579: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_399, _shape_param_646);  permute_399 = _shape_param_646 = None
        bmm_49: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_578, view_579)
        view_580: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_49, _shape_param_647);  bmm_49 = _shape_param_647 = None
        permute_400: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_580, [0, 1, 2, 4, 3]);  view_580 = None
        view_581: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_400, _shape_param_648);  permute_400 = _shape_param_648 = None
        view_582: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_581, _shape_param_649);  view_581 = _shape_param_649 = None
        slice_17: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_582, 2, 1, 9223372036854775807);  view_582 = None
        view_583: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_17, _shape_param_650);  slice_17 = _shape_param_650 = None
        index_16: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_583, [None, None, None, iota_2]);  view_583 = None
        add_178: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_577, index_16);  view_577 = index_16 = None
        add_179: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_178, 0);  add_178 = None
        mul_310: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_179, 0.125)
        convert_element_type_349: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_310, torch.float32);  mul_310 = None
        convert_element_type_350: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_179, torch.float32)
        mul_311: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_350, 1);  convert_element_type_350 = None
        amax_32: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_311, [3], True)
        sub_64: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_311, amax_32);  mul_311 = None
        mul_312: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_64, 0.125);  sub_64 = None
        amax_33: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_349, [3], True)
        sub_65: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_349, amax_33)
        abs_17: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_349)
        ne_16: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_17, inf);  abs_17 = None
        eq_16: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_349, convert_element_type_349);  convert_element_type_349 = None
        mul_313: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_16, ne_16);  eq_16 = ne_16 = None
        logical_not_32: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_313);  mul_313 = None
        any_17: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_32, [3], True);  logical_not_32 = None
        logical_not_33: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        where_16: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_33, mul_312, sub_65);  mul_312 = sub_65 = None
        exp_16: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_16);  where_16 = None
        sum_17: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [3], True)
        div_17: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None
        inductor_lookup_seed_66: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 66)
        inductor_random_66: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_651, inductor_lookup_seed_66, 'rand');  _shape_param_651 = inductor_lookup_seed_66 = None
        gt_66: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_66, 0.1);  inductor_random_66 = None
        mul_314: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_66, div_17);  div_17 = None
        mul_315: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_314, 1.1111111111111112);  mul_314 = None
        convert_element_type_351: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_315, torch.bfloat16);  mul_315 = None
        unsqueeze_407: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_351, 4);  convert_element_type_351 = None
        unsqueeze_408: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_570, 4);  view_570 = None
        permute_401: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_408, [4, 1, 2, 3, 0]);  unsqueeze_408 = None
        view_584: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_407, _shape_param_652);  unsqueeze_407 = _shape_param_652 = None
        permute_402: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_401, [1, 2, 4, 3, 0]);  permute_401 = None
        view_585: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_402, _shape_param_653);  permute_402 = _shape_param_653 = None
        bmm_50: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_584, view_585)
        view_586: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_50, _shape_param_654);  bmm_50 = _shape_param_654 = None
        permute_403: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_586, [2, 0, 1, 4, 3]);  view_586 = None
        view_587: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_403, _shape_param_655);  permute_403 = _shape_param_655 = None
        convert_element_type_352: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.bfloat16);  arg248_1 = None
        unsqueeze_409: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_587, 4);  view_587 = None
        permute_404: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_409, [0, 1, 4, 3, 2]);  unsqueeze_409 = None
        unsqueeze_410: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_352, 3);  convert_element_type_352 = None
        unsqueeze_411: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 4);  unsqueeze_410 = None
        permute_405: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_411, [3, 4, 0, 2, 1]);  unsqueeze_411 = None
        permute_406: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_404, [0, 1, 3, 4, 2]);  permute_404 = None
        clone_33: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_406, memory_format = torch.contiguous_format);  permute_406 = None
        view_588: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_33, _shape_param_656);  clone_33 = _shape_param_656 = None
        permute_407: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_405, [3, 4, 2, 0, 1]);  permute_405 = None
        clone_34: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
        view_589: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_34, _shape_param_657);  clone_34 = _shape_param_657 = None
        squeeze_118: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_588, 0)
        squeeze_119: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_589, 0)
        mm_84: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_118, squeeze_119);  squeeze_118 = squeeze_119 = None
        unsqueeze_412: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_84, 0);  mm_84 = None
        view_590: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_412, _shape_param_658);  unsqueeze_412 = _shape_param_658 = None
        permute_408: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_590, [0, 1, 4, 2, 3]);  view_590 = None
        view_591: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_408, _shape_param_659);  permute_408 = _shape_param_659 = None
        inductor_lookup_seed_67: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 67)
        inductor_random_67: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_660, inductor_lookup_seed_67, 'rand');  _shape_param_660 = inductor_lookup_seed_67 = None
        convert_element_type_353: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_67, torch.bfloat16);  inductor_random_67 = None
        gt_67: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_353, 0.1);  convert_element_type_353 = None
        mul_316: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_67, view_591);  view_591 = None
        mul_317: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_316, 1.1111111111111112);  mul_316 = None
        add_180: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_317, add_175);  mul_317 = add_175 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
        getitem_64: "f32[512, 16, 1]" = var_mean_32[0]
        getitem_65: "f32[512, 16, 1]" = var_mean_32[1];  var_mean_32 = None
        add_181: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_66: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_180, getitem_65);  add_180 = getitem_65 = None
        mul_318: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_32);  sub_66 = None
        mul_319: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_318, arg249_1);  arg249_1 = None
        add_182: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_319, arg250_1);  mul_319 = arg250_1 = None
        convert_element_type_354: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg251_1, torch.bfloat16);  arg251_1 = None
        convert_element_type_355: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.bfloat16);  arg252_1 = None
        convert_element_type_356: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_182, torch.bfloat16)
        view_592: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_356, _shape_param_661);  convert_element_type_356 = _shape_param_661 = None
        permute_409: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_355, [1, 0]);  convert_element_type_355 = None
        addmm_32: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_354, view_592, permute_409);  convert_element_type_354 = None
        view_593: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_32, _shape_param_662);  _shape_param_662 = None
        convert_element_type_357: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_593, torch.float32);  view_593 = None
        mul_320: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_357, 0.5)
        mul_321: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_357, 0.7071067811865476);  convert_element_type_357 = None
        erf_16: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_321);  mul_321 = None
        add_183: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_322: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_320, add_183);  mul_320 = add_183 = None
        convert_element_type_358: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_322, torch.bfloat16);  mul_322 = None
        inductor_lookup_seed_68: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 68)
        inductor_random_68: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_663, inductor_lookup_seed_68, 'rand');  _shape_param_663 = inductor_lookup_seed_68 = None
        convert_element_type_359: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_68, torch.bfloat16);  inductor_random_68 = None
        gt_68: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_359, 0.1);  convert_element_type_359 = None
        mul_323: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_68, convert_element_type_358);  convert_element_type_358 = None
        mul_324: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_323, 1.1111111111111112);  mul_323 = None
        convert_element_type_360: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.bfloat16);  arg253_1 = None
        convert_element_type_361: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg254_1, torch.bfloat16);  arg254_1 = None
        view_594: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_324, _shape_param_664);  mul_324 = _shape_param_664 = None
        permute_410: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        addmm_33: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_360, view_594, permute_410);  convert_element_type_360 = None
        view_595: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_33, _shape_param_665);  addmm_33 = _shape_param_665 = None
        inductor_lookup_seed_69: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 69)
        inductor_random_69: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_666, inductor_lookup_seed_69, 'rand');  _shape_param_666 = inductor_lookup_seed_69 = None
        convert_element_type_362: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_69, torch.bfloat16);  inductor_random_69 = None
        gt_69: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_362, 0.1);  convert_element_type_362 = None
        mul_325: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_69, view_595);  view_595 = None
        mul_326: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_325, 1.1111111111111112);  mul_325 = None
        add_184: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_326, add_182);  mul_326 = add_182 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
        getitem_66: "f32[512, 16, 1]" = var_mean_33[0]
        getitem_67: "f32[512, 16, 1]" = var_mean_33[1];  var_mean_33 = None
        add_185: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        sub_67: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_184, getitem_67);  add_184 = getitem_67 = None
        mul_327: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_33);  sub_67 = None
        mul_328: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_327, arg255_1);  arg255_1 = None
        add_186: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_328, arg256_1);  mul_328 = arg256_1 = None
        convert_element_type_363: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16)
        convert_element_type_364: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.bfloat16);  arg257_1 = None
        unsqueeze_413: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_363, 3);  convert_element_type_363 = None
        unsqueeze_414: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 4);  unsqueeze_413 = None
        unsqueeze_415: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_364, 3);  convert_element_type_364 = None
        unsqueeze_416: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 4);  unsqueeze_415 = None
        view_596: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_414, _shape_param_667);  unsqueeze_414 = _shape_param_667 = None
        view_597: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_416, _shape_param_668);  unsqueeze_416 = _shape_param_668 = None
        squeeze_120: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_596, 0)
        squeeze_121: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_597, 0)
        mm_85: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_120, squeeze_121);  squeeze_121 = None
        unsqueeze_417: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_85, 0);  mm_85 = None
        view_598: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_417, _shape_param_669);  unsqueeze_417 = _shape_param_669 = None
        permute_411: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_598, [0, 1, 3, 4, 2]);  view_598 = None
        view_599: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_411, _shape_param_670);  permute_411 = _shape_param_670 = None
        convert_element_type_365: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.bfloat16);  arg258_1 = None
        unsqueeze_418: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_365, 3);  convert_element_type_365 = None
        unsqueeze_419: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 4);  unsqueeze_418 = None
        view_600: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_419, _shape_param_671);  unsqueeze_419 = _shape_param_671 = None
        squeeze_122: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_600, 0)
        mm_86: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_120, squeeze_122);  squeeze_122 = None
        unsqueeze_420: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_86, 0);  mm_86 = None
        view_601: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_420, _shape_param_672);  unsqueeze_420 = _shape_param_672 = None
        permute_412: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_601, [0, 1, 3, 4, 2]);  view_601 = None
        view_602: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_412, _shape_param_673);  permute_412 = _shape_param_673 = None
        convert_element_type_366: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.bfloat16);  arg259_1 = None
        unsqueeze_421: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_366, 3);  convert_element_type_366 = None
        unsqueeze_422: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 4);  unsqueeze_421 = None
        view_603: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_422, _shape_param_674);  unsqueeze_422 = _shape_param_674 = None
        squeeze_123: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_603, 0)
        mm_87: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_120, squeeze_123);  squeeze_120 = squeeze_123 = None
        unsqueeze_423: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_87, 0);  mm_87 = None
        view_604: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_423, _shape_param_675);  unsqueeze_423 = _shape_param_675 = None
        permute_413: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_604, [0, 1, 3, 4, 2]);  view_604 = None
        view_605: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_413, _shape_param_676);  permute_413 = _shape_param_676 = None
        convert_element_type_367: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg260_1, torch.bfloat16);  arg260_1 = None
        unsqueeze_424: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_367, 3);  convert_element_type_367 = None
        unsqueeze_425: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 4);  unsqueeze_424 = None
        view_606: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_425, _shape_param_677);  unsqueeze_425 = _shape_param_677 = None
        squeeze_124: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_606, 0);  view_606 = None
        mm_88: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_124);  squeeze_124 = None
        unsqueeze_426: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_88, 0);  mm_88 = None
        view_607: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_426, _shape_param_678);  unsqueeze_426 = _shape_param_678 = None
        permute_414: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_607, [0, 1, 3, 4, 2]);  view_607 = None
        view_608: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_414, _shape_param_679);  permute_414 = _shape_param_679 = None
        add_187: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_599, arg261_1);  arg261_1 = None
        convert_element_type_368: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_187, torch.bfloat16);  add_187 = None
        unsqueeze_427: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_368, 4);  convert_element_type_368 = None
        permute_415: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_427, [1, 2, 0, 4, 3]);  unsqueeze_427 = None
        unsqueeze_428: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_602, 4);  view_602 = None
        permute_416: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_428, [1, 2, 4, 0, 3]);  unsqueeze_428 = None
        permute_417: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_415, [0, 1, 2, 4, 3]);  permute_415 = None
        view_609: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_417, _shape_param_680);  permute_417 = _shape_param_680 = None
        permute_418: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_416, [0, 1, 4, 3, 2]);  permute_416 = None
        view_610: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_418, _shape_param_681);  permute_418 = _shape_param_681 = None
        bmm_51: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_609, view_610)
        view_611: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_51, _shape_param_682);  bmm_51 = _shape_param_682 = None
        permute_419: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_611, [0, 1, 2, 4, 3]);  view_611 = None
        view_612: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_419, _shape_param_683);  permute_419 = _shape_param_683 = None
        add_188: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_599, arg262_1);  view_599 = arg262_1 = None
        convert_element_type_369: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_188, torch.bfloat16);  add_188 = None
        unsqueeze_429: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_369, 4);  convert_element_type_369 = None
        permute_420: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_429, [1, 2, 0, 4, 3]);  unsqueeze_429 = None
        unsqueeze_430: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_608, 4);  view_608 = None
        permute_421: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_430, [1, 2, 4, 0, 3]);  unsqueeze_430 = None
        permute_422: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_420, [0, 1, 2, 4, 3]);  permute_420 = None
        view_613: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_422, _shape_param_684);  permute_422 = _shape_param_684 = None
        permute_423: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_421, [0, 1, 4, 3, 2]);  permute_421 = None
        view_614: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_423, _shape_param_685);  permute_423 = _shape_param_685 = None
        bmm_52: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_613, view_614)
        view_615: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_52, _shape_param_686);  bmm_52 = _shape_param_686 = None
        permute_424: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_615, [0, 1, 2, 4, 3]);  view_615 = None
        view_616: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_424, _shape_param_687);  permute_424 = _shape_param_687 = None
        view_617: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_616, _shape_param_688);  view_616 = _shape_param_688 = None
        slice_18: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_617, 2, 1, 9223372036854775807);  view_617 = None
        view_618: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_18, _shape_param_689);  slice_18 = _shape_param_689 = None
        index_17: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_618, [None, None, None, iota_2]);  view_618 = None
        add_189: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_612, index_17);  view_612 = index_17 = None
        add_190: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_189, 0);  add_189 = None
        mul_329: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_190, 0.125)
        convert_element_type_370: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_329, torch.float32);  mul_329 = None
        convert_element_type_371: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_190, torch.float32)
        mul_330: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_371, 1);  convert_element_type_371 = None
        amax_34: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_330, [3], True)
        sub_68: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_330, amax_34);  mul_330 = None
        mul_331: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_68, 0.125);  sub_68 = None
        amax_35: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_370, [3], True)
        sub_69: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_370, amax_35)
        abs_18: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_370)
        ne_17: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_18, inf);  abs_18 = None
        eq_17: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_370, convert_element_type_370);  convert_element_type_370 = None
        mul_332: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_17, ne_17);  eq_17 = ne_17 = None
        logical_not_34: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_332);  mul_332 = None
        any_18: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_34, [3], True);  logical_not_34 = None
        logical_not_35: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        where_17: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_35, mul_331, sub_69);  mul_331 = sub_69 = None
        exp_17: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_17);  where_17 = None
        sum_18: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [3], True)
        div_18: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None
        inductor_lookup_seed_70: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 70)
        inductor_random_70: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_690, inductor_lookup_seed_70, 'rand');  _shape_param_690 = inductor_lookup_seed_70 = None
        gt_70: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_70, 0.1);  inductor_random_70 = None
        mul_333: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_70, div_18);  div_18 = None
        mul_334: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_333, 1.1111111111111112);  mul_333 = None
        convert_element_type_372: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        unsqueeze_431: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_372, 4);  convert_element_type_372 = None
        unsqueeze_432: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_605, 4);  view_605 = None
        permute_425: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_432, [4, 1, 2, 3, 0]);  unsqueeze_432 = None
        view_619: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_431, _shape_param_691);  unsqueeze_431 = _shape_param_691 = None
        permute_426: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_425, [1, 2, 4, 3, 0]);  permute_425 = None
        view_620: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_426, _shape_param_692);  permute_426 = _shape_param_692 = None
        bmm_53: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_619, view_620)
        view_621: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_53, _shape_param_693);  bmm_53 = _shape_param_693 = None
        permute_427: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_621, [2, 0, 1, 4, 3]);  view_621 = None
        view_622: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_427, _shape_param_694);  permute_427 = _shape_param_694 = None
        convert_element_type_373: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.bfloat16);  arg263_1 = None
        unsqueeze_433: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_622, 4);  view_622 = None
        permute_428: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_433, [0, 1, 4, 3, 2]);  unsqueeze_433 = None
        unsqueeze_434: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_373, 3);  convert_element_type_373 = None
        unsqueeze_435: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 4);  unsqueeze_434 = None
        permute_429: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_435, [3, 4, 0, 2, 1]);  unsqueeze_435 = None
        permute_430: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_428, [0, 1, 3, 4, 2]);  permute_428 = None
        clone_35: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_430, memory_format = torch.contiguous_format);  permute_430 = None
        view_623: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_35, _shape_param_695);  clone_35 = _shape_param_695 = None
        permute_431: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_429, [3, 4, 2, 0, 1]);  permute_429 = None
        clone_36: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_431, memory_format = torch.contiguous_format);  permute_431 = None
        view_624: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_36, _shape_param_696);  clone_36 = _shape_param_696 = None
        squeeze_125: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_623, 0)
        squeeze_126: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_624, 0)
        mm_89: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_125, squeeze_126);  squeeze_125 = squeeze_126 = None
        unsqueeze_436: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_89, 0);  mm_89 = None
        view_625: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_436, _shape_param_697);  unsqueeze_436 = _shape_param_697 = None
        permute_432: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_625, [0, 1, 4, 2, 3]);  view_625 = None
        view_626: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_432, _shape_param_698);  permute_432 = _shape_param_698 = None
        inductor_lookup_seed_71: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 71)
        inductor_random_71: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_699, inductor_lookup_seed_71, 'rand');  _shape_param_699 = inductor_lookup_seed_71 = None
        convert_element_type_374: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_71, torch.bfloat16);  inductor_random_71 = None
        gt_71: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_374, 0.1);  convert_element_type_374 = None
        mul_335: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_71, view_626);  view_626 = None
        mul_336: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_335, 1.1111111111111112);  mul_335 = None
        add_191: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_336, add_186);  mul_336 = add_186 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
        getitem_68: "f32[512, 16, 1]" = var_mean_34[0]
        getitem_69: "f32[512, 16, 1]" = var_mean_34[1];  var_mean_34 = None
        add_192: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_70: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_191, getitem_69);  add_191 = getitem_69 = None
        mul_337: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_34);  sub_70 = None
        mul_338: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_337, arg264_1);  arg264_1 = None
        add_193: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_338, arg265_1);  mul_338 = arg265_1 = None
        convert_element_type_375: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg266_1, torch.bfloat16);  arg266_1 = None
        convert_element_type_376: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.bfloat16);  arg267_1 = None
        convert_element_type_377: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16)
        view_627: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_377, _shape_param_700);  convert_element_type_377 = _shape_param_700 = None
        permute_433: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_376, [1, 0]);  convert_element_type_376 = None
        addmm_34: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_375, view_627, permute_433);  convert_element_type_375 = None
        view_628: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_34, _shape_param_701);  _shape_param_701 = None
        convert_element_type_378: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_628, torch.float32);  view_628 = None
        mul_339: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_378, 0.5)
        mul_340: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_378, 0.7071067811865476);  convert_element_type_378 = None
        erf_17: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_340);  mul_340 = None
        add_194: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_341: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_339, add_194);  mul_339 = add_194 = None
        convert_element_type_379: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_341, torch.bfloat16);  mul_341 = None
        inductor_lookup_seed_72: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 72)
        inductor_random_72: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_702, inductor_lookup_seed_72, 'rand');  _shape_param_702 = inductor_lookup_seed_72 = None
        convert_element_type_380: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_72, torch.bfloat16);  inductor_random_72 = None
        gt_72: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_380, 0.1);  convert_element_type_380 = None
        mul_342: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_72, convert_element_type_379);  convert_element_type_379 = None
        mul_343: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_342, 1.1111111111111112);  mul_342 = None
        convert_element_type_381: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.bfloat16);  arg268_1 = None
        convert_element_type_382: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg269_1, torch.bfloat16);  arg269_1 = None
        view_629: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_343, _shape_param_703);  mul_343 = _shape_param_703 = None
        permute_434: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_382, [1, 0]);  convert_element_type_382 = None
        addmm_35: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_381, view_629, permute_434);  convert_element_type_381 = None
        view_630: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_35, _shape_param_704);  addmm_35 = _shape_param_704 = None
        inductor_lookup_seed_73: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 73)
        inductor_random_73: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_705, inductor_lookup_seed_73, 'rand');  _shape_param_705 = inductor_lookup_seed_73 = None
        convert_element_type_383: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_73, torch.bfloat16);  inductor_random_73 = None
        gt_73: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_383, 0.1);  convert_element_type_383 = None
        mul_344: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_73, view_630);  view_630 = None
        mul_345: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_344, 1.1111111111111112);  mul_344 = None
        add_195: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_345, add_193);  mul_345 = add_193 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_195, [2], correction = 0, keepdim = True)
        getitem_70: "f32[512, 16, 1]" = var_mean_35[0]
        getitem_71: "f32[512, 16, 1]" = var_mean_35[1];  var_mean_35 = None
        add_196: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_71: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_195, getitem_71);  add_195 = getitem_71 = None
        mul_346: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_35);  sub_71 = None
        mul_347: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_346, arg270_1);  arg270_1 = None
        add_197: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_347, arg271_1);  mul_347 = arg271_1 = None
        convert_element_type_384: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16)
        convert_element_type_385: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.bfloat16);  arg272_1 = None
        unsqueeze_437: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_384, 3);  convert_element_type_384 = None
        unsqueeze_438: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 4);  unsqueeze_437 = None
        unsqueeze_439: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_385, 3);  convert_element_type_385 = None
        unsqueeze_440: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 4);  unsqueeze_439 = None
        view_631: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_438, _shape_param_706);  unsqueeze_438 = _shape_param_706 = None
        view_632: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_440, _shape_param_707);  unsqueeze_440 = _shape_param_707 = None
        squeeze_127: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_631, 0)
        squeeze_128: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_632, 0)
        mm_90: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_127, squeeze_128);  squeeze_128 = None
        unsqueeze_441: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_90, 0);  mm_90 = None
        view_633: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_441, _shape_param_708);  unsqueeze_441 = _shape_param_708 = None
        permute_435: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_633, [0, 1, 3, 4, 2]);  view_633 = None
        view_634: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_435, _shape_param_709);  permute_435 = _shape_param_709 = None
        convert_element_type_386: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg273_1, torch.bfloat16);  arg273_1 = None
        unsqueeze_442: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_386, 3);  convert_element_type_386 = None
        unsqueeze_443: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 4);  unsqueeze_442 = None
        view_635: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_443, _shape_param_710);  unsqueeze_443 = _shape_param_710 = None
        squeeze_129: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_635, 0)
        mm_91: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_127, squeeze_129);  squeeze_129 = None
        unsqueeze_444: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_91, 0);  mm_91 = None
        view_636: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_444, _shape_param_711);  unsqueeze_444 = _shape_param_711 = None
        permute_436: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_636, [0, 1, 3, 4, 2]);  view_636 = None
        view_637: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_436, _shape_param_712);  permute_436 = _shape_param_712 = None
        convert_element_type_387: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg274_1, torch.bfloat16);  arg274_1 = None
        unsqueeze_445: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_387, 3);  convert_element_type_387 = None
        unsqueeze_446: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 4);  unsqueeze_445 = None
        view_638: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_446, _shape_param_713);  unsqueeze_446 = _shape_param_713 = None
        squeeze_130: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_638, 0)
        mm_92: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_127, squeeze_130);  squeeze_127 = squeeze_130 = None
        unsqueeze_447: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_92, 0);  mm_92 = None
        view_639: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_447, _shape_param_714);  unsqueeze_447 = _shape_param_714 = None
        permute_437: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_639, [0, 1, 3, 4, 2]);  view_639 = None
        view_640: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_437, _shape_param_715);  permute_437 = _shape_param_715 = None
        convert_element_type_388: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.bfloat16);  arg275_1 = None
        unsqueeze_448: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_388, 3);  convert_element_type_388 = None
        unsqueeze_449: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 4);  unsqueeze_448 = None
        view_641: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_449, _shape_param_716);  unsqueeze_449 = _shape_param_716 = None
        squeeze_131: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_641, 0);  view_641 = None
        mm_93: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_131);  squeeze_131 = None
        unsqueeze_450: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_93, 0);  mm_93 = None
        view_642: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_450, _shape_param_717);  unsqueeze_450 = _shape_param_717 = None
        permute_438: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_642, [0, 1, 3, 4, 2]);  view_642 = None
        view_643: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_438, _shape_param_718);  permute_438 = _shape_param_718 = None
        add_198: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_634, arg276_1);  arg276_1 = None
        convert_element_type_389: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16);  add_198 = None
        unsqueeze_451: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_389, 4);  convert_element_type_389 = None
        permute_439: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_451, [1, 2, 0, 4, 3]);  unsqueeze_451 = None
        unsqueeze_452: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_637, 4);  view_637 = None
        permute_440: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_452, [1, 2, 4, 0, 3]);  unsqueeze_452 = None
        permute_441: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_439, [0, 1, 2, 4, 3]);  permute_439 = None
        view_644: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_441, _shape_param_719);  permute_441 = _shape_param_719 = None
        permute_442: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_440, [0, 1, 4, 3, 2]);  permute_440 = None
        view_645: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_442, _shape_param_720);  permute_442 = _shape_param_720 = None
        bmm_54: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_644, view_645)
        view_646: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_54, _shape_param_721);  bmm_54 = _shape_param_721 = None
        permute_443: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_646, [0, 1, 2, 4, 3]);  view_646 = None
        view_647: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_443, _shape_param_722);  permute_443 = _shape_param_722 = None
        add_199: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_634, arg277_1);  view_634 = arg277_1 = None
        convert_element_type_390: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16);  add_199 = None
        unsqueeze_453: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_390, 4);  convert_element_type_390 = None
        permute_444: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_453, [1, 2, 0, 4, 3]);  unsqueeze_453 = None
        unsqueeze_454: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_643, 4);  view_643 = None
        permute_445: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_454, [1, 2, 4, 0, 3]);  unsqueeze_454 = None
        permute_446: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_444, [0, 1, 2, 4, 3]);  permute_444 = None
        view_648: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_446, _shape_param_723);  permute_446 = _shape_param_723 = None
        permute_447: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_445, [0, 1, 4, 3, 2]);  permute_445 = None
        view_649: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_447, _shape_param_724);  permute_447 = _shape_param_724 = None
        bmm_55: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_648, view_649)
        view_650: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_55, _shape_param_725);  bmm_55 = _shape_param_725 = None
        permute_448: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_650, [0, 1, 2, 4, 3]);  view_650 = None
        view_651: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_448, _shape_param_726);  permute_448 = _shape_param_726 = None
        view_652: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_651, _shape_param_727);  view_651 = _shape_param_727 = None
        slice_19: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_652, 2, 1, 9223372036854775807);  view_652 = None
        view_653: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_19, _shape_param_728);  slice_19 = _shape_param_728 = None
        index_18: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_653, [None, None, None, iota_2]);  view_653 = None
        add_200: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_647, index_18);  view_647 = index_18 = None
        add_201: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_200, 0);  add_200 = None
        mul_348: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_201, 0.125)
        convert_element_type_391: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_348, torch.float32);  mul_348 = None
        convert_element_type_392: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_201, torch.float32)
        mul_349: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_392, 1);  convert_element_type_392 = None
        amax_36: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_349, [3], True)
        sub_72: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_349, amax_36);  mul_349 = None
        mul_350: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_72, 0.125);  sub_72 = None
        amax_37: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_391, [3], True)
        sub_73: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_391, amax_37)
        abs_19: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_391)
        ne_18: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_19, inf);  abs_19 = None
        eq_18: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_391, convert_element_type_391);  convert_element_type_391 = None
        mul_351: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_18, ne_18);  eq_18 = ne_18 = None
        logical_not_36: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_351);  mul_351 = None
        any_19: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_36, [3], True);  logical_not_36 = None
        logical_not_37: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        where_18: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_37, mul_350, sub_73);  mul_350 = sub_73 = None
        exp_18: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_18);  where_18 = None
        sum_19: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [3], True)
        div_19: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = None
        inductor_lookup_seed_74: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 74)
        inductor_random_74: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_729, inductor_lookup_seed_74, 'rand');  _shape_param_729 = inductor_lookup_seed_74 = None
        gt_74: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_74, 0.1);  inductor_random_74 = None
        mul_352: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_74, div_19);  div_19 = None
        mul_353: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_352, 1.1111111111111112);  mul_352 = None
        convert_element_type_393: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_353, torch.bfloat16);  mul_353 = None
        unsqueeze_455: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_393, 4);  convert_element_type_393 = None
        unsqueeze_456: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_640, 4);  view_640 = None
        permute_449: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_456, [4, 1, 2, 3, 0]);  unsqueeze_456 = None
        view_654: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_455, _shape_param_730);  unsqueeze_455 = _shape_param_730 = None
        permute_450: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_449, [1, 2, 4, 3, 0]);  permute_449 = None
        view_655: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_450, _shape_param_731);  permute_450 = _shape_param_731 = None
        bmm_56: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_654, view_655)
        view_656: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_56, _shape_param_732);  bmm_56 = _shape_param_732 = None
        permute_451: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_656, [2, 0, 1, 4, 3]);  view_656 = None
        view_657: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_451, _shape_param_733);  permute_451 = _shape_param_733 = None
        convert_element_type_394: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.bfloat16);  arg278_1 = None
        unsqueeze_457: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_657, 4);  view_657 = None
        permute_452: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_457, [0, 1, 4, 3, 2]);  unsqueeze_457 = None
        unsqueeze_458: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_394, 3);  convert_element_type_394 = None
        unsqueeze_459: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 4);  unsqueeze_458 = None
        permute_453: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_459, [3, 4, 0, 2, 1]);  unsqueeze_459 = None
        permute_454: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_452, [0, 1, 3, 4, 2]);  permute_452 = None
        clone_37: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
        view_658: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_37, _shape_param_734);  clone_37 = _shape_param_734 = None
        permute_455: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_453, [3, 4, 2, 0, 1]);  permute_453 = None
        clone_38: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
        view_659: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_38, _shape_param_735);  clone_38 = _shape_param_735 = None
        squeeze_132: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_658, 0)
        squeeze_133: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_659, 0)
        mm_94: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_132, squeeze_133);  squeeze_132 = squeeze_133 = None
        unsqueeze_460: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_94, 0);  mm_94 = None
        view_660: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_460, _shape_param_736);  unsqueeze_460 = _shape_param_736 = None
        permute_456: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_660, [0, 1, 4, 2, 3]);  view_660 = None
        view_661: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_456, _shape_param_737);  permute_456 = _shape_param_737 = None
        inductor_lookup_seed_75: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 75)
        inductor_random_75: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_738, inductor_lookup_seed_75, 'rand');  _shape_param_738 = inductor_lookup_seed_75 = None
        convert_element_type_395: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_75, torch.bfloat16);  inductor_random_75 = None
        gt_75: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_395, 0.1);  convert_element_type_395 = None
        mul_354: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_75, view_661);  view_661 = None
        mul_355: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_354, 1.1111111111111112);  mul_354 = None
        add_202: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_355, add_197);  mul_355 = add_197 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
        getitem_72: "f32[512, 16, 1]" = var_mean_36[0]
        getitem_73: "f32[512, 16, 1]" = var_mean_36[1];  var_mean_36 = None
        add_203: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_74: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_202, getitem_73);  add_202 = getitem_73 = None
        mul_356: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_36);  sub_74 = None
        mul_357: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_356, arg279_1);  arg279_1 = None
        add_204: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_357, arg280_1);  mul_357 = arg280_1 = None
        convert_element_type_396: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.bfloat16);  arg281_1 = None
        convert_element_type_397: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.bfloat16);  arg282_1 = None
        convert_element_type_398: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16)
        view_662: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_398, _shape_param_739);  convert_element_type_398 = _shape_param_739 = None
        permute_457: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_397, [1, 0]);  convert_element_type_397 = None
        addmm_36: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_396, view_662, permute_457);  convert_element_type_396 = None
        view_663: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_36, _shape_param_740);  _shape_param_740 = None
        convert_element_type_399: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_663, torch.float32);  view_663 = None
        mul_358: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_399, 0.5)
        mul_359: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_399, 0.7071067811865476);  convert_element_type_399 = None
        erf_18: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_359);  mul_359 = None
        add_205: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_360: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_358, add_205);  mul_358 = add_205 = None
        convert_element_type_400: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16);  mul_360 = None
        inductor_lookup_seed_76: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 76)
        inductor_random_76: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_741, inductor_lookup_seed_76, 'rand');  _shape_param_741 = inductor_lookup_seed_76 = None
        convert_element_type_401: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_76, torch.bfloat16);  inductor_random_76 = None
        gt_76: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_401, 0.1);  convert_element_type_401 = None
        mul_361: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_76, convert_element_type_400);  convert_element_type_400 = None
        mul_362: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_361, 1.1111111111111112);  mul_361 = None
        convert_element_type_402: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.bfloat16);  arg283_1 = None
        convert_element_type_403: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg284_1, torch.bfloat16);  arg284_1 = None
        view_664: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_362, _shape_param_742);  mul_362 = _shape_param_742 = None
        permute_458: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_403, [1, 0]);  convert_element_type_403 = None
        addmm_37: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_402, view_664, permute_458);  convert_element_type_402 = None
        view_665: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_37, _shape_param_743);  addmm_37 = _shape_param_743 = None
        inductor_lookup_seed_77: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 77)
        inductor_random_77: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_744, inductor_lookup_seed_77, 'rand');  _shape_param_744 = inductor_lookup_seed_77 = None
        convert_element_type_404: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_77, torch.bfloat16);  inductor_random_77 = None
        gt_77: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_404, 0.1);  convert_element_type_404 = None
        mul_363: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_77, view_665);  view_665 = None
        mul_364: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_363, 1.1111111111111112);  mul_363 = None
        add_206: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_364, add_204);  mul_364 = add_204 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_206, [2], correction = 0, keepdim = True)
        getitem_74: "f32[512, 16, 1]" = var_mean_37[0]
        getitem_75: "f32[512, 16, 1]" = var_mean_37[1];  var_mean_37 = None
        add_207: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_75: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_206, getitem_75);  add_206 = getitem_75 = None
        mul_365: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_37);  sub_75 = None
        mul_366: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_365, arg285_1);  arg285_1 = None
        add_208: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_366, arg286_1);  mul_366 = arg286_1 = None
        convert_element_type_405: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16)
        convert_element_type_406: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg287_1, torch.bfloat16);  arg287_1 = None
        unsqueeze_461: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_405, 3);  convert_element_type_405 = None
        unsqueeze_462: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 4);  unsqueeze_461 = None
        unsqueeze_463: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_406, 3);  convert_element_type_406 = None
        unsqueeze_464: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 4);  unsqueeze_463 = None
        view_666: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_462, _shape_param_745);  unsqueeze_462 = _shape_param_745 = None
        view_667: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_464, _shape_param_746);  unsqueeze_464 = _shape_param_746 = None
        squeeze_134: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_666, 0)
        squeeze_135: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_667, 0)
        mm_95: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_134, squeeze_135);  squeeze_135 = None
        unsqueeze_465: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_95, 0);  mm_95 = None
        view_668: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_465, _shape_param_747);  unsqueeze_465 = _shape_param_747 = None
        permute_459: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_668, [0, 1, 3, 4, 2]);  view_668 = None
        view_669: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_459, _shape_param_748);  permute_459 = _shape_param_748 = None
        convert_element_type_407: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.bfloat16);  arg288_1 = None
        unsqueeze_466: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_407, 3);  convert_element_type_407 = None
        unsqueeze_467: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 4);  unsqueeze_466 = None
        view_670: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_467, _shape_param_749);  unsqueeze_467 = _shape_param_749 = None
        squeeze_136: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_670, 0)
        mm_96: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_134, squeeze_136);  squeeze_136 = None
        unsqueeze_468: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_96, 0);  mm_96 = None
        view_671: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_468, _shape_param_750);  unsqueeze_468 = _shape_param_750 = None
        permute_460: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_671, [0, 1, 3, 4, 2]);  view_671 = None
        view_672: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_460, _shape_param_751);  permute_460 = _shape_param_751 = None
        convert_element_type_408: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.bfloat16);  arg289_1 = None
        unsqueeze_469: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_408, 3);  convert_element_type_408 = None
        unsqueeze_470: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 4);  unsqueeze_469 = None
        view_673: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_470, _shape_param_752);  unsqueeze_470 = _shape_param_752 = None
        squeeze_137: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_673, 0)
        mm_97: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_134, squeeze_137);  squeeze_134 = squeeze_137 = None
        unsqueeze_471: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_97, 0);  mm_97 = None
        view_674: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_471, _shape_param_753);  unsqueeze_471 = _shape_param_753 = None
        permute_461: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_674, [0, 1, 3, 4, 2]);  view_674 = None
        view_675: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_461, _shape_param_754);  permute_461 = _shape_param_754 = None
        convert_element_type_409: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg290_1, torch.bfloat16);  arg290_1 = None
        unsqueeze_472: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_409, 3);  convert_element_type_409 = None
        unsqueeze_473: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 4);  unsqueeze_472 = None
        view_676: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_473, _shape_param_755);  unsqueeze_473 = _shape_param_755 = None
        squeeze_138: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_676, 0);  view_676 = None
        mm_98: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_138);  squeeze_138 = None
        unsqueeze_474: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_98, 0);  mm_98 = None
        view_677: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_474, _shape_param_756);  unsqueeze_474 = _shape_param_756 = None
        permute_462: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_677, [0, 1, 3, 4, 2]);  view_677 = None
        view_678: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_462, _shape_param_757);  permute_462 = _shape_param_757 = None
        add_209: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_669, arg291_1);  arg291_1 = None
        convert_element_type_410: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16);  add_209 = None
        unsqueeze_475: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_410, 4);  convert_element_type_410 = None
        permute_463: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_475, [1, 2, 0, 4, 3]);  unsqueeze_475 = None
        unsqueeze_476: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_672, 4);  view_672 = None
        permute_464: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_476, [1, 2, 4, 0, 3]);  unsqueeze_476 = None
        permute_465: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_463, [0, 1, 2, 4, 3]);  permute_463 = None
        view_679: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_465, _shape_param_758);  permute_465 = _shape_param_758 = None
        permute_466: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_464, [0, 1, 4, 3, 2]);  permute_464 = None
        view_680: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_466, _shape_param_759);  permute_466 = _shape_param_759 = None
        bmm_57: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_679, view_680)
        view_681: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_57, _shape_param_760);  bmm_57 = _shape_param_760 = None
        permute_467: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_681, [0, 1, 2, 4, 3]);  view_681 = None
        view_682: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_467, _shape_param_761);  permute_467 = _shape_param_761 = None
        add_210: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_669, arg292_1);  view_669 = arg292_1 = None
        convert_element_type_411: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16);  add_210 = None
        unsqueeze_477: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_411, 4);  convert_element_type_411 = None
        permute_468: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_477, [1, 2, 0, 4, 3]);  unsqueeze_477 = None
        unsqueeze_478: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_678, 4);  view_678 = None
        permute_469: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_478, [1, 2, 4, 0, 3]);  unsqueeze_478 = None
        permute_470: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_468, [0, 1, 2, 4, 3]);  permute_468 = None
        view_683: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_470, _shape_param_762);  permute_470 = _shape_param_762 = None
        permute_471: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_469, [0, 1, 4, 3, 2]);  permute_469 = None
        view_684: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_471, _shape_param_763);  permute_471 = _shape_param_763 = None
        bmm_58: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_683, view_684)
        view_685: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_58, _shape_param_764);  bmm_58 = _shape_param_764 = None
        permute_472: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_685, [0, 1, 2, 4, 3]);  view_685 = None
        view_686: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_472, _shape_param_765);  permute_472 = _shape_param_765 = None
        view_687: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_686, _shape_param_766);  view_686 = _shape_param_766 = None
        slice_20: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_687, 2, 1, 9223372036854775807);  view_687 = None
        view_688: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_20, _shape_param_767);  slice_20 = _shape_param_767 = None
        index_19: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_688, [None, None, None, iota_2]);  view_688 = None
        add_211: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_682, index_19);  view_682 = index_19 = None
        add_212: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_211, 0);  add_211 = None
        mul_367: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_212, 0.125)
        convert_element_type_412: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_367, torch.float32);  mul_367 = None
        convert_element_type_413: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_212, torch.float32)
        mul_368: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_413, 1);  convert_element_type_413 = None
        amax_38: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_368, [3], True)
        sub_76: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_368, amax_38);  mul_368 = None
        mul_369: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_76, 0.125);  sub_76 = None
        amax_39: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_412, [3], True)
        sub_77: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_412, amax_39)
        abs_20: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_412)
        ne_19: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_20, inf);  abs_20 = None
        eq_19: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_412, convert_element_type_412);  convert_element_type_412 = None
        mul_370: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_19, ne_19);  eq_19 = ne_19 = None
        logical_not_38: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_370);  mul_370 = None
        any_20: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_38, [3], True);  logical_not_38 = None
        logical_not_39: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        where_19: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_39, mul_369, sub_77);  mul_369 = sub_77 = None
        exp_19: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_19);  where_19 = None
        sum_20: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [3], True)
        div_20: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = None
        inductor_lookup_seed_78: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 78)
        inductor_random_78: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_768, inductor_lookup_seed_78, 'rand');  _shape_param_768 = inductor_lookup_seed_78 = None
        gt_78: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_78, 0.1);  inductor_random_78 = None
        mul_371: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_78, div_20);  div_20 = None
        mul_372: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_371, 1.1111111111111112);  mul_371 = None
        convert_element_type_414: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_372, torch.bfloat16);  mul_372 = None
        unsqueeze_479: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_414, 4);  convert_element_type_414 = None
        unsqueeze_480: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_675, 4);  view_675 = None
        permute_473: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_480, [4, 1, 2, 3, 0]);  unsqueeze_480 = None
        view_689: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_479, _shape_param_769);  unsqueeze_479 = _shape_param_769 = None
        permute_474: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_473, [1, 2, 4, 3, 0]);  permute_473 = None
        view_690: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_474, _shape_param_770);  permute_474 = _shape_param_770 = None
        bmm_59: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_689, view_690)
        view_691: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_59, _shape_param_771);  bmm_59 = _shape_param_771 = None
        permute_475: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_691, [2, 0, 1, 4, 3]);  view_691 = None
        view_692: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_475, _shape_param_772);  permute_475 = _shape_param_772 = None
        convert_element_type_415: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.bfloat16);  arg293_1 = None
        unsqueeze_481: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_692, 4);  view_692 = None
        permute_476: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_481, [0, 1, 4, 3, 2]);  unsqueeze_481 = None
        unsqueeze_482: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_415, 3);  convert_element_type_415 = None
        unsqueeze_483: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 4);  unsqueeze_482 = None
        permute_477: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_483, [3, 4, 0, 2, 1]);  unsqueeze_483 = None
        permute_478: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_476, [0, 1, 3, 4, 2]);  permute_476 = None
        clone_39: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
        view_693: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_39, _shape_param_773);  clone_39 = _shape_param_773 = None
        permute_479: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_477, [3, 4, 2, 0, 1]);  permute_477 = None
        clone_40: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        view_694: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_40, _shape_param_774);  clone_40 = _shape_param_774 = None
        squeeze_139: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_693, 0)
        squeeze_140: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_694, 0)
        mm_99: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_139, squeeze_140);  squeeze_139 = squeeze_140 = None
        unsqueeze_484: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_99, 0);  mm_99 = None
        view_695: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_484, _shape_param_775);  unsqueeze_484 = _shape_param_775 = None
        permute_480: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_695, [0, 1, 4, 2, 3]);  view_695 = None
        view_696: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_480, _shape_param_776);  permute_480 = _shape_param_776 = None
        inductor_lookup_seed_79: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 79)
        inductor_random_79: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_777, inductor_lookup_seed_79, 'rand');  _shape_param_777 = inductor_lookup_seed_79 = None
        convert_element_type_416: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_79, torch.bfloat16);  inductor_random_79 = None
        gt_79: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_416, 0.1);  convert_element_type_416 = None
        mul_373: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_79, view_696);  view_696 = None
        mul_374: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_373, 1.1111111111111112);  mul_373 = None
        add_213: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_374, add_208);  mul_374 = add_208 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_213, [2], correction = 0, keepdim = True)
        getitem_76: "f32[512, 16, 1]" = var_mean_38[0]
        getitem_77: "f32[512, 16, 1]" = var_mean_38[1];  var_mean_38 = None
        add_214: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        sub_78: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_213, getitem_77);  add_213 = getitem_77 = None
        mul_375: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_38);  sub_78 = None
        mul_376: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_375, arg294_1);  arg294_1 = None
        add_215: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_376, arg295_1);  mul_376 = arg295_1 = None
        convert_element_type_417: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.bfloat16);  arg296_1 = None
        convert_element_type_418: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.bfloat16);  arg297_1 = None
        convert_element_type_419: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16)
        view_697: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_419, _shape_param_778);  convert_element_type_419 = _shape_param_778 = None
        permute_481: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_418, [1, 0]);  convert_element_type_418 = None
        addmm_38: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_417, view_697, permute_481);  convert_element_type_417 = None
        view_698: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_38, _shape_param_779);  _shape_param_779 = None
        convert_element_type_420: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_698, torch.float32);  view_698 = None
        mul_377: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_420, 0.5)
        mul_378: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_420, 0.7071067811865476);  convert_element_type_420 = None
        erf_19: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_378);  mul_378 = None
        add_216: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_379: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_377, add_216);  mul_377 = add_216 = None
        convert_element_type_421: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_379, torch.bfloat16);  mul_379 = None
        inductor_lookup_seed_80: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 80)
        inductor_random_80: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_780, inductor_lookup_seed_80, 'rand');  _shape_param_780 = inductor_lookup_seed_80 = None
        convert_element_type_422: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_80, torch.bfloat16);  inductor_random_80 = None
        gt_80: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_422, 0.1);  convert_element_type_422 = None
        mul_380: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_80, convert_element_type_421);  convert_element_type_421 = None
        mul_381: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_380, 1.1111111111111112);  mul_380 = None
        convert_element_type_423: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.bfloat16);  arg298_1 = None
        convert_element_type_424: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg299_1, torch.bfloat16);  arg299_1 = None
        view_699: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_381, _shape_param_781);  mul_381 = _shape_param_781 = None
        permute_482: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_424, [1, 0]);  convert_element_type_424 = None
        addmm_39: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_423, view_699, permute_482);  convert_element_type_423 = None
        view_700: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_39, _shape_param_782);  addmm_39 = _shape_param_782 = None
        inductor_lookup_seed_81: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 81)
        inductor_random_81: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_783, inductor_lookup_seed_81, 'rand');  _shape_param_783 = inductor_lookup_seed_81 = None
        convert_element_type_425: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_81, torch.bfloat16);  inductor_random_81 = None
        gt_81: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_425, 0.1);  convert_element_type_425 = None
        mul_382: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_81, view_700);  view_700 = None
        mul_383: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_382, 1.1111111111111112);  mul_382 = None
        add_217: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_383, add_215);  mul_383 = add_215 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_217, [2], correction = 0, keepdim = True)
        getitem_78: "f32[512, 16, 1]" = var_mean_39[0]
        getitem_79: "f32[512, 16, 1]" = var_mean_39[1];  var_mean_39 = None
        add_218: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_79: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_217, getitem_79);  add_217 = getitem_79 = None
        mul_384: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_39);  sub_79 = None
        mul_385: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_384, arg300_1);  arg300_1 = None
        add_219: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_385, arg301_1);  mul_385 = arg301_1 = None
        convert_element_type_426: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16)
        convert_element_type_427: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.bfloat16);  arg302_1 = None
        unsqueeze_485: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_426, 3);  convert_element_type_426 = None
        unsqueeze_486: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 4);  unsqueeze_485 = None
        unsqueeze_487: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_427, 3);  convert_element_type_427 = None
        unsqueeze_488: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 4);  unsqueeze_487 = None
        view_701: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_486, _shape_param_784);  unsqueeze_486 = _shape_param_784 = None
        view_702: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_488, _shape_param_785);  unsqueeze_488 = _shape_param_785 = None
        squeeze_141: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_701, 0)
        squeeze_142: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_702, 0)
        mm_100: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_141, squeeze_142);  squeeze_142 = None
        unsqueeze_489: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_100, 0);  mm_100 = None
        view_703: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_489, _shape_param_786);  unsqueeze_489 = _shape_param_786 = None
        permute_483: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_703, [0, 1, 3, 4, 2]);  view_703 = None
        view_704: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_483, _shape_param_787);  permute_483 = _shape_param_787 = None
        convert_element_type_428: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.bfloat16);  arg303_1 = None
        unsqueeze_490: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_428, 3);  convert_element_type_428 = None
        unsqueeze_491: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 4);  unsqueeze_490 = None
        view_705: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_491, _shape_param_788);  unsqueeze_491 = _shape_param_788 = None
        squeeze_143: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_705, 0)
        mm_101: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_141, squeeze_143);  squeeze_143 = None
        unsqueeze_492: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_101, 0);  mm_101 = None
        view_706: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_492, _shape_param_789);  unsqueeze_492 = _shape_param_789 = None
        permute_484: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_706, [0, 1, 3, 4, 2]);  view_706 = None
        view_707: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_484, _shape_param_790);  permute_484 = _shape_param_790 = None
        convert_element_type_429: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg304_1, torch.bfloat16);  arg304_1 = None
        unsqueeze_493: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_429, 3);  convert_element_type_429 = None
        unsqueeze_494: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 4);  unsqueeze_493 = None
        view_708: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_494, _shape_param_791);  unsqueeze_494 = _shape_param_791 = None
        squeeze_144: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_708, 0)
        mm_102: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_141, squeeze_144);  squeeze_141 = squeeze_144 = None
        unsqueeze_495: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_102, 0);  mm_102 = None
        view_709: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_495, _shape_param_792);  unsqueeze_495 = _shape_param_792 = None
        permute_485: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_709, [0, 1, 3, 4, 2]);  view_709 = None
        view_710: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_485, _shape_param_793);  permute_485 = _shape_param_793 = None
        convert_element_type_430: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.bfloat16);  arg305_1 = None
        unsqueeze_496: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_430, 3);  convert_element_type_430 = None
        unsqueeze_497: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 4);  unsqueeze_496 = None
        view_711: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_497, _shape_param_794);  unsqueeze_497 = _shape_param_794 = None
        squeeze_145: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_711, 0);  view_711 = None
        mm_103: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_145);  squeeze_145 = None
        unsqueeze_498: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_103, 0);  mm_103 = None
        view_712: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_498, _shape_param_795);  unsqueeze_498 = _shape_param_795 = None
        permute_486: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_712, [0, 1, 3, 4, 2]);  view_712 = None
        view_713: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_486, _shape_param_796);  permute_486 = _shape_param_796 = None
        add_220: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_704, arg306_1);  arg306_1 = None
        convert_element_type_431: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_220, torch.bfloat16);  add_220 = None
        unsqueeze_499: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_431, 4);  convert_element_type_431 = None
        permute_487: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_499, [1, 2, 0, 4, 3]);  unsqueeze_499 = None
        unsqueeze_500: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_707, 4);  view_707 = None
        permute_488: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_500, [1, 2, 4, 0, 3]);  unsqueeze_500 = None
        permute_489: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_487, [0, 1, 2, 4, 3]);  permute_487 = None
        view_714: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_489, _shape_param_797);  permute_489 = _shape_param_797 = None
        permute_490: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_488, [0, 1, 4, 3, 2]);  permute_488 = None
        view_715: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_490, _shape_param_798);  permute_490 = _shape_param_798 = None
        bmm_60: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_714, view_715)
        view_716: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_60, _shape_param_799);  bmm_60 = _shape_param_799 = None
        permute_491: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_716, [0, 1, 2, 4, 3]);  view_716 = None
        view_717: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_491, _shape_param_800);  permute_491 = _shape_param_800 = None
        add_221: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_704, arg307_1);  view_704 = arg307_1 = None
        convert_element_type_432: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None
        unsqueeze_501: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_432, 4);  convert_element_type_432 = None
        permute_492: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_501, [1, 2, 0, 4, 3]);  unsqueeze_501 = None
        unsqueeze_502: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_713, 4);  view_713 = None
        permute_493: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_502, [1, 2, 4, 0, 3]);  unsqueeze_502 = None
        permute_494: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_492, [0, 1, 2, 4, 3]);  permute_492 = None
        view_718: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_494, _shape_param_801);  permute_494 = _shape_param_801 = None
        permute_495: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_493, [0, 1, 4, 3, 2]);  permute_493 = None
        view_719: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_495, _shape_param_802);  permute_495 = _shape_param_802 = None
        bmm_61: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_718, view_719)
        view_720: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_61, _shape_param_803);  bmm_61 = _shape_param_803 = None
        permute_496: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_720, [0, 1, 2, 4, 3]);  view_720 = None
        view_721: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_496, _shape_param_804);  permute_496 = _shape_param_804 = None
        view_722: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_721, _shape_param_805);  view_721 = _shape_param_805 = None
        slice_21: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_722, 2, 1, 9223372036854775807);  view_722 = None
        view_723: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_21, _shape_param_806);  slice_21 = _shape_param_806 = None
        index_20: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_723, [None, None, None, iota_2]);  view_723 = None
        add_222: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_717, index_20);  view_717 = index_20 = None
        add_223: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_222, 0);  add_222 = None
        mul_386: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_223, 0.125)
        convert_element_type_433: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_386, torch.float32);  mul_386 = None
        convert_element_type_434: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_223, torch.float32)
        mul_387: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_434, 1);  convert_element_type_434 = None
        amax_40: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_387, [3], True)
        sub_80: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_387, amax_40);  mul_387 = None
        mul_388: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_80, 0.125);  sub_80 = None
        amax_41: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_433, [3], True)
        sub_81: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_433, amax_41)
        abs_21: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_433)
        ne_20: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_21, inf);  abs_21 = None
        eq_20: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_433, convert_element_type_433);  convert_element_type_433 = None
        mul_389: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_20, ne_20);  eq_20 = ne_20 = None
        logical_not_40: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_389);  mul_389 = None
        any_21: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_40, [3], True);  logical_not_40 = None
        logical_not_41: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        where_20: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_41, mul_388, sub_81);  mul_388 = sub_81 = None
        exp_20: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_20);  where_20 = None
        sum_21: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [3], True)
        div_21: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = None
        inductor_lookup_seed_82: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 82)
        inductor_random_82: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_807, inductor_lookup_seed_82, 'rand');  _shape_param_807 = inductor_lookup_seed_82 = None
        gt_82: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_82, 0.1);  inductor_random_82 = None
        mul_390: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_82, div_21);  div_21 = None
        mul_391: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_390, 1.1111111111111112);  mul_390 = None
        convert_element_type_435: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_391, torch.bfloat16);  mul_391 = None
        unsqueeze_503: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_435, 4);  convert_element_type_435 = None
        unsqueeze_504: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_710, 4);  view_710 = None
        permute_497: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_504, [4, 1, 2, 3, 0]);  unsqueeze_504 = None
        view_724: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_503, _shape_param_808);  unsqueeze_503 = _shape_param_808 = None
        permute_498: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_497, [1, 2, 4, 3, 0]);  permute_497 = None
        view_725: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_498, _shape_param_809);  permute_498 = _shape_param_809 = None
        bmm_62: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_724, view_725)
        view_726: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_62, _shape_param_810);  bmm_62 = _shape_param_810 = None
        permute_499: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_726, [2, 0, 1, 4, 3]);  view_726 = None
        view_727: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_499, _shape_param_811);  permute_499 = _shape_param_811 = None
        convert_element_type_436: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg308_1, torch.bfloat16);  arg308_1 = None
        unsqueeze_505: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_727, 4);  view_727 = None
        permute_500: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_505, [0, 1, 4, 3, 2]);  unsqueeze_505 = None
        unsqueeze_506: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_436, 3);  convert_element_type_436 = None
        unsqueeze_507: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 4);  unsqueeze_506 = None
        permute_501: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_507, [3, 4, 0, 2, 1]);  unsqueeze_507 = None
        permute_502: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_500, [0, 1, 3, 4, 2]);  permute_500 = None
        clone_41: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
        view_728: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_41, _shape_param_812);  clone_41 = _shape_param_812 = None
        permute_503: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_501, [3, 4, 2, 0, 1]);  permute_501 = None
        clone_42: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_503, memory_format = torch.contiguous_format);  permute_503 = None
        view_729: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_42, _shape_param_813);  clone_42 = _shape_param_813 = None
        squeeze_146: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_728, 0)
        squeeze_147: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_729, 0)
        mm_104: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_146, squeeze_147);  squeeze_146 = squeeze_147 = None
        unsqueeze_508: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_104, 0);  mm_104 = None
        view_730: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_508, _shape_param_814);  unsqueeze_508 = _shape_param_814 = None
        permute_504: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_730, [0, 1, 4, 2, 3]);  view_730 = None
        view_731: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_504, _shape_param_815);  permute_504 = _shape_param_815 = None
        inductor_lookup_seed_83: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 83)
        inductor_random_83: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_816, inductor_lookup_seed_83, 'rand');  _shape_param_816 = inductor_lookup_seed_83 = None
        convert_element_type_437: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_83, torch.bfloat16);  inductor_random_83 = None
        gt_83: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_437, 0.1);  convert_element_type_437 = None
        mul_392: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_83, view_731);  view_731 = None
        mul_393: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_392, 1.1111111111111112);  mul_392 = None
        add_224: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_393, add_219);  mul_393 = add_219 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_224, [2], correction = 0, keepdim = True)
        getitem_80: "f32[512, 16, 1]" = var_mean_40[0]
        getitem_81: "f32[512, 16, 1]" = var_mean_40[1];  var_mean_40 = None
        add_225: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        sub_82: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_224, getitem_81);  add_224 = getitem_81 = None
        mul_394: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_40);  sub_82 = None
        mul_395: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_394, arg309_1);  arg309_1 = None
        add_226: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_395, arg310_1);  mul_395 = arg310_1 = None
        convert_element_type_438: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.bfloat16);  arg311_1 = None
        convert_element_type_439: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.bfloat16);  arg312_1 = None
        convert_element_type_440: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16)
        view_732: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_440, _shape_param_817);  convert_element_type_440 = _shape_param_817 = None
        permute_505: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_439, [1, 0]);  convert_element_type_439 = None
        addmm_40: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_438, view_732, permute_505);  convert_element_type_438 = None
        view_733: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_40, _shape_param_818);  _shape_param_818 = None
        convert_element_type_441: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_733, torch.float32);  view_733 = None
        mul_396: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_441, 0.5)
        mul_397: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_441, 0.7071067811865476);  convert_element_type_441 = None
        erf_20: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_397);  mul_397 = None
        add_227: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_398: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_396, add_227);  mul_396 = add_227 = None
        convert_element_type_442: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_398, torch.bfloat16);  mul_398 = None
        inductor_lookup_seed_84: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 84)
        inductor_random_84: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_819, inductor_lookup_seed_84, 'rand');  _shape_param_819 = inductor_lookup_seed_84 = None
        convert_element_type_443: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_84, torch.bfloat16);  inductor_random_84 = None
        gt_84: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_443, 0.1);  convert_element_type_443 = None
        mul_399: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_84, convert_element_type_442);  convert_element_type_442 = None
        mul_400: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_399, 1.1111111111111112);  mul_399 = None
        convert_element_type_444: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg313_1, torch.bfloat16);  arg313_1 = None
        convert_element_type_445: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg314_1, torch.bfloat16);  arg314_1 = None
        view_734: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_400, _shape_param_820);  mul_400 = _shape_param_820 = None
        permute_506: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_445, [1, 0]);  convert_element_type_445 = None
        addmm_41: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_444, view_734, permute_506);  convert_element_type_444 = None
        view_735: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_41, _shape_param_821);  addmm_41 = _shape_param_821 = None
        inductor_lookup_seed_85: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 85)
        inductor_random_85: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_822, inductor_lookup_seed_85, 'rand');  _shape_param_822 = inductor_lookup_seed_85 = None
        convert_element_type_446: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_85, torch.bfloat16);  inductor_random_85 = None
        gt_85: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_446, 0.1);  convert_element_type_446 = None
        mul_401: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_85, view_735);  view_735 = None
        mul_402: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_401, 1.1111111111111112);  mul_401 = None
        add_228: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_402, add_226);  mul_402 = add_226 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_228, [2], correction = 0, keepdim = True)
        getitem_82: "f32[512, 16, 1]" = var_mean_41[0]
        getitem_83: "f32[512, 16, 1]" = var_mean_41[1];  var_mean_41 = None
        add_229: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_83: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_228, getitem_83);  add_228 = getitem_83 = None
        mul_403: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_41);  sub_83 = None
        mul_404: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_403, arg315_1);  arg315_1 = None
        add_230: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_404, arg316_1);  mul_404 = arg316_1 = None
        convert_element_type_447: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_230, torch.bfloat16)
        convert_element_type_448: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.bfloat16);  arg317_1 = None
        unsqueeze_509: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_447, 3);  convert_element_type_447 = None
        unsqueeze_510: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 4);  unsqueeze_509 = None
        unsqueeze_511: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_448, 3);  convert_element_type_448 = None
        unsqueeze_512: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 4);  unsqueeze_511 = None
        view_736: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_510, _shape_param_823);  unsqueeze_510 = _shape_param_823 = None
        view_737: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_512, _shape_param_824);  unsqueeze_512 = _shape_param_824 = None
        squeeze_148: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_736, 0)
        squeeze_149: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_737, 0)
        mm_105: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_148, squeeze_149);  squeeze_149 = None
        unsqueeze_513: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_105, 0);  mm_105 = None
        view_738: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_513, _shape_param_825);  unsqueeze_513 = _shape_param_825 = None
        permute_507: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_738, [0, 1, 3, 4, 2]);  view_738 = None
        view_739: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_507, _shape_param_826);  permute_507 = _shape_param_826 = None
        convert_element_type_449: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.bfloat16);  arg318_1 = None
        unsqueeze_514: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_449, 3);  convert_element_type_449 = None
        unsqueeze_515: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 4);  unsqueeze_514 = None
        view_740: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_515, _shape_param_827);  unsqueeze_515 = _shape_param_827 = None
        squeeze_150: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_740, 0)
        mm_106: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_148, squeeze_150);  squeeze_150 = None
        unsqueeze_516: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_106, 0);  mm_106 = None
        view_741: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_516, _shape_param_828);  unsqueeze_516 = _shape_param_828 = None
        permute_508: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_741, [0, 1, 3, 4, 2]);  view_741 = None
        view_742: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_508, _shape_param_829);  permute_508 = _shape_param_829 = None
        convert_element_type_450: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg319_1, torch.bfloat16);  arg319_1 = None
        unsqueeze_517: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_450, 3);  convert_element_type_450 = None
        unsqueeze_518: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 4);  unsqueeze_517 = None
        view_743: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_518, _shape_param_830);  unsqueeze_518 = _shape_param_830 = None
        squeeze_151: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_743, 0)
        mm_107: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_148, squeeze_151);  squeeze_148 = squeeze_151 = None
        unsqueeze_519: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_107, 0);  mm_107 = None
        view_744: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_519, _shape_param_831);  unsqueeze_519 = _shape_param_831 = None
        permute_509: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_744, [0, 1, 3, 4, 2]);  view_744 = None
        view_745: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_509, _shape_param_832);  permute_509 = _shape_param_832 = None
        convert_element_type_451: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.bfloat16);  arg320_1 = None
        unsqueeze_520: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_451, 3);  convert_element_type_451 = None
        unsqueeze_521: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 4);  unsqueeze_520 = None
        view_746: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_521, _shape_param_833);  unsqueeze_521 = _shape_param_833 = None
        squeeze_152: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_746, 0);  view_746 = None
        mm_108: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_152);  squeeze_152 = None
        unsqueeze_522: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_108, 0);  mm_108 = None
        view_747: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_522, _shape_param_834);  unsqueeze_522 = _shape_param_834 = None
        permute_510: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_747, [0, 1, 3, 4, 2]);  view_747 = None
        view_748: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_510, _shape_param_835);  permute_510 = _shape_param_835 = None
        add_231: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_739, arg321_1);  arg321_1 = None
        convert_element_type_452: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_231, torch.bfloat16);  add_231 = None
        unsqueeze_523: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_452, 4);  convert_element_type_452 = None
        permute_511: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_523, [1, 2, 0, 4, 3]);  unsqueeze_523 = None
        unsqueeze_524: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_742, 4);  view_742 = None
        permute_512: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_524, [1, 2, 4, 0, 3]);  unsqueeze_524 = None
        permute_513: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_511, [0, 1, 2, 4, 3]);  permute_511 = None
        view_749: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_513, _shape_param_836);  permute_513 = _shape_param_836 = None
        permute_514: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_512, [0, 1, 4, 3, 2]);  permute_512 = None
        view_750: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_514, _shape_param_837);  permute_514 = _shape_param_837 = None
        bmm_63: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_749, view_750)
        view_751: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_63, _shape_param_838);  bmm_63 = _shape_param_838 = None
        permute_515: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_751, [0, 1, 2, 4, 3]);  view_751 = None
        view_752: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_515, _shape_param_839);  permute_515 = _shape_param_839 = None
        add_232: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_739, arg322_1);  view_739 = arg322_1 = None
        convert_element_type_453: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_232, torch.bfloat16);  add_232 = None
        unsqueeze_525: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_453, 4);  convert_element_type_453 = None
        permute_516: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_525, [1, 2, 0, 4, 3]);  unsqueeze_525 = None
        unsqueeze_526: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_748, 4);  view_748 = None
        permute_517: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_526, [1, 2, 4, 0, 3]);  unsqueeze_526 = None
        permute_518: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_516, [0, 1, 2, 4, 3]);  permute_516 = None
        view_753: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_518, _shape_param_840);  permute_518 = _shape_param_840 = None
        permute_519: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_517, [0, 1, 4, 3, 2]);  permute_517 = None
        view_754: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_519, _shape_param_841);  permute_519 = _shape_param_841 = None
        bmm_64: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_753, view_754)
        view_755: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_64, _shape_param_842);  bmm_64 = _shape_param_842 = None
        permute_520: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_755, [0, 1, 2, 4, 3]);  view_755 = None
        view_756: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_520, _shape_param_843);  permute_520 = _shape_param_843 = None
        view_757: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_756, _shape_param_844);  view_756 = _shape_param_844 = None
        slice_22: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_757, 2, 1, 9223372036854775807);  view_757 = None
        view_758: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_22, _shape_param_845);  slice_22 = _shape_param_845 = None
        index_21: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_758, [None, None, None, iota_2]);  view_758 = None
        add_233: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_752, index_21);  view_752 = index_21 = None
        add_234: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_233, 0);  add_233 = None
        mul_405: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_234, 0.125)
        convert_element_type_454: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_405, torch.float32);  mul_405 = None
        convert_element_type_455: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_234, torch.float32)
        mul_406: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_455, 1);  convert_element_type_455 = None
        amax_42: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_406, [3], True)
        sub_84: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_406, amax_42);  mul_406 = None
        mul_407: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_84, 0.125);  sub_84 = None
        amax_43: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_454, [3], True)
        sub_85: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_454, amax_43)
        abs_22: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_454)
        ne_21: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_22, inf);  abs_22 = None
        eq_21: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_454, convert_element_type_454);  convert_element_type_454 = None
        mul_408: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_21, ne_21);  eq_21 = ne_21 = None
        logical_not_42: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_408);  mul_408 = None
        any_22: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_42, [3], True);  logical_not_42 = None
        logical_not_43: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        where_21: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_43, mul_407, sub_85);  mul_407 = sub_85 = None
        exp_21: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_21);  where_21 = None
        sum_22: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [3], True)
        div_22: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = None
        inductor_lookup_seed_86: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 86)
        inductor_random_86: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_846, inductor_lookup_seed_86, 'rand');  _shape_param_846 = inductor_lookup_seed_86 = None
        gt_86: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_86, 0.1);  inductor_random_86 = None
        mul_409: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_86, div_22);  div_22 = None
        mul_410: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_409, 1.1111111111111112);  mul_409 = None
        convert_element_type_456: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_410, torch.bfloat16);  mul_410 = None
        unsqueeze_527: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_456, 4);  convert_element_type_456 = None
        unsqueeze_528: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_745, 4);  view_745 = None
        permute_521: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_528, [4, 1, 2, 3, 0]);  unsqueeze_528 = None
        view_759: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_527, _shape_param_847);  unsqueeze_527 = _shape_param_847 = None
        permute_522: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_521, [1, 2, 4, 3, 0]);  permute_521 = None
        view_760: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_522, _shape_param_848);  permute_522 = _shape_param_848 = None
        bmm_65: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_759, view_760)
        view_761: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_65, _shape_param_849);  bmm_65 = _shape_param_849 = None
        permute_523: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_761, [2, 0, 1, 4, 3]);  view_761 = None
        view_762: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_523, _shape_param_850);  permute_523 = _shape_param_850 = None
        convert_element_type_457: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.bfloat16);  arg323_1 = None
        unsqueeze_529: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_762, 4);  view_762 = None
        permute_524: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_529, [0, 1, 4, 3, 2]);  unsqueeze_529 = None
        unsqueeze_530: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_457, 3);  convert_element_type_457 = None
        unsqueeze_531: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 4);  unsqueeze_530 = None
        permute_525: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_531, [3, 4, 0, 2, 1]);  unsqueeze_531 = None
        permute_526: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_524, [0, 1, 3, 4, 2]);  permute_524 = None
        clone_43: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_526, memory_format = torch.contiguous_format);  permute_526 = None
        view_763: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_43, _shape_param_851);  clone_43 = _shape_param_851 = None
        permute_527: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_525, [3, 4, 2, 0, 1]);  permute_525 = None
        clone_44: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_527, memory_format = torch.contiguous_format);  permute_527 = None
        view_764: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_44, _shape_param_852);  clone_44 = _shape_param_852 = None
        squeeze_153: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_763, 0)
        squeeze_154: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_764, 0)
        mm_109: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_153, squeeze_154);  squeeze_153 = squeeze_154 = None
        unsqueeze_532: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_109, 0);  mm_109 = None
        view_765: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_532, _shape_param_853);  unsqueeze_532 = _shape_param_853 = None
        permute_528: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_765, [0, 1, 4, 2, 3]);  view_765 = None
        view_766: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_528, _shape_param_854);  permute_528 = _shape_param_854 = None
        inductor_lookup_seed_87: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 87)
        inductor_random_87: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_855, inductor_lookup_seed_87, 'rand');  _shape_param_855 = inductor_lookup_seed_87 = None
        convert_element_type_458: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_87, torch.bfloat16);  inductor_random_87 = None
        gt_87: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_458, 0.1);  convert_element_type_458 = None
        mul_411: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_87, view_766);  view_766 = None
        mul_412: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_411, 1.1111111111111112);  mul_411 = None
        add_235: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_412, add_230);  mul_412 = add_230 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_235, [2], correction = 0, keepdim = True)
        getitem_84: "f32[512, 16, 1]" = var_mean_42[0]
        getitem_85: "f32[512, 16, 1]" = var_mean_42[1];  var_mean_42 = None
        add_236: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_86: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_235, getitem_85);  add_235 = getitem_85 = None
        mul_413: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_42);  sub_86 = None
        mul_414: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_413, arg324_1);  arg324_1 = None
        add_237: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_414, arg325_1);  mul_414 = arg325_1 = None
        convert_element_type_459: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg326_1, torch.bfloat16);  arg326_1 = None
        convert_element_type_460: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.bfloat16);  arg327_1 = None
        convert_element_type_461: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16)
        view_767: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_461, _shape_param_856);  convert_element_type_461 = _shape_param_856 = None
        permute_529: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_460, [1, 0]);  convert_element_type_460 = None
        addmm_42: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_459, view_767, permute_529);  convert_element_type_459 = None
        view_768: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_42, _shape_param_857);  _shape_param_857 = None
        convert_element_type_462: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_768, torch.float32);  view_768 = None
        mul_415: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_462, 0.5)
        mul_416: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_462, 0.7071067811865476);  convert_element_type_462 = None
        erf_21: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_416);  mul_416 = None
        add_238: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_417: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_415, add_238);  mul_415 = add_238 = None
        convert_element_type_463: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_417, torch.bfloat16);  mul_417 = None
        inductor_lookup_seed_88: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 88)
        inductor_random_88: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_858, inductor_lookup_seed_88, 'rand');  _shape_param_858 = inductor_lookup_seed_88 = None
        convert_element_type_464: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_88, torch.bfloat16);  inductor_random_88 = None
        gt_88: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_464, 0.1);  convert_element_type_464 = None
        mul_418: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_88, convert_element_type_463);  convert_element_type_463 = None
        mul_419: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_418, 1.1111111111111112);  mul_418 = None
        convert_element_type_465: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.bfloat16);  arg328_1 = None
        convert_element_type_466: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg329_1, torch.bfloat16);  arg329_1 = None
        view_769: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_419, _shape_param_859);  mul_419 = _shape_param_859 = None
        permute_530: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_466, [1, 0]);  convert_element_type_466 = None
        addmm_43: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_465, view_769, permute_530);  convert_element_type_465 = None
        view_770: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_43, _shape_param_860);  addmm_43 = _shape_param_860 = None
        inductor_lookup_seed_89: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 89)
        inductor_random_89: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_861, inductor_lookup_seed_89, 'rand');  _shape_param_861 = inductor_lookup_seed_89 = None
        convert_element_type_467: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_89, torch.bfloat16);  inductor_random_89 = None
        gt_89: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_467, 0.1);  convert_element_type_467 = None
        mul_420: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_89, view_770);  view_770 = None
        mul_421: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_420, 1.1111111111111112);  mul_420 = None
        add_239: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_421, add_237);  mul_421 = add_237 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_239, [2], correction = 0, keepdim = True)
        getitem_86: "f32[512, 16, 1]" = var_mean_43[0]
        getitem_87: "f32[512, 16, 1]" = var_mean_43[1];  var_mean_43 = None
        add_240: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_87: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_239, getitem_87);  add_239 = getitem_87 = None
        mul_422: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_43);  sub_87 = None
        mul_423: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_422, arg330_1);  arg330_1 = None
        add_241: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_423, arg331_1);  mul_423 = arg331_1 = None
        convert_element_type_468: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_241, torch.bfloat16)
        convert_element_type_469: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg332_1, torch.bfloat16);  arg332_1 = None
        unsqueeze_533: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_468, 3);  convert_element_type_468 = None
        unsqueeze_534: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 4);  unsqueeze_533 = None
        unsqueeze_535: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_469, 3);  convert_element_type_469 = None
        unsqueeze_536: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 4);  unsqueeze_535 = None
        view_771: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_534, _shape_param_862);  unsqueeze_534 = _shape_param_862 = None
        view_772: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_536, _shape_param_863);  unsqueeze_536 = _shape_param_863 = None
        squeeze_155: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_771, 0)
        squeeze_156: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_772, 0)
        mm_110: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_155, squeeze_156);  squeeze_156 = None
        unsqueeze_537: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_110, 0);  mm_110 = None
        view_773: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_537, _shape_param_864);  unsqueeze_537 = _shape_param_864 = None
        permute_531: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_773, [0, 1, 3, 4, 2]);  view_773 = None
        view_774: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_531, _shape_param_865);  permute_531 = _shape_param_865 = None
        convert_element_type_470: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.bfloat16);  arg333_1 = None
        unsqueeze_538: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_470, 3);  convert_element_type_470 = None
        unsqueeze_539: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 4);  unsqueeze_538 = None
        view_775: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_539, _shape_param_866);  unsqueeze_539 = _shape_param_866 = None
        squeeze_157: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_775, 0)
        mm_111: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_155, squeeze_157);  squeeze_157 = None
        unsqueeze_540: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_111, 0);  mm_111 = None
        view_776: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_540, _shape_param_867);  unsqueeze_540 = _shape_param_867 = None
        permute_532: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_776, [0, 1, 3, 4, 2]);  view_776 = None
        view_777: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_532, _shape_param_868);  permute_532 = _shape_param_868 = None
        convert_element_type_471: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg334_1, torch.bfloat16);  arg334_1 = None
        unsqueeze_541: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_471, 3);  convert_element_type_471 = None
        unsqueeze_542: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 4);  unsqueeze_541 = None
        view_778: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_542, _shape_param_869);  unsqueeze_542 = _shape_param_869 = None
        squeeze_158: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_778, 0)
        mm_112: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_155, squeeze_158);  squeeze_155 = squeeze_158 = None
        unsqueeze_543: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_112, 0);  mm_112 = None
        view_779: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_543, _shape_param_870);  unsqueeze_543 = _shape_param_870 = None
        permute_533: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_779, [0, 1, 3, 4, 2]);  view_779 = None
        view_780: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_533, _shape_param_871);  permute_533 = _shape_param_871 = None
        convert_element_type_472: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg335_1, torch.bfloat16);  arg335_1 = None
        unsqueeze_544: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_472, 3);  convert_element_type_472 = None
        unsqueeze_545: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 4);  unsqueeze_544 = None
        view_781: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_545, _shape_param_872);  unsqueeze_545 = _shape_param_872 = None
        squeeze_159: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_781, 0);  view_781 = None
        mm_113: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_159);  squeeze_159 = None
        unsqueeze_546: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_113, 0);  mm_113 = None
        view_782: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_546, _shape_param_873);  unsqueeze_546 = _shape_param_873 = None
        permute_534: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_782, [0, 1, 3, 4, 2]);  view_782 = None
        view_783: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_534, _shape_param_874);  permute_534 = _shape_param_874 = None
        add_242: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_774, arg336_1);  arg336_1 = None
        convert_element_type_473: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_242, torch.bfloat16);  add_242 = None
        unsqueeze_547: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_473, 4);  convert_element_type_473 = None
        permute_535: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_547, [1, 2, 0, 4, 3]);  unsqueeze_547 = None
        unsqueeze_548: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_777, 4);  view_777 = None
        permute_536: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_548, [1, 2, 4, 0, 3]);  unsqueeze_548 = None
        permute_537: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_535, [0, 1, 2, 4, 3]);  permute_535 = None
        view_784: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_537, _shape_param_875);  permute_537 = _shape_param_875 = None
        permute_538: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_536, [0, 1, 4, 3, 2]);  permute_536 = None
        view_785: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_538, _shape_param_876);  permute_538 = _shape_param_876 = None
        bmm_66: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_784, view_785)
        view_786: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_66, _shape_param_877);  bmm_66 = _shape_param_877 = None
        permute_539: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_786, [0, 1, 2, 4, 3]);  view_786 = None
        view_787: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_539, _shape_param_878);  permute_539 = _shape_param_878 = None
        add_243: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_774, arg337_1);  view_774 = arg337_1 = None
        convert_element_type_474: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_243, torch.bfloat16);  add_243 = None
        unsqueeze_549: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_474, 4);  convert_element_type_474 = None
        permute_540: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_549, [1, 2, 0, 4, 3]);  unsqueeze_549 = None
        unsqueeze_550: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_783, 4);  view_783 = None
        permute_541: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_550, [1, 2, 4, 0, 3]);  unsqueeze_550 = None
        permute_542: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_540, [0, 1, 2, 4, 3]);  permute_540 = None
        view_788: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_542, _shape_param_879);  permute_542 = _shape_param_879 = None
        permute_543: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_541, [0, 1, 4, 3, 2]);  permute_541 = None
        view_789: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_543, _shape_param_880);  permute_543 = _shape_param_880 = None
        bmm_67: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_788, view_789)
        view_790: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_67, _shape_param_881);  bmm_67 = _shape_param_881 = None
        permute_544: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_790, [0, 1, 2, 4, 3]);  view_790 = None
        view_791: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_544, _shape_param_882);  permute_544 = _shape_param_882 = None
        view_792: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_791, _shape_param_883);  view_791 = _shape_param_883 = None
        slice_23: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_792, 2, 1, 9223372036854775807);  view_792 = None
        view_793: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_23, _shape_param_884);  slice_23 = _shape_param_884 = None
        index_22: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_793, [None, None, None, iota_2]);  view_793 = None
        add_244: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_787, index_22);  view_787 = index_22 = None
        add_245: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_244, 0);  add_244 = None
        mul_424: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_245, 0.125)
        convert_element_type_475: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_424, torch.float32);  mul_424 = None
        convert_element_type_476: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_245, torch.float32)
        mul_425: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_476, 1);  convert_element_type_476 = None
        amax_44: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_425, [3], True)
        sub_88: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_425, amax_44);  mul_425 = None
        mul_426: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_88, 0.125);  sub_88 = None
        amax_45: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_475, [3], True)
        sub_89: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_475, amax_45)
        abs_23: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_475)
        ne_22: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_23, inf);  abs_23 = None
        eq_22: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_475, convert_element_type_475);  convert_element_type_475 = None
        mul_427: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_22, ne_22);  eq_22 = ne_22 = None
        logical_not_44: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_427);  mul_427 = None
        any_23: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_44, [3], True);  logical_not_44 = None
        logical_not_45: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        where_22: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_45, mul_426, sub_89);  mul_426 = sub_89 = None
        exp_22: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_22);  where_22 = None
        sum_23: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [3], True)
        div_23: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = None
        inductor_lookup_seed_90: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 90)
        inductor_random_90: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_885, inductor_lookup_seed_90, 'rand');  _shape_param_885 = inductor_lookup_seed_90 = None
        gt_90: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_90, 0.1);  inductor_random_90 = None
        mul_428: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_90, div_23);  div_23 = None
        mul_429: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_428, 1.1111111111111112);  mul_428 = None
        convert_element_type_477: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_429, torch.bfloat16);  mul_429 = None
        unsqueeze_551: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_477, 4);  convert_element_type_477 = None
        unsqueeze_552: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_780, 4);  view_780 = None
        permute_545: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_552, [4, 1, 2, 3, 0]);  unsqueeze_552 = None
        view_794: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_551, _shape_param_886);  unsqueeze_551 = _shape_param_886 = None
        permute_546: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_545, [1, 2, 4, 3, 0]);  permute_545 = None
        view_795: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_546, _shape_param_887);  permute_546 = _shape_param_887 = None
        bmm_68: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_794, view_795)
        view_796: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_68, _shape_param_888);  bmm_68 = _shape_param_888 = None
        permute_547: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_796, [2, 0, 1, 4, 3]);  view_796 = None
        view_797: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_547, _shape_param_889);  permute_547 = _shape_param_889 = None
        convert_element_type_478: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.bfloat16);  arg338_1 = None
        unsqueeze_553: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_797, 4);  view_797 = None
        permute_548: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_553, [0, 1, 4, 3, 2]);  unsqueeze_553 = None
        unsqueeze_554: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_478, 3);  convert_element_type_478 = None
        unsqueeze_555: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 4);  unsqueeze_554 = None
        permute_549: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_555, [3, 4, 0, 2, 1]);  unsqueeze_555 = None
        permute_550: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_548, [0, 1, 3, 4, 2]);  permute_548 = None
        clone_45: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_550, memory_format = torch.contiguous_format);  permute_550 = None
        view_798: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_45, _shape_param_890);  clone_45 = _shape_param_890 = None
        permute_551: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_549, [3, 4, 2, 0, 1]);  permute_549 = None
        clone_46: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        view_799: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_46, _shape_param_891);  clone_46 = _shape_param_891 = None
        squeeze_160: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_798, 0)
        squeeze_161: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_799, 0)
        mm_114: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_160, squeeze_161);  squeeze_160 = squeeze_161 = None
        unsqueeze_556: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_114, 0);  mm_114 = None
        view_800: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_556, _shape_param_892);  unsqueeze_556 = _shape_param_892 = None
        permute_552: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_800, [0, 1, 4, 2, 3]);  view_800 = None
        view_801: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_552, _shape_param_893);  permute_552 = _shape_param_893 = None
        inductor_lookup_seed_91: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 91)
        inductor_random_91: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_894, inductor_lookup_seed_91, 'rand');  _shape_param_894 = inductor_lookup_seed_91 = None
        convert_element_type_479: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_91, torch.bfloat16);  inductor_random_91 = None
        gt_91: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_479, 0.1);  convert_element_type_479 = None
        mul_430: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_91, view_801);  view_801 = None
        mul_431: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_430, 1.1111111111111112);  mul_430 = None
        add_246: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_431, add_241);  mul_431 = add_241 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_246, [2], correction = 0, keepdim = True)
        getitem_88: "f32[512, 16, 1]" = var_mean_44[0]
        getitem_89: "f32[512, 16, 1]" = var_mean_44[1];  var_mean_44 = None
        add_247: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_90: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_246, getitem_89);  add_246 = getitem_89 = None
        mul_432: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_44);  sub_90 = None
        mul_433: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_432, arg339_1);  arg339_1 = None
        add_248: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_433, arg340_1);  mul_433 = arg340_1 = None
        convert_element_type_480: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg341_1, torch.bfloat16);  arg341_1 = None
        convert_element_type_481: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.bfloat16);  arg342_1 = None
        convert_element_type_482: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_248, torch.bfloat16)
        view_802: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_482, _shape_param_895);  convert_element_type_482 = _shape_param_895 = None
        permute_553: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_481, [1, 0]);  convert_element_type_481 = None
        addmm_44: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_480, view_802, permute_553);  convert_element_type_480 = None
        view_803: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_44, _shape_param_896);  _shape_param_896 = None
        convert_element_type_483: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_803, torch.float32);  view_803 = None
        mul_434: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_483, 0.5)
        mul_435: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_483, 0.7071067811865476);  convert_element_type_483 = None
        erf_22: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_435);  mul_435 = None
        add_249: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_436: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_434, add_249);  mul_434 = add_249 = None
        convert_element_type_484: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_436, torch.bfloat16);  mul_436 = None
        inductor_lookup_seed_92: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 92)
        inductor_random_92: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_897, inductor_lookup_seed_92, 'rand');  _shape_param_897 = inductor_lookup_seed_92 = None
        convert_element_type_485: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_92, torch.bfloat16);  inductor_random_92 = None
        gt_92: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_485, 0.1);  convert_element_type_485 = None
        mul_437: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_92, convert_element_type_484);  convert_element_type_484 = None
        mul_438: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_437, 1.1111111111111112);  mul_437 = None
        convert_element_type_486: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.bfloat16);  arg343_1 = None
        convert_element_type_487: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.bfloat16);  arg344_1 = None
        view_804: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_438, _shape_param_898);  mul_438 = _shape_param_898 = None
        permute_554: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_487, [1, 0]);  convert_element_type_487 = None
        addmm_45: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_486, view_804, permute_554);  convert_element_type_486 = None
        view_805: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_45, _shape_param_899);  addmm_45 = _shape_param_899 = None
        inductor_lookup_seed_93: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 93)
        inductor_random_93: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_900, inductor_lookup_seed_93, 'rand');  _shape_param_900 = inductor_lookup_seed_93 = None
        convert_element_type_488: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_93, torch.bfloat16);  inductor_random_93 = None
        gt_93: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_488, 0.1);  convert_element_type_488 = None
        mul_439: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_93, view_805);  view_805 = None
        mul_440: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_439, 1.1111111111111112);  mul_439 = None
        add_250: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_440, add_248);  mul_440 = add_248 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_250, [2], correction = 0, keepdim = True)
        getitem_90: "f32[512, 16, 1]" = var_mean_45[0]
        getitem_91: "f32[512, 16, 1]" = var_mean_45[1];  var_mean_45 = None
        add_251: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_91: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_250, getitem_91);  add_250 = getitem_91 = None
        mul_441: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_45);  sub_91 = None
        mul_442: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_441, arg345_1);  arg345_1 = None
        add_252: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_442, arg346_1);  mul_442 = arg346_1 = None
        convert_element_type_489: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16)
        convert_element_type_490: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg347_1, torch.bfloat16);  arg347_1 = None
        unsqueeze_557: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_489, 3);  convert_element_type_489 = None
        unsqueeze_558: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 4);  unsqueeze_557 = None
        unsqueeze_559: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_490, 3);  convert_element_type_490 = None
        unsqueeze_560: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 4);  unsqueeze_559 = None
        view_806: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_558, _shape_param_901);  unsqueeze_558 = _shape_param_901 = None
        view_807: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_560, _shape_param_902);  unsqueeze_560 = _shape_param_902 = None
        squeeze_162: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_806, 0)
        squeeze_163: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_807, 0)
        mm_115: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_162, squeeze_163);  squeeze_163 = None
        unsqueeze_561: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_115, 0);  mm_115 = None
        view_808: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_561, _shape_param_903);  unsqueeze_561 = _shape_param_903 = None
        permute_555: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_808, [0, 1, 3, 4, 2]);  view_808 = None
        view_809: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_555, _shape_param_904);  permute_555 = _shape_param_904 = None
        convert_element_type_491: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg348_1, torch.bfloat16);  arg348_1 = None
        unsqueeze_562: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_491, 3);  convert_element_type_491 = None
        unsqueeze_563: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 4);  unsqueeze_562 = None
        view_810: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_563, _shape_param_905);  unsqueeze_563 = _shape_param_905 = None
        squeeze_164: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_810, 0)
        mm_116: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_162, squeeze_164);  squeeze_164 = None
        unsqueeze_564: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_116, 0);  mm_116 = None
        view_811: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_564, _shape_param_906);  unsqueeze_564 = _shape_param_906 = None
        permute_556: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_811, [0, 1, 3, 4, 2]);  view_811 = None
        view_812: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_556, _shape_param_907);  permute_556 = _shape_param_907 = None
        convert_element_type_492: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg349_1, torch.bfloat16);  arg349_1 = None
        unsqueeze_565: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_492, 3);  convert_element_type_492 = None
        unsqueeze_566: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 4);  unsqueeze_565 = None
        view_813: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_566, _shape_param_908);  unsqueeze_566 = _shape_param_908 = None
        squeeze_165: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_813, 0)
        mm_117: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_162, squeeze_165);  squeeze_162 = squeeze_165 = None
        unsqueeze_567: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_117, 0);  mm_117 = None
        view_814: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_567, _shape_param_909);  unsqueeze_567 = _shape_param_909 = None
        permute_557: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_814, [0, 1, 3, 4, 2]);  view_814 = None
        view_815: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_557, _shape_param_910);  permute_557 = _shape_param_910 = None
        convert_element_type_493: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg350_1, torch.bfloat16);  arg350_1 = None
        unsqueeze_568: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_493, 3);  convert_element_type_493 = None
        unsqueeze_569: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 4);  unsqueeze_568 = None
        view_816: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_569, _shape_param_911);  unsqueeze_569 = _shape_param_911 = None
        squeeze_166: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_816, 0);  view_816 = None
        mm_118: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_166);  squeeze_4 = squeeze_166 = None
        unsqueeze_570: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_118, 0);  mm_118 = None
        view_817: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_570, _shape_param_912);  unsqueeze_570 = _shape_param_912 = None
        permute_558: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_817, [0, 1, 3, 4, 2]);  view_817 = None
        view_818: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_558, _shape_param_913);  permute_558 = _shape_param_913 = None
        add_253: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_809, arg351_1);  arg351_1 = None
        convert_element_type_494: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_253, torch.bfloat16);  add_253 = None
        unsqueeze_571: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_494, 4);  convert_element_type_494 = None
        permute_559: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_571, [1, 2, 0, 4, 3]);  unsqueeze_571 = None
        unsqueeze_572: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_812, 4);  view_812 = None
        permute_560: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_572, [1, 2, 4, 0, 3]);  unsqueeze_572 = None
        permute_561: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_559, [0, 1, 2, 4, 3]);  permute_559 = None
        view_819: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_561, _shape_param_914);  permute_561 = _shape_param_914 = None
        permute_562: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_560, [0, 1, 4, 3, 2]);  permute_560 = None
        view_820: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_562, _shape_param_915);  permute_562 = _shape_param_915 = None
        bmm_69: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_819, view_820)
        view_821: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_69, _shape_param_916);  bmm_69 = _shape_param_916 = None
        permute_563: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_821, [0, 1, 2, 4, 3]);  view_821 = None
        view_822: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_563, _shape_param_917);  permute_563 = _shape_param_917 = None
        add_254: "f32[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_809, arg352_1);  view_809 = arg352_1 = None
        convert_element_type_495: "bf16[512, 16, 16, 64]" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16);  add_254 = None
        unsqueeze_573: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_495, 4);  convert_element_type_495 = None
        permute_564: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_573, [1, 2, 0, 4, 3]);  unsqueeze_573 = None
        unsqueeze_574: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_818, 4);  view_818 = None
        permute_565: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_574, [1, 2, 4, 0, 3]);  unsqueeze_574 = None
        permute_566: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_564, [0, 1, 2, 4, 3]);  permute_564 = None
        view_823: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_566, _shape_param_918);  permute_566 = _shape_param_918 = None
        permute_567: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_565, [0, 1, 4, 3, 2]);  permute_565 = None
        view_824: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_567, _shape_param_919);  permute_567 = _shape_param_919 = None
        bmm_70: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_823, view_824)
        view_825: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_70, _shape_param_920);  bmm_70 = _shape_param_920 = None
        permute_568: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_825, [0, 1, 2, 4, 3]);  view_825 = None
        view_826: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_568, _shape_param_921);  permute_568 = _shape_param_921 = None
        view_827: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_826, _shape_param_922);  view_826 = _shape_param_922 = None
        slice_24: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_827, 2, 1, 9223372036854775807);  view_827 = None
        view_828: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_24, _shape_param_923);  slice_24 = _shape_param_923 = None
        index_23: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_828, [None, None, None, iota_2]);  view_828 = None
        add_255: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_822, index_23);  view_822 = index_23 = None
        add_256: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_255, 0);  add_255 = None
        mul_443: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_256, 0.125)
        convert_element_type_496: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_443, torch.float32);  mul_443 = None
        convert_element_type_497: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_256, torch.float32)
        mul_444: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_497, 1);  convert_element_type_497 = None
        amax_46: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_444, [3], True)
        sub_92: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_444, amax_46);  mul_444 = None
        mul_445: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_92, 0.125);  sub_92 = None
        amax_47: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_496, [3], True)
        sub_93: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_496, amax_47)
        abs_24: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_496)
        ne_23: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_24, inf);  abs_24 = None
        eq_23: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_496, convert_element_type_496);  convert_element_type_496 = None
        mul_446: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_23, ne_23);  eq_23 = ne_23 = None
        logical_not_46: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_446);  mul_446 = None
        any_24: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_46, [3], True);  logical_not_46 = None
        logical_not_47: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        where_23: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_47, mul_445, sub_93);  mul_445 = sub_93 = None
        exp_23: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_23);  where_23 = None
        sum_24: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [3], True)
        div_24: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = None
        inductor_lookup_seed_94: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 94)
        inductor_random_94: "f32[16, 16, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_924, inductor_lookup_seed_94, 'rand');  _shape_param_924 = inductor_lookup_seed_94 = None
        gt_94: "b8[16, 16, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_94, 0.1);  inductor_random_94 = None
        mul_447: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(gt_94, div_24);  div_24 = None
        mul_448: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(mul_447, 1.1111111111111112);  mul_447 = None
        convert_element_type_498: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_448, torch.bfloat16);  mul_448 = None
        unsqueeze_575: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_498, 4);  convert_element_type_498 = None
        unsqueeze_576: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_815, 4);  view_815 = None
        permute_569: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_576, [4, 1, 2, 3, 0]);  unsqueeze_576 = None
        view_829: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_575, _shape_param_925);  unsqueeze_575 = _shape_param_925 = None
        permute_570: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_569, [1, 2, 4, 3, 0]);  permute_569 = None
        view_830: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_570, _shape_param_926);  permute_570 = _shape_param_926 = None
        bmm_71: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_829, view_830)
        view_831: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_71, _shape_param_927);  bmm_71 = _shape_param_927 = None
        permute_571: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_831, [2, 0, 1, 4, 3]);  view_831 = None
        view_832: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_571, _shape_param_928);  permute_571 = _shape_param_928 = None
        convert_element_type_499: "bf16[1024, 16, 64]" = torch.ops.prims.convert_element_type.default(arg353_1, torch.bfloat16);  arg353_1 = None
        unsqueeze_577: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_832, 4);  view_832 = None
        permute_572: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_577, [0, 1, 4, 3, 2]);  unsqueeze_577 = None
        unsqueeze_578: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_499, 3);  convert_element_type_499 = None
        unsqueeze_579: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 4);  unsqueeze_578 = None
        permute_573: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_579, [3, 4, 0, 2, 1]);  unsqueeze_579 = None
        permute_574: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_572, [0, 1, 3, 4, 2]);  permute_572 = None
        clone_47: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_574, memory_format = torch.contiguous_format);  permute_574 = None
        view_833: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_47, _shape_param_929);  clone_47 = _shape_param_929 = None
        permute_575: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_573, [3, 4, 2, 0, 1]);  permute_573 = None
        clone_48: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
        view_834: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_48, _shape_param_930);  clone_48 = _shape_param_930 = None
        squeeze_167: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_833, 0)
        squeeze_168: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_834, 0)
        mm_119: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_167, squeeze_168);  squeeze_167 = squeeze_168 = None
        unsqueeze_580: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_119, 0);  mm_119 = None
        view_835: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_580, _shape_param_931);  unsqueeze_580 = _shape_param_931 = None
        permute_576: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_835, [0, 1, 4, 2, 3]);  view_835 = None
        view_836: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_576, _shape_param_932);  permute_576 = _shape_param_932 = None
        inductor_lookup_seed_95: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 95)
        inductor_random_95: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_933, inductor_lookup_seed_95, 'rand');  _shape_param_933 = inductor_lookup_seed_95 = None
        convert_element_type_500: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_95, torch.bfloat16);  inductor_random_95 = None
        gt_95: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_500, 0.1);  convert_element_type_500 = None
        mul_449: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_95, view_836);  view_836 = None
        mul_450: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_449, 1.1111111111111112);  mul_449 = None
        add_257: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_450, add_252);  mul_450 = add_252 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_257, [2], correction = 0, keepdim = True)
        getitem_92: "f32[512, 16, 1]" = var_mean_46[0]
        getitem_93: "f32[512, 16, 1]" = var_mean_46[1];  var_mean_46 = None
        add_258: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        sub_94: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_257, getitem_93);  add_257 = getitem_93 = None
        mul_451: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_46);  sub_94 = None
        mul_452: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_451, arg354_1);  arg354_1 = None
        add_259: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_452, arg355_1);  mul_452 = arg355_1 = None
        convert_element_type_501: "bf16[4096]" = torch.ops.prims.convert_element_type.default(arg356_1, torch.bfloat16);  arg356_1 = None
        convert_element_type_502: "bf16[4096, 1024]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.bfloat16);  arg357_1 = None
        convert_element_type_503: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16)
        view_837: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_503, _shape_param_934);  convert_element_type_503 = _shape_param_934 = None
        permute_577: "bf16[1024, 4096]" = torch.ops.aten.permute.default(convert_element_type_502, [1, 0]);  convert_element_type_502 = None
        addmm_46: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(convert_element_type_501, view_837, permute_577);  convert_element_type_501 = None
        view_838: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_46, _shape_param_935);  _shape_param_935 = None
        convert_element_type_504: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_838, torch.float32);  view_838 = None
        mul_453: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_504, 0.5)
        mul_454: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_504, 0.7071067811865476);  convert_element_type_504 = None
        erf_23: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_454);  mul_454 = None
        add_260: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_455: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_453, add_260);  mul_453 = add_260 = None
        convert_element_type_505: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_455, torch.bfloat16);  mul_455 = None
        inductor_lookup_seed_96: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 96)
        inductor_random_96: "f32[512, 16, 4096]" = torch.ops.prims.inductor_random.default(_shape_param_936, inductor_lookup_seed_96, 'rand');  _shape_param_936 = inductor_lookup_seed_96 = None
        convert_element_type_506: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(inductor_random_96, torch.bfloat16);  inductor_random_96 = None
        gt_96: "b8[512, 16, 4096]" = torch.ops.aten.gt.Scalar(convert_element_type_506, 0.1);  convert_element_type_506 = None
        mul_456: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(gt_96, convert_element_type_505);  convert_element_type_505 = None
        mul_457: "bf16[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_456, 1.1111111111111112);  mul_456 = None
        convert_element_type_507: "bf16[1024]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.bfloat16);  arg358_1 = None
        convert_element_type_508: "bf16[1024, 4096]" = torch.ops.prims.convert_element_type.default(arg359_1, torch.bfloat16);  arg359_1 = None
        view_839: "bf16[8192, 4096]" = torch.ops.aten.view.default(mul_457, _shape_param_937);  mul_457 = _shape_param_937 = None
        permute_578: "bf16[4096, 1024]" = torch.ops.aten.permute.default(convert_element_type_508, [1, 0]);  convert_element_type_508 = None
        addmm_47: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(convert_element_type_507, view_839, permute_578);  convert_element_type_507 = None
        view_840: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_47, _shape_param_938);  addmm_47 = _shape_param_938 = None
        inductor_lookup_seed_97: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 97)
        inductor_random_97: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_939, inductor_lookup_seed_97, 'rand');  _shape_param_939 = inductor_lookup_seed_97 = None
        convert_element_type_509: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(inductor_random_97, torch.bfloat16);  inductor_random_97 = None
        gt_97: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(convert_element_type_509, 0.1);  convert_element_type_509 = None
        mul_458: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_97, view_840);  view_840 = None
        mul_459: "bf16[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_458, 1.1111111111111112);  mul_458 = None
        add_261: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_459, add_259);  mul_459 = add_259 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_261, [2], correction = 0, keepdim = True)
        getitem_94: "f32[512, 16, 1]" = var_mean_47[0]
        getitem_95: "f32[512, 16, 1]" = var_mean_47[1];  var_mean_47 = None
        add_262: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
        sub_95: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(add_261, getitem_95);  add_261 = getitem_95 = None
        mul_460: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_47);  sub_95 = None
        mul_461: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_460, arg360_1);  arg360_1 = None
        add_263: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_461, arg361_1);  mul_461 = arg361_1 = None
        inductor_lookup_seed_98: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 98);  inductor_seeds = None
        inductor_random_98: "f32[512, 16, 1024]" = torch.ops.prims.inductor_random.default(_shape_param_940, inductor_lookup_seed_98, 'rand');  _shape_param_940 = inductor_lookup_seed_98 = None
        gt_98: "b8[512, 16, 1024]" = torch.ops.aten.gt.Scalar(inductor_random_98, 0.1);  inductor_random_98 = None
        mul_462: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(gt_98, add_263);  add_263 = None
        mul_463: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_462, 1.1111111111111112);  mul_462 = None
        permute_579: "f32[16, 512, 1024]" = torch.ops.aten.permute.default(mul_463, [1, 0, 2]);  mul_463 = None
        clone_49: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(permute_579, memory_format = torch.contiguous_format);  permute_579 = None
        convert_element_type_510: "bf16[32000]" = torch.ops.prims.convert_element_type.default(arg362_1, torch.bfloat16);  arg362_1 = None
        convert_element_type_511: "bf16[32000, 1024]" = torch.ops.prims.convert_element_type.default(arg1_1, torch.bfloat16);  arg1_1 = None
        convert_element_type_512: "bf16[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(clone_49, torch.bfloat16);  clone_49 = None
        view_841: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_512, _shape_param_941);  convert_element_type_512 = _shape_param_941 = None
        permute_580: "bf16[1024, 32000]" = torch.ops.aten.permute.default(convert_element_type_511, [1, 0]);  convert_element_type_511 = None
        addmm_48: "bf16[8192, 32000]" = torch.ops.aten.addmm.default(convert_element_type_510, view_841, permute_580);  convert_element_type_510 = None
        view_842: "bf16[16, 512, 32000]" = torch.ops.aten.view.default(addmm_48, _shape_param_942);  addmm_48 = _shape_param_942 = None
        view_843: "bf16[8192, 32000]" = torch.ops.aten.view.default(view_842, _shape_param_943);  _shape_param_943 = None
        view_844: "i64[8192]" = torch.ops.aten.view.default(arg363_1, [-1]);  arg363_1 = None
        convert_element_type_513: "f32[8192, 32000]" = torch.ops.prims.convert_element_type.default(view_843, torch.float32);  view_843 = None
        amax_48: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type_513, [1], True)
        sub_96: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(convert_element_type_513, amax_48);  convert_element_type_513 = None
        exp_24: "f32[8192, 32000]" = torch.ops.aten.exp.default(sub_96)
        sum_25: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_97: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(sub_96, log);  sub_96 = None
        convert_element_type_514: "bf16[8192, 32000]" = torch.ops.prims.convert_element_type.default(sub_97, torch.bfloat16);  sub_97 = None
        convert_element_type_515: "f32[8192, 32000]" = torch.ops.prims.convert_element_type.default(convert_element_type_514, torch.float32);  convert_element_type_514 = None
        ne_24: "b8[8192]" = torch.ops.aten.ne.Scalar(view_844, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[8192]" = torch.ops.aten.where.self(ne_24, view_844, full);  view_844 = full = None
        unsqueeze_581: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather: "f32[8192, 1]" = torch.ops.aten.gather.default(convert_element_type_515, 1, unsqueeze_581);  convert_element_type_515 = unsqueeze_581 = None
        squeeze_169: "f32[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192]" = torch.ops.aten.neg.default(squeeze_169);  squeeze_169 = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[8192]" = torch.ops.aten.where.self(ne_24, neg, full_1);  neg = full_1 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne_24);  ne_24 = None
        convert_element_type_516: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_25: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_516);  sum_27 = None
        permute_581: "bf16[32000, 1024]" = torch.ops.aten.permute.default(permute_580, [1, 0]);  permute_580 = None
        div_26: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None
        permute_582: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_578, [1, 0]);  permute_578 = None
        permute_583: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_577, [1, 0]);  permute_577 = None
        div_27: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None
        permute_584: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_833, [0, 2, 1]);  view_833 = None
        squeeze_170: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_584, 0);  permute_584 = None
        permute_585: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_834, [0, 2, 1]);  view_834 = None
        squeeze_171: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_585, 0);  permute_585 = None
        permute_586: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_829, [0, 2, 1]);  view_829 = None
        permute_587: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_830, [0, 2, 1]);  view_830 = None
        permute_588: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_823, [0, 2, 1]);  view_823 = None
        permute_589: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_824, [0, 2, 1]);  view_824 = None
        permute_590: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_819, [0, 2, 1]);  view_819 = None
        permute_591: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_820, [0, 2, 1]);  view_820 = None
        permute_592: "bf16[1, 1024, 16384]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
        squeeze_172: "bf16[1024, 16384]" = torch.ops.aten.squeeze.dim(permute_592, 0);  permute_592 = None
        permute_593: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_806, [0, 2, 1]);  view_806 = None
        squeeze_173: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_593, 0);  permute_593 = None
        permute_594: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_813, [0, 2, 1]);  view_813 = None
        squeeze_174: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_594, 0);  permute_594 = None
        permute_595: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_810, [0, 2, 1]);  view_810 = None
        squeeze_175: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_595, 0);  permute_595 = None
        permute_596: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_807, [0, 2, 1]);  view_807 = None
        squeeze_176: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_596, 0);  permute_596 = None
        div_28: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None
        permute_597: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_554, [1, 0]);  permute_554 = None
        permute_598: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_553, [1, 0]);  permute_553 = None
        div_29: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None
        permute_599: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_798, [0, 2, 1]);  view_798 = None
        squeeze_177: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_599, 0);  permute_599 = None
        permute_600: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_799, [0, 2, 1]);  view_799 = None
        squeeze_178: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_600, 0);  permute_600 = None
        permute_601: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_794, [0, 2, 1]);  view_794 = None
        permute_602: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_795, [0, 2, 1]);  view_795 = None
        permute_603: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_788, [0, 2, 1]);  view_788 = None
        permute_604: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_789, [0, 2, 1]);  view_789 = None
        permute_605: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_784, [0, 2, 1]);  view_784 = None
        permute_606: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_785, [0, 2, 1]);  view_785 = None
        permute_607: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_771, [0, 2, 1]);  view_771 = None
        squeeze_179: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_607, 0);  permute_607 = None
        permute_608: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_778, [0, 2, 1]);  view_778 = None
        squeeze_180: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_608, 0);  permute_608 = None
        permute_609: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_775, [0, 2, 1]);  view_775 = None
        squeeze_181: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_609, 0);  permute_609 = None
        permute_610: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_772, [0, 2, 1]);  view_772 = None
        squeeze_182: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_610, 0);  permute_610 = None
        div_30: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None
        permute_611: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_530, [1, 0]);  permute_530 = None
        permute_612: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_529, [1, 0]);  permute_529 = None
        div_31: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None
        permute_613: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_763, [0, 2, 1]);  view_763 = None
        squeeze_183: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_613, 0);  permute_613 = None
        permute_614: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_764, [0, 2, 1]);  view_764 = None
        squeeze_184: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_614, 0);  permute_614 = None
        permute_615: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_759, [0, 2, 1]);  view_759 = None
        permute_616: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_760, [0, 2, 1]);  view_760 = None
        permute_617: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_753, [0, 2, 1]);  view_753 = None
        permute_618: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_754, [0, 2, 1]);  view_754 = None
        permute_619: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_749, [0, 2, 1]);  view_749 = None
        permute_620: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_750, [0, 2, 1]);  view_750 = None
        permute_621: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_736, [0, 2, 1]);  view_736 = None
        squeeze_185: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_621, 0);  permute_621 = None
        permute_622: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_743, [0, 2, 1]);  view_743 = None
        squeeze_186: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_622, 0);  permute_622 = None
        permute_623: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_740, [0, 2, 1]);  view_740 = None
        squeeze_187: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_623, 0);  permute_623 = None
        permute_624: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_737, [0, 2, 1]);  view_737 = None
        squeeze_188: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_624, 0);  permute_624 = None
        div_32: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None
        permute_625: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
        permute_626: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_505, [1, 0]);  permute_505 = None
        div_33: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None
        permute_627: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_728, [0, 2, 1]);  view_728 = None
        squeeze_189: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_627, 0);  permute_627 = None
        permute_628: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_729, [0, 2, 1]);  view_729 = None
        squeeze_190: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_628, 0);  permute_628 = None
        permute_629: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_724, [0, 2, 1]);  view_724 = None
        permute_630: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_725, [0, 2, 1]);  view_725 = None
        permute_631: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_718, [0, 2, 1]);  view_718 = None
        permute_632: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_719, [0, 2, 1]);  view_719 = None
        permute_633: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_714, [0, 2, 1]);  view_714 = None
        permute_634: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_715, [0, 2, 1]);  view_715 = None
        permute_635: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_701, [0, 2, 1]);  view_701 = None
        squeeze_191: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_635, 0);  permute_635 = None
        permute_636: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_708, [0, 2, 1]);  view_708 = None
        squeeze_192: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_636, 0);  permute_636 = None
        permute_637: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_705, [0, 2, 1]);  view_705 = None
        squeeze_193: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_637, 0);  permute_637 = None
        permute_638: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_702, [0, 2, 1]);  view_702 = None
        squeeze_194: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_638, 0);  permute_638 = None
        div_34: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None
        permute_639: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
        permute_640: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
        div_35: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None
        permute_641: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_693, [0, 2, 1]);  view_693 = None
        squeeze_195: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_641, 0);  permute_641 = None
        permute_642: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_694, [0, 2, 1]);  view_694 = None
        squeeze_196: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_642, 0);  permute_642 = None
        permute_643: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_689, [0, 2, 1]);  view_689 = None
        permute_644: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_690, [0, 2, 1]);  view_690 = None
        permute_645: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_683, [0, 2, 1]);  view_683 = None
        permute_646: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_684, [0, 2, 1]);  view_684 = None
        permute_647: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_679, [0, 2, 1]);  view_679 = None
        permute_648: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_680, [0, 2, 1]);  view_680 = None
        permute_649: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_666, [0, 2, 1]);  view_666 = None
        squeeze_197: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_649, 0);  permute_649 = None
        permute_650: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_673, [0, 2, 1]);  view_673 = None
        squeeze_198: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_650, 0);  permute_650 = None
        permute_651: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_670, [0, 2, 1]);  view_670 = None
        squeeze_199: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_651, 0);  permute_651 = None
        permute_652: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_667, [0, 2, 1]);  view_667 = None
        squeeze_200: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_652, 0);  permute_652 = None
        div_36: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None
        permute_653: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
        permute_654: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
        div_37: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None
        permute_655: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_658, [0, 2, 1]);  view_658 = None
        squeeze_201: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_655, 0);  permute_655 = None
        permute_656: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_659, [0, 2, 1]);  view_659 = None
        squeeze_202: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_656, 0);  permute_656 = None
        permute_657: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_654, [0, 2, 1]);  view_654 = None
        permute_658: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_655, [0, 2, 1]);  view_655 = None
        permute_659: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_648, [0, 2, 1]);  view_648 = None
        permute_660: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_649, [0, 2, 1]);  view_649 = None
        permute_661: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_644, [0, 2, 1]);  view_644 = None
        permute_662: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_645, [0, 2, 1]);  view_645 = None
        permute_663: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_631, [0, 2, 1]);  view_631 = None
        squeeze_203: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_663, 0);  permute_663 = None
        permute_664: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
        squeeze_204: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_664, 0);  permute_664 = None
        permute_665: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_635, [0, 2, 1]);  view_635 = None
        squeeze_205: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_665, 0);  permute_665 = None
        permute_666: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_632, [0, 2, 1]);  view_632 = None
        squeeze_206: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_666, 0);  permute_666 = None
        div_38: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None
        permute_667: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
        permute_668: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
        div_39: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None
        permute_669: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_623, [0, 2, 1]);  view_623 = None
        squeeze_207: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_669, 0);  permute_669 = None
        permute_670: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_624, [0, 2, 1]);  view_624 = None
        squeeze_208: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_670, 0);  permute_670 = None
        permute_671: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_619, [0, 2, 1]);  view_619 = None
        permute_672: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_620, [0, 2, 1]);  view_620 = None
        permute_673: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_613, [0, 2, 1]);  view_613 = None
        permute_674: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_614, [0, 2, 1]);  view_614 = None
        permute_675: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_609, [0, 2, 1]);  view_609 = None
        permute_676: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_610, [0, 2, 1]);  view_610 = None
        permute_677: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_596, [0, 2, 1]);  view_596 = None
        squeeze_209: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_677, 0);  permute_677 = None
        permute_678: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_603, [0, 2, 1]);  view_603 = None
        squeeze_210: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_678, 0);  permute_678 = None
        permute_679: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_600, [0, 2, 1]);  view_600 = None
        squeeze_211: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_679, 0);  permute_679 = None
        permute_680: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_597, [0, 2, 1]);  view_597 = None
        squeeze_212: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_680, 0);  permute_680 = None
        div_40: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None
        permute_681: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
        permute_682: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_409, [1, 0]);  permute_409 = None
        div_41: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None
        permute_683: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_588, [0, 2, 1]);  view_588 = None
        squeeze_213: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_683, 0);  permute_683 = None
        permute_684: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_589, [0, 2, 1]);  view_589 = None
        squeeze_214: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_684, 0);  permute_684 = None
        permute_685: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_584, [0, 2, 1]);  view_584 = None
        permute_686: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_585, [0, 2, 1]);  view_585 = None
        permute_687: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_578, [0, 2, 1]);  view_578 = None
        permute_688: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_579, [0, 2, 1]);  view_579 = None
        permute_689: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_574, [0, 2, 1]);  view_574 = None
        permute_690: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_575, [0, 2, 1]);  view_575 = None
        permute_691: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_561, [0, 2, 1]);  view_561 = None
        squeeze_215: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_691, 0);  permute_691 = None
        permute_692: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_568, [0, 2, 1]);  view_568 = None
        squeeze_216: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_692, 0);  permute_692 = None
        permute_693: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_565, [0, 2, 1]);  view_565 = None
        squeeze_217: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_693, 0);  permute_693 = None
        permute_694: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_562, [0, 2, 1]);  view_562 = None
        squeeze_218: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_694, 0);  permute_694 = None
        div_42: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None
        permute_695: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_386, [1, 0]);  permute_386 = None
        permute_696: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
        div_43: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None
        permute_697: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_553, [0, 2, 1]);  view_553 = None
        squeeze_219: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_697, 0);  permute_697 = None
        permute_698: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_554, [0, 2, 1]);  view_554 = None
        squeeze_220: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_698, 0);  permute_698 = None
        permute_699: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_549, [0, 2, 1]);  view_549 = None
        permute_700: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_550, [0, 2, 1]);  view_550 = None
        permute_701: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_543, [0, 2, 1]);  view_543 = None
        permute_702: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_544, [0, 2, 1]);  view_544 = None
        permute_703: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_539, [0, 2, 1]);  view_539 = None
        permute_704: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1]);  view_540 = None
        permute_705: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_526, [0, 2, 1]);  view_526 = None
        squeeze_221: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_705, 0);  permute_705 = None
        permute_706: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_533, [0, 2, 1]);  view_533 = None
        squeeze_222: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_706, 0);  permute_706 = None
        permute_707: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_530, [0, 2, 1]);  view_530 = None
        squeeze_223: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_707, 0);  permute_707 = None
        permute_708: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_527, [0, 2, 1]);  view_527 = None
        squeeze_224: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_708, 0);  permute_708 = None
        div_44: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None
        permute_709: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
        permute_710: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
        div_45: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None
        permute_711: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
        squeeze_225: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_711, 0);  permute_711 = None
        permute_712: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_519, [0, 2, 1]);  view_519 = None
        squeeze_226: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_712, 0);  permute_712 = None
        permute_713: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_514, [0, 2, 1]);  view_514 = None
        permute_714: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
        permute_715: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_508, [0, 2, 1]);  view_508 = None
        permute_716: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_509, [0, 2, 1]);  view_509 = None
        permute_717: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_504, [0, 2, 1]);  view_504 = None
        permute_718: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_505, [0, 2, 1]);  view_505 = None
        permute_719: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_491, [0, 2, 1]);  view_491 = None
        squeeze_227: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_719, 0);  permute_719 = None
        permute_720: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_498, [0, 2, 1]);  view_498 = None
        squeeze_228: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_720, 0);  permute_720 = None
        permute_721: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
        squeeze_229: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_721, 0);  permute_721 = None
        permute_722: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_492, [0, 2, 1]);  view_492 = None
        squeeze_230: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_722, 0);  permute_722 = None
        div_46: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None
        permute_723: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
        permute_724: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
        div_47: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None
        permute_725: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_483, [0, 2, 1]);  view_483 = None
        squeeze_231: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_725, 0);  permute_725 = None
        permute_726: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_484, [0, 2, 1]);  view_484 = None
        squeeze_232: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_726, 0);  permute_726 = None
        permute_727: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_479, [0, 2, 1]);  view_479 = None
        permute_728: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_480, [0, 2, 1]);  view_480 = None
        permute_729: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None
        permute_730: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None
        permute_731: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
        permute_732: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1]);  view_470 = None
        permute_733: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None
        squeeze_233: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_733, 0);  permute_733 = None
        permute_734: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_463, [0, 2, 1]);  view_463 = None
        squeeze_234: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_734, 0);  permute_734 = None
        permute_735: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_460, [0, 2, 1]);  view_460 = None
        squeeze_235: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_735, 0);  permute_735 = None
        permute_736: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_457, [0, 2, 1]);  view_457 = None
        squeeze_236: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_736, 0);  permute_736 = None
        div_48: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None
        permute_737: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
        permute_738: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
        div_49: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None
        permute_739: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
        squeeze_237: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_739, 0);  permute_739 = None
        permute_740: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
        squeeze_238: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_740, 0);  permute_740 = None
        permute_741: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
        permute_742: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_445, [0, 2, 1]);  view_445 = None
        permute_743: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
        permute_744: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None
        permute_745: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_434, [0, 2, 1]);  view_434 = None
        permute_746: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_435, [0, 2, 1]);  view_435 = None
        permute_747: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
        squeeze_239: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_747, 0);  permute_747 = None
        permute_748: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_428, [0, 2, 1]);  view_428 = None
        squeeze_240: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_748, 0);  permute_748 = None
        permute_749: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_425, [0, 2, 1]);  view_425 = None
        squeeze_241: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_749, 0);  permute_749 = None
        permute_750: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_422, [0, 2, 1]);  view_422 = None
        squeeze_242: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_750, 0);  permute_750 = None
        div_50: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None
        permute_751: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
        permute_752: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
        div_51: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None
        permute_753: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_413, [0, 2, 1]);  view_413 = None
        squeeze_243: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_753, 0);  permute_753 = None
        permute_754: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_414, [0, 2, 1]);  view_414 = None
        squeeze_244: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_754, 0);  permute_754 = None
        permute_755: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
        permute_756: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
        permute_757: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
        permute_758: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1]);  view_404 = None
        permute_759: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_399, [0, 2, 1]);  view_399 = None
        permute_760: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1]);  view_400 = None
        permute_761: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
        squeeze_245: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_761, 0);  permute_761 = None
        permute_762: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None
        squeeze_246: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_762, 0);  permute_762 = None
        permute_763: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_390, [0, 2, 1]);  view_390 = None
        squeeze_247: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_763, 0);  permute_763 = None
        permute_764: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_387, [0, 2, 1]);  view_387 = None
        squeeze_248: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_764, 0);  permute_764 = None
        div_52: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None
        permute_765: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
        permute_766: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        div_53: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None
        permute_767: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None
        squeeze_249: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_767, 0);  permute_767 = None
        permute_768: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_379, [0, 2, 1]);  view_379 = None
        squeeze_250: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_768, 0);  permute_768 = None
        permute_769: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_374, [0, 2, 1]);  view_374 = None
        permute_770: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_375, [0, 2, 1]);  view_375 = None
        permute_771: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None
        permute_772: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_369, [0, 2, 1]);  view_369 = None
        permute_773: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        permute_774: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1]);  view_365 = None
        permute_775: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
        squeeze_251: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_775, 0);  permute_775 = None
        permute_776: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_358, [0, 2, 1]);  view_358 = None
        squeeze_252: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_776, 0);  permute_776 = None
        permute_777: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_355, [0, 2, 1]);  view_355 = None
        squeeze_253: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_777, 0);  permute_777 = None
        permute_778: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_352, [0, 2, 1]);  view_352 = None
        squeeze_254: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_778, 0);  permute_778 = None
        div_54: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None
        permute_779: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        permute_780: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        div_55: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None
        permute_781: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
        squeeze_255: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_781, 0);  permute_781 = None
        permute_782: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_344, [0, 2, 1]);  view_344 = None
        squeeze_256: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_782, 0);  permute_782 = None
        permute_783: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        permute_784: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None
        permute_785: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_333, [0, 2, 1]);  view_333 = None
        permute_786: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
        permute_787: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_329, [0, 2, 1]);  view_329 = None
        permute_788: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1]);  view_330 = None
        permute_789: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_316, [0, 2, 1]);  view_316 = None
        squeeze_257: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_789, 0);  permute_789 = None
        permute_790: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
        squeeze_258: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_790, 0);  permute_790 = None
        permute_791: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
        squeeze_259: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_791, 0);  permute_791 = None
        permute_792: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        squeeze_260: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_792, 0);  permute_792 = None
        div_56: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None
        permute_793: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        permute_794: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        div_57: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None
        permute_795: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
        squeeze_261: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_795, 0);  permute_795 = None
        permute_796: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
        squeeze_262: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_796, 0);  permute_796 = None
        permute_797: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
        permute_798: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
        permute_799: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
        permute_800: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_299, [0, 2, 1]);  view_299 = None
        permute_801: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_294, [0, 2, 1]);  view_294 = None
        permute_802: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
        permute_803: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_281, [0, 2, 1]);  view_281 = None
        squeeze_263: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_803, 0);  permute_803 = None
        permute_804: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_288, [0, 2, 1]);  view_288 = None
        squeeze_264: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_804, 0);  permute_804 = None
        permute_805: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
        squeeze_265: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_805, 0);  permute_805 = None
        permute_806: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_282, [0, 2, 1]);  view_282 = None
        squeeze_266: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_806, 0);  permute_806 = None
        div_58: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None
        permute_807: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        permute_808: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        div_59: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None
        permute_809: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
        squeeze_267: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_809, 0);  permute_809 = None
        permute_810: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
        squeeze_268: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_810, 0);  permute_810 = None
        permute_811: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
        permute_812: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
        permute_813: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
        permute_814: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1]);  view_264 = None
        permute_815: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
        permute_816: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_260, [0, 2, 1]);  view_260 = None
        permute_817: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_246, [0, 2, 1]);  view_246 = None
        squeeze_269: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_817, 0);  permute_817 = None
        permute_818: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_253, [0, 2, 1]);  view_253 = None
        squeeze_270: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_818, 0);  permute_818 = None
        permute_819: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_250, [0, 2, 1]);  view_250 = None
        squeeze_271: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_819, 0);  permute_819 = None
        permute_820: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_247, [0, 2, 1]);  view_247 = None
        squeeze_272: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_820, 0);  permute_820 = None
        div_60: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None
        permute_821: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        permute_822: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        div_61: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None
        permute_823: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
        squeeze_273: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_823, 0);  permute_823 = None
        permute_824: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
        squeeze_274: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_824, 0);  permute_824 = None
        permute_825: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
        permute_826: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_827: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
        permute_828: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_829: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
        permute_830: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
        permute_831: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
        squeeze_275: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_831, 0);  permute_831 = None
        permute_832: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
        squeeze_276: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_832, 0);  permute_832 = None
        permute_833: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_215, [0, 2, 1]);  view_215 = None
        squeeze_277: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_833, 0);  permute_833 = None
        permute_834: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
        squeeze_278: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_834, 0);  permute_834 = None
        div_62: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None
        permute_835: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
        permute_836: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        div_63: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None
        permute_837: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
        squeeze_279: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_837, 0);  permute_837 = None
        permute_838: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
        squeeze_280: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_838, 0);  permute_838 = None
        permute_839: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
        permute_840: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
        permute_841: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
        permute_842: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_194, [0, 2, 1]);  view_194 = None
        permute_843: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        permute_844: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        permute_845: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_176, [0, 2, 1]);  view_176 = None
        squeeze_281: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_845, 0);  permute_845 = None
        permute_846: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        squeeze_282: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_846, 0);  permute_846 = None
        permute_847: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_180, [0, 2, 1]);  view_180 = None
        squeeze_283: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_847, 0);  permute_847 = None
        permute_848: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_177, [0, 2, 1]);  view_177 = None
        squeeze_284: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_848, 0);  permute_848 = None
        div_64: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None
        permute_849: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        permute_850: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        div_65: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None
        permute_851: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        squeeze_285: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_851, 0);  permute_851 = None
        permute_852: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        squeeze_286: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_852, 0);  permute_852 = None
        permute_853: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
        permute_854: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_165, [0, 2, 1]);  view_165 = None
        permute_855: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_158, [0, 2, 1]);  view_158 = None
        permute_856: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1]);  view_159 = None
        permute_857: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None
        permute_858: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1]);  view_155 = None
        permute_859: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        squeeze_287: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_859, 0);  permute_859 = None
        permute_860: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_148, [0, 2, 1]);  view_148 = None
        squeeze_288: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_860, 0);  permute_860 = None
        permute_861: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        squeeze_289: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_861, 0);  permute_861 = None
        permute_862: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
        squeeze_290: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_862, 0);  permute_862 = None
        div_66: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None
        permute_863: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        permute_864: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        div_67: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None
        permute_865: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
        squeeze_291: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_865, 0);  permute_865 = None
        permute_866: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
        squeeze_292: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_866, 0);  permute_866 = None
        permute_867: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
        permute_868: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_869: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_870: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        permute_871: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        permute_872: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
        permute_873: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
        squeeze_293: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_873, 0);  permute_873 = None
        permute_874: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_113, [0, 2, 1]);  view_113 = None
        squeeze_294: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_874, 0);  permute_874 = None
        permute_875: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_110, [0, 2, 1]);  view_110 = None
        squeeze_295: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_875, 0);  permute_875 = None
        permute_876: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        squeeze_296: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_876, 0);  permute_876 = None
        div_68: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None
        permute_877: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_878: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        div_69: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None
        permute_879: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
        squeeze_297: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_879, 0);  permute_879 = None
        permute_880: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
        squeeze_298: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_880, 0);  permute_880 = None
        permute_881: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
        permute_882: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_95, [0, 2, 1]);  view_95 = None
        permute_883: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
        permute_884: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
        permute_885: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_84, [0, 2, 1]);  view_84 = None
        permute_886: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        permute_887: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_71, [0, 2, 1]);  view_71 = None
        squeeze_299: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_887, 0);  permute_887 = None
        permute_888: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        squeeze_300: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_888, 0);  permute_888 = None
        permute_889: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        squeeze_301: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_889, 0);  permute_889 = None
        permute_890: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
        squeeze_302: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_890, 0);  permute_890 = None
        div_70: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None
        permute_891: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        permute_892: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        div_71: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None
        permute_893: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
        squeeze_303: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_893, 0);  permute_893 = None
        permute_894: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        squeeze_304: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_894, 0);  permute_894 = None
        permute_895: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
        permute_896: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
        permute_897: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_898: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
        permute_899: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
        permute_900: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        permute_901: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
        squeeze_305: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_901, 0);  permute_901 = None
        permute_902: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        squeeze_306: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_902, 0);  permute_902 = None
        permute_903: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_40, [0, 2, 1]);  view_40 = None
        squeeze_307: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_903, 0);  permute_903 = None
        permute_904: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
        squeeze_308: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_904, 0);  permute_904 = None
        div_72: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None
        permute_905: "bf16[1024, 4096]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        permute_906: "bf16[4096, 1024]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        div_73: "f32[512, 16, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        permute_907: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
        squeeze_309: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_907, 0);  permute_907 = None
        permute_908: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
        squeeze_310: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_908, 0);  permute_908 = None
        permute_909: "bf16[256, 512, 512]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
        permute_910: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
        permute_911: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_18, [0, 2, 1]);  view_18 = None
        permute_912: "bf16[256, 1024, 64]" = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None
        permute_913: "bf16[256, 64, 512]" = torch.ops.aten.permute.default(view_14, [0, 2, 1]);  view_14 = None
        permute_914: "bf16[256, 512, 64]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_915: "bf16[1, 1024, 8192]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        squeeze_311: "bf16[1024, 8192]" = torch.ops.aten.squeeze.dim(permute_915, 0);  permute_915 = None
        permute_916: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
        squeeze_312: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_916, 0);  permute_916 = None
        permute_917: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_4, [0, 2, 1]);  view_4 = None
        squeeze_313: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_917, 0);  permute_917 = None
        permute_918: "bf16[1, 1024, 1024]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        squeeze_314: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(permute_918, 0);  permute_918 = None
        return (clone, gt, iota_2, add_3, amax, amax_1, logical_not_1, sum_1, gt_2, gt_3, mul_14, view_32, addmm, gt_4, view_34, gt_5, mul_23, add_14, amax_2, amax_3, logical_not_3, sum_2, gt_6, gt_7, mul_33, view_67, addmm_2, gt_8, view_69, gt_9, mul_42, add_25, amax_4, amax_5, logical_not_5, sum_3, gt_10, gt_11, mul_52, view_102, addmm_4, gt_12, view_104, gt_13, mul_61, add_36, amax_6, amax_7, logical_not_7, sum_4, gt_14, gt_15, mul_71, view_137, addmm_6, gt_16, view_139, gt_17, mul_80, add_47, amax_8, amax_9, logical_not_9, sum_5, gt_18, gt_19, mul_90, view_172, addmm_8, gt_20, view_174, gt_21, mul_99, add_58, amax_10, amax_11, logical_not_11, sum_6, gt_22, gt_23, mul_109, view_207, addmm_10, gt_24, view_209, gt_25, mul_118, add_69, amax_12, amax_13, logical_not_13, sum_7, gt_26, gt_27, mul_128, view_242, addmm_12, gt_28, view_244, gt_29, mul_137, add_80, amax_14, amax_15, logical_not_15, sum_8, gt_30, gt_31, mul_147, view_277, addmm_14, gt_32, view_279, gt_33, mul_156, add_91, amax_16, amax_17, logical_not_17, sum_9, gt_34, gt_35, mul_166, view_312, addmm_16, gt_36, view_314, gt_37, mul_175, add_102, amax_18, amax_19, logical_not_19, sum_10, gt_38, gt_39, mul_185, view_347, addmm_18, gt_40, view_349, gt_41, mul_194, add_113, amax_20, amax_21, logical_not_21, sum_11, gt_42, gt_43, mul_204, view_382, addmm_20, gt_44, view_384, gt_45, mul_213, add_124, amax_22, amax_23, logical_not_23, sum_12, gt_46, gt_47, mul_223, view_417, addmm_22, gt_48, view_419, gt_49, mul_232, add_135, amax_24, amax_25, logical_not_25, sum_13, gt_50, gt_51, mul_242, view_452, addmm_24, gt_52, view_454, gt_53, mul_251, add_146, amax_26, amax_27, logical_not_27, sum_14, gt_54, gt_55, mul_261, view_487, addmm_26, gt_56, view_489, gt_57, mul_270, add_157, amax_28, amax_29, logical_not_29, sum_15, gt_58, gt_59, mul_280, view_522, addmm_28, gt_60, view_524, gt_61, mul_289, add_168, amax_30, amax_31, logical_not_31, sum_16, gt_62, gt_63, mul_299, view_557, addmm_30, gt_64, view_559, gt_65, mul_308, add_179, amax_32, amax_33, logical_not_33, sum_17, gt_66, gt_67, mul_318, view_592, addmm_32, gt_68, view_594, gt_69, mul_327, add_190, amax_34, amax_35, logical_not_35, sum_18, gt_70, gt_71, mul_337, view_627, addmm_34, gt_72, view_629, gt_73, mul_346, add_201, amax_36, amax_37, logical_not_37, sum_19, gt_74, gt_75, mul_356, view_662, addmm_36, gt_76, view_664, gt_77, mul_365, add_212, amax_38, amax_39, logical_not_39, sum_20, gt_78, gt_79, mul_375, view_697, addmm_38, gt_80, view_699, gt_81, mul_384, add_223, amax_40, amax_41, logical_not_41, sum_21, gt_82, gt_83, mul_394, view_732, addmm_40, gt_84, view_734, gt_85, mul_403, add_234, amax_42, amax_43, logical_not_43, sum_22, gt_86, gt_87, mul_413, view_767, addmm_42, gt_88, view_769, gt_89, mul_422, add_245, amax_44, amax_45, logical_not_45, sum_23, gt_90, gt_91, mul_432, view_802, addmm_44, gt_92, view_804, gt_93, mul_441, add_256, amax_46, amax_47, logical_not_47, sum_24, gt_94, gt_95, mul_451, view_837, addmm_46, gt_96, view_839, gt_97, mul_460, gt_98, view_841, view_842, amax_48, log, convert_element_type_516, div_25, permute_581, div_26, permute_582, permute_583, div_27, squeeze_170, squeeze_171, permute_586, permute_587, permute_588, permute_589, permute_590, permute_591, squeeze_172, squeeze_173, squeeze_174, squeeze_175, squeeze_176, div_28, permute_597, permute_598, div_29, squeeze_177, squeeze_178, permute_601, permute_602, permute_603, permute_604, permute_605, permute_606, squeeze_179, squeeze_180, squeeze_181, squeeze_182, div_30, permute_611, permute_612, div_31, squeeze_183, squeeze_184, permute_615, permute_616, permute_617, permute_618, permute_619, permute_620, squeeze_185, squeeze_186, squeeze_187, squeeze_188, div_32, permute_625, permute_626, div_33, squeeze_189, squeeze_190, permute_629, permute_630, permute_631, permute_632, permute_633, permute_634, squeeze_191, squeeze_192, squeeze_193, squeeze_194, div_34, permute_639, permute_640, div_35, squeeze_195, squeeze_196, permute_643, permute_644, permute_645, permute_646, permute_647, permute_648, squeeze_197, squeeze_198, squeeze_199, squeeze_200, div_36, permute_653, permute_654, div_37, squeeze_201, squeeze_202, permute_657, permute_658, permute_659, permute_660, permute_661, permute_662, squeeze_203, squeeze_204, squeeze_205, squeeze_206, div_38, permute_667, permute_668, div_39, squeeze_207, squeeze_208, permute_671, permute_672, permute_673, permute_674, permute_675, permute_676, squeeze_209, squeeze_210, squeeze_211, squeeze_212, div_40, permute_681, permute_682, div_41, squeeze_213, squeeze_214, permute_685, permute_686, permute_687, permute_688, permute_689, permute_690, squeeze_215, squeeze_216, squeeze_217, squeeze_218, div_42, permute_695, permute_696, div_43, squeeze_219, squeeze_220, permute_699, permute_700, permute_701, permute_702, permute_703, permute_704, squeeze_221, squeeze_222, squeeze_223, squeeze_224, div_44, permute_709, permute_710, div_45, squeeze_225, squeeze_226, permute_713, permute_714, permute_715, permute_716, permute_717, permute_718, squeeze_227, squeeze_228, squeeze_229, squeeze_230, div_46, permute_723, permute_724, div_47, squeeze_231, squeeze_232, permute_727, permute_728, permute_729, permute_730, permute_731, permute_732, squeeze_233, squeeze_234, squeeze_235, squeeze_236, div_48, permute_737, permute_738, div_49, squeeze_237, squeeze_238, permute_741, permute_742, permute_743, permute_744, permute_745, permute_746, squeeze_239, squeeze_240, squeeze_241, squeeze_242, div_50, permute_751, permute_752, div_51, squeeze_243, squeeze_244, permute_755, permute_756, permute_757, permute_758, permute_759, permute_760, squeeze_245, squeeze_246, squeeze_247, squeeze_248, div_52, permute_765, permute_766, div_53, squeeze_249, squeeze_250, permute_769, permute_770, permute_771, permute_772, permute_773, permute_774, squeeze_251, squeeze_252, squeeze_253, squeeze_254, div_54, permute_779, permute_780, div_55, squeeze_255, squeeze_256, permute_783, permute_784, permute_785, permute_786, permute_787, permute_788, squeeze_257, squeeze_258, squeeze_259, squeeze_260, div_56, permute_793, permute_794, div_57, squeeze_261, squeeze_262, permute_797, permute_798, permute_799, permute_800, permute_801, permute_802, squeeze_263, squeeze_264, squeeze_265, squeeze_266, div_58, permute_807, permute_808, div_59, squeeze_267, squeeze_268, permute_811, permute_812, permute_813, permute_814, permute_815, permute_816, squeeze_269, squeeze_270, squeeze_271, squeeze_272, div_60, permute_821, permute_822, div_61, squeeze_273, squeeze_274, permute_825, permute_826, permute_827, permute_828, permute_829, permute_830, squeeze_275, squeeze_276, squeeze_277, squeeze_278, div_62, permute_835, permute_836, div_63, squeeze_279, squeeze_280, permute_839, permute_840, permute_841, permute_842, permute_843, permute_844, squeeze_281, squeeze_282, squeeze_283, squeeze_284, div_64, permute_849, permute_850, div_65, squeeze_285, squeeze_286, permute_853, permute_854, permute_855, permute_856, permute_857, permute_858, squeeze_287, squeeze_288, squeeze_289, squeeze_290, div_66, permute_863, permute_864, div_67, squeeze_291, squeeze_292, permute_867, permute_868, permute_869, permute_870, permute_871, permute_872, squeeze_293, squeeze_294, squeeze_295, squeeze_296, div_68, permute_877, permute_878, div_69, squeeze_297, squeeze_298, permute_881, permute_882, permute_883, permute_884, permute_885, permute_886, squeeze_299, squeeze_300, squeeze_301, squeeze_302, div_70, permute_891, permute_892, div_71, squeeze_303, squeeze_304, permute_895, permute_896, permute_897, permute_898, permute_899, permute_900, squeeze_305, squeeze_306, squeeze_307, squeeze_308, div_72, permute_905, permute_906, div_73, squeeze_309, squeeze_310, permute_909, permute_910, permute_911, permute_912, permute_913, permute_914, squeeze_311, squeeze_312, squeeze_313, squeeze_314)



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
