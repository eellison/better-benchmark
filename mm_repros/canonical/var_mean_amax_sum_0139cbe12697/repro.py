"""
Standalone repro captured via capture_hook.
Label: XGLMForCausalLM
Pattern hash: 0139cbe12697
Shape hash: 35e193f9
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
    def forward(self, arg0_1: "bf16[256008, 1024]", arg1_1: "i64[32, 128]", arg2_1: "bf16[2050, 1024]", arg3_1: "bf16[1024]", arg4_1: "bf16[1024]", arg5_1: "bf16[1024, 1024]", arg6_1: "bf16[1024]", arg7_1: "bf16[1024, 1024]", arg8_1: "bf16[1024]", arg9_1: "bf16[1024, 1024]", arg10_1: "bf16[1024]", arg11_1: "bf16[1024, 1024]", arg12_1: "bf16[1024]", arg13_1: "bf16[1024]", arg14_1: "bf16[1024]", arg15_1: "bf16[4096, 1024]", arg16_1: "bf16[4096]", arg17_1: "bf16[1024, 4096]", arg18_1: "bf16[1024]", arg19_1: "bf16[1024]", arg20_1: "bf16[1024]", arg21_1: "bf16[1024, 1024]", arg22_1: "bf16[1024]", arg23_1: "bf16[1024, 1024]", arg24_1: "bf16[1024]", arg25_1: "bf16[1024, 1024]", arg26_1: "bf16[1024]", arg27_1: "bf16[1024, 1024]", arg28_1: "bf16[1024]", arg29_1: "bf16[1024]", arg30_1: "bf16[1024]", arg31_1: "bf16[4096, 1024]", arg32_1: "bf16[4096]", arg33_1: "bf16[1024, 4096]", arg34_1: "bf16[1024]", arg35_1: "bf16[1024]", arg36_1: "bf16[1024]", arg37_1: "bf16[1024, 1024]", arg38_1: "bf16[1024]", arg39_1: "bf16[1024, 1024]", arg40_1: "bf16[1024]", arg41_1: "bf16[1024, 1024]", arg42_1: "bf16[1024]", arg43_1: "bf16[1024, 1024]", arg44_1: "bf16[1024]", arg45_1: "bf16[1024]", arg46_1: "bf16[1024]", arg47_1: "bf16[4096, 1024]", arg48_1: "bf16[4096]", arg49_1: "bf16[1024, 4096]", arg50_1: "bf16[1024]", arg51_1: "bf16[1024]", arg52_1: "bf16[1024]", arg53_1: "bf16[1024, 1024]", arg54_1: "bf16[1024]", arg55_1: "bf16[1024, 1024]", arg56_1: "bf16[1024]", arg57_1: "bf16[1024, 1024]", arg58_1: "bf16[1024]", arg59_1: "bf16[1024, 1024]", arg60_1: "bf16[1024]", arg61_1: "bf16[1024]", arg62_1: "bf16[1024]", arg63_1: "bf16[4096, 1024]", arg64_1: "bf16[4096]", arg65_1: "bf16[1024, 4096]", arg66_1: "bf16[1024]", arg67_1: "bf16[1024]", arg68_1: "bf16[1024]", arg69_1: "bf16[1024, 1024]", arg70_1: "bf16[1024]", arg71_1: "bf16[1024, 1024]", arg72_1: "bf16[1024]", arg73_1: "bf16[1024, 1024]", arg74_1: "bf16[1024]", arg75_1: "bf16[1024, 1024]", arg76_1: "bf16[1024]", arg77_1: "bf16[1024]", arg78_1: "bf16[1024]", arg79_1: "bf16[4096, 1024]", arg80_1: "bf16[4096]", arg81_1: "bf16[1024, 4096]", arg82_1: "bf16[1024]", arg83_1: "bf16[1024]", arg84_1: "bf16[1024]", arg85_1: "bf16[1024, 1024]", arg86_1: "bf16[1024]", arg87_1: "bf16[1024, 1024]", arg88_1: "bf16[1024]", arg89_1: "bf16[1024, 1024]", arg90_1: "bf16[1024]", arg91_1: "bf16[1024, 1024]", arg92_1: "bf16[1024]", arg93_1: "bf16[1024]", arg94_1: "bf16[1024]", arg95_1: "bf16[4096, 1024]", arg96_1: "bf16[4096]", arg97_1: "bf16[1024, 4096]", arg98_1: "bf16[1024]", arg99_1: "bf16[1024]", arg100_1: "bf16[1024]", arg101_1: "bf16[1024, 1024]", arg102_1: "bf16[1024]", arg103_1: "bf16[1024, 1024]", arg104_1: "bf16[1024]", arg105_1: "bf16[1024, 1024]", arg106_1: "bf16[1024]", arg107_1: "bf16[1024, 1024]", arg108_1: "bf16[1024]", arg109_1: "bf16[1024]", arg110_1: "bf16[1024]", arg111_1: "bf16[4096, 1024]", arg112_1: "bf16[4096]", arg113_1: "bf16[1024, 4096]", arg114_1: "bf16[1024]", arg115_1: "bf16[1024]", arg116_1: "bf16[1024]", arg117_1: "bf16[1024, 1024]", arg118_1: "bf16[1024]", arg119_1: "bf16[1024, 1024]", arg120_1: "bf16[1024]", arg121_1: "bf16[1024, 1024]", arg122_1: "bf16[1024]", arg123_1: "bf16[1024, 1024]", arg124_1: "bf16[1024]", arg125_1: "bf16[1024]", arg126_1: "bf16[1024]", arg127_1: "bf16[4096, 1024]", arg128_1: "bf16[4096]", arg129_1: "bf16[1024, 4096]", arg130_1: "bf16[1024]", arg131_1: "bf16[1024]", arg132_1: "bf16[1024]", arg133_1: "bf16[1024, 1024]", arg134_1: "bf16[1024]", arg135_1: "bf16[1024, 1024]", arg136_1: "bf16[1024]", arg137_1: "bf16[1024, 1024]", arg138_1: "bf16[1024]", arg139_1: "bf16[1024, 1024]", arg140_1: "bf16[1024]", arg141_1: "bf16[1024]", arg142_1: "bf16[1024]", arg143_1: "bf16[4096, 1024]", arg144_1: "bf16[4096]", arg145_1: "bf16[1024, 4096]", arg146_1: "bf16[1024]", arg147_1: "bf16[1024]", arg148_1: "bf16[1024]", arg149_1: "bf16[1024, 1024]", arg150_1: "bf16[1024]", arg151_1: "bf16[1024, 1024]", arg152_1: "bf16[1024]", arg153_1: "bf16[1024, 1024]", arg154_1: "bf16[1024]", arg155_1: "bf16[1024, 1024]", arg156_1: "bf16[1024]", arg157_1: "bf16[1024]", arg158_1: "bf16[1024]", arg159_1: "bf16[4096, 1024]", arg160_1: "bf16[4096]", arg161_1: "bf16[1024, 4096]", arg162_1: "bf16[1024]", arg163_1: "bf16[1024]", arg164_1: "bf16[1024]", arg165_1: "bf16[1024, 1024]", arg166_1: "bf16[1024]", arg167_1: "bf16[1024, 1024]", arg168_1: "bf16[1024]", arg169_1: "bf16[1024, 1024]", arg170_1: "bf16[1024]", arg171_1: "bf16[1024, 1024]", arg172_1: "bf16[1024]", arg173_1: "bf16[1024]", arg174_1: "bf16[1024]", arg175_1: "bf16[4096, 1024]", arg176_1: "bf16[4096]", arg177_1: "bf16[1024, 4096]", arg178_1: "bf16[1024]", arg179_1: "bf16[1024]", arg180_1: "bf16[1024]", arg181_1: "bf16[1024, 1024]", arg182_1: "bf16[1024]", arg183_1: "bf16[1024, 1024]", arg184_1: "bf16[1024]", arg185_1: "bf16[1024, 1024]", arg186_1: "bf16[1024]", arg187_1: "bf16[1024, 1024]", arg188_1: "bf16[1024]", arg189_1: "bf16[1024]", arg190_1: "bf16[1024]", arg191_1: "bf16[4096, 1024]", arg192_1: "bf16[4096]", arg193_1: "bf16[1024, 4096]", arg194_1: "bf16[1024]", arg195_1: "bf16[1024]", arg196_1: "bf16[1024]", arg197_1: "bf16[1024, 1024]", arg198_1: "bf16[1024]", arg199_1: "bf16[1024, 1024]", arg200_1: "bf16[1024]", arg201_1: "bf16[1024, 1024]", arg202_1: "bf16[1024]", arg203_1: "bf16[1024, 1024]", arg204_1: "bf16[1024]", arg205_1: "bf16[1024]", arg206_1: "bf16[1024]", arg207_1: "bf16[4096, 1024]", arg208_1: "bf16[4096]", arg209_1: "bf16[1024, 4096]", arg210_1: "bf16[1024]", arg211_1: "bf16[1024]", arg212_1: "bf16[1024]", arg213_1: "bf16[1024, 1024]", arg214_1: "bf16[1024]", arg215_1: "bf16[1024, 1024]", arg216_1: "bf16[1024]", arg217_1: "bf16[1024, 1024]", arg218_1: "bf16[1024]", arg219_1: "bf16[1024, 1024]", arg220_1: "bf16[1024]", arg221_1: "bf16[1024]", arg222_1: "bf16[1024]", arg223_1: "bf16[4096, 1024]", arg224_1: "bf16[4096]", arg225_1: "bf16[1024, 4096]", arg226_1: "bf16[1024]", arg227_1: "bf16[1024]", arg228_1: "bf16[1024]", arg229_1: "bf16[1024, 1024]", arg230_1: "bf16[1024]", arg231_1: "bf16[1024, 1024]", arg232_1: "bf16[1024]", arg233_1: "bf16[1024, 1024]", arg234_1: "bf16[1024]", arg235_1: "bf16[1024, 1024]", arg236_1: "bf16[1024]", arg237_1: "bf16[1024]", arg238_1: "bf16[1024]", arg239_1: "bf16[4096, 1024]", arg240_1: "bf16[4096]", arg241_1: "bf16[1024, 4096]", arg242_1: "bf16[1024]", arg243_1: "bf16[1024]", arg244_1: "bf16[1024]", arg245_1: "bf16[1024, 1024]", arg246_1: "bf16[1024]", arg247_1: "bf16[1024, 1024]", arg248_1: "bf16[1024]", arg249_1: "bf16[1024, 1024]", arg250_1: "bf16[1024]", arg251_1: "bf16[1024, 1024]", arg252_1: "bf16[1024]", arg253_1: "bf16[1024]", arg254_1: "bf16[1024]", arg255_1: "bf16[4096, 1024]", arg256_1: "bf16[4096]", arg257_1: "bf16[1024, 4096]", arg258_1: "bf16[1024]", arg259_1: "bf16[1024]", arg260_1: "bf16[1024]", arg261_1: "bf16[1024, 1024]", arg262_1: "bf16[1024]", arg263_1: "bf16[1024, 1024]", arg264_1: "bf16[1024]", arg265_1: "bf16[1024, 1024]", arg266_1: "bf16[1024]", arg267_1: "bf16[1024, 1024]", arg268_1: "bf16[1024]", arg269_1: "bf16[1024]", arg270_1: "bf16[1024]", arg271_1: "bf16[4096, 1024]", arg272_1: "bf16[4096]", arg273_1: "bf16[1024, 4096]", arg274_1: "bf16[1024]", arg275_1: "bf16[1024]", arg276_1: "bf16[1024]", arg277_1: "bf16[1024, 1024]", arg278_1: "bf16[1024]", arg279_1: "bf16[1024, 1024]", arg280_1: "bf16[1024]", arg281_1: "bf16[1024, 1024]", arg282_1: "bf16[1024]", arg283_1: "bf16[1024, 1024]", arg284_1: "bf16[1024]", arg285_1: "bf16[1024]", arg286_1: "bf16[1024]", arg287_1: "bf16[4096, 1024]", arg288_1: "bf16[4096]", arg289_1: "bf16[1024, 4096]", arg290_1: "bf16[1024]", arg291_1: "bf16[1024]", arg292_1: "bf16[1024]", arg293_1: "bf16[1024, 1024]", arg294_1: "bf16[1024]", arg295_1: "bf16[1024, 1024]", arg296_1: "bf16[1024]", arg297_1: "bf16[1024, 1024]", arg298_1: "bf16[1024]", arg299_1: "bf16[1024, 1024]", arg300_1: "bf16[1024]", arg301_1: "bf16[1024]", arg302_1: "bf16[1024]", arg303_1: "bf16[4096, 1024]", arg304_1: "bf16[4096]", arg305_1: "bf16[1024, 4096]", arg306_1: "bf16[1024]", arg307_1: "bf16[1024]", arg308_1: "bf16[1024]", arg309_1: "bf16[1024, 1024]", arg310_1: "bf16[1024]", arg311_1: "bf16[1024, 1024]", arg312_1: "bf16[1024]", arg313_1: "bf16[1024, 1024]", arg314_1: "bf16[1024]", arg315_1: "bf16[1024, 1024]", arg316_1: "bf16[1024]", arg317_1: "bf16[1024]", arg318_1: "bf16[1024]", arg319_1: "bf16[4096, 1024]", arg320_1: "bf16[4096]", arg321_1: "bf16[1024, 4096]", arg322_1: "bf16[1024]", arg323_1: "bf16[1024]", arg324_1: "bf16[1024]", arg325_1: "bf16[1024, 1024]", arg326_1: "bf16[1024]", arg327_1: "bf16[1024, 1024]", arg328_1: "bf16[1024]", arg329_1: "bf16[1024, 1024]", arg330_1: "bf16[1024]", arg331_1: "bf16[1024, 1024]", arg332_1: "bf16[1024]", arg333_1: "bf16[1024]", arg334_1: "bf16[1024]", arg335_1: "bf16[4096, 1024]", arg336_1: "bf16[4096]", arg337_1: "bf16[1024, 4096]", arg338_1: "bf16[1024]", arg339_1: "bf16[1024]", arg340_1: "bf16[1024]", arg341_1: "bf16[1024, 1024]", arg342_1: "bf16[1024]", arg343_1: "bf16[1024, 1024]", arg344_1: "bf16[1024]", arg345_1: "bf16[1024, 1024]", arg346_1: "bf16[1024]", arg347_1: "bf16[1024, 1024]", arg348_1: "bf16[1024]", arg349_1: "bf16[1024]", arg350_1: "bf16[1024]", arg351_1: "bf16[4096, 1024]", arg352_1: "bf16[4096]", arg353_1: "bf16[1024, 4096]", arg354_1: "bf16[1024]", arg355_1: "bf16[1024]", arg356_1: "bf16[1024]", arg357_1: "bf16[1024, 1024]", arg358_1: "bf16[1024]", arg359_1: "bf16[1024, 1024]", arg360_1: "bf16[1024]", arg361_1: "bf16[1024, 1024]", arg362_1: "bf16[1024]", arg363_1: "bf16[1024, 1024]", arg364_1: "bf16[1024]", arg365_1: "bf16[1024]", arg366_1: "bf16[1024]", arg367_1: "bf16[4096, 1024]", arg368_1: "bf16[4096]", arg369_1: "bf16[1024, 4096]", arg370_1: "bf16[1024]", arg371_1: "bf16[1024]", arg372_1: "bf16[1024]", arg373_1: "bf16[1024, 1024]", arg374_1: "bf16[1024]", arg375_1: "bf16[1024, 1024]", arg376_1: "bf16[1024]", arg377_1: "bf16[1024, 1024]", arg378_1: "bf16[1024]", arg379_1: "bf16[1024, 1024]", arg380_1: "bf16[1024]", arg381_1: "bf16[1024]", arg382_1: "bf16[1024]", arg383_1: "bf16[4096, 1024]", arg384_1: "bf16[4096]", arg385_1: "bf16[1024, 4096]", arg386_1: "bf16[1024]", arg387_1: "i64[32, 128]", arg388_1: "bf16[1024]", arg389_1: "bf16[1024]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532):
        # No stacktrace found for following nodes
        embedding: "bf16[32, 128, 1024]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 1);  arg1_1 = None
        mul: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
        add: "i64[1, 128]" = torch.ops.aten.add.Tensor(unsqueeze, 2);  unsqueeze = None
        view: "i64[128]" = torch.ops.aten.view.default(add, [-1]);  add = None
        index: "bf16[128, 1024]" = torch.ops.aten.index.Tensor(arg2_1, [view]);  arg2_1 = view = None
        view_1: "bf16[1, 128, 1024]" = torch.ops.aten.view.default(index, _shape_param_0);  index = _shape_param_0 = None
        add_1: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul, view_1);  mul = view_1 = None
        convert_element_type: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 128, 1]" = var_mean[0]
        getitem_1: "f32[32, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_2: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_1: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
        add_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
        convert_element_type_1: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view_2: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_1);  _shape_param_1 = None
        permute: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        addmm: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg6_1, view_2, permute);  arg6_1 = view_2 = permute = None
        view_3: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        mul_3: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
        view_4: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_3, _shape_param_3);  mul_3 = _shape_param_3 = None
        permute_1: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None
        clone: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_5: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        view_6: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_5);  _shape_param_5 = None
        permute_2: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm_1: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg8_1, view_6, permute_2);  arg8_1 = view_6 = permute_2 = None
        view_7: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_1, _shape_param_6);  addmm_1 = _shape_param_6 = None
        view_8: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_7, _shape_param_7);  view_7 = _shape_param_7 = None
        permute_3: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_1: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_9: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_8);  clone_1 = _shape_param_8 = None
        permute_4: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_5, permute_4);  view_5 = permute_4 = None
        view_10: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_9);  bmm = _shape_param_9 = None
        iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_4: "i64[128]" = torch.ops.aten.add.Tensor(iota_1, 0);  iota_1 = None
        unsqueeze_1: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_4, 0);  add_4 = None
        unsqueeze_2: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 1);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 2);  unsqueeze_2 = None
        iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_5: "i64[128]" = torch.ops.aten.add.Tensor(iota_2, 0);  iota_2 = None
        unsqueeze_4: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_5, 0);  add_5 = None
        unsqueeze_5: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        unsqueeze_6: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 3);  unsqueeze_5 = None
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_3, unsqueeze_6);  unsqueeze_3 = unsqueeze_6 = None
        expand: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(le, _shape_param_10);  le = _shape_param_10 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_1: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[32, 1, 128, 128]" = torch.ops.aten.where.self(expand, full, full_1);  expand = full = full_1 = None
        add_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_10, where);  view_10 = None
        full_2: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_6, full_2);  add_6 = full_2 = None
        view_11: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum, _shape_param_11);  maximum = _shape_param_11 = None
        convert_element_type_2: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        amax: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub_1: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_3: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        view_12: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_1, _shape_param_12);  convert_element_type_1 = _shape_param_12 = None
        permute_5: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_2: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg10_1, view_12, permute_5);  arg10_1 = view_12 = permute_5 = None
        view_13: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_2, _shape_param_13);  addmm_2 = _shape_param_13 = None
        view_14: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_13, _shape_param_14);  view_13 = _shape_param_14 = None
        permute_6: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_2: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_15: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_15);  clone_2 = _shape_param_15 = None
        bmm_1: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_3, view_15);  convert_element_type_3 = view_15 = None
        view_16: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_16);  bmm_1 = _shape_param_16 = None
        permute_7: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_3: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_17: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_17);  clone_3 = _shape_param_17 = None
        view_18: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_17, _shape_param_18);  view_17 = _shape_param_18 = None
        permute_8: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm_3: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg12_1, view_18, permute_8);  arg12_1 = view_18 = permute_8 = None
        view_19: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_3, _shape_param_19);  addmm_3 = _shape_param_19 = None
        add_7: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_1, view_19);  add_1 = view_19 = None
        convert_element_type_4: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [2], correction = 0, keepdim = True)
        getitem_2: "f32[32, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[32, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_4, getitem_3);  convert_element_type_4 = getitem_3 = None
        add_8: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_4: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_5: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
        add_9: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        convert_element_type_5: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        view_20: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_20);  convert_element_type_5 = _shape_param_20 = None
        permute_9: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_4: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg16_1, view_20, permute_9);  arg16_1 = view_20 = permute_9 = None
        view_21: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_4, _shape_param_21);  addmm_4 = _shape_param_21 = None
        convert_element_type_6: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        mul_6: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.5)
        mul_7: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.7071067811865476);  convert_element_type_6 = None
        erf: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
        add_10: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_8: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_6, add_10);  mul_6 = add_10 = None
        convert_element_type_7: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        view_22: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_22);  convert_element_type_7 = _shape_param_22 = None
        permute_10: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_5: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg18_1, view_22, permute_10);  arg18_1 = view_22 = permute_10 = None
        view_23: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_5, _shape_param_23);  addmm_5 = _shape_param_23 = None
        add_11: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_7, view_23);  add_7 = view_23 = None
        convert_element_type_8: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_11, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_4: "f32[32, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[32, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_5);  convert_element_type_8 = getitem_5 = None
        add_12: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
        mul_9: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_10: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
        add_13: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        convert_element_type_9: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16);  add_13 = None
        view_24: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_24);  _shape_param_24 = None
        permute_11: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg22_1, view_24, permute_11);  arg22_1 = view_24 = permute_11 = None
        view_25: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_6, _shape_param_25);  addmm_6 = _shape_param_25 = None
        mul_11: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_25, 0.125);  view_25 = None
        view_26: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_11, _shape_param_26);  mul_11 = _shape_param_26 = None
        permute_12: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        clone_4: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_27: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_27);  clone_4 = _shape_param_27 = None
        view_28: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_28);  _shape_param_28 = None
        permute_13: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
        addmm_7: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg24_1, view_28, permute_13);  arg24_1 = view_28 = permute_13 = None
        view_29: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_7, _shape_param_29);  addmm_7 = _shape_param_29 = None
        view_30: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_29, _shape_param_30);  view_29 = _shape_param_30 = None
        permute_14: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        clone_5: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_31: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_5, _shape_param_31);  clone_5 = _shape_param_31 = None
        permute_15: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        bmm_2: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_27, permute_15);  view_27 = permute_15 = None
        view_32: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_32);  bmm_2 = _shape_param_32 = None
        add_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_32, where);  view_32 = None
        full_3: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_14, full_3);  add_14 = full_3 = None
        view_33: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_1, _shape_param_33);  maximum_1 = _shape_param_33 = None
        convert_element_type_10: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_33, torch.float32);  view_33 = None
        amax_1: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_10, [-1], True)
        sub_4: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_1);  convert_element_type_10 = amax_1 = None
        exp_1: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_11: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        view_34: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_34);  convert_element_type_9 = _shape_param_34 = None
        permute_16: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_8: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg26_1, view_34, permute_16);  arg26_1 = view_34 = permute_16 = None
        view_35: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_8, _shape_param_35);  addmm_8 = _shape_param_35 = None
        view_36: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_35, _shape_param_36);  view_35 = _shape_param_36 = None
        permute_17: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_6: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_37: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_37);  clone_6 = _shape_param_37 = None
        bmm_3: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_11, view_37);  convert_element_type_11 = view_37 = None
        view_38: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_38);  bmm_3 = _shape_param_38 = None
        permute_18: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_7: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_39: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_7, _shape_param_39);  clone_7 = _shape_param_39 = None
        view_40: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_39, _shape_param_40);  view_39 = _shape_param_40 = None
        permute_19: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_9: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg28_1, view_40, permute_19);  arg28_1 = view_40 = permute_19 = None
        view_41: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_9, _shape_param_41);  addmm_9 = _shape_param_41 = None
        add_15: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_11, view_41);  add_11 = view_41 = None
        convert_element_type_12: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_15, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_12, [2], correction = 0, keepdim = True)
        getitem_6: "f32[32, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[32, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_12, getitem_7);  convert_element_type_12 = getitem_7 = None
        add_16: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_12: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
        add_17: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
        convert_element_type_13: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        view_42: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_13, _shape_param_42);  convert_element_type_13 = _shape_param_42 = None
        permute_20: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_10: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg32_1, view_42, permute_20);  arg32_1 = view_42 = permute_20 = None
        view_43: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_10, _shape_param_43);  addmm_10 = _shape_param_43 = None
        convert_element_type_14: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        mul_14: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5)
        mul_15: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.7071067811865476);  convert_element_type_14 = None
        erf_1: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
        add_18: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_16: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_14, add_18);  mul_14 = add_18 = None
        convert_element_type_15: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_16, torch.bfloat16);  mul_16 = None
        view_44: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_44);  convert_element_type_15 = _shape_param_44 = None
        permute_21: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_11: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg34_1, view_44, permute_21);  arg34_1 = view_44 = permute_21 = None
        view_45: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_11, _shape_param_45);  addmm_11 = _shape_param_45 = None
        add_19: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_15, view_45);  add_15 = view_45 = None
        convert_element_type_16: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_8: "f32[32, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[32, 128, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_6: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_9);  convert_element_type_16 = getitem_9 = None
        add_20: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_17: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_18: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
        add_21: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
        convert_element_type_17: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None
        view_46: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_46);  _shape_param_46 = None
        permute_22: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_12: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg38_1, view_46, permute_22);  arg38_1 = view_46 = permute_22 = None
        view_47: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_12, _shape_param_47);  addmm_12 = _shape_param_47 = None
        mul_19: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_47, 0.125);  view_47 = None
        view_48: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_19, _shape_param_48);  mul_19 = _shape_param_48 = None
        permute_23: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        clone_8: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_49: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_49);  clone_8 = _shape_param_49 = None
        view_50: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_50);  _shape_param_50 = None
        permute_24: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_13: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg40_1, view_50, permute_24);  arg40_1 = view_50 = permute_24 = None
        view_51: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_13, _shape_param_51);  addmm_13 = _shape_param_51 = None
        view_52: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_51, _shape_param_52);  view_51 = _shape_param_52 = None
        permute_25: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_9: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_53: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_9, _shape_param_53);  clone_9 = _shape_param_53 = None
        permute_26: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        bmm_4: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_49, permute_26);  view_49 = permute_26 = None
        view_54: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_54);  bmm_4 = _shape_param_54 = None
        add_22: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_54, where);  view_54 = None
        full_4: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_22, full_4);  add_22 = full_4 = None
        view_55: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_2, _shape_param_55);  maximum_2 = _shape_param_55 = None
        convert_element_type_18: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        amax_2: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_18, [-1], True)
        sub_7: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_2);  convert_element_type_18 = amax_2 = None
        exp_2: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_19: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        view_56: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_56);  convert_element_type_17 = _shape_param_56 = None
        permute_27: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_14: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg42_1, view_56, permute_27);  arg42_1 = view_56 = permute_27 = None
        view_57: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_14, _shape_param_57);  addmm_14 = _shape_param_57 = None
        view_58: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_57, _shape_param_58);  view_57 = _shape_param_58 = None
        permute_28: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_10: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_59: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_59);  clone_10 = _shape_param_59 = None
        bmm_5: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_19, view_59);  convert_element_type_19 = view_59 = None
        view_60: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_60);  bmm_5 = _shape_param_60 = None
        permute_29: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_11: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_61: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_11, _shape_param_61);  clone_11 = _shape_param_61 = None
        view_62: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_61, _shape_param_62);  view_61 = _shape_param_62 = None
        permute_30: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_15: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg44_1, view_62, permute_30);  arg44_1 = view_62 = permute_30 = None
        view_63: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_15, _shape_param_63);  addmm_15 = _shape_param_63 = None
        add_23: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_19, view_63);  add_19 = view_63 = None
        convert_element_type_20: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_23, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_10: "f32[32, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[32, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_8: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_11);  convert_element_type_20 = getitem_11 = None
        add_24: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        mul_20: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
        mul_21: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
        add_25: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
        convert_element_type_21: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None
        view_64: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_64);  convert_element_type_21 = _shape_param_64 = None
        permute_31: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_16: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg48_1, view_64, permute_31);  arg48_1 = view_64 = permute_31 = None
        view_65: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_16, _shape_param_65);  addmm_16 = _shape_param_65 = None
        convert_element_type_22: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_65, torch.float32);  view_65 = None
        mul_22: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.5)
        mul_23: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 0.7071067811865476);  convert_element_type_22 = None
        erf_2: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
        add_26: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_24: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_22, add_26);  mul_22 = add_26 = None
        convert_element_type_23: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        view_66: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_23, _shape_param_66);  convert_element_type_23 = _shape_param_66 = None
        permute_32: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_17: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg50_1, view_66, permute_32);  arg50_1 = view_66 = permute_32 = None
        view_67: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_17, _shape_param_67);  addmm_17 = _shape_param_67 = None
        add_27: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_23, view_67);  add_23 = view_67 = None
        convert_element_type_24: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_27, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_24, [2], correction = 0, keepdim = True)
        getitem_12: "f32[32, 128, 1]" = var_mean_6[0]
        getitem_13: "f32[32, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_9: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_24, getitem_13);  convert_element_type_24 = getitem_13 = None
        add_28: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        mul_25: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
        mul_26: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
        add_29: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
        convert_element_type_25: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        view_68: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_68);  _shape_param_68 = None
        permute_33: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_18: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg54_1, view_68, permute_33);  arg54_1 = view_68 = permute_33 = None
        view_69: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_18, _shape_param_69);  addmm_18 = _shape_param_69 = None
        mul_27: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_69, 0.125);  view_69 = None
        view_70: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_27, _shape_param_70);  mul_27 = _shape_param_70 = None
        permute_34: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_12: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_71: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_71);  clone_12 = _shape_param_71 = None
        view_72: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_72);  _shape_param_72 = None
        permute_35: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_19: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg56_1, view_72, permute_35);  arg56_1 = view_72 = permute_35 = None
        view_73: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_19, _shape_param_73);  addmm_19 = _shape_param_73 = None
        view_74: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_73, _shape_param_74);  view_73 = _shape_param_74 = None
        permute_36: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        clone_13: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_75: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_13, _shape_param_75);  clone_13 = _shape_param_75 = None
        permute_37: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
        bmm_6: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_71, permute_37);  view_71 = permute_37 = None
        view_76: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_76);  bmm_6 = _shape_param_76 = None
        add_30: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_76, where);  view_76 = None
        full_5: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_30, full_5);  add_30 = full_5 = None
        view_77: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_3, _shape_param_77);  maximum_3 = _shape_param_77 = None
        convert_element_type_26: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        amax_3: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_26, [-1], True)
        sub_10: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_3);  convert_element_type_26 = amax_3 = None
        exp_3: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_27: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        view_78: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_25, _shape_param_78);  convert_element_type_25 = _shape_param_78 = None
        permute_38: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_20: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg58_1, view_78, permute_38);  arg58_1 = view_78 = permute_38 = None
        view_79: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_20, _shape_param_79);  addmm_20 = _shape_param_79 = None
        view_80: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_79, _shape_param_80);  view_79 = _shape_param_80 = None
        permute_39: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_14: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_81: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_14, _shape_param_81);  clone_14 = _shape_param_81 = None
        bmm_7: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_27, view_81);  convert_element_type_27 = view_81 = None
        view_82: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_82);  bmm_7 = _shape_param_82 = None
        permute_40: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_15: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_83: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_15, _shape_param_83);  clone_15 = _shape_param_83 = None
        view_84: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_83, _shape_param_84);  view_83 = _shape_param_84 = None
        permute_41: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_21: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg60_1, view_84, permute_41);  arg60_1 = view_84 = permute_41 = None
        view_85: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_21, _shape_param_85);  addmm_21 = _shape_param_85 = None
        add_31: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_27, view_85);  add_27 = view_85 = None
        convert_element_type_28: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_28, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 128, 1]" = var_mean_7[0]
        getitem_15: "f32[32, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_11: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_28, getitem_15);  convert_element_type_28 = getitem_15 = None
        add_32: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_28: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
        mul_29: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
        add_33: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
        convert_element_type_29: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        view_86: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_29, _shape_param_86);  convert_element_type_29 = _shape_param_86 = None
        permute_42: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_22: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg64_1, view_86, permute_42);  arg64_1 = view_86 = permute_42 = None
        view_87: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_22, _shape_param_87);  addmm_22 = _shape_param_87 = None
        convert_element_type_30: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_87, torch.float32);  view_87 = None
        mul_30: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.5)
        mul_31: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 0.7071067811865476);  convert_element_type_30 = None
        erf_3: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_34: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_32: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, add_34);  mul_30 = add_34 = None
        convert_element_type_31: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        view_88: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_88);  convert_element_type_31 = _shape_param_88 = None
        permute_43: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_23: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg66_1, view_88, permute_43);  arg66_1 = view_88 = permute_43 = None
        view_89: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_23, _shape_param_89);  addmm_23 = _shape_param_89 = None
        add_35: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_31, view_89);  add_31 = view_89 = None
        convert_element_type_32: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_32, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 128, 1]" = var_mean_8[0]
        getitem_17: "f32[32, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_12: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_32, getitem_17);  convert_element_type_32 = getitem_17 = None
        add_36: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_33: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
        mul_34: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
        add_37: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
        convert_element_type_33: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16);  add_37 = None
        view_90: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_90);  _shape_param_90 = None
        permute_44: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_24: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg70_1, view_90, permute_44);  arg70_1 = view_90 = permute_44 = None
        view_91: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_24, _shape_param_91);  addmm_24 = _shape_param_91 = None
        mul_35: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_91, 0.125);  view_91 = None
        view_92: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_35, _shape_param_92);  mul_35 = _shape_param_92 = None
        permute_45: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        clone_16: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_93: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_93);  clone_16 = _shape_param_93 = None
        view_94: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_94);  _shape_param_94 = None
        permute_46: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_25: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg72_1, view_94, permute_46);  arg72_1 = view_94 = permute_46 = None
        view_95: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_25, _shape_param_95);  addmm_25 = _shape_param_95 = None
        view_96: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_95, _shape_param_96);  view_95 = _shape_param_96 = None
        permute_47: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        clone_17: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_97: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_17, _shape_param_97);  clone_17 = _shape_param_97 = None
        permute_48: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
        bmm_8: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_93, permute_48);  view_93 = permute_48 = None
        view_98: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_98);  bmm_8 = _shape_param_98 = None
        add_38: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_98, where);  view_98 = None
        full_6: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_38, full_6);  add_38 = full_6 = None
        view_99: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_4, _shape_param_99);  maximum_4 = _shape_param_99 = None
        convert_element_type_34: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        amax_4: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_34, [-1], True)
        sub_13: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_4);  convert_element_type_34 = amax_4 = None
        exp_4: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_35: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        view_100: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_33, _shape_param_100);  convert_element_type_33 = _shape_param_100 = None
        permute_49: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_26: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg74_1, view_100, permute_49);  arg74_1 = view_100 = permute_49 = None
        view_101: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_26, _shape_param_101);  addmm_26 = _shape_param_101 = None
        view_102: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_101, _shape_param_102);  view_101 = _shape_param_102 = None
        permute_50: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_18: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_103: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_103);  clone_18 = _shape_param_103 = None
        bmm_9: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_35, view_103);  convert_element_type_35 = view_103 = None
        view_104: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_104);  bmm_9 = _shape_param_104 = None
        permute_51: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_19: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_105: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_19, _shape_param_105);  clone_19 = _shape_param_105 = None
        view_106: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_105, _shape_param_106);  view_105 = _shape_param_106 = None
        permute_52: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_27: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg76_1, view_106, permute_52);  arg76_1 = view_106 = permute_52 = None
        view_107: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_27, _shape_param_107);  addmm_27 = _shape_param_107 = None
        add_39: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_35, view_107);  add_35 = view_107 = None
        convert_element_type_36: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_39, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_36, [2], correction = 0, keepdim = True)
        getitem_18: "f32[32, 128, 1]" = var_mean_9[0]
        getitem_19: "f32[32, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_14: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_36, getitem_19);  convert_element_type_36 = getitem_19 = None
        add_40: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
        mul_36: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
        mul_37: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
        add_41: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
        convert_element_type_37: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_41, torch.bfloat16);  add_41 = None
        view_108: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_37, _shape_param_108);  convert_element_type_37 = _shape_param_108 = None
        permute_53: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_28: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg80_1, view_108, permute_53);  arg80_1 = view_108 = permute_53 = None
        view_109: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_28, _shape_param_109);  addmm_28 = _shape_param_109 = None
        convert_element_type_38: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        mul_38: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5)
        mul_39: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.7071067811865476);  convert_element_type_38 = None
        erf_4: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_42: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_40: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_38, add_42);  mul_38 = add_42 = None
        convert_element_type_39: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None
        view_110: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_39, _shape_param_110);  convert_element_type_39 = _shape_param_110 = None
        permute_54: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_29: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg82_1, view_110, permute_54);  arg82_1 = view_110 = permute_54 = None
        view_111: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_29, _shape_param_111);  addmm_29 = _shape_param_111 = None
        add_43: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_39, view_111);  add_39 = view_111 = None
        convert_element_type_40: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_43, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_20: "f32[32, 128, 1]" = var_mean_10[0]
        getitem_21: "f32[32, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_15: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_21);  convert_element_type_40 = getitem_21 = None
        add_44: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
        mul_41: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
        mul_42: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
        add_45: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
        convert_element_type_41: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None
        view_112: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_112);  _shape_param_112 = None
        permute_55: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_30: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg86_1, view_112, permute_55);  arg86_1 = view_112 = permute_55 = None
        view_113: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_30, _shape_param_113);  addmm_30 = _shape_param_113 = None
        mul_43: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_113, 0.125);  view_113 = None
        view_114: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_43, _shape_param_114);  mul_43 = _shape_param_114 = None
        permute_56: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None
        clone_20: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_115: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_115);  clone_20 = _shape_param_115 = None
        view_116: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_116);  _shape_param_116 = None
        permute_57: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_31: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg88_1, view_116, permute_57);  arg88_1 = view_116 = permute_57 = None
        view_117: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_31, _shape_param_117);  addmm_31 = _shape_param_117 = None
        view_118: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_117, _shape_param_118);  view_117 = _shape_param_118 = None
        permute_58: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_21: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_119: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_119);  clone_21 = _shape_param_119 = None
        permute_59: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        bmm_10: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_115, permute_59);  view_115 = permute_59 = None
        view_120: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_120);  bmm_10 = _shape_param_120 = None
        add_46: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_120, where);  view_120 = None
        full_7: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_5: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_46, full_7);  add_46 = full_7 = None
        view_121: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_5, _shape_param_121);  maximum_5 = _shape_param_121 = None
        convert_element_type_42: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_121, torch.float32);  view_121 = None
        amax_5: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_42, [-1], True)
        sub_16: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_42, amax_5);  convert_element_type_42 = amax_5 = None
        exp_5: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_43: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        view_122: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_41, _shape_param_122);  convert_element_type_41 = _shape_param_122 = None
        permute_60: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_32: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg90_1, view_122, permute_60);  arg90_1 = view_122 = permute_60 = None
        view_123: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_32, _shape_param_123);  addmm_32 = _shape_param_123 = None
        view_124: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_123, _shape_param_124);  view_123 = _shape_param_124 = None
        permute_61: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        clone_22: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_125: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_125);  clone_22 = _shape_param_125 = None
        bmm_11: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_43, view_125);  convert_element_type_43 = view_125 = None
        view_126: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_126);  bmm_11 = _shape_param_126 = None
        permute_62: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_23: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_127: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_23, _shape_param_127);  clone_23 = _shape_param_127 = None
        view_128: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_127, _shape_param_128);  view_127 = _shape_param_128 = None
        permute_63: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_33: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg92_1, view_128, permute_63);  arg92_1 = view_128 = permute_63 = None
        view_129: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_33, _shape_param_129);  addmm_33 = _shape_param_129 = None
        add_47: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_43, view_129);  add_43 = view_129 = None
        convert_element_type_44: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_44, [2], correction = 0, keepdim = True)
        getitem_22: "f32[32, 128, 1]" = var_mean_11[0]
        getitem_23: "f32[32, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_17: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_44, getitem_23);  convert_element_type_44 = getitem_23 = None
        add_48: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        mul_44: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
        mul_45: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
        add_49: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
        convert_element_type_45: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None
        view_130: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_45, _shape_param_130);  convert_element_type_45 = _shape_param_130 = None
        permute_64: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_34: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg96_1, view_130, permute_64);  arg96_1 = view_130 = permute_64 = None
        view_131: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_34, _shape_param_131);  addmm_34 = _shape_param_131 = None
        convert_element_type_46: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None
        mul_46: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.5)
        mul_47: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 0.7071067811865476);  convert_element_type_46 = None
        erf_5: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_50: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_48: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_46, add_50);  mul_46 = add_50 = None
        convert_element_type_47: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_48, torch.bfloat16);  mul_48 = None
        view_132: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_47, _shape_param_132);  convert_element_type_47 = _shape_param_132 = None
        permute_65: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_35: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg98_1, view_132, permute_65);  arg98_1 = view_132 = permute_65 = None
        view_133: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_35, _shape_param_133);  addmm_35 = _shape_param_133 = None
        add_51: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_47, view_133);  add_47 = view_133 = None
        convert_element_type_48: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_51, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem_24: "f32[32, 128, 1]" = var_mean_12[0]
        getitem_25: "f32[32, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_18: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_25);  convert_element_type_48 = getitem_25 = None
        add_52: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_49: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
        mul_50: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
        add_53: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
        convert_element_type_49: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        view_134: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_134);  _shape_param_134 = None
        permute_66: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_36: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg102_1, view_134, permute_66);  arg102_1 = view_134 = permute_66 = None
        view_135: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_36, _shape_param_135);  addmm_36 = _shape_param_135 = None
        mul_51: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_135, 0.125);  view_135 = None
        view_136: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_51, _shape_param_136);  mul_51 = _shape_param_136 = None
        permute_67: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        clone_24: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_137: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_137);  clone_24 = _shape_param_137 = None
        view_138: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_138);  _shape_param_138 = None
        permute_68: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_37: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg104_1, view_138, permute_68);  arg104_1 = view_138 = permute_68 = None
        view_139: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_37, _shape_param_139);  addmm_37 = _shape_param_139 = None
        view_140: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_139, _shape_param_140);  view_139 = _shape_param_140 = None
        permute_69: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        clone_25: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_141: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_25, _shape_param_141);  clone_25 = _shape_param_141 = None
        permute_70: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
        bmm_12: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_137, permute_70);  view_137 = permute_70 = None
        view_142: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_142);  bmm_12 = _shape_param_142 = None
        add_54: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_142, where);  view_142 = None
        full_8: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_54, full_8);  add_54 = full_8 = None
        view_143: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_6, _shape_param_143);  maximum_6 = _shape_param_143 = None
        convert_element_type_50: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        amax_6: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_50, [-1], True)
        sub_19: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_50, amax_6);  convert_element_type_50 = amax_6 = None
        exp_6: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_51: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        view_144: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_144);  convert_element_type_49 = _shape_param_144 = None
        permute_71: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_38: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg106_1, view_144, permute_71);  arg106_1 = view_144 = permute_71 = None
        view_145: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_38, _shape_param_145);  addmm_38 = _shape_param_145 = None
        view_146: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_145, _shape_param_146);  view_145 = _shape_param_146 = None
        permute_72: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_26: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_147: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_26, _shape_param_147);  clone_26 = _shape_param_147 = None
        bmm_13: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_51, view_147);  convert_element_type_51 = view_147 = None
        view_148: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_148);  bmm_13 = _shape_param_148 = None
        permute_73: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_27: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_149: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_27, _shape_param_149);  clone_27 = _shape_param_149 = None
        view_150: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_149, _shape_param_150);  view_149 = _shape_param_150 = None
        permute_74: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_39: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg108_1, view_150, permute_74);  arg108_1 = view_150 = permute_74 = None
        view_151: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_39, _shape_param_151);  addmm_39 = _shape_param_151 = None
        add_55: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_51, view_151);  add_51 = view_151 = None
        convert_element_type_52: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_55, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_52, [2], correction = 0, keepdim = True)
        getitem_26: "f32[32, 128, 1]" = var_mean_13[0]
        getitem_27: "f32[32, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_20: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_52, getitem_27);  convert_element_type_52 = getitem_27 = None
        add_56: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_52: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
        mul_53: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg109_1);  mul_52 = arg109_1 = None
        add_57: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
        convert_element_type_53: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None
        view_152: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_53, _shape_param_152);  convert_element_type_53 = _shape_param_152 = None
        permute_75: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_40: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg112_1, view_152, permute_75);  arg112_1 = view_152 = permute_75 = None
        view_153: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_40, _shape_param_153);  addmm_40 = _shape_param_153 = None
        convert_element_type_54: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_153, torch.float32);  view_153 = None
        mul_54: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.5)
        mul_55: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 0.7071067811865476);  convert_element_type_54 = None
        erf_6: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
        add_58: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_56: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_58);  mul_54 = add_58 = None
        convert_element_type_55: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None
        view_154: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_55, _shape_param_154);  convert_element_type_55 = _shape_param_154 = None
        permute_76: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_41: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg114_1, view_154, permute_76);  arg114_1 = view_154 = permute_76 = None
        view_155: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_41, _shape_param_155);  addmm_41 = _shape_param_155 = None
        add_59: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_55, view_155);  add_55 = view_155 = None
        convert_element_type_56: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_56, [2], correction = 0, keepdim = True)
        getitem_28: "f32[32, 128, 1]" = var_mean_14[0]
        getitem_29: "f32[32, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_21: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_56, getitem_29);  convert_element_type_56 = getitem_29 = None
        add_60: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_57: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
        mul_58: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_57, arg115_1);  mul_57 = arg115_1 = None
        add_61: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
        convert_element_type_57: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None
        view_156: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_156);  _shape_param_156 = None
        permute_77: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_42: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg118_1, view_156, permute_77);  arg118_1 = view_156 = permute_77 = None
        view_157: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_42, _shape_param_157);  addmm_42 = _shape_param_157 = None
        mul_59: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_157, 0.125);  view_157 = None
        view_158: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_59, _shape_param_158);  mul_59 = _shape_param_158 = None
        permute_78: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_158, [0, 2, 1, 3]);  view_158 = None
        clone_28: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_159: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_159);  clone_28 = _shape_param_159 = None
        view_160: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_160);  _shape_param_160 = None
        permute_79: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_43: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg120_1, view_160, permute_79);  arg120_1 = view_160 = permute_79 = None
        view_161: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_43, _shape_param_161);  addmm_43 = _shape_param_161 = None
        view_162: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_161, _shape_param_162);  view_161 = _shape_param_162 = None
        permute_80: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        clone_29: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_163: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_29, _shape_param_163);  clone_29 = _shape_param_163 = None
        permute_81: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        bmm_14: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_159, permute_81);  view_159 = permute_81 = None
        view_164: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_164);  bmm_14 = _shape_param_164 = None
        add_62: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_164, where);  view_164 = None
        full_9: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_7: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_62, full_9);  add_62 = full_9 = None
        view_165: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_7, _shape_param_165);  maximum_7 = _shape_param_165 = None
        convert_element_type_58: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        amax_7: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_58, [-1], True)
        sub_22: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_58, amax_7);  convert_element_type_58 = amax_7 = None
        exp_7: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_59: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        view_166: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_166);  convert_element_type_57 = _shape_param_166 = None
        permute_82: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_44: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg122_1, view_166, permute_82);  arg122_1 = view_166 = permute_82 = None
        view_167: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_44, _shape_param_167);  addmm_44 = _shape_param_167 = None
        view_168: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_167, _shape_param_168);  view_167 = _shape_param_168 = None
        permute_83: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_30: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_169: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_169);  clone_30 = _shape_param_169 = None
        bmm_15: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_59, view_169);  convert_element_type_59 = view_169 = None
        view_170: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_170);  bmm_15 = _shape_param_170 = None
        permute_84: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_31: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_171: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_31, _shape_param_171);  clone_31 = _shape_param_171 = None
        view_172: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_171, _shape_param_172);  view_171 = _shape_param_172 = None
        permute_85: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_45: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg124_1, view_172, permute_85);  arg124_1 = view_172 = permute_85 = None
        view_173: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_45, _shape_param_173);  addmm_45 = _shape_param_173 = None
        add_63: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_59, view_173);  add_59 = view_173 = None
        convert_element_type_60: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 128, 1]" = var_mean_15[0]
        getitem_31: "f32[32, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_23: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_31);  convert_element_type_60 = getitem_31 = None
        add_64: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_60: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
        mul_61: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_60, arg125_1);  mul_60 = arg125_1 = None
        add_65: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
        convert_element_type_61: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None
        view_174: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_174);  convert_element_type_61 = _shape_param_174 = None
        permute_86: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_46: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg128_1, view_174, permute_86);  arg128_1 = view_174 = permute_86 = None
        view_175: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_46, _shape_param_175);  addmm_46 = _shape_param_175 = None
        convert_element_type_62: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_175, torch.float32);  view_175 = None
        mul_62: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.5)
        mul_63: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.7071067811865476);  convert_element_type_62 = None
        erf_7: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_66: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_64: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_62, add_66);  mul_62 = add_66 = None
        convert_element_type_63: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        view_176: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_63, _shape_param_176);  convert_element_type_63 = _shape_param_176 = None
        permute_87: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_47: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg130_1, view_176, permute_87);  arg130_1 = view_176 = permute_87 = None
        view_177: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_47, _shape_param_177);  addmm_47 = _shape_param_177 = None
        add_67: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_63, view_177);  add_63 = view_177 = None
        convert_element_type_64: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_67, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_64, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 128, 1]" = var_mean_16[0]
        getitem_33: "f32[32, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_24: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_64, getitem_33);  convert_element_type_64 = getitem_33 = None
        add_68: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
        mul_65: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
        mul_66: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
        add_69: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
        convert_element_type_65: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None
        view_178: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_65, _shape_param_178);  _shape_param_178 = None
        permute_88: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg134_1, view_178, permute_88);  arg134_1 = view_178 = permute_88 = None
        view_179: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_48, _shape_param_179);  addmm_48 = _shape_param_179 = None
        mul_67: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_179, 0.125);  view_179 = None
        view_180: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_67, _shape_param_180);  mul_67 = _shape_param_180 = None
        permute_89: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        clone_32: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_181: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_181);  clone_32 = _shape_param_181 = None
        view_182: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_65, _shape_param_182);  _shape_param_182 = None
        permute_90: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg136_1, view_182, permute_90);  arg136_1 = view_182 = permute_90 = None
        view_183: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_49, _shape_param_183);  addmm_49 = _shape_param_183 = None
        view_184: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_183, _shape_param_184);  view_183 = _shape_param_184 = None
        permute_91: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        clone_33: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_185: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_33, _shape_param_185);  clone_33 = _shape_param_185 = None
        permute_92: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
        bmm_16: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_181, permute_92);  view_181 = permute_92 = None
        view_186: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_186);  bmm_16 = _shape_param_186 = None
        add_70: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_186, where);  view_186 = None
        full_10: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_8: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_70, full_10);  add_70 = full_10 = None
        view_187: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_8, _shape_param_187);  maximum_8 = _shape_param_187 = None
        convert_element_type_66: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_187, torch.float32);  view_187 = None
        amax_8: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_66, [-1], True)
        sub_25: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_66, amax_8);  convert_element_type_66 = amax_8 = None
        exp_8: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_67: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        view_188: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_65, _shape_param_188);  convert_element_type_65 = _shape_param_188 = None
        permute_93: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_50: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg138_1, view_188, permute_93);  arg138_1 = view_188 = permute_93 = None
        view_189: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_50, _shape_param_189);  addmm_50 = _shape_param_189 = None
        view_190: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_189, _shape_param_190);  view_189 = _shape_param_190 = None
        permute_94: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_34: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_191: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_191);  clone_34 = _shape_param_191 = None
        bmm_17: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_67, view_191);  convert_element_type_67 = view_191 = None
        view_192: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_192);  bmm_17 = _shape_param_192 = None
        permute_95: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_35: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_193: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_35, _shape_param_193);  clone_35 = _shape_param_193 = None
        view_194: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_193, _shape_param_194);  view_193 = _shape_param_194 = None
        permute_96: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_51: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg140_1, view_194, permute_96);  arg140_1 = view_194 = permute_96 = None
        view_195: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_51, _shape_param_195);  addmm_51 = _shape_param_195 = None
        add_71: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_67, view_195);  add_67 = view_195 = None
        convert_element_type_68: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_71, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_68, [2], correction = 0, keepdim = True)
        getitem_34: "f32[32, 128, 1]" = var_mean_17[0]
        getitem_35: "f32[32, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_26: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_68, getitem_35);  convert_element_type_68 = getitem_35 = None
        add_72: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
        mul_68: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
        mul_69: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_68, arg141_1);  mul_68 = arg141_1 = None
        add_73: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_69, arg142_1);  mul_69 = arg142_1 = None
        convert_element_type_69: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_73, torch.bfloat16);  add_73 = None
        view_196: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_69, _shape_param_196);  convert_element_type_69 = _shape_param_196 = None
        permute_97: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_52: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg144_1, view_196, permute_97);  arg144_1 = view_196 = permute_97 = None
        view_197: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_52, _shape_param_197);  addmm_52 = _shape_param_197 = None
        convert_element_type_70: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        mul_70: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 0.5)
        mul_71: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 0.7071067811865476);  convert_element_type_70 = None
        erf_8: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
        add_74: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_72: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, add_74);  mul_70 = add_74 = None
        convert_element_type_71: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        view_198: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_198);  convert_element_type_71 = _shape_param_198 = None
        permute_98: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_53: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg146_1, view_198, permute_98);  arg146_1 = view_198 = permute_98 = None
        view_199: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_53, _shape_param_199);  addmm_53 = _shape_param_199 = None
        add_75: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_71, view_199);  add_71 = view_199 = None
        convert_element_type_72: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_75, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_36: "f32[32, 128, 1]" = var_mean_18[0]
        getitem_37: "f32[32, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_27: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_37);  convert_element_type_72 = getitem_37 = None
        add_76: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        mul_73: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
        mul_74: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_73, arg147_1);  mul_73 = arg147_1 = None
        add_77: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_74, arg148_1);  mul_74 = arg148_1 = None
        convert_element_type_73: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None
        view_200: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_73, _shape_param_200);  _shape_param_200 = None
        permute_99: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg150_1, view_200, permute_99);  arg150_1 = view_200 = permute_99 = None
        view_201: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_54, _shape_param_201);  addmm_54 = _shape_param_201 = None
        mul_75: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_201, 0.125);  view_201 = None
        view_202: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_75, _shape_param_202);  mul_75 = _shape_param_202 = None
        permute_100: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        clone_36: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_203: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_203);  clone_36 = _shape_param_203 = None
        view_204: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_73, _shape_param_204);  _shape_param_204 = None
        permute_101: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg152_1, view_204, permute_101);  arg152_1 = view_204 = permute_101 = None
        view_205: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_55, _shape_param_205);  addmm_55 = _shape_param_205 = None
        view_206: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_205, _shape_param_206);  view_205 = _shape_param_206 = None
        permute_102: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        clone_37: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_207: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_37, _shape_param_207);  clone_37 = _shape_param_207 = None
        permute_103: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
        bmm_18: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_203, permute_103);  view_203 = permute_103 = None
        view_208: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_208);  bmm_18 = _shape_param_208 = None
        add_78: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_208, where);  view_208 = None
        full_11: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_9: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_78, full_11);  add_78 = full_11 = None
        view_209: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_9, _shape_param_209);  maximum_9 = _shape_param_209 = None
        convert_element_type_74: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        amax_9: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_74, [-1], True)
        sub_28: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_74, amax_9);  convert_element_type_74 = amax_9 = None
        exp_9: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_75: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        view_210: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_73, _shape_param_210);  convert_element_type_73 = _shape_param_210 = None
        permute_104: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_56: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg154_1, view_210, permute_104);  arg154_1 = view_210 = permute_104 = None
        view_211: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_56, _shape_param_211);  addmm_56 = _shape_param_211 = None
        view_212: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_211, _shape_param_212);  view_211 = _shape_param_212 = None
        permute_105: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_38: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_213: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_213);  clone_38 = _shape_param_213 = None
        bmm_19: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_75, view_213);  convert_element_type_75 = view_213 = None
        view_214: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_214);  bmm_19 = _shape_param_214 = None
        permute_106: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_39: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_215: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_39, _shape_param_215);  clone_39 = _shape_param_215 = None
        view_216: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_215, _shape_param_216);  view_215 = _shape_param_216 = None
        permute_107: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_57: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg156_1, view_216, permute_107);  arg156_1 = view_216 = permute_107 = None
        view_217: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_57, _shape_param_217);  addmm_57 = _shape_param_217 = None
        add_79: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_75, view_217);  add_75 = view_217 = None
        convert_element_type_76: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_79, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_76, [2], correction = 0, keepdim = True)
        getitem_38: "f32[32, 128, 1]" = var_mean_19[0]
        getitem_39: "f32[32, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_29: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_76, getitem_39);  convert_element_type_76 = getitem_39 = None
        add_80: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        mul_76: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
        mul_77: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_76, arg157_1);  mul_76 = arg157_1 = None
        add_81: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_77, arg158_1);  mul_77 = arg158_1 = None
        convert_element_type_77: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None
        view_218: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_77, _shape_param_218);  convert_element_type_77 = _shape_param_218 = None
        permute_108: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg160_1, view_218, permute_108);  arg160_1 = view_218 = permute_108 = None
        view_219: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_58, _shape_param_219);  addmm_58 = _shape_param_219 = None
        convert_element_type_78: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_219, torch.float32);  view_219 = None
        mul_78: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.5)
        mul_79: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.7071067811865476);  convert_element_type_78 = None
        erf_9: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
        add_82: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_80: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_78, add_82);  mul_78 = add_82 = None
        convert_element_type_79: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None
        view_220: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_79, _shape_param_220);  convert_element_type_79 = _shape_param_220 = None
        permute_109: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg162_1, view_220, permute_109);  arg162_1 = view_220 = permute_109 = None
        view_221: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_59, _shape_param_221);  addmm_59 = _shape_param_221 = None
        add_83: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_79, view_221);  add_79 = view_221 = None
        convert_element_type_80: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_83, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_40: "f32[32, 128, 1]" = var_mean_20[0]
        getitem_41: "f32[32, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_30: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_41);  convert_element_type_80 = getitem_41 = None
        add_84: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        mul_81: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
        mul_82: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_81, arg163_1);  mul_81 = arg163_1 = None
        add_85: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_82, arg164_1);  mul_82 = arg164_1 = None
        convert_element_type_81: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None
        view_222: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_222);  _shape_param_222 = None
        permute_110: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg166_1, view_222, permute_110);  arg166_1 = view_222 = permute_110 = None
        view_223: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_60, _shape_param_223);  addmm_60 = _shape_param_223 = None
        mul_83: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_223, 0.125);  view_223 = None
        view_224: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_83, _shape_param_224);  mul_83 = _shape_param_224 = None
        permute_111: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        clone_40: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_225: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_225);  clone_40 = _shape_param_225 = None
        view_226: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_226);  _shape_param_226 = None
        permute_112: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg168_1, view_226, permute_112);  arg168_1 = view_226 = permute_112 = None
        view_227: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_61, _shape_param_227);  addmm_61 = _shape_param_227 = None
        view_228: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_227, _shape_param_228);  view_227 = _shape_param_228 = None
        permute_113: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        clone_41: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_229: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_41, _shape_param_229);  clone_41 = _shape_param_229 = None
        permute_114: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        bmm_20: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_225, permute_114);  view_225 = permute_114 = None
        view_230: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_230);  bmm_20 = _shape_param_230 = None
        add_86: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_230, where);  view_230 = None
        full_12: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_10: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_86, full_12);  add_86 = full_12 = None
        view_231: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_10, _shape_param_231);  maximum_10 = _shape_param_231 = None
        convert_element_type_82: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        amax_10: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_82, [-1], True)
        sub_31: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_82, amax_10);  convert_element_type_82 = amax_10 = None
        exp_10: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_83: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        view_232: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_232);  convert_element_type_81 = _shape_param_232 = None
        permute_115: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_62: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg170_1, view_232, permute_115);  arg170_1 = view_232 = permute_115 = None
        view_233: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_62, _shape_param_233);  addmm_62 = _shape_param_233 = None
        view_234: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_233, _shape_param_234);  view_233 = _shape_param_234 = None
        permute_116: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
        clone_42: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_235: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_235);  clone_42 = _shape_param_235 = None
        bmm_21: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_83, view_235);  convert_element_type_83 = view_235 = None
        view_236: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_236);  bmm_21 = _shape_param_236 = None
        permute_117: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        clone_43: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_237: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_43, _shape_param_237);  clone_43 = _shape_param_237 = None
        view_238: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_237, _shape_param_238);  view_237 = _shape_param_238 = None
        permute_118: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_63: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg172_1, view_238, permute_118);  arg172_1 = view_238 = permute_118 = None
        view_239: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_63, _shape_param_239);  addmm_63 = _shape_param_239 = None
        add_87: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_83, view_239);  add_83 = view_239 = None
        convert_element_type_84: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_84, [2], correction = 0, keepdim = True)
        getitem_42: "f32[32, 128, 1]" = var_mean_21[0]
        getitem_43: "f32[32, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_32: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_84, getitem_43);  convert_element_type_84 = getitem_43 = None
        add_88: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_84: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
        mul_85: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_84, arg173_1);  mul_84 = arg173_1 = None
        add_89: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_85, arg174_1);  mul_85 = arg174_1 = None
        convert_element_type_85: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None
        view_240: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_85, _shape_param_240);  convert_element_type_85 = _shape_param_240 = None
        permute_119: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg176_1, view_240, permute_119);  arg176_1 = view_240 = permute_119 = None
        view_241: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_64, _shape_param_241);  addmm_64 = _shape_param_241 = None
        convert_element_type_86: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_241, torch.float32);  view_241 = None
        mul_86: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.5)
        mul_87: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 0.7071067811865476);  convert_element_type_86 = None
        erf_10: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_90: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_88: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_86, add_90);  mul_86 = add_90 = None
        convert_element_type_87: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None
        view_242: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_87, _shape_param_242);  convert_element_type_87 = _shape_param_242 = None
        permute_120: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg178_1, view_242, permute_120);  arg178_1 = view_242 = permute_120 = None
        view_243: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_65, _shape_param_243);  addmm_65 = _shape_param_243 = None
        add_91: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_87, view_243);  add_87 = view_243 = None
        convert_element_type_88: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_88, [2], correction = 0, keepdim = True)
        getitem_44: "f32[32, 128, 1]" = var_mean_22[0]
        getitem_45: "f32[32, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_33: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_88, getitem_45);  convert_element_type_88 = getitem_45 = None
        add_92: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_89: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
        mul_90: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_89, arg179_1);  mul_89 = arg179_1 = None
        add_93: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_90, arg180_1);  mul_90 = arg180_1 = None
        convert_element_type_89: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16);  add_93 = None
        view_244: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_244);  _shape_param_244 = None
        permute_121: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg182_1, view_244, permute_121);  arg182_1 = view_244 = permute_121 = None
        view_245: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_66, _shape_param_245);  addmm_66 = _shape_param_245 = None
        mul_91: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_245, 0.125);  view_245 = None
        view_246: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_91, _shape_param_246);  mul_91 = _shape_param_246 = None
        permute_122: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        clone_44: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_247: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_247);  clone_44 = _shape_param_247 = None
        view_248: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_248);  _shape_param_248 = None
        permute_123: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg184_1, view_248, permute_123);  arg184_1 = view_248 = permute_123 = None
        view_249: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_67, _shape_param_249);  addmm_67 = _shape_param_249 = None
        view_250: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_249, _shape_param_250);  view_249 = _shape_param_250 = None
        permute_124: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        clone_45: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_251: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_45, _shape_param_251);  clone_45 = _shape_param_251 = None
        permute_125: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
        bmm_22: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_247, permute_125);  view_247 = permute_125 = None
        view_252: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_252);  bmm_22 = _shape_param_252 = None
        add_94: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_252, where);  view_252 = None
        full_13: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_11: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_94, full_13);  add_94 = full_13 = None
        view_253: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_11, _shape_param_253);  maximum_11 = _shape_param_253 = None
        convert_element_type_90: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        amax_11: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_90, [-1], True)
        sub_34: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_90, amax_11);  convert_element_type_90 = amax_11 = None
        exp_11: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_91: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        view_254: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_254);  convert_element_type_89 = _shape_param_254 = None
        permute_126: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg186_1, view_254, permute_126);  arg186_1 = view_254 = permute_126 = None
        view_255: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_68, _shape_param_255);  addmm_68 = _shape_param_255 = None
        view_256: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_255, _shape_param_256);  view_255 = _shape_param_256 = None
        permute_127: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        clone_46: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_257: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_257);  clone_46 = _shape_param_257 = None
        bmm_23: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_91, view_257);  convert_element_type_91 = view_257 = None
        view_258: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_258);  bmm_23 = _shape_param_258 = None
        permute_128: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_47: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_259: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_47, _shape_param_259);  clone_47 = _shape_param_259 = None
        view_260: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_259, _shape_param_260);  view_259 = _shape_param_260 = None
        permute_129: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg188_1, view_260, permute_129);  arg188_1 = view_260 = permute_129 = None
        view_261: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_69, _shape_param_261);  addmm_69 = _shape_param_261 = None
        add_95: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_91, view_261);  add_91 = view_261 = None
        convert_element_type_92: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_95, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_92, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 128, 1]" = var_mean_23[0]
        getitem_47: "f32[32, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_35: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_92, getitem_47);  convert_element_type_92 = getitem_47 = None
        add_96: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_92: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
        mul_93: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_92, arg189_1);  mul_92 = arg189_1 = None
        add_97: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_93, arg190_1);  mul_93 = arg190_1 = None
        convert_element_type_93: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None
        view_262: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_93, _shape_param_262);  convert_element_type_93 = _shape_param_262 = None
        permute_130: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg192_1, view_262, permute_130);  arg192_1 = view_262 = permute_130 = None
        view_263: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_70, _shape_param_263);  addmm_70 = _shape_param_263 = None
        convert_element_type_94: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        mul_94: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.5)
        mul_95: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.7071067811865476);  convert_element_type_94 = None
        erf_11: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
        add_98: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_96: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_94, add_98);  mul_94 = add_98 = None
        convert_element_type_95: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_96, torch.bfloat16);  mul_96 = None
        view_264: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_95, _shape_param_264);  convert_element_type_95 = _shape_param_264 = None
        permute_131: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg194_1, view_264, permute_131);  arg194_1 = view_264 = permute_131 = None
        view_265: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_71, _shape_param_265);  addmm_71 = _shape_param_265 = None
        add_99: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_95, view_265);  add_95 = view_265 = None
        convert_element_type_96: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_99, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 128, 1]" = var_mean_24[0]
        getitem_49: "f32[32, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_36: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_49);  convert_element_type_96 = getitem_49 = None
        add_100: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_97: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
        mul_98: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_97, arg195_1);  mul_97 = arg195_1 = None
        add_101: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_98, arg196_1);  mul_98 = arg196_1 = None
        convert_element_type_97: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16);  add_101 = None
        view_266: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_97, _shape_param_266);  _shape_param_266 = None
        permute_132: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_72: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg198_1, view_266, permute_132);  arg198_1 = view_266 = permute_132 = None
        view_267: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_72, _shape_param_267);  addmm_72 = _shape_param_267 = None
        mul_99: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_267, 0.125);  view_267 = None
        view_268: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_99, _shape_param_268);  mul_99 = _shape_param_268 = None
        permute_133: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        clone_48: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
        view_269: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_269);  clone_48 = _shape_param_269 = None
        view_270: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_97, _shape_param_270);  _shape_param_270 = None
        permute_134: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_73: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg200_1, view_270, permute_134);  arg200_1 = view_270 = permute_134 = None
        view_271: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_73, _shape_param_271);  addmm_73 = _shape_param_271 = None
        view_272: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_271, _shape_param_272);  view_271 = _shape_param_272 = None
        permute_135: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        clone_49: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_273: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_49, _shape_param_273);  clone_49 = _shape_param_273 = None
        permute_136: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_273, [0, 2, 1]);  view_273 = None
        bmm_24: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_269, permute_136);  view_269 = permute_136 = None
        view_274: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_274);  bmm_24 = _shape_param_274 = None
        add_102: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_274, where);  view_274 = None
        full_14: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_102, full_14);  add_102 = full_14 = None
        view_275: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_12, _shape_param_275);  maximum_12 = _shape_param_275 = None
        convert_element_type_98: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_275, torch.float32);  view_275 = None
        amax_12: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_98, [-1], True)
        sub_37: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_98, amax_12);  convert_element_type_98 = amax_12 = None
        exp_12: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_99: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        view_276: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_97, _shape_param_276);  convert_element_type_97 = _shape_param_276 = None
        permute_137: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_74: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg202_1, view_276, permute_137);  arg202_1 = view_276 = permute_137 = None
        view_277: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_74, _shape_param_277);  addmm_74 = _shape_param_277 = None
        view_278: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_277, _shape_param_278);  view_277 = _shape_param_278 = None
        permute_138: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
        clone_50: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
        view_279: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_50, _shape_param_279);  clone_50 = _shape_param_279 = None
        bmm_25: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_99, view_279);  convert_element_type_99 = view_279 = None
        view_280: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_25, _shape_param_280);  bmm_25 = _shape_param_280 = None
        permute_139: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
        clone_51: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_281: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_51, _shape_param_281);  clone_51 = _shape_param_281 = None
        view_282: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_281, _shape_param_282);  view_281 = _shape_param_282 = None
        permute_140: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_75: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg204_1, view_282, permute_140);  arg204_1 = view_282 = permute_140 = None
        view_283: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_75, _shape_param_283);  addmm_75 = _shape_param_283 = None
        add_103: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_99, view_283);  add_99 = view_283 = None
        convert_element_type_100: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_103, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_100, [2], correction = 0, keepdim = True)
        getitem_50: "f32[32, 128, 1]" = var_mean_25[0]
        getitem_51: "f32[32, 128, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_38: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_100, getitem_51);  convert_element_type_100 = getitem_51 = None
        add_104: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_25: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_100: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
        mul_101: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_100, arg205_1);  mul_100 = arg205_1 = None
        add_105: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_101, arg206_1);  mul_101 = arg206_1 = None
        convert_element_type_101: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None
        view_284: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_284);  convert_element_type_101 = _shape_param_284 = None
        permute_141: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_76: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg208_1, view_284, permute_141);  arg208_1 = view_284 = permute_141 = None
        view_285: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_76, _shape_param_285);  addmm_76 = _shape_param_285 = None
        convert_element_type_102: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        mul_102: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_102, 0.5)
        mul_103: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_102, 0.7071067811865476);  convert_element_type_102 = None
        erf_12: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_106: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_104: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_102, add_106);  mul_102 = add_106 = None
        convert_element_type_103: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None
        view_286: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_103, _shape_param_286);  convert_element_type_103 = _shape_param_286 = None
        permute_142: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_77: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg210_1, view_286, permute_142);  arg210_1 = view_286 = permute_142 = None
        view_287: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_77, _shape_param_287);  addmm_77 = _shape_param_287 = None
        add_107: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_103, view_287);  add_103 = view_287 = None
        convert_element_type_104: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_104, [2], correction = 0, keepdim = True)
        getitem_52: "f32[32, 128, 1]" = var_mean_26[0]
        getitem_53: "f32[32, 128, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_39: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_104, getitem_53);  convert_element_type_104 = getitem_53 = None
        add_108: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
        rsqrt_26: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_105: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
        mul_106: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_105, arg211_1);  mul_105 = arg211_1 = None
        add_109: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_106, arg212_1);  mul_106 = arg212_1 = None
        convert_element_type_105: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None
        view_288: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_105, _shape_param_288);  _shape_param_288 = None
        permute_143: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_78: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg214_1, view_288, permute_143);  arg214_1 = view_288 = permute_143 = None
        view_289: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_78, _shape_param_289);  addmm_78 = _shape_param_289 = None
        mul_107: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_289, 0.125);  view_289 = None
        view_290: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_107, _shape_param_290);  mul_107 = _shape_param_290 = None
        permute_144: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        clone_52: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_291: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_291);  clone_52 = _shape_param_291 = None
        view_292: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_105, _shape_param_292);  _shape_param_292 = None
        permute_145: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
        addmm_79: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg216_1, view_292, permute_145);  arg216_1 = view_292 = permute_145 = None
        view_293: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_79, _shape_param_293);  addmm_79 = _shape_param_293 = None
        view_294: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_293, _shape_param_294);  view_293 = _shape_param_294 = None
        permute_146: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        clone_53: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_295: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_53, _shape_param_295);  clone_53 = _shape_param_295 = None
        permute_147: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
        bmm_26: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_291, permute_147);  view_291 = permute_147 = None
        view_296: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_296);  bmm_26 = _shape_param_296 = None
        add_110: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_296, where);  view_296 = None
        full_15: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_13: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_110, full_15);  add_110 = full_15 = None
        view_297: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_13, _shape_param_297);  maximum_13 = _shape_param_297 = None
        convert_element_type_106: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_297, torch.float32);  view_297 = None
        amax_13: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_106, [-1], True)
        sub_40: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_106, amax_13);  convert_element_type_106 = amax_13 = None
        exp_13: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_107: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        view_298: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_105, _shape_param_298);  convert_element_type_105 = _shape_param_298 = None
        permute_148: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_80: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg218_1, view_298, permute_148);  arg218_1 = view_298 = permute_148 = None
        view_299: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_80, _shape_param_299);  addmm_80 = _shape_param_299 = None
        view_300: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_299, _shape_param_300);  view_299 = _shape_param_300 = None
        permute_149: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_54: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        view_301: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_54, _shape_param_301);  clone_54 = _shape_param_301 = None
        bmm_27: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_107, view_301);  convert_element_type_107 = view_301 = None
        view_302: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_302);  bmm_27 = _shape_param_302 = None
        permute_150: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        clone_55: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_303: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_55, _shape_param_303);  clone_55 = _shape_param_303 = None
        view_304: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_303, _shape_param_304);  view_303 = _shape_param_304 = None
        permute_151: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_81: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg220_1, view_304, permute_151);  arg220_1 = view_304 = permute_151 = None
        view_305: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_81, _shape_param_305);  addmm_81 = _shape_param_305 = None
        add_111: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_107, view_305);  add_107 = view_305 = None
        convert_element_type_108: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_111, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_108, [2], correction = 0, keepdim = True)
        getitem_54: "f32[32, 128, 1]" = var_mean_27[0]
        getitem_55: "f32[32, 128, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_41: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_108, getitem_55);  convert_element_type_108 = getitem_55 = None
        add_112: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
        rsqrt_27: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        mul_108: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
        mul_109: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_108, arg221_1);  mul_108 = arg221_1 = None
        add_113: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_109, arg222_1);  mul_109 = arg222_1 = None
        convert_element_type_109: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_113, torch.bfloat16);  add_113 = None
        view_306: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_109, _shape_param_306);  convert_element_type_109 = _shape_param_306 = None
        permute_152: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_82: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg224_1, view_306, permute_152);  arg224_1 = view_306 = permute_152 = None
        view_307: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_82, _shape_param_307);  addmm_82 = _shape_param_307 = None
        convert_element_type_110: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        mul_110: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_110, 0.5)
        mul_111: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_110, 0.7071067811865476);  convert_element_type_110 = None
        erf_13: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
        add_114: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_112: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, add_114);  mul_110 = add_114 = None
        convert_element_type_111: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        view_308: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_111, _shape_param_308);  convert_element_type_111 = _shape_param_308 = None
        permute_153: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_83: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg226_1, view_308, permute_153);  arg226_1 = view_308 = permute_153 = None
        view_309: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_83, _shape_param_309);  addmm_83 = _shape_param_309 = None
        add_115: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_111, view_309);  add_111 = view_309 = None
        convert_element_type_112: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_112, [2], correction = 0, keepdim = True)
        getitem_56: "f32[32, 128, 1]" = var_mean_28[0]
        getitem_57: "f32[32, 128, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_42: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_112, getitem_57);  convert_element_type_112 = getitem_57 = None
        add_116: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_28: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_113: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
        mul_114: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_113, arg227_1);  mul_113 = arg227_1 = None
        add_117: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_114, arg228_1);  mul_114 = arg228_1 = None
        convert_element_type_113: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None
        view_310: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_113, _shape_param_310);  _shape_param_310 = None
        permute_154: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_84: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg230_1, view_310, permute_154);  arg230_1 = view_310 = permute_154 = None
        view_311: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_84, _shape_param_311);  addmm_84 = _shape_param_311 = None
        mul_115: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_311, 0.125);  view_311 = None
        view_312: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_115, _shape_param_312);  mul_115 = _shape_param_312 = None
        permute_155: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
        clone_56: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_313: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_313);  clone_56 = _shape_param_313 = None
        view_314: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_113, _shape_param_314);  _shape_param_314 = None
        permute_156: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_85: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg232_1, view_314, permute_156);  arg232_1 = view_314 = permute_156 = None
        view_315: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_85, _shape_param_315);  addmm_85 = _shape_param_315 = None
        view_316: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_315, _shape_param_316);  view_315 = _shape_param_316 = None
        permute_157: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        clone_57: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_317: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_57, _shape_param_317);  clone_57 = _shape_param_317 = None
        permute_158: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
        bmm_28: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_313, permute_158);  view_313 = permute_158 = None
        view_318: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_318);  bmm_28 = _shape_param_318 = None
        add_118: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_318, where);  view_318 = None
        full_16: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_118, full_16);  add_118 = full_16 = None
        view_319: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_14, _shape_param_319);  maximum_14 = _shape_param_319 = None
        convert_element_type_114: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_319, torch.float32);  view_319 = None
        amax_14: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_114, [-1], True)
        sub_43: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_114, amax_14);  convert_element_type_114 = amax_14 = None
        exp_14: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_115: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        view_320: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_113, _shape_param_320);  convert_element_type_113 = _shape_param_320 = None
        permute_159: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_86: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg234_1, view_320, permute_159);  arg234_1 = view_320 = permute_159 = None
        view_321: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_86, _shape_param_321);  addmm_86 = _shape_param_321 = None
        view_322: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_321, _shape_param_322);  view_321 = _shape_param_322 = None
        permute_160: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        clone_58: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_323: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_323);  clone_58 = _shape_param_323 = None
        bmm_29: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_115, view_323);  convert_element_type_115 = view_323 = None
        view_324: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_29, _shape_param_324);  bmm_29 = _shape_param_324 = None
        permute_161: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
        clone_59: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_325: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_59, _shape_param_325);  clone_59 = _shape_param_325 = None
        view_326: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_325, _shape_param_326);  view_325 = _shape_param_326 = None
        permute_162: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_87: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg236_1, view_326, permute_162);  arg236_1 = view_326 = permute_162 = None
        view_327: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_87, _shape_param_327);  addmm_87 = _shape_param_327 = None
        add_119: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_115, view_327);  add_115 = view_327 = None
        convert_element_type_116: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_119, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_116, [2], correction = 0, keepdim = True)
        getitem_58: "f32[32, 128, 1]" = var_mean_29[0]
        getitem_59: "f32[32, 128, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_44: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_116, getitem_59);  convert_element_type_116 = getitem_59 = None
        add_120: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_29: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_116: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
        mul_117: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_116, arg237_1);  mul_116 = arg237_1 = None
        add_121: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_117, arg238_1);  mul_117 = arg238_1 = None
        convert_element_type_117: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None
        view_328: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_117, _shape_param_328);  convert_element_type_117 = _shape_param_328 = None
        permute_163: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_88: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg240_1, view_328, permute_163);  arg240_1 = view_328 = permute_163 = None
        view_329: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_88, _shape_param_329);  addmm_88 = _shape_param_329 = None
        convert_element_type_118: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_329, torch.float32);  view_329 = None
        mul_118: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_118, 0.5)
        mul_119: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_118, 0.7071067811865476);  convert_element_type_118 = None
        erf_14: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_122: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_120: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_118, add_122);  mul_118 = add_122 = None
        convert_element_type_119: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        view_330: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_119, _shape_param_330);  convert_element_type_119 = _shape_param_330 = None
        permute_164: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_89: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg242_1, view_330, permute_164);  arg242_1 = view_330 = permute_164 = None
        view_331: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_89, _shape_param_331);  addmm_89 = _shape_param_331 = None
        add_123: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_119, view_331);  add_119 = view_331 = None
        convert_element_type_120: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_123, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_60: "f32[32, 128, 1]" = var_mean_30[0]
        getitem_61: "f32[32, 128, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_45: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_61);  convert_element_type_120 = getitem_61 = None
        add_124: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_30: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
        mul_121: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
        mul_122: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_121, arg243_1);  mul_121 = arg243_1 = None
        add_125: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_122, arg244_1);  mul_122 = arg244_1 = None
        convert_element_type_121: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None
        view_332: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_121, _shape_param_332);  _shape_param_332 = None
        permute_165: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_90: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg246_1, view_332, permute_165);  arg246_1 = view_332 = permute_165 = None
        view_333: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_90, _shape_param_333);  addmm_90 = _shape_param_333 = None
        mul_123: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_333, 0.125);  view_333 = None
        view_334: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_123, _shape_param_334);  mul_123 = _shape_param_334 = None
        permute_166: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
        clone_60: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_335: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_335);  clone_60 = _shape_param_335 = None
        view_336: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_121, _shape_param_336);  _shape_param_336 = None
        permute_167: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_91: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg248_1, view_336, permute_167);  arg248_1 = view_336 = permute_167 = None
        view_337: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_91, _shape_param_337);  addmm_91 = _shape_param_337 = None
        view_338: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_337, _shape_param_338);  view_337 = _shape_param_338 = None
        permute_168: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_61: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_339: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_61, _shape_param_339);  clone_61 = _shape_param_339 = None
        permute_169: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_339, [0, 2, 1]);  view_339 = None
        bmm_30: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_335, permute_169);  view_335 = permute_169 = None
        view_340: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_340);  bmm_30 = _shape_param_340 = None
        add_126: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_340, where);  view_340 = None
        full_17: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_126, full_17);  add_126 = full_17 = None
        view_341: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_15, _shape_param_341);  maximum_15 = _shape_param_341 = None
        convert_element_type_122: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_341, torch.float32);  view_341 = None
        amax_15: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_122, [-1], True)
        sub_46: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_122, amax_15);  convert_element_type_122 = amax_15 = None
        exp_15: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_123: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        view_342: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_121, _shape_param_342);  convert_element_type_121 = _shape_param_342 = None
        permute_170: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_92: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg250_1, view_342, permute_170);  arg250_1 = view_342 = permute_170 = None
        view_343: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_92, _shape_param_343);  addmm_92 = _shape_param_343 = None
        view_344: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_343, _shape_param_344);  view_343 = _shape_param_344 = None
        permute_171: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_62: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_345: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_62, _shape_param_345);  clone_62 = _shape_param_345 = None
        bmm_31: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_123, view_345);  convert_element_type_123 = view_345 = None
        view_346: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_346);  bmm_31 = _shape_param_346 = None
        permute_172: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        clone_63: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_347: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_63, _shape_param_347);  clone_63 = _shape_param_347 = None
        view_348: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_347, _shape_param_348);  view_347 = _shape_param_348 = None
        permute_173: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_93: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg252_1, view_348, permute_173);  arg252_1 = view_348 = permute_173 = None
        view_349: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_93, _shape_param_349);  addmm_93 = _shape_param_349 = None
        add_127: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_123, view_349);  add_123 = view_349 = None
        convert_element_type_124: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_127, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_124, [2], correction = 0, keepdim = True)
        getitem_62: "f32[32, 128, 1]" = var_mean_31[0]
        getitem_63: "f32[32, 128, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_47: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_124, getitem_63);  convert_element_type_124 = getitem_63 = None
        add_128: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
        rsqrt_31: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        mul_124: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
        mul_125: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_124, arg253_1);  mul_124 = arg253_1 = None
        add_129: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_125, arg254_1);  mul_125 = arg254_1 = None
        convert_element_type_125: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None
        view_350: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_125, _shape_param_350);  convert_element_type_125 = _shape_param_350 = None
        permute_174: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_94: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg256_1, view_350, permute_174);  arg256_1 = view_350 = permute_174 = None
        view_351: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_94, _shape_param_351);  addmm_94 = _shape_param_351 = None
        convert_element_type_126: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_351, torch.float32);  view_351 = None
        mul_126: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.5)
        mul_127: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.7071067811865476);  convert_element_type_126 = None
        erf_15: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_127);  mul_127 = None
        add_130: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_128: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_126, add_130);  mul_126 = add_130 = None
        convert_element_type_127: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_128, torch.bfloat16);  mul_128 = None
        view_352: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_352);  convert_element_type_127 = _shape_param_352 = None
        permute_175: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_95: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg258_1, view_352, permute_175);  arg258_1 = view_352 = permute_175 = None
        view_353: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_95, _shape_param_353);  addmm_95 = _shape_param_353 = None
        add_131: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_127, view_353);  add_127 = view_353 = None
        convert_element_type_128: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_131, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_128, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 128, 1]" = var_mean_32[0]
        getitem_65: "f32[32, 128, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_48: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_128, getitem_65);  convert_element_type_128 = getitem_65 = None
        add_132: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
        rsqrt_32: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
        mul_129: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
        mul_130: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_129, arg259_1);  mul_129 = arg259_1 = None
        add_133: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_130, arg260_1);  mul_130 = arg260_1 = None
        convert_element_type_129: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None
        view_354: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_129, _shape_param_354);  _shape_param_354 = None
        permute_176: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_96: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg262_1, view_354, permute_176);  arg262_1 = view_354 = permute_176 = None
        view_355: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_96, _shape_param_355);  addmm_96 = _shape_param_355 = None
        mul_131: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_355, 0.125);  view_355 = None
        view_356: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_131, _shape_param_356);  mul_131 = _shape_param_356 = None
        permute_177: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        clone_64: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_357: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_357);  clone_64 = _shape_param_357 = None
        view_358: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_129, _shape_param_358);  _shape_param_358 = None
        permute_178: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_97: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg264_1, view_358, permute_178);  arg264_1 = view_358 = permute_178 = None
        view_359: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_97, _shape_param_359);  addmm_97 = _shape_param_359 = None
        view_360: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_359, _shape_param_360);  view_359 = _shape_param_360 = None
        permute_179: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        clone_65: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_361: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_65, _shape_param_361);  clone_65 = _shape_param_361 = None
        permute_180: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_361, [0, 2, 1]);  view_361 = None
        bmm_32: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_357, permute_180);  view_357 = permute_180 = None
        view_362: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_362);  bmm_32 = _shape_param_362 = None
        add_134: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_362, where);  view_362 = None
        full_18: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_16: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_134, full_18);  add_134 = full_18 = None
        view_363: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_16, _shape_param_363);  maximum_16 = _shape_param_363 = None
        convert_element_type_130: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None
        amax_16: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_130, [-1], True)
        sub_49: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_130, amax_16);  convert_element_type_130 = amax_16 = None
        exp_16: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_131: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        view_364: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_129, _shape_param_364);  convert_element_type_129 = _shape_param_364 = None
        permute_181: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_98: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg266_1, view_364, permute_181);  arg266_1 = view_364 = permute_181 = None
        view_365: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_98, _shape_param_365);  addmm_98 = _shape_param_365 = None
        view_366: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_365, _shape_param_366);  view_365 = _shape_param_366 = None
        permute_182: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        clone_66: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_367: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_367);  clone_66 = _shape_param_367 = None
        bmm_33: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_131, view_367);  convert_element_type_131 = view_367 = None
        view_368: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_33, _shape_param_368);  bmm_33 = _shape_param_368 = None
        permute_183: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        clone_67: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_369: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_67, _shape_param_369);  clone_67 = _shape_param_369 = None
        view_370: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_369, _shape_param_370);  view_369 = _shape_param_370 = None
        permute_184: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_99: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg268_1, view_370, permute_184);  arg268_1 = view_370 = permute_184 = None
        view_371: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_99, _shape_param_371);  addmm_99 = _shape_param_371 = None
        add_135: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_131, view_371);  add_131 = view_371 = None
        convert_element_type_132: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_135, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_132, [2], correction = 0, keepdim = True)
        getitem_66: "f32[32, 128, 1]" = var_mean_33[0]
        getitem_67: "f32[32, 128, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_50: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_132, getitem_67);  convert_element_type_132 = getitem_67 = None
        add_136: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_33: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        mul_132: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
        mul_133: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_132, arg269_1);  mul_132 = arg269_1 = None
        add_137: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_133, arg270_1);  mul_133 = arg270_1 = None
        convert_element_type_133: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_137, torch.bfloat16);  add_137 = None
        view_372: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_133, _shape_param_372);  convert_element_type_133 = _shape_param_372 = None
        permute_185: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_100: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg272_1, view_372, permute_185);  arg272_1 = view_372 = permute_185 = None
        view_373: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_100, _shape_param_373);  addmm_100 = _shape_param_373 = None
        convert_element_type_134: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_373, torch.float32);  view_373 = None
        mul_134: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_134, 0.5)
        mul_135: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_134, 0.7071067811865476);  convert_element_type_134 = None
        erf_16: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
        add_138: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_136: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_134, add_138);  mul_134 = add_138 = None
        convert_element_type_135: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None
        view_374: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_135, _shape_param_374);  convert_element_type_135 = _shape_param_374 = None
        permute_186: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_101: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg274_1, view_374, permute_186);  arg274_1 = view_374 = permute_186 = None
        view_375: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_101, _shape_param_375);  addmm_101 = _shape_param_375 = None
        add_139: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_135, view_375);  add_135 = view_375 = None
        convert_element_type_136: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_139, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_136, [2], correction = 0, keepdim = True)
        getitem_68: "f32[32, 128, 1]" = var_mean_34[0]
        getitem_69: "f32[32, 128, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_51: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_136, getitem_69);  convert_element_type_136 = getitem_69 = None
        add_140: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_34: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_137: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
        mul_138: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_137, arg275_1);  mul_137 = arg275_1 = None
        add_141: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_138, arg276_1);  mul_138 = arg276_1 = None
        convert_element_type_137: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16);  add_141 = None
        view_376: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_137, _shape_param_376);  _shape_param_376 = None
        permute_187: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_102: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg278_1, view_376, permute_187);  arg278_1 = view_376 = permute_187 = None
        view_377: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_102, _shape_param_377);  addmm_102 = _shape_param_377 = None
        mul_139: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_377, 0.125);  view_377 = None
        view_378: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_139, _shape_param_378);  mul_139 = _shape_param_378 = None
        permute_188: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        clone_68: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_379: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_379);  clone_68 = _shape_param_379 = None
        view_380: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_137, _shape_param_380);  _shape_param_380 = None
        permute_189: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_103: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg280_1, view_380, permute_189);  arg280_1 = view_380 = permute_189 = None
        view_381: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_103, _shape_param_381);  addmm_103 = _shape_param_381 = None
        view_382: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_381, _shape_param_382);  view_381 = _shape_param_382 = None
        permute_190: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        clone_69: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_383: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_69, _shape_param_383);  clone_69 = _shape_param_383 = None
        permute_191: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
        bmm_34: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_379, permute_191);  view_379 = permute_191 = None
        view_384: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_384);  bmm_34 = _shape_param_384 = None
        add_142: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_384, where);  view_384 = None
        full_19: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_17: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_142, full_19);  add_142 = full_19 = None
        view_385: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_17, _shape_param_385);  maximum_17 = _shape_param_385 = None
        convert_element_type_138: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_385, torch.float32);  view_385 = None
        amax_17: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_138, [-1], True)
        sub_52: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_138, amax_17);  convert_element_type_138 = amax_17 = None
        exp_17: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_139: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        view_386: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_137, _shape_param_386);  convert_element_type_137 = _shape_param_386 = None
        permute_192: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_104: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg282_1, view_386, permute_192);  arg282_1 = view_386 = permute_192 = None
        view_387: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_104, _shape_param_387);  addmm_104 = _shape_param_387 = None
        view_388: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_387, _shape_param_388);  view_387 = _shape_param_388 = None
        permute_193: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        clone_70: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_389: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_389);  clone_70 = _shape_param_389 = None
        bmm_35: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_139, view_389);  convert_element_type_139 = view_389 = None
        view_390: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_390);  bmm_35 = _shape_param_390 = None
        permute_194: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
        clone_71: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_391: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_71, _shape_param_391);  clone_71 = _shape_param_391 = None
        view_392: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_391, _shape_param_392);  view_391 = _shape_param_392 = None
        permute_195: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_105: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg284_1, view_392, permute_195);  arg284_1 = view_392 = permute_195 = None
        view_393: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_105, _shape_param_393);  addmm_105 = _shape_param_393 = None
        add_143: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_139, view_393);  add_139 = view_393 = None
        convert_element_type_140: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_143, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_140, [2], correction = 0, keepdim = True)
        getitem_70: "f32[32, 128, 1]" = var_mean_35[0]
        getitem_71: "f32[32, 128, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_53: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_140, getitem_71);  convert_element_type_140 = getitem_71 = None
        add_144: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_35: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_140: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
        mul_141: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_140, arg285_1);  mul_140 = arg285_1 = None
        add_145: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_141, arg286_1);  mul_141 = arg286_1 = None
        convert_element_type_141: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16);  add_145 = None
        view_394: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_141, _shape_param_394);  convert_element_type_141 = _shape_param_394 = None
        permute_196: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_106: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg288_1, view_394, permute_196);  arg288_1 = view_394 = permute_196 = None
        view_395: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_106, _shape_param_395);  addmm_106 = _shape_param_395 = None
        convert_element_type_142: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_395, torch.float32);  view_395 = None
        mul_142: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_142, 0.5)
        mul_143: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_142, 0.7071067811865476);  convert_element_type_142 = None
        erf_17: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
        add_146: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_144: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_142, add_146);  mul_142 = add_146 = None
        convert_element_type_143: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_144, torch.bfloat16);  mul_144 = None
        view_396: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_143, _shape_param_396);  convert_element_type_143 = _shape_param_396 = None
        permute_197: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_107: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg290_1, view_396, permute_197);  arg290_1 = view_396 = permute_197 = None
        view_397: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_107, _shape_param_397);  addmm_107 = _shape_param_397 = None
        add_147: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_143, view_397);  add_143 = view_397 = None
        convert_element_type_144: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_147, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_72: "f32[32, 128, 1]" = var_mean_36[0]
        getitem_73: "f32[32, 128, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_54: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_73);  convert_element_type_144 = getitem_73 = None
        add_148: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
        rsqrt_36: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_145: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
        mul_146: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_145, arg291_1);  mul_145 = arg291_1 = None
        add_149: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_146, arg292_1);  mul_146 = arg292_1 = None
        convert_element_type_145: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None
        view_398: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_145, _shape_param_398);  _shape_param_398 = None
        permute_198: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_108: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg294_1, view_398, permute_198);  arg294_1 = view_398 = permute_198 = None
        view_399: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_108, _shape_param_399);  addmm_108 = _shape_param_399 = None
        mul_147: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_399, 0.125);  view_399 = None
        view_400: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_147, _shape_param_400);  mul_147 = _shape_param_400 = None
        permute_199: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
        clone_72: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_401: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_401);  clone_72 = _shape_param_401 = None
        view_402: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_145, _shape_param_402);  _shape_param_402 = None
        permute_200: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_109: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg296_1, view_402, permute_200);  arg296_1 = view_402 = permute_200 = None
        view_403: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_109, _shape_param_403);  addmm_109 = _shape_param_403 = None
        view_404: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_403, _shape_param_404);  view_403 = _shape_param_404 = None
        permute_201: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        clone_73: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_405: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_73, _shape_param_405);  clone_73 = _shape_param_405 = None
        permute_202: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1]);  view_405 = None
        bmm_36: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_401, permute_202);  view_401 = permute_202 = None
        view_406: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_406);  bmm_36 = _shape_param_406 = None
        add_150: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_406, where);  view_406 = None
        full_20: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_18: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_150, full_20);  add_150 = full_20 = None
        view_407: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_18, _shape_param_407);  maximum_18 = _shape_param_407 = None
        convert_element_type_146: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        amax_18: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_146, [-1], True)
        sub_55: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_146, amax_18);  convert_element_type_146 = amax_18 = None
        exp_18: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_147: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        view_408: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_145, _shape_param_408);  convert_element_type_145 = _shape_param_408 = None
        permute_203: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_110: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg298_1, view_408, permute_203);  arg298_1 = view_408 = permute_203 = None
        view_409: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_110, _shape_param_409);  addmm_110 = _shape_param_409 = None
        view_410: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_409, _shape_param_410);  view_409 = _shape_param_410 = None
        permute_204: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_74: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_411: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_74, _shape_param_411);  clone_74 = _shape_param_411 = None
        bmm_37: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_147, view_411);  convert_element_type_147 = view_411 = None
        view_412: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_37, _shape_param_412);  bmm_37 = _shape_param_412 = None
        permute_205: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        clone_75: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_413: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_75, _shape_param_413);  clone_75 = _shape_param_413 = None
        view_414: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_413, _shape_param_414);  view_413 = _shape_param_414 = None
        permute_206: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
        addmm_111: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg300_1, view_414, permute_206);  arg300_1 = view_414 = permute_206 = None
        view_415: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_111, _shape_param_415);  addmm_111 = _shape_param_415 = None
        add_151: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_147, view_415);  add_147 = view_415 = None
        convert_element_type_148: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_151, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_148, [2], correction = 0, keepdim = True)
        getitem_74: "f32[32, 128, 1]" = var_mean_37[0]
        getitem_75: "f32[32, 128, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_56: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_148, getitem_75);  convert_element_type_148 = getitem_75 = None
        add_152: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
        rsqrt_37: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_148: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
        mul_149: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_148, arg301_1);  mul_148 = arg301_1 = None
        add_153: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_149, arg302_1);  mul_149 = arg302_1 = None
        convert_element_type_149: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16);  add_153 = None
        view_416: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_149, _shape_param_416);  convert_element_type_149 = _shape_param_416 = None
        permute_207: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_112: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg304_1, view_416, permute_207);  arg304_1 = view_416 = permute_207 = None
        view_417: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_112, _shape_param_417);  addmm_112 = _shape_param_417 = None
        convert_element_type_150: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None
        mul_150: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_150, 0.5)
        mul_151: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_150, 0.7071067811865476);  convert_element_type_150 = None
        erf_18: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
        add_154: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_152: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, add_154);  mul_150 = add_154 = None
        convert_element_type_151: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        view_418: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_151, _shape_param_418);  convert_element_type_151 = _shape_param_418 = None
        permute_208: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_113: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg306_1, view_418, permute_208);  arg306_1 = view_418 = permute_208 = None
        view_419: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_113, _shape_param_419);  addmm_113 = _shape_param_419 = None
        add_155: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_151, view_419);  add_151 = view_419 = None
        convert_element_type_152: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_155, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_152, [2], correction = 0, keepdim = True)
        getitem_76: "f32[32, 128, 1]" = var_mean_38[0]
        getitem_77: "f32[32, 128, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_57: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_152, getitem_77);  convert_element_type_152 = getitem_77 = None
        add_156: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_38: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        mul_153: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
        mul_154: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_153, arg307_1);  mul_153 = arg307_1 = None
        add_157: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_154, arg308_1);  mul_154 = arg308_1 = None
        convert_element_type_153: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16);  add_157 = None
        view_420: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_153, _shape_param_420);  _shape_param_420 = None
        permute_209: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_114: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg310_1, view_420, permute_209);  arg310_1 = view_420 = permute_209 = None
        view_421: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_114, _shape_param_421);  addmm_114 = _shape_param_421 = None
        mul_155: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_421, 0.125);  view_421 = None
        view_422: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_155, _shape_param_422);  mul_155 = _shape_param_422 = None
        permute_210: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
        clone_76: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_423: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_423);  clone_76 = _shape_param_423 = None
        view_424: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_153, _shape_param_424);  _shape_param_424 = None
        permute_211: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_115: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg312_1, view_424, permute_211);  arg312_1 = view_424 = permute_211 = None
        view_425: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_115, _shape_param_425);  addmm_115 = _shape_param_425 = None
        view_426: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_425, _shape_param_426);  view_425 = _shape_param_426 = None
        permute_212: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
        clone_77: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_427: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_77, _shape_param_427);  clone_77 = _shape_param_427 = None
        permute_213: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
        bmm_38: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_423, permute_213);  view_423 = permute_213 = None
        view_428: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_428);  bmm_38 = _shape_param_428 = None
        add_158: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_428, where);  view_428 = None
        full_21: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_19: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_158, full_21);  add_158 = full_21 = None
        view_429: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_19, _shape_param_429);  maximum_19 = _shape_param_429 = None
        convert_element_type_154: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_429, torch.float32);  view_429 = None
        amax_19: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_154, [-1], True)
        sub_58: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_154, amax_19);  convert_element_type_154 = amax_19 = None
        exp_19: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_155: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        view_430: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_153, _shape_param_430);  convert_element_type_153 = _shape_param_430 = None
        permute_214: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_116: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg314_1, view_430, permute_214);  arg314_1 = view_430 = permute_214 = None
        view_431: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_116, _shape_param_431);  addmm_116 = _shape_param_431 = None
        view_432: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_431, _shape_param_432);  view_431 = _shape_param_432 = None
        permute_215: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        clone_78: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_433: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_78, _shape_param_433);  clone_78 = _shape_param_433 = None
        bmm_39: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_155, view_433);  convert_element_type_155 = view_433 = None
        view_434: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_434);  bmm_39 = _shape_param_434 = None
        permute_216: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
        clone_79: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_435: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_79, _shape_param_435);  clone_79 = _shape_param_435 = None
        view_436: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_435, _shape_param_436);  view_435 = _shape_param_436 = None
        permute_217: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_117: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg316_1, view_436, permute_217);  arg316_1 = view_436 = permute_217 = None
        view_437: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_117, _shape_param_437);  addmm_117 = _shape_param_437 = None
        add_159: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_155, view_437);  add_155 = view_437 = None
        convert_element_type_156: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_159, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_156, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 128, 1]" = var_mean_39[0]
        getitem_79: "f32[32, 128, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_59: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_156, getitem_79);  convert_element_type_156 = getitem_79 = None
        add_160: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_39: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
        mul_156: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
        mul_157: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_156, arg317_1);  mul_156 = arg317_1 = None
        add_161: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_157, arg318_1);  mul_157 = arg318_1 = None
        convert_element_type_157: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_161, torch.bfloat16);  add_161 = None
        view_438: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_157, _shape_param_438);  convert_element_type_157 = _shape_param_438 = None
        permute_218: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_118: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg320_1, view_438, permute_218);  arg320_1 = view_438 = permute_218 = None
        view_439: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_118, _shape_param_439);  addmm_118 = _shape_param_439 = None
        convert_element_type_158: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_439, torch.float32);  view_439 = None
        mul_158: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_158, 0.5)
        mul_159: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_158, 0.7071067811865476);  convert_element_type_158 = None
        erf_19: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
        add_162: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_160: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_158, add_162);  mul_158 = add_162 = None
        convert_element_type_159: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_160, torch.bfloat16);  mul_160 = None
        view_440: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_159, _shape_param_440);  convert_element_type_159 = _shape_param_440 = None
        permute_219: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_119: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg322_1, view_440, permute_219);  arg322_1 = view_440 = permute_219 = None
        view_441: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_119, _shape_param_441);  addmm_119 = _shape_param_441 = None
        add_163: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_159, view_441);  add_159 = view_441 = None
        convert_element_type_160: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_163, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_160, [2], correction = 0, keepdim = True)
        getitem_80: "f32[32, 128, 1]" = var_mean_40[0]
        getitem_81: "f32[32, 128, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_60: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_160, getitem_81);  convert_element_type_160 = getitem_81 = None
        add_164: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_40: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
        mul_161: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
        mul_162: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_161, arg323_1);  mul_161 = arg323_1 = None
        add_165: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_162, arg324_1);  mul_162 = arg324_1 = None
        convert_element_type_161: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16);  add_165 = None
        view_442: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_161, _shape_param_442);  _shape_param_442 = None
        permute_220: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_120: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg326_1, view_442, permute_220);  arg326_1 = view_442 = permute_220 = None
        view_443: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_120, _shape_param_443);  addmm_120 = _shape_param_443 = None
        mul_163: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_443, 0.125);  view_443 = None
        view_444: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_163, _shape_param_444);  mul_163 = _shape_param_444 = None
        permute_221: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
        clone_80: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_445: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_80, _shape_param_445);  clone_80 = _shape_param_445 = None
        view_446: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_161, _shape_param_446);  _shape_param_446 = None
        permute_222: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_121: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg328_1, view_446, permute_222);  arg328_1 = view_446 = permute_222 = None
        view_447: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_121, _shape_param_447);  addmm_121 = _shape_param_447 = None
        view_448: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_447, _shape_param_448);  view_447 = _shape_param_448 = None
        permute_223: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        clone_81: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_449: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_81, _shape_param_449);  clone_81 = _shape_param_449 = None
        permute_224: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
        bmm_40: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_445, permute_224);  view_445 = permute_224 = None
        view_450: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_450);  bmm_40 = _shape_param_450 = None
        add_166: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_450, where);  view_450 = None
        full_22: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_20: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_166, full_22);  add_166 = full_22 = None
        view_451: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_20, _shape_param_451);  maximum_20 = _shape_param_451 = None
        convert_element_type_162: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_451, torch.float32);  view_451 = None
        amax_20: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_162, [-1], True)
        sub_61: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_162, amax_20);  convert_element_type_162 = amax_20 = None
        exp_20: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_163: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        view_452: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_161, _shape_param_452);  convert_element_type_161 = _shape_param_452 = None
        permute_225: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_122: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg330_1, view_452, permute_225);  arg330_1 = view_452 = permute_225 = None
        view_453: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_122, _shape_param_453);  addmm_122 = _shape_param_453 = None
        view_454: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_453, _shape_param_454);  view_453 = _shape_param_454 = None
        permute_226: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        clone_82: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_455: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_82, _shape_param_455);  clone_82 = _shape_param_455 = None
        bmm_41: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_163, view_455);  convert_element_type_163 = view_455 = None
        view_456: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_41, _shape_param_456);  bmm_41 = _shape_param_456 = None
        permute_227: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_83: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_457: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_83, _shape_param_457);  clone_83 = _shape_param_457 = None
        view_458: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_457, _shape_param_458);  view_457 = _shape_param_458 = None
        permute_228: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_123: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg332_1, view_458, permute_228);  arg332_1 = view_458 = permute_228 = None
        view_459: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_123, _shape_param_459);  addmm_123 = _shape_param_459 = None
        add_167: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_163, view_459);  add_163 = view_459 = None
        convert_element_type_164: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_167, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_164, [2], correction = 0, keepdim = True)
        getitem_82: "f32[32, 128, 1]" = var_mean_41[0]
        getitem_83: "f32[32, 128, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_62: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_164, getitem_83);  convert_element_type_164 = getitem_83 = None
        add_168: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
        rsqrt_41: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
        mul_164: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
        mul_165: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_164, arg333_1);  mul_164 = arg333_1 = None
        add_169: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_165, arg334_1);  mul_165 = arg334_1 = None
        convert_element_type_165: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16);  add_169 = None
        view_460: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_165, _shape_param_460);  convert_element_type_165 = _shape_param_460 = None
        permute_229: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_124: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg336_1, view_460, permute_229);  arg336_1 = view_460 = permute_229 = None
        view_461: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_124, _shape_param_461);  addmm_124 = _shape_param_461 = None
        convert_element_type_166: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_461, torch.float32);  view_461 = None
        mul_166: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_166, 0.5)
        mul_167: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_166, 0.7071067811865476);  convert_element_type_166 = None
        erf_20: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_170: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_168: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_166, add_170);  mul_166 = add_170 = None
        convert_element_type_167: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        view_462: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_167, _shape_param_462);  convert_element_type_167 = _shape_param_462 = None
        permute_230: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_125: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg338_1, view_462, permute_230);  arg338_1 = view_462 = permute_230 = None
        view_463: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_125, _shape_param_463);  addmm_125 = _shape_param_463 = None
        add_171: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_167, view_463);  add_167 = view_463 = None
        convert_element_type_168: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_171, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_168, [2], correction = 0, keepdim = True)
        getitem_84: "f32[32, 128, 1]" = var_mean_42[0]
        getitem_85: "f32[32, 128, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_63: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_168, getitem_85);  convert_element_type_168 = getitem_85 = None
        add_172: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
        rsqrt_42: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        mul_169: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
        mul_170: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_169, arg339_1);  mul_169 = arg339_1 = None
        add_173: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_170, arg340_1);  mul_170 = arg340_1 = None
        convert_element_type_169: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16);  add_173 = None
        view_464: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_169, _shape_param_464);  _shape_param_464 = None
        permute_231: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_126: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg342_1, view_464, permute_231);  arg342_1 = view_464 = permute_231 = None
        view_465: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_126, _shape_param_465);  addmm_126 = _shape_param_465 = None
        mul_171: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_465, 0.125);  view_465 = None
        view_466: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_171, _shape_param_466);  mul_171 = _shape_param_466 = None
        permute_232: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None
        clone_84: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_467: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_84, _shape_param_467);  clone_84 = _shape_param_467 = None
        view_468: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_169, _shape_param_468);  _shape_param_468 = None
        permute_233: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_127: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg344_1, view_468, permute_233);  arg344_1 = view_468 = permute_233 = None
        view_469: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_127, _shape_param_469);  addmm_127 = _shape_param_469 = None
        view_470: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_469, _shape_param_470);  view_469 = _shape_param_470 = None
        permute_234: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        clone_85: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_471: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_85, _shape_param_471);  clone_85 = _shape_param_471 = None
        permute_235: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_471, [0, 2, 1]);  view_471 = None
        bmm_42: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_467, permute_235);  view_467 = permute_235 = None
        view_472: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_472);  bmm_42 = _shape_param_472 = None
        add_174: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_472, where);  view_472 = None
        full_23: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_21: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_174, full_23);  add_174 = full_23 = None
        view_473: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_21, _shape_param_473);  maximum_21 = _shape_param_473 = None
        convert_element_type_170: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_473, torch.float32);  view_473 = None
        amax_21: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_170, [-1], True)
        sub_64: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_170, amax_21);  convert_element_type_170 = amax_21 = None
        exp_21: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_171: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        view_474: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_169, _shape_param_474);  convert_element_type_169 = _shape_param_474 = None
        permute_236: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_128: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg346_1, view_474, permute_236);  arg346_1 = view_474 = permute_236 = None
        view_475: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_128, _shape_param_475);  addmm_128 = _shape_param_475 = None
        view_476: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_475, _shape_param_476);  view_475 = _shape_param_476 = None
        permute_237: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_86: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_477: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_86, _shape_param_477);  clone_86 = _shape_param_477 = None
        bmm_43: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_171, view_477);  convert_element_type_171 = view_477 = None
        view_478: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_478);  bmm_43 = _shape_param_478 = None
        permute_238: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
        clone_87: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_479: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_87, _shape_param_479);  clone_87 = _shape_param_479 = None
        view_480: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_479, _shape_param_480);  view_479 = _shape_param_480 = None
        permute_239: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_129: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg348_1, view_480, permute_239);  arg348_1 = view_480 = permute_239 = None
        view_481: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_129, _shape_param_481);  addmm_129 = _shape_param_481 = None
        add_175: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_171, view_481);  add_171 = view_481 = None
        convert_element_type_172: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_175, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_172, [2], correction = 0, keepdim = True)
        getitem_86: "f32[32, 128, 1]" = var_mean_43[0]
        getitem_87: "f32[32, 128, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_65: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_172, getitem_87);  convert_element_type_172 = getitem_87 = None
        add_176: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_43: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        mul_172: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
        mul_173: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_172, arg349_1);  mul_172 = arg349_1 = None
        add_177: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_173, arg350_1);  mul_173 = arg350_1 = None
        convert_element_type_173: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_177, torch.bfloat16);  add_177 = None
        view_482: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_173, _shape_param_482);  convert_element_type_173 = _shape_param_482 = None
        permute_240: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_130: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg352_1, view_482, permute_240);  arg352_1 = view_482 = permute_240 = None
        view_483: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_130, _shape_param_483);  addmm_130 = _shape_param_483 = None
        convert_element_type_174: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_483, torch.float32);  view_483 = None
        mul_174: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_174, 0.5)
        mul_175: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_174, 0.7071067811865476);  convert_element_type_174 = None
        erf_21: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
        add_178: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_176: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_174, add_178);  mul_174 = add_178 = None
        convert_element_type_175: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_176, torch.bfloat16);  mul_176 = None
        view_484: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_175, _shape_param_484);  convert_element_type_175 = _shape_param_484 = None
        permute_241: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_131: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg354_1, view_484, permute_241);  arg354_1 = view_484 = permute_241 = None
        view_485: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_131, _shape_param_485);  addmm_131 = _shape_param_485 = None
        add_179: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_175, view_485);  add_175 = view_485 = None
        convert_element_type_176: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_179, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_176, [2], correction = 0, keepdim = True)
        getitem_88: "f32[32, 128, 1]" = var_mean_44[0]
        getitem_89: "f32[32, 128, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_66: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_176, getitem_89);  convert_element_type_176 = getitem_89 = None
        add_180: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_44: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
        mul_177: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
        mul_178: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_177, arg355_1);  mul_177 = arg355_1 = None
        add_181: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_178, arg356_1);  mul_178 = arg356_1 = None
        convert_element_type_177: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_181, torch.bfloat16);  add_181 = None
        view_486: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_177, _shape_param_486);  _shape_param_486 = None
        permute_242: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_132: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg358_1, view_486, permute_242);  arg358_1 = view_486 = permute_242 = None
        view_487: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_132, _shape_param_487);  addmm_132 = _shape_param_487 = None
        mul_179: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_487, 0.125);  view_487 = None
        view_488: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_179, _shape_param_488);  mul_179 = _shape_param_488 = None
        permute_243: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
        clone_88: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_489: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_88, _shape_param_489);  clone_88 = _shape_param_489 = None
        view_490: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_177, _shape_param_490);  _shape_param_490 = None
        permute_244: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_133: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg360_1, view_490, permute_244);  arg360_1 = view_490 = permute_244 = None
        view_491: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_133, _shape_param_491);  addmm_133 = _shape_param_491 = None
        view_492: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_491, _shape_param_492);  view_491 = _shape_param_492 = None
        permute_245: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
        clone_89: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
        view_493: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_89, _shape_param_493);  clone_89 = _shape_param_493 = None
        permute_246: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_493, [0, 2, 1]);  view_493 = None
        bmm_44: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_489, permute_246);  view_489 = permute_246 = None
        view_494: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_494);  bmm_44 = _shape_param_494 = None
        add_182: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_494, where);  view_494 = None
        full_24: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_22: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_182, full_24);  add_182 = full_24 = None
        view_495: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_22, _shape_param_495);  maximum_22 = _shape_param_495 = None
        convert_element_type_178: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_495, torch.float32);  view_495 = None
        amax_22: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_178, [-1], True)
        sub_67: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_178, amax_22);  convert_element_type_178 = amax_22 = None
        exp_22: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_179: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        view_496: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_177, _shape_param_496);  convert_element_type_177 = _shape_param_496 = None
        permute_247: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_134: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg362_1, view_496, permute_247);  arg362_1 = view_496 = permute_247 = None
        view_497: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_134, _shape_param_497);  addmm_134 = _shape_param_497 = None
        view_498: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_497, _shape_param_498);  view_497 = _shape_param_498 = None
        permute_248: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_90: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_499: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_90, _shape_param_499);  clone_90 = _shape_param_499 = None
        bmm_45: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_179, view_499);  convert_element_type_179 = view_499 = None
        view_500: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_45, _shape_param_500);  bmm_45 = _shape_param_500 = None
        permute_249: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
        clone_91: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_501: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_91, _shape_param_501);  clone_91 = _shape_param_501 = None
        view_502: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_501, _shape_param_502);  view_501 = _shape_param_502 = None
        permute_250: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_135: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg364_1, view_502, permute_250);  arg364_1 = view_502 = permute_250 = None
        view_503: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_135, _shape_param_503);  addmm_135 = _shape_param_503 = None
        add_183: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_179, view_503);  add_179 = view_503 = None
        convert_element_type_180: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_183, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_180, [2], correction = 0, keepdim = True)
        getitem_90: "f32[32, 128, 1]" = var_mean_45[0]
        getitem_91: "f32[32, 128, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_68: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_180, getitem_91);  convert_element_type_180 = getitem_91 = None
        add_184: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_45: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        mul_180: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
        mul_181: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_180, arg365_1);  mul_180 = arg365_1 = None
        add_185: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_181, arg366_1);  mul_181 = arg366_1 = None
        convert_element_type_181: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_185, torch.bfloat16);  add_185 = None
        view_504: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_181, _shape_param_504);  convert_element_type_181 = _shape_param_504 = None
        permute_251: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_136: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg368_1, view_504, permute_251);  arg368_1 = view_504 = permute_251 = None
        view_505: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_136, _shape_param_505);  addmm_136 = _shape_param_505 = None
        convert_element_type_182: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_505, torch.float32);  view_505 = None
        mul_182: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.5)
        mul_183: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.7071067811865476);  convert_element_type_182 = None
        erf_22: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_183);  mul_183 = None
        add_186: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_184: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_182, add_186);  mul_182 = add_186 = None
        convert_element_type_183: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_184, torch.bfloat16);  mul_184 = None
        view_506: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_183, _shape_param_506);  convert_element_type_183 = _shape_param_506 = None
        permute_252: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_137: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg370_1, view_506, permute_252);  arg370_1 = view_506 = permute_252 = None
        view_507: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_137, _shape_param_507);  addmm_137 = _shape_param_507 = None
        add_187: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_183, view_507);  add_183 = view_507 = None
        convert_element_type_184: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_187, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_184, [2], correction = 0, keepdim = True)
        getitem_92: "f32[32, 128, 1]" = var_mean_46[0]
        getitem_93: "f32[32, 128, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_69: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_184, getitem_93);  convert_element_type_184 = getitem_93 = None
        add_188: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
        rsqrt_46: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_185: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
        mul_186: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_185, arg371_1);  mul_185 = arg371_1 = None
        add_189: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_186, arg372_1);  mul_186 = arg372_1 = None
        convert_element_type_185: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None
        view_508: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_185, _shape_param_508);  _shape_param_508 = None
        permute_253: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_138: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg374_1, view_508, permute_253);  arg374_1 = view_508 = permute_253 = None
        view_509: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_138, _shape_param_509);  addmm_138 = _shape_param_509 = None
        mul_187: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_509, 0.125);  view_509 = None
        view_510: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(mul_187, _shape_param_510);  mul_187 = _shape_param_510 = None
        permute_254: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
        clone_92: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_511: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_92, _shape_param_511);  clone_92 = _shape_param_511 = None
        view_512: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_185, _shape_param_512);  _shape_param_512 = None
        permute_255: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        addmm_139: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg376_1, view_512, permute_255);  arg376_1 = view_512 = permute_255 = None
        view_513: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_139, _shape_param_513);  addmm_139 = _shape_param_513 = None
        view_514: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_513, _shape_param_514);  view_513 = _shape_param_514 = None
        permute_256: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        clone_93: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_515: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_93, _shape_param_515);  clone_93 = _shape_param_515 = None
        permute_257: "bf16[512, 64, 128]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
        bmm_46: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_511, permute_257);  view_511 = permute_257 = None
        view_516: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_516);  bmm_46 = _shape_param_516 = None
        add_190: "bf16[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_516, where);  view_516 = where = None
        full_25: "f32[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        maximum_23: "bf16[32, 16, 128, 128]" = torch.ops.aten.maximum.default(add_190, full_25);  add_190 = full_25 = None
        view_517: "bf16[512, 128, 128]" = torch.ops.aten.view.default(maximum_23, _shape_param_517);  maximum_23 = _shape_param_517 = None
        convert_element_type_186: "f32[512, 128, 128]" = torch.ops.prims.convert_element_type.default(view_517, torch.float32);  view_517 = None
        amax_23: "f32[512, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_186, [-1], True)
        sub_70: "f32[512, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_186, amax_23);  convert_element_type_186 = amax_23 = None
        exp_23: "f32[512, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24: "f32[512, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[512, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_187: "bf16[512, 128, 128]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        view_518: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_185, _shape_param_518);  convert_element_type_185 = _shape_param_518 = None
        permute_258: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_140: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg378_1, view_518, permute_258);  arg378_1 = view_518 = permute_258 = None
        view_519: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_140, _shape_param_519);  addmm_140 = _shape_param_519 = None
        view_520: "bf16[32, 128, 16, 64]" = torch.ops.aten.view.default(view_519, _shape_param_520);  view_519 = _shape_param_520 = None
        permute_259: "bf16[32, 16, 128, 64]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
        clone_94: "bf16[32, 16, 128, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_521: "bf16[512, 128, 64]" = torch.ops.aten.view.default(clone_94, _shape_param_521);  clone_94 = _shape_param_521 = None
        bmm_47: "bf16[512, 128, 64]" = torch.ops.aten.bmm.default(convert_element_type_187, view_521);  convert_element_type_187 = view_521 = None
        view_522: "bf16[32, 16, 128, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_522);  bmm_47 = _shape_param_522 = None
        permute_260: "bf16[32, 128, 16, 64]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
        clone_95: "bf16[32, 128, 16, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_523: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(clone_95, _shape_param_523);  clone_95 = _shape_param_523 = None
        view_524: "bf16[4096, 1024]" = torch.ops.aten.view.default(view_523, _shape_param_524);  view_523 = _shape_param_524 = None
        permute_261: "bf16[1024, 1024]" = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_141: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg380_1, view_524, permute_261);  arg380_1 = view_524 = permute_261 = None
        view_525: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_141, _shape_param_525);  addmm_141 = _shape_param_525 = None
        add_191: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_187, view_525);  add_187 = view_525 = None
        convert_element_type_188: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_191, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_188, [2], correction = 0, keepdim = True)
        getitem_94: "f32[32, 128, 1]" = var_mean_47[0]
        getitem_95: "f32[32, 128, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_71: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_188, getitem_95);  convert_element_type_188 = getitem_95 = None
        add_192: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
        rsqrt_47: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_188: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
        mul_189: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_188, arg381_1);  mul_188 = arg381_1 = None
        add_193: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_189, arg382_1);  mul_189 = arg382_1 = None
        convert_element_type_189: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16);  add_193 = None
        view_526: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_189, _shape_param_526);  convert_element_type_189 = _shape_param_526 = None
        permute_262: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_142: "bf16[4096, 4096]" = torch.ops.aten.addmm.default(arg384_1, view_526, permute_262);  arg384_1 = view_526 = permute_262 = None
        view_527: "bf16[32, 128, 4096]" = torch.ops.aten.view.default(addmm_142, _shape_param_527);  addmm_142 = _shape_param_527 = None
        convert_element_type_190: "f32[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_527, torch.float32);  view_527 = None
        mul_190: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_190, 0.5)
        mul_191: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_190, 0.7071067811865476);  convert_element_type_190 = None
        erf_23: "f32[32, 128, 4096]" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
        add_194: "f32[32, 128, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_192: "f32[32, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, add_194);  mul_190 = add_194 = None
        convert_element_type_191: "bf16[32, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None
        view_528: "bf16[4096, 4096]" = torch.ops.aten.view.default(convert_element_type_191, _shape_param_528);  convert_element_type_191 = _shape_param_528 = None
        permute_263: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_143: "bf16[4096, 1024]" = torch.ops.aten.addmm.default(arg386_1, view_528, permute_263);  arg386_1 = view_528 = permute_263 = None
        view_529: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(addmm_143, _shape_param_529);  addmm_143 = _shape_param_529 = None
        add_195: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_191, view_529);  add_191 = view_529 = None
        convert_element_type_192: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_195, torch.float32);  add_195 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_192, [2], correction = 0, keepdim = True)
        getitem_96: "f32[32, 128, 1]" = var_mean_48[0]
        getitem_97: "f32[32, 128, 1]" = var_mean_48[1];  var_mean_48 = None
        constant_pad_nd: "i64[32, 129]" = torch.ops.aten.constant_pad_nd.default(arg387_1, [0, 1], -100.0);  arg387_1 = None
        slice_1: "i64[32, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_96: "i64[32, 128]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None
        view_530: "i64[4096]" = torch.ops.aten.view.default(clone_96, [-1]);  clone_96 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_530, -100)
        sub_72: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_192, getitem_97);  convert_element_type_192 = getitem_97 = None
        add_196: "f32[32, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_48: "f32[32, 128, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        mul_193: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
        mul_194: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_193, arg388_1);  mul_193 = arg388_1 = None
        add_197: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_194, arg389_1);  mul_194 = arg389_1 = None
        convert_element_type_193: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16);  add_197 = None
        view_531: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_193, _shape_param_530);  convert_element_type_193 = _shape_param_530 = None
        permute_264: "bf16[1024, 256008]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        mm: "bf16[4096, 256008]" = torch.ops.aten.mm.default(view_531, permute_264);  view_531 = permute_264 = None
        view_532: "bf16[32, 128, 256008]" = torch.ops.aten.view.default(mm, _shape_param_531);  mm = _shape_param_531 = None
        convert_element_type_194: "f32[32, 128, 256008]" = torch.ops.prims.convert_element_type.default(view_532, torch.float32)
        view_533: "f32[4096, 256008]" = torch.ops.aten.view.default(convert_element_type_194, _shape_param_532);  convert_element_type_194 = _shape_param_532 = None
        amax_24: "f32[4096, 1]" = torch.ops.aten.amax.default(view_533, [1], True)
        sub_73: "f32[4096, 256008]" = torch.ops.aten.sub.Tensor(view_533, amax_24);  view_533 = amax_24 = None
        exp_24: "f32[4096, 256008]" = torch.ops.aten.exp.default(sub_73)
        sum_25: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_74: "f32[4096, 256008]" = torch.ops.aten.sub.Tensor(sub_73, log);  sub_73 = log = None
        ne_1: "b8[4096]" = torch.ops.aten.ne.Scalar(view_530, -100)
        full_26: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "i64[4096]" = torch.ops.aten.where.self(ne_1, view_530, full_26);  ne_1 = full_26 = None
        unsqueeze_7: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(sub_74, 1, unsqueeze_7);  sub_74 = unsqueeze_7 = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_27: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4096]" = torch.ops.aten.where.self(ne, neg, full_27);  ne = neg = full_27 = None
        sum_26: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
        ne_2: "b8[4096]" = torch.ops.aten.ne.Scalar(view_530, -100);  view_530 = None
        sum_27: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_195: "f32[]" = torch.ops.prims.convert_element_type.default(sum_27, torch.float32);  sum_27 = None
        div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_26, convert_element_type_195);  sum_26 = convert_element_type_195 = None
        return (view_532, div_24)



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
