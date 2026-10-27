"""
Standalone repro captured via capture_hook.
Label: DebertaV2ForMaskedLM
Pattern hash: 73d20ec79a74
Shape hash: d3fce788
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
    def forward(self, arg0_1: "f32[128100, 1536]", arg1_1: "i64[8, 512]", arg2_1: "f32[512, 1536]", arg3_1: "i64[1, 512]", arg4_1: "f32[1536]", arg5_1: "f32[1536]", arg6_1: "f32[1536]", arg7_1: "f32[1536, 1536]", arg8_1: "f32[1536]", arg9_1: "f32[1536, 1536]", arg10_1: "f32[1536]", arg11_1: "f32[1536, 1536]", arg12_1: "f32[1536]", arg13_1: "f32[1536, 1536]", arg14_1: "f32[1536]", arg15_1: "f32[1536]", arg16_1: "f32[6144]", arg17_1: "f32[6144, 1536]", arg18_1: "f32[1536]", arg19_1: "f32[1536, 6144]", arg20_1: "f32[1536]", arg21_1: "f32[1536]", arg22_1: "f32[1536]", arg23_1: "f32[1536, 1536]", arg24_1: "f32[1536]", arg25_1: "f32[1536, 1536]", arg26_1: "f32[1536]", arg27_1: "f32[1536, 1536]", arg28_1: "f32[1536]", arg29_1: "f32[1536, 1536]", arg30_1: "f32[1536]", arg31_1: "f32[1536]", arg32_1: "f32[6144]", arg33_1: "f32[6144, 1536]", arg34_1: "f32[1536]", arg35_1: "f32[1536, 6144]", arg36_1: "f32[1536]", arg37_1: "f32[1536]", arg38_1: "f32[1536]", arg39_1: "f32[1536, 1536]", arg40_1: "f32[1536]", arg41_1: "f32[1536, 1536]", arg42_1: "f32[1536]", arg43_1: "f32[1536, 1536]", arg44_1: "f32[1536]", arg45_1: "f32[1536, 1536]", arg46_1: "f32[1536]", arg47_1: "f32[1536]", arg48_1: "f32[6144]", arg49_1: "f32[6144, 1536]", arg50_1: "f32[1536]", arg51_1: "f32[1536, 6144]", arg52_1: "f32[1536]", arg53_1: "f32[1536]", arg54_1: "f32[1536]", arg55_1: "f32[1536, 1536]", arg56_1: "f32[1536]", arg57_1: "f32[1536, 1536]", arg58_1: "f32[1536]", arg59_1: "f32[1536, 1536]", arg60_1: "f32[1536]", arg61_1: "f32[1536, 1536]", arg62_1: "f32[1536]", arg63_1: "f32[1536]", arg64_1: "f32[6144]", arg65_1: "f32[6144, 1536]", arg66_1: "f32[1536]", arg67_1: "f32[1536, 6144]", arg68_1: "f32[1536]", arg69_1: "f32[1536]", arg70_1: "f32[1536]", arg71_1: "f32[1536, 1536]", arg72_1: "f32[1536]", arg73_1: "f32[1536, 1536]", arg74_1: "f32[1536]", arg75_1: "f32[1536, 1536]", arg76_1: "f32[1536]", arg77_1: "f32[1536, 1536]", arg78_1: "f32[1536]", arg79_1: "f32[1536]", arg80_1: "f32[6144]", arg81_1: "f32[6144, 1536]", arg82_1: "f32[1536]", arg83_1: "f32[1536, 6144]", arg84_1: "f32[1536]", arg85_1: "f32[1536]", arg86_1: "f32[1536]", arg87_1: "f32[1536, 1536]", arg88_1: "f32[1536]", arg89_1: "f32[1536, 1536]", arg90_1: "f32[1536]", arg91_1: "f32[1536, 1536]", arg92_1: "f32[1536]", arg93_1: "f32[1536, 1536]", arg94_1: "f32[1536]", arg95_1: "f32[1536]", arg96_1: "f32[6144]", arg97_1: "f32[6144, 1536]", arg98_1: "f32[1536]", arg99_1: "f32[1536, 6144]", arg100_1: "f32[1536]", arg101_1: "f32[1536]", arg102_1: "f32[1536]", arg103_1: "f32[1536, 1536]", arg104_1: "f32[1536]", arg105_1: "f32[1536, 1536]", arg106_1: "f32[1536]", arg107_1: "f32[1536, 1536]", arg108_1: "f32[1536]", arg109_1: "f32[1536, 1536]", arg110_1: "f32[1536]", arg111_1: "f32[1536]", arg112_1: "f32[6144]", arg113_1: "f32[6144, 1536]", arg114_1: "f32[1536]", arg115_1: "f32[1536, 6144]", arg116_1: "f32[1536]", arg117_1: "f32[1536]", arg118_1: "f32[1536]", arg119_1: "f32[1536, 1536]", arg120_1: "f32[1536]", arg121_1: "f32[1536, 1536]", arg122_1: "f32[1536]", arg123_1: "f32[1536, 1536]", arg124_1: "f32[1536]", arg125_1: "f32[1536, 1536]", arg126_1: "f32[1536]", arg127_1: "f32[1536]", arg128_1: "f32[6144]", arg129_1: "f32[6144, 1536]", arg130_1: "f32[1536]", arg131_1: "f32[1536, 6144]", arg132_1: "f32[1536]", arg133_1: "f32[1536]", arg134_1: "f32[1536]", arg135_1: "f32[1536, 1536]", arg136_1: "f32[1536]", arg137_1: "f32[1536, 1536]", arg138_1: "f32[1536]", arg139_1: "f32[1536, 1536]", arg140_1: "f32[1536]", arg141_1: "f32[1536, 1536]", arg142_1: "f32[1536]", arg143_1: "f32[1536]", arg144_1: "f32[6144]", arg145_1: "f32[6144, 1536]", arg146_1: "f32[1536]", arg147_1: "f32[1536, 6144]", arg148_1: "f32[1536]", arg149_1: "f32[1536]", arg150_1: "f32[1536]", arg151_1: "f32[1536, 1536]", arg152_1: "f32[1536]", arg153_1: "f32[1536, 1536]", arg154_1: "f32[1536]", arg155_1: "f32[1536, 1536]", arg156_1: "f32[1536]", arg157_1: "f32[1536, 1536]", arg158_1: "f32[1536]", arg159_1: "f32[1536]", arg160_1: "f32[6144]", arg161_1: "f32[6144, 1536]", arg162_1: "f32[1536]", arg163_1: "f32[1536, 6144]", arg164_1: "f32[1536]", arg165_1: "f32[1536]", arg166_1: "f32[1536]", arg167_1: "f32[1536, 1536]", arg168_1: "f32[1536]", arg169_1: "f32[1536, 1536]", arg170_1: "f32[1536]", arg171_1: "f32[1536, 1536]", arg172_1: "f32[1536]", arg173_1: "f32[1536, 1536]", arg174_1: "f32[1536]", arg175_1: "f32[1536]", arg176_1: "f32[6144]", arg177_1: "f32[6144, 1536]", arg178_1: "f32[1536]", arg179_1: "f32[1536, 6144]", arg180_1: "f32[1536]", arg181_1: "f32[1536]", arg182_1: "f32[1536]", arg183_1: "f32[1536, 1536]", arg184_1: "f32[1536]", arg185_1: "f32[1536, 1536]", arg186_1: "f32[1536]", arg187_1: "f32[1536, 1536]", arg188_1: "f32[1536]", arg189_1: "f32[1536, 1536]", arg190_1: "f32[1536]", arg191_1: "f32[1536]", arg192_1: "f32[6144]", arg193_1: "f32[6144, 1536]", arg194_1: "f32[1536]", arg195_1: "f32[1536, 6144]", arg196_1: "f32[1536]", arg197_1: "f32[1536]", arg198_1: "f32[1536]", arg199_1: "f32[1536, 1536]", arg200_1: "f32[1536]", arg201_1: "f32[1536, 1536]", arg202_1: "f32[1536]", arg203_1: "f32[1536, 1536]", arg204_1: "f32[1536]", arg205_1: "f32[1536, 1536]", arg206_1: "f32[1536]", arg207_1: "f32[1536]", arg208_1: "f32[6144]", arg209_1: "f32[6144, 1536]", arg210_1: "f32[1536]", arg211_1: "f32[1536, 6144]", arg212_1: "f32[1536]", arg213_1: "f32[1536]", arg214_1: "f32[1536]", arg215_1: "f32[1536, 1536]", arg216_1: "f32[1536]", arg217_1: "f32[1536, 1536]", arg218_1: "f32[1536]", arg219_1: "f32[1536, 1536]", arg220_1: "f32[1536]", arg221_1: "f32[1536, 1536]", arg222_1: "f32[1536]", arg223_1: "f32[1536]", arg224_1: "f32[6144]", arg225_1: "f32[6144, 1536]", arg226_1: "f32[1536]", arg227_1: "f32[1536, 6144]", arg228_1: "f32[1536]", arg229_1: "f32[1536]", arg230_1: "f32[1536]", arg231_1: "f32[1536, 1536]", arg232_1: "f32[1536]", arg233_1: "f32[1536, 1536]", arg234_1: "f32[1536]", arg235_1: "f32[1536, 1536]", arg236_1: "f32[1536]", arg237_1: "f32[1536, 1536]", arg238_1: "f32[1536]", arg239_1: "f32[1536]", arg240_1: "f32[6144]", arg241_1: "f32[6144, 1536]", arg242_1: "f32[1536]", arg243_1: "f32[1536, 6144]", arg244_1: "f32[1536]", arg245_1: "f32[1536]", arg246_1: "f32[1536]", arg247_1: "f32[1536, 1536]", arg248_1: "f32[1536]", arg249_1: "f32[1536, 1536]", arg250_1: "f32[1536]", arg251_1: "f32[1536, 1536]", arg252_1: "f32[1536]", arg253_1: "f32[1536, 1536]", arg254_1: "f32[1536]", arg255_1: "f32[1536]", arg256_1: "f32[6144]", arg257_1: "f32[6144, 1536]", arg258_1: "f32[1536]", arg259_1: "f32[1536, 6144]", arg260_1: "f32[1536]", arg261_1: "f32[1536]", arg262_1: "f32[1536]", arg263_1: "f32[1536, 1536]", arg264_1: "f32[1536]", arg265_1: "f32[1536, 1536]", arg266_1: "f32[1536]", arg267_1: "f32[1536, 1536]", arg268_1: "f32[1536]", arg269_1: "f32[1536, 1536]", arg270_1: "f32[1536]", arg271_1: "f32[1536]", arg272_1: "f32[6144]", arg273_1: "f32[6144, 1536]", arg274_1: "f32[1536]", arg275_1: "f32[1536, 6144]", arg276_1: "f32[1536]", arg277_1: "f32[1536]", arg278_1: "f32[1536]", arg279_1: "f32[1536, 1536]", arg280_1: "f32[1536]", arg281_1: "f32[1536, 1536]", arg282_1: "f32[1536]", arg283_1: "f32[1536, 1536]", arg284_1: "f32[1536]", arg285_1: "f32[1536, 1536]", arg286_1: "f32[1536]", arg287_1: "f32[1536]", arg288_1: "f32[6144]", arg289_1: "f32[6144, 1536]", arg290_1: "f32[1536]", arg291_1: "f32[1536, 6144]", arg292_1: "f32[1536]", arg293_1: "f32[1536]", arg294_1: "f32[1536]", arg295_1: "f32[1536, 1536]", arg296_1: "f32[1536]", arg297_1: "f32[1536, 1536]", arg298_1: "f32[1536]", arg299_1: "f32[1536, 1536]", arg300_1: "f32[1536]", arg301_1: "f32[1536, 1536]", arg302_1: "f32[1536]", arg303_1: "f32[1536]", arg304_1: "f32[6144]", arg305_1: "f32[6144, 1536]", arg306_1: "f32[1536]", arg307_1: "f32[1536, 6144]", arg308_1: "f32[1536]", arg309_1: "f32[1536]", arg310_1: "f32[1536]", arg311_1: "f32[1536, 1536]", arg312_1: "f32[1536]", arg313_1: "f32[1536, 1536]", arg314_1: "f32[1536]", arg315_1: "f32[1536, 1536]", arg316_1: "f32[1536]", arg317_1: "f32[1536, 1536]", arg318_1: "f32[1536]", arg319_1: "f32[1536]", arg320_1: "f32[6144]", arg321_1: "f32[6144, 1536]", arg322_1: "f32[1536]", arg323_1: "f32[1536, 6144]", arg324_1: "f32[1536]", arg325_1: "f32[1536]", arg326_1: "f32[1536]", arg327_1: "f32[1536, 1536]", arg328_1: "f32[1536]", arg329_1: "f32[1536, 1536]", arg330_1: "f32[1536]", arg331_1: "f32[1536, 1536]", arg332_1: "f32[1536]", arg333_1: "f32[1536, 1536]", arg334_1: "f32[1536]", arg335_1: "f32[1536]", arg336_1: "f32[6144]", arg337_1: "f32[6144, 1536]", arg338_1: "f32[1536]", arg339_1: "f32[1536, 6144]", arg340_1: "f32[1536]", arg341_1: "f32[1536]", arg342_1: "f32[1536]", arg343_1: "f32[1536, 1536]", arg344_1: "f32[1536]", arg345_1: "f32[1536, 1536]", arg346_1: "f32[1536]", arg347_1: "f32[1536, 1536]", arg348_1: "f32[1536]", arg349_1: "f32[1536, 1536]", arg350_1: "f32[1536]", arg351_1: "f32[1536]", arg352_1: "f32[6144]", arg353_1: "f32[6144, 1536]", arg354_1: "f32[1536]", arg355_1: "f32[1536, 6144]", arg356_1: "f32[1536]", arg357_1: "f32[1536]", arg358_1: "f32[1536]", arg359_1: "f32[1536, 1536]", arg360_1: "f32[1536]", arg361_1: "f32[1536, 1536]", arg362_1: "f32[1536]", arg363_1: "f32[1536, 1536]", arg364_1: "f32[1536]", arg365_1: "f32[1536, 1536]", arg366_1: "f32[1536]", arg367_1: "f32[1536]", arg368_1: "f32[6144]", arg369_1: "f32[6144, 1536]", arg370_1: "f32[1536]", arg371_1: "f32[1536, 6144]", arg372_1: "f32[1536]", arg373_1: "f32[1536]", arg374_1: "f32[1536]", arg375_1: "f32[1536, 1536]", arg376_1: "f32[1536]", arg377_1: "f32[1536, 1536]", arg378_1: "f32[1536]", arg379_1: "f32[1536, 1536]", arg380_1: "f32[1536]", arg381_1: "f32[1536, 1536]", arg382_1: "f32[1536]", arg383_1: "f32[1536]", arg384_1: "f32[6144]", arg385_1: "f32[6144, 1536]", arg386_1: "f32[1536]", arg387_1: "f32[1536, 6144]", arg388_1: "f32[1536]", arg389_1: "f32[1536]", arg390_1: "f32[1536]", arg391_1: "f32[1536, 1536]", arg392_1: "f32[1536]", arg393_1: "f32[1536]", arg394_1: "f32[128100]", arg395_1: "i64[8, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560):
        # No stacktrace found for following nodes
        embedding: "f32[8, 512, 1536]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg1_1 = None
        embedding_1: "f32[1, 512, 1536]" = torch.ops.aten.embedding.default(arg2_1, arg3_1);  arg2_1 = arg3_1 = None
        add: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(embedding, embedding_1)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[8, 512, 1]" = var_mean[0]
        getitem_1: "f32[8, 512, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-07);  getitem = None
        rsqrt: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = None
        mul: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul, arg4_1);  mul = arg4_1 = None
        add_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_1, arg5_1);  mul_1 = arg5_1 = None
        inductor_seeds: "i64[73]" = torch.ops.prims.inductor_seeds.default(73, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_0, inductor_lookup_seed, 'rand');  _shape_param_0 = inductor_lookup_seed = None
        gt: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(inductor_random, 0.1);  inductor_random = None
        mul_2: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt, add_2);  add_2 = None
        mul_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_2, 1.1111111111111112);  mul_2 = None
        convert_element_type: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg6_1, torch.bfloat16);  arg6_1 = None
        convert_element_type_1: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.bfloat16);  arg7_1 = None
        convert_element_type_2: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16)
        view: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        permute: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm, _shape_param_2);  addmm = _shape_param_2 = None
        view_2: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_1, _shape_param_3);  view_1 = _shape_param_3 = None
        permute_1: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        clone: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        view_3: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone, _shape_param_4);  clone = _shape_param_4 = None
        convert_element_type_3: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.bfloat16);  arg8_1 = None
        convert_element_type_4: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        permute_2: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm_1: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_3, view, permute_2);  convert_element_type_3 = None
        view_4: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_1, _shape_param_5);  addmm_1 = _shape_param_5 = None
        view_5: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_4, _shape_param_6);  view_4 = _shape_param_6 = None
        permute_3: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        clone_1: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        view_6: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_7);  clone_1 = _shape_param_7 = None
        convert_element_type_5: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        convert_element_type_6: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg11_1, torch.bfloat16);  arg11_1 = None
        permute_4: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_6, [1, 0]);  convert_element_type_6 = None
        addmm_2: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_5, view, permute_4);  convert_element_type_5 = None
        view_7: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_2, _shape_param_8);  addmm_2 = _shape_param_8 = None
        view_8: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_7, _shape_param_9);  view_7 = _shape_param_9 = None
        permute_5: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        clone_2: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_9: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_2, _shape_param_10);  clone_2 = _shape_param_10 = None
        permute_6: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
        full: "bf16[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        div: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_6, full);  permute_6 = None
        bmm: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_3, div)
        view_10: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm, _shape_param_11);  _shape_param_11 = None
        full_1: "b8[8, 1, 512, 512]" = torch.ops.aten.full.default(_shape_param_12, False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_12 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_10);  view_10 = None
        convert_element_type_7: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        amax: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_7, [-1], True)
        sub_1: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_7, amax);  convert_element_type_7 = None
        exp: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_13, inductor_lookup_seed_1, 'rand');  _shape_param_13 = inductor_lookup_seed_1 = None
        gt_1: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_1, 0.1);  inductor_random_1 = None
        mul_4: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_1, div_1);  div_1 = None
        mul_5: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None
        view_11: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_5, _shape_param_14);  mul_5 = _shape_param_14 = None
        convert_element_type_8: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        bmm_1: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_8, view_9)
        view_12: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_1, _shape_param_15);  bmm_1 = _shape_param_15 = None
        permute_7: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone_3: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_13: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_3, _shape_param_16);  clone_3 = _shape_param_16 = None
        convert_element_type_9: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        convert_element_type_10: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        view_14: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_13, _shape_param_17);  view_13 = _shape_param_17 = None
        permute_8: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_10, [1, 0]);  convert_element_type_10 = None
        addmm_3: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_9, view_14, permute_8);  convert_element_type_9 = None
        view_15: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_3, _shape_param_18);  addmm_3 = _shape_param_18 = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2)
        inductor_random_2: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_19, inductor_lookup_seed_2, 'rand');  _shape_param_19 = inductor_lookup_seed_2 = None
        convert_element_type_11: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_2, torch.bfloat16);  inductor_random_2 = None
        gt_2: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_11, 0.1);  convert_element_type_11 = None
        mul_6: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_2, view_15);  view_15 = None
        mul_7: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_6, 1.1111111111111112);  mul_6 = None
        add_3: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_7, mul_3);  mul_7 = mul_3 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
        getitem_2: "f32[8, 512, 1]" = var_mean_1[0]
        getitem_3: "f32[8, 512, 1]" = var_mean_1[1];  var_mean_1 = None
        add_4: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-07);  getitem_2 = None
        rsqrt_1: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        sub_2: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
        mul_8: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_9: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_8, arg14_1);  arg14_1 = None
        add_5: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_9, arg15_1);  mul_9 = arg15_1 = None
        convert_element_type_12: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        convert_element_type_13: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        convert_element_type_14: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        view_16: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_14, _shape_param_20);  convert_element_type_14 = _shape_param_20 = None
        permute_9: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_4: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_12, view_16, permute_9);  convert_element_type_12 = None
        view_17: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_4, _shape_param_21);  _shape_param_21 = None
        convert_element_type_15: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        mul_10: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.5)
        mul_11: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 0.7071067811865476);  convert_element_type_15 = None
        erf: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_6: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_12: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_10, add_6);  mul_10 = add_6 = None
        convert_element_type_16: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        convert_element_type_17: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        convert_element_type_18: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        view_18: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_16, _shape_param_22);  convert_element_type_16 = _shape_param_22 = None
        permute_10: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_18, [1, 0]);  convert_element_type_18 = None
        addmm_5: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_17, view_18, permute_10);  convert_element_type_17 = None
        view_19: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_5, _shape_param_23);  addmm_5 = _shape_param_23 = None
        inductor_lookup_seed_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
        inductor_random_3: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_24, inductor_lookup_seed_3, 'rand');  _shape_param_24 = inductor_lookup_seed_3 = None
        convert_element_type_19: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_3, torch.bfloat16);  inductor_random_3 = None
        gt_3: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_19, 0.1);  convert_element_type_19 = None
        mul_13: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_3, view_19);  view_19 = None
        mul_14: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_13, 1.1111111111111112);  mul_13 = None
        add_7: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_14, add_5);  mul_14 = add_5 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
        getitem_4: "f32[8, 512, 1]" = var_mean_2[0]
        getitem_5: "f32[8, 512, 1]" = var_mean_2[1];  var_mean_2 = None
        add_8: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-07);  getitem_4 = None
        rsqrt_2: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        sub_3: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  add_7 = getitem_5 = None
        mul_15: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
        mul_16: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_15, arg20_1);  arg20_1 = None
        add_9: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_16, arg21_1);  mul_16 = arg21_1 = None
        convert_element_type_20: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        convert_element_type_21: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        convert_element_type_22: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16)
        view_20: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_22, _shape_param_25);  convert_element_type_22 = _shape_param_25 = None
        permute_11: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_21, [1, 0]);  convert_element_type_21 = None
        addmm_6: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_20, view_20, permute_11);  convert_element_type_20 = None
        view_21: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_6, _shape_param_26);  addmm_6 = _shape_param_26 = None
        view_22: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_21, _shape_param_27);  view_21 = _shape_param_27 = None
        permute_12: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        clone_4: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_12, memory_format = torch.contiguous_format);  permute_12 = None
        view_23: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_28);  clone_4 = _shape_param_28 = None
        convert_element_type_23: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        convert_element_type_24: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        permute_13: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_7: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_23, view_20, permute_13);  convert_element_type_23 = None
        view_24: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_7, _shape_param_29);  addmm_7 = _shape_param_29 = None
        view_25: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_24, _shape_param_30);  view_24 = _shape_param_30 = None
        permute_14: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        clone_5: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_26: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_5, _shape_param_31);  clone_5 = _shape_param_31 = None
        convert_element_type_25: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg26_1, torch.bfloat16);  arg26_1 = None
        convert_element_type_26: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        permute_15: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_26, [1, 0]);  convert_element_type_26 = None
        addmm_8: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_25, view_20, permute_15);  convert_element_type_25 = None
        view_27: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_8, _shape_param_32);  addmm_8 = _shape_param_32 = None
        view_28: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_27, _shape_param_33);  view_27 = _shape_param_33 = None
        permute_16: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_6: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_29: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_34);  clone_6 = _shape_param_34 = None
        permute_17: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
        div_2: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_17, full);  permute_17 = None
        bmm_2: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_23, div_2)
        view_30: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_2, _shape_param_35);  bmm_2 = _shape_param_35 = None
        where_1: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_30);  view_30 = None
        convert_element_type_27: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_1, torch.float32)
        amax_1: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_27, [-1], True)
        sub_4: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_27, amax_1);  convert_element_type_27 = None
        exp_1: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_3: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None
        inductor_lookup_seed_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 4)
        inductor_random_4: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_36, inductor_lookup_seed_4, 'rand');  _shape_param_36 = inductor_lookup_seed_4 = None
        gt_4: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_4, 0.1);  inductor_random_4 = None
        mul_17: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_4, div_3);  div_3 = None
        mul_18: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_17, 1.1111111111111112);  mul_17 = None
        view_31: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_18, _shape_param_37);  mul_18 = _shape_param_37 = None
        convert_element_type_28: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_31, torch.bfloat16);  view_31 = None
        bmm_3: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_28, view_29)
        view_32: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_38);  bmm_3 = _shape_param_38 = None
        permute_18: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
        clone_7: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_33: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_7, _shape_param_39);  clone_7 = _shape_param_39 = None
        convert_element_type_29: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        convert_element_type_30: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        view_34: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_33, _shape_param_40);  view_33 = _shape_param_40 = None
        permute_19: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_30, [1, 0]);  convert_element_type_30 = None
        addmm_9: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_29, view_34, permute_19);  convert_element_type_29 = None
        view_35: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_9, _shape_param_41);  addmm_9 = _shape_param_41 = None
        inductor_lookup_seed_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 5)
        inductor_random_5: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_42, inductor_lookup_seed_5, 'rand');  _shape_param_42 = inductor_lookup_seed_5 = None
        convert_element_type_31: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_5, torch.bfloat16);  inductor_random_5 = None
        gt_5: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_31, 0.1);  convert_element_type_31 = None
        mul_19: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_5, view_35);  view_35 = None
        mul_20: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_19, 1.1111111111111112);  mul_19 = None
        add_10: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_20, add_9);  mul_20 = add_9 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_6: "f32[8, 512, 1]" = var_mean_3[0]
        getitem_7: "f32[8, 512, 1]" = var_mean_3[1];  var_mean_3 = None
        add_11: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-07);  getitem_6 = None
        rsqrt_3: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_5: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_10, getitem_7);  add_10 = getitem_7 = None
        mul_21: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_22: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_21, arg30_1);  arg30_1 = None
        add_12: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_22, arg31_1);  mul_22 = arg31_1 = None
        convert_element_type_32: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.bfloat16);  arg32_1 = None
        convert_element_type_33: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.bfloat16);  arg33_1 = None
        convert_element_type_34: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16)
        view_36: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_34, _shape_param_43);  convert_element_type_34 = _shape_param_43 = None
        permute_20: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_33, [1, 0]);  convert_element_type_33 = None
        addmm_10: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_32, view_36, permute_20);  convert_element_type_32 = None
        view_37: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_10, _shape_param_44);  _shape_param_44 = None
        convert_element_type_35: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        mul_23: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.5)
        mul_24: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 0.7071067811865476);  convert_element_type_35 = None
        erf_1: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
        add_13: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_25: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_23, add_13);  mul_23 = add_13 = None
        convert_element_type_36: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_25, torch.bfloat16);  mul_25 = None
        convert_element_type_37: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        convert_element_type_38: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg35_1, torch.bfloat16);  arg35_1 = None
        view_38: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_45);  convert_element_type_36 = _shape_param_45 = None
        permute_21: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_38, [1, 0]);  convert_element_type_38 = None
        addmm_11: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_37, view_38, permute_21);  convert_element_type_37 = None
        view_39: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_11, _shape_param_46);  addmm_11 = _shape_param_46 = None
        inductor_lookup_seed_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 6)
        inductor_random_6: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_47, inductor_lookup_seed_6, 'rand');  _shape_param_47 = inductor_lookup_seed_6 = None
        convert_element_type_39: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_6, torch.bfloat16);  inductor_random_6 = None
        gt_6: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_39, 0.1);  convert_element_type_39 = None
        mul_26: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_6, view_39);  view_39 = None
        mul_27: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_26, 1.1111111111111112);  mul_26 = None
        add_14: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_27, add_12);  mul_27 = add_12 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_8: "f32[8, 512, 1]" = var_mean_4[0]
        getitem_9: "f32[8, 512, 1]" = var_mean_4[1];  var_mean_4 = None
        add_15: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-07);  getitem_8 = None
        rsqrt_4: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        sub_6: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  add_14 = getitem_9 = None
        mul_28: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
        mul_29: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_28, arg36_1);  arg36_1 = None
        add_16: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_29, arg37_1);  mul_29 = arg37_1 = None
        convert_element_type_40: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.bfloat16);  arg38_1 = None
        convert_element_type_41: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg39_1, torch.bfloat16);  arg39_1 = None
        convert_element_type_42: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16)
        view_40: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_42, _shape_param_48);  convert_element_type_42 = _shape_param_48 = None
        permute_22: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_41, [1, 0]);  convert_element_type_41 = None
        addmm_12: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_40, view_40, permute_22);  convert_element_type_40 = None
        view_41: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_12, _shape_param_49);  addmm_12 = _shape_param_49 = None
        view_42: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_41, _shape_param_50);  view_41 = _shape_param_50 = None
        permute_23: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        clone_8: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_43: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_51);  clone_8 = _shape_param_51 = None
        convert_element_type_43: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg40_1, torch.bfloat16);  arg40_1 = None
        convert_element_type_44: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        permute_24: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_44, [1, 0]);  convert_element_type_44 = None
        addmm_13: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_43, view_40, permute_24);  convert_element_type_43 = None
        view_44: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_13, _shape_param_52);  addmm_13 = _shape_param_52 = None
        view_45: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_44, _shape_param_53);  view_44 = _shape_param_53 = None
        permute_25: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        clone_9: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_46: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_9, _shape_param_54);  clone_9 = _shape_param_54 = None
        convert_element_type_45: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        convert_element_type_46: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        permute_26: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_46, [1, 0]);  convert_element_type_46 = None
        addmm_14: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_45, view_40, permute_26);  convert_element_type_45 = None
        view_47: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_14, _shape_param_55);  addmm_14 = _shape_param_55 = None
        view_48: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_47, _shape_param_56);  view_47 = _shape_param_56 = None
        permute_27: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        clone_10: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_49: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_57);  clone_10 = _shape_param_57 = None
        permute_28: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
        div_4: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_28, full);  permute_28 = None
        bmm_4: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_43, div_4)
        view_50: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_4, _shape_param_58);  bmm_4 = _shape_param_58 = None
        where_2: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_50);  view_50 = None
        convert_element_type_47: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_2, torch.float32)
        amax_2: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_47, [-1], True)
        sub_7: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_47, amax_2);  convert_element_type_47 = None
        exp_2: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_3: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_5: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None
        inductor_lookup_seed_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 7)
        inductor_random_7: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_59, inductor_lookup_seed_7, 'rand');  _shape_param_59 = inductor_lookup_seed_7 = None
        gt_7: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_7, 0.1);  inductor_random_7 = None
        mul_30: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_7, div_5);  div_5 = None
        mul_31: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_30, 1.1111111111111112);  mul_30 = None
        view_51: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_31, _shape_param_60);  mul_31 = _shape_param_60 = None
        convert_element_type_48: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_51, torch.bfloat16);  view_51 = None
        bmm_5: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_48, view_49)
        view_52: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_61);  bmm_5 = _shape_param_61 = None
        permute_29: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        clone_11: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_53: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_11, _shape_param_62);  clone_11 = _shape_param_62 = None
        convert_element_type_49: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg44_1, torch.bfloat16);  arg44_1 = None
        convert_element_type_50: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        view_54: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_53, _shape_param_63);  view_53 = _shape_param_63 = None
        permute_30: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_50, [1, 0]);  convert_element_type_50 = None
        addmm_15: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_49, view_54, permute_30);  convert_element_type_49 = None
        view_55: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_15, _shape_param_64);  addmm_15 = _shape_param_64 = None
        inductor_lookup_seed_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 8)
        inductor_random_8: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_65, inductor_lookup_seed_8, 'rand');  _shape_param_65 = inductor_lookup_seed_8 = None
        convert_element_type_51: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_8, torch.bfloat16);  inductor_random_8 = None
        gt_8: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_51, 0.1);  convert_element_type_51 = None
        mul_32: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_8, view_55);  view_55 = None
        mul_33: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_32, 1.1111111111111112);  mul_32 = None
        add_17: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_33, add_16);  mul_33 = add_16 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_10: "f32[8, 512, 1]" = var_mean_5[0]
        getitem_11: "f32[8, 512, 1]" = var_mean_5[1];  var_mean_5 = None
        add_18: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-07);  getitem_10 = None
        rsqrt_5: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_8: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  add_17 = getitem_11 = None
        mul_34: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_35: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_34, arg46_1);  arg46_1 = None
        add_19: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_35, arg47_1);  mul_35 = arg47_1 = None
        convert_element_type_52: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.bfloat16);  arg48_1 = None
        convert_element_type_53: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.bfloat16);  arg49_1 = None
        convert_element_type_54: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16)
        view_56: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_54, _shape_param_66);  convert_element_type_54 = _shape_param_66 = None
        permute_31: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_53, [1, 0]);  convert_element_type_53 = None
        addmm_16: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_52, view_56, permute_31);  convert_element_type_52 = None
        view_57: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_16, _shape_param_67);  _shape_param_67 = None
        convert_element_type_55: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        mul_36: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.5)
        mul_37: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 0.7071067811865476);  convert_element_type_55 = None
        erf_2: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
        add_20: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_38: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_36, add_20);  mul_36 = add_20 = None
        convert_element_type_56: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16);  mul_38 = None
        convert_element_type_57: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        convert_element_type_58: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.bfloat16);  arg51_1 = None
        view_58: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_56, _shape_param_68);  convert_element_type_56 = _shape_param_68 = None
        permute_32: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_58, [1, 0]);  convert_element_type_58 = None
        addmm_17: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_57, view_58, permute_32);  convert_element_type_57 = None
        view_59: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_17, _shape_param_69);  addmm_17 = _shape_param_69 = None
        inductor_lookup_seed_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 9)
        inductor_random_9: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_70, inductor_lookup_seed_9, 'rand');  _shape_param_70 = inductor_lookup_seed_9 = None
        convert_element_type_59: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_9, torch.bfloat16);  inductor_random_9 = None
        gt_9: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_59, 0.1);  convert_element_type_59 = None
        mul_39: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_9, view_59);  view_59 = None
        mul_40: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_39, 1.1111111111111112);  mul_39 = None
        add_21: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_40, add_19);  mul_40 = add_19 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_12: "f32[8, 512, 1]" = var_mean_6[0]
        getitem_13: "f32[8, 512, 1]" = var_mean_6[1];  var_mean_6 = None
        add_22: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-07);  getitem_12 = None
        rsqrt_6: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_21, getitem_13);  add_21 = getitem_13 = None
        mul_41: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
        mul_42: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_41, arg52_1);  arg52_1 = None
        add_23: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_42, arg53_1);  mul_42 = arg53_1 = None
        convert_element_type_60: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.bfloat16);  arg54_1 = None
        convert_element_type_61: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        convert_element_type_62: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16)
        view_60: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_62, _shape_param_71);  convert_element_type_62 = _shape_param_71 = None
        permute_33: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        addmm_18: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_60, view_60, permute_33);  convert_element_type_60 = None
        view_61: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_18, _shape_param_72);  addmm_18 = _shape_param_72 = None
        view_62: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_61, _shape_param_73);  view_61 = _shape_param_73 = None
        permute_34: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        clone_12: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_63: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_74);  clone_12 = _shape_param_74 = None
        convert_element_type_63: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        convert_element_type_64: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.bfloat16);  arg57_1 = None
        permute_35: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        addmm_19: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_63, view_60, permute_35);  convert_element_type_63 = None
        view_64: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_19, _shape_param_75);  addmm_19 = _shape_param_75 = None
        view_65: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_64, _shape_param_76);  view_64 = _shape_param_76 = None
        permute_36: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
        clone_13: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_66: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_13, _shape_param_77);  clone_13 = _shape_param_77 = None
        convert_element_type_65: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.bfloat16);  arg58_1 = None
        convert_element_type_66: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.bfloat16);  arg59_1 = None
        permute_37: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        addmm_20: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_65, view_60, permute_37);  convert_element_type_65 = None
        view_67: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_20, _shape_param_78);  addmm_20 = _shape_param_78 = None
        view_68: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_67, _shape_param_79);  view_67 = _shape_param_79 = None
        permute_38: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_14: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_69: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_14, _shape_param_80);  clone_14 = _shape_param_80 = None
        permute_39: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_66, [0, 2, 1]);  view_66 = None
        div_6: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_39, full);  permute_39 = None
        bmm_6: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_63, div_6)
        view_70: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_81);  bmm_6 = _shape_param_81 = None
        where_3: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_70);  view_70 = None
        convert_element_type_67: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_3, torch.float32)
        amax_3: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_67, [-1], True)
        sub_10: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_67, amax_3);  convert_element_type_67 = None
        exp_3: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_4: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_7: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None
        inductor_lookup_seed_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 10)
        inductor_random_10: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_82, inductor_lookup_seed_10, 'rand');  _shape_param_82 = inductor_lookup_seed_10 = None
        gt_10: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_10, 0.1);  inductor_random_10 = None
        mul_43: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_10, div_7);  div_7 = None
        mul_44: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_43, 1.1111111111111112);  mul_43 = None
        view_71: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_44, _shape_param_83);  mul_44 = _shape_param_83 = None
        convert_element_type_68: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_71, torch.bfloat16);  view_71 = None
        bmm_7: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_68, view_69)
        view_72: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_84);  bmm_7 = _shape_param_84 = None
        permute_40: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
        clone_15: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_73: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_15, _shape_param_85);  clone_15 = _shape_param_85 = None
        convert_element_type_69: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.bfloat16);  arg60_1 = None
        convert_element_type_70: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        view_74: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_73, _shape_param_86);  view_73 = _shape_param_86 = None
        permute_41: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        addmm_21: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_69, view_74, permute_41);  convert_element_type_69 = None
        view_75: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_21, _shape_param_87);  addmm_21 = _shape_param_87 = None
        inductor_lookup_seed_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11)
        inductor_random_11: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_88, inductor_lookup_seed_11, 'rand');  _shape_param_88 = inductor_lookup_seed_11 = None
        convert_element_type_71: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_11, torch.bfloat16);  inductor_random_11 = None
        gt_11: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_71, 0.1);  convert_element_type_71 = None
        mul_45: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_11, view_75);  view_75 = None
        mul_46: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_45, 1.1111111111111112);  mul_45 = None
        add_24: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_46, add_23);  mul_46 = add_23 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
        getitem_14: "f32[8, 512, 1]" = var_mean_7[0]
        getitem_15: "f32[8, 512, 1]" = var_mean_7[1];  var_mean_7 = None
        add_25: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-07);  getitem_14 = None
        rsqrt_7: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        sub_11: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_24, getitem_15);  add_24 = getitem_15 = None
        mul_47: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_48: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_47, arg62_1);  arg62_1 = None
        add_26: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_48, arg63_1);  mul_48 = arg63_1 = None
        convert_element_type_72: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        convert_element_type_73: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg65_1, torch.bfloat16);  arg65_1 = None
        convert_element_type_74: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16)
        view_76: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_74, _shape_param_89);  convert_element_type_74 = _shape_param_89 = None
        permute_42: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_73, [1, 0]);  convert_element_type_73 = None
        addmm_22: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_72, view_76, permute_42);  convert_element_type_72 = None
        view_77: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_22, _shape_param_90);  _shape_param_90 = None
        convert_element_type_75: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        mul_49: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.5)
        mul_50: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.7071067811865476);  convert_element_type_75 = None
        erf_3: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_27: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_51: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_49, add_27);  mul_49 = add_27 = None
        convert_element_type_76: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_51, torch.bfloat16);  mul_51 = None
        convert_element_type_77: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg66_1, torch.bfloat16);  arg66_1 = None
        convert_element_type_78: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        view_78: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_76, _shape_param_91);  convert_element_type_76 = _shape_param_91 = None
        permute_43: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_78, [1, 0]);  convert_element_type_78 = None
        addmm_23: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_77, view_78, permute_43);  convert_element_type_77 = None
        view_79: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_23, _shape_param_92);  addmm_23 = _shape_param_92 = None
        inductor_lookup_seed_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 12)
        inductor_random_12: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_93, inductor_lookup_seed_12, 'rand');  _shape_param_93 = inductor_lookup_seed_12 = None
        convert_element_type_79: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_12, torch.bfloat16);  inductor_random_12 = None
        gt_12: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_79, 0.1);  convert_element_type_79 = None
        mul_52: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_12, view_79);  view_79 = None
        mul_53: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_52, 1.1111111111111112);  mul_52 = None
        add_28: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_53, add_26);  mul_53 = add_26 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
        getitem_16: "f32[8, 512, 1]" = var_mean_8[0]
        getitem_17: "f32[8, 512, 1]" = var_mean_8[1];  var_mean_8 = None
        add_29: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-07);  getitem_16 = None
        rsqrt_8: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        sub_12: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_28, getitem_17);  add_28 = getitem_17 = None
        mul_54: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
        mul_55: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_54, arg68_1);  arg68_1 = None
        add_30: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_55, arg69_1);  mul_55 = arg69_1 = None
        convert_element_type_80: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg70_1, torch.bfloat16);  arg70_1 = None
        convert_element_type_81: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg71_1, torch.bfloat16);  arg71_1 = None
        convert_element_type_82: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16)
        view_80: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_82, _shape_param_94);  convert_element_type_82 = _shape_param_94 = None
        permute_44: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_81, [1, 0]);  convert_element_type_81 = None
        addmm_24: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_80, view_80, permute_44);  convert_element_type_80 = None
        view_81: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_24, _shape_param_95);  addmm_24 = _shape_param_95 = None
        view_82: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_81, _shape_param_96);  view_81 = _shape_param_96 = None
        permute_45: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_16: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_83: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_97);  clone_16 = _shape_param_97 = None
        convert_element_type_83: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.bfloat16);  arg72_1 = None
        convert_element_type_84: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        permute_46: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        addmm_25: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_83, view_80, permute_46);  convert_element_type_83 = None
        view_84: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_25, _shape_param_98);  addmm_25 = _shape_param_98 = None
        view_85: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_84, _shape_param_99);  view_84 = _shape_param_99 = None
        permute_47: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        clone_17: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_86: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_17, _shape_param_100);  clone_17 = _shape_param_100 = None
        convert_element_type_85: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        convert_element_type_86: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg75_1, torch.bfloat16);  arg75_1 = None
        permute_48: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_86, [1, 0]);  convert_element_type_86 = None
        addmm_26: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_85, view_80, permute_48);  convert_element_type_85 = None
        view_87: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_26, _shape_param_101);  addmm_26 = _shape_param_101 = None
        view_88: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_87, _shape_param_102);  view_87 = _shape_param_102 = None
        permute_49: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        clone_18: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_89: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_103);  clone_18 = _shape_param_103 = None
        permute_50: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None
        div_8: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_50, full);  permute_50 = None
        bmm_8: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_83, div_8)
        view_90: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_8, _shape_param_104);  bmm_8 = _shape_param_104 = None
        where_4: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_90);  view_90 = None
        convert_element_type_87: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_4, torch.float32)
        amax_4: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_87, [-1], True)
        sub_13: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_87, amax_4);  convert_element_type_87 = None
        exp_4: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_5: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_9: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None
        inductor_lookup_seed_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 13)
        inductor_random_13: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_105, inductor_lookup_seed_13, 'rand');  _shape_param_105 = inductor_lookup_seed_13 = None
        gt_13: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_13, 0.1);  inductor_random_13 = None
        mul_56: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_13, div_9);  div_9 = None
        mul_57: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_56, 1.1111111111111112);  mul_56 = None
        view_91: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_57, _shape_param_106);  mul_57 = _shape_param_106 = None
        convert_element_type_88: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_91, torch.bfloat16);  view_91 = None
        bmm_9: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_88, view_89)
        view_92: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_9, _shape_param_107);  bmm_9 = _shape_param_107 = None
        permute_51: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        clone_19: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_93: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_19, _shape_param_108);  clone_19 = _shape_param_108 = None
        convert_element_type_89: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg76_1, torch.bfloat16);  arg76_1 = None
        convert_element_type_90: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.bfloat16);  arg77_1 = None
        view_94: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_93, _shape_param_109);  view_93 = _shape_param_109 = None
        permute_52: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_90, [1, 0]);  convert_element_type_90 = None
        addmm_27: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_89, view_94, permute_52);  convert_element_type_89 = None
        view_95: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_27, _shape_param_110);  addmm_27 = _shape_param_110 = None
        inductor_lookup_seed_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 14)
        inductor_random_14: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_111, inductor_lookup_seed_14, 'rand');  _shape_param_111 = inductor_lookup_seed_14 = None
        convert_element_type_91: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_14, torch.bfloat16);  inductor_random_14 = None
        gt_14: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_91, 0.1);  convert_element_type_91 = None
        mul_58: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_14, view_95);  view_95 = None
        mul_59: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_58, 1.1111111111111112);  mul_58 = None
        add_31: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_59, add_30);  mul_59 = add_30 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
        getitem_18: "f32[8, 512, 1]" = var_mean_9[0]
        getitem_19: "f32[8, 512, 1]" = var_mean_9[1];  var_mean_9 = None
        add_32: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-07);  getitem_18 = None
        rsqrt_9: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_14: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_31, getitem_19);  add_31 = getitem_19 = None
        mul_60: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_61: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_60, arg78_1);  arg78_1 = None
        add_33: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_61, arg79_1);  mul_61 = arg79_1 = None
        convert_element_type_92: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.bfloat16);  arg80_1 = None
        convert_element_type_93: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg81_1, torch.bfloat16);  arg81_1 = None
        convert_element_type_94: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16)
        view_96: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_94, _shape_param_112);  convert_element_type_94 = _shape_param_112 = None
        permute_53: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_93, [1, 0]);  convert_element_type_93 = None
        addmm_28: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_92, view_96, permute_53);  convert_element_type_92 = None
        view_97: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_28, _shape_param_113);  _shape_param_113 = None
        convert_element_type_95: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        mul_62: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_95, 0.5)
        mul_63: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_95, 0.7071067811865476);  convert_element_type_95 = None
        erf_4: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_34: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_64: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_62, add_34);  mul_62 = add_34 = None
        convert_element_type_96: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        convert_element_type_97: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg82_1, torch.bfloat16);  arg82_1 = None
        convert_element_type_98: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.bfloat16);  arg83_1 = None
        view_98: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_96, _shape_param_114);  convert_element_type_96 = _shape_param_114 = None
        permute_54: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_98, [1, 0]);  convert_element_type_98 = None
        addmm_29: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_97, view_98, permute_54);  convert_element_type_97 = None
        view_99: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_29, _shape_param_115);  addmm_29 = _shape_param_115 = None
        inductor_lookup_seed_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 15)
        inductor_random_15: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_116, inductor_lookup_seed_15, 'rand');  _shape_param_116 = inductor_lookup_seed_15 = None
        convert_element_type_99: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_15, torch.bfloat16);  inductor_random_15 = None
        gt_15: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_99, 0.1);  convert_element_type_99 = None
        mul_65: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_15, view_99);  view_99 = None
        mul_66: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_65, 1.1111111111111112);  mul_65 = None
        add_35: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_66, add_33);  mul_66 = add_33 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
        getitem_20: "f32[8, 512, 1]" = var_mean_10[0]
        getitem_21: "f32[8, 512, 1]" = var_mean_10[1];  var_mean_10 = None
        add_36: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-07);  getitem_20 = None
        rsqrt_10: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_15: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_35, getitem_21);  add_35 = getitem_21 = None
        mul_67: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
        mul_68: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_67, arg84_1);  arg84_1 = None
        add_37: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_68, arg85_1);  mul_68 = arg85_1 = None
        convert_element_type_100: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.bfloat16);  arg86_1 = None
        convert_element_type_101: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.bfloat16);  arg87_1 = None
        convert_element_type_102: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16)
        view_100: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_102, _shape_param_117);  convert_element_type_102 = _shape_param_117 = None
        permute_55: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_101, [1, 0]);  convert_element_type_101 = None
        addmm_30: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_100, view_100, permute_55);  convert_element_type_100 = None
        view_101: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_30, _shape_param_118);  addmm_30 = _shape_param_118 = None
        view_102: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_101, _shape_param_119);  view_101 = _shape_param_119 = None
        permute_56: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
        clone_20: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_103: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_120);  clone_20 = _shape_param_120 = None
        convert_element_type_103: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.bfloat16);  arg88_1 = None
        convert_element_type_104: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg89_1, torch.bfloat16);  arg89_1 = None
        permute_57: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_104, [1, 0]);  convert_element_type_104 = None
        addmm_31: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_103, view_100, permute_57);  convert_element_type_103 = None
        view_104: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_31, _shape_param_121);  addmm_31 = _shape_param_121 = None
        view_105: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_104, _shape_param_122);  view_104 = _shape_param_122 = None
        permute_58: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
        clone_21: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_106: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_123);  clone_21 = _shape_param_123 = None
        convert_element_type_105: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg90_1, torch.bfloat16);  arg90_1 = None
        convert_element_type_106: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg91_1, torch.bfloat16);  arg91_1 = None
        permute_59: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_106, [1, 0]);  convert_element_type_106 = None
        addmm_32: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_105, view_100, permute_59);  convert_element_type_105 = None
        view_107: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_32, _shape_param_124);  addmm_32 = _shape_param_124 = None
        view_108: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_107, _shape_param_125);  view_107 = _shape_param_125 = None
        permute_60: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        clone_22: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_109: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_126);  clone_22 = _shape_param_126 = None
        permute_61: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
        div_10: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_61, full);  permute_61 = None
        bmm_10: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_103, div_10)
        view_110: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_10, _shape_param_127);  bmm_10 = _shape_param_127 = None
        where_5: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_110);  view_110 = None
        convert_element_type_107: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_5, torch.float32)
        amax_5: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_107, [-1], True)
        sub_16: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_107, amax_5);  convert_element_type_107 = None
        exp_5: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_6: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_11: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None
        inductor_lookup_seed_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 16)
        inductor_random_16: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_128, inductor_lookup_seed_16, 'rand');  _shape_param_128 = inductor_lookup_seed_16 = None
        gt_16: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_16, 0.1);  inductor_random_16 = None
        mul_69: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_16, div_11);  div_11 = None
        mul_70: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_69, 1.1111111111111112);  mul_69 = None
        view_111: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_70, _shape_param_129);  mul_70 = _shape_param_129 = None
        convert_element_type_108: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_111, torch.bfloat16);  view_111 = None
        bmm_11: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_108, view_109)
        view_112: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_130);  bmm_11 = _shape_param_130 = None
        permute_62: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        clone_23: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_113: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_23, _shape_param_131);  clone_23 = _shape_param_131 = None
        convert_element_type_109: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.bfloat16);  arg92_1 = None
        convert_element_type_110: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        view_114: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_113, _shape_param_132);  view_113 = _shape_param_132 = None
        permute_63: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_110, [1, 0]);  convert_element_type_110 = None
        addmm_33: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_109, view_114, permute_63);  convert_element_type_109 = None
        view_115: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_33, _shape_param_133);  addmm_33 = _shape_param_133 = None
        inductor_lookup_seed_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 17)
        inductor_random_17: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_134, inductor_lookup_seed_17, 'rand');  _shape_param_134 = inductor_lookup_seed_17 = None
        convert_element_type_111: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_17, torch.bfloat16);  inductor_random_17 = None
        gt_17: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_111, 0.1);  convert_element_type_111 = None
        mul_71: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_17, view_115);  view_115 = None
        mul_72: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_71, 1.1111111111111112);  mul_71 = None
        add_38: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_72, add_37);  mul_72 = add_37 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
        getitem_22: "f32[8, 512, 1]" = var_mean_11[0]
        getitem_23: "f32[8, 512, 1]" = var_mean_11[1];  var_mean_11 = None
        add_39: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-07);  getitem_22 = None
        rsqrt_11: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        sub_17: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_38, getitem_23);  add_38 = getitem_23 = None
        mul_73: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_74: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_73, arg94_1);  arg94_1 = None
        add_40: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_74, arg95_1);  mul_74 = arg95_1 = None
        convert_element_type_112: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.bfloat16);  arg96_1 = None
        convert_element_type_113: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg97_1, torch.bfloat16);  arg97_1 = None
        convert_element_type_114: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16)
        view_116: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_114, _shape_param_135);  convert_element_type_114 = _shape_param_135 = None
        permute_64: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_113, [1, 0]);  convert_element_type_113 = None
        addmm_34: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_112, view_116, permute_64);  convert_element_type_112 = None
        view_117: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_34, _shape_param_136);  _shape_param_136 = None
        convert_element_type_115: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_117, torch.float32);  view_117 = None
        mul_75: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5)
        mul_76: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.7071067811865476);  convert_element_type_115 = None
        erf_5: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
        add_41: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_77: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_75, add_41);  mul_75 = add_41 = None
        convert_element_type_116: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        convert_element_type_117: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.bfloat16);  arg98_1 = None
        convert_element_type_118: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.bfloat16);  arg99_1 = None
        view_118: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_116, _shape_param_137);  convert_element_type_116 = _shape_param_137 = None
        permute_65: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_118, [1, 0]);  convert_element_type_118 = None
        addmm_35: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_117, view_118, permute_65);  convert_element_type_117 = None
        view_119: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_35, _shape_param_138);  addmm_35 = _shape_param_138 = None
        inductor_lookup_seed_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 18)
        inductor_random_18: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_139, inductor_lookup_seed_18, 'rand');  _shape_param_139 = inductor_lookup_seed_18 = None
        convert_element_type_119: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_18, torch.bfloat16);  inductor_random_18 = None
        gt_18: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_119, 0.1);  convert_element_type_119 = None
        mul_78: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_18, view_119);  view_119 = None
        mul_79: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_78, 1.1111111111111112);  mul_78 = None
        add_42: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_79, add_40);  mul_79 = add_40 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
        getitem_24: "f32[8, 512, 1]" = var_mean_12[0]
        getitem_25: "f32[8, 512, 1]" = var_mean_12[1];  var_mean_12 = None
        add_43: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-07);  getitem_24 = None
        rsqrt_12: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_18: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_42, getitem_25);  add_42 = getitem_25 = None
        mul_80: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
        mul_81: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_80, arg100_1);  arg100_1 = None
        add_44: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_81, arg101_1);  mul_81 = arg101_1 = None
        convert_element_type_120: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        convert_element_type_121: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg103_1, torch.bfloat16);  arg103_1 = None
        convert_element_type_122: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16)
        view_120: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_122, _shape_param_140);  convert_element_type_122 = _shape_param_140 = None
        permute_66: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        addmm_36: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_120, view_120, permute_66);  convert_element_type_120 = None
        view_121: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_36, _shape_param_141);  addmm_36 = _shape_param_141 = None
        view_122: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_121, _shape_param_142);  view_121 = _shape_param_142 = None
        permute_67: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        clone_24: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_123: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_143);  clone_24 = _shape_param_143 = None
        convert_element_type_123: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg104_1, torch.bfloat16);  arg104_1 = None
        convert_element_type_124: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg105_1, torch.bfloat16);  arg105_1 = None
        permute_68: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_124, [1, 0]);  convert_element_type_124 = None
        addmm_37: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_123, view_120, permute_68);  convert_element_type_123 = None
        view_124: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_37, _shape_param_144);  addmm_37 = _shape_param_144 = None
        view_125: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_124, _shape_param_145);  view_124 = _shape_param_145 = None
        permute_69: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        clone_25: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_126: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_25, _shape_param_146);  clone_25 = _shape_param_146 = None
        convert_element_type_125: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.bfloat16);  arg106_1 = None
        convert_element_type_126: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        permute_70: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_126, [1, 0]);  convert_element_type_126 = None
        addmm_38: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_125, view_120, permute_70);  convert_element_type_125 = None
        view_127: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_38, _shape_param_147);  addmm_38 = _shape_param_147 = None
        view_128: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_127, _shape_param_148);  view_127 = _shape_param_148 = None
        permute_71: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        clone_26: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_129: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_26, _shape_param_149);  clone_26 = _shape_param_149 = None
        permute_72: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
        div_12: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_72, full);  permute_72 = None
        bmm_12: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_123, div_12)
        view_130: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_12, _shape_param_150);  bmm_12 = _shape_param_150 = None
        where_6: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_130);  view_130 = None
        convert_element_type_127: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_6, torch.float32)
        amax_6: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_127, [-1], True)
        sub_19: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_127, amax_6);  convert_element_type_127 = None
        exp_6: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_7: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_13: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None
        inductor_lookup_seed_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 19)
        inductor_random_19: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_151, inductor_lookup_seed_19, 'rand');  _shape_param_151 = inductor_lookup_seed_19 = None
        gt_19: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_19, 0.1);  inductor_random_19 = None
        mul_82: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_19, div_13);  div_13 = None
        mul_83: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_82, 1.1111111111111112);  mul_82 = None
        view_131: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_83, _shape_param_152);  mul_83 = _shape_param_152 = None
        convert_element_type_128: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_131, torch.bfloat16);  view_131 = None
        bmm_13: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_128, view_129)
        view_132: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_13, _shape_param_153);  bmm_13 = _shape_param_153 = None
        permute_73: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
        clone_27: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_133: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_27, _shape_param_154);  clone_27 = _shape_param_154 = None
        convert_element_type_129: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.bfloat16);  arg108_1 = None
        convert_element_type_130: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        view_134: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_133, _shape_param_155);  view_133 = _shape_param_155 = None
        permute_74: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        addmm_39: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_129, view_134, permute_74);  convert_element_type_129 = None
        view_135: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_39, _shape_param_156);  addmm_39 = _shape_param_156 = None
        inductor_lookup_seed_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 20)
        inductor_random_20: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_157, inductor_lookup_seed_20, 'rand');  _shape_param_157 = inductor_lookup_seed_20 = None
        convert_element_type_131: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_20, torch.bfloat16);  inductor_random_20 = None
        gt_20: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_131, 0.1);  convert_element_type_131 = None
        mul_84: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_20, view_135);  view_135 = None
        mul_85: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_84, 1.1111111111111112);  mul_84 = None
        add_45: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_85, add_44);  mul_85 = add_44 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_26: "f32[8, 512, 1]" = var_mean_13[0]
        getitem_27: "f32[8, 512, 1]" = var_mean_13[1];  var_mean_13 = None
        add_46: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-07);  getitem_26 = None
        rsqrt_13: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_20: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_45, getitem_27);  add_45 = getitem_27 = None
        mul_86: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_87: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_86, arg110_1);  arg110_1 = None
        add_47: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_87, arg111_1);  mul_87 = arg111_1 = None
        convert_element_type_132: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        convert_element_type_133: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.bfloat16);  arg113_1 = None
        convert_element_type_134: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16)
        view_136: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_134, _shape_param_158);  convert_element_type_134 = _shape_param_158 = None
        permute_75: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_133, [1, 0]);  convert_element_type_133 = None
        addmm_40: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_132, view_136, permute_75);  convert_element_type_132 = None
        view_137: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_40, _shape_param_159);  _shape_param_159 = None
        convert_element_type_135: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        mul_88: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_135, 0.5)
        mul_89: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_135, 0.7071067811865476);  convert_element_type_135 = None
        erf_6: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_48: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_90: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_88, add_48);  mul_88 = add_48 = None
        convert_element_type_136: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None
        convert_element_type_137: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.bfloat16);  arg114_1 = None
        convert_element_type_138: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg115_1, torch.bfloat16);  arg115_1 = None
        view_138: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_136, _shape_param_160);  convert_element_type_136 = _shape_param_160 = None
        permute_76: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_138, [1, 0]);  convert_element_type_138 = None
        addmm_41: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_137, view_138, permute_76);  convert_element_type_137 = None
        view_139: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_41, _shape_param_161);  addmm_41 = _shape_param_161 = None
        inductor_lookup_seed_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 21)
        inductor_random_21: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_162, inductor_lookup_seed_21, 'rand');  _shape_param_162 = inductor_lookup_seed_21 = None
        convert_element_type_139: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_21, torch.bfloat16);  inductor_random_21 = None
        gt_21: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_139, 0.1);  convert_element_type_139 = None
        mul_91: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_21, view_139);  view_139 = None
        mul_92: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_91, 1.1111111111111112);  mul_91 = None
        add_49: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_92, add_47);  mul_92 = add_47 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_28: "f32[8, 512, 1]" = var_mean_14[0]
        getitem_29: "f32[8, 512, 1]" = var_mean_14[1];  var_mean_14 = None
        add_50: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-07);  getitem_28 = None
        rsqrt_14: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_21: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_49, getitem_29);  add_49 = getitem_29 = None
        mul_93: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
        mul_94: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_93, arg116_1);  arg116_1 = None
        add_51: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_94, arg117_1);  mul_94 = arg117_1 = None
        convert_element_type_140: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.bfloat16);  arg118_1 = None
        convert_element_type_141: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg119_1, torch.bfloat16);  arg119_1 = None
        convert_element_type_142: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16)
        view_140: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_142, _shape_param_163);  convert_element_type_142 = _shape_param_163 = None
        permute_77: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        addmm_42: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_140, view_140, permute_77);  convert_element_type_140 = None
        view_141: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_42, _shape_param_164);  addmm_42 = _shape_param_164 = None
        view_142: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_141, _shape_param_165);  view_141 = _shape_param_165 = None
        permute_78: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        clone_28: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_143: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_166);  clone_28 = _shape_param_166 = None
        convert_element_type_143: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg120_1, torch.bfloat16);  arg120_1 = None
        convert_element_type_144: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.bfloat16);  arg121_1 = None
        permute_79: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        addmm_43: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_143, view_140, permute_79);  convert_element_type_143 = None
        view_144: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_43, _shape_param_167);  addmm_43 = _shape_param_167 = None
        view_145: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_144, _shape_param_168);  view_144 = _shape_param_168 = None
        permute_80: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
        clone_29: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
        view_146: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_29, _shape_param_169);  clone_29 = _shape_param_169 = None
        convert_element_type_145: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        convert_element_type_146: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.bfloat16);  arg123_1 = None
        permute_81: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_146, [1, 0]);  convert_element_type_146 = None
        addmm_44: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_145, view_140, permute_81);  convert_element_type_145 = None
        view_147: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_44, _shape_param_170);  addmm_44 = _shape_param_170 = None
        view_148: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_147, _shape_param_171);  view_147 = _shape_param_171 = None
        permute_82: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_30: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_149: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_172);  clone_30 = _shape_param_172 = None
        permute_83: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
        div_14: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_83, full);  permute_83 = None
        bmm_14: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_143, div_14)
        view_150: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_14, _shape_param_173);  bmm_14 = _shape_param_173 = None
        where_7: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_150);  view_150 = None
        convert_element_type_147: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_7, torch.float32)
        amax_7: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_147, [-1], True)
        sub_22: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_147, amax_7);  convert_element_type_147 = None
        exp_7: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_8: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_15: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None
        inductor_lookup_seed_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22)
        inductor_random_22: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_174, inductor_lookup_seed_22, 'rand');  _shape_param_174 = inductor_lookup_seed_22 = None
        gt_22: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_22, 0.1);  inductor_random_22 = None
        mul_95: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_22, div_15);  div_15 = None
        mul_96: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_95, 1.1111111111111112);  mul_95 = None
        view_151: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_96, _shape_param_175);  mul_96 = _shape_param_175 = None
        convert_element_type_148: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_151, torch.bfloat16);  view_151 = None
        bmm_15: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_148, view_149)
        view_152: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_176);  bmm_15 = _shape_param_176 = None
        permute_84: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        clone_31: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_153: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_31, _shape_param_177);  clone_31 = _shape_param_177 = None
        convert_element_type_149: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg124_1, torch.bfloat16);  arg124_1 = None
        convert_element_type_150: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.bfloat16);  arg125_1 = None
        view_154: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_153, _shape_param_178);  view_153 = _shape_param_178 = None
        permute_85: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_150, [1, 0]);  convert_element_type_150 = None
        addmm_45: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_149, view_154, permute_85);  convert_element_type_149 = None
        view_155: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_45, _shape_param_179);  addmm_45 = _shape_param_179 = None
        inductor_lookup_seed_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 23)
        inductor_random_23: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_180, inductor_lookup_seed_23, 'rand');  _shape_param_180 = inductor_lookup_seed_23 = None
        convert_element_type_151: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_23, torch.bfloat16);  inductor_random_23 = None
        gt_23: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_151, 0.1);  convert_element_type_151 = None
        mul_97: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_23, view_155);  view_155 = None
        mul_98: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_97, 1.1111111111111112);  mul_97 = None
        add_52: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_98, add_51);  mul_98 = add_51 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_30: "f32[8, 512, 1]" = var_mean_15[0]
        getitem_31: "f32[8, 512, 1]" = var_mean_15[1];  var_mean_15 = None
        add_53: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-07);  getitem_30 = None
        rsqrt_15: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_23: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_52, getitem_31);  add_52 = getitem_31 = None
        mul_99: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_100: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_99, arg126_1);  arg126_1 = None
        add_54: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_100, arg127_1);  mul_100 = arg127_1 = None
        convert_element_type_152: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        convert_element_type_153: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg129_1, torch.bfloat16);  arg129_1 = None
        convert_element_type_154: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16)
        view_156: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_154, _shape_param_181);  convert_element_type_154 = _shape_param_181 = None
        permute_86: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_153, [1, 0]);  convert_element_type_153 = None
        addmm_46: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_152, view_156, permute_86);  convert_element_type_152 = None
        view_157: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_46, _shape_param_182);  _shape_param_182 = None
        convert_element_type_155: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        mul_101: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.5)
        mul_102: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.7071067811865476);  convert_element_type_155 = None
        erf_7: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_102);  mul_102 = None
        add_55: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_103: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_101, add_55);  mul_101 = add_55 = None
        convert_element_type_156: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_103, torch.bfloat16);  mul_103 = None
        convert_element_type_157: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg130_1, torch.bfloat16);  arg130_1 = None
        convert_element_type_158: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.bfloat16);  arg131_1 = None
        view_158: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_156, _shape_param_183);  convert_element_type_156 = _shape_param_183 = None
        permute_87: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_158, [1, 0]);  convert_element_type_158 = None
        addmm_47: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_157, view_158, permute_87);  convert_element_type_157 = None
        view_159: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_47, _shape_param_184);  addmm_47 = _shape_param_184 = None
        inductor_lookup_seed_24: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 24)
        inductor_random_24: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_185, inductor_lookup_seed_24, 'rand');  _shape_param_185 = inductor_lookup_seed_24 = None
        convert_element_type_159: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_24, torch.bfloat16);  inductor_random_24 = None
        gt_24: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_159, 0.1);  convert_element_type_159 = None
        mul_104: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_24, view_159);  view_159 = None
        mul_105: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_104, 1.1111111111111112);  mul_104 = None
        add_56: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_105, add_54);  mul_105 = add_54 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
        getitem_32: "f32[8, 512, 1]" = var_mean_16[0]
        getitem_33: "f32[8, 512, 1]" = var_mean_16[1];  var_mean_16 = None
        add_57: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-07);  getitem_32 = None
        rsqrt_16: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        sub_24: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_56, getitem_33);  add_56 = getitem_33 = None
        mul_106: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
        mul_107: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_106, arg132_1);  arg132_1 = None
        add_58: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_107, arg133_1);  mul_107 = arg133_1 = None
        convert_element_type_160: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.bfloat16);  arg134_1 = None
        convert_element_type_161: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.bfloat16);  arg135_1 = None
        convert_element_type_162: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_58, torch.bfloat16)
        view_160: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_162, _shape_param_186);  convert_element_type_162 = _shape_param_186 = None
        permute_88: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_161, [1, 0]);  convert_element_type_161 = None
        addmm_48: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_160, view_160, permute_88);  convert_element_type_160 = None
        view_161: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_48, _shape_param_187);  addmm_48 = _shape_param_187 = None
        view_162: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_161, _shape_param_188);  view_161 = _shape_param_188 = None
        permute_89: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        clone_32: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_163: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_189);  clone_32 = _shape_param_189 = None
        convert_element_type_163: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg136_1, torch.bfloat16);  arg136_1 = None
        convert_element_type_164: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        permute_90: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        addmm_49: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_163, view_160, permute_90);  convert_element_type_163 = None
        view_164: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_49, _shape_param_190);  addmm_49 = _shape_param_190 = None
        view_165: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_164, _shape_param_191);  view_164 = _shape_param_191 = None
        permute_91: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        clone_33: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_166: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_33, _shape_param_192);  clone_33 = _shape_param_192 = None
        convert_element_type_165: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.bfloat16);  arg138_1 = None
        convert_element_type_166: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg139_1, torch.bfloat16);  arg139_1 = None
        permute_92: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_166, [1, 0]);  convert_element_type_166 = None
        addmm_50: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_165, view_160, permute_92);  convert_element_type_165 = None
        view_167: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_50, _shape_param_193);  addmm_50 = _shape_param_193 = None
        view_168: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_167, _shape_param_194);  view_167 = _shape_param_194 = None
        permute_93: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_34: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_169: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_195);  clone_34 = _shape_param_195 = None
        permute_94: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        div_16: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_94, full);  permute_94 = None
        bmm_16: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_163, div_16)
        view_170: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_16, _shape_param_196);  bmm_16 = _shape_param_196 = None
        where_8: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_170);  view_170 = None
        convert_element_type_167: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_8, torch.float32)
        amax_8: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_167, [-1], True)
        sub_25: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_167, amax_8);  convert_element_type_167 = None
        exp_8: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_9: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_17: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None
        inductor_lookup_seed_25: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 25)
        inductor_random_25: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_197, inductor_lookup_seed_25, 'rand');  _shape_param_197 = inductor_lookup_seed_25 = None
        gt_25: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_25, 0.1);  inductor_random_25 = None
        mul_108: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_25, div_17);  div_17 = None
        mul_109: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_108, 1.1111111111111112);  mul_108 = None
        view_171: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_109, _shape_param_198);  mul_109 = _shape_param_198 = None
        convert_element_type_168: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_171, torch.bfloat16);  view_171 = None
        bmm_17: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_168, view_169)
        view_172: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_199);  bmm_17 = _shape_param_199 = None
        permute_95: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        clone_35: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_173: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_35, _shape_param_200);  clone_35 = _shape_param_200 = None
        convert_element_type_169: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.bfloat16);  arg140_1 = None
        convert_element_type_170: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.bfloat16);  arg141_1 = None
        view_174: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_173, _shape_param_201);  view_173 = _shape_param_201 = None
        permute_96: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_170, [1, 0]);  convert_element_type_170 = None
        addmm_51: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_169, view_174, permute_96);  convert_element_type_169 = None
        view_175: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_51, _shape_param_202);  addmm_51 = _shape_param_202 = None
        inductor_lookup_seed_26: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 26)
        inductor_random_26: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_203, inductor_lookup_seed_26, 'rand');  _shape_param_203 = inductor_lookup_seed_26 = None
        convert_element_type_171: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_26, torch.bfloat16);  inductor_random_26 = None
        gt_26: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_171, 0.1);  convert_element_type_171 = None
        mul_110: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_26, view_175);  view_175 = None
        mul_111: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_110, 1.1111111111111112);  mul_110 = None
        add_59: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_111, add_58);  mul_111 = add_58 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_34: "f32[8, 512, 1]" = var_mean_17[0]
        getitem_35: "f32[8, 512, 1]" = var_mean_17[1];  var_mean_17 = None
        add_60: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-07);  getitem_34 = None
        rsqrt_17: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_26: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_59, getitem_35);  add_59 = getitem_35 = None
        mul_112: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_113: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_112, arg142_1);  arg142_1 = None
        add_61: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_113, arg143_1);  mul_113 = arg143_1 = None
        convert_element_type_172: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.bfloat16);  arg144_1 = None
        convert_element_type_173: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg145_1, torch.bfloat16);  arg145_1 = None
        convert_element_type_174: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        view_176: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_174, _shape_param_204);  convert_element_type_174 = _shape_param_204 = None
        permute_97: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_173, [1, 0]);  convert_element_type_173 = None
        addmm_52: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_172, view_176, permute_97);  convert_element_type_172 = None
        view_177: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_52, _shape_param_205);  _shape_param_205 = None
        convert_element_type_175: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        mul_114: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.5)
        mul_115: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.7071067811865476);  convert_element_type_175 = None
        erf_8: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
        add_62: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_116: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_114, add_62);  mul_114 = add_62 = None
        convert_element_type_176: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_116, torch.bfloat16);  mul_116 = None
        convert_element_type_177: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.bfloat16);  arg146_1 = None
        convert_element_type_178: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.bfloat16);  arg147_1 = None
        view_178: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_176, _shape_param_206);  convert_element_type_176 = _shape_param_206 = None
        permute_98: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_178, [1, 0]);  convert_element_type_178 = None
        addmm_53: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_177, view_178, permute_98);  convert_element_type_177 = None
        view_179: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_53, _shape_param_207);  addmm_53 = _shape_param_207 = None
        inductor_lookup_seed_27: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 27)
        inductor_random_27: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_208, inductor_lookup_seed_27, 'rand');  _shape_param_208 = inductor_lookup_seed_27 = None
        convert_element_type_179: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_27, torch.bfloat16);  inductor_random_27 = None
        gt_27: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_179, 0.1);  convert_element_type_179 = None
        mul_117: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_27, view_179);  view_179 = None
        mul_118: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_117, 1.1111111111111112);  mul_117 = None
        add_63: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_118, add_61);  mul_118 = add_61 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_36: "f32[8, 512, 1]" = var_mean_18[0]
        getitem_37: "f32[8, 512, 1]" = var_mean_18[1];  var_mean_18 = None
        add_64: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-07);  getitem_36 = None
        rsqrt_18: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_27: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_63, getitem_37);  add_63 = getitem_37 = None
        mul_119: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
        mul_120: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_119, arg148_1);  arg148_1 = None
        add_65: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_120, arg149_1);  mul_120 = arg149_1 = None
        convert_element_type_180: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg150_1, torch.bfloat16);  arg150_1 = None
        convert_element_type_181: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.bfloat16);  arg151_1 = None
        convert_element_type_182: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16)
        view_180: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_182, _shape_param_209);  convert_element_type_182 = _shape_param_209 = None
        permute_99: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_181, [1, 0]);  convert_element_type_181 = None
        addmm_54: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_180, view_180, permute_99);  convert_element_type_180 = None
        view_181: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_54, _shape_param_210);  addmm_54 = _shape_param_210 = None
        view_182: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_181, _shape_param_211);  view_181 = _shape_param_211 = None
        permute_100: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_36: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_183: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_212);  clone_36 = _shape_param_212 = None
        convert_element_type_183: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.bfloat16);  arg152_1 = None
        convert_element_type_184: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.bfloat16);  arg153_1 = None
        permute_101: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_184, [1, 0]);  convert_element_type_184 = None
        addmm_55: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_183, view_180, permute_101);  convert_element_type_183 = None
        view_184: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_55, _shape_param_213);  addmm_55 = _shape_param_213 = None
        view_185: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_184, _shape_param_214);  view_184 = _shape_param_214 = None
        permute_102: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_37: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
        view_186: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_37, _shape_param_215);  clone_37 = _shape_param_215 = None
        convert_element_type_185: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.bfloat16);  arg154_1 = None
        convert_element_type_186: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.bfloat16);  arg155_1 = None
        permute_103: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_186, [1, 0]);  convert_element_type_186 = None
        addmm_56: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_185, view_180, permute_103);  convert_element_type_185 = None
        view_187: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_56, _shape_param_216);  addmm_56 = _shape_param_216 = None
        view_188: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_187, _shape_param_217);  view_187 = _shape_param_217 = None
        permute_104: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        clone_38: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_189: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_38, _shape_param_218);  clone_38 = _shape_param_218 = None
        permute_105: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
        div_18: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_105, full);  permute_105 = None
        bmm_18: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_183, div_18)
        view_190: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_219);  bmm_18 = _shape_param_219 = None
        where_9: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_190);  view_190 = None
        convert_element_type_187: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_9, torch.float32)
        amax_9: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_187, [-1], True)
        sub_28: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_187, amax_9);  convert_element_type_187 = None
        exp_9: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        sum_10: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_19: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None
        inductor_lookup_seed_28: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 28)
        inductor_random_28: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_220, inductor_lookup_seed_28, 'rand');  _shape_param_220 = inductor_lookup_seed_28 = None
        gt_28: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_28, 0.1);  inductor_random_28 = None
        mul_121: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_28, div_19);  div_19 = None
        mul_122: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None
        view_191: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_122, _shape_param_221);  mul_122 = _shape_param_221 = None
        convert_element_type_188: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_191, torch.bfloat16);  view_191 = None
        bmm_19: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_188, view_189)
        view_192: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_222);  bmm_19 = _shape_param_222 = None
        permute_106: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        clone_39: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_193: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_39, _shape_param_223);  clone_39 = _shape_param_223 = None
        convert_element_type_189: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg156_1, torch.bfloat16);  arg156_1 = None
        convert_element_type_190: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        view_194: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_193, _shape_param_224);  view_193 = _shape_param_224 = None
        permute_107: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_190, [1, 0]);  convert_element_type_190 = None
        addmm_57: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_189, view_194, permute_107);  convert_element_type_189 = None
        view_195: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_57, _shape_param_225);  addmm_57 = _shape_param_225 = None
        inductor_lookup_seed_29: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 29)
        inductor_random_29: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_226, inductor_lookup_seed_29, 'rand');  _shape_param_226 = inductor_lookup_seed_29 = None
        convert_element_type_191: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_29, torch.bfloat16);  inductor_random_29 = None
        gt_29: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_191, 0.1);  convert_element_type_191 = None
        mul_123: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_29, view_195);  view_195 = None
        mul_124: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_123, 1.1111111111111112);  mul_123 = None
        add_66: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_124, add_65);  mul_124 = add_65 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_38: "f32[8, 512, 1]" = var_mean_19[0]
        getitem_39: "f32[8, 512, 1]" = var_mean_19[1];  var_mean_19 = None
        add_67: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-07);  getitem_38 = None
        rsqrt_19: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        sub_29: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_66, getitem_39);  add_66 = getitem_39 = None
        mul_125: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_126: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_125, arg158_1);  arg158_1 = None
        add_68: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_126, arg159_1);  mul_126 = arg159_1 = None
        convert_element_type_192: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg160_1, torch.bfloat16);  arg160_1 = None
        convert_element_type_193: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg161_1, torch.bfloat16);  arg161_1 = None
        convert_element_type_194: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16)
        view_196: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_194, _shape_param_227);  convert_element_type_194 = _shape_param_227 = None
        permute_108: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None
        addmm_58: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_192, view_196, permute_108);  convert_element_type_192 = None
        view_197: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_58, _shape_param_228);  _shape_param_228 = None
        convert_element_type_195: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        mul_127: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_195, 0.5)
        mul_128: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_195, 0.7071067811865476);  convert_element_type_195 = None
        erf_9: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
        add_69: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_129: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_127, add_69);  mul_127 = add_69 = None
        convert_element_type_196: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        convert_element_type_197: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.bfloat16);  arg162_1 = None
        convert_element_type_198: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.bfloat16);  arg163_1 = None
        view_198: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_196, _shape_param_229);  convert_element_type_196 = _shape_param_229 = None
        permute_109: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_198, [1, 0]);  convert_element_type_198 = None
        addmm_59: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_197, view_198, permute_109);  convert_element_type_197 = None
        view_199: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_59, _shape_param_230);  addmm_59 = _shape_param_230 = None
        inductor_lookup_seed_30: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 30)
        inductor_random_30: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_231, inductor_lookup_seed_30, 'rand');  _shape_param_231 = inductor_lookup_seed_30 = None
        convert_element_type_199: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_30, torch.bfloat16);  inductor_random_30 = None
        gt_30: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_199, 0.1);  convert_element_type_199 = None
        mul_130: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_30, view_199);  view_199 = None
        mul_131: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_130, 1.1111111111111112);  mul_130 = None
        add_70: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_131, add_68);  mul_131 = add_68 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_40: "f32[8, 512, 1]" = var_mean_20[0]
        getitem_41: "f32[8, 512, 1]" = var_mean_20[1];  var_mean_20 = None
        add_71: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-07);  getitem_40 = None
        rsqrt_20: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_30: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_70, getitem_41);  add_70 = getitem_41 = None
        mul_132: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
        mul_133: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_132, arg164_1);  arg164_1 = None
        add_72: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_133, arg165_1);  mul_133 = arg165_1 = None
        convert_element_type_200: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg166_1, torch.bfloat16);  arg166_1 = None
        convert_element_type_201: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        convert_element_type_202: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16)
        view_200: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_202, _shape_param_232);  convert_element_type_202 = _shape_param_232 = None
        permute_110: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_201, [1, 0]);  convert_element_type_201 = None
        addmm_60: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_200, view_200, permute_110);  convert_element_type_200 = None
        view_201: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_60, _shape_param_233);  addmm_60 = _shape_param_233 = None
        view_202: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_201, _shape_param_234);  view_201 = _shape_param_234 = None
        permute_111: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        clone_40: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_111, memory_format = torch.contiguous_format);  permute_111 = None
        view_203: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_235);  clone_40 = _shape_param_235 = None
        convert_element_type_203: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.bfloat16);  arg168_1 = None
        convert_element_type_204: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg169_1, torch.bfloat16);  arg169_1 = None
        permute_112: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_204, [1, 0]);  convert_element_type_204 = None
        addmm_61: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_203, view_200, permute_112);  convert_element_type_203 = None
        view_204: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_61, _shape_param_236);  addmm_61 = _shape_param_236 = None
        view_205: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_204, _shape_param_237);  view_204 = _shape_param_237 = None
        permute_113: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        clone_41: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_206: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_41, _shape_param_238);  clone_41 = _shape_param_238 = None
        convert_element_type_205: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.bfloat16);  arg170_1 = None
        convert_element_type_206: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.bfloat16);  arg171_1 = None
        permute_114: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        addmm_62: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_205, view_200, permute_114);  convert_element_type_205 = None
        view_207: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_62, _shape_param_239);  addmm_62 = _shape_param_239 = None
        view_208: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_207, _shape_param_240);  view_207 = _shape_param_240 = None
        permute_115: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        clone_42: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_209: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_241);  clone_42 = _shape_param_241 = None
        permute_116: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
        div_20: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_116, full);  permute_116 = None
        bmm_20: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_203, div_20)
        view_210: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_20, _shape_param_242);  bmm_20 = _shape_param_242 = None
        where_10: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_210);  view_210 = None
        convert_element_type_207: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_10, torch.float32)
        amax_10: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_207, [-1], True)
        sub_31: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_207, amax_10);  convert_element_type_207 = None
        exp_10: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
        sum_11: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_21: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None
        inductor_lookup_seed_31: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 31)
        inductor_random_31: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_243, inductor_lookup_seed_31, 'rand');  _shape_param_243 = inductor_lookup_seed_31 = None
        gt_31: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_31, 0.1);  inductor_random_31 = None
        mul_134: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_31, div_21);  div_21 = None
        mul_135: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_134, 1.1111111111111112);  mul_134 = None
        view_211: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_135, _shape_param_244);  mul_135 = _shape_param_244 = None
        convert_element_type_208: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_211, torch.bfloat16);  view_211 = None
        bmm_21: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_208, view_209)
        view_212: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_21, _shape_param_245);  bmm_21 = _shape_param_245 = None
        permute_117: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
        clone_43: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_213: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_43, _shape_param_246);  clone_43 = _shape_param_246 = None
        convert_element_type_209: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.bfloat16);  arg172_1 = None
        convert_element_type_210: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.bfloat16);  arg173_1 = None
        view_214: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_213, _shape_param_247);  view_213 = _shape_param_247 = None
        permute_118: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_210, [1, 0]);  convert_element_type_210 = None
        addmm_63: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_209, view_214, permute_118);  convert_element_type_209 = None
        view_215: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_63, _shape_param_248);  addmm_63 = _shape_param_248 = None
        inductor_lookup_seed_32: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 32)
        inductor_random_32: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_249, inductor_lookup_seed_32, 'rand');  _shape_param_249 = inductor_lookup_seed_32 = None
        convert_element_type_211: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_32, torch.bfloat16);  inductor_random_32 = None
        gt_32: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_211, 0.1);  convert_element_type_211 = None
        mul_136: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_32, view_215);  view_215 = None
        mul_137: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_136, 1.1111111111111112);  mul_136 = None
        add_73: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_137, add_72);  mul_137 = add_72 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_42: "f32[8, 512, 1]" = var_mean_21[0]
        getitem_43: "f32[8, 512, 1]" = var_mean_21[1];  var_mean_21 = None
        add_74: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-07);  getitem_42 = None
        rsqrt_21: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_32: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_73, getitem_43);  add_73 = getitem_43 = None
        mul_138: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_139: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_138, arg174_1);  arg174_1 = None
        add_75: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_139, arg175_1);  mul_139 = arg175_1 = None
        convert_element_type_212: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.bfloat16);  arg176_1 = None
        convert_element_type_213: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        convert_element_type_214: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16)
        view_216: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_214, _shape_param_250);  convert_element_type_214 = _shape_param_250 = None
        permute_119: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_213, [1, 0]);  convert_element_type_213 = None
        addmm_64: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_212, view_216, permute_119);  convert_element_type_212 = None
        view_217: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_64, _shape_param_251);  _shape_param_251 = None
        convert_element_type_215: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_140: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_215, 0.5)
        mul_141: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_215, 0.7071067811865476);  convert_element_type_215 = None
        erf_10: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_76: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_142: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_140, add_76);  mul_140 = add_76 = None
        convert_element_type_216: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        convert_element_type_217: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg178_1, torch.bfloat16);  arg178_1 = None
        convert_element_type_218: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.bfloat16);  arg179_1 = None
        view_218: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_216, _shape_param_252);  convert_element_type_216 = _shape_param_252 = None
        permute_120: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        addmm_65: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_217, view_218, permute_120);  convert_element_type_217 = None
        view_219: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_65, _shape_param_253);  addmm_65 = _shape_param_253 = None
        inductor_lookup_seed_33: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 33)
        inductor_random_33: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_254, inductor_lookup_seed_33, 'rand');  _shape_param_254 = inductor_lookup_seed_33 = None
        convert_element_type_219: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_33, torch.bfloat16);  inductor_random_33 = None
        gt_33: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_219, 0.1);  convert_element_type_219 = None
        mul_143: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_33, view_219);  view_219 = None
        mul_144: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_143, 1.1111111111111112);  mul_143 = None
        add_77: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_144, add_75);  mul_144 = add_75 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_44: "f32[8, 512, 1]" = var_mean_22[0]
        getitem_45: "f32[8, 512, 1]" = var_mean_22[1];  var_mean_22 = None
        add_78: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-07);  getitem_44 = None
        rsqrt_22: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_33: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_77, getitem_45);  add_77 = getitem_45 = None
        mul_145: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
        mul_146: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_145, arg180_1);  arg180_1 = None
        add_79: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_146, arg181_1);  mul_146 = arg181_1 = None
        convert_element_type_220: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.bfloat16);  arg182_1 = None
        convert_element_type_221: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.bfloat16);  arg183_1 = None
        convert_element_type_222: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16)
        view_220: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_222, _shape_param_255);  convert_element_type_222 = _shape_param_255 = None
        permute_121: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_221, [1, 0]);  convert_element_type_221 = None
        addmm_66: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_220, view_220, permute_121);  convert_element_type_220 = None
        view_221: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_66, _shape_param_256);  addmm_66 = _shape_param_256 = None
        view_222: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_221, _shape_param_257);  view_221 = _shape_param_257 = None
        permute_122: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        clone_44: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_223: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_258);  clone_44 = _shape_param_258 = None
        convert_element_type_223: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.bfloat16);  arg184_1 = None
        convert_element_type_224: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.bfloat16);  arg185_1 = None
        permute_123: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_224, [1, 0]);  convert_element_type_224 = None
        addmm_67: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_223, view_220, permute_123);  convert_element_type_223 = None
        view_224: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_67, _shape_param_259);  addmm_67 = _shape_param_259 = None
        view_225: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_224, _shape_param_260);  view_224 = _shape_param_260 = None
        permute_124: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        clone_45: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_226: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_45, _shape_param_261);  clone_45 = _shape_param_261 = None
        convert_element_type_225: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.bfloat16);  arg186_1 = None
        convert_element_type_226: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.bfloat16);  arg187_1 = None
        permute_125: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_226, [1, 0]);  convert_element_type_226 = None
        addmm_68: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_225, view_220, permute_125);  convert_element_type_225 = None
        view_227: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_68, _shape_param_262);  addmm_68 = _shape_param_262 = None
        view_228: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_227, _shape_param_263);  view_227 = _shape_param_263 = None
        permute_126: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        clone_46: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_229: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_264);  clone_46 = _shape_param_264 = None
        permute_127: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_226, [0, 2, 1]);  view_226 = None
        div_22: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_127, full);  permute_127 = None
        bmm_22: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_223, div_22)
        view_230: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_265);  bmm_22 = _shape_param_265 = None
        where_11: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_230);  view_230 = None
        convert_element_type_227: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_11, torch.float32)
        amax_11: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_227, [-1], True)
        sub_34: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_227, amax_11);  convert_element_type_227 = None
        exp_11: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        sum_12: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_23: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None
        inductor_lookup_seed_34: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 34)
        inductor_random_34: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_266, inductor_lookup_seed_34, 'rand');  _shape_param_266 = inductor_lookup_seed_34 = None
        gt_34: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_34, 0.1);  inductor_random_34 = None
        mul_147: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_34, div_23);  div_23 = None
        mul_148: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_147, 1.1111111111111112);  mul_147 = None
        view_231: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_148, _shape_param_267);  mul_148 = _shape_param_267 = None
        convert_element_type_228: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_231, torch.bfloat16);  view_231 = None
        bmm_23: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_228, view_229)
        view_232: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_268);  bmm_23 = _shape_param_268 = None
        permute_128: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
        clone_47: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_233: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_47, _shape_param_269);  clone_47 = _shape_param_269 = None
        convert_element_type_229: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.bfloat16);  arg188_1 = None
        convert_element_type_230: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.bfloat16);  arg189_1 = None
        view_234: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_233, _shape_param_270);  view_233 = _shape_param_270 = None
        permute_129: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_230, [1, 0]);  convert_element_type_230 = None
        addmm_69: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_229, view_234, permute_129);  convert_element_type_229 = None
        view_235: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_69, _shape_param_271);  addmm_69 = _shape_param_271 = None
        inductor_lookup_seed_35: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 35)
        inductor_random_35: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_272, inductor_lookup_seed_35, 'rand');  _shape_param_272 = inductor_lookup_seed_35 = None
        convert_element_type_231: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_35, torch.bfloat16);  inductor_random_35 = None
        gt_35: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_231, 0.1);  convert_element_type_231 = None
        mul_149: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_35, view_235);  view_235 = None
        mul_150: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_149, 1.1111111111111112);  mul_149 = None
        add_80: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_150, add_79);  mul_150 = add_79 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
        getitem_46: "f32[8, 512, 1]" = var_mean_23[0]
        getitem_47: "f32[8, 512, 1]" = var_mean_23[1];  var_mean_23 = None
        add_81: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-07);  getitem_46 = None
        rsqrt_23: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_35: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_80, getitem_47);  add_80 = getitem_47 = None
        mul_151: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_152: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_151, arg190_1);  arg190_1 = None
        add_82: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_152, arg191_1);  mul_152 = arg191_1 = None
        convert_element_type_232: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.bfloat16);  arg192_1 = None
        convert_element_type_233: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.bfloat16);  arg193_1 = None
        convert_element_type_234: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16)
        view_236: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_234, _shape_param_273);  convert_element_type_234 = _shape_param_273 = None
        permute_130: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_233, [1, 0]);  convert_element_type_233 = None
        addmm_70: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_232, view_236, permute_130);  convert_element_type_232 = None
        view_237: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_70, _shape_param_274);  _shape_param_274 = None
        convert_element_type_235: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None
        mul_153: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.5)
        mul_154: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.7071067811865476);  convert_element_type_235 = None
        erf_11: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
        add_83: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_155: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_153, add_83);  mul_153 = add_83 = None
        convert_element_type_236: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_155, torch.bfloat16);  mul_155 = None
        convert_element_type_237: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.bfloat16);  arg194_1 = None
        convert_element_type_238: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg195_1, torch.bfloat16);  arg195_1 = None
        view_238: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_236, _shape_param_275);  convert_element_type_236 = _shape_param_275 = None
        permute_131: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_238, [1, 0]);  convert_element_type_238 = None
        addmm_71: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_237, view_238, permute_131);  convert_element_type_237 = None
        view_239: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_71, _shape_param_276);  addmm_71 = _shape_param_276 = None
        inductor_lookup_seed_36: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 36)
        inductor_random_36: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_277, inductor_lookup_seed_36, 'rand');  _shape_param_277 = inductor_lookup_seed_36 = None
        convert_element_type_239: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_36, torch.bfloat16);  inductor_random_36 = None
        gt_36: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_239, 0.1);  convert_element_type_239 = None
        mul_156: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_36, view_239);  view_239 = None
        mul_157: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_156, 1.1111111111111112);  mul_156 = None
        add_84: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_157, add_82);  mul_157 = add_82 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_48: "f32[8, 512, 1]" = var_mean_24[0]
        getitem_49: "f32[8, 512, 1]" = var_mean_24[1];  var_mean_24 = None
        add_85: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-07);  getitem_48 = None
        rsqrt_24: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_36: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_84, getitem_49);  add_84 = getitem_49 = None
        mul_158: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
        mul_159: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_158, arg196_1);  arg196_1 = None
        add_86: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_159, arg197_1);  mul_159 = arg197_1 = None
        convert_element_type_240: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg198_1, torch.bfloat16);  arg198_1 = None
        convert_element_type_241: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.bfloat16);  arg199_1 = None
        convert_element_type_242: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16)
        view_240: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_242, _shape_param_278);  convert_element_type_242 = _shape_param_278 = None
        permute_132: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None
        addmm_72: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_240, view_240, permute_132);  convert_element_type_240 = None
        view_241: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_72, _shape_param_279);  addmm_72 = _shape_param_279 = None
        view_242: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_241, _shape_param_280);  view_241 = _shape_param_280 = None
        permute_133: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
        clone_48: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
        view_243: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_281);  clone_48 = _shape_param_281 = None
        convert_element_type_243: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.bfloat16);  arg200_1 = None
        convert_element_type_244: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg201_1, torch.bfloat16);  arg201_1 = None
        permute_134: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        addmm_73: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_243, view_240, permute_134);  convert_element_type_243 = None
        view_244: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_73, _shape_param_282);  addmm_73 = _shape_param_282 = None
        view_245: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_244, _shape_param_283);  view_244 = _shape_param_283 = None
        permute_135: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
        clone_49: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
        view_246: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_49, _shape_param_284);  clone_49 = _shape_param_284 = None
        convert_element_type_245: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg202_1, torch.bfloat16);  arg202_1 = None
        convert_element_type_246: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg203_1, torch.bfloat16);  arg203_1 = None
        permute_136: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_246, [1, 0]);  convert_element_type_246 = None
        addmm_74: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_245, view_240, permute_136);  convert_element_type_245 = None
        view_247: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_74, _shape_param_285);  addmm_74 = _shape_param_285 = None
        view_248: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_247, _shape_param_286);  view_247 = _shape_param_286 = None
        permute_137: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        clone_50: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_249: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_50, _shape_param_287);  clone_50 = _shape_param_287 = None
        permute_138: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_246, [0, 2, 1]);  view_246 = None
        div_24: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_138, full);  permute_138 = None
        bmm_24: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_243, div_24)
        view_250: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_24, _shape_param_288);  bmm_24 = _shape_param_288 = None
        where_12: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_250);  view_250 = None
        convert_element_type_247: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_12, torch.float32)
        amax_12: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_247, [-1], True)
        sub_37: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_247, amax_12);  convert_element_type_247 = None
        exp_12: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_13: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_25: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None
        inductor_lookup_seed_37: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 37)
        inductor_random_37: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_289, inductor_lookup_seed_37, 'rand');  _shape_param_289 = inductor_lookup_seed_37 = None
        gt_37: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_37, 0.1);  inductor_random_37 = None
        mul_160: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_37, div_25);  div_25 = None
        mul_161: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_160, 1.1111111111111112);  mul_160 = None
        view_251: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_161, _shape_param_290);  mul_161 = _shape_param_290 = None
        convert_element_type_248: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_251, torch.bfloat16);  view_251 = None
        bmm_25: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_248, view_249)
        view_252: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_25, _shape_param_291);  bmm_25 = _shape_param_291 = None
        permute_139: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        clone_51: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_253: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_51, _shape_param_292);  clone_51 = _shape_param_292 = None
        convert_element_type_249: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg204_1, torch.bfloat16);  arg204_1 = None
        convert_element_type_250: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg205_1, torch.bfloat16);  arg205_1 = None
        view_254: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_253, _shape_param_293);  view_253 = _shape_param_293 = None
        permute_140: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_250, [1, 0]);  convert_element_type_250 = None
        addmm_75: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_249, view_254, permute_140);  convert_element_type_249 = None
        view_255: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_75, _shape_param_294);  addmm_75 = _shape_param_294 = None
        inductor_lookup_seed_38: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 38)
        inductor_random_38: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_295, inductor_lookup_seed_38, 'rand');  _shape_param_295 = inductor_lookup_seed_38 = None
        convert_element_type_251: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_38, torch.bfloat16);  inductor_random_38 = None
        gt_38: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_251, 0.1);  convert_element_type_251 = None
        mul_162: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_38, view_255);  view_255 = None
        mul_163: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_162, 1.1111111111111112);  mul_162 = None
        add_87: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_163, add_86);  mul_163 = add_86 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
        getitem_50: "f32[8, 512, 1]" = var_mean_25[0]
        getitem_51: "f32[8, 512, 1]" = var_mean_25[1];  var_mean_25 = None
        add_88: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-07);  getitem_50 = None
        rsqrt_25: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        sub_38: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_87, getitem_51);  add_87 = getitem_51 = None
        mul_164: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
        mul_165: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_164, arg206_1);  arg206_1 = None
        add_89: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_165, arg207_1);  mul_165 = arg207_1 = None
        convert_element_type_252: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg208_1, torch.bfloat16);  arg208_1 = None
        convert_element_type_253: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.bfloat16);  arg209_1 = None
        convert_element_type_254: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16)
        view_256: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_254, _shape_param_296);  convert_element_type_254 = _shape_param_296 = None
        permute_141: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_253, [1, 0]);  convert_element_type_253 = None
        addmm_76: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_252, view_256, permute_141);  convert_element_type_252 = None
        view_257: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_76, _shape_param_297);  _shape_param_297 = None
        convert_element_type_255: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None
        mul_166: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_255, 0.5)
        mul_167: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_255, 0.7071067811865476);  convert_element_type_255 = None
        erf_12: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
        add_90: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_168: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_166, add_90);  mul_166 = add_90 = None
        convert_element_type_256: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        convert_element_type_257: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg210_1, torch.bfloat16);  arg210_1 = None
        convert_element_type_258: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg211_1, torch.bfloat16);  arg211_1 = None
        view_258: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_256, _shape_param_298);  convert_element_type_256 = _shape_param_298 = None
        permute_142: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_258, [1, 0]);  convert_element_type_258 = None
        addmm_77: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_257, view_258, permute_142);  convert_element_type_257 = None
        view_259: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_77, _shape_param_299);  addmm_77 = _shape_param_299 = None
        inductor_lookup_seed_39: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 39)
        inductor_random_39: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_300, inductor_lookup_seed_39, 'rand');  _shape_param_300 = inductor_lookup_seed_39 = None
        convert_element_type_259: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_39, torch.bfloat16);  inductor_random_39 = None
        gt_39: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_259, 0.1);  convert_element_type_259 = None
        mul_169: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_39, view_259);  view_259 = None
        mul_170: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None
        add_91: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_170, add_89);  mul_170 = add_89 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
        getitem_52: "f32[8, 512, 1]" = var_mean_26[0]
        getitem_53: "f32[8, 512, 1]" = var_mean_26[1];  var_mean_26 = None
        add_92: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-07);  getitem_52 = None
        rsqrt_26: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        sub_39: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_91, getitem_53);  add_91 = getitem_53 = None
        mul_171: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = None
        mul_172: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_171, arg212_1);  arg212_1 = None
        add_93: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_172, arg213_1);  mul_172 = arg213_1 = None
        convert_element_type_260: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg214_1, torch.bfloat16);  arg214_1 = None
        convert_element_type_261: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg215_1, torch.bfloat16);  arg215_1 = None
        convert_element_type_262: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16)
        view_260: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_262, _shape_param_301);  convert_element_type_262 = _shape_param_301 = None
        permute_143: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_261, [1, 0]);  convert_element_type_261 = None
        addmm_78: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_260, view_260, permute_143);  convert_element_type_260 = None
        view_261: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_78, _shape_param_302);  addmm_78 = _shape_param_302 = None
        view_262: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_261, _shape_param_303);  view_261 = _shape_param_303 = None
        permute_144: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        clone_52: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_263: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_304);  clone_52 = _shape_param_304 = None
        convert_element_type_263: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg216_1, torch.bfloat16);  arg216_1 = None
        convert_element_type_264: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg217_1, torch.bfloat16);  arg217_1 = None
        permute_145: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        addmm_79: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_263, view_260, permute_145);  convert_element_type_263 = None
        view_264: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_79, _shape_param_305);  addmm_79 = _shape_param_305 = None
        view_265: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_264, _shape_param_306);  view_264 = _shape_param_306 = None
        permute_146: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
        clone_53: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_266: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_53, _shape_param_307);  clone_53 = _shape_param_307 = None
        convert_element_type_265: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg218_1, torch.bfloat16);  arg218_1 = None
        convert_element_type_266: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg219_1, torch.bfloat16);  arg219_1 = None
        permute_147: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_266, [1, 0]);  convert_element_type_266 = None
        addmm_80: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_265, view_260, permute_147);  convert_element_type_265 = None
        view_267: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_80, _shape_param_308);  addmm_80 = _shape_param_308 = None
        view_268: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_267, _shape_param_309);  view_267 = _shape_param_309 = None
        permute_148: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        clone_54: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_269: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_54, _shape_param_310);  clone_54 = _shape_param_310 = None
        permute_149: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
        div_26: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_149, full);  permute_149 = None
        bmm_26: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_263, div_26)
        view_270: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_26, _shape_param_311);  bmm_26 = _shape_param_311 = None
        where_13: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_270);  view_270 = None
        convert_element_type_267: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_13, torch.float32)
        amax_13: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_267, [-1], True)
        sub_40: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_267, amax_13);  convert_element_type_267 = None
        exp_13: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        sum_14: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_27: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None
        inductor_lookup_seed_40: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 40)
        inductor_random_40: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_312, inductor_lookup_seed_40, 'rand');  _shape_param_312 = inductor_lookup_seed_40 = None
        gt_40: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_40, 0.1);  inductor_random_40 = None
        mul_173: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_40, div_27);  div_27 = None
        mul_174: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_173, 1.1111111111111112);  mul_173 = None
        view_271: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_174, _shape_param_313);  mul_174 = _shape_param_313 = None
        convert_element_type_268: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_271, torch.bfloat16);  view_271 = None
        bmm_27: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_268, view_269)
        view_272: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_314);  bmm_27 = _shape_param_314 = None
        permute_150: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        clone_55: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_273: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_55, _shape_param_315);  clone_55 = _shape_param_315 = None
        convert_element_type_269: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg220_1, torch.bfloat16);  arg220_1 = None
        convert_element_type_270: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg221_1, torch.bfloat16);  arg221_1 = None
        view_274: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_273, _shape_param_316);  view_273 = _shape_param_316 = None
        permute_151: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        addmm_81: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_269, view_274, permute_151);  convert_element_type_269 = None
        view_275: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_81, _shape_param_317);  addmm_81 = _shape_param_317 = None
        inductor_lookup_seed_41: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 41)
        inductor_random_41: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_318, inductor_lookup_seed_41, 'rand');  _shape_param_318 = inductor_lookup_seed_41 = None
        convert_element_type_271: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_41, torch.bfloat16);  inductor_random_41 = None
        gt_41: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_271, 0.1);  convert_element_type_271 = None
        mul_175: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_41, view_275);  view_275 = None
        mul_176: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_175, 1.1111111111111112);  mul_175 = None
        add_94: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_176, add_93);  mul_176 = add_93 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
        getitem_54: "f32[8, 512, 1]" = var_mean_27[0]
        getitem_55: "f32[8, 512, 1]" = var_mean_27[1];  var_mean_27 = None
        add_95: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-07);  getitem_54 = None
        rsqrt_27: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        sub_41: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_94, getitem_55);  add_94 = getitem_55 = None
        mul_177: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
        mul_178: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_177, arg222_1);  arg222_1 = None
        add_96: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_178, arg223_1);  mul_178 = arg223_1 = None
        convert_element_type_272: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg224_1, torch.bfloat16);  arg224_1 = None
        convert_element_type_273: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg225_1, torch.bfloat16);  arg225_1 = None
        convert_element_type_274: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16)
        view_276: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_274, _shape_param_319);  convert_element_type_274 = _shape_param_319 = None
        permute_152: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_273, [1, 0]);  convert_element_type_273 = None
        addmm_82: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_272, view_276, permute_152);  convert_element_type_272 = None
        view_277: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_82, _shape_param_320);  _shape_param_320 = None
        convert_element_type_275: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None
        mul_179: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_275, 0.5)
        mul_180: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_275, 0.7071067811865476);  convert_element_type_275 = None
        erf_13: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
        add_97: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_181: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_179, add_97);  mul_179 = add_97 = None
        convert_element_type_276: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        convert_element_type_277: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg226_1, torch.bfloat16);  arg226_1 = None
        convert_element_type_278: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.bfloat16);  arg227_1 = None
        view_278: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_276, _shape_param_321);  convert_element_type_276 = _shape_param_321 = None
        permute_153: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_278, [1, 0]);  convert_element_type_278 = None
        addmm_83: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_277, view_278, permute_153);  convert_element_type_277 = None
        view_279: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_83, _shape_param_322);  addmm_83 = _shape_param_322 = None
        inductor_lookup_seed_42: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 42)
        inductor_random_42: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_323, inductor_lookup_seed_42, 'rand');  _shape_param_323 = inductor_lookup_seed_42 = None
        convert_element_type_279: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_42, torch.bfloat16);  inductor_random_42 = None
        gt_42: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_279, 0.1);  convert_element_type_279 = None
        mul_182: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_42, view_279);  view_279 = None
        mul_183: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_182, 1.1111111111111112);  mul_182 = None
        add_98: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_183, add_96);  mul_183 = add_96 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
        getitem_56: "f32[8, 512, 1]" = var_mean_28[0]
        getitem_57: "f32[8, 512, 1]" = var_mean_28[1];  var_mean_28 = None
        add_99: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-07);  getitem_56 = None
        rsqrt_28: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        sub_42: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_98, getitem_57);  add_98 = getitem_57 = None
        mul_184: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = None
        mul_185: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_184, arg228_1);  arg228_1 = None
        add_100: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_185, arg229_1);  mul_185 = arg229_1 = None
        convert_element_type_280: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg230_1, torch.bfloat16);  arg230_1 = None
        convert_element_type_281: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg231_1, torch.bfloat16);  arg231_1 = None
        convert_element_type_282: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16)
        view_280: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_282, _shape_param_324);  convert_element_type_282 = _shape_param_324 = None
        permute_154: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_281, [1, 0]);  convert_element_type_281 = None
        addmm_84: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_280, view_280, permute_154);  convert_element_type_280 = None
        view_281: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_84, _shape_param_325);  addmm_84 = _shape_param_325 = None
        view_282: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_281, _shape_param_326);  view_281 = _shape_param_326 = None
        permute_155: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
        clone_56: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_283: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_327);  clone_56 = _shape_param_327 = None
        convert_element_type_283: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.bfloat16);  arg232_1 = None
        convert_element_type_284: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.bfloat16);  arg233_1 = None
        permute_156: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_284, [1, 0]);  convert_element_type_284 = None
        addmm_85: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_283, view_280, permute_156);  convert_element_type_283 = None
        view_284: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_85, _shape_param_328);  addmm_85 = _shape_param_328 = None
        view_285: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_284, _shape_param_329);  view_284 = _shape_param_329 = None
        permute_157: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
        clone_57: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_286: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_57, _shape_param_330);  clone_57 = _shape_param_330 = None
        convert_element_type_285: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg234_1, torch.bfloat16);  arg234_1 = None
        convert_element_type_286: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg235_1, torch.bfloat16);  arg235_1 = None
        permute_158: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_286, [1, 0]);  convert_element_type_286 = None
        addmm_86: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_285, view_280, permute_158);  convert_element_type_285 = None
        view_287: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_86, _shape_param_331);  addmm_86 = _shape_param_331 = None
        view_288: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_287, _shape_param_332);  view_287 = _shape_param_332 = None
        permute_159: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        clone_58: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_289: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_333);  clone_58 = _shape_param_333 = None
        permute_160: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
        div_28: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_160, full);  permute_160 = None
        bmm_28: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_283, div_28)
        view_290: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_28, _shape_param_334);  bmm_28 = _shape_param_334 = None
        where_14: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_290);  view_290 = None
        convert_element_type_287: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_14, torch.float32)
        amax_14: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_287, [-1], True)
        sub_43: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_287, amax_14);  convert_element_type_287 = None
        exp_14: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
        sum_15: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_29: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None
        inductor_lookup_seed_43: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 43)
        inductor_random_43: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_335, inductor_lookup_seed_43, 'rand');  _shape_param_335 = inductor_lookup_seed_43 = None
        gt_43: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_43, 0.1);  inductor_random_43 = None
        mul_186: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_43, div_29);  div_29 = None
        mul_187: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_186, 1.1111111111111112);  mul_186 = None
        view_291: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_187, _shape_param_336);  mul_187 = _shape_param_336 = None
        convert_element_type_288: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_291, torch.bfloat16);  view_291 = None
        bmm_29: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_288, view_289)
        view_292: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_29, _shape_param_337);  bmm_29 = _shape_param_337 = None
        permute_161: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_59: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_293: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_59, _shape_param_338);  clone_59 = _shape_param_338 = None
        convert_element_type_289: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg236_1, torch.bfloat16);  arg236_1 = None
        convert_element_type_290: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.bfloat16);  arg237_1 = None
        view_294: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_293, _shape_param_339);  view_293 = _shape_param_339 = None
        permute_162: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_290, [1, 0]);  convert_element_type_290 = None
        addmm_87: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_289, view_294, permute_162);  convert_element_type_289 = None
        view_295: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_87, _shape_param_340);  addmm_87 = _shape_param_340 = None
        inductor_lookup_seed_44: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 44)
        inductor_random_44: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_341, inductor_lookup_seed_44, 'rand');  _shape_param_341 = inductor_lookup_seed_44 = None
        convert_element_type_291: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_44, torch.bfloat16);  inductor_random_44 = None
        gt_44: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_291, 0.1);  convert_element_type_291 = None
        mul_188: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_44, view_295);  view_295 = None
        mul_189: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_188, 1.1111111111111112);  mul_188 = None
        add_101: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_189, add_100);  mul_189 = add_100 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_58: "f32[8, 512, 1]" = var_mean_29[0]
        getitem_59: "f32[8, 512, 1]" = var_mean_29[1];  var_mean_29 = None
        add_102: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-07);  getitem_58 = None
        rsqrt_29: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_44: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_101, getitem_59);  add_101 = getitem_59 = None
        mul_190: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
        mul_191: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_190, arg238_1);  arg238_1 = None
        add_103: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_191, arg239_1);  mul_191 = arg239_1 = None
        convert_element_type_292: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg240_1, torch.bfloat16);  arg240_1 = None
        convert_element_type_293: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg241_1, torch.bfloat16);  arg241_1 = None
        convert_element_type_294: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16)
        view_296: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_294, _shape_param_342);  convert_element_type_294 = _shape_param_342 = None
        permute_163: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_293, [1, 0]);  convert_element_type_293 = None
        addmm_88: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_292, view_296, permute_163);  convert_element_type_292 = None
        view_297: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_88, _shape_param_343);  _shape_param_343 = None
        convert_element_type_295: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_297, torch.float32);  view_297 = None
        mul_192: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_295, 0.5)
        mul_193: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_295, 0.7071067811865476);  convert_element_type_295 = None
        erf_14: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_193);  mul_193 = None
        add_104: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_194: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_192, add_104);  mul_192 = add_104 = None
        convert_element_type_296: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_194, torch.bfloat16);  mul_194 = None
        convert_element_type_297: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.bfloat16);  arg242_1 = None
        convert_element_type_298: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.bfloat16);  arg243_1 = None
        view_298: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_296, _shape_param_344);  convert_element_type_296 = _shape_param_344 = None
        permute_164: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_298, [1, 0]);  convert_element_type_298 = None
        addmm_89: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_297, view_298, permute_164);  convert_element_type_297 = None
        view_299: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_89, _shape_param_345);  addmm_89 = _shape_param_345 = None
        inductor_lookup_seed_45: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 45)
        inductor_random_45: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_346, inductor_lookup_seed_45, 'rand');  _shape_param_346 = inductor_lookup_seed_45 = None
        convert_element_type_299: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_45, torch.bfloat16);  inductor_random_45 = None
        gt_45: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_299, 0.1);  convert_element_type_299 = None
        mul_195: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_45, view_299);  view_299 = None
        mul_196: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_195, 1.1111111111111112);  mul_195 = None
        add_105: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_196, add_103);  mul_196 = add_103 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_60: "f32[8, 512, 1]" = var_mean_30[0]
        getitem_61: "f32[8, 512, 1]" = var_mean_30[1];  var_mean_30 = None
        add_106: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-07);  getitem_60 = None
        rsqrt_30: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_45: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_105, getitem_61);  add_105 = getitem_61 = None
        mul_197: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = None
        mul_198: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_197, arg244_1);  arg244_1 = None
        add_107: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_198, arg245_1);  mul_198 = arg245_1 = None
        convert_element_type_300: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg246_1, torch.bfloat16);  arg246_1 = None
        convert_element_type_301: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg247_1, torch.bfloat16);  arg247_1 = None
        convert_element_type_302: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16)
        view_300: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_302, _shape_param_347);  convert_element_type_302 = _shape_param_347 = None
        permute_165: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_301, [1, 0]);  convert_element_type_301 = None
        addmm_90: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_300, view_300, permute_165);  convert_element_type_300 = None
        view_301: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_90, _shape_param_348);  addmm_90 = _shape_param_348 = None
        view_302: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_301, _shape_param_349);  view_301 = _shape_param_349 = None
        permute_166: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        clone_60: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_303: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_350);  clone_60 = _shape_param_350 = None
        convert_element_type_303: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.bfloat16);  arg248_1 = None
        convert_element_type_304: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg249_1, torch.bfloat16);  arg249_1 = None
        permute_167: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_304, [1, 0]);  convert_element_type_304 = None
        addmm_91: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_303, view_300, permute_167);  convert_element_type_303 = None
        view_304: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_91, _shape_param_351);  addmm_91 = _shape_param_351 = None
        view_305: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_304, _shape_param_352);  view_304 = _shape_param_352 = None
        permute_168: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
        clone_61: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
        view_306: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_61, _shape_param_353);  clone_61 = _shape_param_353 = None
        convert_element_type_305: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg250_1, torch.bfloat16);  arg250_1 = None
        convert_element_type_306: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg251_1, torch.bfloat16);  arg251_1 = None
        permute_169: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_306, [1, 0]);  convert_element_type_306 = None
        addmm_92: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_305, view_300, permute_169);  convert_element_type_305 = None
        view_307: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_92, _shape_param_354);  addmm_92 = _shape_param_354 = None
        view_308: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_307, _shape_param_355);  view_307 = _shape_param_355 = None
        permute_170: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        clone_62: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_309: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_62, _shape_param_356);  clone_62 = _shape_param_356 = None
        permute_171: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_306, [0, 2, 1]);  view_306 = None
        div_30: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_171, full);  permute_171 = None
        bmm_30: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_303, div_30)
        view_310: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_30, _shape_param_357);  bmm_30 = _shape_param_357 = None
        where_15: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_310);  view_310 = None
        convert_element_type_307: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_15, torch.float32)
        amax_15: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_307, [-1], True)
        sub_46: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_307, amax_15);  convert_element_type_307 = None
        exp_15: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
        sum_16: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_31: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None
        inductor_lookup_seed_46: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 46)
        inductor_random_46: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_358, inductor_lookup_seed_46, 'rand');  _shape_param_358 = inductor_lookup_seed_46 = None
        gt_46: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_46, 0.1);  inductor_random_46 = None
        mul_199: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_46, div_31);  div_31 = None
        mul_200: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_199, 1.1111111111111112);  mul_199 = None
        view_311: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_200, _shape_param_359);  mul_200 = _shape_param_359 = None
        convert_element_type_308: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_311, torch.bfloat16);  view_311 = None
        bmm_31: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_308, view_309)
        view_312: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_360);  bmm_31 = _shape_param_360 = None
        permute_172: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1, 3]);  view_312 = None
        clone_63: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_313: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_63, _shape_param_361);  clone_63 = _shape_param_361 = None
        convert_element_type_309: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.bfloat16);  arg252_1 = None
        convert_element_type_310: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.bfloat16);  arg253_1 = None
        view_314: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_313, _shape_param_362);  view_313 = _shape_param_362 = None
        permute_173: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_310, [1, 0]);  convert_element_type_310 = None
        addmm_93: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_309, view_314, permute_173);  convert_element_type_309 = None
        view_315: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_93, _shape_param_363);  addmm_93 = _shape_param_363 = None
        inductor_lookup_seed_47: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 47)
        inductor_random_47: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_364, inductor_lookup_seed_47, 'rand');  _shape_param_364 = inductor_lookup_seed_47 = None
        convert_element_type_311: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_47, torch.bfloat16);  inductor_random_47 = None
        gt_47: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_311, 0.1);  convert_element_type_311 = None
        mul_201: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_47, view_315);  view_315 = None
        mul_202: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_201, 1.1111111111111112);  mul_201 = None
        add_108: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_202, add_107);  mul_202 = add_107 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
        getitem_62: "f32[8, 512, 1]" = var_mean_31[0]
        getitem_63: "f32[8, 512, 1]" = var_mean_31[1];  var_mean_31 = None
        add_109: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-07);  getitem_62 = None
        rsqrt_31: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        sub_47: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_108, getitem_63);  add_108 = getitem_63 = None
        mul_203: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
        mul_204: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_203, arg254_1);  arg254_1 = None
        add_110: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_204, arg255_1);  mul_204 = arg255_1 = None
        convert_element_type_312: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg256_1, torch.bfloat16);  arg256_1 = None
        convert_element_type_313: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.bfloat16);  arg257_1 = None
        convert_element_type_314: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_110, torch.bfloat16)
        view_316: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_314, _shape_param_365);  convert_element_type_314 = _shape_param_365 = None
        permute_174: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_313, [1, 0]);  convert_element_type_313 = None
        addmm_94: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_312, view_316, permute_174);  convert_element_type_312 = None
        view_317: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_94, _shape_param_366);  _shape_param_366 = None
        convert_element_type_315: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None
        mul_205: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_315, 0.5)
        mul_206: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_315, 0.7071067811865476);  convert_element_type_315 = None
        erf_15: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_206);  mul_206 = None
        add_111: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_207: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_205, add_111);  mul_205 = add_111 = None
        convert_element_type_316: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_207, torch.bfloat16);  mul_207 = None
        convert_element_type_317: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.bfloat16);  arg258_1 = None
        convert_element_type_318: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.bfloat16);  arg259_1 = None
        view_318: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_316, _shape_param_367);  convert_element_type_316 = _shape_param_367 = None
        permute_175: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_318, [1, 0]);  convert_element_type_318 = None
        addmm_95: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_317, view_318, permute_175);  convert_element_type_317 = None
        view_319: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_95, _shape_param_368);  addmm_95 = _shape_param_368 = None
        inductor_lookup_seed_48: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 48)
        inductor_random_48: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_369, inductor_lookup_seed_48, 'rand');  _shape_param_369 = inductor_lookup_seed_48 = None
        convert_element_type_319: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_48, torch.bfloat16);  inductor_random_48 = None
        gt_48: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_319, 0.1);  convert_element_type_319 = None
        mul_208: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_48, view_319);  view_319 = None
        mul_209: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_208, 1.1111111111111112);  mul_208 = None
        add_112: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_209, add_110);  mul_209 = add_110 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
        getitem_64: "f32[8, 512, 1]" = var_mean_32[0]
        getitem_65: "f32[8, 512, 1]" = var_mean_32[1];  var_mean_32 = None
        add_113: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-07);  getitem_64 = None
        rsqrt_32: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        sub_48: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_112, getitem_65);  add_112 = getitem_65 = None
        mul_210: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = None
        mul_211: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_210, arg260_1);  arg260_1 = None
        add_114: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_211, arg261_1);  mul_211 = arg261_1 = None
        convert_element_type_320: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg262_1, torch.bfloat16);  arg262_1 = None
        convert_element_type_321: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.bfloat16);  arg263_1 = None
        convert_element_type_322: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16)
        view_320: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_322, _shape_param_370);  convert_element_type_322 = _shape_param_370 = None
        permute_176: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_321, [1, 0]);  convert_element_type_321 = None
        addmm_96: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_320, view_320, permute_176);  convert_element_type_320 = None
        view_321: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_96, _shape_param_371);  addmm_96 = _shape_param_371 = None
        view_322: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_321, _shape_param_372);  view_321 = _shape_param_372 = None
        permute_177: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        clone_64: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_323: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_373);  clone_64 = _shape_param_373 = None
        convert_element_type_323: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg264_1, torch.bfloat16);  arg264_1 = None
        convert_element_type_324: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg265_1, torch.bfloat16);  arg265_1 = None
        permute_178: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_324, [1, 0]);  convert_element_type_324 = None
        addmm_97: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_323, view_320, permute_178);  convert_element_type_323 = None
        view_324: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_97, _shape_param_374);  addmm_97 = _shape_param_374 = None
        view_325: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_324, _shape_param_375);  view_324 = _shape_param_375 = None
        permute_179: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
        clone_65: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
        view_326: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_65, _shape_param_376);  clone_65 = _shape_param_376 = None
        convert_element_type_325: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg266_1, torch.bfloat16);  arg266_1 = None
        convert_element_type_326: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.bfloat16);  arg267_1 = None
        permute_180: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_326, [1, 0]);  convert_element_type_326 = None
        addmm_98: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_325, view_320, permute_180);  convert_element_type_325 = None
        view_327: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_98, _shape_param_377);  addmm_98 = _shape_param_377 = None
        view_328: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_327, _shape_param_378);  view_327 = _shape_param_378 = None
        permute_181: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        clone_66: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_329: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_379);  clone_66 = _shape_param_379 = None
        permute_182: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_326, [0, 2, 1]);  view_326 = None
        div_32: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_182, full);  permute_182 = None
        bmm_32: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_323, div_32)
        view_330: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_32, _shape_param_380);  bmm_32 = _shape_param_380 = None
        where_16: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_330);  view_330 = None
        convert_element_type_327: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_16, torch.float32)
        amax_16: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_327, [-1], True)
        sub_49: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_327, amax_16);  convert_element_type_327 = None
        exp_16: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_17: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_33: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None
        inductor_lookup_seed_49: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 49)
        inductor_random_49: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_381, inductor_lookup_seed_49, 'rand');  _shape_param_381 = inductor_lookup_seed_49 = None
        gt_49: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_49, 0.1);  inductor_random_49 = None
        mul_212: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_49, div_33);  div_33 = None
        mul_213: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_212, 1.1111111111111112);  mul_212 = None
        view_331: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_213, _shape_param_382);  mul_213 = _shape_param_382 = None
        convert_element_type_328: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_331, torch.bfloat16);  view_331 = None
        bmm_33: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_328, view_329)
        view_332: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_33, _shape_param_383);  bmm_33 = _shape_param_383 = None
        permute_183: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        clone_67: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
        view_333: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_67, _shape_param_384);  clone_67 = _shape_param_384 = None
        convert_element_type_329: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.bfloat16);  arg268_1 = None
        convert_element_type_330: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg269_1, torch.bfloat16);  arg269_1 = None
        view_334: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_333, _shape_param_385);  view_333 = _shape_param_385 = None
        permute_184: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_330, [1, 0]);  convert_element_type_330 = None
        addmm_99: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_329, view_334, permute_184);  convert_element_type_329 = None
        view_335: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_99, _shape_param_386);  addmm_99 = _shape_param_386 = None
        inductor_lookup_seed_50: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 50)
        inductor_random_50: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_387, inductor_lookup_seed_50, 'rand');  _shape_param_387 = inductor_lookup_seed_50 = None
        convert_element_type_331: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_50, torch.bfloat16);  inductor_random_50 = None
        gt_50: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_331, 0.1);  convert_element_type_331 = None
        mul_214: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_50, view_335);  view_335 = None
        mul_215: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_214, 1.1111111111111112);  mul_214 = None
        add_115: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_215, add_114);  mul_215 = add_114 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
        getitem_66: "f32[8, 512, 1]" = var_mean_33[0]
        getitem_67: "f32[8, 512, 1]" = var_mean_33[1];  var_mean_33 = None
        add_116: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-07);  getitem_66 = None
        rsqrt_33: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_50: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_115, getitem_67);  add_115 = getitem_67 = None
        mul_216: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
        mul_217: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_216, arg270_1);  arg270_1 = None
        add_117: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_217, arg271_1);  mul_217 = arg271_1 = None
        convert_element_type_332: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.bfloat16);  arg272_1 = None
        convert_element_type_333: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg273_1, torch.bfloat16);  arg273_1 = None
        convert_element_type_334: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16)
        view_336: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_334, _shape_param_388);  convert_element_type_334 = _shape_param_388 = None
        permute_185: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_333, [1, 0]);  convert_element_type_333 = None
        addmm_100: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_332, view_336, permute_185);  convert_element_type_332 = None
        view_337: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_100, _shape_param_389);  _shape_param_389 = None
        convert_element_type_335: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_337, torch.float32);  view_337 = None
        mul_218: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_335, 0.5)
        mul_219: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_335, 0.7071067811865476);  convert_element_type_335 = None
        erf_16: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
        add_118: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_220: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_218, add_118);  mul_218 = add_118 = None
        convert_element_type_336: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_220, torch.bfloat16);  mul_220 = None
        convert_element_type_337: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg274_1, torch.bfloat16);  arg274_1 = None
        convert_element_type_338: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.bfloat16);  arg275_1 = None
        view_338: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_336, _shape_param_390);  convert_element_type_336 = _shape_param_390 = None
        permute_186: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_338, [1, 0]);  convert_element_type_338 = None
        addmm_101: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_337, view_338, permute_186);  convert_element_type_337 = None
        view_339: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_101, _shape_param_391);  addmm_101 = _shape_param_391 = None
        inductor_lookup_seed_51: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 51)
        inductor_random_51: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_392, inductor_lookup_seed_51, 'rand');  _shape_param_392 = inductor_lookup_seed_51 = None
        convert_element_type_339: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_51, torch.bfloat16);  inductor_random_51 = None
        gt_51: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_339, 0.1);  convert_element_type_339 = None
        mul_221: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_51, view_339);  view_339 = None
        mul_222: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_221, 1.1111111111111112);  mul_221 = None
        add_119: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_222, add_117);  mul_222 = add_117 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
        getitem_68: "f32[8, 512, 1]" = var_mean_34[0]
        getitem_69: "f32[8, 512, 1]" = var_mean_34[1];  var_mean_34 = None
        add_120: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-07);  getitem_68 = None
        rsqrt_34: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        sub_51: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_119, getitem_69);  add_119 = getitem_69 = None
        mul_223: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = None
        mul_224: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_223, arg276_1);  arg276_1 = None
        add_121: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_224, arg277_1);  mul_224 = arg277_1 = None
        convert_element_type_340: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.bfloat16);  arg278_1 = None
        convert_element_type_341: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg279_1, torch.bfloat16);  arg279_1 = None
        convert_element_type_342: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16)
        view_340: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_342, _shape_param_393);  convert_element_type_342 = _shape_param_393 = None
        permute_187: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_341, [1, 0]);  convert_element_type_341 = None
        addmm_102: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_340, view_340, permute_187);  convert_element_type_340 = None
        view_341: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_102, _shape_param_394);  addmm_102 = _shape_param_394 = None
        view_342: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_341, _shape_param_395);  view_341 = _shape_param_395 = None
        permute_188: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
        clone_68: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_343: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_396);  clone_68 = _shape_param_396 = None
        convert_element_type_343: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg280_1, torch.bfloat16);  arg280_1 = None
        convert_element_type_344: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.bfloat16);  arg281_1 = None
        permute_189: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_344, [1, 0]);  convert_element_type_344 = None
        addmm_103: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_343, view_340, permute_189);  convert_element_type_343 = None
        view_344: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_103, _shape_param_397);  addmm_103 = _shape_param_397 = None
        view_345: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_344, _shape_param_398);  view_344 = _shape_param_398 = None
        permute_190: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
        clone_69: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_346: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_69, _shape_param_399);  clone_69 = _shape_param_399 = None
        convert_element_type_345: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.bfloat16);  arg282_1 = None
        convert_element_type_346: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.bfloat16);  arg283_1 = None
        permute_191: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_346, [1, 0]);  convert_element_type_346 = None
        addmm_104: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_345, view_340, permute_191);  convert_element_type_345 = None
        view_347: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_104, _shape_param_400);  addmm_104 = _shape_param_400 = None
        view_348: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_347, _shape_param_401);  view_347 = _shape_param_401 = None
        permute_192: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None
        clone_70: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
        view_349: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_402);  clone_70 = _shape_param_402 = None
        permute_193: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_346, [0, 2, 1]);  view_346 = None
        div_34: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_193, full);  permute_193 = None
        bmm_34: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_343, div_34)
        view_350: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_34, _shape_param_403);  bmm_34 = _shape_param_403 = None
        where_17: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_350);  view_350 = None
        convert_element_type_347: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_17, torch.float32)
        amax_17: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_347, [-1], True)
        sub_52: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_347, amax_17);  convert_element_type_347 = None
        exp_17: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        sum_18: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_35: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None
        inductor_lookup_seed_52: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 52)
        inductor_random_52: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_404, inductor_lookup_seed_52, 'rand');  _shape_param_404 = inductor_lookup_seed_52 = None
        gt_52: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_52, 0.1);  inductor_random_52 = None
        mul_225: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_52, div_35);  div_35 = None
        mul_226: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_225, 1.1111111111111112);  mul_225 = None
        view_351: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_226, _shape_param_405);  mul_226 = _shape_param_405 = None
        convert_element_type_348: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_351, torch.bfloat16);  view_351 = None
        bmm_35: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_348, view_349)
        view_352: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_406);  bmm_35 = _shape_param_406 = None
        permute_194: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        clone_71: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
        view_353: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_71, _shape_param_407);  clone_71 = _shape_param_407 = None
        convert_element_type_349: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg284_1, torch.bfloat16);  arg284_1 = None
        convert_element_type_350: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg285_1, torch.bfloat16);  arg285_1 = None
        view_354: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_353, _shape_param_408);  view_353 = _shape_param_408 = None
        permute_195: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_350, [1, 0]);  convert_element_type_350 = None
        addmm_105: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_349, view_354, permute_195);  convert_element_type_349 = None
        view_355: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_105, _shape_param_409);  addmm_105 = _shape_param_409 = None
        inductor_lookup_seed_53: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 53)
        inductor_random_53: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_410, inductor_lookup_seed_53, 'rand');  _shape_param_410 = inductor_lookup_seed_53 = None
        convert_element_type_351: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_53, torch.bfloat16);  inductor_random_53 = None
        gt_53: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_351, 0.1);  convert_element_type_351 = None
        mul_227: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_53, view_355);  view_355 = None
        mul_228: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_227, 1.1111111111111112);  mul_227 = None
        add_122: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_228, add_121);  mul_228 = add_121 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
        getitem_70: "f32[8, 512, 1]" = var_mean_35[0]
        getitem_71: "f32[8, 512, 1]" = var_mean_35[1];  var_mean_35 = None
        add_123: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-07);  getitem_70 = None
        rsqrt_35: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        sub_53: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_122, getitem_71);  add_122 = getitem_71 = None
        mul_229: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
        mul_230: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_229, arg286_1);  arg286_1 = None
        add_124: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_230, arg287_1);  mul_230 = arg287_1 = None
        convert_element_type_352: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.bfloat16);  arg288_1 = None
        convert_element_type_353: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.bfloat16);  arg289_1 = None
        convert_element_type_354: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16)
        view_356: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_354, _shape_param_411);  convert_element_type_354 = _shape_param_411 = None
        permute_196: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_353, [1, 0]);  convert_element_type_353 = None
        addmm_106: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_352, view_356, permute_196);  convert_element_type_352 = None
        view_357: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_106, _shape_param_412);  _shape_param_412 = None
        convert_element_type_355: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_357, torch.float32);  view_357 = None
        mul_231: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.5)
        mul_232: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.7071067811865476);  convert_element_type_355 = None
        erf_17: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_232);  mul_232 = None
        add_125: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_233: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_231, add_125);  mul_231 = add_125 = None
        convert_element_type_356: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_233, torch.bfloat16);  mul_233 = None
        convert_element_type_357: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg290_1, torch.bfloat16);  arg290_1 = None
        convert_element_type_358: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg291_1, torch.bfloat16);  arg291_1 = None
        view_358: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_356, _shape_param_413);  convert_element_type_356 = _shape_param_413 = None
        permute_197: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_358, [1, 0]);  convert_element_type_358 = None
        addmm_107: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_357, view_358, permute_197);  convert_element_type_357 = None
        view_359: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_107, _shape_param_414);  addmm_107 = _shape_param_414 = None
        inductor_lookup_seed_54: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 54)
        inductor_random_54: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_415, inductor_lookup_seed_54, 'rand');  _shape_param_415 = inductor_lookup_seed_54 = None
        convert_element_type_359: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_54, torch.bfloat16);  inductor_random_54 = None
        gt_54: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_359, 0.1);  convert_element_type_359 = None
        mul_234: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_54, view_359);  view_359 = None
        mul_235: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_234, 1.1111111111111112);  mul_234 = None
        add_126: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_235, add_124);  mul_235 = add_124 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
        getitem_72: "f32[8, 512, 1]" = var_mean_36[0]
        getitem_73: "f32[8, 512, 1]" = var_mean_36[1];  var_mean_36 = None
        add_127: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-07);  getitem_72 = None
        rsqrt_36: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        sub_54: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_126, getitem_73);  add_126 = getitem_73 = None
        mul_236: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = None
        mul_237: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_236, arg292_1);  arg292_1 = None
        add_128: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_237, arg293_1);  mul_237 = arg293_1 = None
        convert_element_type_360: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg294_1, torch.bfloat16);  arg294_1 = None
        convert_element_type_361: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg295_1, torch.bfloat16);  arg295_1 = None
        convert_element_type_362: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16)
        view_360: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_362, _shape_param_416);  convert_element_type_362 = _shape_param_416 = None
        permute_198: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        addmm_108: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_360, view_360, permute_198);  convert_element_type_360 = None
        view_361: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_108, _shape_param_417);  addmm_108 = _shape_param_417 = None
        view_362: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_361, _shape_param_418);  view_361 = _shape_param_418 = None
        permute_199: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
        clone_72: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_363: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_419);  clone_72 = _shape_param_419 = None
        convert_element_type_363: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.bfloat16);  arg296_1 = None
        convert_element_type_364: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.bfloat16);  arg297_1 = None
        permute_200: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_364, [1, 0]);  convert_element_type_364 = None
        addmm_109: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_363, view_360, permute_200);  convert_element_type_363 = None
        view_364: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_109, _shape_param_420);  addmm_109 = _shape_param_420 = None
        view_365: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_364, _shape_param_421);  view_364 = _shape_param_421 = None
        permute_201: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
        clone_73: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_366: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_73, _shape_param_422);  clone_73 = _shape_param_422 = None
        convert_element_type_365: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.bfloat16);  arg298_1 = None
        convert_element_type_366: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg299_1, torch.bfloat16);  arg299_1 = None
        permute_202: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_366, [1, 0]);  convert_element_type_366 = None
        addmm_110: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_365, view_360, permute_202);  convert_element_type_365 = None
        view_367: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_110, _shape_param_423);  addmm_110 = _shape_param_423 = None
        view_368: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_367, _shape_param_424);  view_367 = _shape_param_424 = None
        permute_203: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        clone_74: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_369: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_74, _shape_param_425);  clone_74 = _shape_param_425 = None
        permute_204: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_366, [0, 2, 1]);  view_366 = None
        div_36: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_204, full);  permute_204 = None
        bmm_36: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_363, div_36)
        view_370: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_36, _shape_param_426);  bmm_36 = _shape_param_426 = None
        where_18: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_370);  view_370 = None
        convert_element_type_367: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_18, torch.float32)
        amax_18: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_367, [-1], True)
        sub_55: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_367, amax_18);  convert_element_type_367 = None
        exp_18: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
        sum_19: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_37: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = None
        inductor_lookup_seed_55: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 55)
        inductor_random_55: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_427, inductor_lookup_seed_55, 'rand');  _shape_param_427 = inductor_lookup_seed_55 = None
        gt_55: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_55, 0.1);  inductor_random_55 = None
        mul_238: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_55, div_37);  div_37 = None
        mul_239: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_238, 1.1111111111111112);  mul_238 = None
        view_371: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_239, _shape_param_428);  mul_239 = _shape_param_428 = None
        convert_element_type_368: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_371, torch.bfloat16);  view_371 = None
        bmm_37: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_368, view_369)
        view_372: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_37, _shape_param_429);  bmm_37 = _shape_param_429 = None
        permute_205: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
        clone_75: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_373: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_75, _shape_param_430);  clone_75 = _shape_param_430 = None
        convert_element_type_369: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg300_1, torch.bfloat16);  arg300_1 = None
        convert_element_type_370: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg301_1, torch.bfloat16);  arg301_1 = None
        view_374: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_373, _shape_param_431);  view_373 = _shape_param_431 = None
        permute_206: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_370, [1, 0]);  convert_element_type_370 = None
        addmm_111: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_369, view_374, permute_206);  convert_element_type_369 = None
        view_375: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_111, _shape_param_432);  addmm_111 = _shape_param_432 = None
        inductor_lookup_seed_56: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 56)
        inductor_random_56: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_433, inductor_lookup_seed_56, 'rand');  _shape_param_433 = inductor_lookup_seed_56 = None
        convert_element_type_371: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_56, torch.bfloat16);  inductor_random_56 = None
        gt_56: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_371, 0.1);  convert_element_type_371 = None
        mul_240: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_56, view_375);  view_375 = None
        mul_241: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_240, 1.1111111111111112);  mul_240 = None
        add_129: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_241, add_128);  mul_241 = add_128 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_74: "f32[8, 512, 1]" = var_mean_37[0]
        getitem_75: "f32[8, 512, 1]" = var_mean_37[1];  var_mean_37 = None
        add_130: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-07);  getitem_74 = None
        rsqrt_37: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_56: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_129, getitem_75);  add_129 = getitem_75 = None
        mul_242: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
        mul_243: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_242, arg302_1);  arg302_1 = None
        add_131: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_243, arg303_1);  mul_243 = arg303_1 = None
        convert_element_type_372: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg304_1, torch.bfloat16);  arg304_1 = None
        convert_element_type_373: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.bfloat16);  arg305_1 = None
        convert_element_type_374: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16)
        view_376: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_374, _shape_param_434);  convert_element_type_374 = _shape_param_434 = None
        permute_207: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_373, [1, 0]);  convert_element_type_373 = None
        addmm_112: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_372, view_376, permute_207);  convert_element_type_372 = None
        view_377: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_112, _shape_param_435);  _shape_param_435 = None
        convert_element_type_375: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None
        mul_244: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_375, 0.5)
        mul_245: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_375, 0.7071067811865476);  convert_element_type_375 = None
        erf_18: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_245);  mul_245 = None
        add_132: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_246: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_244, add_132);  mul_244 = add_132 = None
        convert_element_type_376: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None
        convert_element_type_377: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg306_1, torch.bfloat16);  arg306_1 = None
        convert_element_type_378: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.bfloat16);  arg307_1 = None
        view_378: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_376, _shape_param_436);  convert_element_type_376 = _shape_param_436 = None
        permute_208: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_378, [1, 0]);  convert_element_type_378 = None
        addmm_113: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_377, view_378, permute_208);  convert_element_type_377 = None
        view_379: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_113, _shape_param_437);  addmm_113 = _shape_param_437 = None
        inductor_lookup_seed_57: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 57)
        inductor_random_57: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_438, inductor_lookup_seed_57, 'rand');  _shape_param_438 = inductor_lookup_seed_57 = None
        convert_element_type_379: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_57, torch.bfloat16);  inductor_random_57 = None
        gt_57: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_379, 0.1);  convert_element_type_379 = None
        mul_247: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_57, view_379);  view_379 = None
        mul_248: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_247, 1.1111111111111112);  mul_247 = None
        add_133: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_248, add_131);  mul_248 = add_131 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_76: "f32[8, 512, 1]" = var_mean_38[0]
        getitem_77: "f32[8, 512, 1]" = var_mean_38[1];  var_mean_38 = None
        add_134: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-07);  getitem_76 = None
        rsqrt_38: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_57: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_133, getitem_77);  add_133 = getitem_77 = None
        mul_249: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = None
        mul_250: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_249, arg308_1);  arg308_1 = None
        add_135: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_250, arg309_1);  mul_250 = arg309_1 = None
        convert_element_type_380: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.bfloat16);  arg310_1 = None
        convert_element_type_381: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.bfloat16);  arg311_1 = None
        convert_element_type_382: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16)
        view_380: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_382, _shape_param_439);  convert_element_type_382 = _shape_param_439 = None
        permute_209: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_381, [1, 0]);  convert_element_type_381 = None
        addmm_114: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_380, view_380, permute_209);  convert_element_type_380 = None
        view_381: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_114, _shape_param_440);  addmm_114 = _shape_param_440 = None
        view_382: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_381, _shape_param_441);  view_381 = _shape_param_441 = None
        permute_210: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        clone_76: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_383: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_442);  clone_76 = _shape_param_442 = None
        convert_element_type_383: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.bfloat16);  arg312_1 = None
        convert_element_type_384: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg313_1, torch.bfloat16);  arg313_1 = None
        permute_211: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_384, [1, 0]);  convert_element_type_384 = None
        addmm_115: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_383, view_380, permute_211);  convert_element_type_383 = None
        view_384: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_115, _shape_param_443);  addmm_115 = _shape_param_443 = None
        view_385: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_384, _shape_param_444);  view_384 = _shape_param_444 = None
        permute_212: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_385, [0, 2, 1, 3]);  view_385 = None
        clone_77: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
        view_386: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_77, _shape_param_445);  clone_77 = _shape_param_445 = None
        convert_element_type_385: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg314_1, torch.bfloat16);  arg314_1 = None
        convert_element_type_386: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.bfloat16);  arg315_1 = None
        permute_213: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_386, [1, 0]);  convert_element_type_386 = None
        addmm_116: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_385, view_380, permute_213);  convert_element_type_385 = None
        view_387: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_116, _shape_param_446);  addmm_116 = _shape_param_446 = None
        view_388: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_387, _shape_param_447);  view_387 = _shape_param_447 = None
        permute_214: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        clone_78: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
        view_389: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_78, _shape_param_448);  clone_78 = _shape_param_448 = None
        permute_215: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
        div_38: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_215, full);  permute_215 = None
        bmm_38: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_383, div_38)
        view_390: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_38, _shape_param_449);  bmm_38 = _shape_param_449 = None
        where_19: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_390);  view_390 = None
        convert_element_type_387: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_19, torch.float32)
        amax_19: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_387, [-1], True)
        sub_58: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_387, amax_19);  convert_element_type_387 = None
        exp_19: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
        sum_20: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_39: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = None
        inductor_lookup_seed_58: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 58)
        inductor_random_58: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_450, inductor_lookup_seed_58, 'rand');  _shape_param_450 = inductor_lookup_seed_58 = None
        gt_58: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_58, 0.1);  inductor_random_58 = None
        mul_251: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_58, div_39);  div_39 = None
        mul_252: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_251, 1.1111111111111112);  mul_251 = None
        view_391: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_252, _shape_param_451);  mul_252 = _shape_param_451 = None
        convert_element_type_388: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_391, torch.bfloat16);  view_391 = None
        bmm_39: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_388, view_389)
        view_392: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_452);  bmm_39 = _shape_param_452 = None
        permute_216: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
        clone_79: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
        view_393: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_79, _shape_param_453);  clone_79 = _shape_param_453 = None
        convert_element_type_389: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.bfloat16);  arg316_1 = None
        convert_element_type_390: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.bfloat16);  arg317_1 = None
        view_394: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_393, _shape_param_454);  view_393 = _shape_param_454 = None
        permute_217: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_390, [1, 0]);  convert_element_type_390 = None
        addmm_117: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_389, view_394, permute_217);  convert_element_type_389 = None
        view_395: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_117, _shape_param_455);  addmm_117 = _shape_param_455 = None
        inductor_lookup_seed_59: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 59)
        inductor_random_59: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_456, inductor_lookup_seed_59, 'rand');  _shape_param_456 = inductor_lookup_seed_59 = None
        convert_element_type_391: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_59, torch.bfloat16);  inductor_random_59 = None
        gt_59: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_391, 0.1);  convert_element_type_391 = None
        mul_253: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_59, view_395);  view_395 = None
        mul_254: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_253, 1.1111111111111112);  mul_253 = None
        add_136: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_254, add_135);  mul_254 = add_135 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
        getitem_78: "f32[8, 512, 1]" = var_mean_39[0]
        getitem_79: "f32[8, 512, 1]" = var_mean_39[1];  var_mean_39 = None
        add_137: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-07);  getitem_78 = None
        rsqrt_39: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_59: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_136, getitem_79);  add_136 = getitem_79 = None
        mul_255: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
        mul_256: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_255, arg318_1);  arg318_1 = None
        add_138: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_256, arg319_1);  mul_256 = arg319_1 = None
        convert_element_type_392: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.bfloat16);  arg320_1 = None
        convert_element_type_393: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg321_1, torch.bfloat16);  arg321_1 = None
        convert_element_type_394: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16)
        view_396: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_394, _shape_param_457);  convert_element_type_394 = _shape_param_457 = None
        permute_218: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_393, [1, 0]);  convert_element_type_393 = None
        addmm_118: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_392, view_396, permute_218);  convert_element_type_392 = None
        view_397: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_118, _shape_param_458);  _shape_param_458 = None
        convert_element_type_395: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_397, torch.float32);  view_397 = None
        mul_257: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_395, 0.5)
        mul_258: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_395, 0.7071067811865476);  convert_element_type_395 = None
        erf_19: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_258);  mul_258 = None
        add_139: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_259: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_257, add_139);  mul_257 = add_139 = None
        convert_element_type_396: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_259, torch.bfloat16);  mul_259 = None
        convert_element_type_397: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg322_1, torch.bfloat16);  arg322_1 = None
        convert_element_type_398: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.bfloat16);  arg323_1 = None
        view_398: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_396, _shape_param_459);  convert_element_type_396 = _shape_param_459 = None
        permute_219: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_398, [1, 0]);  convert_element_type_398 = None
        addmm_119: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_397, view_398, permute_219);  convert_element_type_397 = None
        view_399: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_119, _shape_param_460);  addmm_119 = _shape_param_460 = None
        inductor_lookup_seed_60: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 60)
        inductor_random_60: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_461, inductor_lookup_seed_60, 'rand');  _shape_param_461 = inductor_lookup_seed_60 = None
        convert_element_type_399: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_60, torch.bfloat16);  inductor_random_60 = None
        gt_60: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_399, 0.1);  convert_element_type_399 = None
        mul_260: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_60, view_399);  view_399 = None
        mul_261: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_260, 1.1111111111111112);  mul_260 = None
        add_140: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_261, add_138);  mul_261 = add_138 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
        getitem_80: "f32[8, 512, 1]" = var_mean_40[0]
        getitem_81: "f32[8, 512, 1]" = var_mean_40[1];  var_mean_40 = None
        add_141: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-07);  getitem_80 = None
        rsqrt_40: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_60: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_140, getitem_81);  add_140 = getitem_81 = None
        mul_262: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = None
        mul_263: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_262, arg324_1);  arg324_1 = None
        add_142: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_263, arg325_1);  mul_263 = arg325_1 = None
        convert_element_type_400: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg326_1, torch.bfloat16);  arg326_1 = None
        convert_element_type_401: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.bfloat16);  arg327_1 = None
        convert_element_type_402: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16)
        view_400: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_402, _shape_param_462);  convert_element_type_402 = _shape_param_462 = None
        permute_220: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_401, [1, 0]);  convert_element_type_401 = None
        addmm_120: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_400, view_400, permute_220);  convert_element_type_400 = None
        view_401: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_120, _shape_param_463);  addmm_120 = _shape_param_463 = None
        view_402: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_401, _shape_param_464);  view_401 = _shape_param_464 = None
        permute_221: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_80: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_403: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_80, _shape_param_465);  clone_80 = _shape_param_465 = None
        convert_element_type_403: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.bfloat16);  arg328_1 = None
        convert_element_type_404: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg329_1, torch.bfloat16);  arg329_1 = None
        permute_222: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_404, [1, 0]);  convert_element_type_404 = None
        addmm_121: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_403, view_400, permute_222);  convert_element_type_403 = None
        view_404: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_121, _shape_param_466);  addmm_121 = _shape_param_466 = None
        view_405: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_404, _shape_param_467);  view_404 = _shape_param_467 = None
        permute_223: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
        clone_81: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_406: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_81, _shape_param_468);  clone_81 = _shape_param_468 = None
        convert_element_type_405: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg330_1, torch.bfloat16);  arg330_1 = None
        convert_element_type_406: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg331_1, torch.bfloat16);  arg331_1 = None
        permute_224: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_406, [1, 0]);  convert_element_type_406 = None
        addmm_122: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_405, view_400, permute_224);  convert_element_type_405 = None
        view_407: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_122, _shape_param_469);  addmm_122 = _shape_param_469 = None
        view_408: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_407, _shape_param_470);  view_407 = _shape_param_470 = None
        permute_225: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None
        clone_82: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_409: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_82, _shape_param_471);  clone_82 = _shape_param_471 = None
        permute_226: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
        div_40: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_226, full);  permute_226 = None
        bmm_40: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_403, div_40)
        view_410: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_40, _shape_param_472);  bmm_40 = _shape_param_472 = None
        where_20: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_410);  view_410 = None
        convert_element_type_407: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_20, torch.float32)
        amax_20: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_407, [-1], True)
        sub_61: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_407, amax_20);  convert_element_type_407 = None
        exp_20: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_21: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_41: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = None
        inductor_lookup_seed_61: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 61)
        inductor_random_61: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_473, inductor_lookup_seed_61, 'rand');  _shape_param_473 = inductor_lookup_seed_61 = None
        gt_61: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_61, 0.1);  inductor_random_61 = None
        mul_264: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_61, div_41);  div_41 = None
        mul_265: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_264, 1.1111111111111112);  mul_264 = None
        view_411: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_265, _shape_param_474);  mul_265 = _shape_param_474 = None
        convert_element_type_408: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_411, torch.bfloat16);  view_411 = None
        bmm_41: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_408, view_409)
        view_412: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_41, _shape_param_475);  bmm_41 = _shape_param_475 = None
        permute_227: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
        clone_83: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_413: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_83, _shape_param_476);  clone_83 = _shape_param_476 = None
        convert_element_type_409: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg332_1, torch.bfloat16);  arg332_1 = None
        convert_element_type_410: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.bfloat16);  arg333_1 = None
        view_414: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_413, _shape_param_477);  view_413 = _shape_param_477 = None
        permute_228: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_410, [1, 0]);  convert_element_type_410 = None
        addmm_123: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_409, view_414, permute_228);  convert_element_type_409 = None
        view_415: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_123, _shape_param_478);  addmm_123 = _shape_param_478 = None
        inductor_lookup_seed_62: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 62)
        inductor_random_62: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_479, inductor_lookup_seed_62, 'rand');  _shape_param_479 = inductor_lookup_seed_62 = None
        convert_element_type_411: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_62, torch.bfloat16);  inductor_random_62 = None
        gt_62: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_411, 0.1);  convert_element_type_411 = None
        mul_266: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_62, view_415);  view_415 = None
        mul_267: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_266, 1.1111111111111112);  mul_266 = None
        add_143: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_267, add_142);  mul_267 = add_142 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_82: "f32[8, 512, 1]" = var_mean_41[0]
        getitem_83: "f32[8, 512, 1]" = var_mean_41[1];  var_mean_41 = None
        add_144: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-07);  getitem_82 = None
        rsqrt_41: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        sub_62: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_143, getitem_83);  add_143 = getitem_83 = None
        mul_268: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
        mul_269: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_268, arg334_1);  arg334_1 = None
        add_145: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_269, arg335_1);  mul_269 = arg335_1 = None
        convert_element_type_412: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg336_1, torch.bfloat16);  arg336_1 = None
        convert_element_type_413: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg337_1, torch.bfloat16);  arg337_1 = None
        convert_element_type_414: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_145, torch.bfloat16)
        view_416: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_414, _shape_param_480);  convert_element_type_414 = _shape_param_480 = None
        permute_229: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_413, [1, 0]);  convert_element_type_413 = None
        addmm_124: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_412, view_416, permute_229);  convert_element_type_412 = None
        view_417: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_124, _shape_param_481);  _shape_param_481 = None
        convert_element_type_415: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None
        mul_270: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_415, 0.5)
        mul_271: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_415, 0.7071067811865476);  convert_element_type_415 = None
        erf_20: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_271);  mul_271 = None
        add_146: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_272: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_270, add_146);  mul_270 = add_146 = None
        convert_element_type_416: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_272, torch.bfloat16);  mul_272 = None
        convert_element_type_417: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.bfloat16);  arg338_1 = None
        convert_element_type_418: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg339_1, torch.bfloat16);  arg339_1 = None
        view_418: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_416, _shape_param_482);  convert_element_type_416 = _shape_param_482 = None
        permute_230: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_418, [1, 0]);  convert_element_type_418 = None
        addmm_125: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_417, view_418, permute_230);  convert_element_type_417 = None
        view_419: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_125, _shape_param_483);  addmm_125 = _shape_param_483 = None
        inductor_lookup_seed_63: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 63)
        inductor_random_63: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_484, inductor_lookup_seed_63, 'rand');  _shape_param_484 = inductor_lookup_seed_63 = None
        convert_element_type_419: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_63, torch.bfloat16);  inductor_random_63 = None
        gt_63: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_419, 0.1);  convert_element_type_419 = None
        mul_273: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_63, view_419);  view_419 = None
        mul_274: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_273, 1.1111111111111112);  mul_273 = None
        add_147: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_274, add_145);  mul_274 = add_145 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_84: "f32[8, 512, 1]" = var_mean_42[0]
        getitem_85: "f32[8, 512, 1]" = var_mean_42[1];  var_mean_42 = None
        add_148: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-07);  getitem_84 = None
        rsqrt_42: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_63: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_147, getitem_85);  add_147 = getitem_85 = None
        mul_275: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = None
        mul_276: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_275, arg340_1);  arg340_1 = None
        add_149: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_276, arg341_1);  mul_276 = arg341_1 = None
        convert_element_type_420: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.bfloat16);  arg342_1 = None
        convert_element_type_421: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.bfloat16);  arg343_1 = None
        convert_element_type_422: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16)
        view_420: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_422, _shape_param_485);  convert_element_type_422 = _shape_param_485 = None
        permute_231: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_421, [1, 0]);  convert_element_type_421 = None
        addmm_126: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_420, view_420, permute_231);  convert_element_type_420 = None
        view_421: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_126, _shape_param_486);  addmm_126 = _shape_param_486 = None
        view_422: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_421, _shape_param_487);  view_421 = _shape_param_487 = None
        permute_232: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
        clone_84: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
        view_423: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_84, _shape_param_488);  clone_84 = _shape_param_488 = None
        convert_element_type_423: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.bfloat16);  arg344_1 = None
        convert_element_type_424: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg345_1, torch.bfloat16);  arg345_1 = None
        permute_233: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_424, [1, 0]);  convert_element_type_424 = None
        addmm_127: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_423, view_420, permute_233);  convert_element_type_423 = None
        view_424: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_127, _shape_param_489);  addmm_127 = _shape_param_489 = None
        view_425: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_424, _shape_param_490);  view_424 = _shape_param_490 = None
        permute_234: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
        clone_85: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
        view_426: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_85, _shape_param_491);  clone_85 = _shape_param_491 = None
        convert_element_type_425: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg346_1, torch.bfloat16);  arg346_1 = None
        convert_element_type_426: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg347_1, torch.bfloat16);  arg347_1 = None
        permute_235: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_426, [1, 0]);  convert_element_type_426 = None
        addmm_128: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_425, view_420, permute_235);  convert_element_type_425 = None
        view_427: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_128, _shape_param_492);  addmm_128 = _shape_param_492 = None
        view_428: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_427, _shape_param_493);  view_427 = _shape_param_493 = None
        permute_236: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        clone_86: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_429: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_86, _shape_param_494);  clone_86 = _shape_param_494 = None
        permute_237: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_426, [0, 2, 1]);  view_426 = None
        div_42: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_237, full);  permute_237 = None
        bmm_42: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_423, div_42)
        view_430: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_42, _shape_param_495);  bmm_42 = _shape_param_495 = None
        where_21: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_430);  view_430 = None
        convert_element_type_427: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_21, torch.float32)
        amax_21: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_427, [-1], True)
        sub_64: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_427, amax_21);  convert_element_type_427 = None
        exp_21: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
        sum_22: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_43: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = None
        inductor_lookup_seed_64: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 64)
        inductor_random_64: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_496, inductor_lookup_seed_64, 'rand');  _shape_param_496 = inductor_lookup_seed_64 = None
        gt_64: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_64, 0.1);  inductor_random_64 = None
        mul_277: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_64, div_43);  div_43 = None
        mul_278: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_277, 1.1111111111111112);  mul_277 = None
        view_431: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_278, _shape_param_497);  mul_278 = _shape_param_497 = None
        convert_element_type_428: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_431, torch.bfloat16);  view_431 = None
        bmm_43: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_428, view_429)
        view_432: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_498);  bmm_43 = _shape_param_498 = None
        permute_238: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        clone_87: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_433: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_87, _shape_param_499);  clone_87 = _shape_param_499 = None
        convert_element_type_429: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg348_1, torch.bfloat16);  arg348_1 = None
        convert_element_type_430: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg349_1, torch.bfloat16);  arg349_1 = None
        view_434: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_433, _shape_param_500);  view_433 = _shape_param_500 = None
        permute_239: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_430, [1, 0]);  convert_element_type_430 = None
        addmm_129: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_429, view_434, permute_239);  convert_element_type_429 = None
        view_435: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_129, _shape_param_501);  addmm_129 = _shape_param_501 = None
        inductor_lookup_seed_65: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 65)
        inductor_random_65: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_502, inductor_lookup_seed_65, 'rand');  _shape_param_502 = inductor_lookup_seed_65 = None
        convert_element_type_431: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_65, torch.bfloat16);  inductor_random_65 = None
        gt_65: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_431, 0.1);  convert_element_type_431 = None
        mul_279: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_65, view_435);  view_435 = None
        mul_280: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_279, 1.1111111111111112);  mul_279 = None
        add_150: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_280, add_149);  mul_280 = add_149 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
        getitem_86: "f32[8, 512, 1]" = var_mean_43[0]
        getitem_87: "f32[8, 512, 1]" = var_mean_43[1];  var_mean_43 = None
        add_151: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-07);  getitem_86 = None
        rsqrt_43: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_65: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_150, getitem_87);  add_150 = getitem_87 = None
        mul_281: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
        mul_282: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_281, arg350_1);  arg350_1 = None
        add_152: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_282, arg351_1);  mul_282 = arg351_1 = None
        convert_element_type_432: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg352_1, torch.bfloat16);  arg352_1 = None
        convert_element_type_433: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg353_1, torch.bfloat16);  arg353_1 = None
        convert_element_type_434: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16)
        view_436: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_434, _shape_param_503);  convert_element_type_434 = _shape_param_503 = None
        permute_240: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_433, [1, 0]);  convert_element_type_433 = None
        addmm_130: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_432, view_436, permute_240);  convert_element_type_432 = None
        view_437: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_130, _shape_param_504);  _shape_param_504 = None
        convert_element_type_435: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        mul_283: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_435, 0.5)
        mul_284: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_435, 0.7071067811865476);  convert_element_type_435 = None
        erf_21: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_284);  mul_284 = None
        add_153: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_285: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_283, add_153);  mul_283 = add_153 = None
        convert_element_type_436: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_285, torch.bfloat16);  mul_285 = None
        convert_element_type_437: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg354_1, torch.bfloat16);  arg354_1 = None
        convert_element_type_438: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg355_1, torch.bfloat16);  arg355_1 = None
        view_438: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_436, _shape_param_505);  convert_element_type_436 = _shape_param_505 = None
        permute_241: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_438, [1, 0]);  convert_element_type_438 = None
        addmm_131: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_437, view_438, permute_241);  convert_element_type_437 = None
        view_439: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_131, _shape_param_506);  addmm_131 = _shape_param_506 = None
        inductor_lookup_seed_66: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 66)
        inductor_random_66: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_507, inductor_lookup_seed_66, 'rand');  _shape_param_507 = inductor_lookup_seed_66 = None
        convert_element_type_439: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_66, torch.bfloat16);  inductor_random_66 = None
        gt_66: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_439, 0.1);  convert_element_type_439 = None
        mul_286: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_66, view_439);  view_439 = None
        mul_287: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_286, 1.1111111111111112);  mul_286 = None
        add_154: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_287, add_152);  mul_287 = add_152 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
        getitem_88: "f32[8, 512, 1]" = var_mean_44[0]
        getitem_89: "f32[8, 512, 1]" = var_mean_44[1];  var_mean_44 = None
        add_155: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-07);  getitem_88 = None
        rsqrt_44: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        sub_66: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_154, getitem_89);  add_154 = getitem_89 = None
        mul_288: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = None
        mul_289: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_288, arg356_1);  arg356_1 = None
        add_156: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_289, arg357_1);  mul_289 = arg357_1 = None
        convert_element_type_440: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.bfloat16);  arg358_1 = None
        convert_element_type_441: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg359_1, torch.bfloat16);  arg359_1 = None
        convert_element_type_442: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_156, torch.bfloat16)
        view_440: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_442, _shape_param_508);  convert_element_type_442 = _shape_param_508 = None
        permute_242: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_441, [1, 0]);  convert_element_type_441 = None
        addmm_132: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_440, view_440, permute_242);  convert_element_type_440 = None
        view_441: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_132, _shape_param_509);  addmm_132 = _shape_param_509 = None
        view_442: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_441, _shape_param_510);  view_441 = _shape_param_510 = None
        permute_243: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        clone_88: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_443: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_88, _shape_param_511);  clone_88 = _shape_param_511 = None
        convert_element_type_443: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg360_1, torch.bfloat16);  arg360_1 = None
        convert_element_type_444: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg361_1, torch.bfloat16);  arg361_1 = None
        permute_244: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_444, [1, 0]);  convert_element_type_444 = None
        addmm_133: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_443, view_440, permute_244);  convert_element_type_443 = None
        view_444: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_133, _shape_param_512);  addmm_133 = _shape_param_512 = None
        view_445: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_444, _shape_param_513);  view_444 = _shape_param_513 = None
        permute_245: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
        clone_89: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_245, memory_format = torch.contiguous_format);  permute_245 = None
        view_446: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_89, _shape_param_514);  clone_89 = _shape_param_514 = None
        convert_element_type_445: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg362_1, torch.bfloat16);  arg362_1 = None
        convert_element_type_446: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg363_1, torch.bfloat16);  arg363_1 = None
        permute_246: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_446, [1, 0]);  convert_element_type_446 = None
        addmm_134: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_445, view_440, permute_246);  convert_element_type_445 = None
        view_447: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_134, _shape_param_515);  addmm_134 = _shape_param_515 = None
        view_448: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_447, _shape_param_516);  view_447 = _shape_param_516 = None
        permute_247: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        clone_90: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_449: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_90, _shape_param_517);  clone_90 = _shape_param_517 = None
        permute_248: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_446, [0, 2, 1]);  view_446 = None
        div_44: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_248, full);  permute_248 = None
        bmm_44: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_443, div_44)
        view_450: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_44, _shape_param_518);  bmm_44 = _shape_param_518 = None
        where_22: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_450);  view_450 = None
        convert_element_type_447: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_22, torch.float32)
        amax_22: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_447, [-1], True)
        sub_67: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_447, amax_22);  convert_element_type_447 = None
        exp_22: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
        sum_23: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_45: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = None
        inductor_lookup_seed_67: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 67)
        inductor_random_67: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_519, inductor_lookup_seed_67, 'rand');  _shape_param_519 = inductor_lookup_seed_67 = None
        gt_67: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_67, 0.1);  inductor_random_67 = None
        mul_290: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_67, div_45);  div_45 = None
        mul_291: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_290, 1.1111111111111112);  mul_290 = None
        view_451: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_291, _shape_param_520);  mul_291 = _shape_param_520 = None
        convert_element_type_448: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_451, torch.bfloat16);  view_451 = None
        bmm_45: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_448, view_449)
        view_452: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_45, _shape_param_521);  bmm_45 = _shape_param_521 = None
        permute_249: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
        clone_91: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_453: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_91, _shape_param_522);  clone_91 = _shape_param_522 = None
        convert_element_type_449: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg364_1, torch.bfloat16);  arg364_1 = None
        convert_element_type_450: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg365_1, torch.bfloat16);  arg365_1 = None
        view_454: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_453, _shape_param_523);  view_453 = _shape_param_523 = None
        permute_250: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_450, [1, 0]);  convert_element_type_450 = None
        addmm_135: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_449, view_454, permute_250);  convert_element_type_449 = None
        view_455: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_135, _shape_param_524);  addmm_135 = _shape_param_524 = None
        inductor_lookup_seed_68: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 68)
        inductor_random_68: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_525, inductor_lookup_seed_68, 'rand');  _shape_param_525 = inductor_lookup_seed_68 = None
        convert_element_type_451: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_68, torch.bfloat16);  inductor_random_68 = None
        gt_68: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_451, 0.1);  convert_element_type_451 = None
        mul_292: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_68, view_455);  view_455 = None
        mul_293: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_292, 1.1111111111111112);  mul_292 = None
        add_157: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_293, add_156);  mul_293 = add_156 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_90: "f32[8, 512, 1]" = var_mean_45[0]
        getitem_91: "f32[8, 512, 1]" = var_mean_45[1];  var_mean_45 = None
        add_158: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-07);  getitem_90 = None
        rsqrt_45: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_68: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_157, getitem_91);  add_157 = getitem_91 = None
        mul_294: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
        mul_295: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_294, arg366_1);  arg366_1 = None
        add_159: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_295, arg367_1);  mul_295 = arg367_1 = None
        convert_element_type_452: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg368_1, torch.bfloat16);  arg368_1 = None
        convert_element_type_453: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg369_1, torch.bfloat16);  arg369_1 = None
        convert_element_type_454: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16)
        view_456: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_454, _shape_param_526);  convert_element_type_454 = _shape_param_526 = None
        permute_251: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_453, [1, 0]);  convert_element_type_453 = None
        addmm_136: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_452, view_456, permute_251);  convert_element_type_452 = None
        view_457: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_136, _shape_param_527);  _shape_param_527 = None
        convert_element_type_455: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_457, torch.float32);  view_457 = None
        mul_296: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_455, 0.5)
        mul_297: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_455, 0.7071067811865476);  convert_element_type_455 = None
        erf_22: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_297);  mul_297 = None
        add_160: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_298: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_296, add_160);  mul_296 = add_160 = None
        convert_element_type_456: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_298, torch.bfloat16);  mul_298 = None
        convert_element_type_457: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg370_1, torch.bfloat16);  arg370_1 = None
        convert_element_type_458: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg371_1, torch.bfloat16);  arg371_1 = None
        view_458: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_456, _shape_param_528);  convert_element_type_456 = _shape_param_528 = None
        permute_252: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_458, [1, 0]);  convert_element_type_458 = None
        addmm_137: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_457, view_458, permute_252);  convert_element_type_457 = None
        view_459: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_137, _shape_param_529);  addmm_137 = _shape_param_529 = None
        inductor_lookup_seed_69: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 69)
        inductor_random_69: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_530, inductor_lookup_seed_69, 'rand');  _shape_param_530 = inductor_lookup_seed_69 = None
        convert_element_type_459: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_69, torch.bfloat16);  inductor_random_69 = None
        gt_69: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_459, 0.1);  convert_element_type_459 = None
        mul_299: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_69, view_459);  view_459 = None
        mul_300: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_299, 1.1111111111111112);  mul_299 = None
        add_161: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_300, add_159);  mul_300 = add_159 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_92: "f32[8, 512, 1]" = var_mean_46[0]
        getitem_93: "f32[8, 512, 1]" = var_mean_46[1];  var_mean_46 = None
        add_162: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-07);  getitem_92 = None
        rsqrt_46: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_69: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_161, getitem_93);  add_161 = getitem_93 = None
        mul_301: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = None
        mul_302: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_301, arg372_1);  arg372_1 = None
        add_163: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_302, arg373_1);  mul_302 = arg373_1 = None
        convert_element_type_460: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg374_1, torch.bfloat16);  arg374_1 = None
        convert_element_type_461: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg375_1, torch.bfloat16);  arg375_1 = None
        convert_element_type_462: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_163, torch.bfloat16)
        view_460: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_462, _shape_param_531);  convert_element_type_462 = _shape_param_531 = None
        permute_253: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_461, [1, 0]);  convert_element_type_461 = None
        addmm_138: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_460, view_460, permute_253);  convert_element_type_460 = None
        view_461: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_138, _shape_param_532);  addmm_138 = _shape_param_532 = None
        view_462: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_461, _shape_param_533);  view_461 = _shape_param_533 = None
        permute_254: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
        clone_92: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_463: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_92, _shape_param_534);  clone_92 = _shape_param_534 = None
        convert_element_type_463: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg376_1, torch.bfloat16);  arg376_1 = None
        convert_element_type_464: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg377_1, torch.bfloat16);  arg377_1 = None
        permute_255: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_464, [1, 0]);  convert_element_type_464 = None
        addmm_139: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_463, view_460, permute_255);  convert_element_type_463 = None
        view_464: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_139, _shape_param_535);  addmm_139 = _shape_param_535 = None
        view_465: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_464, _shape_param_536);  view_464 = _shape_param_536 = None
        permute_256: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
        clone_93: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_466: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_93, _shape_param_537);  clone_93 = _shape_param_537 = None
        convert_element_type_465: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg378_1, torch.bfloat16);  arg378_1 = None
        convert_element_type_466: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg379_1, torch.bfloat16);  arg379_1 = None
        permute_257: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_466, [1, 0]);  convert_element_type_466 = None
        addmm_140: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_465, view_460, permute_257);  convert_element_type_465 = None
        view_467: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_140, _shape_param_538);  addmm_140 = _shape_param_538 = None
        view_468: "bf16[8, 512, 24, 64]" = torch.ops.aten.view.default(view_467, _shape_param_539);  view_467 = _shape_param_539 = None
        permute_258: "bf16[8, 24, 512, 64]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        clone_94: "bf16[8, 24, 512, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
        view_469: "bf16[192, 512, 64]" = torch.ops.aten.view.default(clone_94, _shape_param_540);  clone_94 = _shape_param_540 = None
        permute_259: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_466, [0, 2, 1]);  view_466 = None
        div_46: "bf16[192, 64, 512]" = torch.ops.aten.div.Tensor(permute_259, full);  permute_259 = full = None
        bmm_46: "bf16[192, 512, 512]" = torch.ops.aten.bmm.default(view_463, div_46)
        view_470: "bf16[8, 24, 512, 512]" = torch.ops.aten.view.default(bmm_46, _shape_param_541);  bmm_46 = _shape_param_541 = None
        where_23: "bf16[8, 24, 512, 512]" = torch.ops.aten.where.self(full_1, full_2, view_470);  full_1 = full_2 = view_470 = None
        convert_element_type_467: "f32[8, 24, 512, 512]" = torch.ops.prims.convert_element_type.default(where_23, torch.float32)
        amax_23: "f32[8, 24, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_467, [-1], True)
        sub_70: "f32[8, 24, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_467, amax_23);  convert_element_type_467 = None
        exp_23: "f32[8, 24, 512, 512]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
        sum_24: "f32[8, 24, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_47: "f32[8, 24, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = None
        inductor_lookup_seed_70: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 70)
        inductor_random_70: "f32[8, 24, 512, 512]" = torch.ops.prims.inductor_random.default(_shape_param_542, inductor_lookup_seed_70, 'rand');  _shape_param_542 = inductor_lookup_seed_70 = None
        gt_70: "b8[8, 24, 512, 512]" = torch.ops.aten.gt.Scalar(inductor_random_70, 0.1);  inductor_random_70 = None
        mul_303: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(gt_70, div_47);  div_47 = None
        mul_304: "f32[8, 24, 512, 512]" = torch.ops.aten.mul.Tensor(mul_303, 1.1111111111111112);  mul_303 = None
        view_471: "f32[192, 512, 512]" = torch.ops.aten.view.default(mul_304, _shape_param_543);  mul_304 = _shape_param_543 = None
        convert_element_type_468: "bf16[192, 512, 512]" = torch.ops.prims.convert_element_type.default(view_471, torch.bfloat16);  view_471 = None
        bmm_47: "bf16[192, 512, 64]" = torch.ops.aten.bmm.default(convert_element_type_468, view_469)
        view_472: "bf16[8, 24, 512, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_544);  bmm_47 = _shape_param_544 = None
        permute_260: "bf16[8, 512, 24, 64]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
        clone_95: "bf16[8, 512, 24, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
        view_473: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(clone_95, _shape_param_545);  clone_95 = _shape_param_545 = None
        convert_element_type_469: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg380_1, torch.bfloat16);  arg380_1 = None
        convert_element_type_470: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg381_1, torch.bfloat16);  arg381_1 = None
        view_474: "bf16[4096, 1536]" = torch.ops.aten.view.default(view_473, _shape_param_546);  view_473 = _shape_param_546 = None
        permute_261: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_470, [1, 0]);  convert_element_type_470 = None
        addmm_141: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_469, view_474, permute_261);  convert_element_type_469 = None
        view_475: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_141, _shape_param_547);  addmm_141 = _shape_param_547 = None
        inductor_lookup_seed_71: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 71)
        inductor_random_71: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_548, inductor_lookup_seed_71, 'rand');  _shape_param_548 = inductor_lookup_seed_71 = None
        convert_element_type_471: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_71, torch.bfloat16);  inductor_random_71 = None
        gt_71: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_471, 0.1);  convert_element_type_471 = None
        mul_305: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_71, view_475);  view_475 = None
        mul_306: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_305, 1.1111111111111112);  mul_305 = None
        add_164: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_306, add_163);  mul_306 = add_163 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
        getitem_94: "f32[8, 512, 1]" = var_mean_47[0]
        getitem_95: "f32[8, 512, 1]" = var_mean_47[1];  var_mean_47 = None
        add_165: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-07);  getitem_94 = None
        rsqrt_47: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_71: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_164, getitem_95);  add_164 = getitem_95 = None
        mul_307: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
        mul_308: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_307, arg382_1);  arg382_1 = None
        add_166: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_308, arg383_1);  mul_308 = arg383_1 = None
        convert_element_type_472: "bf16[6144]" = torch.ops.prims.convert_element_type.default(arg384_1, torch.bfloat16);  arg384_1 = None
        convert_element_type_473: "bf16[6144, 1536]" = torch.ops.prims.convert_element_type.default(arg385_1, torch.bfloat16);  arg385_1 = None
        convert_element_type_474: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_166, torch.bfloat16)
        view_476: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_474, _shape_param_549);  convert_element_type_474 = _shape_param_549 = None
        permute_262: "bf16[1536, 6144]" = torch.ops.aten.permute.default(convert_element_type_473, [1, 0]);  convert_element_type_473 = None
        addmm_142: "bf16[4096, 6144]" = torch.ops.aten.addmm.default(convert_element_type_472, view_476, permute_262);  convert_element_type_472 = None
        view_477: "bf16[8, 512, 6144]" = torch.ops.aten.view.default(addmm_142, _shape_param_550);  _shape_param_550 = None
        convert_element_type_475: "f32[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(view_477, torch.float32);  view_477 = None
        mul_309: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_475, 0.5)
        mul_310: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_475, 0.7071067811865476);  convert_element_type_475 = None
        erf_23: "f32[8, 512, 6144]" = torch.ops.aten.erf.default(mul_310);  mul_310 = None
        add_167: "f32[8, 512, 6144]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_311: "f32[8, 512, 6144]" = torch.ops.aten.mul.Tensor(mul_309, add_167);  mul_309 = add_167 = None
        convert_element_type_476: "bf16[8, 512, 6144]" = torch.ops.prims.convert_element_type.default(mul_311, torch.bfloat16);  mul_311 = None
        convert_element_type_477: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg386_1, torch.bfloat16);  arg386_1 = None
        convert_element_type_478: "bf16[1536, 6144]" = torch.ops.prims.convert_element_type.default(arg387_1, torch.bfloat16);  arg387_1 = None
        view_478: "bf16[4096, 6144]" = torch.ops.aten.view.default(convert_element_type_476, _shape_param_551);  convert_element_type_476 = _shape_param_551 = None
        permute_263: "bf16[6144, 1536]" = torch.ops.aten.permute.default(convert_element_type_478, [1, 0]);  convert_element_type_478 = None
        addmm_143: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_477, view_478, permute_263);  convert_element_type_477 = None
        view_479: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_143, _shape_param_552);  addmm_143 = _shape_param_552 = None
        inductor_lookup_seed_72: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 72);  inductor_seeds = None
        inductor_random_72: "f32[8, 512, 1536]" = torch.ops.prims.inductor_random.default(_shape_param_553, inductor_lookup_seed_72, 'rand');  _shape_param_553 = inductor_lookup_seed_72 = None
        convert_element_type_479: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(inductor_random_72, torch.bfloat16);  inductor_random_72 = None
        gt_72: "b8[8, 512, 1536]" = torch.ops.aten.gt.Scalar(convert_element_type_479, 0.1);  convert_element_type_479 = None
        mul_312: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(gt_72, view_479);  view_479 = None
        mul_313: "bf16[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_312, 1.1111111111111112);  mul_312 = None
        add_168: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_313, add_166);  mul_313 = add_166 = None
        var_mean_48 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
        getitem_96: "f32[8, 512, 1]" = var_mean_48[0]
        getitem_97: "f32[8, 512, 1]" = var_mean_48[1];  var_mean_48 = None
        add_169: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-07);  getitem_96 = None
        rsqrt_48: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        sub_72: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_168, getitem_97);  add_168 = getitem_97 = None
        mul_314: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = None
        mul_315: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_314, arg388_1);  arg388_1 = None
        add_170: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_315, arg389_1);  mul_315 = arg389_1 = None
        convert_element_type_480: "bf16[1536]" = torch.ops.prims.convert_element_type.default(arg390_1, torch.bfloat16);  arg390_1 = None
        convert_element_type_481: "bf16[1536, 1536]" = torch.ops.prims.convert_element_type.default(arg391_1, torch.bfloat16);  arg391_1 = None
        convert_element_type_482: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16);  add_170 = None
        view_480: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_482, _shape_param_554);  convert_element_type_482 = _shape_param_554 = None
        permute_264: "bf16[1536, 1536]" = torch.ops.aten.permute.default(convert_element_type_481, [1, 0]);  convert_element_type_481 = None
        addmm_144: "bf16[4096, 1536]" = torch.ops.aten.addmm.default(convert_element_type_480, view_480, permute_264);  convert_element_type_480 = None
        view_481: "bf16[8, 512, 1536]" = torch.ops.aten.view.default(addmm_144, _shape_param_555);  _shape_param_555 = None
        convert_element_type_483: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(view_481, torch.float32);  view_481 = None
        mul_316: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_483, 0.5)
        mul_317: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_483, 0.7071067811865476);  convert_element_type_483 = None
        erf_24: "f32[8, 512, 1536]" = torch.ops.aten.erf.default(mul_317);  mul_317 = None
        add_171: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_318: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_316, add_171);  mul_316 = add_171 = None
        convert_element_type_484: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(mul_318, torch.bfloat16);  mul_318 = None
        convert_element_type_485: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(convert_element_type_484, torch.float32);  convert_element_type_484 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_485, [2], correction = 0, keepdim = True)
        getitem_98: "f32[8, 512, 1]" = var_mean_49[0]
        getitem_99: "f32[8, 512, 1]" = var_mean_49[1];  var_mean_49 = None
        add_172: "f32[8, 512, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-07);  getitem_98 = None
        rsqrt_49: "f32[8, 512, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        sub_73: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(convert_element_type_485, getitem_99);  convert_element_type_485 = None
        mul_319: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_49);  sub_73 = None
        mul_320: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_319, arg392_1);  mul_319 = arg392_1 = None
        add_173: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_320, arg393_1);  mul_320 = arg393_1 = None
        convert_element_type_486: "bf16[128100]" = torch.ops.prims.convert_element_type.default(arg394_1, torch.bfloat16);  arg394_1 = None
        convert_element_type_487: "bf16[128100, 1536]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.bfloat16);  arg0_1 = None
        convert_element_type_488: "bf16[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16);  add_173 = None
        view_482: "bf16[4096, 1536]" = torch.ops.aten.view.default(convert_element_type_488, _shape_param_556);  convert_element_type_488 = _shape_param_556 = None
        permute_265: "bf16[1536, 128100]" = torch.ops.aten.permute.default(convert_element_type_487, [1, 0]);  convert_element_type_487 = None
        constant_pad_nd: "bf16[1536, 128104]" = torch.ops.aten.constant_pad_nd.default(permute_265, _shape_param_557);  _shape_param_557 = None
        full_3: "bf16[4]" = torch.ops.aten.full.default(_shape_param_558, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_558 = None
        cat: "bf16[128104]" = torch.ops.aten.cat.default([convert_element_type_486, full_3]);  convert_element_type_486 = full_3 = None
        addmm_145: "bf16[4096, 128104]" = torch.ops.aten.addmm.default(cat, view_482, constant_pad_nd);  cat = constant_pad_nd = None
        slice_1: "bf16[4096, 128100]" = torch.ops.aten.slice.Tensor(addmm_145, 1, 0, -4);  addmm_145 = None
        view_483: "bf16[8, 512, 128100]" = torch.ops.aten.view.default(slice_1, _shape_param_559);  slice_1 = _shape_param_559 = None
        view_484: "bf16[4096, 128100]" = torch.ops.aten.view.default(view_483, _shape_param_560);  _shape_param_560 = None
        view_485: "i64[4096]" = torch.ops.aten.view.default(arg395_1, [-1]);  arg395_1 = None
        convert_element_type_489: "f32[4096, 128100]" = torch.ops.prims.convert_element_type.default(view_484, torch.float32);  view_484 = None
        amax_24: "f32[4096, 1]" = torch.ops.aten.amax.default(convert_element_type_489, [1], True)
        sub_74: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(convert_element_type_489, amax_24);  convert_element_type_489 = None
        exp_24: "f32[4096, 128100]" = torch.ops.aten.exp.default(sub_74)
        sum_25: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[4096, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_75: "f32[4096, 128100]" = torch.ops.aten.sub.Tensor(sub_74, log);  sub_74 = None
        convert_element_type_490: "bf16[4096, 128100]" = torch.ops.prims.convert_element_type.default(sub_75, torch.bfloat16);  sub_75 = None
        convert_element_type_491: "f32[4096, 128100]" = torch.ops.prims.convert_element_type.default(convert_element_type_490, torch.float32);  convert_element_type_490 = None
        ne: "b8[4096]" = torch.ops.aten.ne.Scalar(view_485, -100)
        full_4: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[4096]" = torch.ops.aten.where.self(ne, view_485, full_4);  view_485 = full_4 = None
        unsqueeze: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather: "f32[4096, 1]" = torch.ops.aten.gather.default(convert_element_type_491, 1, unsqueeze);  convert_element_type_491 = unsqueeze = None
        squeeze: "f32[4096]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[4096]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[4096]" = torch.ops.aten.where.self(ne, neg, full_5);  neg = full_5 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_492: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_48: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_492);  sum_27 = None
        permute_266: "bf16[128100, 1536]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        permute_267: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        div_49: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_48, 1536);  rsqrt_48 = None
        permute_268: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        permute_269: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        div_50: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_47, 1536);  rsqrt_47 = None
        permute_270: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        permute_271: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_468, [0, 2, 1]);  convert_element_type_468 = None
        permute_272: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
        permute_273: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_463, [0, 2, 1]);  view_463 = None
        permute_274: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_46, [0, 2, 1]);  div_46 = None
        permute_275: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
        permute_276: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        permute_277: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        div_51: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_46, 1536);  rsqrt_46 = None
        permute_278: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        permute_279: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        div_52: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_45, 1536);  rsqrt_45 = None
        permute_280: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        permute_281: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_448, [0, 2, 1]);  convert_element_type_448 = None
        permute_282: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
        permute_283: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_443, [0, 2, 1]);  view_443 = None
        permute_284: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_44, [0, 2, 1]);  div_44 = None
        permute_285: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        permute_286: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        permute_287: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        div_53: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_44, 1536);  rsqrt_44 = None
        permute_288: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        permute_289: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        div_54: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 1536);  rsqrt_43 = None
        permute_290: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        permute_291: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_428, [0, 2, 1]);  convert_element_type_428 = None
        permute_292: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_429, [0, 2, 1]);  view_429 = None
        permute_293: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_423, [0, 2, 1]);  view_423 = None
        permute_294: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_42, [0, 2, 1]);  div_42 = None
        permute_295: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        permute_296: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        permute_297: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        div_55: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 1536);  rsqrt_42 = None
        permute_298: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        permute_299: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        div_56: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 1536);  rsqrt_41 = None
        permute_300: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        permute_301: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_408, [0, 2, 1]);  convert_element_type_408 = None
        permute_302: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
        permute_303: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
        permute_304: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_40, [0, 2, 1]);  div_40 = None
        permute_305: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        permute_306: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        permute_307: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        div_57: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 1536);  rsqrt_40 = None
        permute_308: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        permute_309: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        div_58: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1536);  rsqrt_39 = None
        permute_310: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        permute_311: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_388, [0, 2, 1]);  convert_element_type_388 = None
        permute_312: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_389, [0, 2, 1]);  view_389 = None
        permute_313: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
        permute_314: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_38, [0, 2, 1]);  div_38 = None
        permute_315: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        permute_316: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        permute_317: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        div_59: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1536);  rsqrt_38 = None
        permute_318: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        permute_319: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        div_60: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1536);  rsqrt_37 = None
        permute_320: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        permute_321: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_368, [0, 2, 1]);  convert_element_type_368 = None
        permute_322: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_369, [0, 2, 1]);  view_369 = None
        permute_323: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
        permute_324: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_36, [0, 2, 1]);  div_36 = None
        permute_325: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        permute_326: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        permute_327: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        div_61: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 1536);  rsqrt_36 = None
        permute_328: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        permute_329: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        div_62: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 1536);  rsqrt_35 = None
        permute_330: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        permute_331: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_348, [0, 2, 1]);  convert_element_type_348 = None
        permute_332: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
        permute_333: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
        permute_334: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_34, [0, 2, 1]);  div_34 = None
        permute_335: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        permute_336: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        permute_337: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        div_63: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 1536);  rsqrt_34 = None
        permute_338: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        permute_339: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        div_64: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 1536);  rsqrt_33 = None
        permute_340: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        permute_341: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_328, [0, 2, 1]);  convert_element_type_328 = None
        permute_342: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_329, [0, 2, 1]);  view_329 = None
        permute_343: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_323, [0, 2, 1]);  view_323 = None
        permute_344: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_32, [0, 2, 1]);  div_32 = None
        permute_345: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        permute_346: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        permute_347: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        div_65: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 1536);  rsqrt_32 = None
        permute_348: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        permute_349: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        div_66: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 1536);  rsqrt_31 = None
        permute_350: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        permute_351: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_308, [0, 2, 1]);  convert_element_type_308 = None
        permute_352: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
        permute_353: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_303, [0, 2, 1]);  view_303 = None
        permute_354: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_30, [0, 2, 1]);  div_30 = None
        permute_355: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        permute_356: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        permute_357: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        div_67: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 1536);  rsqrt_30 = None
        permute_358: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        permute_359: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        div_68: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 1536);  rsqrt_29 = None
        permute_360: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        permute_361: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_288, [0, 2, 1]);  convert_element_type_288 = None
        permute_362: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_289, [0, 2, 1]);  view_289 = None
        permute_363: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
        permute_364: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_28, [0, 2, 1]);  div_28 = None
        permute_365: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        permute_366: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        permute_367: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        div_69: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 1536);  rsqrt_28 = None
        permute_368: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        permute_369: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        div_70: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 1536);  rsqrt_27 = None
        permute_370: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        permute_371: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_268, [0, 2, 1]);  convert_element_type_268 = None
        permute_372: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_269, [0, 2, 1]);  view_269 = None
        permute_373: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_263, [0, 2, 1]);  view_263 = None
        permute_374: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_26, [0, 2, 1]);  div_26 = None
        permute_375: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        permute_376: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        permute_377: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        div_71: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1536);  rsqrt_26 = None
        permute_378: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        permute_379: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        div_72: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1536);  rsqrt_25 = None
        permute_380: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        permute_381: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_248, [0, 2, 1]);  convert_element_type_248 = None
        permute_382: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
        permute_383: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_243, [0, 2, 1]);  view_243 = None
        permute_384: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_24, [0, 2, 1]);  div_24 = None
        permute_385: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        permute_386: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        permute_387: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        div_73: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1536);  rsqrt_24 = None
        permute_388: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        permute_389: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        div_74: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1536);  rsqrt_23 = None
        permute_390: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        permute_391: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_228, [0, 2, 1]);  convert_element_type_228 = None
        permute_392: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        permute_393: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
        permute_394: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_22, [0, 2, 1]);  div_22 = None
        permute_395: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        permute_396: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_397: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        div_75: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1536);  rsqrt_22 = None
        permute_398: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        permute_399: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        div_76: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1536);  rsqrt_21 = None
        permute_400: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_401: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_208, [0, 2, 1]);  convert_element_type_208 = None
        permute_402: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_209, [0, 2, 1]);  view_209 = None
        permute_403: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_203, [0, 2, 1]);  view_203 = None
        permute_404: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_20, [0, 2, 1]);  div_20 = None
        permute_405: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        permute_406: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        permute_407: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        div_77: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1536);  rsqrt_20 = None
        permute_408: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        permute_409: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        div_78: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1536);  rsqrt_19 = None
        permute_410: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_411: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_188, [0, 2, 1]);  convert_element_type_188 = None
        permute_412: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
        permute_413: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        permute_414: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_18, [0, 2, 1]);  div_18 = None
        permute_415: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        permute_416: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_417: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        div_79: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1536);  rsqrt_18 = None
        permute_418: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        permute_419: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        div_80: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 1536);  rsqrt_17 = None
        permute_420: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_421: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_168, [0, 2, 1]);  convert_element_type_168 = None
        permute_422: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
        permute_423: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_424: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_16, [0, 2, 1]);  div_16 = None
        permute_425: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        permute_426: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        permute_427: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        div_81: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 1536);  rsqrt_16 = None
        permute_428: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_429: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        div_82: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 1536);  rsqrt_15 = None
        permute_430: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_431: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_148, [0, 2, 1]);  convert_element_type_148 = None
        permute_432: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_149, [0, 2, 1]);  view_149 = None
        permute_433: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
        permute_434: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_14, [0, 2, 1]);  div_14 = None
        permute_435: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_436: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        permute_437: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        div_83: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 1536);  rsqrt_14 = None
        permute_438: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_439: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        div_84: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 1536);  rsqrt_13 = None
        permute_440: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_441: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_128, [0, 2, 1]);  convert_element_type_128 = None
        permute_442: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
        permute_443: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_444: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_12, [0, 2, 1]);  div_12 = None
        permute_445: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        permute_446: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        permute_447: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        div_85: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 1536);  rsqrt_12 = None
        permute_448: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_449: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        div_86: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 1536);  rsqrt_11 = None
        permute_450: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_451: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_108, [0, 2, 1]);  convert_element_type_108 = None
        permute_452: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_109, [0, 2, 1]);  view_109 = None
        permute_453: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None
        permute_454: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_10, [0, 2, 1]);  div_10 = None
        permute_455: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        permute_456: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        permute_457: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        div_87: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 1536);  rsqrt_10 = None
        permute_458: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_459: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        div_88: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 1536);  rsqrt_9 = None
        permute_460: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_461: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_88, [0, 2, 1]);  convert_element_type_88 = None
        permute_462: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
        permute_463: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
        permute_464: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_8, [0, 2, 1]);  div_8 = None
        permute_465: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        permute_466: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        permute_467: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        div_89: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 1536);  rsqrt_8 = None
        permute_468: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_469: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        div_90: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 1536);  rsqrt_7 = None
        permute_470: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_471: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_68, [0, 2, 1]);  convert_element_type_68 = None
        permute_472: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
        permute_473: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
        permute_474: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_6, [0, 2, 1]);  div_6 = None
        permute_475: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        permute_476: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        permute_477: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        div_91: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 1536);  rsqrt_6 = None
        permute_478: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_479: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        div_92: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 1536);  rsqrt_5 = None
        permute_480: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        permute_481: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_48, [0, 2, 1]);  convert_element_type_48 = None
        permute_482: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
        permute_483: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        permute_484: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_4, [0, 2, 1]);  div_4 = None
        permute_485: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        permute_486: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        permute_487: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        div_93: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 1536);  rsqrt_4 = None
        permute_488: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_489: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        div_94: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 1536);  rsqrt_3 = None
        permute_490: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        permute_491: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_28, [0, 2, 1]);  convert_element_type_28 = None
        permute_492: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_29, [0, 2, 1]);  view_29 = None
        permute_493: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_23, [0, 2, 1]);  view_23 = None
        permute_494: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div_2, [0, 2, 1]);  div_2 = None
        permute_495: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        permute_496: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        permute_497: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        div_95: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 1536);  rsqrt_2 = None
        permute_498: "bf16[1536, 6144]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        permute_499: "bf16[6144, 1536]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        div_96: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 1536);  rsqrt_1 = None
        permute_500: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        permute_501: "bf16[192, 512, 512]" = torch.ops.aten.permute.default(convert_element_type_8, [0, 2, 1]);  convert_element_type_8 = None
        permute_502: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        permute_503: "bf16[192, 64, 512]" = torch.ops.aten.permute.default(view_3, [0, 2, 1]);  view_3 = None
        permute_504: "bf16[192, 512, 64]" = torch.ops.aten.permute.default(div, [0, 2, 1]);  div = None
        permute_505: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        permute_506: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_507: "bf16[1536, 1536]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (embedding, embedding_1, getitem_1, rsqrt, gt, view, bmm, amax, sum_1, gt_1, view_14, gt_2, mul_8, view_16, addmm_4, view_18, gt_3, mul_15, view_20, where_1, amax_1, sum_2, gt_4, view_34, gt_5, mul_21, view_36, addmm_10, view_38, gt_6, mul_28, view_40, where_2, amax_2, sum_3, gt_7, view_54, gt_8, mul_34, view_56, addmm_16, view_58, gt_9, mul_41, view_60, where_3, amax_3, sum_4, gt_10, view_74, gt_11, mul_47, view_76, addmm_22, view_78, gt_12, mul_54, view_80, where_4, amax_4, sum_5, gt_13, view_94, gt_14, mul_60, view_96, addmm_28, view_98, gt_15, mul_67, view_100, where_5, amax_5, sum_6, gt_16, view_114, gt_17, mul_73, view_116, addmm_34, view_118, gt_18, mul_80, view_120, where_6, amax_6, sum_7, gt_19, view_134, gt_20, mul_86, view_136, addmm_40, view_138, gt_21, mul_93, view_140, where_7, amax_7, sum_8, gt_22, view_154, gt_23, mul_99, view_156, addmm_46, view_158, gt_24, mul_106, view_160, where_8, amax_8, sum_9, gt_25, view_174, gt_26, mul_112, view_176, addmm_52, view_178, gt_27, mul_119, view_180, where_9, amax_9, sum_10, gt_28, view_194, gt_29, mul_125, view_196, addmm_58, view_198, gt_30, mul_132, view_200, where_10, amax_10, sum_11, gt_31, view_214, gt_32, mul_138, view_216, addmm_64, view_218, gt_33, mul_145, view_220, where_11, amax_11, sum_12, gt_34, view_234, gt_35, mul_151, view_236, addmm_70, view_238, gt_36, mul_158, view_240, where_12, amax_12, sum_13, gt_37, view_254, gt_38, mul_164, view_256, addmm_76, view_258, gt_39, mul_171, view_260, where_13, amax_13, sum_14, gt_40, view_274, gt_41, mul_177, view_276, addmm_82, view_278, gt_42, mul_184, view_280, where_14, amax_14, sum_15, gt_43, view_294, gt_44, mul_190, view_296, addmm_88, view_298, gt_45, mul_197, view_300, where_15, amax_15, sum_16, gt_46, view_314, gt_47, mul_203, view_316, addmm_94, view_318, gt_48, mul_210, view_320, where_16, amax_16, sum_17, gt_49, view_334, gt_50, mul_216, view_336, addmm_100, view_338, gt_51, mul_223, view_340, where_17, amax_17, sum_18, gt_52, view_354, gt_53, mul_229, view_356, addmm_106, view_358, gt_54, mul_236, view_360, where_18, amax_18, sum_19, gt_55, view_374, gt_56, mul_242, view_376, addmm_112, view_378, gt_57, mul_249, view_380, where_19, amax_19, sum_20, gt_58, view_394, gt_59, mul_255, view_396, addmm_118, view_398, gt_60, mul_262, view_400, where_20, amax_20, sum_21, gt_61, view_414, gt_62, mul_268, view_416, addmm_124, view_418, gt_63, mul_275, view_420, where_21, amax_21, sum_22, gt_64, view_434, gt_65, mul_281, view_436, addmm_130, view_438, gt_66, mul_288, view_440, where_22, amax_22, sum_23, gt_67, view_454, gt_68, mul_294, view_456, addmm_136, view_458, gt_69, mul_301, view_460, where_23, amax_23, sum_24, gt_70, view_474, gt_71, mul_307, view_476, addmm_142, view_478, gt_72, mul_314, view_480, addmm_144, getitem_99, rsqrt_49, view_482, view_483, amax_24, log, convert_element_type_492, div_48, permute_266, permute_267, div_49, permute_268, permute_269, div_50, permute_270, permute_271, permute_272, permute_273, permute_274, permute_275, permute_276, permute_277, div_51, permute_278, permute_279, div_52, permute_280, permute_281, permute_282, permute_283, permute_284, permute_285, permute_286, permute_287, div_53, permute_288, permute_289, div_54, permute_290, permute_291, permute_292, permute_293, permute_294, permute_295, permute_296, permute_297, div_55, permute_298, permute_299, div_56, permute_300, permute_301, permute_302, permute_303, permute_304, permute_305, permute_306, permute_307, div_57, permute_308, permute_309, div_58, permute_310, permute_311, permute_312, permute_313, permute_314, permute_315, permute_316, permute_317, div_59, permute_318, permute_319, div_60, permute_320, permute_321, permute_322, permute_323, permute_324, permute_325, permute_326, permute_327, div_61, permute_328, permute_329, div_62, permute_330, permute_331, permute_332, permute_333, permute_334, permute_335, permute_336, permute_337, div_63, permute_338, permute_339, div_64, permute_340, permute_341, permute_342, permute_343, permute_344, permute_345, permute_346, permute_347, div_65, permute_348, permute_349, div_66, permute_350, permute_351, permute_352, permute_353, permute_354, permute_355, permute_356, permute_357, div_67, permute_358, permute_359, div_68, permute_360, permute_361, permute_362, permute_363, permute_364, permute_365, permute_366, permute_367, div_69, permute_368, permute_369, div_70, permute_370, permute_371, permute_372, permute_373, permute_374, permute_375, permute_376, permute_377, div_71, permute_378, permute_379, div_72, permute_380, permute_381, permute_382, permute_383, permute_384, permute_385, permute_386, permute_387, div_73, permute_388, permute_389, div_74, permute_390, permute_391, permute_392, permute_393, permute_394, permute_395, permute_396, permute_397, div_75, permute_398, permute_399, div_76, permute_400, permute_401, permute_402, permute_403, permute_404, permute_405, permute_406, permute_407, div_77, permute_408, permute_409, div_78, permute_410, permute_411, permute_412, permute_413, permute_414, permute_415, permute_416, permute_417, div_79, permute_418, permute_419, div_80, permute_420, permute_421, permute_422, permute_423, permute_424, permute_425, permute_426, permute_427, div_81, permute_428, permute_429, div_82, permute_430, permute_431, permute_432, permute_433, permute_434, permute_435, permute_436, permute_437, div_83, permute_438, permute_439, div_84, permute_440, permute_441, permute_442, permute_443, permute_444, permute_445, permute_446, permute_447, div_85, permute_448, permute_449, div_86, permute_450, permute_451, permute_452, permute_453, permute_454, permute_455, permute_456, permute_457, div_87, permute_458, permute_459, div_88, permute_460, permute_461, permute_462, permute_463, permute_464, permute_465, permute_466, permute_467, div_89, permute_468, permute_469, div_90, permute_470, permute_471, permute_472, permute_473, permute_474, permute_475, permute_476, permute_477, div_91, permute_478, permute_479, div_92, permute_480, permute_481, permute_482, permute_483, permute_484, permute_485, permute_486, permute_487, div_93, permute_488, permute_489, div_94, permute_490, permute_491, permute_492, permute_493, permute_494, permute_495, permute_496, permute_497, div_95, permute_498, permute_499, div_96, permute_500, permute_501, permute_502, permute_503, permute_504, permute_505, permute_506, permute_507)



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
