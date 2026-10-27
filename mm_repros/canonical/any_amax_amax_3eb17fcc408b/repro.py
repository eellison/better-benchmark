"""
Standalone repro captured via capture_hook.
Label: XLNetLMHeadModel
Pattern hash: 3eb17fcc408b
Shape hash: 73d3140e
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
    def forward(self, arg0_1: "i64[16, 512]", arg1_1: "bf16[32000, 1024]", arg2_1: "bf16[1024, 16, 64]", arg3_1: "bf16[16, 64]", arg4_1: "bf16[1024, 16, 64]", arg5_1: "bf16[16, 64]", arg6_1: "bf16[1024, 16, 64]", arg7_1: "bf16[1024, 16, 64]", arg8_1: "bf16[1024, 16, 64]", arg9_1: "bf16[1024]", arg10_1: "bf16[1024]", arg11_1: "bf16[4096, 1024]", arg12_1: "bf16[4096]", arg13_1: "bf16[1024, 4096]", arg14_1: "bf16[1024]", arg15_1: "bf16[1024]", arg16_1: "bf16[1024]", arg17_1: "bf16[1024, 16, 64]", arg18_1: "bf16[16, 64]", arg19_1: "bf16[1024, 16, 64]", arg20_1: "bf16[16, 64]", arg21_1: "bf16[1024, 16, 64]", arg22_1: "bf16[1024, 16, 64]", arg23_1: "bf16[1024, 16, 64]", arg24_1: "bf16[1024]", arg25_1: "bf16[1024]", arg26_1: "bf16[4096, 1024]", arg27_1: "bf16[4096]", arg28_1: "bf16[1024, 4096]", arg29_1: "bf16[1024]", arg30_1: "bf16[1024]", arg31_1: "bf16[1024]", arg32_1: "bf16[1024, 16, 64]", arg33_1: "bf16[16, 64]", arg34_1: "bf16[1024, 16, 64]", arg35_1: "bf16[16, 64]", arg36_1: "bf16[1024, 16, 64]", arg37_1: "bf16[1024, 16, 64]", arg38_1: "bf16[1024, 16, 64]", arg39_1: "bf16[1024]", arg40_1: "bf16[1024]", arg41_1: "bf16[4096, 1024]", arg42_1: "bf16[4096]", arg43_1: "bf16[1024, 4096]", arg44_1: "bf16[1024]", arg45_1: "bf16[1024]", arg46_1: "bf16[1024]", arg47_1: "bf16[1024, 16, 64]", arg48_1: "bf16[16, 64]", arg49_1: "bf16[1024, 16, 64]", arg50_1: "bf16[16, 64]", arg51_1: "bf16[1024, 16, 64]", arg52_1: "bf16[1024, 16, 64]", arg53_1: "bf16[1024, 16, 64]", arg54_1: "bf16[1024]", arg55_1: "bf16[1024]", arg56_1: "bf16[4096, 1024]", arg57_1: "bf16[4096]", arg58_1: "bf16[1024, 4096]", arg59_1: "bf16[1024]", arg60_1: "bf16[1024]", arg61_1: "bf16[1024]", arg62_1: "bf16[1024, 16, 64]", arg63_1: "bf16[16, 64]", arg64_1: "bf16[1024, 16, 64]", arg65_1: "bf16[16, 64]", arg66_1: "bf16[1024, 16, 64]", arg67_1: "bf16[1024, 16, 64]", arg68_1: "bf16[1024, 16, 64]", arg69_1: "bf16[1024]", arg70_1: "bf16[1024]", arg71_1: "bf16[4096, 1024]", arg72_1: "bf16[4096]", arg73_1: "bf16[1024, 4096]", arg74_1: "bf16[1024]", arg75_1: "bf16[1024]", arg76_1: "bf16[1024]", arg77_1: "bf16[1024, 16, 64]", arg78_1: "bf16[16, 64]", arg79_1: "bf16[1024, 16, 64]", arg80_1: "bf16[16, 64]", arg81_1: "bf16[1024, 16, 64]", arg82_1: "bf16[1024, 16, 64]", arg83_1: "bf16[1024, 16, 64]", arg84_1: "bf16[1024]", arg85_1: "bf16[1024]", arg86_1: "bf16[4096, 1024]", arg87_1: "bf16[4096]", arg88_1: "bf16[1024, 4096]", arg89_1: "bf16[1024]", arg90_1: "bf16[1024]", arg91_1: "bf16[1024]", arg92_1: "bf16[1024, 16, 64]", arg93_1: "bf16[16, 64]", arg94_1: "bf16[1024, 16, 64]", arg95_1: "bf16[16, 64]", arg96_1: "bf16[1024, 16, 64]", arg97_1: "bf16[1024, 16, 64]", arg98_1: "bf16[1024, 16, 64]", arg99_1: "bf16[1024]", arg100_1: "bf16[1024]", arg101_1: "bf16[4096, 1024]", arg102_1: "bf16[4096]", arg103_1: "bf16[1024, 4096]", arg104_1: "bf16[1024]", arg105_1: "bf16[1024]", arg106_1: "bf16[1024]", arg107_1: "bf16[1024, 16, 64]", arg108_1: "bf16[16, 64]", arg109_1: "bf16[1024, 16, 64]", arg110_1: "bf16[16, 64]", arg111_1: "bf16[1024, 16, 64]", arg112_1: "bf16[1024, 16, 64]", arg113_1: "bf16[1024, 16, 64]", arg114_1: "bf16[1024]", arg115_1: "bf16[1024]", arg116_1: "bf16[4096, 1024]", arg117_1: "bf16[4096]", arg118_1: "bf16[1024, 4096]", arg119_1: "bf16[1024]", arg120_1: "bf16[1024]", arg121_1: "bf16[1024]", arg122_1: "bf16[1024, 16, 64]", arg123_1: "bf16[16, 64]", arg124_1: "bf16[1024, 16, 64]", arg125_1: "bf16[16, 64]", arg126_1: "bf16[1024, 16, 64]", arg127_1: "bf16[1024, 16, 64]", arg128_1: "bf16[1024, 16, 64]", arg129_1: "bf16[1024]", arg130_1: "bf16[1024]", arg131_1: "bf16[4096, 1024]", arg132_1: "bf16[4096]", arg133_1: "bf16[1024, 4096]", arg134_1: "bf16[1024]", arg135_1: "bf16[1024]", arg136_1: "bf16[1024]", arg137_1: "bf16[1024, 16, 64]", arg138_1: "bf16[16, 64]", arg139_1: "bf16[1024, 16, 64]", arg140_1: "bf16[16, 64]", arg141_1: "bf16[1024, 16, 64]", arg142_1: "bf16[1024, 16, 64]", arg143_1: "bf16[1024, 16, 64]", arg144_1: "bf16[1024]", arg145_1: "bf16[1024]", arg146_1: "bf16[4096, 1024]", arg147_1: "bf16[4096]", arg148_1: "bf16[1024, 4096]", arg149_1: "bf16[1024]", arg150_1: "bf16[1024]", arg151_1: "bf16[1024]", arg152_1: "bf16[1024, 16, 64]", arg153_1: "bf16[16, 64]", arg154_1: "bf16[1024, 16, 64]", arg155_1: "bf16[16, 64]", arg156_1: "bf16[1024, 16, 64]", arg157_1: "bf16[1024, 16, 64]", arg158_1: "bf16[1024, 16, 64]", arg159_1: "bf16[1024]", arg160_1: "bf16[1024]", arg161_1: "bf16[4096, 1024]", arg162_1: "bf16[4096]", arg163_1: "bf16[1024, 4096]", arg164_1: "bf16[1024]", arg165_1: "bf16[1024]", arg166_1: "bf16[1024]", arg167_1: "bf16[1024, 16, 64]", arg168_1: "bf16[16, 64]", arg169_1: "bf16[1024, 16, 64]", arg170_1: "bf16[16, 64]", arg171_1: "bf16[1024, 16, 64]", arg172_1: "bf16[1024, 16, 64]", arg173_1: "bf16[1024, 16, 64]", arg174_1: "bf16[1024]", arg175_1: "bf16[1024]", arg176_1: "bf16[4096, 1024]", arg177_1: "bf16[4096]", arg178_1: "bf16[1024, 4096]", arg179_1: "bf16[1024]", arg180_1: "bf16[1024]", arg181_1: "bf16[1024]", arg182_1: "bf16[1024, 16, 64]", arg183_1: "bf16[16, 64]", arg184_1: "bf16[1024, 16, 64]", arg185_1: "bf16[16, 64]", arg186_1: "bf16[1024, 16, 64]", arg187_1: "bf16[1024, 16, 64]", arg188_1: "bf16[1024, 16, 64]", arg189_1: "bf16[1024]", arg190_1: "bf16[1024]", arg191_1: "bf16[4096, 1024]", arg192_1: "bf16[4096]", arg193_1: "bf16[1024, 4096]", arg194_1: "bf16[1024]", arg195_1: "bf16[1024]", arg196_1: "bf16[1024]", arg197_1: "bf16[1024, 16, 64]", arg198_1: "bf16[16, 64]", arg199_1: "bf16[1024, 16, 64]", arg200_1: "bf16[16, 64]", arg201_1: "bf16[1024, 16, 64]", arg202_1: "bf16[1024, 16, 64]", arg203_1: "bf16[1024, 16, 64]", arg204_1: "bf16[1024]", arg205_1: "bf16[1024]", arg206_1: "bf16[4096, 1024]", arg207_1: "bf16[4096]", arg208_1: "bf16[1024, 4096]", arg209_1: "bf16[1024]", arg210_1: "bf16[1024]", arg211_1: "bf16[1024]", arg212_1: "bf16[1024, 16, 64]", arg213_1: "bf16[16, 64]", arg214_1: "bf16[1024, 16, 64]", arg215_1: "bf16[16, 64]", arg216_1: "bf16[1024, 16, 64]", arg217_1: "bf16[1024, 16, 64]", arg218_1: "bf16[1024, 16, 64]", arg219_1: "bf16[1024]", arg220_1: "bf16[1024]", arg221_1: "bf16[4096, 1024]", arg222_1: "bf16[4096]", arg223_1: "bf16[1024, 4096]", arg224_1: "bf16[1024]", arg225_1: "bf16[1024]", arg226_1: "bf16[1024]", arg227_1: "bf16[1024, 16, 64]", arg228_1: "bf16[16, 64]", arg229_1: "bf16[1024, 16, 64]", arg230_1: "bf16[16, 64]", arg231_1: "bf16[1024, 16, 64]", arg232_1: "bf16[1024, 16, 64]", arg233_1: "bf16[1024, 16, 64]", arg234_1: "bf16[1024]", arg235_1: "bf16[1024]", arg236_1: "bf16[4096, 1024]", arg237_1: "bf16[4096]", arg238_1: "bf16[1024, 4096]", arg239_1: "bf16[1024]", arg240_1: "bf16[1024]", arg241_1: "bf16[1024]", arg242_1: "bf16[1024, 16, 64]", arg243_1: "bf16[16, 64]", arg244_1: "bf16[1024, 16, 64]", arg245_1: "bf16[16, 64]", arg246_1: "bf16[1024, 16, 64]", arg247_1: "bf16[1024, 16, 64]", arg248_1: "bf16[1024, 16, 64]", arg249_1: "bf16[1024]", arg250_1: "bf16[1024]", arg251_1: "bf16[4096, 1024]", arg252_1: "bf16[4096]", arg253_1: "bf16[1024, 4096]", arg254_1: "bf16[1024]", arg255_1: "bf16[1024]", arg256_1: "bf16[1024]", arg257_1: "bf16[1024, 16, 64]", arg258_1: "bf16[16, 64]", arg259_1: "bf16[1024, 16, 64]", arg260_1: "bf16[16, 64]", arg261_1: "bf16[1024, 16, 64]", arg262_1: "bf16[1024, 16, 64]", arg263_1: "bf16[1024, 16, 64]", arg264_1: "bf16[1024]", arg265_1: "bf16[1024]", arg266_1: "bf16[4096, 1024]", arg267_1: "bf16[4096]", arg268_1: "bf16[1024, 4096]", arg269_1: "bf16[1024]", arg270_1: "bf16[1024]", arg271_1: "bf16[1024]", arg272_1: "bf16[1024, 16, 64]", arg273_1: "bf16[16, 64]", arg274_1: "bf16[1024, 16, 64]", arg275_1: "bf16[16, 64]", arg276_1: "bf16[1024, 16, 64]", arg277_1: "bf16[1024, 16, 64]", arg278_1: "bf16[1024, 16, 64]", arg279_1: "bf16[1024]", arg280_1: "bf16[1024]", arg281_1: "bf16[4096, 1024]", arg282_1: "bf16[4096]", arg283_1: "bf16[1024, 4096]", arg284_1: "bf16[1024]", arg285_1: "bf16[1024]", arg286_1: "bf16[1024]", arg287_1: "bf16[1024, 16, 64]", arg288_1: "bf16[16, 64]", arg289_1: "bf16[1024, 16, 64]", arg290_1: "bf16[16, 64]", arg291_1: "bf16[1024, 16, 64]", arg292_1: "bf16[1024, 16, 64]", arg293_1: "bf16[1024, 16, 64]", arg294_1: "bf16[1024]", arg295_1: "bf16[1024]", arg296_1: "bf16[4096, 1024]", arg297_1: "bf16[4096]", arg298_1: "bf16[1024, 4096]", arg299_1: "bf16[1024]", arg300_1: "bf16[1024]", arg301_1: "bf16[1024]", arg302_1: "bf16[1024, 16, 64]", arg303_1: "bf16[16, 64]", arg304_1: "bf16[1024, 16, 64]", arg305_1: "bf16[16, 64]", arg306_1: "bf16[1024, 16, 64]", arg307_1: "bf16[1024, 16, 64]", arg308_1: "bf16[1024, 16, 64]", arg309_1: "bf16[1024]", arg310_1: "bf16[1024]", arg311_1: "bf16[4096, 1024]", arg312_1: "bf16[4096]", arg313_1: "bf16[1024, 4096]", arg314_1: "bf16[1024]", arg315_1: "bf16[1024]", arg316_1: "bf16[1024]", arg317_1: "bf16[1024, 16, 64]", arg318_1: "bf16[16, 64]", arg319_1: "bf16[1024, 16, 64]", arg320_1: "bf16[16, 64]", arg321_1: "bf16[1024, 16, 64]", arg322_1: "bf16[1024, 16, 64]", arg323_1: "bf16[1024, 16, 64]", arg324_1: "bf16[1024]", arg325_1: "bf16[1024]", arg326_1: "bf16[4096, 1024]", arg327_1: "bf16[4096]", arg328_1: "bf16[1024, 4096]", arg329_1: "bf16[1024]", arg330_1: "bf16[1024]", arg331_1: "bf16[1024]", arg332_1: "bf16[1024, 16, 64]", arg333_1: "bf16[16, 64]", arg334_1: "bf16[1024, 16, 64]", arg335_1: "bf16[16, 64]", arg336_1: "bf16[1024, 16, 64]", arg337_1: "bf16[1024, 16, 64]", arg338_1: "bf16[1024, 16, 64]", arg339_1: "bf16[1024]", arg340_1: "bf16[1024]", arg341_1: "bf16[4096, 1024]", arg342_1: "bf16[4096]", arg343_1: "bf16[1024, 4096]", arg344_1: "bf16[1024]", arg345_1: "bf16[1024]", arg346_1: "bf16[1024]", arg347_1: "bf16[1024, 16, 64]", arg348_1: "bf16[16, 64]", arg349_1: "bf16[1024, 16, 64]", arg350_1: "bf16[16, 64]", arg351_1: "bf16[1024, 16, 64]", arg352_1: "bf16[1024, 16, 64]", arg353_1: "bf16[1024, 16, 64]", arg354_1: "bf16[1024]", arg355_1: "bf16[1024]", arg356_1: "bf16[4096, 1024]", arg357_1: "bf16[4096]", arg358_1: "bf16[1024, 4096]", arg359_1: "bf16[1024]", arg360_1: "i64[16, 512]", arg361_1: "bf16[1024]", arg362_1: "bf16[1024]", arg363_1: "bf16[32000]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833, _shape_param_834, _shape_param_835, _shape_param_836, _shape_param_837, _shape_param_838, _shape_param_839, _shape_param_840, _shape_param_841, _shape_param_842, _shape_param_843, _shape_param_844, _shape_param_845, _shape_param_846, _shape_param_847, _shape_param_848, _shape_param_849, _shape_param_850, _shape_param_851, _shape_param_852, _shape_param_853, _shape_param_854, _shape_param_855, _shape_param_856, _shape_param_857, _shape_param_858, _shape_param_859, _shape_param_860, _shape_param_861, _shape_param_862, _shape_param_863, _shape_param_864, _shape_param_865, _shape_param_866, _shape_param_867, _shape_param_868, _shape_param_869, _shape_param_870, _shape_param_871, _shape_param_872, _shape_param_873, _shape_param_874, _shape_param_875, _shape_param_876, _shape_param_877, _shape_param_878, _shape_param_879, _shape_param_880, _shape_param_881, _shape_param_882, _shape_param_883, _shape_param_884, _shape_param_885, _shape_param_886, _shape_param_887, _shape_param_888, _shape_param_889, _shape_param_890, _shape_param_891, _shape_param_892, _shape_param_893, _shape_param_894, _shape_param_895, _shape_param_896, _shape_param_897, _shape_param_898, _shape_param_899, _shape_param_900, _shape_param_901, _shape_param_902, _shape_param_903, _shape_param_904, _shape_param_905, _shape_param_906, _shape_param_907, _shape_param_908, _shape_param_909, _shape_param_910, _shape_param_911, _shape_param_912, _shape_param_913, _shape_param_914, _shape_param_915):
        # No stacktrace found for following nodes
        permute: "i64[512, 16]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        clone: "i64[512, 16]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        embedding: "bf16[512, 16, 1024]" = torch.ops.aten.embedding.default(arg1_1, clone);  clone = None
        unsqueeze: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_1: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 4);  unsqueeze = None
        view: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_1, _shape_param_0);  unsqueeze_1 = _shape_param_0 = None
        squeeze: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view, 0);  view = None
        unsqueeze_2: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg2_1, 3);  arg2_1 = None
        unsqueeze_3: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 4);  unsqueeze_2 = None
        view_1: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_3, _shape_param_1);  unsqueeze_3 = _shape_param_1 = None
        squeeze_1: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_1, 0);  view_1 = None
        mm: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze, squeeze_1);  squeeze = squeeze_1 = None
        unsqueeze_4: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm, 0);  mm = None
        view_2: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_4, _shape_param_2);  unsqueeze_4 = _shape_param_2 = None
        permute_1: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_1, _shape_param_3);  permute_1 = _shape_param_3 = None
        add: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg3_1);  arg3_1 = None
        unsqueeze_5: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add, 4);  add = None
        permute_2: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_5, [1, 2, 0, 4, 3]);  unsqueeze_5 = None
        permute_3: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_2, [0, 1, 2, 4, 3]);  permute_2 = None
        view_4: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_3, _shape_param_4);  permute_3 = _shape_param_4 = None
        unsqueeze_6: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_7: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 4);  unsqueeze_6 = None
        view_5: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_7, _shape_param_5);  unsqueeze_7 = _shape_param_5 = None
        squeeze_2: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_5, 0);  view_5 = None
        unsqueeze_8: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, 3);  arg4_1 = None
        unsqueeze_9: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 4);  unsqueeze_8 = None
        view_6: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_9, _shape_param_6);  unsqueeze_9 = _shape_param_6 = None
        squeeze_3: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_6, 0);  view_6 = None
        mm_1: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_2, squeeze_3);  squeeze_2 = squeeze_3 = None
        unsqueeze_10: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_1, 0);  mm_1 = None
        view_7: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_10, _shape_param_7);  unsqueeze_10 = _shape_param_7 = None
        permute_4: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_7, [0, 1, 3, 4, 2]);  view_7 = None
        view_8: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_4, _shape_param_8);  permute_4 = _shape_param_8 = None
        unsqueeze_11: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_8, 4);  view_8 = None
        permute_5: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_11, [1, 2, 4, 0, 3]);  unsqueeze_11 = None
        permute_6: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_5, [0, 1, 4, 3, 2]);  permute_5 = None
        view_9: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_6, _shape_param_9);  permute_6 = _shape_param_9 = None
        bmm: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_4, view_9);  view_4 = view_9 = None
        view_10: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm, _shape_param_10);  bmm = _shape_param_10 = None
        permute_7: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_10, [0, 1, 2, 4, 3]);  view_10 = None
        view_11: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_7, _shape_param_11);  permute_7 = _shape_param_11 = None
        add_1: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_3, arg5_1);  view_3 = arg5_1 = None
        unsqueeze_12: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_1, 4);  add_1 = None
        permute_8: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_12, [1, 2, 0, 4, 3]);  unsqueeze_12 = None
        permute_9: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_8, [0, 1, 2, 4, 3]);  permute_8 = None
        view_12: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_9, _shape_param_12);  permute_9 = _shape_param_12 = None
        iota: "i64[1024]" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[1024]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        unsqueeze_13: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, 1);  convert_element_type = None
        permute_10: "f32[1024, 1]" = torch.ops.aten.permute.default(unsqueeze_13, [0, 1]);  unsqueeze_13 = None
        iota_1: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_1: "f32[512]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        div: "f32[512]" = torch.ops.aten.div.Tensor(convert_element_type_1, 1024);  convert_element_type_1 = None
        pow_1: "f32[512]" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512]" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_14: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul, 1);  mul = None
        permute_11: "f32[1, 512]" = torch.ops.aten.permute.default(unsqueeze_14, [1, 0]);  unsqueeze_14 = None
        mul_1: "f32[1024, 512]" = torch.ops.aten.mul.Tensor(permute_10, permute_11);  permute_10 = permute_11 = None
        sin: "f32[1024, 512]" = torch.ops.aten.sin.default(mul_1)
        cos: "f32[1024, 512]" = torch.ops.aten.cos.default(mul_1);  mul_1 = None
        cat: "f32[1024, 1024]" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None
        unsqueeze_15: "f32[1024, 1, 1024]" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None
        expand: "f32[1024, 16, 1024]" = torch.ops.aten.expand.default(unsqueeze_15, _shape_param_13);  unsqueeze_15 = _shape_param_13 = None
        clone_1: "f32[1024, 16, 1024]" = torch.ops.aten.clone.default(expand);  expand = None
        convert_element_type_2: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_16: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 3);  convert_element_type_2 = None
        unsqueeze_17: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, 4);  unsqueeze_16 = None
        view_13: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_17, _shape_param_14);  unsqueeze_17 = _shape_param_14 = None
        squeeze_4: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_13, 0);  view_13 = None
        unsqueeze_18: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, 3);  arg6_1 = None
        unsqueeze_19: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, 4);  unsqueeze_18 = None
        view_14: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_19, _shape_param_15);  unsqueeze_19 = _shape_param_15 = None
        squeeze_5: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_14, 0);  view_14 = None
        mm_2: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_4, squeeze_5);  squeeze_4 = squeeze_5 = None
        unsqueeze_20: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_2, 0);  mm_2 = None
        view_15: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_20, _shape_param_16);  unsqueeze_20 = _shape_param_16 = None
        permute_12: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_15, [0, 1, 3, 4, 2]);  view_15 = None
        view_16: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_12, _shape_param_17);  permute_12 = _shape_param_17 = None
        unsqueeze_21: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_16, 4);  view_16 = None
        permute_13: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_21, [1, 2, 4, 0, 3]);  unsqueeze_21 = None
        permute_14: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_13, [0, 1, 4, 3, 2]);  permute_13 = None
        view_17: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_14, _shape_param_18);  permute_14 = _shape_param_18 = None
        bmm_1: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_12, view_17);  view_12 = view_17 = None
        view_18: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_1, _shape_param_19);  bmm_1 = _shape_param_19 = None
        permute_15: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_18, [0, 1, 2, 4, 3]);  view_18 = None
        view_19: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_15, _shape_param_20);  permute_15 = _shape_param_20 = None
        view_20: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_19, _shape_param_21);  view_19 = _shape_param_21 = None
        slice_1: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_20, 2, 1, 9223372036854775807);  view_20 = None
        view_21: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_1, _shape_param_22);  slice_1 = _shape_param_22 = None
        iota_2: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_21, [None, None, None, iota_2]);  view_21 = iota_2 = None
        add_2: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_11, index);  view_11 = index = None
        add_3: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_2, 0);  add_2 = None
        mul_2: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_3, 0.125)
        convert_element_type_3: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_2, torch.float32);  mul_2 = None
        eq: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_3, convert_element_type_3)
        abs_1: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_3)
        ne: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_1, inf);  abs_1 = None
        mul_3: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq, ne);  eq = ne = None
        logical_not: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_3);  mul_3 = None
        any_1: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not, [3], True);  logical_not = None
        logical_not_1: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        convert_element_type_4: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32);  add_3 = None
        mul_4: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1);  convert_element_type_4 = None
        amax: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_4, [3], True)
        sub: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_4, amax);  mul_4 = amax = None
        mul_5: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub, 0.125);  sub = None
        amax_1: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_3, [3], True)
        sub_1: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_3, amax_1);  convert_element_type_3 = amax_1 = None
        where: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_1, mul_5, sub_1);  logical_not_1 = mul_5 = sub_1 = None
        exp: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where);  where = None
        sum_1: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div_1: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_5: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        unsqueeze_22: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 4);  convert_element_type_5 = None
        view_22: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_22, _shape_param_23);  unsqueeze_22 = _shape_param_23 = None
        unsqueeze_23: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(embedding, 3)
        unsqueeze_24: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 4);  unsqueeze_23 = None
        view_23: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_24, _shape_param_24);  unsqueeze_24 = _shape_param_24 = None
        squeeze_6: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_23, 0);  view_23 = None
        unsqueeze_25: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg7_1, 3);  arg7_1 = None
        unsqueeze_26: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_25, 4);  unsqueeze_25 = None
        view_24: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_26, _shape_param_25);  unsqueeze_26 = _shape_param_25 = None
        squeeze_7: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_24, 0);  view_24 = None
        mm_3: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_6, squeeze_7);  squeeze_6 = squeeze_7 = None
        unsqueeze_27: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_3, 0);  mm_3 = None
        view_25: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_27, _shape_param_26);  unsqueeze_27 = _shape_param_26 = None
        permute_16: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_25, [0, 1, 3, 4, 2]);  view_25 = None
        view_26: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_16, _shape_param_27);  permute_16 = _shape_param_27 = None
        unsqueeze_28: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_26, 4);  view_26 = None
        permute_17: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_28, [4, 1, 2, 3, 0]);  unsqueeze_28 = None
        permute_18: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_17, [1, 2, 4, 3, 0]);  permute_17 = None
        view_27: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_18, _shape_param_28);  permute_18 = _shape_param_28 = None
        bmm_2: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_22, view_27);  view_22 = view_27 = None
        view_28: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_2, _shape_param_29);  bmm_2 = _shape_param_29 = None
        permute_19: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_28, [2, 0, 1, 4, 3]);  view_28 = None
        view_29: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_19, _shape_param_30);  permute_19 = _shape_param_30 = None
        unsqueeze_29: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_29, 4);  view_29 = None
        permute_20: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_29, [0, 1, 4, 3, 2]);  unsqueeze_29 = None
        permute_21: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_20, [0, 1, 3, 4, 2]);  permute_20 = None
        clone_2: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_30: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_2, _shape_param_31);  clone_2 = _shape_param_31 = None
        squeeze_8: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_30, 0);  view_30 = None
        unsqueeze_30: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg8_1, 3);  arg8_1 = None
        unsqueeze_31: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 4);  unsqueeze_30 = None
        permute_22: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_31, [3, 4, 0, 2, 1]);  unsqueeze_31 = None
        permute_23: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_22, [3, 4, 2, 0, 1]);  permute_22 = None
        clone_3: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_31: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_3, _shape_param_32);  clone_3 = _shape_param_32 = None
        squeeze_9: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_31, 0);  view_31 = None
        mm_4: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_8, squeeze_9);  squeeze_8 = squeeze_9 = None
        unsqueeze_32: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_4, 0);  mm_4 = None
        view_32: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_32, _shape_param_33);  unsqueeze_32 = _shape_param_33 = None
        permute_24: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_32, [0, 1, 4, 2, 3]);  view_32 = None
        view_33: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_24, _shape_param_34);  permute_24 = _shape_param_34 = None
        add_4: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_33, embedding);  view_33 = None
        convert_element_type_6: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_4, torch.float32);  add_4 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_6, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1]" = var_mean[0]
        getitem_1: "f32[512, 16, 1]" = var_mean[1];  var_mean = None
        sub_2: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_6, getitem_1);  convert_element_type_6 = getitem_1 = None
        add_5: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        mul_6: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_7: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_6, arg9_1);  mul_6 = arg9_1 = None
        add_6: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_7, arg10_1);  mul_7 = arg10_1 = None
        convert_element_type_7: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16);  add_6 = None
        view_34: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_7, _shape_param_35);  _shape_param_35 = None
        permute_25: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        addmm: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg12_1, view_34, permute_25);  arg12_1 = view_34 = permute_25 = None
        view_35: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm, _shape_param_36);  addmm = _shape_param_36 = None
        convert_element_type_8: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        mul_8: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 0.5)
        mul_9: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 0.7071067811865476);  convert_element_type_8 = None
        erf: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
        add_7: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_10: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_8, add_7);  mul_8 = add_7 = None
        convert_element_type_9: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_10, torch.bfloat16);  mul_10 = None
        view_36: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_9, _shape_param_37);  convert_element_type_9 = _shape_param_37 = None
        permute_26: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_1: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg14_1, view_36, permute_26);  arg14_1 = view_36 = permute_26 = None
        view_37: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_1, _shape_param_38);  addmm_1 = _shape_param_38 = None
        add_8: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_37, convert_element_type_7);  view_37 = convert_element_type_7 = None
        convert_element_type_10: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_8, torch.float32);  add_8 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_10, [2], correction = 0, keepdim = True)
        getitem_2: "f32[512, 16, 1]" = var_mean_1[0]
        getitem_3: "f32[512, 16, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_3: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_10, getitem_3);  convert_element_type_10 = getitem_3 = None
        add_9: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        mul_11: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = rsqrt_1 = None
        mul_12: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_11, arg15_1);  mul_11 = arg15_1 = None
        add_10: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_12, arg16_1);  mul_12 = arg16_1 = None
        convert_element_type_11: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16);  add_10 = None
        unsqueeze_33: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_11, 3)
        unsqueeze_34: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_33, 4);  unsqueeze_33 = None
        view_38: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_34, _shape_param_39);  unsqueeze_34 = _shape_param_39 = None
        squeeze_10: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_38, 0);  view_38 = None
        unsqueeze_35: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg17_1, 3);  arg17_1 = None
        unsqueeze_36: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_35, 4);  unsqueeze_35 = None
        view_39: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_36, _shape_param_40);  unsqueeze_36 = _shape_param_40 = None
        squeeze_11: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_39, 0);  view_39 = None
        mm_5: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_10, squeeze_11);  squeeze_10 = squeeze_11 = None
        unsqueeze_37: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_5, 0);  mm_5 = None
        view_40: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_37, _shape_param_41);  unsqueeze_37 = _shape_param_41 = None
        permute_27: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_40, [0, 1, 3, 4, 2]);  view_40 = None
        view_41: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_27, _shape_param_42);  permute_27 = _shape_param_42 = None
        add_11: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_41, arg18_1);  arg18_1 = None
        unsqueeze_38: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_11, 4);  add_11 = None
        permute_28: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_38, [1, 2, 0, 4, 3]);  unsqueeze_38 = None
        permute_29: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_28, [0, 1, 2, 4, 3]);  permute_28 = None
        view_42: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_29, _shape_param_43);  permute_29 = _shape_param_43 = None
        unsqueeze_39: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_11, 3)
        unsqueeze_40: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_39, 4);  unsqueeze_39 = None
        view_43: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_40, _shape_param_44);  unsqueeze_40 = _shape_param_44 = None
        squeeze_12: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_43, 0);  view_43 = None
        unsqueeze_41: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, 3);  arg19_1 = None
        unsqueeze_42: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_41, 4);  unsqueeze_41 = None
        view_44: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_42, _shape_param_45);  unsqueeze_42 = _shape_param_45 = None
        squeeze_13: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_44, 0);  view_44 = None
        mm_6: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_12, squeeze_13);  squeeze_12 = squeeze_13 = None
        unsqueeze_43: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_6, 0);  mm_6 = None
        view_45: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_43, _shape_param_46);  unsqueeze_43 = _shape_param_46 = None
        permute_30: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_45, [0, 1, 3, 4, 2]);  view_45 = None
        view_46: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_30, _shape_param_47);  permute_30 = _shape_param_47 = None
        unsqueeze_44: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_46, 4);  view_46 = None
        permute_31: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_44, [1, 2, 4, 0, 3]);  unsqueeze_44 = None
        permute_32: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_31, [0, 1, 4, 3, 2]);  permute_31 = None
        view_47: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_32, _shape_param_48);  permute_32 = _shape_param_48 = None
        bmm_3: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_42, view_47);  view_42 = view_47 = None
        view_48: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_3, _shape_param_49);  bmm_3 = _shape_param_49 = None
        permute_33: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_48, [0, 1, 2, 4, 3]);  view_48 = None
        view_49: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_33, _shape_param_50);  permute_33 = _shape_param_50 = None
        add_12: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_41, arg20_1);  view_41 = arg20_1 = None
        unsqueeze_45: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_12, 4);  add_12 = None
        permute_34: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_45, [1, 2, 0, 4, 3]);  unsqueeze_45 = None
        permute_35: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_34, [0, 1, 2, 4, 3]);  permute_34 = None
        view_50: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_35, _shape_param_51);  permute_35 = _shape_param_51 = None
        convert_element_type_12: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_46: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, 3);  convert_element_type_12 = None
        unsqueeze_47: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, 4);  unsqueeze_46 = None
        view_51: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_47, _shape_param_52);  unsqueeze_47 = _shape_param_52 = None
        squeeze_14: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_51, 0);  view_51 = None
        unsqueeze_48: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg21_1, 3);  arg21_1 = None
        unsqueeze_49: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, 4);  unsqueeze_48 = None
        view_52: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_49, _shape_param_53);  unsqueeze_49 = _shape_param_53 = None
        squeeze_15: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_52, 0);  view_52 = None
        mm_7: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_14, squeeze_15);  squeeze_14 = squeeze_15 = None
        unsqueeze_50: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_7, 0);  mm_7 = None
        view_53: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_50, _shape_param_54);  unsqueeze_50 = _shape_param_54 = None
        permute_36: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_53, [0, 1, 3, 4, 2]);  view_53 = None
        view_54: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_36, _shape_param_55);  permute_36 = _shape_param_55 = None
        unsqueeze_51: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_54, 4);  view_54 = None
        permute_37: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_51, [1, 2, 4, 0, 3]);  unsqueeze_51 = None
        permute_38: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_37, [0, 1, 4, 3, 2]);  permute_37 = None
        view_55: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_38, _shape_param_56);  permute_38 = _shape_param_56 = None
        bmm_4: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_50, view_55);  view_50 = view_55 = None
        view_56: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_4, _shape_param_57);  bmm_4 = _shape_param_57 = None
        permute_39: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_56, [0, 1, 2, 4, 3]);  view_56 = None
        view_57: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_39, _shape_param_58);  permute_39 = _shape_param_58 = None
        view_58: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_57, _shape_param_59);  view_57 = _shape_param_59 = None
        slice_2: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_58, 2, 1, 9223372036854775807);  view_58 = None
        view_59: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_2, _shape_param_60);  slice_2 = _shape_param_60 = None
        iota_3: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_1: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_59, [None, None, None, iota_3]);  view_59 = iota_3 = None
        add_13: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_49, index_1);  view_49 = index_1 = None
        add_14: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_13, 0);  add_13 = None
        mul_13: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_14, 0.125)
        convert_element_type_13: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_13, torch.float32);  mul_13 = None
        eq_1: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_13, convert_element_type_13)
        abs_2: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_13)
        ne_1: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_2, inf);  abs_2 = None
        mul_14: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_1, ne_1);  eq_1 = ne_1 = None
        logical_not_2: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_14);  mul_14 = None
        any_2: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_2, [3], True);  logical_not_2 = None
        logical_not_3: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        convert_element_type_14: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32);  add_14 = None
        mul_15: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1);  convert_element_type_14 = None
        amax_2: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_15, [3], True)
        sub_4: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_15, amax_2);  mul_15 = amax_2 = None
        mul_16: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_4, 0.125);  sub_4 = None
        amax_3: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_13, [3], True)
        sub_5: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_13, amax_3);  convert_element_type_13 = amax_3 = None
        where_1: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_3, mul_16, sub_5);  logical_not_3 = mul_16 = sub_5 = None
        exp_1: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_1);  where_1 = None
        sum_2: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [3], True)
        div_2: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_15: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        unsqueeze_52: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_15, 4);  convert_element_type_15 = None
        view_60: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_52, _shape_param_61);  unsqueeze_52 = _shape_param_61 = None
        unsqueeze_53: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_11, 3)
        unsqueeze_54: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 4);  unsqueeze_53 = None
        view_61: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_54, _shape_param_62);  unsqueeze_54 = _shape_param_62 = None
        squeeze_16: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_61, 0);  view_61 = None
        unsqueeze_55: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg22_1, 3);  arg22_1 = None
        unsqueeze_56: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
        view_62: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_56, _shape_param_63);  unsqueeze_56 = _shape_param_63 = None
        squeeze_17: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_62, 0);  view_62 = None
        mm_8: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_16, squeeze_17);  squeeze_16 = squeeze_17 = None
        unsqueeze_57: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_8, 0);  mm_8 = None
        view_63: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_57, _shape_param_64);  unsqueeze_57 = _shape_param_64 = None
        permute_40: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_63, [0, 1, 3, 4, 2]);  view_63 = None
        view_64: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_40, _shape_param_65);  permute_40 = _shape_param_65 = None
        unsqueeze_58: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_64, 4);  view_64 = None
        permute_41: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_58, [4, 1, 2, 3, 0]);  unsqueeze_58 = None
        permute_42: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_41, [1, 2, 4, 3, 0]);  permute_41 = None
        view_65: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_42, _shape_param_66);  permute_42 = _shape_param_66 = None
        bmm_5: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_60, view_65);  view_60 = view_65 = None
        view_66: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_5, _shape_param_67);  bmm_5 = _shape_param_67 = None
        permute_43: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_66, [2, 0, 1, 4, 3]);  view_66 = None
        view_67: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_43, _shape_param_68);  permute_43 = _shape_param_68 = None
        unsqueeze_59: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_67, 4);  view_67 = None
        permute_44: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_59, [0, 1, 4, 3, 2]);  unsqueeze_59 = None
        permute_45: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_44, [0, 1, 3, 4, 2]);  permute_44 = None
        clone_4: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
        view_68: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_4, _shape_param_69);  clone_4 = _shape_param_69 = None
        squeeze_18: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_68, 0);  view_68 = None
        unsqueeze_60: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg23_1, 3);  arg23_1 = None
        unsqueeze_61: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, 4);  unsqueeze_60 = None
        permute_46: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_61, [3, 4, 0, 2, 1]);  unsqueeze_61 = None
        permute_47: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_46, [3, 4, 2, 0, 1]);  permute_46 = None
        clone_5: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_69: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_5, _shape_param_70);  clone_5 = _shape_param_70 = None
        squeeze_19: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_69, 0);  view_69 = None
        mm_9: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_18, squeeze_19);  squeeze_18 = squeeze_19 = None
        unsqueeze_62: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_9, 0);  mm_9 = None
        view_70: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_62, _shape_param_71);  unsqueeze_62 = _shape_param_71 = None
        permute_48: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_70, [0, 1, 4, 2, 3]);  view_70 = None
        view_71: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_48, _shape_param_72);  permute_48 = _shape_param_72 = None
        add_15: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_71, convert_element_type_11);  view_71 = None
        convert_element_type_16: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_15, torch.float32);  add_15 = None
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_16, [2], correction = 0, keepdim = True)
        getitem_4: "f32[512, 16, 1]" = var_mean_2[0]
        getitem_5: "f32[512, 16, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_6: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_16, getitem_5);  convert_element_type_16 = getitem_5 = None
        add_16: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        mul_17: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_2);  sub_6 = rsqrt_2 = None
        mul_18: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg24_1);  mul_17 = arg24_1 = None
        add_17: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg25_1);  mul_18 = arg25_1 = None
        convert_element_type_17: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16);  add_17 = None
        view_72: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_73);  _shape_param_73 = None
        permute_49: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_2: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg27_1, view_72, permute_49);  arg27_1 = view_72 = permute_49 = None
        view_73: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_2, _shape_param_74);  addmm_2 = _shape_param_74 = None
        convert_element_type_18: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        mul_19: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.5)
        mul_20: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.7071067811865476);  convert_element_type_18 = None
        erf_1: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
        add_18: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_21: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_19, add_18);  mul_19 = add_18 = None
        convert_element_type_19: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        view_74: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_75);  convert_element_type_19 = _shape_param_75 = None
        permute_50: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_3: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg29_1, view_74, permute_50);  arg29_1 = view_74 = permute_50 = None
        view_75: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_3, _shape_param_76);  addmm_3 = _shape_param_76 = None
        add_19: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_75, convert_element_type_17);  view_75 = convert_element_type_17 = None
        convert_element_type_20: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_19, torch.float32);  add_19 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_20, [2], correction = 0, keepdim = True)
        getitem_6: "f32[512, 16, 1]" = var_mean_3[0]
        getitem_7: "f32[512, 16, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_7: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_7);  convert_element_type_20 = getitem_7 = None
        add_20: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        mul_22: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_3);  sub_7 = rsqrt_3 = None
        mul_23: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_22, arg30_1);  mul_22 = arg30_1 = None
        add_21: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_23, arg31_1);  mul_23 = arg31_1 = None
        convert_element_type_21: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16);  add_21 = None
        unsqueeze_63: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_21, 3)
        unsqueeze_64: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 4);  unsqueeze_63 = None
        view_76: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_64, _shape_param_77);  unsqueeze_64 = _shape_param_77 = None
        squeeze_20: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_76, 0);  view_76 = None
        unsqueeze_65: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg32_1, 3);  arg32_1 = None
        unsqueeze_66: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_65, 4);  unsqueeze_65 = None
        view_77: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_66, _shape_param_78);  unsqueeze_66 = _shape_param_78 = None
        squeeze_21: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_77, 0);  view_77 = None
        mm_10: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_20, squeeze_21);  squeeze_20 = squeeze_21 = None
        unsqueeze_67: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_10, 0);  mm_10 = None
        view_78: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_67, _shape_param_79);  unsqueeze_67 = _shape_param_79 = None
        permute_51: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_78, [0, 1, 3, 4, 2]);  view_78 = None
        view_79: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_51, _shape_param_80);  permute_51 = _shape_param_80 = None
        add_22: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_79, arg33_1);  arg33_1 = None
        unsqueeze_68: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_22, 4);  add_22 = None
        permute_52: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_68, [1, 2, 0, 4, 3]);  unsqueeze_68 = None
        permute_53: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_52, [0, 1, 2, 4, 3]);  permute_52 = None
        view_80: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_53, _shape_param_81);  permute_53 = _shape_param_81 = None
        unsqueeze_69: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_21, 3)
        unsqueeze_70: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 4);  unsqueeze_69 = None
        view_81: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_70, _shape_param_82);  unsqueeze_70 = _shape_param_82 = None
        squeeze_22: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_81, 0);  view_81 = None
        unsqueeze_71: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, 3);  arg34_1 = None
        unsqueeze_72: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 4);  unsqueeze_71 = None
        view_82: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_72, _shape_param_83);  unsqueeze_72 = _shape_param_83 = None
        squeeze_23: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_82, 0);  view_82 = None
        mm_11: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_22, squeeze_23);  squeeze_22 = squeeze_23 = None
        unsqueeze_73: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_11, 0);  mm_11 = None
        view_83: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_73, _shape_param_84);  unsqueeze_73 = _shape_param_84 = None
        permute_54: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_83, [0, 1, 3, 4, 2]);  view_83 = None
        view_84: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_54, _shape_param_85);  permute_54 = _shape_param_85 = None
        unsqueeze_74: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_84, 4);  view_84 = None
        permute_55: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_74, [1, 2, 4, 0, 3]);  unsqueeze_74 = None
        permute_56: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_55, [0, 1, 4, 3, 2]);  permute_55 = None
        view_85: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_56, _shape_param_86);  permute_56 = _shape_param_86 = None
        bmm_6: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_80, view_85);  view_80 = view_85 = None
        view_86: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_87);  bmm_6 = _shape_param_87 = None
        permute_57: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_86, [0, 1, 2, 4, 3]);  view_86 = None
        view_87: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_57, _shape_param_88);  permute_57 = _shape_param_88 = None
        add_23: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_79, arg35_1);  view_79 = arg35_1 = None
        unsqueeze_75: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_23, 4);  add_23 = None
        permute_58: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_75, [1, 2, 0, 4, 3]);  unsqueeze_75 = None
        permute_59: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_58, [0, 1, 2, 4, 3]);  permute_58 = None
        view_88: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_59, _shape_param_89);  permute_59 = _shape_param_89 = None
        convert_element_type_22: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_76: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_22, 3);  convert_element_type_22 = None
        unsqueeze_77: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 4);  unsqueeze_76 = None
        view_89: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_77, _shape_param_90);  unsqueeze_77 = _shape_param_90 = None
        squeeze_24: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_89, 0);  view_89 = None
        unsqueeze_78: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg36_1, 3);  arg36_1 = None
        unsqueeze_79: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 4);  unsqueeze_78 = None
        view_90: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_79, _shape_param_91);  unsqueeze_79 = _shape_param_91 = None
        squeeze_25: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_90, 0);  view_90 = None
        mm_12: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_24, squeeze_25);  squeeze_24 = squeeze_25 = None
        unsqueeze_80: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_12, 0);  mm_12 = None
        view_91: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_80, _shape_param_92);  unsqueeze_80 = _shape_param_92 = None
        permute_60: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_91, [0, 1, 3, 4, 2]);  view_91 = None
        view_92: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_60, _shape_param_93);  permute_60 = _shape_param_93 = None
        unsqueeze_81: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_92, 4);  view_92 = None
        permute_61: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_81, [1, 2, 4, 0, 3]);  unsqueeze_81 = None
        permute_62: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_61, [0, 1, 4, 3, 2]);  permute_61 = None
        view_93: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_62, _shape_param_94);  permute_62 = _shape_param_94 = None
        bmm_7: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_88, view_93);  view_88 = view_93 = None
        view_94: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_7, _shape_param_95);  bmm_7 = _shape_param_95 = None
        permute_63: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_94, [0, 1, 2, 4, 3]);  view_94 = None
        view_95: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_63, _shape_param_96);  permute_63 = _shape_param_96 = None
        view_96: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_95, _shape_param_97);  view_95 = _shape_param_97 = None
        slice_3: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_96, 2, 1, 9223372036854775807);  view_96 = None
        view_97: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_3, _shape_param_98);  slice_3 = _shape_param_98 = None
        iota_4: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_2: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_97, [None, None, None, iota_4]);  view_97 = iota_4 = None
        add_24: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_87, index_2);  view_87 = index_2 = None
        add_25: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_24, 0);  add_24 = None
        mul_24: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_25, 0.125)
        convert_element_type_23: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_24, torch.float32);  mul_24 = None
        eq_2: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_23, convert_element_type_23)
        abs_3: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_23)
        ne_2: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_3, inf);  abs_3 = None
        mul_25: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_2, ne_2);  eq_2 = ne_2 = None
        logical_not_4: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_25);  mul_25 = None
        any_3: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_4, [3], True);  logical_not_4 = None
        logical_not_5: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        convert_element_type_24: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_25, torch.float32);  add_25 = None
        mul_26: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1);  convert_element_type_24 = None
        amax_4: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_26, [3], True)
        sub_8: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_26, amax_4);  mul_26 = amax_4 = None
        mul_27: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_8, 0.125);  sub_8 = None
        amax_5: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_23, [3], True)
        sub_9: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_23, amax_5);  convert_element_type_23 = amax_5 = None
        where_2: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_5, mul_27, sub_9);  logical_not_5 = mul_27 = sub_9 = None
        exp_2: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_2);  where_2 = None
        sum_3: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [3], True)
        div_3: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_25: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        unsqueeze_82: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_25, 4);  convert_element_type_25 = None
        view_98: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_82, _shape_param_99);  unsqueeze_82 = _shape_param_99 = None
        unsqueeze_83: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_21, 3)
        unsqueeze_84: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 4);  unsqueeze_83 = None
        view_99: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_84, _shape_param_100);  unsqueeze_84 = _shape_param_100 = None
        squeeze_26: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_99, 0);  view_99 = None
        unsqueeze_85: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg37_1, 3);  arg37_1 = None
        unsqueeze_86: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 4);  unsqueeze_85 = None
        view_100: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_86, _shape_param_101);  unsqueeze_86 = _shape_param_101 = None
        squeeze_27: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_100, 0);  view_100 = None
        mm_13: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_26, squeeze_27);  squeeze_26 = squeeze_27 = None
        unsqueeze_87: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_13, 0);  mm_13 = None
        view_101: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_87, _shape_param_102);  unsqueeze_87 = _shape_param_102 = None
        permute_64: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_101, [0, 1, 3, 4, 2]);  view_101 = None
        view_102: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_64, _shape_param_103);  permute_64 = _shape_param_103 = None
        unsqueeze_88: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_102, 4);  view_102 = None
        permute_65: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_88, [4, 1, 2, 3, 0]);  unsqueeze_88 = None
        permute_66: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_65, [1, 2, 4, 3, 0]);  permute_65 = None
        view_103: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_66, _shape_param_104);  permute_66 = _shape_param_104 = None
        bmm_8: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_98, view_103);  view_98 = view_103 = None
        view_104: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_105);  bmm_8 = _shape_param_105 = None
        permute_67: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_104, [2, 0, 1, 4, 3]);  view_104 = None
        view_105: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_67, _shape_param_106);  permute_67 = _shape_param_106 = None
        unsqueeze_89: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_105, 4);  view_105 = None
        permute_68: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_89, [0, 1, 4, 3, 2]);  unsqueeze_89 = None
        permute_69: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_68, [0, 1, 3, 4, 2]);  permute_68 = None
        clone_6: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_106: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_6, _shape_param_107);  clone_6 = _shape_param_107 = None
        squeeze_28: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_106, 0);  view_106 = None
        unsqueeze_90: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg38_1, 3);  arg38_1 = None
        unsqueeze_91: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 4);  unsqueeze_90 = None
        permute_70: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_91, [3, 4, 0, 2, 1]);  unsqueeze_91 = None
        permute_71: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_70, [3, 4, 2, 0, 1]);  permute_70 = None
        clone_7: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_107: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_7, _shape_param_108);  clone_7 = _shape_param_108 = None
        squeeze_29: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_107, 0);  view_107 = None
        mm_14: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_28, squeeze_29);  squeeze_28 = squeeze_29 = None
        unsqueeze_92: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_14, 0);  mm_14 = None
        view_108: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_92, _shape_param_109);  unsqueeze_92 = _shape_param_109 = None
        permute_72: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_108, [0, 1, 4, 2, 3]);  view_108 = None
        view_109: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_72, _shape_param_110);  permute_72 = _shape_param_110 = None
        add_26: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_109, convert_element_type_21);  view_109 = None
        convert_element_type_26: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_26, torch.float32);  add_26 = None
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_26, [2], correction = 0, keepdim = True)
        getitem_8: "f32[512, 16, 1]" = var_mean_4[0]
        getitem_9: "f32[512, 16, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_10: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_26, getitem_9);  convert_element_type_26 = getitem_9 = None
        add_27: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_28: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_4);  sub_10 = rsqrt_4 = None
        mul_29: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_28, arg39_1);  mul_28 = arg39_1 = None
        add_28: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_29, arg40_1);  mul_29 = arg40_1 = None
        convert_element_type_27: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None
        view_110: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_111);  _shape_param_111 = None
        permute_73: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_4: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg42_1, view_110, permute_73);  arg42_1 = view_110 = permute_73 = None
        view_111: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_4, _shape_param_112);  addmm_4 = _shape_param_112 = None
        convert_element_type_28: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_111, torch.float32);  view_111 = None
        mul_30: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.5)
        mul_31: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 0.7071067811865476);  convert_element_type_28 = None
        erf_2: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_29: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_32: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_30, add_29);  mul_30 = add_29 = None
        convert_element_type_29: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_32, torch.bfloat16);  mul_32 = None
        view_112: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_29, _shape_param_113);  convert_element_type_29 = _shape_param_113 = None
        permute_74: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_5: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg44_1, view_112, permute_74);  arg44_1 = view_112 = permute_74 = None
        view_113: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_5, _shape_param_114);  addmm_5 = _shape_param_114 = None
        add_30: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_113, convert_element_type_27);  view_113 = convert_element_type_27 = None
        convert_element_type_30: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_30, torch.float32);  add_30 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_30, [2], correction = 0, keepdim = True)
        getitem_10: "f32[512, 16, 1]" = var_mean_5[0]
        getitem_11: "f32[512, 16, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_11: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_30, getitem_11);  convert_element_type_30 = getitem_11 = None
        add_31: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        mul_33: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_5);  sub_11 = rsqrt_5 = None
        mul_34: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg45_1);  mul_33 = arg45_1 = None
        add_32: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_34, arg46_1);  mul_34 = arg46_1 = None
        convert_element_type_31: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None
        unsqueeze_93: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_31, 3)
        unsqueeze_94: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 4);  unsqueeze_93 = None
        view_114: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_94, _shape_param_115);  unsqueeze_94 = _shape_param_115 = None
        squeeze_30: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_114, 0);  view_114 = None
        unsqueeze_95: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg47_1, 3);  arg47_1 = None
        unsqueeze_96: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 4);  unsqueeze_95 = None
        view_115: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_96, _shape_param_116);  unsqueeze_96 = _shape_param_116 = None
        squeeze_31: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_115, 0);  view_115 = None
        mm_15: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_30, squeeze_31);  squeeze_30 = squeeze_31 = None
        unsqueeze_97: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_15, 0);  mm_15 = None
        view_116: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_97, _shape_param_117);  unsqueeze_97 = _shape_param_117 = None
        permute_75: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_116, [0, 1, 3, 4, 2]);  view_116 = None
        view_117: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_75, _shape_param_118);  permute_75 = _shape_param_118 = None
        add_33: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_117, arg48_1);  arg48_1 = None
        unsqueeze_98: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_33, 4);  add_33 = None
        permute_76: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_98, [1, 2, 0, 4, 3]);  unsqueeze_98 = None
        permute_77: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_76, [0, 1, 2, 4, 3]);  permute_76 = None
        view_118: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_77, _shape_param_119);  permute_77 = _shape_param_119 = None
        unsqueeze_99: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_31, 3)
        unsqueeze_100: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 4);  unsqueeze_99 = None
        view_119: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_100, _shape_param_120);  unsqueeze_100 = _shape_param_120 = None
        squeeze_32: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_119, 0);  view_119 = None
        unsqueeze_101: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, 3);  arg49_1 = None
        unsqueeze_102: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 4);  unsqueeze_101 = None
        view_120: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_102, _shape_param_121);  unsqueeze_102 = _shape_param_121 = None
        squeeze_33: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_120, 0);  view_120 = None
        mm_16: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_32, squeeze_33);  squeeze_32 = squeeze_33 = None
        unsqueeze_103: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_16, 0);  mm_16 = None
        view_121: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_103, _shape_param_122);  unsqueeze_103 = _shape_param_122 = None
        permute_78: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_121, [0, 1, 3, 4, 2]);  view_121 = None
        view_122: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_78, _shape_param_123);  permute_78 = _shape_param_123 = None
        unsqueeze_104: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_122, 4);  view_122 = None
        permute_79: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_104, [1, 2, 4, 0, 3]);  unsqueeze_104 = None
        permute_80: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_79, [0, 1, 4, 3, 2]);  permute_79 = None
        view_123: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_80, _shape_param_124);  permute_80 = _shape_param_124 = None
        bmm_9: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_118, view_123);  view_118 = view_123 = None
        view_124: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_9, _shape_param_125);  bmm_9 = _shape_param_125 = None
        permute_81: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_124, [0, 1, 2, 4, 3]);  view_124 = None
        view_125: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_81, _shape_param_126);  permute_81 = _shape_param_126 = None
        add_34: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_117, arg50_1);  view_117 = arg50_1 = None
        unsqueeze_105: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_34, 4);  add_34 = None
        permute_82: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_105, [1, 2, 0, 4, 3]);  unsqueeze_105 = None
        permute_83: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_82, [0, 1, 2, 4, 3]);  permute_82 = None
        view_126: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_83, _shape_param_127);  permute_83 = _shape_param_127 = None
        convert_element_type_32: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_106: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_32, 3);  convert_element_type_32 = None
        unsqueeze_107: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 4);  unsqueeze_106 = None
        view_127: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_107, _shape_param_128);  unsqueeze_107 = _shape_param_128 = None
        squeeze_34: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_127, 0);  view_127 = None
        unsqueeze_108: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg51_1, 3);  arg51_1 = None
        unsqueeze_109: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 4);  unsqueeze_108 = None
        view_128: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_109, _shape_param_129);  unsqueeze_109 = _shape_param_129 = None
        squeeze_35: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_128, 0);  view_128 = None
        mm_17: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_34, squeeze_35);  squeeze_34 = squeeze_35 = None
        unsqueeze_110: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_17, 0);  mm_17 = None
        view_129: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_110, _shape_param_130);  unsqueeze_110 = _shape_param_130 = None
        permute_84: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_129, [0, 1, 3, 4, 2]);  view_129 = None
        view_130: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_84, _shape_param_131);  permute_84 = _shape_param_131 = None
        unsqueeze_111: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_130, 4);  view_130 = None
        permute_85: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_111, [1, 2, 4, 0, 3]);  unsqueeze_111 = None
        permute_86: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_85, [0, 1, 4, 3, 2]);  permute_85 = None
        view_131: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_86, _shape_param_132);  permute_86 = _shape_param_132 = None
        bmm_10: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_126, view_131);  view_126 = view_131 = None
        view_132: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_10, _shape_param_133);  bmm_10 = _shape_param_133 = None
        permute_87: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_132, [0, 1, 2, 4, 3]);  view_132 = None
        view_133: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_87, _shape_param_134);  permute_87 = _shape_param_134 = None
        view_134: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_133, _shape_param_135);  view_133 = _shape_param_135 = None
        slice_4: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_134, 2, 1, 9223372036854775807);  view_134 = None
        view_135: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_4, _shape_param_136);  slice_4 = _shape_param_136 = None
        iota_5: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_3: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_135, [None, None, None, iota_5]);  view_135 = iota_5 = None
        add_35: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_125, index_3);  view_125 = index_3 = None
        add_36: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_35, 0);  add_35 = None
        mul_35: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_36, 0.125)
        convert_element_type_33: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_35, torch.float32);  mul_35 = None
        eq_3: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_33, convert_element_type_33)
        abs_4: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_33)
        ne_3: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_4, inf);  abs_4 = None
        mul_36: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_3, ne_3);  eq_3 = ne_3 = None
        logical_not_6: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_36);  mul_36 = None
        any_4: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_6, [3], True);  logical_not_6 = None
        logical_not_7: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        convert_element_type_34: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_36, torch.float32);  add_36 = None
        mul_37: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1);  convert_element_type_34 = None
        amax_6: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_37, [3], True)
        sub_12: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_37, amax_6);  mul_37 = amax_6 = None
        mul_38: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_12, 0.125);  sub_12 = None
        amax_7: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_33, [3], True)
        sub_13: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_33, amax_7);  convert_element_type_33 = amax_7 = None
        where_3: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_7, mul_38, sub_13);  logical_not_7 = mul_38 = sub_13 = None
        exp_3: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_3);  where_3 = None
        sum_4: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [3], True)
        div_4: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_35: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        unsqueeze_112: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_35, 4);  convert_element_type_35 = None
        view_136: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_112, _shape_param_137);  unsqueeze_112 = _shape_param_137 = None
        unsqueeze_113: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_31, 3)
        unsqueeze_114: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 4);  unsqueeze_113 = None
        view_137: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_114, _shape_param_138);  unsqueeze_114 = _shape_param_138 = None
        squeeze_36: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_137, 0);  view_137 = None
        unsqueeze_115: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg52_1, 3);  arg52_1 = None
        unsqueeze_116: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 4);  unsqueeze_115 = None
        view_138: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_116, _shape_param_139);  unsqueeze_116 = _shape_param_139 = None
        squeeze_37: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_138, 0);  view_138 = None
        mm_18: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_36, squeeze_37);  squeeze_36 = squeeze_37 = None
        unsqueeze_117: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_18, 0);  mm_18 = None
        view_139: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_117, _shape_param_140);  unsqueeze_117 = _shape_param_140 = None
        permute_88: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_139, [0, 1, 3, 4, 2]);  view_139 = None
        view_140: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_88, _shape_param_141);  permute_88 = _shape_param_141 = None
        unsqueeze_118: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_140, 4);  view_140 = None
        permute_89: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_118, [4, 1, 2, 3, 0]);  unsqueeze_118 = None
        permute_90: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_89, [1, 2, 4, 3, 0]);  permute_89 = None
        view_141: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_90, _shape_param_142);  permute_90 = _shape_param_142 = None
        bmm_11: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_136, view_141);  view_136 = view_141 = None
        view_142: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_143);  bmm_11 = _shape_param_143 = None
        permute_91: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_142, [2, 0, 1, 4, 3]);  view_142 = None
        view_143: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_91, _shape_param_144);  permute_91 = _shape_param_144 = None
        unsqueeze_119: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_143, 4);  view_143 = None
        permute_92: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_119, [0, 1, 4, 3, 2]);  unsqueeze_119 = None
        permute_93: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 4, 2]);  permute_92 = None
        clone_8: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_144: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_8, _shape_param_145);  clone_8 = _shape_param_145 = None
        squeeze_38: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_144, 0);  view_144 = None
        unsqueeze_120: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg53_1, 3);  arg53_1 = None
        unsqueeze_121: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 4);  unsqueeze_120 = None
        permute_94: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_121, [3, 4, 0, 2, 1]);  unsqueeze_121 = None
        permute_95: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_94, [3, 4, 2, 0, 1]);  permute_94 = None
        clone_9: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_145: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_9, _shape_param_146);  clone_9 = _shape_param_146 = None
        squeeze_39: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_145, 0);  view_145 = None
        mm_19: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_38, squeeze_39);  squeeze_38 = squeeze_39 = None
        unsqueeze_122: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_19, 0);  mm_19 = None
        view_146: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_122, _shape_param_147);  unsqueeze_122 = _shape_param_147 = None
        permute_96: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_146, [0, 1, 4, 2, 3]);  view_146 = None
        view_147: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_96, _shape_param_148);  permute_96 = _shape_param_148 = None
        add_37: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_147, convert_element_type_31);  view_147 = None
        convert_element_type_36: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_37, torch.float32);  add_37 = None
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_36, [2], correction = 0, keepdim = True)
        getitem_12: "f32[512, 16, 1]" = var_mean_6[0]
        getitem_13: "f32[512, 16, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_14: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_36, getitem_13);  convert_element_type_36 = getitem_13 = None
        add_38: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_39: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_6);  sub_14 = rsqrt_6 = None
        mul_40: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_39, arg54_1);  mul_39 = arg54_1 = None
        add_39: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_40, arg55_1);  mul_40 = arg55_1 = None
        convert_element_type_37: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        view_148: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_37, _shape_param_149);  _shape_param_149 = None
        permute_97: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
        addmm_6: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg57_1, view_148, permute_97);  arg57_1 = view_148 = permute_97 = None
        view_149: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_6, _shape_param_150);  addmm_6 = _shape_param_150 = None
        convert_element_type_38: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_149, torch.float32);  view_149 = None
        mul_41: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5)
        mul_42: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.7071067811865476);  convert_element_type_38 = None
        erf_3: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_40: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_43: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_41, add_40);  mul_41 = add_40 = None
        convert_element_type_39: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        view_150: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_39, _shape_param_151);  convert_element_type_39 = _shape_param_151 = None
        permute_98: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
        addmm_7: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg59_1, view_150, permute_98);  arg59_1 = view_150 = permute_98 = None
        view_151: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_7, _shape_param_152);  addmm_7 = _shape_param_152 = None
        add_41: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_151, convert_element_type_37);  view_151 = convert_element_type_37 = None
        convert_element_type_40: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_41, torch.float32);  add_41 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_40, [2], correction = 0, keepdim = True)
        getitem_14: "f32[512, 16, 1]" = var_mean_7[0]
        getitem_15: "f32[512, 16, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_15: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_40, getitem_15);  convert_element_type_40 = getitem_15 = None
        add_42: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        mul_44: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_7);  sub_15 = rsqrt_7 = None
        mul_45: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_44, arg60_1);  mul_44 = arg60_1 = None
        add_43: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_45, arg61_1);  mul_45 = arg61_1 = None
        convert_element_type_41: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        unsqueeze_123: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_41, 3)
        unsqueeze_124: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 4);  unsqueeze_123 = None
        view_152: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_124, _shape_param_153);  unsqueeze_124 = _shape_param_153 = None
        squeeze_40: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_152, 0);  view_152 = None
        unsqueeze_125: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg62_1, 3);  arg62_1 = None
        unsqueeze_126: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 4);  unsqueeze_125 = None
        view_153: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_126, _shape_param_154);  unsqueeze_126 = _shape_param_154 = None
        squeeze_41: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_153, 0);  view_153 = None
        mm_20: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_40, squeeze_41);  squeeze_40 = squeeze_41 = None
        unsqueeze_127: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_20, 0);  mm_20 = None
        view_154: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_127, _shape_param_155);  unsqueeze_127 = _shape_param_155 = None
        permute_99: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_154, [0, 1, 3, 4, 2]);  view_154 = None
        view_155: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_99, _shape_param_156);  permute_99 = _shape_param_156 = None
        add_44: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_155, arg63_1);  arg63_1 = None
        unsqueeze_128: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_44, 4);  add_44 = None
        permute_100: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_128, [1, 2, 0, 4, 3]);  unsqueeze_128 = None
        permute_101: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_100, [0, 1, 2, 4, 3]);  permute_100 = None
        view_156: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_101, _shape_param_157);  permute_101 = _shape_param_157 = None
        unsqueeze_129: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_41, 3)
        unsqueeze_130: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 4);  unsqueeze_129 = None
        view_157: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_130, _shape_param_158);  unsqueeze_130 = _shape_param_158 = None
        squeeze_42: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_157, 0);  view_157 = None
        unsqueeze_131: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, 3);  arg64_1 = None
        unsqueeze_132: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 4);  unsqueeze_131 = None
        view_158: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_132, _shape_param_159);  unsqueeze_132 = _shape_param_159 = None
        squeeze_43: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_158, 0);  view_158 = None
        mm_21: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_42, squeeze_43);  squeeze_42 = squeeze_43 = None
        unsqueeze_133: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_21, 0);  mm_21 = None
        view_159: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_133, _shape_param_160);  unsqueeze_133 = _shape_param_160 = None
        permute_102: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_159, [0, 1, 3, 4, 2]);  view_159 = None
        view_160: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_102, _shape_param_161);  permute_102 = _shape_param_161 = None
        unsqueeze_134: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_160, 4);  view_160 = None
        permute_103: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_134, [1, 2, 4, 0, 3]);  unsqueeze_134 = None
        permute_104: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_103, [0, 1, 4, 3, 2]);  permute_103 = None
        view_161: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_104, _shape_param_162);  permute_104 = _shape_param_162 = None
        bmm_12: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_156, view_161);  view_156 = view_161 = None
        view_162: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_12, _shape_param_163);  bmm_12 = _shape_param_163 = None
        permute_105: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_162, [0, 1, 2, 4, 3]);  view_162 = None
        view_163: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_105, _shape_param_164);  permute_105 = _shape_param_164 = None
        add_45: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_155, arg65_1);  view_155 = arg65_1 = None
        unsqueeze_135: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_45, 4);  add_45 = None
        permute_106: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_135, [1, 2, 0, 4, 3]);  unsqueeze_135 = None
        permute_107: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_106, [0, 1, 2, 4, 3]);  permute_106 = None
        view_164: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_107, _shape_param_165);  permute_107 = _shape_param_165 = None
        convert_element_type_42: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_136: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, 3);  convert_element_type_42 = None
        unsqueeze_137: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, 4);  unsqueeze_136 = None
        view_165: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_137, _shape_param_166);  unsqueeze_137 = _shape_param_166 = None
        squeeze_44: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_165, 0);  view_165 = None
        unsqueeze_138: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg66_1, 3);  arg66_1 = None
        unsqueeze_139: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 4);  unsqueeze_138 = None
        view_166: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_139, _shape_param_167);  unsqueeze_139 = _shape_param_167 = None
        squeeze_45: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_166, 0);  view_166 = None
        mm_22: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_44, squeeze_45);  squeeze_44 = squeeze_45 = None
        unsqueeze_140: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_22, 0);  mm_22 = None
        view_167: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_140, _shape_param_168);  unsqueeze_140 = _shape_param_168 = None
        permute_108: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_167, [0, 1, 3, 4, 2]);  view_167 = None
        view_168: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_108, _shape_param_169);  permute_108 = _shape_param_169 = None
        unsqueeze_141: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_168, 4);  view_168 = None
        permute_109: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_141, [1, 2, 4, 0, 3]);  unsqueeze_141 = None
        permute_110: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_109, [0, 1, 4, 3, 2]);  permute_109 = None
        view_169: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_110, _shape_param_170);  permute_110 = _shape_param_170 = None
        bmm_13: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_164, view_169);  view_164 = view_169 = None
        view_170: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_13, _shape_param_171);  bmm_13 = _shape_param_171 = None
        permute_111: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_170, [0, 1, 2, 4, 3]);  view_170 = None
        view_171: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_111, _shape_param_172);  permute_111 = _shape_param_172 = None
        view_172: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_171, _shape_param_173);  view_171 = _shape_param_173 = None
        slice_5: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_172, 2, 1, 9223372036854775807);  view_172 = None
        view_173: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_5, _shape_param_174);  slice_5 = _shape_param_174 = None
        iota_6: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_4: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_173, [None, None, None, iota_6]);  view_173 = iota_6 = None
        add_46: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_163, index_4);  view_163 = index_4 = None
        add_47: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_46, 0);  add_46 = None
        mul_46: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_47, 0.125)
        convert_element_type_43: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_46, torch.float32);  mul_46 = None
        eq_4: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_43, convert_element_type_43)
        abs_5: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_43)
        ne_4: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_5, inf);  abs_5 = None
        mul_47: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_4, ne_4);  eq_4 = ne_4 = None
        logical_not_8: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_47);  mul_47 = None
        any_5: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_8, [3], True);  logical_not_8 = None
        logical_not_9: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        convert_element_type_44: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_47, torch.float32);  add_47 = None
        mul_48: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1);  convert_element_type_44 = None
        amax_8: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_48, [3], True)
        sub_16: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_48, amax_8);  mul_48 = amax_8 = None
        mul_49: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_16, 0.125);  sub_16 = None
        amax_9: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_43, [3], True)
        sub_17: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_43, amax_9);  convert_element_type_43 = amax_9 = None
        where_4: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_9, mul_49, sub_17);  logical_not_9 = mul_49 = sub_17 = None
        exp_4: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_4);  where_4 = None
        sum_5: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [3], True)
        div_5: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_45: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        unsqueeze_142: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_45, 4);  convert_element_type_45 = None
        view_174: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_142, _shape_param_175);  unsqueeze_142 = _shape_param_175 = None
        unsqueeze_143: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_41, 3)
        unsqueeze_144: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 4);  unsqueeze_143 = None
        view_175: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_144, _shape_param_176);  unsqueeze_144 = _shape_param_176 = None
        squeeze_46: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_175, 0);  view_175 = None
        unsqueeze_145: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg67_1, 3);  arg67_1 = None
        unsqueeze_146: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 4);  unsqueeze_145 = None
        view_176: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_146, _shape_param_177);  unsqueeze_146 = _shape_param_177 = None
        squeeze_47: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_176, 0);  view_176 = None
        mm_23: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_46, squeeze_47);  squeeze_46 = squeeze_47 = None
        unsqueeze_147: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_23, 0);  mm_23 = None
        view_177: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_147, _shape_param_178);  unsqueeze_147 = _shape_param_178 = None
        permute_112: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_177, [0, 1, 3, 4, 2]);  view_177 = None
        view_178: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_112, _shape_param_179);  permute_112 = _shape_param_179 = None
        unsqueeze_148: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_178, 4);  view_178 = None
        permute_113: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_148, [4, 1, 2, 3, 0]);  unsqueeze_148 = None
        permute_114: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_113, [1, 2, 4, 3, 0]);  permute_113 = None
        view_179: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_114, _shape_param_180);  permute_114 = _shape_param_180 = None
        bmm_14: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_174, view_179);  view_174 = view_179 = None
        view_180: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_14, _shape_param_181);  bmm_14 = _shape_param_181 = None
        permute_115: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_180, [2, 0, 1, 4, 3]);  view_180 = None
        view_181: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_115, _shape_param_182);  permute_115 = _shape_param_182 = None
        unsqueeze_149: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_181, 4);  view_181 = None
        permute_116: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_149, [0, 1, 4, 3, 2]);  unsqueeze_149 = None
        permute_117: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_116, [0, 1, 3, 4, 2]);  permute_116 = None
        clone_10: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_182: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_10, _shape_param_183);  clone_10 = _shape_param_183 = None
        squeeze_48: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_182, 0);  view_182 = None
        unsqueeze_150: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg68_1, 3);  arg68_1 = None
        unsqueeze_151: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 4);  unsqueeze_150 = None
        permute_118: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_151, [3, 4, 0, 2, 1]);  unsqueeze_151 = None
        permute_119: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_118, [3, 4, 2, 0, 1]);  permute_118 = None
        clone_11: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
        view_183: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_11, _shape_param_184);  clone_11 = _shape_param_184 = None
        squeeze_49: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_183, 0);  view_183 = None
        mm_24: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_48, squeeze_49);  squeeze_48 = squeeze_49 = None
        unsqueeze_152: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_24, 0);  mm_24 = None
        view_184: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_152, _shape_param_185);  unsqueeze_152 = _shape_param_185 = None
        permute_120: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_184, [0, 1, 4, 2, 3]);  view_184 = None
        view_185: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_120, _shape_param_186);  permute_120 = _shape_param_186 = None
        add_48: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_185, convert_element_type_41);  view_185 = None
        convert_element_type_46: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_48, torch.float32);  add_48 = None
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_46, [2], correction = 0, keepdim = True)
        getitem_16: "f32[512, 16, 1]" = var_mean_8[0]
        getitem_17: "f32[512, 16, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_18: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_46, getitem_17);  convert_element_type_46 = getitem_17 = None
        add_49: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_50: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_8);  sub_18 = rsqrt_8 = None
        mul_51: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_50, arg69_1);  mul_50 = arg69_1 = None
        add_50: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_51, arg70_1);  mul_51 = arg70_1 = None
        convert_element_type_47: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None
        view_186: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_47, _shape_param_187);  _shape_param_187 = None
        permute_121: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_8: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg72_1, view_186, permute_121);  arg72_1 = view_186 = permute_121 = None
        view_187: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_8, _shape_param_188);  addmm_8 = _shape_param_188 = None
        convert_element_type_48: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_187, torch.float32);  view_187 = None
        mul_52: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 0.5)
        mul_53: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 0.7071067811865476);  convert_element_type_48 = None
        erf_4: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
        add_51: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_54: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_52, add_51);  mul_52 = add_51 = None
        convert_element_type_49: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_54, torch.bfloat16);  mul_54 = None
        view_188: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_189);  convert_element_type_49 = _shape_param_189 = None
        permute_122: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_9: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg74_1, view_188, permute_122);  arg74_1 = view_188 = permute_122 = None
        view_189: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_9, _shape_param_190);  addmm_9 = _shape_param_190 = None
        add_52: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_189, convert_element_type_47);  view_189 = convert_element_type_47 = None
        convert_element_type_50: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_52, torch.float32);  add_52 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_50, [2], correction = 0, keepdim = True)
        getitem_18: "f32[512, 16, 1]" = var_mean_9[0]
        getitem_19: "f32[512, 16, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_19: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_50, getitem_19);  convert_element_type_50 = getitem_19 = None
        add_53: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_55: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_9);  sub_19 = rsqrt_9 = None
        mul_56: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_55, arg75_1);  mul_55 = arg75_1 = None
        add_54: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_56, arg76_1);  mul_56 = arg76_1 = None
        convert_element_type_51: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None
        unsqueeze_153: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_51, 3)
        unsqueeze_154: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 4);  unsqueeze_153 = None
        view_190: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_154, _shape_param_191);  unsqueeze_154 = _shape_param_191 = None
        squeeze_50: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_190, 0);  view_190 = None
        unsqueeze_155: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg77_1, 3);  arg77_1 = None
        unsqueeze_156: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 4);  unsqueeze_155 = None
        view_191: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_156, _shape_param_192);  unsqueeze_156 = _shape_param_192 = None
        squeeze_51: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_191, 0);  view_191 = None
        mm_25: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_50, squeeze_51);  squeeze_50 = squeeze_51 = None
        unsqueeze_157: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_25, 0);  mm_25 = None
        view_192: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_157, _shape_param_193);  unsqueeze_157 = _shape_param_193 = None
        permute_123: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_192, [0, 1, 3, 4, 2]);  view_192 = None
        view_193: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_123, _shape_param_194);  permute_123 = _shape_param_194 = None
        add_55: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_193, arg78_1);  arg78_1 = None
        unsqueeze_158: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_55, 4);  add_55 = None
        permute_124: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_158, [1, 2, 0, 4, 3]);  unsqueeze_158 = None
        permute_125: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_124, [0, 1, 2, 4, 3]);  permute_124 = None
        view_194: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_125, _shape_param_195);  permute_125 = _shape_param_195 = None
        unsqueeze_159: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_51, 3)
        unsqueeze_160: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 4);  unsqueeze_159 = None
        view_195: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_160, _shape_param_196);  unsqueeze_160 = _shape_param_196 = None
        squeeze_52: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_195, 0);  view_195 = None
        unsqueeze_161: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, 3);  arg79_1 = None
        unsqueeze_162: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 4);  unsqueeze_161 = None
        view_196: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_162, _shape_param_197);  unsqueeze_162 = _shape_param_197 = None
        squeeze_53: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_196, 0);  view_196 = None
        mm_26: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_52, squeeze_53);  squeeze_52 = squeeze_53 = None
        unsqueeze_163: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_26, 0);  mm_26 = None
        view_197: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_163, _shape_param_198);  unsqueeze_163 = _shape_param_198 = None
        permute_126: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_197, [0, 1, 3, 4, 2]);  view_197 = None
        view_198: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_126, _shape_param_199);  permute_126 = _shape_param_199 = None
        unsqueeze_164: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_198, 4);  view_198 = None
        permute_127: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_164, [1, 2, 4, 0, 3]);  unsqueeze_164 = None
        permute_128: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_127, [0, 1, 4, 3, 2]);  permute_127 = None
        view_199: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_128, _shape_param_200);  permute_128 = _shape_param_200 = None
        bmm_15: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_194, view_199);  view_194 = view_199 = None
        view_200: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_15, _shape_param_201);  bmm_15 = _shape_param_201 = None
        permute_129: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_200, [0, 1, 2, 4, 3]);  view_200 = None
        view_201: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_129, _shape_param_202);  permute_129 = _shape_param_202 = None
        add_56: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_193, arg80_1);  view_193 = arg80_1 = None
        unsqueeze_165: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_56, 4);  add_56 = None
        permute_130: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_165, [1, 2, 0, 4, 3]);  unsqueeze_165 = None
        permute_131: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_130, [0, 1, 2, 4, 3]);  permute_130 = None
        view_202: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_131, _shape_param_203);  permute_131 = _shape_param_203 = None
        convert_element_type_52: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_166: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_52, 3);  convert_element_type_52 = None
        unsqueeze_167: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 4);  unsqueeze_166 = None
        view_203: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_167, _shape_param_204);  unsqueeze_167 = _shape_param_204 = None
        squeeze_54: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_203, 0);  view_203 = None
        unsqueeze_168: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg81_1, 3);  arg81_1 = None
        unsqueeze_169: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 4);  unsqueeze_168 = None
        view_204: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_169, _shape_param_205);  unsqueeze_169 = _shape_param_205 = None
        squeeze_55: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_204, 0);  view_204 = None
        mm_27: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_54, squeeze_55);  squeeze_54 = squeeze_55 = None
        unsqueeze_170: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_27, 0);  mm_27 = None
        view_205: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_170, _shape_param_206);  unsqueeze_170 = _shape_param_206 = None
        permute_132: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_205, [0, 1, 3, 4, 2]);  view_205 = None
        view_206: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_132, _shape_param_207);  permute_132 = _shape_param_207 = None
        unsqueeze_171: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_206, 4);  view_206 = None
        permute_133: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_171, [1, 2, 4, 0, 3]);  unsqueeze_171 = None
        permute_134: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_133, [0, 1, 4, 3, 2]);  permute_133 = None
        view_207: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_134, _shape_param_208);  permute_134 = _shape_param_208 = None
        bmm_16: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_202, view_207);  view_202 = view_207 = None
        view_208: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_16, _shape_param_209);  bmm_16 = _shape_param_209 = None
        permute_135: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_208, [0, 1, 2, 4, 3]);  view_208 = None
        view_209: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_135, _shape_param_210);  permute_135 = _shape_param_210 = None
        view_210: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_209, _shape_param_211);  view_209 = _shape_param_211 = None
        slice_6: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_210, 2, 1, 9223372036854775807);  view_210 = None
        view_211: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_6, _shape_param_212);  slice_6 = _shape_param_212 = None
        iota_7: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_5: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_211, [None, None, None, iota_7]);  view_211 = iota_7 = None
        add_57: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_201, index_5);  view_201 = index_5 = None
        add_58: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_57, 0);  add_57 = None
        mul_57: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_58, 0.125)
        convert_element_type_53: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_57, torch.float32);  mul_57 = None
        eq_5: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_53, convert_element_type_53)
        abs_6: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_53)
        ne_5: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_6, inf);  abs_6 = None
        mul_58: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_5, ne_5);  eq_5 = ne_5 = None
        logical_not_10: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_58);  mul_58 = None
        any_6: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_10, [3], True);  logical_not_10 = None
        logical_not_11: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        convert_element_type_54: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_58, torch.float32);  add_58 = None
        mul_59: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1);  convert_element_type_54 = None
        amax_10: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_59, [3], True)
        sub_20: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_59, amax_10);  mul_59 = amax_10 = None
        mul_60: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_20, 0.125);  sub_20 = None
        amax_11: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_53, [3], True)
        sub_21: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_53, amax_11);  convert_element_type_53 = amax_11 = None
        where_5: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_11, mul_60, sub_21);  logical_not_11 = mul_60 = sub_21 = None
        exp_5: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_5);  where_5 = None
        sum_6: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [3], True)
        div_6: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_55: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        unsqueeze_172: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_55, 4);  convert_element_type_55 = None
        view_212: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_172, _shape_param_213);  unsqueeze_172 = _shape_param_213 = None
        unsqueeze_173: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_51, 3)
        unsqueeze_174: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 4);  unsqueeze_173 = None
        view_213: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_174, _shape_param_214);  unsqueeze_174 = _shape_param_214 = None
        squeeze_56: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_213, 0);  view_213 = None
        unsqueeze_175: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg82_1, 3);  arg82_1 = None
        unsqueeze_176: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 4);  unsqueeze_175 = None
        view_214: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_176, _shape_param_215);  unsqueeze_176 = _shape_param_215 = None
        squeeze_57: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_214, 0);  view_214 = None
        mm_28: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_56, squeeze_57);  squeeze_56 = squeeze_57 = None
        unsqueeze_177: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_28, 0);  mm_28 = None
        view_215: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_177, _shape_param_216);  unsqueeze_177 = _shape_param_216 = None
        permute_136: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_215, [0, 1, 3, 4, 2]);  view_215 = None
        view_216: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_136, _shape_param_217);  permute_136 = _shape_param_217 = None
        unsqueeze_178: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_216, 4);  view_216 = None
        permute_137: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_178, [4, 1, 2, 3, 0]);  unsqueeze_178 = None
        permute_138: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_137, [1, 2, 4, 3, 0]);  permute_137 = None
        view_217: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_138, _shape_param_218);  permute_138 = _shape_param_218 = None
        bmm_17: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_212, view_217);  view_212 = view_217 = None
        view_218: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_17, _shape_param_219);  bmm_17 = _shape_param_219 = None
        permute_139: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_218, [2, 0, 1, 4, 3]);  view_218 = None
        view_219: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_139, _shape_param_220);  permute_139 = _shape_param_220 = None
        unsqueeze_179: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_219, 4);  view_219 = None
        permute_140: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_179, [0, 1, 4, 3, 2]);  unsqueeze_179 = None
        permute_141: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_140, [0, 1, 3, 4, 2]);  permute_140 = None
        clone_12: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
        view_220: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_12, _shape_param_221);  clone_12 = _shape_param_221 = None
        squeeze_58: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_220, 0);  view_220 = None
        unsqueeze_180: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg83_1, 3);  arg83_1 = None
        unsqueeze_181: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 4);  unsqueeze_180 = None
        permute_142: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_181, [3, 4, 0, 2, 1]);  unsqueeze_181 = None
        permute_143: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_142, [3, 4, 2, 0, 1]);  permute_142 = None
        clone_13: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
        view_221: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_13, _shape_param_222);  clone_13 = _shape_param_222 = None
        squeeze_59: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_221, 0);  view_221 = None
        mm_29: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_58, squeeze_59);  squeeze_58 = squeeze_59 = None
        unsqueeze_182: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_29, 0);  mm_29 = None
        view_222: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_182, _shape_param_223);  unsqueeze_182 = _shape_param_223 = None
        permute_144: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_222, [0, 1, 4, 2, 3]);  view_222 = None
        view_223: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_144, _shape_param_224);  permute_144 = _shape_param_224 = None
        add_59: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_223, convert_element_type_51);  view_223 = None
        convert_element_type_56: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32);  add_59 = None
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_56, [2], correction = 0, keepdim = True)
        getitem_20: "f32[512, 16, 1]" = var_mean_10[0]
        getitem_21: "f32[512, 16, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_22: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_56, getitem_21);  convert_element_type_56 = getitem_21 = None
        add_60: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_61: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_10);  sub_22 = rsqrt_10 = None
        mul_62: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_61, arg84_1);  mul_61 = arg84_1 = None
        add_61: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_62, arg85_1);  mul_62 = arg85_1 = None
        convert_element_type_57: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16);  add_61 = None
        view_224: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_225);  _shape_param_225 = None
        permute_145: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
        addmm_10: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg87_1, view_224, permute_145);  arg87_1 = view_224 = permute_145 = None
        view_225: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_10, _shape_param_226);  addmm_10 = _shape_param_226 = None
        convert_element_type_58: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_225, torch.float32);  view_225 = None
        mul_63: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.5)
        mul_64: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.7071067811865476);  convert_element_type_58 = None
        erf_5: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_62: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_65: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_63, add_62);  mul_63 = add_62 = None
        convert_element_type_59: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None
        view_226: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_59, _shape_param_227);  convert_element_type_59 = _shape_param_227 = None
        permute_146: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
        addmm_11: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg89_1, view_226, permute_146);  arg89_1 = view_226 = permute_146 = None
        view_227: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_11, _shape_param_228);  addmm_11 = _shape_param_228 = None
        add_63: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_227, convert_element_type_57);  view_227 = convert_element_type_57 = None
        convert_element_type_60: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_60, [2], correction = 0, keepdim = True)
        getitem_22: "f32[512, 16, 1]" = var_mean_11[0]
        getitem_23: "f32[512, 16, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_23: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_60, getitem_23);  convert_element_type_60 = getitem_23 = None
        add_64: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_66: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_11);  sub_23 = rsqrt_11 = None
        mul_67: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_66, arg90_1);  mul_66 = arg90_1 = None
        add_65: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_67, arg91_1);  mul_67 = arg91_1 = None
        convert_element_type_61: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None
        unsqueeze_183: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_61, 3)
        unsqueeze_184: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 4);  unsqueeze_183 = None
        view_228: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_184, _shape_param_229);  unsqueeze_184 = _shape_param_229 = None
        squeeze_60: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_228, 0);  view_228 = None
        unsqueeze_185: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg92_1, 3);  arg92_1 = None
        unsqueeze_186: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 4);  unsqueeze_185 = None
        view_229: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_186, _shape_param_230);  unsqueeze_186 = _shape_param_230 = None
        squeeze_61: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_229, 0);  view_229 = None
        mm_30: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_60, squeeze_61);  squeeze_60 = squeeze_61 = None
        unsqueeze_187: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_30, 0);  mm_30 = None
        view_230: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_187, _shape_param_231);  unsqueeze_187 = _shape_param_231 = None
        permute_147: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_230, [0, 1, 3, 4, 2]);  view_230 = None
        view_231: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_147, _shape_param_232);  permute_147 = _shape_param_232 = None
        add_66: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_231, arg93_1);  arg93_1 = None
        unsqueeze_188: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_66, 4);  add_66 = None
        permute_148: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_188, [1, 2, 0, 4, 3]);  unsqueeze_188 = None
        permute_149: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_148, [0, 1, 2, 4, 3]);  permute_148 = None
        view_232: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_149, _shape_param_233);  permute_149 = _shape_param_233 = None
        unsqueeze_189: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_61, 3)
        unsqueeze_190: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 4);  unsqueeze_189 = None
        view_233: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_190, _shape_param_234);  unsqueeze_190 = _shape_param_234 = None
        squeeze_62: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_233, 0);  view_233 = None
        unsqueeze_191: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, 3);  arg94_1 = None
        unsqueeze_192: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 4);  unsqueeze_191 = None
        view_234: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_192, _shape_param_235);  unsqueeze_192 = _shape_param_235 = None
        squeeze_63: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_234, 0);  view_234 = None
        mm_31: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_62, squeeze_63);  squeeze_62 = squeeze_63 = None
        unsqueeze_193: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_31, 0);  mm_31 = None
        view_235: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_193, _shape_param_236);  unsqueeze_193 = _shape_param_236 = None
        permute_150: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_235, [0, 1, 3, 4, 2]);  view_235 = None
        view_236: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_150, _shape_param_237);  permute_150 = _shape_param_237 = None
        unsqueeze_194: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_236, 4);  view_236 = None
        permute_151: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_194, [1, 2, 4, 0, 3]);  unsqueeze_194 = None
        permute_152: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_151, [0, 1, 4, 3, 2]);  permute_151 = None
        view_237: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_152, _shape_param_238);  permute_152 = _shape_param_238 = None
        bmm_18: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_232, view_237);  view_232 = view_237 = None
        view_238: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_239);  bmm_18 = _shape_param_239 = None
        permute_153: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_238, [0, 1, 2, 4, 3]);  view_238 = None
        view_239: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_153, _shape_param_240);  permute_153 = _shape_param_240 = None
        add_67: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_231, arg95_1);  view_231 = arg95_1 = None
        unsqueeze_195: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_67, 4);  add_67 = None
        permute_154: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_195, [1, 2, 0, 4, 3]);  unsqueeze_195 = None
        permute_155: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_154, [0, 1, 2, 4, 3]);  permute_154 = None
        view_240: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_155, _shape_param_241);  permute_155 = _shape_param_241 = None
        convert_element_type_62: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_196: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_62, 3);  convert_element_type_62 = None
        unsqueeze_197: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 4);  unsqueeze_196 = None
        view_241: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_197, _shape_param_242);  unsqueeze_197 = _shape_param_242 = None
        squeeze_64: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_241, 0);  view_241 = None
        unsqueeze_198: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg96_1, 3);  arg96_1 = None
        unsqueeze_199: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 4);  unsqueeze_198 = None
        view_242: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_199, _shape_param_243);  unsqueeze_199 = _shape_param_243 = None
        squeeze_65: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_242, 0);  view_242 = None
        mm_32: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_64, squeeze_65);  squeeze_64 = squeeze_65 = None
        unsqueeze_200: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_32, 0);  mm_32 = None
        view_243: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_200, _shape_param_244);  unsqueeze_200 = _shape_param_244 = None
        permute_156: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_243, [0, 1, 3, 4, 2]);  view_243 = None
        view_244: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_156, _shape_param_245);  permute_156 = _shape_param_245 = None
        unsqueeze_201: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_244, 4);  view_244 = None
        permute_157: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_201, [1, 2, 4, 0, 3]);  unsqueeze_201 = None
        permute_158: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_157, [0, 1, 4, 3, 2]);  permute_157 = None
        view_245: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_158, _shape_param_246);  permute_158 = _shape_param_246 = None
        bmm_19: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_240, view_245);  view_240 = view_245 = None
        view_246: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_19, _shape_param_247);  bmm_19 = _shape_param_247 = None
        permute_159: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_246, [0, 1, 2, 4, 3]);  view_246 = None
        view_247: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_159, _shape_param_248);  permute_159 = _shape_param_248 = None
        view_248: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_247, _shape_param_249);  view_247 = _shape_param_249 = None
        slice_7: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_248, 2, 1, 9223372036854775807);  view_248 = None
        view_249: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_7, _shape_param_250);  slice_7 = _shape_param_250 = None
        iota_8: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_6: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_249, [None, None, None, iota_8]);  view_249 = iota_8 = None
        add_68: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_239, index_6);  view_239 = index_6 = None
        add_69: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_68, 0);  add_68 = None
        mul_68: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_69, 0.125)
        convert_element_type_63: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_68, torch.float32);  mul_68 = None
        eq_6: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_63, convert_element_type_63)
        abs_7: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_63)
        ne_6: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_7, inf);  abs_7 = None
        mul_69: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_6, ne_6);  eq_6 = ne_6 = None
        logical_not_12: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_69);  mul_69 = None
        any_7: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_12, [3], True);  logical_not_12 = None
        logical_not_13: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        convert_element_type_64: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_69, torch.float32);  add_69 = None
        mul_70: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1);  convert_element_type_64 = None
        amax_12: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_70, [3], True)
        sub_24: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_70, amax_12);  mul_70 = amax_12 = None
        mul_71: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_24, 0.125);  sub_24 = None
        amax_13: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_63, [3], True)
        sub_25: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_63, amax_13);  convert_element_type_63 = amax_13 = None
        where_6: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_13, mul_71, sub_25);  logical_not_13 = mul_71 = sub_25 = None
        exp_6: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_6);  where_6 = None
        sum_7: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [3], True)
        div_7: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_65: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        unsqueeze_202: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_65, 4);  convert_element_type_65 = None
        view_250: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_202, _shape_param_251);  unsqueeze_202 = _shape_param_251 = None
        unsqueeze_203: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_61, 3)
        unsqueeze_204: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 4);  unsqueeze_203 = None
        view_251: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_204, _shape_param_252);  unsqueeze_204 = _shape_param_252 = None
        squeeze_66: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_251, 0);  view_251 = None
        unsqueeze_205: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg97_1, 3);  arg97_1 = None
        unsqueeze_206: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 4);  unsqueeze_205 = None
        view_252: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_206, _shape_param_253);  unsqueeze_206 = _shape_param_253 = None
        squeeze_67: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_252, 0);  view_252 = None
        mm_33: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_66, squeeze_67);  squeeze_66 = squeeze_67 = None
        unsqueeze_207: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_33, 0);  mm_33 = None
        view_253: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_207, _shape_param_254);  unsqueeze_207 = _shape_param_254 = None
        permute_160: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_253, [0, 1, 3, 4, 2]);  view_253 = None
        view_254: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_160, _shape_param_255);  permute_160 = _shape_param_255 = None
        unsqueeze_208: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_254, 4);  view_254 = None
        permute_161: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_208, [4, 1, 2, 3, 0]);  unsqueeze_208 = None
        permute_162: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_161, [1, 2, 4, 3, 0]);  permute_161 = None
        view_255: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_162, _shape_param_256);  permute_162 = _shape_param_256 = None
        bmm_20: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_250, view_255);  view_250 = view_255 = None
        view_256: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_257);  bmm_20 = _shape_param_257 = None
        permute_163: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_256, [2, 0, 1, 4, 3]);  view_256 = None
        view_257: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_163, _shape_param_258);  permute_163 = _shape_param_258 = None
        unsqueeze_209: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_257, 4);  view_257 = None
        permute_164: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_209, [0, 1, 4, 3, 2]);  unsqueeze_209 = None
        permute_165: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 4, 2]);  permute_164 = None
        clone_14: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_258: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_14, _shape_param_259);  clone_14 = _shape_param_259 = None
        squeeze_68: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_258, 0);  view_258 = None
        unsqueeze_210: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg98_1, 3);  arg98_1 = None
        unsqueeze_211: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 4);  unsqueeze_210 = None
        permute_166: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_211, [3, 4, 0, 2, 1]);  unsqueeze_211 = None
        permute_167: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_166, [3, 4, 2, 0, 1]);  permute_166 = None
        clone_15: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_259: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_15, _shape_param_260);  clone_15 = _shape_param_260 = None
        squeeze_69: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_259, 0);  view_259 = None
        mm_34: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_68, squeeze_69);  squeeze_68 = squeeze_69 = None
        unsqueeze_212: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_34, 0);  mm_34 = None
        view_260: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_212, _shape_param_261);  unsqueeze_212 = _shape_param_261 = None
        permute_168: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_260, [0, 1, 4, 2, 3]);  view_260 = None
        view_261: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_168, _shape_param_262);  permute_168 = _shape_param_262 = None
        add_70: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_261, convert_element_type_61);  view_261 = None
        convert_element_type_66: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_70, torch.float32);  add_70 = None
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_66, [2], correction = 0, keepdim = True)
        getitem_24: "f32[512, 16, 1]" = var_mean_12[0]
        getitem_25: "f32[512, 16, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_26: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_66, getitem_25);  convert_element_type_66 = getitem_25 = None
        add_71: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_72: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_12);  sub_26 = rsqrt_12 = None
        mul_73: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_72, arg99_1);  mul_72 = arg99_1 = None
        add_72: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_73, arg100_1);  mul_73 = arg100_1 = None
        convert_element_type_67: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16);  add_72 = None
        view_262: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_67, _shape_param_263);  _shape_param_263 = None
        permute_169: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_12: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg102_1, view_262, permute_169);  arg102_1 = view_262 = permute_169 = None
        view_263: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_12, _shape_param_264);  addmm_12 = _shape_param_264 = None
        convert_element_type_68: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        mul_74: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.5)
        mul_75: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 0.7071067811865476);  convert_element_type_68 = None
        erf_6: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_73: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_76: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_74, add_73);  mul_74 = add_73 = None
        convert_element_type_69: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None
        view_264: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_69, _shape_param_265);  convert_element_type_69 = _shape_param_265 = None
        permute_170: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_13: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg104_1, view_264, permute_170);  arg104_1 = view_264 = permute_170 = None
        view_265: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_13, _shape_param_266);  addmm_13 = _shape_param_266 = None
        add_74: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_265, convert_element_type_67);  view_265 = convert_element_type_67 = None
        convert_element_type_70: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_74, torch.float32);  add_74 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_70, [2], correction = 0, keepdim = True)
        getitem_26: "f32[512, 16, 1]" = var_mean_13[0]
        getitem_27: "f32[512, 16, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_27: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_70, getitem_27);  convert_element_type_70 = getitem_27 = None
        add_75: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        mul_77: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_13);  sub_27 = rsqrt_13 = None
        mul_78: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_77, arg105_1);  mul_77 = arg105_1 = None
        add_76: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_78, arg106_1);  mul_78 = arg106_1 = None
        convert_element_type_71: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16);  add_76 = None
        unsqueeze_213: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_71, 3)
        unsqueeze_214: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 4);  unsqueeze_213 = None
        view_266: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_214, _shape_param_267);  unsqueeze_214 = _shape_param_267 = None
        squeeze_70: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_266, 0);  view_266 = None
        unsqueeze_215: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg107_1, 3);  arg107_1 = None
        unsqueeze_216: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 4);  unsqueeze_215 = None
        view_267: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_216, _shape_param_268);  unsqueeze_216 = _shape_param_268 = None
        squeeze_71: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_267, 0);  view_267 = None
        mm_35: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_70, squeeze_71);  squeeze_70 = squeeze_71 = None
        unsqueeze_217: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_35, 0);  mm_35 = None
        view_268: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_217, _shape_param_269);  unsqueeze_217 = _shape_param_269 = None
        permute_171: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_268, [0, 1, 3, 4, 2]);  view_268 = None
        view_269: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_171, _shape_param_270);  permute_171 = _shape_param_270 = None
        add_77: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_269, arg108_1);  arg108_1 = None
        unsqueeze_218: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_77, 4);  add_77 = None
        permute_172: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_218, [1, 2, 0, 4, 3]);  unsqueeze_218 = None
        permute_173: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_172, [0, 1, 2, 4, 3]);  permute_172 = None
        view_270: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_173, _shape_param_271);  permute_173 = _shape_param_271 = None
        unsqueeze_219: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_71, 3)
        unsqueeze_220: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 4);  unsqueeze_219 = None
        view_271: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_220, _shape_param_272);  unsqueeze_220 = _shape_param_272 = None
        squeeze_72: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_271, 0);  view_271 = None
        unsqueeze_221: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, 3);  arg109_1 = None
        unsqueeze_222: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 4);  unsqueeze_221 = None
        view_272: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_222, _shape_param_273);  unsqueeze_222 = _shape_param_273 = None
        squeeze_73: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_272, 0);  view_272 = None
        mm_36: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_72, squeeze_73);  squeeze_72 = squeeze_73 = None
        unsqueeze_223: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_36, 0);  mm_36 = None
        view_273: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_223, _shape_param_274);  unsqueeze_223 = _shape_param_274 = None
        permute_174: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_273, [0, 1, 3, 4, 2]);  view_273 = None
        view_274: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_174, _shape_param_275);  permute_174 = _shape_param_275 = None
        unsqueeze_224: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_274, 4);  view_274 = None
        permute_175: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_224, [1, 2, 4, 0, 3]);  unsqueeze_224 = None
        permute_176: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_175, [0, 1, 4, 3, 2]);  permute_175 = None
        view_275: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_176, _shape_param_276);  permute_176 = _shape_param_276 = None
        bmm_21: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_270, view_275);  view_270 = view_275 = None
        view_276: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_21, _shape_param_277);  bmm_21 = _shape_param_277 = None
        permute_177: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_276, [0, 1, 2, 4, 3]);  view_276 = None
        view_277: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_177, _shape_param_278);  permute_177 = _shape_param_278 = None
        add_78: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_269, arg110_1);  view_269 = arg110_1 = None
        unsqueeze_225: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_78, 4);  add_78 = None
        permute_178: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_225, [1, 2, 0, 4, 3]);  unsqueeze_225 = None
        permute_179: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_178, [0, 1, 2, 4, 3]);  permute_178 = None
        view_278: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_179, _shape_param_279);  permute_179 = _shape_param_279 = None
        convert_element_type_72: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_226: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, 3);  convert_element_type_72 = None
        unsqueeze_227: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 4);  unsqueeze_226 = None
        view_279: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_227, _shape_param_280);  unsqueeze_227 = _shape_param_280 = None
        squeeze_74: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_279, 0);  view_279 = None
        unsqueeze_228: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg111_1, 3);  arg111_1 = None
        unsqueeze_229: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 4);  unsqueeze_228 = None
        view_280: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_229, _shape_param_281);  unsqueeze_229 = _shape_param_281 = None
        squeeze_75: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_280, 0);  view_280 = None
        mm_37: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_74, squeeze_75);  squeeze_74 = squeeze_75 = None
        unsqueeze_230: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_37, 0);  mm_37 = None
        view_281: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_230, _shape_param_282);  unsqueeze_230 = _shape_param_282 = None
        permute_180: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_281, [0, 1, 3, 4, 2]);  view_281 = None
        view_282: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_180, _shape_param_283);  permute_180 = _shape_param_283 = None
        unsqueeze_231: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_282, 4);  view_282 = None
        permute_181: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_231, [1, 2, 4, 0, 3]);  unsqueeze_231 = None
        permute_182: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_181, [0, 1, 4, 3, 2]);  permute_181 = None
        view_283: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_182, _shape_param_284);  permute_182 = _shape_param_284 = None
        bmm_22: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_278, view_283);  view_278 = view_283 = None
        view_284: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_22, _shape_param_285);  bmm_22 = _shape_param_285 = None
        permute_183: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_284, [0, 1, 2, 4, 3]);  view_284 = None
        view_285: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_183, _shape_param_286);  permute_183 = _shape_param_286 = None
        view_286: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_285, _shape_param_287);  view_285 = _shape_param_287 = None
        slice_8: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_286, 2, 1, 9223372036854775807);  view_286 = None
        view_287: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_8, _shape_param_288);  slice_8 = _shape_param_288 = None
        iota_9: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_7: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_287, [None, None, None, iota_9]);  view_287 = iota_9 = None
        add_79: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_277, index_7);  view_277 = index_7 = None
        add_80: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_79, 0);  add_79 = None
        mul_79: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_80, 0.125)
        convert_element_type_73: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_79, torch.float32);  mul_79 = None
        eq_7: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_73, convert_element_type_73)
        abs_8: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_73)
        ne_7: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_8, inf);  abs_8 = None
        mul_80: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_7, ne_7);  eq_7 = ne_7 = None
        logical_not_14: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_80);  mul_80 = None
        any_8: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_14, [3], True);  logical_not_14 = None
        logical_not_15: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        convert_element_type_74: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_80, torch.float32);  add_80 = None
        mul_81: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1);  convert_element_type_74 = None
        amax_14: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_81, [3], True)
        sub_28: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_81, amax_14);  mul_81 = amax_14 = None
        mul_82: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_28, 0.125);  sub_28 = None
        amax_15: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_73, [3], True)
        sub_29: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_73, amax_15);  convert_element_type_73 = amax_15 = None
        where_7: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_15, mul_82, sub_29);  logical_not_15 = mul_82 = sub_29 = None
        exp_7: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_7);  where_7 = None
        sum_8: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [3], True)
        div_8: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_75: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        unsqueeze_232: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_75, 4);  convert_element_type_75 = None
        view_288: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_232, _shape_param_289);  unsqueeze_232 = _shape_param_289 = None
        unsqueeze_233: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_71, 3)
        unsqueeze_234: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 4);  unsqueeze_233 = None
        view_289: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_234, _shape_param_290);  unsqueeze_234 = _shape_param_290 = None
        squeeze_76: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_289, 0);  view_289 = None
        unsqueeze_235: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg112_1, 3);  arg112_1 = None
        unsqueeze_236: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 4);  unsqueeze_235 = None
        view_290: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_236, _shape_param_291);  unsqueeze_236 = _shape_param_291 = None
        squeeze_77: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_290, 0);  view_290 = None
        mm_38: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_76, squeeze_77);  squeeze_76 = squeeze_77 = None
        unsqueeze_237: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_38, 0);  mm_38 = None
        view_291: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_237, _shape_param_292);  unsqueeze_237 = _shape_param_292 = None
        permute_184: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_291, [0, 1, 3, 4, 2]);  view_291 = None
        view_292: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_184, _shape_param_293);  permute_184 = _shape_param_293 = None
        unsqueeze_238: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_292, 4);  view_292 = None
        permute_185: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_238, [4, 1, 2, 3, 0]);  unsqueeze_238 = None
        permute_186: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_185, [1, 2, 4, 3, 0]);  permute_185 = None
        view_293: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_186, _shape_param_294);  permute_186 = _shape_param_294 = None
        bmm_23: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_288, view_293);  view_288 = view_293 = None
        view_294: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_295);  bmm_23 = _shape_param_295 = None
        permute_187: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_294, [2, 0, 1, 4, 3]);  view_294 = None
        view_295: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_187, _shape_param_296);  permute_187 = _shape_param_296 = None
        unsqueeze_239: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_295, 4);  view_295 = None
        permute_188: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_239, [0, 1, 4, 3, 2]);  unsqueeze_239 = None
        permute_189: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_188, [0, 1, 3, 4, 2]);  permute_188 = None
        clone_16: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_296: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_16, _shape_param_297);  clone_16 = _shape_param_297 = None
        squeeze_78: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_296, 0);  view_296 = None
        unsqueeze_240: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg113_1, 3);  arg113_1 = None
        unsqueeze_241: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 4);  unsqueeze_240 = None
        permute_190: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_241, [3, 4, 0, 2, 1]);  unsqueeze_241 = None
        permute_191: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_190, [3, 4, 2, 0, 1]);  permute_190 = None
        clone_17: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
        view_297: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_17, _shape_param_298);  clone_17 = _shape_param_298 = None
        squeeze_79: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_297, 0);  view_297 = None
        mm_39: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_78, squeeze_79);  squeeze_78 = squeeze_79 = None
        unsqueeze_242: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_39, 0);  mm_39 = None
        view_298: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_242, _shape_param_299);  unsqueeze_242 = _shape_param_299 = None
        permute_192: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_298, [0, 1, 4, 2, 3]);  view_298 = None
        view_299: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_192, _shape_param_300);  permute_192 = _shape_param_300 = None
        add_81: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_299, convert_element_type_71);  view_299 = None
        convert_element_type_76: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_76, [2], correction = 0, keepdim = True)
        getitem_28: "f32[512, 16, 1]" = var_mean_14[0]
        getitem_29: "f32[512, 16, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_30: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_76, getitem_29);  convert_element_type_76 = getitem_29 = None
        add_82: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_83: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_14);  sub_30 = rsqrt_14 = None
        mul_84: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_83, arg114_1);  mul_83 = arg114_1 = None
        add_83: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_84, arg115_1);  mul_84 = arg115_1 = None
        convert_element_type_77: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        view_300: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_77, _shape_param_301);  _shape_param_301 = None
        permute_193: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
        addmm_14: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg117_1, view_300, permute_193);  arg117_1 = view_300 = permute_193 = None
        view_301: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_14, _shape_param_302);  addmm_14 = _shape_param_302 = None
        convert_element_type_78: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_301, torch.float32);  view_301 = None
        mul_85: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.5)
        mul_86: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 0.7071067811865476);  convert_element_type_78 = None
        erf_7: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_84: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_87: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_85, add_84);  mul_85 = add_84 = None
        convert_element_type_79: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None
        view_302: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_79, _shape_param_303);  convert_element_type_79 = _shape_param_303 = None
        permute_194: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
        addmm_15: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg119_1, view_302, permute_194);  arg119_1 = view_302 = permute_194 = None
        view_303: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_15, _shape_param_304);  addmm_15 = _shape_param_304 = None
        add_85: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_303, convert_element_type_77);  view_303 = convert_element_type_77 = None
        convert_element_type_80: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_85, torch.float32);  add_85 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_30: "f32[512, 16, 1]" = var_mean_15[0]
        getitem_31: "f32[512, 16, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_31: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_31);  convert_element_type_80 = getitem_31 = None
        add_86: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        mul_88: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_15);  sub_31 = rsqrt_15 = None
        mul_89: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_88, arg120_1);  mul_88 = arg120_1 = None
        add_87: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_89, arg121_1);  mul_89 = arg121_1 = None
        convert_element_type_81: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None
        unsqueeze_243: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_81, 3)
        unsqueeze_244: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 4);  unsqueeze_243 = None
        view_304: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_244, _shape_param_305);  unsqueeze_244 = _shape_param_305 = None
        squeeze_80: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_304, 0);  view_304 = None
        unsqueeze_245: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg122_1, 3);  arg122_1 = None
        unsqueeze_246: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 4);  unsqueeze_245 = None
        view_305: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_246, _shape_param_306);  unsqueeze_246 = _shape_param_306 = None
        squeeze_81: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_305, 0);  view_305 = None
        mm_40: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_80, squeeze_81);  squeeze_80 = squeeze_81 = None
        unsqueeze_247: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_40, 0);  mm_40 = None
        view_306: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_247, _shape_param_307);  unsqueeze_247 = _shape_param_307 = None
        permute_195: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_306, [0, 1, 3, 4, 2]);  view_306 = None
        view_307: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_195, _shape_param_308);  permute_195 = _shape_param_308 = None
        add_88: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_307, arg123_1);  arg123_1 = None
        unsqueeze_248: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_88, 4);  add_88 = None
        permute_196: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_248, [1, 2, 0, 4, 3]);  unsqueeze_248 = None
        permute_197: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_196, [0, 1, 2, 4, 3]);  permute_196 = None
        view_308: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_197, _shape_param_309);  permute_197 = _shape_param_309 = None
        unsqueeze_249: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_81, 3)
        unsqueeze_250: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 4);  unsqueeze_249 = None
        view_309: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_250, _shape_param_310);  unsqueeze_250 = _shape_param_310 = None
        squeeze_82: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_309, 0);  view_309 = None
        unsqueeze_251: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, 3);  arg124_1 = None
        unsqueeze_252: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 4);  unsqueeze_251 = None
        view_310: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_252, _shape_param_311);  unsqueeze_252 = _shape_param_311 = None
        squeeze_83: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_310, 0);  view_310 = None
        mm_41: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_82, squeeze_83);  squeeze_82 = squeeze_83 = None
        unsqueeze_253: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_41, 0);  mm_41 = None
        view_311: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_253, _shape_param_312);  unsqueeze_253 = _shape_param_312 = None
        permute_198: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_311, [0, 1, 3, 4, 2]);  view_311 = None
        view_312: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_198, _shape_param_313);  permute_198 = _shape_param_313 = None
        unsqueeze_254: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_312, 4);  view_312 = None
        permute_199: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_254, [1, 2, 4, 0, 3]);  unsqueeze_254 = None
        permute_200: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_199, [0, 1, 4, 3, 2]);  permute_199 = None
        view_313: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_200, _shape_param_314);  permute_200 = _shape_param_314 = None
        bmm_24: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_308, view_313);  view_308 = view_313 = None
        view_314: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_24, _shape_param_315);  bmm_24 = _shape_param_315 = None
        permute_201: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_314, [0, 1, 2, 4, 3]);  view_314 = None
        view_315: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_201, _shape_param_316);  permute_201 = _shape_param_316 = None
        add_89: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_307, arg125_1);  view_307 = arg125_1 = None
        unsqueeze_255: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_89, 4);  add_89 = None
        permute_202: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_255, [1, 2, 0, 4, 3]);  unsqueeze_255 = None
        permute_203: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_202, [0, 1, 2, 4, 3]);  permute_202 = None
        view_316: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_203, _shape_param_317);  permute_203 = _shape_param_317 = None
        convert_element_type_82: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_256: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_82, 3);  convert_element_type_82 = None
        unsqueeze_257: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 4);  unsqueeze_256 = None
        view_317: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_257, _shape_param_318);  unsqueeze_257 = _shape_param_318 = None
        squeeze_84: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_317, 0);  view_317 = None
        unsqueeze_258: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg126_1, 3);  arg126_1 = None
        unsqueeze_259: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 4);  unsqueeze_258 = None
        view_318: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_259, _shape_param_319);  unsqueeze_259 = _shape_param_319 = None
        squeeze_85: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_318, 0);  view_318 = None
        mm_42: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_84, squeeze_85);  squeeze_84 = squeeze_85 = None
        unsqueeze_260: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_42, 0);  mm_42 = None
        view_319: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_260, _shape_param_320);  unsqueeze_260 = _shape_param_320 = None
        permute_204: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_319, [0, 1, 3, 4, 2]);  view_319 = None
        view_320: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_204, _shape_param_321);  permute_204 = _shape_param_321 = None
        unsqueeze_261: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_320, 4);  view_320 = None
        permute_205: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_261, [1, 2, 4, 0, 3]);  unsqueeze_261 = None
        permute_206: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_205, [0, 1, 4, 3, 2]);  permute_205 = None
        view_321: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_206, _shape_param_322);  permute_206 = _shape_param_322 = None
        bmm_25: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_316, view_321);  view_316 = view_321 = None
        view_322: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_25, _shape_param_323);  bmm_25 = _shape_param_323 = None
        permute_207: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_322, [0, 1, 2, 4, 3]);  view_322 = None
        view_323: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_207, _shape_param_324);  permute_207 = _shape_param_324 = None
        view_324: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_323, _shape_param_325);  view_323 = _shape_param_325 = None
        slice_9: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_324, 2, 1, 9223372036854775807);  view_324 = None
        view_325: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_9, _shape_param_326);  slice_9 = _shape_param_326 = None
        iota_10: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_8: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_325, [None, None, None, iota_10]);  view_325 = iota_10 = None
        add_90: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_315, index_8);  view_315 = index_8 = None
        add_91: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_90, 0);  add_90 = None
        mul_90: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_91, 0.125)
        convert_element_type_83: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_90, torch.float32);  mul_90 = None
        eq_8: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_83, convert_element_type_83)
        abs_9: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_83)
        ne_8: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_9, inf);  abs_9 = None
        mul_91: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_8, ne_8);  eq_8 = ne_8 = None
        logical_not_16: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_91);  mul_91 = None
        any_9: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_16, [3], True);  logical_not_16 = None
        logical_not_17: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        convert_element_type_84: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32);  add_91 = None
        mul_92: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1);  convert_element_type_84 = None
        amax_16: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_92, [3], True)
        sub_32: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_92, amax_16);  mul_92 = amax_16 = None
        mul_93: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_32, 0.125);  sub_32 = None
        amax_17: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_83, [3], True)
        sub_33: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_83, amax_17);  convert_element_type_83 = amax_17 = None
        where_8: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_17, mul_93, sub_33);  logical_not_17 = mul_93 = sub_33 = None
        exp_8: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_8);  where_8 = None
        sum_9: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [3], True)
        div_9: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_85: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        unsqueeze_262: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_85, 4);  convert_element_type_85 = None
        view_326: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_262, _shape_param_327);  unsqueeze_262 = _shape_param_327 = None
        unsqueeze_263: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_81, 3)
        unsqueeze_264: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 4);  unsqueeze_263 = None
        view_327: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_264, _shape_param_328);  unsqueeze_264 = _shape_param_328 = None
        squeeze_86: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_327, 0);  view_327 = None
        unsqueeze_265: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg127_1, 3);  arg127_1 = None
        unsqueeze_266: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 4);  unsqueeze_265 = None
        view_328: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_266, _shape_param_329);  unsqueeze_266 = _shape_param_329 = None
        squeeze_87: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_328, 0);  view_328 = None
        mm_43: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_86, squeeze_87);  squeeze_86 = squeeze_87 = None
        unsqueeze_267: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_43, 0);  mm_43 = None
        view_329: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_267, _shape_param_330);  unsqueeze_267 = _shape_param_330 = None
        permute_208: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_329, [0, 1, 3, 4, 2]);  view_329 = None
        view_330: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_208, _shape_param_331);  permute_208 = _shape_param_331 = None
        unsqueeze_268: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_330, 4);  view_330 = None
        permute_209: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_268, [4, 1, 2, 3, 0]);  unsqueeze_268 = None
        permute_210: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_209, [1, 2, 4, 3, 0]);  permute_209 = None
        view_331: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_210, _shape_param_332);  permute_210 = _shape_param_332 = None
        bmm_26: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_326, view_331);  view_326 = view_331 = None
        view_332: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_26, _shape_param_333);  bmm_26 = _shape_param_333 = None
        permute_211: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_332, [2, 0, 1, 4, 3]);  view_332 = None
        view_333: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_211, _shape_param_334);  permute_211 = _shape_param_334 = None
        unsqueeze_269: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_333, 4);  view_333 = None
        permute_212: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_269, [0, 1, 4, 3, 2]);  unsqueeze_269 = None
        permute_213: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_212, [0, 1, 3, 4, 2]);  permute_212 = None
        clone_18: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_334: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_18, _shape_param_335);  clone_18 = _shape_param_335 = None
        squeeze_88: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_334, 0);  view_334 = None
        unsqueeze_270: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg128_1, 3);  arg128_1 = None
        unsqueeze_271: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 4);  unsqueeze_270 = None
        permute_214: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_271, [3, 4, 0, 2, 1]);  unsqueeze_271 = None
        permute_215: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_214, [3, 4, 2, 0, 1]);  permute_214 = None
        clone_19: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_335: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_19, _shape_param_336);  clone_19 = _shape_param_336 = None
        squeeze_89: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_335, 0);  view_335 = None
        mm_44: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_88, squeeze_89);  squeeze_88 = squeeze_89 = None
        unsqueeze_272: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_44, 0);  mm_44 = None
        view_336: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_272, _shape_param_337);  unsqueeze_272 = _shape_param_337 = None
        permute_216: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_336, [0, 1, 4, 2, 3]);  view_336 = None
        view_337: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_216, _shape_param_338);  permute_216 = _shape_param_338 = None
        add_92: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_337, convert_element_type_81);  view_337 = None
        convert_element_type_86: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_92, torch.float32);  add_92 = None
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_86, [2], correction = 0, keepdim = True)
        getitem_32: "f32[512, 16, 1]" = var_mean_16[0]
        getitem_33: "f32[512, 16, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_34: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_86, getitem_33);  convert_element_type_86 = getitem_33 = None
        add_93: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_94: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_16);  sub_34 = rsqrt_16 = None
        mul_95: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_94, arg129_1);  mul_94 = arg129_1 = None
        add_94: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_95, arg130_1);  mul_95 = arg130_1 = None
        convert_element_type_87: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None
        view_338: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_87, _shape_param_339);  _shape_param_339 = None
        permute_217: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_16: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg132_1, view_338, permute_217);  arg132_1 = view_338 = permute_217 = None
        view_339: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_16, _shape_param_340);  addmm_16 = _shape_param_340 = None
        convert_element_type_88: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        mul_96: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.5)
        mul_97: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.7071067811865476);  convert_element_type_88 = None
        erf_8: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_95: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_98: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_96, add_95);  mul_96 = add_95 = None
        convert_element_type_89: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_98, torch.bfloat16);  mul_98 = None
        view_340: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_341);  convert_element_type_89 = _shape_param_341 = None
        permute_218: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_17: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg134_1, view_340, permute_218);  arg134_1 = view_340 = permute_218 = None
        view_341: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_17, _shape_param_342);  addmm_17 = _shape_param_342 = None
        add_96: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_341, convert_element_type_87);  view_341 = convert_element_type_87 = None
        convert_element_type_90: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_96, torch.float32);  add_96 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_34: "f32[512, 16, 1]" = var_mean_17[0]
        getitem_35: "f32[512, 16, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_35: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_35);  convert_element_type_90 = getitem_35 = None
        add_97: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        mul_99: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_17);  sub_35 = rsqrt_17 = None
        mul_100: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_99, arg135_1);  mul_99 = arg135_1 = None
        add_98: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_100, arg136_1);  mul_100 = arg136_1 = None
        convert_element_type_91: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16);  add_98 = None
        unsqueeze_273: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_91, 3)
        unsqueeze_274: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 4);  unsqueeze_273 = None
        view_342: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_274, _shape_param_343);  unsqueeze_274 = _shape_param_343 = None
        squeeze_90: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_342, 0);  view_342 = None
        unsqueeze_275: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg137_1, 3);  arg137_1 = None
        unsqueeze_276: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 4);  unsqueeze_275 = None
        view_343: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_276, _shape_param_344);  unsqueeze_276 = _shape_param_344 = None
        squeeze_91: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_343, 0);  view_343 = None
        mm_45: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_90, squeeze_91);  squeeze_90 = squeeze_91 = None
        unsqueeze_277: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_45, 0);  mm_45 = None
        view_344: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_277, _shape_param_345);  unsqueeze_277 = _shape_param_345 = None
        permute_219: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_344, [0, 1, 3, 4, 2]);  view_344 = None
        view_345: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_219, _shape_param_346);  permute_219 = _shape_param_346 = None
        add_99: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_345, arg138_1);  arg138_1 = None
        unsqueeze_278: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_99, 4);  add_99 = None
        permute_220: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_278, [1, 2, 0, 4, 3]);  unsqueeze_278 = None
        permute_221: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_220, [0, 1, 2, 4, 3]);  permute_220 = None
        view_346: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_221, _shape_param_347);  permute_221 = _shape_param_347 = None
        unsqueeze_279: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_91, 3)
        unsqueeze_280: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 4);  unsqueeze_279 = None
        view_347: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_280, _shape_param_348);  unsqueeze_280 = _shape_param_348 = None
        squeeze_92: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_347, 0);  view_347 = None
        unsqueeze_281: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, 3);  arg139_1 = None
        unsqueeze_282: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 4);  unsqueeze_281 = None
        view_348: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_282, _shape_param_349);  unsqueeze_282 = _shape_param_349 = None
        squeeze_93: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_348, 0);  view_348 = None
        mm_46: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_92, squeeze_93);  squeeze_92 = squeeze_93 = None
        unsqueeze_283: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_46, 0);  mm_46 = None
        view_349: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_283, _shape_param_350);  unsqueeze_283 = _shape_param_350 = None
        permute_222: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_349, [0, 1, 3, 4, 2]);  view_349 = None
        view_350: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_222, _shape_param_351);  permute_222 = _shape_param_351 = None
        unsqueeze_284: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_350, 4);  view_350 = None
        permute_223: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_284, [1, 2, 4, 0, 3]);  unsqueeze_284 = None
        permute_224: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_223, [0, 1, 4, 3, 2]);  permute_223 = None
        view_351: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_224, _shape_param_352);  permute_224 = _shape_param_352 = None
        bmm_27: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_346, view_351);  view_346 = view_351 = None
        view_352: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_27, _shape_param_353);  bmm_27 = _shape_param_353 = None
        permute_225: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_352, [0, 1, 2, 4, 3]);  view_352 = None
        view_353: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_225, _shape_param_354);  permute_225 = _shape_param_354 = None
        add_100: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_345, arg140_1);  view_345 = arg140_1 = None
        unsqueeze_285: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_100, 4);  add_100 = None
        permute_226: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_285, [1, 2, 0, 4, 3]);  unsqueeze_285 = None
        permute_227: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_226, [0, 1, 2, 4, 3]);  permute_226 = None
        view_354: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_227, _shape_param_355);  permute_227 = _shape_param_355 = None
        convert_element_type_92: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_286: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_92, 3);  convert_element_type_92 = None
        unsqueeze_287: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 4);  unsqueeze_286 = None
        view_355: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_287, _shape_param_356);  unsqueeze_287 = _shape_param_356 = None
        squeeze_94: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_355, 0);  view_355 = None
        unsqueeze_288: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg141_1, 3);  arg141_1 = None
        unsqueeze_289: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 4);  unsqueeze_288 = None
        view_356: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_289, _shape_param_357);  unsqueeze_289 = _shape_param_357 = None
        squeeze_95: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_356, 0);  view_356 = None
        mm_47: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_94, squeeze_95);  squeeze_94 = squeeze_95 = None
        unsqueeze_290: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_47, 0);  mm_47 = None
        view_357: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_290, _shape_param_358);  unsqueeze_290 = _shape_param_358 = None
        permute_228: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_357, [0, 1, 3, 4, 2]);  view_357 = None
        view_358: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_228, _shape_param_359);  permute_228 = _shape_param_359 = None
        unsqueeze_291: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_358, 4);  view_358 = None
        permute_229: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_291, [1, 2, 4, 0, 3]);  unsqueeze_291 = None
        permute_230: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_229, [0, 1, 4, 3, 2]);  permute_229 = None
        view_359: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_230, _shape_param_360);  permute_230 = _shape_param_360 = None
        bmm_28: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_354, view_359);  view_354 = view_359 = None
        view_360: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_28, _shape_param_361);  bmm_28 = _shape_param_361 = None
        permute_231: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_360, [0, 1, 2, 4, 3]);  view_360 = None
        view_361: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_231, _shape_param_362);  permute_231 = _shape_param_362 = None
        view_362: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_361, _shape_param_363);  view_361 = _shape_param_363 = None
        slice_10: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_362, 2, 1, 9223372036854775807);  view_362 = None
        view_363: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_10, _shape_param_364);  slice_10 = _shape_param_364 = None
        iota_11: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_9: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_363, [None, None, None, iota_11]);  view_363 = iota_11 = None
        add_101: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_353, index_9);  view_353 = index_9 = None
        add_102: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_101, 0);  add_101 = None
        mul_101: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_102, 0.125)
        convert_element_type_93: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_101, torch.float32);  mul_101 = None
        eq_9: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_93, convert_element_type_93)
        abs_10: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_93)
        ne_9: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_10, inf);  abs_10 = None
        mul_102: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_9, ne_9);  eq_9 = ne_9 = None
        logical_not_18: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_102);  mul_102 = None
        any_10: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_18, [3], True);  logical_not_18 = None
        logical_not_19: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        convert_element_type_94: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_102, torch.float32);  add_102 = None
        mul_103: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 1);  convert_element_type_94 = None
        amax_18: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_103, [3], True)
        sub_36: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_103, amax_18);  mul_103 = amax_18 = None
        mul_104: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_36, 0.125);  sub_36 = None
        amax_19: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_93, [3], True)
        sub_37: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_93, amax_19);  convert_element_type_93 = amax_19 = None
        where_9: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_19, mul_104, sub_37);  logical_not_19 = mul_104 = sub_37 = None
        exp_9: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_9);  where_9 = None
        sum_10: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [3], True)
        div_10: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_95: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        unsqueeze_292: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_95, 4);  convert_element_type_95 = None
        view_364: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_292, _shape_param_365);  unsqueeze_292 = _shape_param_365 = None
        unsqueeze_293: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_91, 3)
        unsqueeze_294: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 4);  unsqueeze_293 = None
        view_365: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_294, _shape_param_366);  unsqueeze_294 = _shape_param_366 = None
        squeeze_96: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_365, 0);  view_365 = None
        unsqueeze_295: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg142_1, 3);  arg142_1 = None
        unsqueeze_296: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 4);  unsqueeze_295 = None
        view_366: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_296, _shape_param_367);  unsqueeze_296 = _shape_param_367 = None
        squeeze_97: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_366, 0);  view_366 = None
        mm_48: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_96, squeeze_97);  squeeze_96 = squeeze_97 = None
        unsqueeze_297: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_48, 0);  mm_48 = None
        view_367: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_297, _shape_param_368);  unsqueeze_297 = _shape_param_368 = None
        permute_232: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_367, [0, 1, 3, 4, 2]);  view_367 = None
        view_368: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_232, _shape_param_369);  permute_232 = _shape_param_369 = None
        unsqueeze_298: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_368, 4);  view_368 = None
        permute_233: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_298, [4, 1, 2, 3, 0]);  unsqueeze_298 = None
        permute_234: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_233, [1, 2, 4, 3, 0]);  permute_233 = None
        view_369: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_234, _shape_param_370);  permute_234 = _shape_param_370 = None
        bmm_29: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_364, view_369);  view_364 = view_369 = None
        view_370: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_29, _shape_param_371);  bmm_29 = _shape_param_371 = None
        permute_235: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_370, [2, 0, 1, 4, 3]);  view_370 = None
        view_371: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_235, _shape_param_372);  permute_235 = _shape_param_372 = None
        unsqueeze_299: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_371, 4);  view_371 = None
        permute_236: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_299, [0, 1, 4, 3, 2]);  unsqueeze_299 = None
        permute_237: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 4, 2]);  permute_236 = None
        clone_20: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_372: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_20, _shape_param_373);  clone_20 = _shape_param_373 = None
        squeeze_98: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_372, 0);  view_372 = None
        unsqueeze_300: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg143_1, 3);  arg143_1 = None
        unsqueeze_301: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 4);  unsqueeze_300 = None
        permute_238: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_301, [3, 4, 0, 2, 1]);  unsqueeze_301 = None
        permute_239: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_238, [3, 4, 2, 0, 1]);  permute_238 = None
        clone_21: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_373: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_21, _shape_param_374);  clone_21 = _shape_param_374 = None
        squeeze_99: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_373, 0);  view_373 = None
        mm_49: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_98, squeeze_99);  squeeze_98 = squeeze_99 = None
        unsqueeze_302: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_49, 0);  mm_49 = None
        view_374: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_302, _shape_param_375);  unsqueeze_302 = _shape_param_375 = None
        permute_240: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_374, [0, 1, 4, 2, 3]);  view_374 = None
        view_375: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_240, _shape_param_376);  permute_240 = _shape_param_376 = None
        add_103: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_375, convert_element_type_91);  view_375 = None
        convert_element_type_96: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_103, torch.float32);  add_103 = None
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_96, [2], correction = 0, keepdim = True)
        getitem_36: "f32[512, 16, 1]" = var_mean_18[0]
        getitem_37: "f32[512, 16, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_38: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_96, getitem_37);  convert_element_type_96 = getitem_37 = None
        add_104: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_105: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_18);  sub_38 = rsqrt_18 = None
        mul_106: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_105, arg144_1);  mul_105 = arg144_1 = None
        add_105: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_106, arg145_1);  mul_106 = arg145_1 = None
        convert_element_type_97: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16);  add_105 = None
        view_376: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_97, _shape_param_377);  _shape_param_377 = None
        permute_241: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
        addmm_18: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg147_1, view_376, permute_241);  arg147_1 = view_376 = permute_241 = None
        view_377: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_18, _shape_param_378);  addmm_18 = _shape_param_378 = None
        convert_element_type_98: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None
        mul_107: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 0.5)
        mul_108: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 0.7071067811865476);  convert_element_type_98 = None
        erf_9: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
        add_106: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_109: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_107, add_106);  mul_107 = add_106 = None
        convert_element_type_99: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16);  mul_109 = None
        view_378: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_99, _shape_param_379);  convert_element_type_99 = _shape_param_379 = None
        permute_242: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
        addmm_19: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg149_1, view_378, permute_242);  arg149_1 = view_378 = permute_242 = None
        view_379: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_19, _shape_param_380);  addmm_19 = _shape_param_380 = None
        add_107: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_379, convert_element_type_97);  view_379 = convert_element_type_97 = None
        convert_element_type_100: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_107, torch.float32);  add_107 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_100, [2], correction = 0, keepdim = True)
        getitem_38: "f32[512, 16, 1]" = var_mean_19[0]
        getitem_39: "f32[512, 16, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_39: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_100, getitem_39);  convert_element_type_100 = getitem_39 = None
        add_108: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        mul_110: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_19);  sub_39 = rsqrt_19 = None
        mul_111: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_110, arg150_1);  mul_110 = arg150_1 = None
        add_109: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_111, arg151_1);  mul_111 = arg151_1 = None
        convert_element_type_101: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None
        unsqueeze_303: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_101, 3)
        unsqueeze_304: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 4);  unsqueeze_303 = None
        view_380: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_304, _shape_param_381);  unsqueeze_304 = _shape_param_381 = None
        squeeze_100: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_380, 0);  view_380 = None
        unsqueeze_305: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg152_1, 3);  arg152_1 = None
        unsqueeze_306: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 4);  unsqueeze_305 = None
        view_381: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_306, _shape_param_382);  unsqueeze_306 = _shape_param_382 = None
        squeeze_101: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_381, 0);  view_381 = None
        mm_50: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_100, squeeze_101);  squeeze_100 = squeeze_101 = None
        unsqueeze_307: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_50, 0);  mm_50 = None
        view_382: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_307, _shape_param_383);  unsqueeze_307 = _shape_param_383 = None
        permute_243: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_382, [0, 1, 3, 4, 2]);  view_382 = None
        view_383: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_243, _shape_param_384);  permute_243 = _shape_param_384 = None
        add_110: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_383, arg153_1);  arg153_1 = None
        unsqueeze_308: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_110, 4);  add_110 = None
        permute_244: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_308, [1, 2, 0, 4, 3]);  unsqueeze_308 = None
        permute_245: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_244, [0, 1, 2, 4, 3]);  permute_244 = None
        view_384: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_245, _shape_param_385);  permute_245 = _shape_param_385 = None
        unsqueeze_309: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_101, 3)
        unsqueeze_310: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 4);  unsqueeze_309 = None
        view_385: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_310, _shape_param_386);  unsqueeze_310 = _shape_param_386 = None
        squeeze_102: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_385, 0);  view_385 = None
        unsqueeze_311: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, 3);  arg154_1 = None
        unsqueeze_312: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 4);  unsqueeze_311 = None
        view_386: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_312, _shape_param_387);  unsqueeze_312 = _shape_param_387 = None
        squeeze_103: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_386, 0);  view_386 = None
        mm_51: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_102, squeeze_103);  squeeze_102 = squeeze_103 = None
        unsqueeze_313: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_51, 0);  mm_51 = None
        view_387: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_313, _shape_param_388);  unsqueeze_313 = _shape_param_388 = None
        permute_246: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_387, [0, 1, 3, 4, 2]);  view_387 = None
        view_388: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_246, _shape_param_389);  permute_246 = _shape_param_389 = None
        unsqueeze_314: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_388, 4);  view_388 = None
        permute_247: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_314, [1, 2, 4, 0, 3]);  unsqueeze_314 = None
        permute_248: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_247, [0, 1, 4, 3, 2]);  permute_247 = None
        view_389: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_248, _shape_param_390);  permute_248 = _shape_param_390 = None
        bmm_30: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_384, view_389);  view_384 = view_389 = None
        view_390: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_30, _shape_param_391);  bmm_30 = _shape_param_391 = None
        permute_249: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_390, [0, 1, 2, 4, 3]);  view_390 = None
        view_391: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_249, _shape_param_392);  permute_249 = _shape_param_392 = None
        add_111: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_383, arg155_1);  view_383 = arg155_1 = None
        unsqueeze_315: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_111, 4);  add_111 = None
        permute_250: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_315, [1, 2, 0, 4, 3]);  unsqueeze_315 = None
        permute_251: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_250, [0, 1, 2, 4, 3]);  permute_250 = None
        view_392: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_251, _shape_param_393);  permute_251 = _shape_param_393 = None
        convert_element_type_102: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_316: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, 3);  convert_element_type_102 = None
        unsqueeze_317: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 4);  unsqueeze_316 = None
        view_393: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_317, _shape_param_394);  unsqueeze_317 = _shape_param_394 = None
        squeeze_104: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_393, 0);  view_393 = None
        unsqueeze_318: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg156_1, 3);  arg156_1 = None
        unsqueeze_319: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 4);  unsqueeze_318 = None
        view_394: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_319, _shape_param_395);  unsqueeze_319 = _shape_param_395 = None
        squeeze_105: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_394, 0);  view_394 = None
        mm_52: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_104, squeeze_105);  squeeze_104 = squeeze_105 = None
        unsqueeze_320: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_52, 0);  mm_52 = None
        view_395: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_320, _shape_param_396);  unsqueeze_320 = _shape_param_396 = None
        permute_252: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_395, [0, 1, 3, 4, 2]);  view_395 = None
        view_396: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_252, _shape_param_397);  permute_252 = _shape_param_397 = None
        unsqueeze_321: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_396, 4);  view_396 = None
        permute_253: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_321, [1, 2, 4, 0, 3]);  unsqueeze_321 = None
        permute_254: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_253, [0, 1, 4, 3, 2]);  permute_253 = None
        view_397: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_254, _shape_param_398);  permute_254 = _shape_param_398 = None
        bmm_31: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_392, view_397);  view_392 = view_397 = None
        view_398: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_31, _shape_param_399);  bmm_31 = _shape_param_399 = None
        permute_255: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_398, [0, 1, 2, 4, 3]);  view_398 = None
        view_399: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_255, _shape_param_400);  permute_255 = _shape_param_400 = None
        view_400: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_399, _shape_param_401);  view_399 = _shape_param_401 = None
        slice_11: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_400, 2, 1, 9223372036854775807);  view_400 = None
        view_401: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_11, _shape_param_402);  slice_11 = _shape_param_402 = None
        iota_12: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_10: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_401, [None, None, None, iota_12]);  view_401 = iota_12 = None
        add_112: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_391, index_10);  view_391 = index_10 = None
        add_113: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_112, 0);  add_112 = None
        mul_112: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_113, 0.125)
        convert_element_type_103: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_112, torch.float32);  mul_112 = None
        eq_10: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_103, convert_element_type_103)
        abs_11: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_103)
        ne_10: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_11, inf);  abs_11 = None
        mul_113: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_10, ne_10);  eq_10 = ne_10 = None
        logical_not_20: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_113);  mul_113 = None
        any_11: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_20, [3], True);  logical_not_20 = None
        logical_not_21: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        convert_element_type_104: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_113, torch.float32);  add_113 = None
        mul_114: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_104, 1);  convert_element_type_104 = None
        amax_20: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_114, [3], True)
        sub_40: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_114, amax_20);  mul_114 = amax_20 = None
        mul_115: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_40, 0.125);  sub_40 = None
        amax_21: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_103, [3], True)
        sub_41: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_103, amax_21);  convert_element_type_103 = amax_21 = None
        where_10: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_21, mul_115, sub_41);  logical_not_21 = mul_115 = sub_41 = None
        exp_10: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_10);  where_10 = None
        sum_11: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [3], True)
        div_11: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_105: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        unsqueeze_322: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_105, 4);  convert_element_type_105 = None
        view_402: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_322, _shape_param_403);  unsqueeze_322 = _shape_param_403 = None
        unsqueeze_323: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_101, 3)
        unsqueeze_324: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 4);  unsqueeze_323 = None
        view_403: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_324, _shape_param_404);  unsqueeze_324 = _shape_param_404 = None
        squeeze_106: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_403, 0);  view_403 = None
        unsqueeze_325: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg157_1, 3);  arg157_1 = None
        unsqueeze_326: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 4);  unsqueeze_325 = None
        view_404: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_326, _shape_param_405);  unsqueeze_326 = _shape_param_405 = None
        squeeze_107: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_404, 0);  view_404 = None
        mm_53: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_106, squeeze_107);  squeeze_106 = squeeze_107 = None
        unsqueeze_327: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_53, 0);  mm_53 = None
        view_405: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_327, _shape_param_406);  unsqueeze_327 = _shape_param_406 = None
        permute_256: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_405, [0, 1, 3, 4, 2]);  view_405 = None
        view_406: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_256, _shape_param_407);  permute_256 = _shape_param_407 = None
        unsqueeze_328: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_406, 4);  view_406 = None
        permute_257: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_328, [4, 1, 2, 3, 0]);  unsqueeze_328 = None
        permute_258: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_257, [1, 2, 4, 3, 0]);  permute_257 = None
        view_407: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_258, _shape_param_408);  permute_258 = _shape_param_408 = None
        bmm_32: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_402, view_407);  view_402 = view_407 = None
        view_408: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_32, _shape_param_409);  bmm_32 = _shape_param_409 = None
        permute_259: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_408, [2, 0, 1, 4, 3]);  view_408 = None
        view_409: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_259, _shape_param_410);  permute_259 = _shape_param_410 = None
        unsqueeze_329: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_409, 4);  view_409 = None
        permute_260: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_329, [0, 1, 4, 3, 2]);  unsqueeze_329 = None
        permute_261: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_260, [0, 1, 3, 4, 2]);  permute_260 = None
        clone_22: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_410: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_22, _shape_param_411);  clone_22 = _shape_param_411 = None
        squeeze_108: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_410, 0);  view_410 = None
        unsqueeze_330: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg158_1, 3);  arg158_1 = None
        unsqueeze_331: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 4);  unsqueeze_330 = None
        permute_262: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_331, [3, 4, 0, 2, 1]);  unsqueeze_331 = None
        permute_263: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_262, [3, 4, 2, 0, 1]);  permute_262 = None
        clone_23: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        view_411: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_23, _shape_param_412);  clone_23 = _shape_param_412 = None
        squeeze_109: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_411, 0);  view_411 = None
        mm_54: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_108, squeeze_109);  squeeze_108 = squeeze_109 = None
        unsqueeze_332: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_54, 0);  mm_54 = None
        view_412: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_332, _shape_param_413);  unsqueeze_332 = _shape_param_413 = None
        permute_264: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_412, [0, 1, 4, 2, 3]);  view_412 = None
        view_413: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_264, _shape_param_414);  permute_264 = _shape_param_414 = None
        add_114: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_413, convert_element_type_101);  view_413 = None
        convert_element_type_106: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_106, [2], correction = 0, keepdim = True)
        getitem_40: "f32[512, 16, 1]" = var_mean_20[0]
        getitem_41: "f32[512, 16, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_42: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_106, getitem_41);  convert_element_type_106 = getitem_41 = None
        add_115: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_116: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_20);  sub_42 = rsqrt_20 = None
        mul_117: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_116, arg159_1);  mul_116 = arg159_1 = None
        add_116: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_117, arg160_1);  mul_117 = arg160_1 = None
        convert_element_type_107: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16);  add_116 = None
        view_414: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_107, _shape_param_415);  _shape_param_415 = None
        permute_265: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_20: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg162_1, view_414, permute_265);  arg162_1 = view_414 = permute_265 = None
        view_415: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_20, _shape_param_416);  addmm_20 = _shape_param_416 = None
        convert_element_type_108: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_118: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_108, 0.5)
        mul_119: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_108, 0.7071067811865476);  convert_element_type_108 = None
        erf_10: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_117: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_120: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_118, add_117);  mul_118 = add_117 = None
        convert_element_type_109: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        view_416: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_109, _shape_param_417);  convert_element_type_109 = _shape_param_417 = None
        permute_266: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_21: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg164_1, view_416, permute_266);  arg164_1 = view_416 = permute_266 = None
        view_417: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_21, _shape_param_418);  addmm_21 = _shape_param_418 = None
        add_118: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_417, convert_element_type_107);  view_417 = convert_element_type_107 = None
        convert_element_type_110: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_118, torch.float32);  add_118 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_110, [2], correction = 0, keepdim = True)
        getitem_42: "f32[512, 16, 1]" = var_mean_21[0]
        getitem_43: "f32[512, 16, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_43: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_110, getitem_43);  convert_element_type_110 = getitem_43 = None
        add_119: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        mul_121: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_21);  sub_43 = rsqrt_21 = None
        mul_122: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_121, arg165_1);  mul_121 = arg165_1 = None
        add_120: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_122, arg166_1);  mul_122 = arg166_1 = None
        convert_element_type_111: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16);  add_120 = None
        unsqueeze_333: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_111, 3)
        unsqueeze_334: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 4);  unsqueeze_333 = None
        view_418: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_334, _shape_param_419);  unsqueeze_334 = _shape_param_419 = None
        squeeze_110: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_418, 0);  view_418 = None
        unsqueeze_335: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg167_1, 3);  arg167_1 = None
        unsqueeze_336: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 4);  unsqueeze_335 = None
        view_419: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_336, _shape_param_420);  unsqueeze_336 = _shape_param_420 = None
        squeeze_111: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_419, 0);  view_419 = None
        mm_55: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_110, squeeze_111);  squeeze_110 = squeeze_111 = None
        unsqueeze_337: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_55, 0);  mm_55 = None
        view_420: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_337, _shape_param_421);  unsqueeze_337 = _shape_param_421 = None
        permute_267: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_420, [0, 1, 3, 4, 2]);  view_420 = None
        view_421: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_267, _shape_param_422);  permute_267 = _shape_param_422 = None
        add_121: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_421, arg168_1);  arg168_1 = None
        unsqueeze_338: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_121, 4);  add_121 = None
        permute_268: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_338, [1, 2, 0, 4, 3]);  unsqueeze_338 = None
        permute_269: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_268, [0, 1, 2, 4, 3]);  permute_268 = None
        view_422: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_269, _shape_param_423);  permute_269 = _shape_param_423 = None
        unsqueeze_339: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_111, 3)
        unsqueeze_340: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 4);  unsqueeze_339 = None
        view_423: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_340, _shape_param_424);  unsqueeze_340 = _shape_param_424 = None
        squeeze_112: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_423, 0);  view_423 = None
        unsqueeze_341: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, 3);  arg169_1 = None
        unsqueeze_342: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 4);  unsqueeze_341 = None
        view_424: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_342, _shape_param_425);  unsqueeze_342 = _shape_param_425 = None
        squeeze_113: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_424, 0);  view_424 = None
        mm_56: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_112, squeeze_113);  squeeze_112 = squeeze_113 = None
        unsqueeze_343: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_56, 0);  mm_56 = None
        view_425: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_343, _shape_param_426);  unsqueeze_343 = _shape_param_426 = None
        permute_270: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_425, [0, 1, 3, 4, 2]);  view_425 = None
        view_426: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_270, _shape_param_427);  permute_270 = _shape_param_427 = None
        unsqueeze_344: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_426, 4);  view_426 = None
        permute_271: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_344, [1, 2, 4, 0, 3]);  unsqueeze_344 = None
        permute_272: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_271, [0, 1, 4, 3, 2]);  permute_271 = None
        view_427: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_272, _shape_param_428);  permute_272 = _shape_param_428 = None
        bmm_33: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_422, view_427);  view_422 = view_427 = None
        view_428: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_33, _shape_param_429);  bmm_33 = _shape_param_429 = None
        permute_273: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_428, [0, 1, 2, 4, 3]);  view_428 = None
        view_429: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_273, _shape_param_430);  permute_273 = _shape_param_430 = None
        add_122: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_421, arg170_1);  view_421 = arg170_1 = None
        unsqueeze_345: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_122, 4);  add_122 = None
        permute_274: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_345, [1, 2, 0, 4, 3]);  unsqueeze_345 = None
        permute_275: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_274, [0, 1, 2, 4, 3]);  permute_274 = None
        view_430: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_275, _shape_param_431);  permute_275 = _shape_param_431 = None
        convert_element_type_112: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_346: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_112, 3);  convert_element_type_112 = None
        unsqueeze_347: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 4);  unsqueeze_346 = None
        view_431: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_347, _shape_param_432);  unsqueeze_347 = _shape_param_432 = None
        squeeze_114: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_431, 0);  view_431 = None
        unsqueeze_348: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg171_1, 3);  arg171_1 = None
        unsqueeze_349: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 4);  unsqueeze_348 = None
        view_432: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_349, _shape_param_433);  unsqueeze_349 = _shape_param_433 = None
        squeeze_115: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_432, 0);  view_432 = None
        mm_57: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_114, squeeze_115);  squeeze_114 = squeeze_115 = None
        unsqueeze_350: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_57, 0);  mm_57 = None
        view_433: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_350, _shape_param_434);  unsqueeze_350 = _shape_param_434 = None
        permute_276: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_433, [0, 1, 3, 4, 2]);  view_433 = None
        view_434: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_276, _shape_param_435);  permute_276 = _shape_param_435 = None
        unsqueeze_351: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_434, 4);  view_434 = None
        permute_277: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_351, [1, 2, 4, 0, 3]);  unsqueeze_351 = None
        permute_278: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_277, [0, 1, 4, 3, 2]);  permute_277 = None
        view_435: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_278, _shape_param_436);  permute_278 = _shape_param_436 = None
        bmm_34: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_430, view_435);  view_430 = view_435 = None
        view_436: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_34, _shape_param_437);  bmm_34 = _shape_param_437 = None
        permute_279: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_436, [0, 1, 2, 4, 3]);  view_436 = None
        view_437: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_279, _shape_param_438);  permute_279 = _shape_param_438 = None
        view_438: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_437, _shape_param_439);  view_437 = _shape_param_439 = None
        slice_12: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_438, 2, 1, 9223372036854775807);  view_438 = None
        view_439: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_12, _shape_param_440);  slice_12 = _shape_param_440 = None
        iota_13: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_11: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_439, [None, None, None, iota_13]);  view_439 = iota_13 = None
        add_123: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_429, index_11);  view_429 = index_11 = None
        add_124: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_123, 0);  add_123 = None
        mul_123: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_124, 0.125)
        convert_element_type_113: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_123, torch.float32);  mul_123 = None
        eq_11: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_113, convert_element_type_113)
        abs_12: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_113)
        ne_11: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_12, inf);  abs_12 = None
        mul_124: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_11, ne_11);  eq_11 = ne_11 = None
        logical_not_22: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_124);  mul_124 = None
        any_12: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_22, [3], True);  logical_not_22 = None
        logical_not_23: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        convert_element_type_114: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_124, torch.float32);  add_124 = None
        mul_125: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_114, 1);  convert_element_type_114 = None
        amax_22: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_125, [3], True)
        sub_44: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_125, amax_22);  mul_125 = amax_22 = None
        mul_126: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_44, 0.125);  sub_44 = None
        amax_23: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_113, [3], True)
        sub_45: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_113, amax_23);  convert_element_type_113 = amax_23 = None
        where_11: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_23, mul_126, sub_45);  logical_not_23 = mul_126 = sub_45 = None
        exp_11: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_11);  where_11 = None
        sum_12: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [3], True)
        div_12: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_115: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        unsqueeze_352: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_115, 4);  convert_element_type_115 = None
        view_440: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_352, _shape_param_441);  unsqueeze_352 = _shape_param_441 = None
        unsqueeze_353: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_111, 3)
        unsqueeze_354: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 4);  unsqueeze_353 = None
        view_441: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_354, _shape_param_442);  unsqueeze_354 = _shape_param_442 = None
        squeeze_116: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_441, 0);  view_441 = None
        unsqueeze_355: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg172_1, 3);  arg172_1 = None
        unsqueeze_356: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 4);  unsqueeze_355 = None
        view_442: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_356, _shape_param_443);  unsqueeze_356 = _shape_param_443 = None
        squeeze_117: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_442, 0);  view_442 = None
        mm_58: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_116, squeeze_117);  squeeze_116 = squeeze_117 = None
        unsqueeze_357: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_58, 0);  mm_58 = None
        view_443: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_357, _shape_param_444);  unsqueeze_357 = _shape_param_444 = None
        permute_280: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_443, [0, 1, 3, 4, 2]);  view_443 = None
        view_444: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_280, _shape_param_445);  permute_280 = _shape_param_445 = None
        unsqueeze_358: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_444, 4);  view_444 = None
        permute_281: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_358, [4, 1, 2, 3, 0]);  unsqueeze_358 = None
        permute_282: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_281, [1, 2, 4, 3, 0]);  permute_281 = None
        view_445: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_282, _shape_param_446);  permute_282 = _shape_param_446 = None
        bmm_35: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_440, view_445);  view_440 = view_445 = None
        view_446: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_447);  bmm_35 = _shape_param_447 = None
        permute_283: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_446, [2, 0, 1, 4, 3]);  view_446 = None
        view_447: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_283, _shape_param_448);  permute_283 = _shape_param_448 = None
        unsqueeze_359: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_447, 4);  view_447 = None
        permute_284: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_359, [0, 1, 4, 3, 2]);  unsqueeze_359 = None
        permute_285: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_284, [0, 1, 3, 4, 2]);  permute_284 = None
        clone_24: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_285, memory_format = torch.contiguous_format);  permute_285 = None
        view_448: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_24, _shape_param_449);  clone_24 = _shape_param_449 = None
        squeeze_118: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_448, 0);  view_448 = None
        unsqueeze_360: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg173_1, 3);  arg173_1 = None
        unsqueeze_361: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 4);  unsqueeze_360 = None
        permute_286: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_361, [3, 4, 0, 2, 1]);  unsqueeze_361 = None
        permute_287: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_286, [3, 4, 2, 0, 1]);  permute_286 = None
        clone_25: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_449: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_25, _shape_param_450);  clone_25 = _shape_param_450 = None
        squeeze_119: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_449, 0);  view_449 = None
        mm_59: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_118, squeeze_119);  squeeze_118 = squeeze_119 = None
        unsqueeze_362: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_59, 0);  mm_59 = None
        view_450: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_362, _shape_param_451);  unsqueeze_362 = _shape_param_451 = None
        permute_288: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_450, [0, 1, 4, 2, 3]);  view_450 = None
        view_451: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_288, _shape_param_452);  permute_288 = _shape_param_452 = None
        add_125: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_451, convert_element_type_111);  view_451 = None
        convert_element_type_116: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_125, torch.float32);  add_125 = None
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_116, [2], correction = 0, keepdim = True)
        getitem_44: "f32[512, 16, 1]" = var_mean_22[0]
        getitem_45: "f32[512, 16, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_46: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_116, getitem_45);  convert_element_type_116 = getitem_45 = None
        add_126: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_127: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_22);  sub_46 = rsqrt_22 = None
        mul_128: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_127, arg174_1);  mul_127 = arg174_1 = None
        add_127: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_128, arg175_1);  mul_128 = arg175_1 = None
        convert_element_type_117: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None
        view_452: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_117, _shape_param_453);  _shape_param_453 = None
        permute_289: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
        addmm_22: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg177_1, view_452, permute_289);  arg177_1 = view_452 = permute_289 = None
        view_453: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_22, _shape_param_454);  addmm_22 = _shape_param_454 = None
        convert_element_type_118: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_453, torch.float32);  view_453 = None
        mul_129: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_118, 0.5)
        mul_130: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_118, 0.7071067811865476);  convert_element_type_118 = None
        erf_11: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
        add_128: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_131: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_129, add_128);  mul_129 = add_128 = None
        convert_element_type_119: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_131, torch.bfloat16);  mul_131 = None
        view_454: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_119, _shape_param_455);  convert_element_type_119 = _shape_param_455 = None
        permute_290: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
        addmm_23: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg179_1, view_454, permute_290);  arg179_1 = view_454 = permute_290 = None
        view_455: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_23, _shape_param_456);  addmm_23 = _shape_param_456 = None
        add_129: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_455, convert_element_type_117);  view_455 = convert_element_type_117 = None
        convert_element_type_120: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_129, torch.float32);  add_129 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_120, [2], correction = 0, keepdim = True)
        getitem_46: "f32[512, 16, 1]" = var_mean_23[0]
        getitem_47: "f32[512, 16, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_47: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_120, getitem_47);  convert_element_type_120 = getitem_47 = None
        add_130: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_132: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_23);  sub_47 = rsqrt_23 = None
        mul_133: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_132, arg180_1);  mul_132 = arg180_1 = None
        add_131: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_133, arg181_1);  mul_133 = arg181_1 = None
        convert_element_type_121: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None
        unsqueeze_363: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_121, 3)
        unsqueeze_364: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 4);  unsqueeze_363 = None
        view_456: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_364, _shape_param_457);  unsqueeze_364 = _shape_param_457 = None
        squeeze_120: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_456, 0);  view_456 = None
        unsqueeze_365: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg182_1, 3);  arg182_1 = None
        unsqueeze_366: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 4);  unsqueeze_365 = None
        view_457: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_366, _shape_param_458);  unsqueeze_366 = _shape_param_458 = None
        squeeze_121: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_457, 0);  view_457 = None
        mm_60: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_120, squeeze_121);  squeeze_120 = squeeze_121 = None
        unsqueeze_367: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_60, 0);  mm_60 = None
        view_458: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_367, _shape_param_459);  unsqueeze_367 = _shape_param_459 = None
        permute_291: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 4, 2]);  view_458 = None
        view_459: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_291, _shape_param_460);  permute_291 = _shape_param_460 = None
        add_132: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_459, arg183_1);  arg183_1 = None
        unsqueeze_368: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_132, 4);  add_132 = None
        permute_292: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_368, [1, 2, 0, 4, 3]);  unsqueeze_368 = None
        permute_293: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_292, [0, 1, 2, 4, 3]);  permute_292 = None
        view_460: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_293, _shape_param_461);  permute_293 = _shape_param_461 = None
        unsqueeze_369: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_121, 3)
        unsqueeze_370: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 4);  unsqueeze_369 = None
        view_461: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_370, _shape_param_462);  unsqueeze_370 = _shape_param_462 = None
        squeeze_122: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_461, 0);  view_461 = None
        unsqueeze_371: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, 3);  arg184_1 = None
        unsqueeze_372: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 4);  unsqueeze_371 = None
        view_462: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_372, _shape_param_463);  unsqueeze_372 = _shape_param_463 = None
        squeeze_123: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_462, 0);  view_462 = None
        mm_61: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_122, squeeze_123);  squeeze_122 = squeeze_123 = None
        unsqueeze_373: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_61, 0);  mm_61 = None
        view_463: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_373, _shape_param_464);  unsqueeze_373 = _shape_param_464 = None
        permute_294: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_463, [0, 1, 3, 4, 2]);  view_463 = None
        view_464: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_294, _shape_param_465);  permute_294 = _shape_param_465 = None
        unsqueeze_374: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_464, 4);  view_464 = None
        permute_295: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_374, [1, 2, 4, 0, 3]);  unsqueeze_374 = None
        permute_296: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_295, [0, 1, 4, 3, 2]);  permute_295 = None
        view_465: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_296, _shape_param_466);  permute_296 = _shape_param_466 = None
        bmm_36: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_460, view_465);  view_460 = view_465 = None
        view_466: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_36, _shape_param_467);  bmm_36 = _shape_param_467 = None
        permute_297: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_466, [0, 1, 2, 4, 3]);  view_466 = None
        view_467: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_297, _shape_param_468);  permute_297 = _shape_param_468 = None
        add_133: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_459, arg185_1);  view_459 = arg185_1 = None
        unsqueeze_375: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_133, 4);  add_133 = None
        permute_298: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_375, [1, 2, 0, 4, 3]);  unsqueeze_375 = None
        permute_299: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_298, [0, 1, 2, 4, 3]);  permute_298 = None
        view_468: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_299, _shape_param_469);  permute_299 = _shape_param_469 = None
        convert_element_type_122: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_376: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_122, 3);  convert_element_type_122 = None
        unsqueeze_377: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 4);  unsqueeze_376 = None
        view_469: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_377, _shape_param_470);  unsqueeze_377 = _shape_param_470 = None
        squeeze_124: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_469, 0);  view_469 = None
        unsqueeze_378: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg186_1, 3);  arg186_1 = None
        unsqueeze_379: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 4);  unsqueeze_378 = None
        view_470: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_379, _shape_param_471);  unsqueeze_379 = _shape_param_471 = None
        squeeze_125: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_470, 0);  view_470 = None
        mm_62: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_124, squeeze_125);  squeeze_124 = squeeze_125 = None
        unsqueeze_380: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_62, 0);  mm_62 = None
        view_471: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_380, _shape_param_472);  unsqueeze_380 = _shape_param_472 = None
        permute_300: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_471, [0, 1, 3, 4, 2]);  view_471 = None
        view_472: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_300, _shape_param_473);  permute_300 = _shape_param_473 = None
        unsqueeze_381: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_472, 4);  view_472 = None
        permute_301: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_381, [1, 2, 4, 0, 3]);  unsqueeze_381 = None
        permute_302: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_301, [0, 1, 4, 3, 2]);  permute_301 = None
        view_473: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_302, _shape_param_474);  permute_302 = _shape_param_474 = None
        bmm_37: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_468, view_473);  view_468 = view_473 = None
        view_474: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_37, _shape_param_475);  bmm_37 = _shape_param_475 = None
        permute_303: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_474, [0, 1, 2, 4, 3]);  view_474 = None
        view_475: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_303, _shape_param_476);  permute_303 = _shape_param_476 = None
        view_476: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_475, _shape_param_477);  view_475 = _shape_param_477 = None
        slice_13: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_476, 2, 1, 9223372036854775807);  view_476 = None
        view_477: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_13, _shape_param_478);  slice_13 = _shape_param_478 = None
        iota_14: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_12: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_477, [None, None, None, iota_14]);  view_477 = iota_14 = None
        add_134: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_467, index_12);  view_467 = index_12 = None
        add_135: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_134, 0);  add_134 = None
        mul_134: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_135, 0.125)
        convert_element_type_123: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_134, torch.float32);  mul_134 = None
        eq_12: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_123, convert_element_type_123)
        abs_13: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_123)
        ne_12: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_13, inf);  abs_13 = None
        mul_135: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_12, ne_12);  eq_12 = ne_12 = None
        logical_not_24: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_135);  mul_135 = None
        any_13: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_24, [3], True);  logical_not_24 = None
        logical_not_25: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        convert_element_type_124: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_135, torch.float32);  add_135 = None
        mul_136: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 1);  convert_element_type_124 = None
        amax_24: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_136, [3], True)
        sub_48: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_136, amax_24);  mul_136 = amax_24 = None
        mul_137: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_48, 0.125);  sub_48 = None
        amax_25: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_123, [3], True)
        sub_49: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_123, amax_25);  convert_element_type_123 = amax_25 = None
        where_12: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_25, mul_137, sub_49);  logical_not_25 = mul_137 = sub_49 = None
        exp_12: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_12);  where_12 = None
        sum_13: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [3], True)
        div_13: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_125: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        unsqueeze_382: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_125, 4);  convert_element_type_125 = None
        view_478: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_382, _shape_param_479);  unsqueeze_382 = _shape_param_479 = None
        unsqueeze_383: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_121, 3)
        unsqueeze_384: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 4);  unsqueeze_383 = None
        view_479: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_384, _shape_param_480);  unsqueeze_384 = _shape_param_480 = None
        squeeze_126: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_479, 0);  view_479 = None
        unsqueeze_385: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg187_1, 3);  arg187_1 = None
        unsqueeze_386: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 4);  unsqueeze_385 = None
        view_480: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_386, _shape_param_481);  unsqueeze_386 = _shape_param_481 = None
        squeeze_127: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_480, 0);  view_480 = None
        mm_63: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_126, squeeze_127);  squeeze_126 = squeeze_127 = None
        unsqueeze_387: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_63, 0);  mm_63 = None
        view_481: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_387, _shape_param_482);  unsqueeze_387 = _shape_param_482 = None
        permute_304: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_481, [0, 1, 3, 4, 2]);  view_481 = None
        view_482: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_304, _shape_param_483);  permute_304 = _shape_param_483 = None
        unsqueeze_388: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_482, 4);  view_482 = None
        permute_305: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_388, [4, 1, 2, 3, 0]);  unsqueeze_388 = None
        permute_306: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_305, [1, 2, 4, 3, 0]);  permute_305 = None
        view_483: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_306, _shape_param_484);  permute_306 = _shape_param_484 = None
        bmm_38: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_478, view_483);  view_478 = view_483 = None
        view_484: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_38, _shape_param_485);  bmm_38 = _shape_param_485 = None
        permute_307: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_484, [2, 0, 1, 4, 3]);  view_484 = None
        view_485: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_307, _shape_param_486);  permute_307 = _shape_param_486 = None
        unsqueeze_389: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_485, 4);  view_485 = None
        permute_308: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_389, [0, 1, 4, 3, 2]);  unsqueeze_389 = None
        permute_309: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_308, [0, 1, 3, 4, 2]);  permute_308 = None
        clone_26: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
        view_486: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_26, _shape_param_487);  clone_26 = _shape_param_487 = None
        squeeze_128: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_486, 0);  view_486 = None
        unsqueeze_390: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg188_1, 3);  arg188_1 = None
        unsqueeze_391: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 4);  unsqueeze_390 = None
        permute_310: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_391, [3, 4, 0, 2, 1]);  unsqueeze_391 = None
        permute_311: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_310, [3, 4, 2, 0, 1]);  permute_310 = None
        clone_27: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_311, memory_format = torch.contiguous_format);  permute_311 = None
        view_487: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_27, _shape_param_488);  clone_27 = _shape_param_488 = None
        squeeze_129: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_487, 0);  view_487 = None
        mm_64: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_128, squeeze_129);  squeeze_128 = squeeze_129 = None
        unsqueeze_392: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_64, 0);  mm_64 = None
        view_488: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_392, _shape_param_489);  unsqueeze_392 = _shape_param_489 = None
        permute_312: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_488, [0, 1, 4, 2, 3]);  view_488 = None
        view_489: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_312, _shape_param_490);  permute_312 = _shape_param_490 = None
        add_136: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_489, convert_element_type_121);  view_489 = None
        convert_element_type_126: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_136, torch.float32);  add_136 = None
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_126, [2], correction = 0, keepdim = True)
        getitem_48: "f32[512, 16, 1]" = var_mean_24[0]
        getitem_49: "f32[512, 16, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_50: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_126, getitem_49);  convert_element_type_126 = getitem_49 = None
        add_137: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        mul_138: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_24);  sub_50 = rsqrt_24 = None
        mul_139: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_138, arg189_1);  mul_138 = arg189_1 = None
        add_138: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_139, arg190_1);  mul_139 = arg190_1 = None
        convert_element_type_127: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16);  add_138 = None
        view_490: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_491);  _shape_param_491 = None
        permute_313: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_24: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg192_1, view_490, permute_313);  arg192_1 = view_490 = permute_313 = None
        view_491: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_24, _shape_param_492);  addmm_24 = _shape_param_492 = None
        convert_element_type_128: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_491, torch.float32);  view_491 = None
        mul_140: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_128, 0.5)
        mul_141: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_128, 0.7071067811865476);  convert_element_type_128 = None
        erf_12: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_139: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_142: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_140, add_139);  mul_140 = add_139 = None
        convert_element_type_129: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        view_492: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_129, _shape_param_493);  convert_element_type_129 = _shape_param_493 = None
        permute_314: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_25: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg194_1, view_492, permute_314);  arg194_1 = view_492 = permute_314 = None
        view_493: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_25, _shape_param_494);  addmm_25 = _shape_param_494 = None
        add_140: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_493, convert_element_type_127);  view_493 = convert_element_type_127 = None
        convert_element_type_130: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_140, torch.float32);  add_140 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_130, [2], correction = 0, keepdim = True)
        getitem_50: "f32[512, 16, 1]" = var_mean_25[0]
        getitem_51: "f32[512, 16, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_51: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_130, getitem_51);  convert_element_type_130 = getitem_51 = None
        add_141: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_143: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_25);  sub_51 = rsqrt_25 = None
        mul_144: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_143, arg195_1);  mul_143 = arg195_1 = None
        add_142: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_144, arg196_1);  mul_144 = arg196_1 = None
        convert_element_type_131: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16);  add_142 = None
        unsqueeze_393: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_131, 3)
        unsqueeze_394: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 4);  unsqueeze_393 = None
        view_494: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_394, _shape_param_495);  unsqueeze_394 = _shape_param_495 = None
        squeeze_130: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_494, 0);  view_494 = None
        unsqueeze_395: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg197_1, 3);  arg197_1 = None
        unsqueeze_396: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 4);  unsqueeze_395 = None
        view_495: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_396, _shape_param_496);  unsqueeze_396 = _shape_param_496 = None
        squeeze_131: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_495, 0);  view_495 = None
        mm_65: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_130, squeeze_131);  squeeze_130 = squeeze_131 = None
        unsqueeze_397: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_65, 0);  mm_65 = None
        view_496: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_397, _shape_param_497);  unsqueeze_397 = _shape_param_497 = None
        permute_315: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_496, [0, 1, 3, 4, 2]);  view_496 = None
        view_497: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_315, _shape_param_498);  permute_315 = _shape_param_498 = None
        add_143: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_497, arg198_1);  arg198_1 = None
        unsqueeze_398: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_143, 4);  add_143 = None
        permute_316: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_398, [1, 2, 0, 4, 3]);  unsqueeze_398 = None
        permute_317: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_316, [0, 1, 2, 4, 3]);  permute_316 = None
        view_498: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_317, _shape_param_499);  permute_317 = _shape_param_499 = None
        unsqueeze_399: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_131, 3)
        unsqueeze_400: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 4);  unsqueeze_399 = None
        view_499: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_400, _shape_param_500);  unsqueeze_400 = _shape_param_500 = None
        squeeze_132: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_499, 0);  view_499 = None
        unsqueeze_401: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, 3);  arg199_1 = None
        unsqueeze_402: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 4);  unsqueeze_401 = None
        view_500: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_402, _shape_param_501);  unsqueeze_402 = _shape_param_501 = None
        squeeze_133: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_500, 0);  view_500 = None
        mm_66: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_132, squeeze_133);  squeeze_132 = squeeze_133 = None
        unsqueeze_403: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_66, 0);  mm_66 = None
        view_501: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_403, _shape_param_502);  unsqueeze_403 = _shape_param_502 = None
        permute_318: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_501, [0, 1, 3, 4, 2]);  view_501 = None
        view_502: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_318, _shape_param_503);  permute_318 = _shape_param_503 = None
        unsqueeze_404: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_502, 4);  view_502 = None
        permute_319: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_404, [1, 2, 4, 0, 3]);  unsqueeze_404 = None
        permute_320: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_319, [0, 1, 4, 3, 2]);  permute_319 = None
        view_503: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_320, _shape_param_504);  permute_320 = _shape_param_504 = None
        bmm_39: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_498, view_503);  view_498 = view_503 = None
        view_504: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_39, _shape_param_505);  bmm_39 = _shape_param_505 = None
        permute_321: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_504, [0, 1, 2, 4, 3]);  view_504 = None
        view_505: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_321, _shape_param_506);  permute_321 = _shape_param_506 = None
        add_144: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_497, arg200_1);  view_497 = arg200_1 = None
        unsqueeze_405: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_144, 4);  add_144 = None
        permute_322: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_405, [1, 2, 0, 4, 3]);  unsqueeze_405 = None
        permute_323: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_322, [0, 1, 2, 4, 3]);  permute_322 = None
        view_506: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_323, _shape_param_507);  permute_323 = _shape_param_507 = None
        convert_element_type_132: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_406: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_132, 3);  convert_element_type_132 = None
        unsqueeze_407: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 4);  unsqueeze_406 = None
        view_507: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_407, _shape_param_508);  unsqueeze_407 = _shape_param_508 = None
        squeeze_134: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_507, 0);  view_507 = None
        unsqueeze_408: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg201_1, 3);  arg201_1 = None
        unsqueeze_409: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 4);  unsqueeze_408 = None
        view_508: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_409, _shape_param_509);  unsqueeze_409 = _shape_param_509 = None
        squeeze_135: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_508, 0);  view_508 = None
        mm_67: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_134, squeeze_135);  squeeze_134 = squeeze_135 = None
        unsqueeze_410: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_67, 0);  mm_67 = None
        view_509: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_410, _shape_param_510);  unsqueeze_410 = _shape_param_510 = None
        permute_324: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_509, [0, 1, 3, 4, 2]);  view_509 = None
        view_510: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_324, _shape_param_511);  permute_324 = _shape_param_511 = None
        unsqueeze_411: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_510, 4);  view_510 = None
        permute_325: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_411, [1, 2, 4, 0, 3]);  unsqueeze_411 = None
        permute_326: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_325, [0, 1, 4, 3, 2]);  permute_325 = None
        view_511: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_326, _shape_param_512);  permute_326 = _shape_param_512 = None
        bmm_40: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_506, view_511);  view_506 = view_511 = None
        view_512: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_40, _shape_param_513);  bmm_40 = _shape_param_513 = None
        permute_327: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_512, [0, 1, 2, 4, 3]);  view_512 = None
        view_513: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_327, _shape_param_514);  permute_327 = _shape_param_514 = None
        view_514: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_513, _shape_param_515);  view_513 = _shape_param_515 = None
        slice_14: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_514, 2, 1, 9223372036854775807);  view_514 = None
        view_515: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_14, _shape_param_516);  slice_14 = _shape_param_516 = None
        iota_15: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_13: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_515, [None, None, None, iota_15]);  view_515 = iota_15 = None
        add_145: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_505, index_13);  view_505 = index_13 = None
        add_146: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_145, 0);  add_145 = None
        mul_145: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_146, 0.125)
        convert_element_type_133: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_145, torch.float32);  mul_145 = None
        eq_13: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_133, convert_element_type_133)
        abs_14: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_133)
        ne_13: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_14, inf);  abs_14 = None
        mul_146: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_13, ne_13);  eq_13 = ne_13 = None
        logical_not_26: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_146);  mul_146 = None
        any_14: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_26, [3], True);  logical_not_26 = None
        logical_not_27: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        convert_element_type_134: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_146, torch.float32);  add_146 = None
        mul_147: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_134, 1);  convert_element_type_134 = None
        amax_26: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_147, [3], True)
        sub_52: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_147, amax_26);  mul_147 = amax_26 = None
        mul_148: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_52, 0.125);  sub_52 = None
        amax_27: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_133, [3], True)
        sub_53: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_133, amax_27);  convert_element_type_133 = amax_27 = None
        where_13: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_27, mul_148, sub_53);  logical_not_27 = mul_148 = sub_53 = None
        exp_13: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_13);  where_13 = None
        sum_14: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [3], True)
        div_14: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_135: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        unsqueeze_412: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_135, 4);  convert_element_type_135 = None
        view_516: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_412, _shape_param_517);  unsqueeze_412 = _shape_param_517 = None
        unsqueeze_413: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_131, 3)
        unsqueeze_414: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 4);  unsqueeze_413 = None
        view_517: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_414, _shape_param_518);  unsqueeze_414 = _shape_param_518 = None
        squeeze_136: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_517, 0);  view_517 = None
        unsqueeze_415: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg202_1, 3);  arg202_1 = None
        unsqueeze_416: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 4);  unsqueeze_415 = None
        view_518: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_416, _shape_param_519);  unsqueeze_416 = _shape_param_519 = None
        squeeze_137: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_518, 0);  view_518 = None
        mm_68: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_136, squeeze_137);  squeeze_136 = squeeze_137 = None
        unsqueeze_417: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_68, 0);  mm_68 = None
        view_519: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_417, _shape_param_520);  unsqueeze_417 = _shape_param_520 = None
        permute_328: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_519, [0, 1, 3, 4, 2]);  view_519 = None
        view_520: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_328, _shape_param_521);  permute_328 = _shape_param_521 = None
        unsqueeze_418: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_520, 4);  view_520 = None
        permute_329: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_418, [4, 1, 2, 3, 0]);  unsqueeze_418 = None
        permute_330: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_329, [1, 2, 4, 3, 0]);  permute_329 = None
        view_521: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_330, _shape_param_522);  permute_330 = _shape_param_522 = None
        bmm_41: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_516, view_521);  view_516 = view_521 = None
        view_522: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_41, _shape_param_523);  bmm_41 = _shape_param_523 = None
        permute_331: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_522, [2, 0, 1, 4, 3]);  view_522 = None
        view_523: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_331, _shape_param_524);  permute_331 = _shape_param_524 = None
        unsqueeze_419: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_523, 4);  view_523 = None
        permute_332: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_419, [0, 1, 4, 3, 2]);  unsqueeze_419 = None
        permute_333: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_332, [0, 1, 3, 4, 2]);  permute_332 = None
        clone_28: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
        view_524: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_28, _shape_param_525);  clone_28 = _shape_param_525 = None
        squeeze_138: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_524, 0);  view_524 = None
        unsqueeze_420: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg203_1, 3);  arg203_1 = None
        unsqueeze_421: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 4);  unsqueeze_420 = None
        permute_334: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_421, [3, 4, 0, 2, 1]);  unsqueeze_421 = None
        permute_335: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_334, [3, 4, 2, 0, 1]);  permute_334 = None
        clone_29: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_525: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_29, _shape_param_526);  clone_29 = _shape_param_526 = None
        squeeze_139: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_525, 0);  view_525 = None
        mm_69: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_138, squeeze_139);  squeeze_138 = squeeze_139 = None
        unsqueeze_422: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_69, 0);  mm_69 = None
        view_526: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_422, _shape_param_527);  unsqueeze_422 = _shape_param_527 = None
        permute_336: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_526, [0, 1, 4, 2, 3]);  view_526 = None
        view_527: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_336, _shape_param_528);  permute_336 = _shape_param_528 = None
        add_147: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_527, convert_element_type_131);  view_527 = None
        convert_element_type_136: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_147, torch.float32);  add_147 = None
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_136, [2], correction = 0, keepdim = True)
        getitem_52: "f32[512, 16, 1]" = var_mean_26[0]
        getitem_53: "f32[512, 16, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_54: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_136, getitem_53);  convert_element_type_136 = getitem_53 = None
        add_148: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_149: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_26);  sub_54 = rsqrt_26 = None
        mul_150: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_149, arg204_1);  mul_149 = arg204_1 = None
        add_149: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_150, arg205_1);  mul_150 = arg205_1 = None
        convert_element_type_137: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None
        view_528: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_137, _shape_param_529);  _shape_param_529 = None
        permute_337: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
        addmm_26: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg207_1, view_528, permute_337);  arg207_1 = view_528 = permute_337 = None
        view_529: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_26, _shape_param_530);  addmm_26 = _shape_param_530 = None
        convert_element_type_138: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        mul_151: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_138, 0.5)
        mul_152: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_138, 0.7071067811865476);  convert_element_type_138 = None
        erf_13: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
        add_150: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_153: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_151, add_150);  mul_151 = add_150 = None
        convert_element_type_139: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_153, torch.bfloat16);  mul_153 = None
        view_530: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_139, _shape_param_531);  convert_element_type_139 = _shape_param_531 = None
        permute_338: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
        addmm_27: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg209_1, view_530, permute_338);  arg209_1 = view_530 = permute_338 = None
        view_531: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_27, _shape_param_532);  addmm_27 = _shape_param_532 = None
        add_151: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_531, convert_element_type_137);  view_531 = convert_element_type_137 = None
        convert_element_type_140: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_151, torch.float32);  add_151 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_140, [2], correction = 0, keepdim = True)
        getitem_54: "f32[512, 16, 1]" = var_mean_27[0]
        getitem_55: "f32[512, 16, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_55: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_140, getitem_55);  convert_element_type_140 = getitem_55 = None
        add_152: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        mul_154: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_27);  sub_55 = rsqrt_27 = None
        mul_155: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_154, arg210_1);  mul_154 = arg210_1 = None
        add_153: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_155, arg211_1);  mul_155 = arg211_1 = None
        convert_element_type_141: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16);  add_153 = None
        unsqueeze_423: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_141, 3)
        unsqueeze_424: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 4);  unsqueeze_423 = None
        view_532: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_424, _shape_param_533);  unsqueeze_424 = _shape_param_533 = None
        squeeze_140: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_532, 0);  view_532 = None
        unsqueeze_425: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg212_1, 3);  arg212_1 = None
        unsqueeze_426: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 4);  unsqueeze_425 = None
        view_533: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_426, _shape_param_534);  unsqueeze_426 = _shape_param_534 = None
        squeeze_141: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_533, 0);  view_533 = None
        mm_70: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_140, squeeze_141);  squeeze_140 = squeeze_141 = None
        unsqueeze_427: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_70, 0);  mm_70 = None
        view_534: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_427, _shape_param_535);  unsqueeze_427 = _shape_param_535 = None
        permute_339: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_534, [0, 1, 3, 4, 2]);  view_534 = None
        view_535: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_339, _shape_param_536);  permute_339 = _shape_param_536 = None
        add_154: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_535, arg213_1);  arg213_1 = None
        unsqueeze_428: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_154, 4);  add_154 = None
        permute_340: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_428, [1, 2, 0, 4, 3]);  unsqueeze_428 = None
        permute_341: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_340, [0, 1, 2, 4, 3]);  permute_340 = None
        view_536: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_341, _shape_param_537);  permute_341 = _shape_param_537 = None
        unsqueeze_429: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_141, 3)
        unsqueeze_430: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 4);  unsqueeze_429 = None
        view_537: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_430, _shape_param_538);  unsqueeze_430 = _shape_param_538 = None
        squeeze_142: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_537, 0);  view_537 = None
        unsqueeze_431: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, 3);  arg214_1 = None
        unsqueeze_432: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 4);  unsqueeze_431 = None
        view_538: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_432, _shape_param_539);  unsqueeze_432 = _shape_param_539 = None
        squeeze_143: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_538, 0);  view_538 = None
        mm_71: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_142, squeeze_143);  squeeze_142 = squeeze_143 = None
        unsqueeze_433: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_71, 0);  mm_71 = None
        view_539: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_433, _shape_param_540);  unsqueeze_433 = _shape_param_540 = None
        permute_342: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_539, [0, 1, 3, 4, 2]);  view_539 = None
        view_540: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_342, _shape_param_541);  permute_342 = _shape_param_541 = None
        unsqueeze_434: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_540, 4);  view_540 = None
        permute_343: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_434, [1, 2, 4, 0, 3]);  unsqueeze_434 = None
        permute_344: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_343, [0, 1, 4, 3, 2]);  permute_343 = None
        view_541: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_344, _shape_param_542);  permute_344 = _shape_param_542 = None
        bmm_42: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_536, view_541);  view_536 = view_541 = None
        view_542: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_42, _shape_param_543);  bmm_42 = _shape_param_543 = None
        permute_345: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_542, [0, 1, 2, 4, 3]);  view_542 = None
        view_543: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_345, _shape_param_544);  permute_345 = _shape_param_544 = None
        add_155: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_535, arg215_1);  view_535 = arg215_1 = None
        unsqueeze_435: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_155, 4);  add_155 = None
        permute_346: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_435, [1, 2, 0, 4, 3]);  unsqueeze_435 = None
        permute_347: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_346, [0, 1, 2, 4, 3]);  permute_346 = None
        view_544: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_347, _shape_param_545);  permute_347 = _shape_param_545 = None
        convert_element_type_142: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_436: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_142, 3);  convert_element_type_142 = None
        unsqueeze_437: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 4);  unsqueeze_436 = None
        view_545: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_437, _shape_param_546);  unsqueeze_437 = _shape_param_546 = None
        squeeze_144: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_545, 0);  view_545 = None
        unsqueeze_438: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg216_1, 3);  arg216_1 = None
        unsqueeze_439: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 4);  unsqueeze_438 = None
        view_546: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_439, _shape_param_547);  unsqueeze_439 = _shape_param_547 = None
        squeeze_145: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_546, 0);  view_546 = None
        mm_72: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_144, squeeze_145);  squeeze_144 = squeeze_145 = None
        unsqueeze_440: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_72, 0);  mm_72 = None
        view_547: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_440, _shape_param_548);  unsqueeze_440 = _shape_param_548 = None
        permute_348: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_547, [0, 1, 3, 4, 2]);  view_547 = None
        view_548: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_348, _shape_param_549);  permute_348 = _shape_param_549 = None
        unsqueeze_441: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_548, 4);  view_548 = None
        permute_349: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_441, [1, 2, 4, 0, 3]);  unsqueeze_441 = None
        permute_350: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_349, [0, 1, 4, 3, 2]);  permute_349 = None
        view_549: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_350, _shape_param_550);  permute_350 = _shape_param_550 = None
        bmm_43: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_544, view_549);  view_544 = view_549 = None
        view_550: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_43, _shape_param_551);  bmm_43 = _shape_param_551 = None
        permute_351: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_550, [0, 1, 2, 4, 3]);  view_550 = None
        view_551: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_351, _shape_param_552);  permute_351 = _shape_param_552 = None
        view_552: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_551, _shape_param_553);  view_551 = _shape_param_553 = None
        slice_15: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_552, 2, 1, 9223372036854775807);  view_552 = None
        view_553: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_15, _shape_param_554);  slice_15 = _shape_param_554 = None
        iota_16: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_14: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_553, [None, None, None, iota_16]);  view_553 = iota_16 = None
        add_156: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_543, index_14);  view_543 = index_14 = None
        add_157: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_156, 0);  add_156 = None
        mul_156: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_157, 0.125)
        convert_element_type_143: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_156, torch.float32);  mul_156 = None
        eq_14: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_143, convert_element_type_143)
        abs_15: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_143)
        ne_14: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_15, inf);  abs_15 = None
        mul_157: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_14, ne_14);  eq_14 = ne_14 = None
        logical_not_28: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_157);  mul_157 = None
        any_15: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_28, [3], True);  logical_not_28 = None
        logical_not_29: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        convert_element_type_144: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_157, torch.float32);  add_157 = None
        mul_158: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_144, 1);  convert_element_type_144 = None
        amax_28: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_158, [3], True)
        sub_56: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_158, amax_28);  mul_158 = amax_28 = None
        mul_159: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_56, 0.125);  sub_56 = None
        amax_29: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_143, [3], True)
        sub_57: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_143, amax_29);  convert_element_type_143 = amax_29 = None
        where_14: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_29, mul_159, sub_57);  logical_not_29 = mul_159 = sub_57 = None
        exp_14: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_14);  where_14 = None
        sum_15: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [3], True)
        div_15: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_145: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        unsqueeze_442: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_145, 4);  convert_element_type_145 = None
        view_554: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_442, _shape_param_555);  unsqueeze_442 = _shape_param_555 = None
        unsqueeze_443: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_141, 3)
        unsqueeze_444: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 4);  unsqueeze_443 = None
        view_555: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_444, _shape_param_556);  unsqueeze_444 = _shape_param_556 = None
        squeeze_146: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_555, 0);  view_555 = None
        unsqueeze_445: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg217_1, 3);  arg217_1 = None
        unsqueeze_446: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 4);  unsqueeze_445 = None
        view_556: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_446, _shape_param_557);  unsqueeze_446 = _shape_param_557 = None
        squeeze_147: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_556, 0);  view_556 = None
        mm_73: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_146, squeeze_147);  squeeze_146 = squeeze_147 = None
        unsqueeze_447: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_73, 0);  mm_73 = None
        view_557: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_447, _shape_param_558);  unsqueeze_447 = _shape_param_558 = None
        permute_352: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_557, [0, 1, 3, 4, 2]);  view_557 = None
        view_558: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_352, _shape_param_559);  permute_352 = _shape_param_559 = None
        unsqueeze_448: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_558, 4);  view_558 = None
        permute_353: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_448, [4, 1, 2, 3, 0]);  unsqueeze_448 = None
        permute_354: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_353, [1, 2, 4, 3, 0]);  permute_353 = None
        view_559: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_354, _shape_param_560);  permute_354 = _shape_param_560 = None
        bmm_44: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_554, view_559);  view_554 = view_559 = None
        view_560: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_561);  bmm_44 = _shape_param_561 = None
        permute_355: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_560, [2, 0, 1, 4, 3]);  view_560 = None
        view_561: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_355, _shape_param_562);  permute_355 = _shape_param_562 = None
        unsqueeze_449: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_561, 4);  view_561 = None
        permute_356: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_449, [0, 1, 4, 3, 2]);  unsqueeze_449 = None
        permute_357: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_356, [0, 1, 3, 4, 2]);  permute_356 = None
        clone_30: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_357, memory_format = torch.contiguous_format);  permute_357 = None
        view_562: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_30, _shape_param_563);  clone_30 = _shape_param_563 = None
        squeeze_148: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_562, 0);  view_562 = None
        unsqueeze_450: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg218_1, 3);  arg218_1 = None
        unsqueeze_451: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 4);  unsqueeze_450 = None
        permute_358: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_451, [3, 4, 0, 2, 1]);  unsqueeze_451 = None
        permute_359: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_358, [3, 4, 2, 0, 1]);  permute_358 = None
        clone_31: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
        view_563: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_31, _shape_param_564);  clone_31 = _shape_param_564 = None
        squeeze_149: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_563, 0);  view_563 = None
        mm_74: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_148, squeeze_149);  squeeze_148 = squeeze_149 = None
        unsqueeze_452: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_74, 0);  mm_74 = None
        view_564: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_452, _shape_param_565);  unsqueeze_452 = _shape_param_565 = None
        permute_360: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_564, [0, 1, 4, 2, 3]);  view_564 = None
        view_565: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_360, _shape_param_566);  permute_360 = _shape_param_566 = None
        add_158: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_565, convert_element_type_141);  view_565 = None
        convert_element_type_146: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_158, torch.float32);  add_158 = None
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_146, [2], correction = 0, keepdim = True)
        getitem_56: "f32[512, 16, 1]" = var_mean_28[0]
        getitem_57: "f32[512, 16, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_58: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_146, getitem_57);  convert_element_type_146 = getitem_57 = None
        add_159: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_160: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_28);  sub_58 = rsqrt_28 = None
        mul_161: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_160, arg219_1);  mul_160 = arg219_1 = None
        add_160: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_161, arg220_1);  mul_161 = arg220_1 = None
        convert_element_type_147: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16);  add_160 = None
        view_566: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_147, _shape_param_567);  _shape_param_567 = None
        permute_361: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
        addmm_28: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg222_1, view_566, permute_361);  arg222_1 = view_566 = permute_361 = None
        view_567: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_28, _shape_param_568);  addmm_28 = _shape_param_568 = None
        convert_element_type_148: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_567, torch.float32);  view_567 = None
        mul_162: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.5)
        mul_163: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_148, 0.7071067811865476);  convert_element_type_148 = None
        erf_14: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_161: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_164: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_162, add_161);  mul_162 = add_161 = None
        convert_element_type_149: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_164, torch.bfloat16);  mul_164 = None
        view_568: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_149, _shape_param_569);  convert_element_type_149 = _shape_param_569 = None
        permute_362: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_29: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg224_1, view_568, permute_362);  arg224_1 = view_568 = permute_362 = None
        view_569: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_29, _shape_param_570);  addmm_29 = _shape_param_570 = None
        add_162: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_569, convert_element_type_147);  view_569 = convert_element_type_147 = None
        convert_element_type_150: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_162, torch.float32);  add_162 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_150, [2], correction = 0, keepdim = True)
        getitem_58: "f32[512, 16, 1]" = var_mean_29[0]
        getitem_59: "f32[512, 16, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_59: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_150, getitem_59);  convert_element_type_150 = getitem_59 = None
        add_163: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        mul_165: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_29);  sub_59 = rsqrt_29 = None
        mul_166: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_165, arg225_1);  mul_165 = arg225_1 = None
        add_164: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_166, arg226_1);  mul_166 = arg226_1 = None
        convert_element_type_151: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None
        unsqueeze_453: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_151, 3)
        unsqueeze_454: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 4);  unsqueeze_453 = None
        view_570: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_454, _shape_param_571);  unsqueeze_454 = _shape_param_571 = None
        squeeze_150: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_570, 0);  view_570 = None
        unsqueeze_455: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg227_1, 3);  arg227_1 = None
        unsqueeze_456: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 4);  unsqueeze_455 = None
        view_571: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_456, _shape_param_572);  unsqueeze_456 = _shape_param_572 = None
        squeeze_151: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_571, 0);  view_571 = None
        mm_75: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_150, squeeze_151);  squeeze_150 = squeeze_151 = None
        unsqueeze_457: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_75, 0);  mm_75 = None
        view_572: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_457, _shape_param_573);  unsqueeze_457 = _shape_param_573 = None
        permute_363: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_572, [0, 1, 3, 4, 2]);  view_572 = None
        view_573: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_363, _shape_param_574);  permute_363 = _shape_param_574 = None
        add_165: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_573, arg228_1);  arg228_1 = None
        unsqueeze_458: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_165, 4);  add_165 = None
        permute_364: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_458, [1, 2, 0, 4, 3]);  unsqueeze_458 = None
        permute_365: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_364, [0, 1, 2, 4, 3]);  permute_364 = None
        view_574: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_365, _shape_param_575);  permute_365 = _shape_param_575 = None
        unsqueeze_459: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_151, 3)
        unsqueeze_460: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 4);  unsqueeze_459 = None
        view_575: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_460, _shape_param_576);  unsqueeze_460 = _shape_param_576 = None
        squeeze_152: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_575, 0);  view_575 = None
        unsqueeze_461: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, 3);  arg229_1 = None
        unsqueeze_462: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 4);  unsqueeze_461 = None
        view_576: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_462, _shape_param_577);  unsqueeze_462 = _shape_param_577 = None
        squeeze_153: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_576, 0);  view_576 = None
        mm_76: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_152, squeeze_153);  squeeze_152 = squeeze_153 = None
        unsqueeze_463: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_76, 0);  mm_76 = None
        view_577: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_463, _shape_param_578);  unsqueeze_463 = _shape_param_578 = None
        permute_366: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_577, [0, 1, 3, 4, 2]);  view_577 = None
        view_578: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_366, _shape_param_579);  permute_366 = _shape_param_579 = None
        unsqueeze_464: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_578, 4);  view_578 = None
        permute_367: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_464, [1, 2, 4, 0, 3]);  unsqueeze_464 = None
        permute_368: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_367, [0, 1, 4, 3, 2]);  permute_367 = None
        view_579: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_368, _shape_param_580);  permute_368 = _shape_param_580 = None
        bmm_45: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_574, view_579);  view_574 = view_579 = None
        view_580: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_45, _shape_param_581);  bmm_45 = _shape_param_581 = None
        permute_369: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_580, [0, 1, 2, 4, 3]);  view_580 = None
        view_581: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_369, _shape_param_582);  permute_369 = _shape_param_582 = None
        add_166: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_573, arg230_1);  view_573 = arg230_1 = None
        unsqueeze_465: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_166, 4);  add_166 = None
        permute_370: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_465, [1, 2, 0, 4, 3]);  unsqueeze_465 = None
        permute_371: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_370, [0, 1, 2, 4, 3]);  permute_370 = None
        view_582: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_371, _shape_param_583);  permute_371 = _shape_param_583 = None
        convert_element_type_152: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_466: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_152, 3);  convert_element_type_152 = None
        unsqueeze_467: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 4);  unsqueeze_466 = None
        view_583: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_467, _shape_param_584);  unsqueeze_467 = _shape_param_584 = None
        squeeze_154: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_583, 0);  view_583 = None
        unsqueeze_468: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg231_1, 3);  arg231_1 = None
        unsqueeze_469: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 4);  unsqueeze_468 = None
        view_584: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_469, _shape_param_585);  unsqueeze_469 = _shape_param_585 = None
        squeeze_155: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_584, 0);  view_584 = None
        mm_77: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_154, squeeze_155);  squeeze_154 = squeeze_155 = None
        unsqueeze_470: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_77, 0);  mm_77 = None
        view_585: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_470, _shape_param_586);  unsqueeze_470 = _shape_param_586 = None
        permute_372: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_585, [0, 1, 3, 4, 2]);  view_585 = None
        view_586: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_372, _shape_param_587);  permute_372 = _shape_param_587 = None
        unsqueeze_471: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_586, 4);  view_586 = None
        permute_373: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_471, [1, 2, 4, 0, 3]);  unsqueeze_471 = None
        permute_374: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_373, [0, 1, 4, 3, 2]);  permute_373 = None
        view_587: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_374, _shape_param_588);  permute_374 = _shape_param_588 = None
        bmm_46: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_582, view_587);  view_582 = view_587 = None
        view_588: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_46, _shape_param_589);  bmm_46 = _shape_param_589 = None
        permute_375: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_588, [0, 1, 2, 4, 3]);  view_588 = None
        view_589: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_375, _shape_param_590);  permute_375 = _shape_param_590 = None
        view_590: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_589, _shape_param_591);  view_589 = _shape_param_591 = None
        slice_16: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_590, 2, 1, 9223372036854775807);  view_590 = None
        view_591: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_16, _shape_param_592);  slice_16 = _shape_param_592 = None
        iota_17: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_15: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_591, [None, None, None, iota_17]);  view_591 = iota_17 = None
        add_167: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_581, index_15);  view_581 = index_15 = None
        add_168: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_167, 0);  add_167 = None
        mul_167: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_168, 0.125)
        convert_element_type_153: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_167, torch.float32);  mul_167 = None
        eq_15: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_153, convert_element_type_153)
        abs_16: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_153)
        ne_15: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_16, inf);  abs_16 = None
        mul_168: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_15, ne_15);  eq_15 = ne_15 = None
        logical_not_30: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_168);  mul_168 = None
        any_16: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_30, [3], True);  logical_not_30 = None
        logical_not_31: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        convert_element_type_154: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_168, torch.float32);  add_168 = None
        mul_169: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_154, 1);  convert_element_type_154 = None
        amax_30: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_169, [3], True)
        sub_60: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_169, amax_30);  mul_169 = amax_30 = None
        mul_170: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_60, 0.125);  sub_60 = None
        amax_31: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_153, [3], True)
        sub_61: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_153, amax_31);  convert_element_type_153 = amax_31 = None
        where_15: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_31, mul_170, sub_61);  logical_not_31 = mul_170 = sub_61 = None
        exp_15: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_15);  where_15 = None
        sum_16: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [3], True)
        div_16: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_155: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        unsqueeze_472: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_155, 4);  convert_element_type_155 = None
        view_592: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_472, _shape_param_593);  unsqueeze_472 = _shape_param_593 = None
        unsqueeze_473: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_151, 3)
        unsqueeze_474: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 4);  unsqueeze_473 = None
        view_593: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_474, _shape_param_594);  unsqueeze_474 = _shape_param_594 = None
        squeeze_156: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_593, 0);  view_593 = None
        unsqueeze_475: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg232_1, 3);  arg232_1 = None
        unsqueeze_476: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 4);  unsqueeze_475 = None
        view_594: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_476, _shape_param_595);  unsqueeze_476 = _shape_param_595 = None
        squeeze_157: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_594, 0);  view_594 = None
        mm_78: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_156, squeeze_157);  squeeze_156 = squeeze_157 = None
        unsqueeze_477: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_78, 0);  mm_78 = None
        view_595: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_477, _shape_param_596);  unsqueeze_477 = _shape_param_596 = None
        permute_376: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_595, [0, 1, 3, 4, 2]);  view_595 = None
        view_596: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_376, _shape_param_597);  permute_376 = _shape_param_597 = None
        unsqueeze_478: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_596, 4);  view_596 = None
        permute_377: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_478, [4, 1, 2, 3, 0]);  unsqueeze_478 = None
        permute_378: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_377, [1, 2, 4, 3, 0]);  permute_377 = None
        view_597: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_378, _shape_param_598);  permute_378 = _shape_param_598 = None
        bmm_47: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_592, view_597);  view_592 = view_597 = None
        view_598: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_599);  bmm_47 = _shape_param_599 = None
        permute_379: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_598, [2, 0, 1, 4, 3]);  view_598 = None
        view_599: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_379, _shape_param_600);  permute_379 = _shape_param_600 = None
        unsqueeze_479: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_599, 4);  view_599 = None
        permute_380: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_479, [0, 1, 4, 3, 2]);  unsqueeze_479 = None
        permute_381: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_380, [0, 1, 3, 4, 2]);  permute_380 = None
        clone_32: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_381, memory_format = torch.contiguous_format);  permute_381 = None
        view_600: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_32, _shape_param_601);  clone_32 = _shape_param_601 = None
        squeeze_158: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_600, 0);  view_600 = None
        unsqueeze_480: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg233_1, 3);  arg233_1 = None
        unsqueeze_481: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 4);  unsqueeze_480 = None
        permute_382: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_481, [3, 4, 0, 2, 1]);  unsqueeze_481 = None
        permute_383: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_382, [3, 4, 2, 0, 1]);  permute_382 = None
        clone_33: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
        view_601: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_33, _shape_param_602);  clone_33 = _shape_param_602 = None
        squeeze_159: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_601, 0);  view_601 = None
        mm_79: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_158, squeeze_159);  squeeze_158 = squeeze_159 = None
        unsqueeze_482: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_79, 0);  mm_79 = None
        view_602: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_482, _shape_param_603);  unsqueeze_482 = _shape_param_603 = None
        permute_384: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_602, [0, 1, 4, 2, 3]);  view_602 = None
        view_603: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_384, _shape_param_604);  permute_384 = _shape_param_604 = None
        add_169: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_603, convert_element_type_151);  view_603 = None
        convert_element_type_156: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_169, torch.float32);  add_169 = None
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_156, [2], correction = 0, keepdim = True)
        getitem_60: "f32[512, 16, 1]" = var_mean_30[0]
        getitem_61: "f32[512, 16, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_62: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_156, getitem_61);  convert_element_type_156 = getitem_61 = None
        add_170: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_171: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_30);  sub_62 = rsqrt_30 = None
        mul_172: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_171, arg234_1);  mul_171 = arg234_1 = None
        add_171: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_172, arg235_1);  mul_172 = arg235_1 = None
        convert_element_type_157: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16);  add_171 = None
        view_604: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_157, _shape_param_605);  _shape_param_605 = None
        permute_385: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
        addmm_30: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg237_1, view_604, permute_385);  arg237_1 = view_604 = permute_385 = None
        view_605: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_30, _shape_param_606);  addmm_30 = _shape_param_606 = None
        convert_element_type_158: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_605, torch.float32);  view_605 = None
        mul_173: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_158, 0.5)
        mul_174: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_158, 0.7071067811865476);  convert_element_type_158 = None
        erf_15: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_172: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_175: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_173, add_172);  mul_173 = add_172 = None
        convert_element_type_159: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None
        view_606: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_159, _shape_param_607);  convert_element_type_159 = _shape_param_607 = None
        permute_386: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
        addmm_31: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg239_1, view_606, permute_386);  arg239_1 = view_606 = permute_386 = None
        view_607: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_31, _shape_param_608);  addmm_31 = _shape_param_608 = None
        add_173: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_607, convert_element_type_157);  view_607 = convert_element_type_157 = None
        convert_element_type_160: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_173, torch.float32);  add_173 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_160, [2], correction = 0, keepdim = True)
        getitem_62: "f32[512, 16, 1]" = var_mean_31[0]
        getitem_63: "f32[512, 16, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_63: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_160, getitem_63);  convert_element_type_160 = getitem_63 = None
        add_174: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        mul_176: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_31);  sub_63 = rsqrt_31 = None
        mul_177: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_176, arg240_1);  mul_176 = arg240_1 = None
        add_175: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_177, arg241_1);  mul_177 = arg241_1 = None
        convert_element_type_161: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None
        unsqueeze_483: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_161, 3)
        unsqueeze_484: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 4);  unsqueeze_483 = None
        view_608: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_484, _shape_param_609);  unsqueeze_484 = _shape_param_609 = None
        squeeze_160: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_608, 0);  view_608 = None
        unsqueeze_485: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg242_1, 3);  arg242_1 = None
        unsqueeze_486: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 4);  unsqueeze_485 = None
        view_609: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_486, _shape_param_610);  unsqueeze_486 = _shape_param_610 = None
        squeeze_161: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_609, 0);  view_609 = None
        mm_80: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_160, squeeze_161);  squeeze_160 = squeeze_161 = None
        unsqueeze_487: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_80, 0);  mm_80 = None
        view_610: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_487, _shape_param_611);  unsqueeze_487 = _shape_param_611 = None
        permute_387: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_610, [0, 1, 3, 4, 2]);  view_610 = None
        view_611: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_387, _shape_param_612);  permute_387 = _shape_param_612 = None
        add_176: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_611, arg243_1);  arg243_1 = None
        unsqueeze_488: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_176, 4);  add_176 = None
        permute_388: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_488, [1, 2, 0, 4, 3]);  unsqueeze_488 = None
        permute_389: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_388, [0, 1, 2, 4, 3]);  permute_388 = None
        view_612: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_389, _shape_param_613);  permute_389 = _shape_param_613 = None
        unsqueeze_489: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_161, 3)
        unsqueeze_490: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 4);  unsqueeze_489 = None
        view_613: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_490, _shape_param_614);  unsqueeze_490 = _shape_param_614 = None
        squeeze_162: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_613, 0);  view_613 = None
        unsqueeze_491: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, 3);  arg244_1 = None
        unsqueeze_492: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 4);  unsqueeze_491 = None
        view_614: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_492, _shape_param_615);  unsqueeze_492 = _shape_param_615 = None
        squeeze_163: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_614, 0);  view_614 = None
        mm_81: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_162, squeeze_163);  squeeze_162 = squeeze_163 = None
        unsqueeze_493: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_81, 0);  mm_81 = None
        view_615: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_493, _shape_param_616);  unsqueeze_493 = _shape_param_616 = None
        permute_390: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_615, [0, 1, 3, 4, 2]);  view_615 = None
        view_616: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_390, _shape_param_617);  permute_390 = _shape_param_617 = None
        unsqueeze_494: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_616, 4);  view_616 = None
        permute_391: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_494, [1, 2, 4, 0, 3]);  unsqueeze_494 = None
        permute_392: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_391, [0, 1, 4, 3, 2]);  permute_391 = None
        view_617: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_392, _shape_param_618);  permute_392 = _shape_param_618 = None
        bmm_48: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_612, view_617);  view_612 = view_617 = None
        view_618: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_48, _shape_param_619);  bmm_48 = _shape_param_619 = None
        permute_393: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_618, [0, 1, 2, 4, 3]);  view_618 = None
        view_619: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_393, _shape_param_620);  permute_393 = _shape_param_620 = None
        add_177: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_611, arg245_1);  view_611 = arg245_1 = None
        unsqueeze_495: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_177, 4);  add_177 = None
        permute_394: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_495, [1, 2, 0, 4, 3]);  unsqueeze_495 = None
        permute_395: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_394, [0, 1, 2, 4, 3]);  permute_394 = None
        view_620: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_395, _shape_param_621);  permute_395 = _shape_param_621 = None
        convert_element_type_162: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_496: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_162, 3);  convert_element_type_162 = None
        unsqueeze_497: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 4);  unsqueeze_496 = None
        view_621: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_497, _shape_param_622);  unsqueeze_497 = _shape_param_622 = None
        squeeze_164: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_621, 0);  view_621 = None
        unsqueeze_498: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg246_1, 3);  arg246_1 = None
        unsqueeze_499: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 4);  unsqueeze_498 = None
        view_622: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_499, _shape_param_623);  unsqueeze_499 = _shape_param_623 = None
        squeeze_165: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_622, 0);  view_622 = None
        mm_82: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_164, squeeze_165);  squeeze_164 = squeeze_165 = None
        unsqueeze_500: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_82, 0);  mm_82 = None
        view_623: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_500, _shape_param_624);  unsqueeze_500 = _shape_param_624 = None
        permute_396: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_623, [0, 1, 3, 4, 2]);  view_623 = None
        view_624: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_396, _shape_param_625);  permute_396 = _shape_param_625 = None
        unsqueeze_501: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_624, 4);  view_624 = None
        permute_397: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_501, [1, 2, 4, 0, 3]);  unsqueeze_501 = None
        permute_398: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_397, [0, 1, 4, 3, 2]);  permute_397 = None
        view_625: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_398, _shape_param_626);  permute_398 = _shape_param_626 = None
        bmm_49: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_620, view_625);  view_620 = view_625 = None
        view_626: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_49, _shape_param_627);  bmm_49 = _shape_param_627 = None
        permute_399: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_626, [0, 1, 2, 4, 3]);  view_626 = None
        view_627: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_399, _shape_param_628);  permute_399 = _shape_param_628 = None
        view_628: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_627, _shape_param_629);  view_627 = _shape_param_629 = None
        slice_17: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_628, 2, 1, 9223372036854775807);  view_628 = None
        view_629: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_17, _shape_param_630);  slice_17 = _shape_param_630 = None
        iota_18: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_16: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_629, [None, None, None, iota_18]);  view_629 = iota_18 = None
        add_178: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_619, index_16);  view_619 = index_16 = None
        add_179: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_178, 0);  add_178 = None
        mul_178: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_179, 0.125)
        convert_element_type_163: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_178, torch.float32);  mul_178 = None
        eq_16: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_163, convert_element_type_163)
        abs_17: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_163)
        ne_16: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_17, inf);  abs_17 = None
        mul_179: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_16, ne_16);  eq_16 = ne_16 = None
        logical_not_32: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_179);  mul_179 = None
        any_17: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_32, [3], True);  logical_not_32 = None
        logical_not_33: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        convert_element_type_164: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_179, torch.float32);  add_179 = None
        mul_180: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_164, 1);  convert_element_type_164 = None
        amax_32: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_180, [3], True)
        sub_64: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_180, amax_32);  mul_180 = amax_32 = None
        mul_181: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_64, 0.125);  sub_64 = None
        amax_33: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_163, [3], True)
        sub_65: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_163, amax_33);  convert_element_type_163 = amax_33 = None
        where_16: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_33, mul_181, sub_65);  logical_not_33 = mul_181 = sub_65 = None
        exp_16: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_16);  where_16 = None
        sum_17: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [3], True)
        div_17: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_165: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        unsqueeze_502: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_165, 4);  convert_element_type_165 = None
        view_630: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_502, _shape_param_631);  unsqueeze_502 = _shape_param_631 = None
        unsqueeze_503: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_161, 3)
        unsqueeze_504: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 4);  unsqueeze_503 = None
        view_631: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_504, _shape_param_632);  unsqueeze_504 = _shape_param_632 = None
        squeeze_166: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_631, 0);  view_631 = None
        unsqueeze_505: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg247_1, 3);  arg247_1 = None
        unsqueeze_506: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 4);  unsqueeze_505 = None
        view_632: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_506, _shape_param_633);  unsqueeze_506 = _shape_param_633 = None
        squeeze_167: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_632, 0);  view_632 = None
        mm_83: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_166, squeeze_167);  squeeze_166 = squeeze_167 = None
        unsqueeze_507: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_83, 0);  mm_83 = None
        view_633: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_507, _shape_param_634);  unsqueeze_507 = _shape_param_634 = None
        permute_400: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_633, [0, 1, 3, 4, 2]);  view_633 = None
        view_634: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_400, _shape_param_635);  permute_400 = _shape_param_635 = None
        unsqueeze_508: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_634, 4);  view_634 = None
        permute_401: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_508, [4, 1, 2, 3, 0]);  unsqueeze_508 = None
        permute_402: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_401, [1, 2, 4, 3, 0]);  permute_401 = None
        view_635: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_402, _shape_param_636);  permute_402 = _shape_param_636 = None
        bmm_50: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_630, view_635);  view_630 = view_635 = None
        view_636: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_50, _shape_param_637);  bmm_50 = _shape_param_637 = None
        permute_403: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_636, [2, 0, 1, 4, 3]);  view_636 = None
        view_637: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_403, _shape_param_638);  permute_403 = _shape_param_638 = None
        unsqueeze_509: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_637, 4);  view_637 = None
        permute_404: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_509, [0, 1, 4, 3, 2]);  unsqueeze_509 = None
        permute_405: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_404, [0, 1, 3, 4, 2]);  permute_404 = None
        clone_34: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_405, memory_format = torch.contiguous_format);  permute_405 = None
        view_638: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_34, _shape_param_639);  clone_34 = _shape_param_639 = None
        squeeze_168: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_638, 0);  view_638 = None
        unsqueeze_510: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg248_1, 3);  arg248_1 = None
        unsqueeze_511: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 4);  unsqueeze_510 = None
        permute_406: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_511, [3, 4, 0, 2, 1]);  unsqueeze_511 = None
        permute_407: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_406, [3, 4, 2, 0, 1]);  permute_406 = None
        clone_35: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
        view_639: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_35, _shape_param_640);  clone_35 = _shape_param_640 = None
        squeeze_169: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_639, 0);  view_639 = None
        mm_84: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_168, squeeze_169);  squeeze_168 = squeeze_169 = None
        unsqueeze_512: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_84, 0);  mm_84 = None
        view_640: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_512, _shape_param_641);  unsqueeze_512 = _shape_param_641 = None
        permute_408: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_640, [0, 1, 4, 2, 3]);  view_640 = None
        view_641: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_408, _shape_param_642);  permute_408 = _shape_param_642 = None
        add_180: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_641, convert_element_type_161);  view_641 = None
        convert_element_type_166: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_180, torch.float32);  add_180 = None
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_166, [2], correction = 0, keepdim = True)
        getitem_64: "f32[512, 16, 1]" = var_mean_32[0]
        getitem_65: "f32[512, 16, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_66: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_166, getitem_65);  convert_element_type_166 = getitem_65 = None
        add_181: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        mul_182: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_32);  sub_66 = rsqrt_32 = None
        mul_183: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_182, arg249_1);  mul_182 = arg249_1 = None
        add_182: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_183, arg250_1);  mul_183 = arg250_1 = None
        convert_element_type_167: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_182, torch.bfloat16);  add_182 = None
        view_642: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_167, _shape_param_643);  _shape_param_643 = None
        permute_409: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_32: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg252_1, view_642, permute_409);  arg252_1 = view_642 = permute_409 = None
        view_643: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_32, _shape_param_644);  addmm_32 = _shape_param_644 = None
        convert_element_type_168: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_643, torch.float32);  view_643 = None
        mul_184: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_168, 0.5)
        mul_185: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_168, 0.7071067811865476);  convert_element_type_168 = None
        erf_16: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_183: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_186: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_184, add_183);  mul_184 = add_183 = None
        convert_element_type_169: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_186, torch.bfloat16);  mul_186 = None
        view_644: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_169, _shape_param_645);  convert_element_type_169 = _shape_param_645 = None
        permute_410: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_33: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg254_1, view_644, permute_410);  arg254_1 = view_644 = permute_410 = None
        view_645: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_33, _shape_param_646);  addmm_33 = _shape_param_646 = None
        add_184: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_645, convert_element_type_167);  view_645 = convert_element_type_167 = None
        convert_element_type_170: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_184, torch.float32);  add_184 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True)
        getitem_66: "f32[512, 16, 1]" = var_mean_33[0]
        getitem_67: "f32[512, 16, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_67: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_67);  convert_element_type_170 = getitem_67 = None
        add_185: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        mul_187: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_33);  sub_67 = rsqrt_33 = None
        mul_188: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_187, arg255_1);  mul_187 = arg255_1 = None
        add_186: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_188, arg256_1);  mul_188 = arg256_1 = None
        convert_element_type_171: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16);  add_186 = None
        unsqueeze_513: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_171, 3)
        unsqueeze_514: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 4);  unsqueeze_513 = None
        view_646: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_514, _shape_param_647);  unsqueeze_514 = _shape_param_647 = None
        squeeze_170: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_646, 0);  view_646 = None
        unsqueeze_515: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg257_1, 3);  arg257_1 = None
        unsqueeze_516: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 4);  unsqueeze_515 = None
        view_647: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_516, _shape_param_648);  unsqueeze_516 = _shape_param_648 = None
        squeeze_171: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_647, 0);  view_647 = None
        mm_85: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_170, squeeze_171);  squeeze_170 = squeeze_171 = None
        unsqueeze_517: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_85, 0);  mm_85 = None
        view_648: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_517, _shape_param_649);  unsqueeze_517 = _shape_param_649 = None
        permute_411: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_648, [0, 1, 3, 4, 2]);  view_648 = None
        view_649: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_411, _shape_param_650);  permute_411 = _shape_param_650 = None
        add_187: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_649, arg258_1);  arg258_1 = None
        unsqueeze_518: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_187, 4);  add_187 = None
        permute_412: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_518, [1, 2, 0, 4, 3]);  unsqueeze_518 = None
        permute_413: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_412, [0, 1, 2, 4, 3]);  permute_412 = None
        view_650: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_413, _shape_param_651);  permute_413 = _shape_param_651 = None
        unsqueeze_519: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_171, 3)
        unsqueeze_520: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 4);  unsqueeze_519 = None
        view_651: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_520, _shape_param_652);  unsqueeze_520 = _shape_param_652 = None
        squeeze_172: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_651, 0);  view_651 = None
        unsqueeze_521: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, 3);  arg259_1 = None
        unsqueeze_522: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 4);  unsqueeze_521 = None
        view_652: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_522, _shape_param_653);  unsqueeze_522 = _shape_param_653 = None
        squeeze_173: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_652, 0);  view_652 = None
        mm_86: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_172, squeeze_173);  squeeze_172 = squeeze_173 = None
        unsqueeze_523: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_86, 0);  mm_86 = None
        view_653: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_523, _shape_param_654);  unsqueeze_523 = _shape_param_654 = None
        permute_414: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_653, [0, 1, 3, 4, 2]);  view_653 = None
        view_654: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_414, _shape_param_655);  permute_414 = _shape_param_655 = None
        unsqueeze_524: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_654, 4);  view_654 = None
        permute_415: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_524, [1, 2, 4, 0, 3]);  unsqueeze_524 = None
        permute_416: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_415, [0, 1, 4, 3, 2]);  permute_415 = None
        view_655: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_416, _shape_param_656);  permute_416 = _shape_param_656 = None
        bmm_51: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_650, view_655);  view_650 = view_655 = None
        view_656: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_51, _shape_param_657);  bmm_51 = _shape_param_657 = None
        permute_417: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_656, [0, 1, 2, 4, 3]);  view_656 = None
        view_657: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_417, _shape_param_658);  permute_417 = _shape_param_658 = None
        add_188: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_649, arg260_1);  view_649 = arg260_1 = None
        unsqueeze_525: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_188, 4);  add_188 = None
        permute_418: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_525, [1, 2, 0, 4, 3]);  unsqueeze_525 = None
        permute_419: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_418, [0, 1, 2, 4, 3]);  permute_418 = None
        view_658: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_419, _shape_param_659);  permute_419 = _shape_param_659 = None
        convert_element_type_172: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_526: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_172, 3);  convert_element_type_172 = None
        unsqueeze_527: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 4);  unsqueeze_526 = None
        view_659: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_527, _shape_param_660);  unsqueeze_527 = _shape_param_660 = None
        squeeze_174: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_659, 0);  view_659 = None
        unsqueeze_528: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg261_1, 3);  arg261_1 = None
        unsqueeze_529: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 4);  unsqueeze_528 = None
        view_660: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_529, _shape_param_661);  unsqueeze_529 = _shape_param_661 = None
        squeeze_175: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_660, 0);  view_660 = None
        mm_87: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_174, squeeze_175);  squeeze_174 = squeeze_175 = None
        unsqueeze_530: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_87, 0);  mm_87 = None
        view_661: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_530, _shape_param_662);  unsqueeze_530 = _shape_param_662 = None
        permute_420: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_661, [0, 1, 3, 4, 2]);  view_661 = None
        view_662: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_420, _shape_param_663);  permute_420 = _shape_param_663 = None
        unsqueeze_531: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_662, 4);  view_662 = None
        permute_421: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_531, [1, 2, 4, 0, 3]);  unsqueeze_531 = None
        permute_422: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_421, [0, 1, 4, 3, 2]);  permute_421 = None
        view_663: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_422, _shape_param_664);  permute_422 = _shape_param_664 = None
        bmm_52: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_658, view_663);  view_658 = view_663 = None
        view_664: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_52, _shape_param_665);  bmm_52 = _shape_param_665 = None
        permute_423: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_664, [0, 1, 2, 4, 3]);  view_664 = None
        view_665: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_423, _shape_param_666);  permute_423 = _shape_param_666 = None
        view_666: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_665, _shape_param_667);  view_665 = _shape_param_667 = None
        slice_18: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_666, 2, 1, 9223372036854775807);  view_666 = None
        view_667: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_18, _shape_param_668);  slice_18 = _shape_param_668 = None
        iota_19: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_17: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_667, [None, None, None, iota_19]);  view_667 = iota_19 = None
        add_189: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_657, index_17);  view_657 = index_17 = None
        add_190: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_189, 0);  add_189 = None
        mul_189: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_190, 0.125)
        convert_element_type_173: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_189, torch.float32);  mul_189 = None
        eq_17: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_173, convert_element_type_173)
        abs_18: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_173)
        ne_17: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_18, inf);  abs_18 = None
        mul_190: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_17, ne_17);  eq_17 = ne_17 = None
        logical_not_34: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_190);  mul_190 = None
        any_18: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_34, [3], True);  logical_not_34 = None
        logical_not_35: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        convert_element_type_174: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_190, torch.float32);  add_190 = None
        mul_191: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_174, 1);  convert_element_type_174 = None
        amax_34: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_191, [3], True)
        sub_68: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_191, amax_34);  mul_191 = amax_34 = None
        mul_192: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_68, 0.125);  sub_68 = None
        amax_35: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_173, [3], True)
        sub_69: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_173, amax_35);  convert_element_type_173 = amax_35 = None
        where_17: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_35, mul_192, sub_69);  logical_not_35 = mul_192 = sub_69 = None
        exp_17: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_17);  where_17 = None
        sum_18: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [3], True)
        div_18: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_175: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        unsqueeze_532: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_175, 4);  convert_element_type_175 = None
        view_668: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_532, _shape_param_669);  unsqueeze_532 = _shape_param_669 = None
        unsqueeze_533: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_171, 3)
        unsqueeze_534: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 4);  unsqueeze_533 = None
        view_669: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_534, _shape_param_670);  unsqueeze_534 = _shape_param_670 = None
        squeeze_176: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_669, 0);  view_669 = None
        unsqueeze_535: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg262_1, 3);  arg262_1 = None
        unsqueeze_536: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 4);  unsqueeze_535 = None
        view_670: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_536, _shape_param_671);  unsqueeze_536 = _shape_param_671 = None
        squeeze_177: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_670, 0);  view_670 = None
        mm_88: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_176, squeeze_177);  squeeze_176 = squeeze_177 = None
        unsqueeze_537: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_88, 0);  mm_88 = None
        view_671: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_537, _shape_param_672);  unsqueeze_537 = _shape_param_672 = None
        permute_424: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_671, [0, 1, 3, 4, 2]);  view_671 = None
        view_672: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_424, _shape_param_673);  permute_424 = _shape_param_673 = None
        unsqueeze_538: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_672, 4);  view_672 = None
        permute_425: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_538, [4, 1, 2, 3, 0]);  unsqueeze_538 = None
        permute_426: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_425, [1, 2, 4, 3, 0]);  permute_425 = None
        view_673: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_426, _shape_param_674);  permute_426 = _shape_param_674 = None
        bmm_53: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_668, view_673);  view_668 = view_673 = None
        view_674: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_53, _shape_param_675);  bmm_53 = _shape_param_675 = None
        permute_427: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_674, [2, 0, 1, 4, 3]);  view_674 = None
        view_675: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_427, _shape_param_676);  permute_427 = _shape_param_676 = None
        unsqueeze_539: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_675, 4);  view_675 = None
        permute_428: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_539, [0, 1, 4, 3, 2]);  unsqueeze_539 = None
        permute_429: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_428, [0, 1, 3, 4, 2]);  permute_428 = None
        clone_36: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_429, memory_format = torch.contiguous_format);  permute_429 = None
        view_676: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_36, _shape_param_677);  clone_36 = _shape_param_677 = None
        squeeze_178: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_676, 0);  view_676 = None
        unsqueeze_540: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg263_1, 3);  arg263_1 = None
        unsqueeze_541: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 4);  unsqueeze_540 = None
        permute_430: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_541, [3, 4, 0, 2, 1]);  unsqueeze_541 = None
        permute_431: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_430, [3, 4, 2, 0, 1]);  permute_430 = None
        clone_37: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_431, memory_format = torch.contiguous_format);  permute_431 = None
        view_677: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_37, _shape_param_678);  clone_37 = _shape_param_678 = None
        squeeze_179: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_677, 0);  view_677 = None
        mm_89: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_178, squeeze_179);  squeeze_178 = squeeze_179 = None
        unsqueeze_542: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_89, 0);  mm_89 = None
        view_678: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_542, _shape_param_679);  unsqueeze_542 = _shape_param_679 = None
        permute_432: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_678, [0, 1, 4, 2, 3]);  view_678 = None
        view_679: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_432, _shape_param_680);  permute_432 = _shape_param_680 = None
        add_191: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_679, convert_element_type_171);  view_679 = None
        convert_element_type_176: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_191, torch.float32);  add_191 = None
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_176, [2], correction = 0, keepdim = True)
        getitem_68: "f32[512, 16, 1]" = var_mean_34[0]
        getitem_69: "f32[512, 16, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_70: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_176, getitem_69);  convert_element_type_176 = getitem_69 = None
        add_192: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_193: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_34);  sub_70 = rsqrt_34 = None
        mul_194: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_193, arg264_1);  mul_193 = arg264_1 = None
        add_193: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_194, arg265_1);  mul_194 = arg265_1 = None
        convert_element_type_177: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16);  add_193 = None
        view_680: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_177, _shape_param_681);  _shape_param_681 = None
        permute_433: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
        addmm_34: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg267_1, view_680, permute_433);  arg267_1 = view_680 = permute_433 = None
        view_681: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_34, _shape_param_682);  addmm_34 = _shape_param_682 = None
        convert_element_type_178: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_681, torch.float32);  view_681 = None
        mul_195: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_178, 0.5)
        mul_196: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_178, 0.7071067811865476);  convert_element_type_178 = None
        erf_17: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_196);  mul_196 = None
        add_194: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_197: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_195, add_194);  mul_195 = add_194 = None
        convert_element_type_179: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_197, torch.bfloat16);  mul_197 = None
        view_682: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_179, _shape_param_683);  convert_element_type_179 = _shape_param_683 = None
        permute_434: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg268_1, [1, 0]);  arg268_1 = None
        addmm_35: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg269_1, view_682, permute_434);  arg269_1 = view_682 = permute_434 = None
        view_683: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_35, _shape_param_684);  addmm_35 = _shape_param_684 = None
        add_195: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_683, convert_element_type_177);  view_683 = convert_element_type_177 = None
        convert_element_type_180: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_195, torch.float32);  add_195 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_180, [2], correction = 0, keepdim = True)
        getitem_70: "f32[512, 16, 1]" = var_mean_35[0]
        getitem_71: "f32[512, 16, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_71: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_180, getitem_71);  convert_element_type_180 = getitem_71 = None
        add_196: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        mul_198: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_35);  sub_71 = rsqrt_35 = None
        mul_199: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_198, arg270_1);  mul_198 = arg270_1 = None
        add_197: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_199, arg271_1);  mul_199 = arg271_1 = None
        convert_element_type_181: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16);  add_197 = None
        unsqueeze_543: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_181, 3)
        unsqueeze_544: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 4);  unsqueeze_543 = None
        view_684: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_544, _shape_param_685);  unsqueeze_544 = _shape_param_685 = None
        squeeze_180: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_684, 0);  view_684 = None
        unsqueeze_545: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg272_1, 3);  arg272_1 = None
        unsqueeze_546: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 4);  unsqueeze_545 = None
        view_685: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_546, _shape_param_686);  unsqueeze_546 = _shape_param_686 = None
        squeeze_181: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_685, 0);  view_685 = None
        mm_90: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_180, squeeze_181);  squeeze_180 = squeeze_181 = None
        unsqueeze_547: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_90, 0);  mm_90 = None
        view_686: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_547, _shape_param_687);  unsqueeze_547 = _shape_param_687 = None
        permute_435: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_686, [0, 1, 3, 4, 2]);  view_686 = None
        view_687: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_435, _shape_param_688);  permute_435 = _shape_param_688 = None
        add_198: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_687, arg273_1);  arg273_1 = None
        unsqueeze_548: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_198, 4);  add_198 = None
        permute_436: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_548, [1, 2, 0, 4, 3]);  unsqueeze_548 = None
        permute_437: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_436, [0, 1, 2, 4, 3]);  permute_436 = None
        view_688: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_437, _shape_param_689);  permute_437 = _shape_param_689 = None
        unsqueeze_549: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_181, 3)
        unsqueeze_550: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 4);  unsqueeze_549 = None
        view_689: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_550, _shape_param_690);  unsqueeze_550 = _shape_param_690 = None
        squeeze_182: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_689, 0);  view_689 = None
        unsqueeze_551: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, 3);  arg274_1 = None
        unsqueeze_552: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 4);  unsqueeze_551 = None
        view_690: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_552, _shape_param_691);  unsqueeze_552 = _shape_param_691 = None
        squeeze_183: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_690, 0);  view_690 = None
        mm_91: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_182, squeeze_183);  squeeze_182 = squeeze_183 = None
        unsqueeze_553: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_91, 0);  mm_91 = None
        view_691: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_553, _shape_param_692);  unsqueeze_553 = _shape_param_692 = None
        permute_438: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_691, [0, 1, 3, 4, 2]);  view_691 = None
        view_692: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_438, _shape_param_693);  permute_438 = _shape_param_693 = None
        unsqueeze_554: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_692, 4);  view_692 = None
        permute_439: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_554, [1, 2, 4, 0, 3]);  unsqueeze_554 = None
        permute_440: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_439, [0, 1, 4, 3, 2]);  permute_439 = None
        view_693: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_440, _shape_param_694);  permute_440 = _shape_param_694 = None
        bmm_54: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_688, view_693);  view_688 = view_693 = None
        view_694: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_54, _shape_param_695);  bmm_54 = _shape_param_695 = None
        permute_441: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_694, [0, 1, 2, 4, 3]);  view_694 = None
        view_695: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_441, _shape_param_696);  permute_441 = _shape_param_696 = None
        add_199: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_687, arg275_1);  view_687 = arg275_1 = None
        unsqueeze_555: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_199, 4);  add_199 = None
        permute_442: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_555, [1, 2, 0, 4, 3]);  unsqueeze_555 = None
        permute_443: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_442, [0, 1, 2, 4, 3]);  permute_442 = None
        view_696: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_443, _shape_param_697);  permute_443 = _shape_param_697 = None
        convert_element_type_182: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_556: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_182, 3);  convert_element_type_182 = None
        unsqueeze_557: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 4);  unsqueeze_556 = None
        view_697: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_557, _shape_param_698);  unsqueeze_557 = _shape_param_698 = None
        squeeze_184: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_697, 0);  view_697 = None
        unsqueeze_558: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg276_1, 3);  arg276_1 = None
        unsqueeze_559: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 4);  unsqueeze_558 = None
        view_698: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_559, _shape_param_699);  unsqueeze_559 = _shape_param_699 = None
        squeeze_185: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_698, 0);  view_698 = None
        mm_92: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_184, squeeze_185);  squeeze_184 = squeeze_185 = None
        unsqueeze_560: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_92, 0);  mm_92 = None
        view_699: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_560, _shape_param_700);  unsqueeze_560 = _shape_param_700 = None
        permute_444: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_699, [0, 1, 3, 4, 2]);  view_699 = None
        view_700: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_444, _shape_param_701);  permute_444 = _shape_param_701 = None
        unsqueeze_561: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_700, 4);  view_700 = None
        permute_445: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_561, [1, 2, 4, 0, 3]);  unsqueeze_561 = None
        permute_446: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_445, [0, 1, 4, 3, 2]);  permute_445 = None
        view_701: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_446, _shape_param_702);  permute_446 = _shape_param_702 = None
        bmm_55: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_696, view_701);  view_696 = view_701 = None
        view_702: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_55, _shape_param_703);  bmm_55 = _shape_param_703 = None
        permute_447: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_702, [0, 1, 2, 4, 3]);  view_702 = None
        view_703: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_447, _shape_param_704);  permute_447 = _shape_param_704 = None
        view_704: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_703, _shape_param_705);  view_703 = _shape_param_705 = None
        slice_19: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_704, 2, 1, 9223372036854775807);  view_704 = None
        view_705: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_19, _shape_param_706);  slice_19 = _shape_param_706 = None
        iota_20: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_18: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_705, [None, None, None, iota_20]);  view_705 = iota_20 = None
        add_200: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_695, index_18);  view_695 = index_18 = None
        add_201: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_200, 0);  add_200 = None
        mul_200: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_201, 0.125)
        convert_element_type_183: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_200, torch.float32);  mul_200 = None
        eq_18: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_183, convert_element_type_183)
        abs_19: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_183)
        ne_18: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_19, inf);  abs_19 = None
        mul_201: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_18, ne_18);  eq_18 = ne_18 = None
        logical_not_36: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_201);  mul_201 = None
        any_19: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_36, [3], True);  logical_not_36 = None
        logical_not_37: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        convert_element_type_184: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_201, torch.float32);  add_201 = None
        mul_202: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_184, 1);  convert_element_type_184 = None
        amax_36: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_202, [3], True)
        sub_72: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_202, amax_36);  mul_202 = amax_36 = None
        mul_203: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_72, 0.125);  sub_72 = None
        amax_37: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_183, [3], True)
        sub_73: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_183, amax_37);  convert_element_type_183 = amax_37 = None
        where_18: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_37, mul_203, sub_73);  logical_not_37 = mul_203 = sub_73 = None
        exp_18: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_18);  where_18 = None
        sum_19: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [3], True)
        div_19: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_185: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        unsqueeze_562: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_185, 4);  convert_element_type_185 = None
        view_706: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_562, _shape_param_707);  unsqueeze_562 = _shape_param_707 = None
        unsqueeze_563: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_181, 3)
        unsqueeze_564: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 4);  unsqueeze_563 = None
        view_707: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_564, _shape_param_708);  unsqueeze_564 = _shape_param_708 = None
        squeeze_186: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_707, 0);  view_707 = None
        unsqueeze_565: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg277_1, 3);  arg277_1 = None
        unsqueeze_566: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 4);  unsqueeze_565 = None
        view_708: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_566, _shape_param_709);  unsqueeze_566 = _shape_param_709 = None
        squeeze_187: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_708, 0);  view_708 = None
        mm_93: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_186, squeeze_187);  squeeze_186 = squeeze_187 = None
        unsqueeze_567: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_93, 0);  mm_93 = None
        view_709: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_567, _shape_param_710);  unsqueeze_567 = _shape_param_710 = None
        permute_448: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_709, [0, 1, 3, 4, 2]);  view_709 = None
        view_710: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_448, _shape_param_711);  permute_448 = _shape_param_711 = None
        unsqueeze_568: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_710, 4);  view_710 = None
        permute_449: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_568, [4, 1, 2, 3, 0]);  unsqueeze_568 = None
        permute_450: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_449, [1, 2, 4, 3, 0]);  permute_449 = None
        view_711: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_450, _shape_param_712);  permute_450 = _shape_param_712 = None
        bmm_56: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_706, view_711);  view_706 = view_711 = None
        view_712: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_56, _shape_param_713);  bmm_56 = _shape_param_713 = None
        permute_451: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_712, [2, 0, 1, 4, 3]);  view_712 = None
        view_713: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_451, _shape_param_714);  permute_451 = _shape_param_714 = None
        unsqueeze_569: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_713, 4);  view_713 = None
        permute_452: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_569, [0, 1, 4, 3, 2]);  unsqueeze_569 = None
        permute_453: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_452, [0, 1, 3, 4, 2]);  permute_452 = None
        clone_38: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
        view_714: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_38, _shape_param_715);  clone_38 = _shape_param_715 = None
        squeeze_188: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_714, 0);  view_714 = None
        unsqueeze_570: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg278_1, 3);  arg278_1 = None
        unsqueeze_571: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 4);  unsqueeze_570 = None
        permute_454: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_571, [3, 4, 0, 2, 1]);  unsqueeze_571 = None
        permute_455: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_454, [3, 4, 2, 0, 1]);  permute_454 = None
        clone_39: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
        view_715: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_39, _shape_param_716);  clone_39 = _shape_param_716 = None
        squeeze_189: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_715, 0);  view_715 = None
        mm_94: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_188, squeeze_189);  squeeze_188 = squeeze_189 = None
        unsqueeze_572: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_94, 0);  mm_94 = None
        view_716: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_572, _shape_param_717);  unsqueeze_572 = _shape_param_717 = None
        permute_456: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_716, [0, 1, 4, 2, 3]);  view_716 = None
        view_717: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_456, _shape_param_718);  permute_456 = _shape_param_718 = None
        add_202: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_717, convert_element_type_181);  view_717 = None
        convert_element_type_186: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_202, torch.float32);  add_202 = None
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_186, [2], correction = 0, keepdim = True)
        getitem_72: "f32[512, 16, 1]" = var_mean_36[0]
        getitem_73: "f32[512, 16, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_74: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_186, getitem_73);  convert_element_type_186 = getitem_73 = None
        add_203: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_204: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_36);  sub_74 = rsqrt_36 = None
        mul_205: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_204, arg279_1);  mul_204 = arg279_1 = None
        add_204: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_205, arg280_1);  mul_205 = arg280_1 = None
        convert_element_type_187: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None
        view_718: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_187, _shape_param_719);  _shape_param_719 = None
        permute_457: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_36: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg282_1, view_718, permute_457);  arg282_1 = view_718 = permute_457 = None
        view_719: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_36, _shape_param_720);  addmm_36 = _shape_param_720 = None
        convert_element_type_188: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_719, torch.float32);  view_719 = None
        mul_206: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_188, 0.5)
        mul_207: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_188, 0.7071067811865476);  convert_element_type_188 = None
        erf_18: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_205: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_208: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_206, add_205);  mul_206 = add_205 = None
        convert_element_type_189: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None
        view_720: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_189, _shape_param_721);  convert_element_type_189 = _shape_param_721 = None
        permute_458: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_37: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg284_1, view_720, permute_458);  arg284_1 = view_720 = permute_458 = None
        view_721: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_37, _shape_param_722);  addmm_37 = _shape_param_722 = None
        add_206: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_721, convert_element_type_187);  view_721 = convert_element_type_187 = None
        convert_element_type_190: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_206, torch.float32);  add_206 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_190, [2], correction = 0, keepdim = True)
        getitem_74: "f32[512, 16, 1]" = var_mean_37[0]
        getitem_75: "f32[512, 16, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_75: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_190, getitem_75);  convert_element_type_190 = getitem_75 = None
        add_207: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        mul_209: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_37);  sub_75 = rsqrt_37 = None
        mul_210: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_209, arg285_1);  mul_209 = arg285_1 = None
        add_208: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_210, arg286_1);  mul_210 = arg286_1 = None
        convert_element_type_191: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16);  add_208 = None
        unsqueeze_573: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_191, 3)
        unsqueeze_574: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 4);  unsqueeze_573 = None
        view_722: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_574, _shape_param_723);  unsqueeze_574 = _shape_param_723 = None
        squeeze_190: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_722, 0);  view_722 = None
        unsqueeze_575: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg287_1, 3);  arg287_1 = None
        unsqueeze_576: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 4);  unsqueeze_575 = None
        view_723: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_576, _shape_param_724);  unsqueeze_576 = _shape_param_724 = None
        squeeze_191: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_723, 0);  view_723 = None
        mm_95: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_190, squeeze_191);  squeeze_190 = squeeze_191 = None
        unsqueeze_577: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_95, 0);  mm_95 = None
        view_724: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_577, _shape_param_725);  unsqueeze_577 = _shape_param_725 = None
        permute_459: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_724, [0, 1, 3, 4, 2]);  view_724 = None
        view_725: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_459, _shape_param_726);  permute_459 = _shape_param_726 = None
        add_209: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_725, arg288_1);  arg288_1 = None
        unsqueeze_578: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_209, 4);  add_209 = None
        permute_460: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_578, [1, 2, 0, 4, 3]);  unsqueeze_578 = None
        permute_461: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_460, [0, 1, 2, 4, 3]);  permute_460 = None
        view_726: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_461, _shape_param_727);  permute_461 = _shape_param_727 = None
        unsqueeze_579: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_191, 3)
        unsqueeze_580: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 4);  unsqueeze_579 = None
        view_727: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_580, _shape_param_728);  unsqueeze_580 = _shape_param_728 = None
        squeeze_192: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_727, 0);  view_727 = None
        unsqueeze_581: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, 3);  arg289_1 = None
        unsqueeze_582: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 4);  unsqueeze_581 = None
        view_728: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_582, _shape_param_729);  unsqueeze_582 = _shape_param_729 = None
        squeeze_193: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_728, 0);  view_728 = None
        mm_96: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_192, squeeze_193);  squeeze_192 = squeeze_193 = None
        unsqueeze_583: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_96, 0);  mm_96 = None
        view_729: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_583, _shape_param_730);  unsqueeze_583 = _shape_param_730 = None
        permute_462: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_729, [0, 1, 3, 4, 2]);  view_729 = None
        view_730: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_462, _shape_param_731);  permute_462 = _shape_param_731 = None
        unsqueeze_584: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_730, 4);  view_730 = None
        permute_463: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_584, [1, 2, 4, 0, 3]);  unsqueeze_584 = None
        permute_464: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_463, [0, 1, 4, 3, 2]);  permute_463 = None
        view_731: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_464, _shape_param_732);  permute_464 = _shape_param_732 = None
        bmm_57: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_726, view_731);  view_726 = view_731 = None
        view_732: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_57, _shape_param_733);  bmm_57 = _shape_param_733 = None
        permute_465: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_732, [0, 1, 2, 4, 3]);  view_732 = None
        view_733: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_465, _shape_param_734);  permute_465 = _shape_param_734 = None
        add_210: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_725, arg290_1);  view_725 = arg290_1 = None
        unsqueeze_585: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_210, 4);  add_210 = None
        permute_466: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_585, [1, 2, 0, 4, 3]);  unsqueeze_585 = None
        permute_467: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_466, [0, 1, 2, 4, 3]);  permute_466 = None
        view_734: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_467, _shape_param_735);  permute_467 = _shape_param_735 = None
        convert_element_type_192: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_586: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_192, 3);  convert_element_type_192 = None
        unsqueeze_587: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 4);  unsqueeze_586 = None
        view_735: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_587, _shape_param_736);  unsqueeze_587 = _shape_param_736 = None
        squeeze_194: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_735, 0);  view_735 = None
        unsqueeze_588: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg291_1, 3);  arg291_1 = None
        unsqueeze_589: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 4);  unsqueeze_588 = None
        view_736: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_589, _shape_param_737);  unsqueeze_589 = _shape_param_737 = None
        squeeze_195: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_736, 0);  view_736 = None
        mm_97: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_194, squeeze_195);  squeeze_194 = squeeze_195 = None
        unsqueeze_590: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_97, 0);  mm_97 = None
        view_737: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_590, _shape_param_738);  unsqueeze_590 = _shape_param_738 = None
        permute_468: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_737, [0, 1, 3, 4, 2]);  view_737 = None
        view_738: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_468, _shape_param_739);  permute_468 = _shape_param_739 = None
        unsqueeze_591: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_738, 4);  view_738 = None
        permute_469: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_591, [1, 2, 4, 0, 3]);  unsqueeze_591 = None
        permute_470: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_469, [0, 1, 4, 3, 2]);  permute_469 = None
        view_739: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_470, _shape_param_740);  permute_470 = _shape_param_740 = None
        bmm_58: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_734, view_739);  view_734 = view_739 = None
        view_740: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_58, _shape_param_741);  bmm_58 = _shape_param_741 = None
        permute_471: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_740, [0, 1, 2, 4, 3]);  view_740 = None
        view_741: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_471, _shape_param_742);  permute_471 = _shape_param_742 = None
        view_742: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_741, _shape_param_743);  view_741 = _shape_param_743 = None
        slice_20: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_742, 2, 1, 9223372036854775807);  view_742 = None
        view_743: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_20, _shape_param_744);  slice_20 = _shape_param_744 = None
        iota_21: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_19: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_743, [None, None, None, iota_21]);  view_743 = iota_21 = None
        add_211: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_733, index_19);  view_733 = index_19 = None
        add_212: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_211, 0);  add_211 = None
        mul_211: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_212, 0.125)
        convert_element_type_193: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_211, torch.float32);  mul_211 = None
        eq_19: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_193, convert_element_type_193)
        abs_20: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_193)
        ne_19: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_20, inf);  abs_20 = None
        mul_212: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_19, ne_19);  eq_19 = ne_19 = None
        logical_not_38: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_212);  mul_212 = None
        any_20: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_38, [3], True);  logical_not_38 = None
        logical_not_39: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        convert_element_type_194: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_212, torch.float32);  add_212 = None
        mul_213: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_194, 1);  convert_element_type_194 = None
        amax_38: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_213, [3], True)
        sub_76: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_213, amax_38);  mul_213 = amax_38 = None
        mul_214: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_76, 0.125);  sub_76 = None
        amax_39: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_193, [3], True)
        sub_77: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_193, amax_39);  convert_element_type_193 = amax_39 = None
        where_19: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_39, mul_214, sub_77);  logical_not_39 = mul_214 = sub_77 = None
        exp_19: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_19);  where_19 = None
        sum_20: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [3], True)
        div_20: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_195: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        unsqueeze_592: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_195, 4);  convert_element_type_195 = None
        view_744: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_592, _shape_param_745);  unsqueeze_592 = _shape_param_745 = None
        unsqueeze_593: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_191, 3)
        unsqueeze_594: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 4);  unsqueeze_593 = None
        view_745: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_594, _shape_param_746);  unsqueeze_594 = _shape_param_746 = None
        squeeze_196: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_745, 0);  view_745 = None
        unsqueeze_595: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg292_1, 3);  arg292_1 = None
        unsqueeze_596: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 4);  unsqueeze_595 = None
        view_746: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_596, _shape_param_747);  unsqueeze_596 = _shape_param_747 = None
        squeeze_197: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_746, 0);  view_746 = None
        mm_98: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_196, squeeze_197);  squeeze_196 = squeeze_197 = None
        unsqueeze_597: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_98, 0);  mm_98 = None
        view_747: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_597, _shape_param_748);  unsqueeze_597 = _shape_param_748 = None
        permute_472: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_747, [0, 1, 3, 4, 2]);  view_747 = None
        view_748: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_472, _shape_param_749);  permute_472 = _shape_param_749 = None
        unsqueeze_598: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_748, 4);  view_748 = None
        permute_473: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_598, [4, 1, 2, 3, 0]);  unsqueeze_598 = None
        permute_474: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_473, [1, 2, 4, 3, 0]);  permute_473 = None
        view_749: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_474, _shape_param_750);  permute_474 = _shape_param_750 = None
        bmm_59: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_744, view_749);  view_744 = view_749 = None
        view_750: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_59, _shape_param_751);  bmm_59 = _shape_param_751 = None
        permute_475: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_750, [2, 0, 1, 4, 3]);  view_750 = None
        view_751: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_475, _shape_param_752);  permute_475 = _shape_param_752 = None
        unsqueeze_599: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_751, 4);  view_751 = None
        permute_476: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_599, [0, 1, 4, 3, 2]);  unsqueeze_599 = None
        permute_477: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_476, [0, 1, 3, 4, 2]);  permute_476 = None
        clone_40: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_477, memory_format = torch.contiguous_format);  permute_477 = None
        view_752: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_40, _shape_param_753);  clone_40 = _shape_param_753 = None
        squeeze_198: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_752, 0);  view_752 = None
        unsqueeze_600: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg293_1, 3);  arg293_1 = None
        unsqueeze_601: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 4);  unsqueeze_600 = None
        permute_478: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_601, [3, 4, 0, 2, 1]);  unsqueeze_601 = None
        permute_479: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_478, [3, 4, 2, 0, 1]);  permute_478 = None
        clone_41: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        view_753: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_41, _shape_param_754);  clone_41 = _shape_param_754 = None
        squeeze_199: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_753, 0);  view_753 = None
        mm_99: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_198, squeeze_199);  squeeze_198 = squeeze_199 = None
        unsqueeze_602: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_99, 0);  mm_99 = None
        view_754: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_602, _shape_param_755);  unsqueeze_602 = _shape_param_755 = None
        permute_480: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_754, [0, 1, 4, 2, 3]);  view_754 = None
        view_755: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_480, _shape_param_756);  permute_480 = _shape_param_756 = None
        add_213: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_755, convert_element_type_191);  view_755 = None
        convert_element_type_196: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_213, torch.float32);  add_213 = None
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_196, [2], correction = 0, keepdim = True)
        getitem_76: "f32[512, 16, 1]" = var_mean_38[0]
        getitem_77: "f32[512, 16, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_78: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_196, getitem_77);  convert_element_type_196 = getitem_77 = None
        add_214: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        mul_215: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_38);  sub_78 = rsqrt_38 = None
        mul_216: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_215, arg294_1);  mul_215 = arg294_1 = None
        add_215: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_216, arg295_1);  mul_216 = arg295_1 = None
        convert_element_type_197: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16);  add_215 = None
        view_756: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_197, _shape_param_757);  _shape_param_757 = None
        permute_481: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg296_1, [1, 0]);  arg296_1 = None
        addmm_38: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg297_1, view_756, permute_481);  arg297_1 = view_756 = permute_481 = None
        view_757: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_38, _shape_param_758);  addmm_38 = _shape_param_758 = None
        convert_element_type_198: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_757, torch.float32);  view_757 = None
        mul_217: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.5)
        mul_218: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.7071067811865476);  convert_element_type_198 = None
        erf_19: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
        add_216: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_219: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_217, add_216);  mul_217 = add_216 = None
        convert_element_type_199: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_219, torch.bfloat16);  mul_219 = None
        view_758: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_199, _shape_param_759);  convert_element_type_199 = _shape_param_759 = None
        permute_482: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg298_1, [1, 0]);  arg298_1 = None
        addmm_39: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg299_1, view_758, permute_482);  arg299_1 = view_758 = permute_482 = None
        view_759: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_39, _shape_param_760);  addmm_39 = _shape_param_760 = None
        add_217: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_759, convert_element_type_197);  view_759 = convert_element_type_197 = None
        convert_element_type_200: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_217, torch.float32);  add_217 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_200, [2], correction = 0, keepdim = True)
        getitem_78: "f32[512, 16, 1]" = var_mean_39[0]
        getitem_79: "f32[512, 16, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_79: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_200, getitem_79);  convert_element_type_200 = getitem_79 = None
        add_218: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        mul_220: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_39);  sub_79 = rsqrt_39 = None
        mul_221: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_220, arg300_1);  mul_220 = arg300_1 = None
        add_219: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_221, arg301_1);  mul_221 = arg301_1 = None
        convert_element_type_201: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None
        unsqueeze_603: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_201, 3)
        unsqueeze_604: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 4);  unsqueeze_603 = None
        view_760: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_604, _shape_param_761);  unsqueeze_604 = _shape_param_761 = None
        squeeze_200: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_760, 0);  view_760 = None
        unsqueeze_605: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg302_1, 3);  arg302_1 = None
        unsqueeze_606: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 4);  unsqueeze_605 = None
        view_761: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_606, _shape_param_762);  unsqueeze_606 = _shape_param_762 = None
        squeeze_201: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_761, 0);  view_761 = None
        mm_100: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_200, squeeze_201);  squeeze_200 = squeeze_201 = None
        unsqueeze_607: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_100, 0);  mm_100 = None
        view_762: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_607, _shape_param_763);  unsqueeze_607 = _shape_param_763 = None
        permute_483: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_762, [0, 1, 3, 4, 2]);  view_762 = None
        view_763: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_483, _shape_param_764);  permute_483 = _shape_param_764 = None
        add_220: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_763, arg303_1);  arg303_1 = None
        unsqueeze_608: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_220, 4);  add_220 = None
        permute_484: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_608, [1, 2, 0, 4, 3]);  unsqueeze_608 = None
        permute_485: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_484, [0, 1, 2, 4, 3]);  permute_484 = None
        view_764: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_485, _shape_param_765);  permute_485 = _shape_param_765 = None
        unsqueeze_609: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_201, 3)
        unsqueeze_610: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 4);  unsqueeze_609 = None
        view_765: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_610, _shape_param_766);  unsqueeze_610 = _shape_param_766 = None
        squeeze_202: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_765, 0);  view_765 = None
        unsqueeze_611: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, 3);  arg304_1 = None
        unsqueeze_612: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 4);  unsqueeze_611 = None
        view_766: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_612, _shape_param_767);  unsqueeze_612 = _shape_param_767 = None
        squeeze_203: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_766, 0);  view_766 = None
        mm_101: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_202, squeeze_203);  squeeze_202 = squeeze_203 = None
        unsqueeze_613: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_101, 0);  mm_101 = None
        view_767: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_613, _shape_param_768);  unsqueeze_613 = _shape_param_768 = None
        permute_486: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_767, [0, 1, 3, 4, 2]);  view_767 = None
        view_768: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_486, _shape_param_769);  permute_486 = _shape_param_769 = None
        unsqueeze_614: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_768, 4);  view_768 = None
        permute_487: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_614, [1, 2, 4, 0, 3]);  unsqueeze_614 = None
        permute_488: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_487, [0, 1, 4, 3, 2]);  permute_487 = None
        view_769: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_488, _shape_param_770);  permute_488 = _shape_param_770 = None
        bmm_60: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_764, view_769);  view_764 = view_769 = None
        view_770: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_60, _shape_param_771);  bmm_60 = _shape_param_771 = None
        permute_489: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_770, [0, 1, 2, 4, 3]);  view_770 = None
        view_771: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_489, _shape_param_772);  permute_489 = _shape_param_772 = None
        add_221: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_763, arg305_1);  view_763 = arg305_1 = None
        unsqueeze_615: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_221, 4);  add_221 = None
        permute_490: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_615, [1, 2, 0, 4, 3]);  unsqueeze_615 = None
        permute_491: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_490, [0, 1, 2, 4, 3]);  permute_490 = None
        view_772: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_491, _shape_param_773);  permute_491 = _shape_param_773 = None
        convert_element_type_202: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_616: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_202, 3);  convert_element_type_202 = None
        unsqueeze_617: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 4);  unsqueeze_616 = None
        view_773: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_617, _shape_param_774);  unsqueeze_617 = _shape_param_774 = None
        squeeze_204: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_773, 0);  view_773 = None
        unsqueeze_618: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg306_1, 3);  arg306_1 = None
        unsqueeze_619: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 4);  unsqueeze_618 = None
        view_774: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_619, _shape_param_775);  unsqueeze_619 = _shape_param_775 = None
        squeeze_205: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_774, 0);  view_774 = None
        mm_102: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_204, squeeze_205);  squeeze_204 = squeeze_205 = None
        unsqueeze_620: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_102, 0);  mm_102 = None
        view_775: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_620, _shape_param_776);  unsqueeze_620 = _shape_param_776 = None
        permute_492: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_775, [0, 1, 3, 4, 2]);  view_775 = None
        view_776: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_492, _shape_param_777);  permute_492 = _shape_param_777 = None
        unsqueeze_621: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_776, 4);  view_776 = None
        permute_493: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_621, [1, 2, 4, 0, 3]);  unsqueeze_621 = None
        permute_494: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_493, [0, 1, 4, 3, 2]);  permute_493 = None
        view_777: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_494, _shape_param_778);  permute_494 = _shape_param_778 = None
        bmm_61: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_772, view_777);  view_772 = view_777 = None
        view_778: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_61, _shape_param_779);  bmm_61 = _shape_param_779 = None
        permute_495: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_778, [0, 1, 2, 4, 3]);  view_778 = None
        view_779: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_495, _shape_param_780);  permute_495 = _shape_param_780 = None
        view_780: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_779, _shape_param_781);  view_779 = _shape_param_781 = None
        slice_21: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_780, 2, 1, 9223372036854775807);  view_780 = None
        view_781: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_21, _shape_param_782);  slice_21 = _shape_param_782 = None
        iota_22: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_20: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_781, [None, None, None, iota_22]);  view_781 = iota_22 = None
        add_222: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_771, index_20);  view_771 = index_20 = None
        add_223: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_222, 0);  add_222 = None
        mul_222: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_223, 0.125)
        convert_element_type_203: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_222, torch.float32);  mul_222 = None
        eq_20: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_203, convert_element_type_203)
        abs_21: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_203)
        ne_20: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_21, inf);  abs_21 = None
        mul_223: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_20, ne_20);  eq_20 = ne_20 = None
        logical_not_40: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_223);  mul_223 = None
        any_21: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_40, [3], True);  logical_not_40 = None
        logical_not_41: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        convert_element_type_204: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_223, torch.float32);  add_223 = None
        mul_224: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_204, 1);  convert_element_type_204 = None
        amax_40: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_224, [3], True)
        sub_80: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_224, amax_40);  mul_224 = amax_40 = None
        mul_225: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_80, 0.125);  sub_80 = None
        amax_41: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_203, [3], True)
        sub_81: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_203, amax_41);  convert_element_type_203 = amax_41 = None
        where_20: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_41, mul_225, sub_81);  logical_not_41 = mul_225 = sub_81 = None
        exp_20: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_20);  where_20 = None
        sum_21: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [3], True)
        div_21: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_205: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        unsqueeze_622: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_205, 4);  convert_element_type_205 = None
        view_782: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_622, _shape_param_783);  unsqueeze_622 = _shape_param_783 = None
        unsqueeze_623: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_201, 3)
        unsqueeze_624: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 4);  unsqueeze_623 = None
        view_783: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_624, _shape_param_784);  unsqueeze_624 = _shape_param_784 = None
        squeeze_206: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_783, 0);  view_783 = None
        unsqueeze_625: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg307_1, 3);  arg307_1 = None
        unsqueeze_626: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 4);  unsqueeze_625 = None
        view_784: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_626, _shape_param_785);  unsqueeze_626 = _shape_param_785 = None
        squeeze_207: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_784, 0);  view_784 = None
        mm_103: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_206, squeeze_207);  squeeze_206 = squeeze_207 = None
        unsqueeze_627: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_103, 0);  mm_103 = None
        view_785: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_627, _shape_param_786);  unsqueeze_627 = _shape_param_786 = None
        permute_496: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_785, [0, 1, 3, 4, 2]);  view_785 = None
        view_786: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_496, _shape_param_787);  permute_496 = _shape_param_787 = None
        unsqueeze_628: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_786, 4);  view_786 = None
        permute_497: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_628, [4, 1, 2, 3, 0]);  unsqueeze_628 = None
        permute_498: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_497, [1, 2, 4, 3, 0]);  permute_497 = None
        view_787: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_498, _shape_param_788);  permute_498 = _shape_param_788 = None
        bmm_62: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_782, view_787);  view_782 = view_787 = None
        view_788: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_62, _shape_param_789);  bmm_62 = _shape_param_789 = None
        permute_499: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_788, [2, 0, 1, 4, 3]);  view_788 = None
        view_789: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_499, _shape_param_790);  permute_499 = _shape_param_790 = None
        unsqueeze_629: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_789, 4);  view_789 = None
        permute_500: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_629, [0, 1, 4, 3, 2]);  unsqueeze_629 = None
        permute_501: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_500, [0, 1, 3, 4, 2]);  permute_500 = None
        clone_42: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
        view_790: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_42, _shape_param_791);  clone_42 = _shape_param_791 = None
        squeeze_208: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_790, 0);  view_790 = None
        unsqueeze_630: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg308_1, 3);  arg308_1 = None
        unsqueeze_631: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 4);  unsqueeze_630 = None
        permute_502: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_631, [3, 4, 0, 2, 1]);  unsqueeze_631 = None
        permute_503: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_502, [3, 4, 2, 0, 1]);  permute_502 = None
        clone_43: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_503, memory_format = torch.contiguous_format);  permute_503 = None
        view_791: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_43, _shape_param_792);  clone_43 = _shape_param_792 = None
        squeeze_209: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_791, 0);  view_791 = None
        mm_104: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_208, squeeze_209);  squeeze_208 = squeeze_209 = None
        unsqueeze_632: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_104, 0);  mm_104 = None
        view_792: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_632, _shape_param_793);  unsqueeze_632 = _shape_param_793 = None
        permute_504: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_792, [0, 1, 4, 2, 3]);  view_792 = None
        view_793: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_504, _shape_param_794);  permute_504 = _shape_param_794 = None
        add_224: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_793, convert_element_type_201);  view_793 = None
        convert_element_type_206: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_224, torch.float32);  add_224 = None
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_206, [2], correction = 0, keepdim = True)
        getitem_80: "f32[512, 16, 1]" = var_mean_40[0]
        getitem_81: "f32[512, 16, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_82: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_206, getitem_81);  convert_element_type_206 = getitem_81 = None
        add_225: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_226: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_40);  sub_82 = rsqrt_40 = None
        mul_227: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_226, arg309_1);  mul_226 = arg309_1 = None
        add_226: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_227, arg310_1);  mul_227 = arg310_1 = None
        convert_element_type_207: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16);  add_226 = None
        view_794: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_207, _shape_param_795);  _shape_param_795 = None
        permute_505: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_40: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg312_1, view_794, permute_505);  arg312_1 = view_794 = permute_505 = None
        view_795: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_40, _shape_param_796);  addmm_40 = _shape_param_796 = None
        convert_element_type_208: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_795, torch.float32);  view_795 = None
        mul_228: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_208, 0.5)
        mul_229: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_208, 0.7071067811865476);  convert_element_type_208 = None
        erf_20: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_229);  mul_229 = None
        add_227: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_230: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_228, add_227);  mul_228 = add_227 = None
        convert_element_type_209: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_230, torch.bfloat16);  mul_230 = None
        view_796: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_209, _shape_param_797);  convert_element_type_209 = _shape_param_797 = None
        permute_506: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_41: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg314_1, view_796, permute_506);  arg314_1 = view_796 = permute_506 = None
        view_797: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_41, _shape_param_798);  addmm_41 = _shape_param_798 = None
        add_228: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_797, convert_element_type_207);  view_797 = convert_element_type_207 = None
        convert_element_type_210: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_228, torch.float32);  add_228 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_210, [2], correction = 0, keepdim = True)
        getitem_82: "f32[512, 16, 1]" = var_mean_41[0]
        getitem_83: "f32[512, 16, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_83: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_210, getitem_83);  convert_element_type_210 = getitem_83 = None
        add_229: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        mul_231: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_41);  sub_83 = rsqrt_41 = None
        mul_232: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_231, arg315_1);  mul_231 = arg315_1 = None
        add_230: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_232, arg316_1);  mul_232 = arg316_1 = None
        convert_element_type_211: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_230, torch.bfloat16);  add_230 = None
        unsqueeze_633: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_211, 3)
        unsqueeze_634: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 4);  unsqueeze_633 = None
        view_798: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_634, _shape_param_799);  unsqueeze_634 = _shape_param_799 = None
        squeeze_210: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_798, 0);  view_798 = None
        unsqueeze_635: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg317_1, 3);  arg317_1 = None
        unsqueeze_636: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 4);  unsqueeze_635 = None
        view_799: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_636, _shape_param_800);  unsqueeze_636 = _shape_param_800 = None
        squeeze_211: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_799, 0);  view_799 = None
        mm_105: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_210, squeeze_211);  squeeze_210 = squeeze_211 = None
        unsqueeze_637: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_105, 0);  mm_105 = None
        view_800: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_637, _shape_param_801);  unsqueeze_637 = _shape_param_801 = None
        permute_507: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_800, [0, 1, 3, 4, 2]);  view_800 = None
        view_801: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_507, _shape_param_802);  permute_507 = _shape_param_802 = None
        add_231: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_801, arg318_1);  arg318_1 = None
        unsqueeze_638: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_231, 4);  add_231 = None
        permute_508: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_638, [1, 2, 0, 4, 3]);  unsqueeze_638 = None
        permute_509: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_508, [0, 1, 2, 4, 3]);  permute_508 = None
        view_802: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_509, _shape_param_803);  permute_509 = _shape_param_803 = None
        unsqueeze_639: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_211, 3)
        unsqueeze_640: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 4);  unsqueeze_639 = None
        view_803: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_640, _shape_param_804);  unsqueeze_640 = _shape_param_804 = None
        squeeze_212: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_803, 0);  view_803 = None
        unsqueeze_641: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, 3);  arg319_1 = None
        unsqueeze_642: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 4);  unsqueeze_641 = None
        view_804: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_642, _shape_param_805);  unsqueeze_642 = _shape_param_805 = None
        squeeze_213: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_804, 0);  view_804 = None
        mm_106: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_212, squeeze_213);  squeeze_212 = squeeze_213 = None
        unsqueeze_643: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_106, 0);  mm_106 = None
        view_805: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_643, _shape_param_806);  unsqueeze_643 = _shape_param_806 = None
        permute_510: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_805, [0, 1, 3, 4, 2]);  view_805 = None
        view_806: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_510, _shape_param_807);  permute_510 = _shape_param_807 = None
        unsqueeze_644: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_806, 4);  view_806 = None
        permute_511: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_644, [1, 2, 4, 0, 3]);  unsqueeze_644 = None
        permute_512: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_511, [0, 1, 4, 3, 2]);  permute_511 = None
        view_807: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_512, _shape_param_808);  permute_512 = _shape_param_808 = None
        bmm_63: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_802, view_807);  view_802 = view_807 = None
        view_808: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_63, _shape_param_809);  bmm_63 = _shape_param_809 = None
        permute_513: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_808, [0, 1, 2, 4, 3]);  view_808 = None
        view_809: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_513, _shape_param_810);  permute_513 = _shape_param_810 = None
        add_232: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_801, arg320_1);  view_801 = arg320_1 = None
        unsqueeze_645: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_232, 4);  add_232 = None
        permute_514: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_645, [1, 2, 0, 4, 3]);  unsqueeze_645 = None
        permute_515: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_514, [0, 1, 2, 4, 3]);  permute_514 = None
        view_810: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_515, _shape_param_811);  permute_515 = _shape_param_811 = None
        convert_element_type_212: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_646: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_212, 3);  convert_element_type_212 = None
        unsqueeze_647: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 4);  unsqueeze_646 = None
        view_811: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_647, _shape_param_812);  unsqueeze_647 = _shape_param_812 = None
        squeeze_214: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_811, 0);  view_811 = None
        unsqueeze_648: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg321_1, 3);  arg321_1 = None
        unsqueeze_649: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 4);  unsqueeze_648 = None
        view_812: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_649, _shape_param_813);  unsqueeze_649 = _shape_param_813 = None
        squeeze_215: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_812, 0);  view_812 = None
        mm_107: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_214, squeeze_215);  squeeze_214 = squeeze_215 = None
        unsqueeze_650: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_107, 0);  mm_107 = None
        view_813: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_650, _shape_param_814);  unsqueeze_650 = _shape_param_814 = None
        permute_516: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_813, [0, 1, 3, 4, 2]);  view_813 = None
        view_814: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_516, _shape_param_815);  permute_516 = _shape_param_815 = None
        unsqueeze_651: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_814, 4);  view_814 = None
        permute_517: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_651, [1, 2, 4, 0, 3]);  unsqueeze_651 = None
        permute_518: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_517, [0, 1, 4, 3, 2]);  permute_517 = None
        view_815: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_518, _shape_param_816);  permute_518 = _shape_param_816 = None
        bmm_64: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_810, view_815);  view_810 = view_815 = None
        view_816: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_64, _shape_param_817);  bmm_64 = _shape_param_817 = None
        permute_519: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_816, [0, 1, 2, 4, 3]);  view_816 = None
        view_817: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_519, _shape_param_818);  permute_519 = _shape_param_818 = None
        view_818: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_817, _shape_param_819);  view_817 = _shape_param_819 = None
        slice_22: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_818, 2, 1, 9223372036854775807);  view_818 = None
        view_819: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_22, _shape_param_820);  slice_22 = _shape_param_820 = None
        iota_23: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_21: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_819, [None, None, None, iota_23]);  view_819 = iota_23 = None
        add_233: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_809, index_21);  view_809 = index_21 = None
        add_234: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_233, 0);  add_233 = None
        mul_233: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_234, 0.125)
        convert_element_type_213: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_233, torch.float32);  mul_233 = None
        eq_21: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_213, convert_element_type_213)
        abs_22: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_213)
        ne_21: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_22, inf);  abs_22 = None
        mul_234: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_21, ne_21);  eq_21 = ne_21 = None
        logical_not_42: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_234);  mul_234 = None
        any_22: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_42, [3], True);  logical_not_42 = None
        logical_not_43: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        convert_element_type_214: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_234, torch.float32);  add_234 = None
        mul_235: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_214, 1);  convert_element_type_214 = None
        amax_42: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_235, [3], True)
        sub_84: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_235, amax_42);  mul_235 = amax_42 = None
        mul_236: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_84, 0.125);  sub_84 = None
        amax_43: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_213, [3], True)
        sub_85: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_213, amax_43);  convert_element_type_213 = amax_43 = None
        where_21: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_43, mul_236, sub_85);  logical_not_43 = mul_236 = sub_85 = None
        exp_21: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_21);  where_21 = None
        sum_22: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [3], True)
        div_22: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_215: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        unsqueeze_652: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_215, 4);  convert_element_type_215 = None
        view_820: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_652, _shape_param_821);  unsqueeze_652 = _shape_param_821 = None
        unsqueeze_653: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_211, 3)
        unsqueeze_654: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 4);  unsqueeze_653 = None
        view_821: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_654, _shape_param_822);  unsqueeze_654 = _shape_param_822 = None
        squeeze_216: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_821, 0);  view_821 = None
        unsqueeze_655: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg322_1, 3);  arg322_1 = None
        unsqueeze_656: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 4);  unsqueeze_655 = None
        view_822: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_656, _shape_param_823);  unsqueeze_656 = _shape_param_823 = None
        squeeze_217: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_822, 0);  view_822 = None
        mm_108: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_216, squeeze_217);  squeeze_216 = squeeze_217 = None
        unsqueeze_657: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_108, 0);  mm_108 = None
        view_823: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_657, _shape_param_824);  unsqueeze_657 = _shape_param_824 = None
        permute_520: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_823, [0, 1, 3, 4, 2]);  view_823 = None
        view_824: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_520, _shape_param_825);  permute_520 = _shape_param_825 = None
        unsqueeze_658: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_824, 4);  view_824 = None
        permute_521: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_658, [4, 1, 2, 3, 0]);  unsqueeze_658 = None
        permute_522: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_521, [1, 2, 4, 3, 0]);  permute_521 = None
        view_825: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_522, _shape_param_826);  permute_522 = _shape_param_826 = None
        bmm_65: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_820, view_825);  view_820 = view_825 = None
        view_826: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_65, _shape_param_827);  bmm_65 = _shape_param_827 = None
        permute_523: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_826, [2, 0, 1, 4, 3]);  view_826 = None
        view_827: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_523, _shape_param_828);  permute_523 = _shape_param_828 = None
        unsqueeze_659: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_827, 4);  view_827 = None
        permute_524: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_659, [0, 1, 4, 3, 2]);  unsqueeze_659 = None
        permute_525: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_524, [0, 1, 3, 4, 2]);  permute_524 = None
        clone_44: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
        view_828: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_44, _shape_param_829);  clone_44 = _shape_param_829 = None
        squeeze_218: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_828, 0);  view_828 = None
        unsqueeze_660: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg323_1, 3);  arg323_1 = None
        unsqueeze_661: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 4);  unsqueeze_660 = None
        permute_526: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_661, [3, 4, 0, 2, 1]);  unsqueeze_661 = None
        permute_527: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_526, [3, 4, 2, 0, 1]);  permute_526 = None
        clone_45: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_527, memory_format = torch.contiguous_format);  permute_527 = None
        view_829: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_45, _shape_param_830);  clone_45 = _shape_param_830 = None
        squeeze_219: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_829, 0);  view_829 = None
        mm_109: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_218, squeeze_219);  squeeze_218 = squeeze_219 = None
        unsqueeze_662: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_109, 0);  mm_109 = None
        view_830: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_662, _shape_param_831);  unsqueeze_662 = _shape_param_831 = None
        permute_528: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_830, [0, 1, 4, 2, 3]);  view_830 = None
        view_831: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_528, _shape_param_832);  permute_528 = _shape_param_832 = None
        add_235: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_831, convert_element_type_211);  view_831 = None
        convert_element_type_216: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_235, torch.float32);  add_235 = None
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_216, [2], correction = 0, keepdim = True)
        getitem_84: "f32[512, 16, 1]" = var_mean_42[0]
        getitem_85: "f32[512, 16, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_86: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_216, getitem_85);  convert_element_type_216 = getitem_85 = None
        add_236: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_237: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_42);  sub_86 = rsqrt_42 = None
        mul_238: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_237, arg324_1);  mul_237 = arg324_1 = None
        add_237: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_238, arg325_1);  mul_238 = arg325_1 = None
        convert_element_type_217: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16);  add_237 = None
        view_832: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_217, _shape_param_833);  _shape_param_833 = None
        permute_529: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg326_1, [1, 0]);  arg326_1 = None
        addmm_42: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg327_1, view_832, permute_529);  arg327_1 = view_832 = permute_529 = None
        view_833: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_42, _shape_param_834);  addmm_42 = _shape_param_834 = None
        convert_element_type_218: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_833, torch.float32);  view_833 = None
        mul_239: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_218, 0.5)
        mul_240: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_218, 0.7071067811865476);  convert_element_type_218 = None
        erf_21: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
        add_238: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_241: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_239, add_238);  mul_239 = add_238 = None
        convert_element_type_219: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_241, torch.bfloat16);  mul_241 = None
        view_834: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_219, _shape_param_835);  convert_element_type_219 = _shape_param_835 = None
        permute_530: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg328_1, [1, 0]);  arg328_1 = None
        addmm_43: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg329_1, view_834, permute_530);  arg329_1 = view_834 = permute_530 = None
        view_835: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_43, _shape_param_836);  addmm_43 = _shape_param_836 = None
        add_239: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_835, convert_element_type_217);  view_835 = convert_element_type_217 = None
        convert_element_type_220: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_239, torch.float32);  add_239 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_220, [2], correction = 0, keepdim = True)
        getitem_86: "f32[512, 16, 1]" = var_mean_43[0]
        getitem_87: "f32[512, 16, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_87: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_220, getitem_87);  convert_element_type_220 = getitem_87 = None
        add_240: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        mul_242: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_43);  sub_87 = rsqrt_43 = None
        mul_243: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_242, arg330_1);  mul_242 = arg330_1 = None
        add_241: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_243, arg331_1);  mul_243 = arg331_1 = None
        convert_element_type_221: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_241, torch.bfloat16);  add_241 = None
        unsqueeze_663: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_221, 3)
        unsqueeze_664: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 4);  unsqueeze_663 = None
        view_836: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_664, _shape_param_837);  unsqueeze_664 = _shape_param_837 = None
        squeeze_220: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_836, 0);  view_836 = None
        unsqueeze_665: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg332_1, 3);  arg332_1 = None
        unsqueeze_666: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 4);  unsqueeze_665 = None
        view_837: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_666, _shape_param_838);  unsqueeze_666 = _shape_param_838 = None
        squeeze_221: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_837, 0);  view_837 = None
        mm_110: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_220, squeeze_221);  squeeze_220 = squeeze_221 = None
        unsqueeze_667: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_110, 0);  mm_110 = None
        view_838: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_667, _shape_param_839);  unsqueeze_667 = _shape_param_839 = None
        permute_531: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_838, [0, 1, 3, 4, 2]);  view_838 = None
        view_839: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_531, _shape_param_840);  permute_531 = _shape_param_840 = None
        add_242: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_839, arg333_1);  arg333_1 = None
        unsqueeze_668: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_242, 4);  add_242 = None
        permute_532: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_668, [1, 2, 0, 4, 3]);  unsqueeze_668 = None
        permute_533: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_532, [0, 1, 2, 4, 3]);  permute_532 = None
        view_840: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_533, _shape_param_841);  permute_533 = _shape_param_841 = None
        unsqueeze_669: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_221, 3)
        unsqueeze_670: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 4);  unsqueeze_669 = None
        view_841: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_670, _shape_param_842);  unsqueeze_670 = _shape_param_842 = None
        squeeze_222: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_841, 0);  view_841 = None
        unsqueeze_671: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, 3);  arg334_1 = None
        unsqueeze_672: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 4);  unsqueeze_671 = None
        view_842: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_672, _shape_param_843);  unsqueeze_672 = _shape_param_843 = None
        squeeze_223: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_842, 0);  view_842 = None
        mm_111: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_222, squeeze_223);  squeeze_222 = squeeze_223 = None
        unsqueeze_673: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_111, 0);  mm_111 = None
        view_843: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_673, _shape_param_844);  unsqueeze_673 = _shape_param_844 = None
        permute_534: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_843, [0, 1, 3, 4, 2]);  view_843 = None
        view_844: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_534, _shape_param_845);  permute_534 = _shape_param_845 = None
        unsqueeze_674: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_844, 4);  view_844 = None
        permute_535: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_674, [1, 2, 4, 0, 3]);  unsqueeze_674 = None
        permute_536: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_535, [0, 1, 4, 3, 2]);  permute_535 = None
        view_845: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_536, _shape_param_846);  permute_536 = _shape_param_846 = None
        bmm_66: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_840, view_845);  view_840 = view_845 = None
        view_846: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_66, _shape_param_847);  bmm_66 = _shape_param_847 = None
        permute_537: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_846, [0, 1, 2, 4, 3]);  view_846 = None
        view_847: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_537, _shape_param_848);  permute_537 = _shape_param_848 = None
        add_243: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_839, arg335_1);  view_839 = arg335_1 = None
        unsqueeze_675: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_243, 4);  add_243 = None
        permute_538: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_675, [1, 2, 0, 4, 3]);  unsqueeze_675 = None
        permute_539: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_538, [0, 1, 2, 4, 3]);  permute_538 = None
        view_848: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_539, _shape_param_849);  permute_539 = _shape_param_849 = None
        convert_element_type_222: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16)
        unsqueeze_676: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_222, 3);  convert_element_type_222 = None
        unsqueeze_677: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 4);  unsqueeze_676 = None
        view_849: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_677, _shape_param_850);  unsqueeze_677 = _shape_param_850 = None
        squeeze_224: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_849, 0);  view_849 = None
        unsqueeze_678: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg336_1, 3);  arg336_1 = None
        unsqueeze_679: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 4);  unsqueeze_678 = None
        view_850: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_679, _shape_param_851);  unsqueeze_679 = _shape_param_851 = None
        squeeze_225: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_850, 0);  view_850 = None
        mm_112: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_224, squeeze_225);  squeeze_224 = squeeze_225 = None
        unsqueeze_680: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_112, 0);  mm_112 = None
        view_851: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_680, _shape_param_852);  unsqueeze_680 = _shape_param_852 = None
        permute_540: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_851, [0, 1, 3, 4, 2]);  view_851 = None
        view_852: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_540, _shape_param_853);  permute_540 = _shape_param_853 = None
        unsqueeze_681: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_852, 4);  view_852 = None
        permute_541: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_681, [1, 2, 4, 0, 3]);  unsqueeze_681 = None
        permute_542: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_541, [0, 1, 4, 3, 2]);  permute_541 = None
        view_853: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_542, _shape_param_854);  permute_542 = _shape_param_854 = None
        bmm_67: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_848, view_853);  view_848 = view_853 = None
        view_854: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_67, _shape_param_855);  bmm_67 = _shape_param_855 = None
        permute_543: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_854, [0, 1, 2, 4, 3]);  view_854 = None
        view_855: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_543, _shape_param_856);  permute_543 = _shape_param_856 = None
        view_856: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_855, _shape_param_857);  view_855 = _shape_param_857 = None
        slice_23: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_856, 2, 1, 9223372036854775807);  view_856 = None
        view_857: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_23, _shape_param_858);  slice_23 = _shape_param_858 = None
        iota_24: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_22: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_857, [None, None, None, iota_24]);  view_857 = iota_24 = None
        add_244: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_847, index_22);  view_847 = index_22 = None
        add_245: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_244, 0);  add_244 = None
        mul_244: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_245, 0.125)
        convert_element_type_223: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_244, torch.float32);  mul_244 = None
        eq_22: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_223, convert_element_type_223)
        abs_23: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_223)
        ne_22: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_23, inf);  abs_23 = None
        mul_245: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_22, ne_22);  eq_22 = ne_22 = None
        logical_not_44: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_245);  mul_245 = None
        any_23: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_44, [3], True);  logical_not_44 = None
        logical_not_45: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        convert_element_type_224: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_245, torch.float32);  add_245 = None
        mul_246: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_224, 1);  convert_element_type_224 = None
        amax_44: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_246, [3], True)
        sub_88: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_246, amax_44);  mul_246 = amax_44 = None
        mul_247: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_88, 0.125);  sub_88 = None
        amax_45: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_223, [3], True)
        sub_89: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_223, amax_45);  convert_element_type_223 = amax_45 = None
        where_22: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_45, mul_247, sub_89);  logical_not_45 = mul_247 = sub_89 = None
        exp_22: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_22);  where_22 = None
        sum_23: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [3], True)
        div_23: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_225: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        unsqueeze_682: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_225, 4);  convert_element_type_225 = None
        view_858: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_682, _shape_param_859);  unsqueeze_682 = _shape_param_859 = None
        unsqueeze_683: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_221, 3)
        unsqueeze_684: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 4);  unsqueeze_683 = None
        view_859: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_684, _shape_param_860);  unsqueeze_684 = _shape_param_860 = None
        squeeze_226: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_859, 0);  view_859 = None
        unsqueeze_685: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg337_1, 3);  arg337_1 = None
        unsqueeze_686: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 4);  unsqueeze_685 = None
        view_860: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_686, _shape_param_861);  unsqueeze_686 = _shape_param_861 = None
        squeeze_227: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_860, 0);  view_860 = None
        mm_113: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_226, squeeze_227);  squeeze_226 = squeeze_227 = None
        unsqueeze_687: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_113, 0);  mm_113 = None
        view_861: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_687, _shape_param_862);  unsqueeze_687 = _shape_param_862 = None
        permute_544: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_861, [0, 1, 3, 4, 2]);  view_861 = None
        view_862: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_544, _shape_param_863);  permute_544 = _shape_param_863 = None
        unsqueeze_688: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_862, 4);  view_862 = None
        permute_545: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_688, [4, 1, 2, 3, 0]);  unsqueeze_688 = None
        permute_546: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_545, [1, 2, 4, 3, 0]);  permute_545 = None
        view_863: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_546, _shape_param_864);  permute_546 = _shape_param_864 = None
        bmm_68: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_858, view_863);  view_858 = view_863 = None
        view_864: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_68, _shape_param_865);  bmm_68 = _shape_param_865 = None
        permute_547: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_864, [2, 0, 1, 4, 3]);  view_864 = None
        view_865: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_547, _shape_param_866);  permute_547 = _shape_param_866 = None
        unsqueeze_689: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_865, 4);  view_865 = None
        permute_548: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_689, [0, 1, 4, 3, 2]);  unsqueeze_689 = None
        permute_549: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_548, [0, 1, 3, 4, 2]);  permute_548 = None
        clone_46: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_549, memory_format = torch.contiguous_format);  permute_549 = None
        view_866: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_46, _shape_param_867);  clone_46 = _shape_param_867 = None
        squeeze_228: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_866, 0);  view_866 = None
        unsqueeze_690: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg338_1, 3);  arg338_1 = None
        unsqueeze_691: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 4);  unsqueeze_690 = None
        permute_550: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_691, [3, 4, 0, 2, 1]);  unsqueeze_691 = None
        permute_551: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_550, [3, 4, 2, 0, 1]);  permute_550 = None
        clone_47: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        view_867: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_47, _shape_param_868);  clone_47 = _shape_param_868 = None
        squeeze_229: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_867, 0);  view_867 = None
        mm_114: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_228, squeeze_229);  squeeze_228 = squeeze_229 = None
        unsqueeze_692: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_114, 0);  mm_114 = None
        view_868: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_692, _shape_param_869);  unsqueeze_692 = _shape_param_869 = None
        permute_552: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_868, [0, 1, 4, 2, 3]);  view_868 = None
        view_869: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_552, _shape_param_870);  permute_552 = _shape_param_870 = None
        add_246: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_869, convert_element_type_221);  view_869 = None
        convert_element_type_226: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_246, torch.float32);  add_246 = None
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_226, [2], correction = 0, keepdim = True)
        getitem_88: "f32[512, 16, 1]" = var_mean_44[0]
        getitem_89: "f32[512, 16, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_90: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_226, getitem_89);  convert_element_type_226 = getitem_89 = None
        add_247: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_248: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_44);  sub_90 = rsqrt_44 = None
        mul_249: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_248, arg339_1);  mul_248 = arg339_1 = None
        add_248: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_249, arg340_1);  mul_249 = arg340_1 = None
        convert_element_type_227: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_248, torch.bfloat16);  add_248 = None
        view_870: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_227, _shape_param_871);  _shape_param_871 = None
        permute_553: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_44: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg342_1, view_870, permute_553);  arg342_1 = view_870 = permute_553 = None
        view_871: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_44, _shape_param_872);  addmm_44 = _shape_param_872 = None
        convert_element_type_228: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_871, torch.float32);  view_871 = None
        mul_250: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.5)
        mul_251: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.7071067811865476);  convert_element_type_228 = None
        erf_22: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_251);  mul_251 = None
        add_249: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_252: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_250, add_249);  mul_250 = add_249 = None
        convert_element_type_229: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_252, torch.bfloat16);  mul_252 = None
        view_872: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_229, _shape_param_873);  convert_element_type_229 = _shape_param_873 = None
        permute_554: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_45: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg344_1, view_872, permute_554);  arg344_1 = view_872 = permute_554 = None
        view_873: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_45, _shape_param_874);  addmm_45 = _shape_param_874 = None
        add_250: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_873, convert_element_type_227);  view_873 = convert_element_type_227 = None
        convert_element_type_230: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_250, torch.float32);  add_250 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_230, [2], correction = 0, keepdim = True)
        getitem_90: "f32[512, 16, 1]" = var_mean_45[0]
        getitem_91: "f32[512, 16, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_91: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_230, getitem_91);  convert_element_type_230 = getitem_91 = None
        add_251: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        mul_253: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_45);  sub_91 = rsqrt_45 = None
        mul_254: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_253, arg345_1);  mul_253 = arg345_1 = None
        add_252: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_254, arg346_1);  mul_254 = arg346_1 = None
        convert_element_type_231: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16);  add_252 = None
        unsqueeze_693: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_231, 3)
        unsqueeze_694: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 4);  unsqueeze_693 = None
        view_874: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_694, _shape_param_875);  unsqueeze_694 = _shape_param_875 = None
        squeeze_230: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_874, 0);  view_874 = None
        unsqueeze_695: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg347_1, 3);  arg347_1 = None
        unsqueeze_696: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 4);  unsqueeze_695 = None
        view_875: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_696, _shape_param_876);  unsqueeze_696 = _shape_param_876 = None
        squeeze_231: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_875, 0);  view_875 = None
        mm_115: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_230, squeeze_231);  squeeze_230 = squeeze_231 = None
        unsqueeze_697: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_115, 0);  mm_115 = None
        view_876: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_697, _shape_param_877);  unsqueeze_697 = _shape_param_877 = None
        permute_555: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_876, [0, 1, 3, 4, 2]);  view_876 = None
        view_877: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_555, _shape_param_878);  permute_555 = _shape_param_878 = None
        add_253: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_877, arg348_1);  arg348_1 = None
        unsqueeze_698: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_253, 4);  add_253 = None
        permute_556: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_698, [1, 2, 0, 4, 3]);  unsqueeze_698 = None
        permute_557: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_556, [0, 1, 2, 4, 3]);  permute_556 = None
        view_878: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_557, _shape_param_879);  permute_557 = _shape_param_879 = None
        unsqueeze_699: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_231, 3)
        unsqueeze_700: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 4);  unsqueeze_699 = None
        view_879: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_700, _shape_param_880);  unsqueeze_700 = _shape_param_880 = None
        squeeze_232: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_879, 0);  view_879 = None
        unsqueeze_701: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, 3);  arg349_1 = None
        unsqueeze_702: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 4);  unsqueeze_701 = None
        view_880: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_702, _shape_param_881);  unsqueeze_702 = _shape_param_881 = None
        squeeze_233: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_880, 0);  view_880 = None
        mm_116: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_232, squeeze_233);  squeeze_232 = squeeze_233 = None
        unsqueeze_703: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_116, 0);  mm_116 = None
        view_881: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_703, _shape_param_882);  unsqueeze_703 = _shape_param_882 = None
        permute_558: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_881, [0, 1, 3, 4, 2]);  view_881 = None
        view_882: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_558, _shape_param_883);  permute_558 = _shape_param_883 = None
        unsqueeze_704: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_882, 4);  view_882 = None
        permute_559: "bf16[16, 16, 1, 512, 64]" = torch.ops.aten.permute.default(unsqueeze_704, [1, 2, 4, 0, 3]);  unsqueeze_704 = None
        permute_560: "bf16[16, 16, 64, 512, 1]" = torch.ops.aten.permute.default(permute_559, [0, 1, 4, 3, 2]);  permute_559 = None
        view_883: "bf16[256, 64, 512]" = torch.ops.aten.view.default(permute_560, _shape_param_884);  permute_560 = _shape_param_884 = None
        bmm_69: "bf16[256, 512, 512]" = torch.ops.aten.bmm.default(view_878, view_883);  view_878 = view_883 = None
        view_884: "bf16[16, 16, 512, 1, 512]" = torch.ops.aten.view.default(bmm_69, _shape_param_885);  bmm_69 = _shape_param_885 = None
        permute_561: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.permute.default(view_884, [0, 1, 2, 4, 3]);  view_884 = None
        view_885: "bf16[16, 16, 512, 512]" = torch.ops.aten.view.default(permute_561, _shape_param_886);  permute_561 = _shape_param_886 = None
        add_254: "bf16[512, 16, 16, 64]" = torch.ops.aten.add.Tensor(view_877, arg350_1);  view_877 = arg350_1 = None
        unsqueeze_705: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(add_254, 4);  add_254 = None
        permute_562: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.permute.default(unsqueeze_705, [1, 2, 0, 4, 3]);  unsqueeze_705 = None
        permute_563: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_562, [0, 1, 2, 4, 3]);  permute_562 = None
        view_886: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_563, _shape_param_887);  permute_563 = _shape_param_887 = None
        convert_element_type_232: "bf16[1024, 16, 1024]" = torch.ops.prims.convert_element_type.default(clone_1, torch.bfloat16);  clone_1 = None
        unsqueeze_706: "bf16[1024, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_232, 3);  convert_element_type_232 = None
        unsqueeze_707: "bf16[1024, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 4);  unsqueeze_706 = None
        view_887: "bf16[1, 16384, 1024]" = torch.ops.aten.view.default(unsqueeze_707, _shape_param_888);  unsqueeze_707 = _shape_param_888 = None
        squeeze_234: "bf16[16384, 1024]" = torch.ops.aten.squeeze.dim(view_887, 0);  view_887 = None
        unsqueeze_708: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg351_1, 3);  arg351_1 = None
        unsqueeze_709: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 4);  unsqueeze_708 = None
        view_888: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_709, _shape_param_889);  unsqueeze_709 = _shape_param_889 = None
        squeeze_235: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_888, 0);  view_888 = None
        mm_117: "bf16[16384, 1024]" = torch.ops.aten.mm.default(squeeze_234, squeeze_235);  squeeze_234 = squeeze_235 = None
        unsqueeze_710: "bf16[1, 16384, 1024]" = torch.ops.aten.unsqueeze.default(mm_117, 0);  mm_117 = None
        view_889: "bf16[1024, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_710, _shape_param_890);  unsqueeze_710 = _shape_param_890 = None
        permute_564: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_889, [0, 1, 3, 4, 2]);  view_889 = None
        view_890: "bf16[1024, 16, 16, 64]" = torch.ops.aten.view.default(permute_564, _shape_param_891);  permute_564 = _shape_param_891 = None
        unsqueeze_711: "bf16[1024, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_890, 4);  view_890 = None
        permute_565: "bf16[16, 16, 1, 1024, 64]" = torch.ops.aten.permute.default(unsqueeze_711, [1, 2, 4, 0, 3]);  unsqueeze_711 = None
        permute_566: "bf16[16, 16, 64, 1024, 1]" = torch.ops.aten.permute.default(permute_565, [0, 1, 4, 3, 2]);  permute_565 = None
        view_891: "bf16[256, 64, 1024]" = torch.ops.aten.view.default(permute_566, _shape_param_892);  permute_566 = _shape_param_892 = None
        bmm_70: "bf16[256, 512, 1024]" = torch.ops.aten.bmm.default(view_886, view_891);  view_886 = view_891 = None
        view_892: "bf16[16, 16, 512, 1, 1024]" = torch.ops.aten.view.default(bmm_70, _shape_param_893);  bmm_70 = _shape_param_893 = None
        permute_567: "bf16[16, 16, 512, 1024, 1]" = torch.ops.aten.permute.default(view_892, [0, 1, 2, 4, 3]);  view_892 = None
        view_893: "bf16[16, 16, 512, 1024]" = torch.ops.aten.view.default(permute_567, _shape_param_894);  permute_567 = _shape_param_894 = None
        view_894: "bf16[16, 16, 1024, 512]" = torch.ops.aten.view.default(view_893, _shape_param_895);  view_893 = _shape_param_895 = None
        slice_24: "bf16[16, 16, 1023, 512]" = torch.ops.aten.slice.Tensor(view_894, 2, 1, 9223372036854775807);  view_894 = None
        view_895: "bf16[16, 16, 512, 1023]" = torch.ops.aten.view.default(slice_24, _shape_param_896);  slice_24 = _shape_param_896 = None
        iota_25: "i64[512]" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_23: "bf16[16, 16, 512, 512]" = torch.ops.aten.index.Tensor(view_895, [None, None, None, iota_25]);  view_895 = iota_25 = None
        add_255: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(view_885, index_23);  view_885 = index_23 = None
        add_256: "bf16[16, 16, 512, 512]" = torch.ops.aten.add.Tensor(add_255, 0);  add_255 = None
        mul_255: "bf16[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(add_256, 0.125)
        convert_element_type_233: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_255, torch.float32);  mul_255 = None
        eq_23: "b8[16, 16, 512, 512]" = torch.ops.aten.eq.Tensor(convert_element_type_233, convert_element_type_233)
        abs_24: "f32[16, 16, 512, 512]" = torch.ops.aten.abs.default(convert_element_type_233)
        ne_23: "b8[16, 16, 512, 512]" = torch.ops.aten.ne.Scalar(abs_24, inf);  abs_24 = None
        mul_256: "b8[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(eq_23, ne_23);  eq_23 = ne_23 = None
        logical_not_46: "b8[16, 16, 512, 512]" = torch.ops.aten.logical_not.default(mul_256);  mul_256 = None
        any_24: "b8[16, 16, 512, 1]" = torch.ops.aten.any.dims(logical_not_46, [3], True);  logical_not_46 = None
        logical_not_47: "b8[16, 16, 512, 1]" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        convert_element_type_234: "f32[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(add_256, torch.float32);  add_256 = None
        mul_257: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_234, 1);  convert_element_type_234 = None
        amax_46: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(mul_257, [3], True)
        sub_92: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(mul_257, amax_46);  mul_257 = amax_46 = None
        mul_258: "f32[16, 16, 512, 512]" = torch.ops.aten.mul.Tensor(sub_92, 0.125);  sub_92 = None
        amax_47: "f32[16, 16, 512, 1]" = torch.ops.aten.amax.default(convert_element_type_233, [3], True)
        sub_93: "f32[16, 16, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_233, amax_47);  convert_element_type_233 = amax_47 = None
        where_23: "f32[16, 16, 512, 512]" = torch.ops.aten.where.self(logical_not_47, mul_258, sub_93);  logical_not_47 = mul_258 = sub_93 = None
        exp_23: "f32[16, 16, 512, 512]" = torch.ops.aten.exp.default(where_23);  where_23 = None
        sum_24: "f32[16, 16, 512, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [3], True)
        div_24: "f32[16, 16, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_235: "bf16[16, 16, 512, 512]" = torch.ops.prims.convert_element_type.default(div_24, torch.bfloat16);  div_24 = None
        unsqueeze_712: "bf16[16, 16, 512, 512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_235, 4);  convert_element_type_235 = None
        view_896: "bf16[256, 512, 512]" = torch.ops.aten.view.default(unsqueeze_712, _shape_param_897);  unsqueeze_712 = _shape_param_897 = None
        unsqueeze_713: "bf16[512, 16, 1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_231, 3)
        unsqueeze_714: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 4);  unsqueeze_713 = None
        view_897: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(unsqueeze_714, _shape_param_898);  unsqueeze_714 = _shape_param_898 = None
        squeeze_236: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_897, 0);  view_897 = None
        unsqueeze_715: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg352_1, 3);  arg352_1 = None
        unsqueeze_716: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 4);  unsqueeze_715 = None
        view_898: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(unsqueeze_716, _shape_param_899);  unsqueeze_716 = _shape_param_899 = None
        squeeze_237: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_898, 0);  view_898 = None
        mm_118: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_236, squeeze_237);  squeeze_236 = squeeze_237 = None
        unsqueeze_717: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_118, 0);  mm_118 = None
        view_899: "bf16[512, 16, 1, 16, 64]" = torch.ops.aten.view.default(unsqueeze_717, _shape_param_900);  unsqueeze_717 = _shape_param_900 = None
        permute_568: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_899, [0, 1, 3, 4, 2]);  view_899 = None
        view_900: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_568, _shape_param_901);  permute_568 = _shape_param_901 = None
        unsqueeze_718: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_900, 4);  view_900 = None
        permute_569: "bf16[1, 16, 16, 64, 512]" = torch.ops.aten.permute.default(unsqueeze_718, [4, 1, 2, 3, 0]);  unsqueeze_718 = None
        permute_570: "bf16[16, 16, 512, 64, 1]" = torch.ops.aten.permute.default(permute_569, [1, 2, 4, 3, 0]);  permute_569 = None
        view_901: "bf16[256, 512, 64]" = torch.ops.aten.view.default(permute_570, _shape_param_902);  permute_570 = _shape_param_902 = None
        bmm_71: "bf16[256, 512, 64]" = torch.ops.aten.bmm.default(view_896, view_901);  view_896 = view_901 = None
        view_902: "bf16[16, 16, 512, 1, 64]" = torch.ops.aten.view.default(bmm_71, _shape_param_903);  bmm_71 = _shape_param_903 = None
        permute_571: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.permute.default(view_902, [2, 0, 1, 4, 3]);  view_902 = None
        view_903: "bf16[512, 16, 16, 64]" = torch.ops.aten.view.default(permute_571, _shape_param_904);  permute_571 = _shape_param_904 = None
        unsqueeze_719: "bf16[512, 16, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(view_903, 4);  view_903 = None
        permute_572: "bf16[512, 16, 1, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_719, [0, 1, 4, 3, 2]);  unsqueeze_719 = None
        permute_573: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.permute.default(permute_572, [0, 1, 3, 4, 2]);  permute_572 = None
        clone_48: "bf16[512, 16, 64, 16, 1]" = torch.ops.aten.clone.default(permute_573, memory_format = torch.contiguous_format);  permute_573 = None
        view_904: "bf16[1, 8192, 1024]" = torch.ops.aten.view.default(clone_48, _shape_param_905);  clone_48 = _shape_param_905 = None
        squeeze_238: "bf16[8192, 1024]" = torch.ops.aten.squeeze.dim(view_904, 0);  view_904 = None
        unsqueeze_720: "bf16[1024, 16, 64, 1]" = torch.ops.aten.unsqueeze.default(arg353_1, 3);  arg353_1 = None
        unsqueeze_721: "bf16[1024, 16, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 4);  unsqueeze_720 = None
        permute_574: "bf16[1, 1, 1024, 64, 16]" = torch.ops.aten.permute.default(unsqueeze_721, [3, 4, 0, 2, 1]);  unsqueeze_721 = None
        permute_575: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(permute_574, [3, 4, 2, 0, 1]);  permute_574 = None
        clone_49: "bf16[64, 16, 1024, 1, 1]" = torch.ops.aten.clone.default(permute_575, memory_format = torch.contiguous_format);  permute_575 = None
        view_905: "bf16[1, 1024, 1024]" = torch.ops.aten.view.default(clone_49, _shape_param_906);  clone_49 = _shape_param_906 = None
        squeeze_239: "bf16[1024, 1024]" = torch.ops.aten.squeeze.dim(view_905, 0);  view_905 = None
        mm_119: "bf16[8192, 1024]" = torch.ops.aten.mm.default(squeeze_238, squeeze_239);  squeeze_238 = squeeze_239 = None
        unsqueeze_722: "bf16[1, 8192, 1024]" = torch.ops.aten.unsqueeze.default(mm_119, 0);  mm_119 = None
        view_906: "bf16[512, 16, 1, 1, 1024]" = torch.ops.aten.view.default(unsqueeze_722, _shape_param_907);  unsqueeze_722 = _shape_param_907 = None
        permute_576: "bf16[512, 16, 1024, 1, 1]" = torch.ops.aten.permute.default(view_906, [0, 1, 4, 2, 3]);  view_906 = None
        view_907: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(permute_576, _shape_param_908);  permute_576 = _shape_param_908 = None
        add_257: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_907, convert_element_type_231);  view_907 = None
        convert_element_type_236: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_257, torch.float32);  add_257 = None
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_236, [2], correction = 0, keepdim = True)
        getitem_92: "f32[512, 16, 1]" = var_mean_46[0]
        getitem_93: "f32[512, 16, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_94: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_236, getitem_93);  convert_element_type_236 = getitem_93 = None
        add_258: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        mul_259: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_46);  sub_94 = rsqrt_46 = None
        mul_260: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_259, arg354_1);  mul_259 = arg354_1 = None
        add_259: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_260, arg355_1);  mul_260 = arg355_1 = None
        convert_element_type_237: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None
        view_908: "bf16[8192, 1024]" = torch.ops.aten.view.default(convert_element_type_237, _shape_param_909);  _shape_param_909 = None
        permute_577: "bf16[1024, 4096]" = torch.ops.aten.permute.default(arg356_1, [1, 0]);  arg356_1 = None
        addmm_46: "bf16[8192, 4096]" = torch.ops.aten.addmm.default(arg357_1, view_908, permute_577);  arg357_1 = view_908 = permute_577 = None
        view_909: "bf16[512, 16, 4096]" = torch.ops.aten.view.default(addmm_46, _shape_param_910);  addmm_46 = _shape_param_910 = None
        convert_element_type_238: "f32[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(view_909, torch.float32);  view_909 = None
        mul_261: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_238, 0.5)
        mul_262: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_238, 0.7071067811865476);  convert_element_type_238 = None
        erf_23: "f32[512, 16, 4096]" = torch.ops.aten.erf.default(mul_262);  mul_262 = None
        add_260: "f32[512, 16, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_263: "f32[512, 16, 4096]" = torch.ops.aten.mul.Tensor(mul_261, add_260);  mul_261 = add_260 = None
        convert_element_type_239: "bf16[512, 16, 4096]" = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None
        view_910: "bf16[8192, 4096]" = torch.ops.aten.view.default(convert_element_type_239, _shape_param_911);  convert_element_type_239 = _shape_param_911 = None
        permute_578: "bf16[4096, 1024]" = torch.ops.aten.permute.default(arg358_1, [1, 0]);  arg358_1 = None
        addmm_47: "bf16[8192, 1024]" = torch.ops.aten.addmm.default(arg359_1, view_910, permute_578);  arg359_1 = view_910 = permute_578 = None
        view_911: "bf16[512, 16, 1024]" = torch.ops.aten.view.default(addmm_47, _shape_param_912);  addmm_47 = _shape_param_912 = None
        add_261: "bf16[512, 16, 1024]" = torch.ops.aten.add.Tensor(view_911, convert_element_type_237);  view_911 = convert_element_type_237 = None
        convert_element_type_240: "f32[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_261, torch.float32);  add_261 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_240, [2], correction = 0, keepdim = True)
        getitem_94: "f32[512, 16, 1]" = var_mean_47[0]
        getitem_95: "f32[512, 16, 1]" = var_mean_47[1];  var_mean_47 = None
        view_912: "i64[8192]" = torch.ops.aten.view.default(arg360_1, [-1]);  arg360_1 = None
        ne_24: "b8[8192]" = torch.ops.aten.ne.Scalar(view_912, -100)
        sub_95: "f32[512, 16, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_240, getitem_95);  convert_element_type_240 = getitem_95 = None
        add_262: "f32[512, 16, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47: "f32[512, 16, 1]" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
        mul_264: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_47);  sub_95 = rsqrt_47 = None
        mul_265: "f32[512, 16, 1024]" = torch.ops.aten.mul.Tensor(mul_264, arg361_1);  mul_264 = arg361_1 = None
        add_263: "f32[512, 16, 1024]" = torch.ops.aten.add.Tensor(mul_265, arg362_1);  mul_265 = arg362_1 = None
        convert_element_type_241: "bf16[512, 16, 1024]" = torch.ops.prims.convert_element_type.default(add_263, torch.bfloat16);  add_263 = None
        permute_579: "bf16[16, 512, 1024]" = torch.ops.aten.permute.default(convert_element_type_241, [1, 0, 2]);  convert_element_type_241 = None
        clone_50: "bf16[16, 512, 1024]" = torch.ops.aten.clone.default(permute_579, memory_format = torch.contiguous_format);  permute_579 = None
        view_913: "bf16[8192, 1024]" = torch.ops.aten.view.default(clone_50, _shape_param_913);  clone_50 = _shape_param_913 = None
        permute_580: "bf16[1024, 32000]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        addmm_48: "bf16[8192, 32000]" = torch.ops.aten.addmm.default(arg363_1, view_913, permute_580);  arg363_1 = view_913 = permute_580 = None
        view_914: "bf16[16, 512, 32000]" = torch.ops.aten.view.default(addmm_48, _shape_param_914);  addmm_48 = _shape_param_914 = None
        view_915: "bf16[8192, 32000]" = torch.ops.aten.view.default(view_914, _shape_param_915);  _shape_param_915 = None
        convert_element_type_242: "f32[8192, 32000]" = torch.ops.prims.convert_element_type.default(view_915, torch.float32);  view_915 = None
        amax_48: "f32[8192, 1]" = torch.ops.aten.amax.default(convert_element_type_242, [1], True)
        sub_96: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(convert_element_type_242, amax_48);  convert_element_type_242 = amax_48 = None
        exp_24: "f32[8192, 32000]" = torch.ops.aten.exp.default(sub_96)
        sum_25: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[8192, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_97: "f32[8192, 32000]" = torch.ops.aten.sub.Tensor(sub_96, log);  sub_96 = log = None
        convert_element_type_243: "bf16[8192, 32000]" = torch.ops.prims.convert_element_type.default(sub_97, torch.bfloat16);  sub_97 = None
        ne_25: "b8[8192]" = torch.ops.aten.ne.Scalar(view_912, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[8192]" = torch.ops.aten.where.self(ne_25, view_912, full);  ne_25 = full = None
        unsqueeze_723: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather: "bf16[8192, 1]" = torch.ops.aten.gather.default(convert_element_type_243, 1, unsqueeze_723);  convert_element_type_243 = unsqueeze_723 = None
        squeeze_240: "bf16[8192]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[8192]" = torch.ops.aten.neg.default(squeeze_240);  squeeze_240 = None
        full_1: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "bf16[8192]" = torch.ops.aten.where.self(ne_24, neg, full_1);  ne_24 = neg = full_1 = None
        sum_26: "bf16[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        ne_26: "b8[8192]" = torch.ops.aten.ne.Scalar(view_912, -100);  view_912 = None
        sum_27: "i64[]" = torch.ops.aten.sum.default(ne_26);  ne_26 = None
        convert_element_type_244: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_27, torch.bfloat16);  sum_27 = None
        div_25: "bf16[]" = torch.ops.aten.div.Tensor(sum_26, convert_element_type_244);  sum_26 = convert_element_type_244 = None
        slice_25: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(embedding, 0, -512, 9223372036854775807);  embedding = None
        slice_26: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_11, 0, -512, 9223372036854775807);  convert_element_type_11 = None
        slice_27: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_21, 0, -512, 9223372036854775807);  convert_element_type_21 = None
        slice_28: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_31, 0, -512, 9223372036854775807);  convert_element_type_31 = None
        slice_29: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_41, 0, -512, 9223372036854775807);  convert_element_type_41 = None
        slice_30: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_51, 0, -512, 9223372036854775807);  convert_element_type_51 = None
        slice_31: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_61, 0, -512, 9223372036854775807);  convert_element_type_61 = None
        slice_32: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_71, 0, -512, 9223372036854775807);  convert_element_type_71 = None
        slice_33: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_81, 0, -512, 9223372036854775807);  convert_element_type_81 = None
        slice_34: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_91, 0, -512, 9223372036854775807);  convert_element_type_91 = None
        slice_35: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_101, 0, -512, 9223372036854775807);  convert_element_type_101 = None
        slice_36: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_111, 0, -512, 9223372036854775807);  convert_element_type_111 = None
        slice_37: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_121, 0, -512, 9223372036854775807);  convert_element_type_121 = None
        slice_38: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_131, 0, -512, 9223372036854775807);  convert_element_type_131 = None
        slice_39: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_141, 0, -512, 9223372036854775807);  convert_element_type_141 = None
        slice_40: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_151, 0, -512, 9223372036854775807);  convert_element_type_151 = None
        slice_41: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_161, 0, -512, 9223372036854775807);  convert_element_type_161 = None
        slice_42: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_171, 0, -512, 9223372036854775807);  convert_element_type_171 = None
        slice_43: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_181, 0, -512, 9223372036854775807);  convert_element_type_181 = None
        slice_44: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_191, 0, -512, 9223372036854775807);  convert_element_type_191 = None
        slice_45: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_201, 0, -512, 9223372036854775807);  convert_element_type_201 = None
        slice_46: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_211, 0, -512, 9223372036854775807);  convert_element_type_211 = None
        slice_47: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_221, 0, -512, 9223372036854775807);  convert_element_type_221 = None
        slice_48: "bf16[512, 16, 1024]" = torch.ops.aten.slice.Tensor(convert_element_type_231, 0, -512, 9223372036854775807);  convert_element_type_231 = None
        return (view_914, div_25, slice_25, slice_26, slice_27, slice_28, slice_29, slice_30, slice_31, slice_32, slice_33, slice_34, slice_35, slice_36, slice_37, slice_38, slice_39, slice_40, slice_41, slice_42, slice_43, slice_44, slice_45, slice_46, slice_47, slice_48)



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
