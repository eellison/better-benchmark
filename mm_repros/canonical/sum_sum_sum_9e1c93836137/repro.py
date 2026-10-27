"""
Standalone repro captured via capture_hook.
Label: BERT_pytorch
Pattern hash: 9e1c93836137
Shape hash: c9ac8fca
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
    def forward(self, arg0_1: "f32[16, 128, 20005]", arg1_1: "f32[16, 128, 20005]", arg2_1: "bf16[20005, 768]", arg3_1: "bf16[2048, 768]", arg4_1: "f32[16, 2]", arg5_1: "f32[16, 2]", arg6_1: "bf16[2, 768]", arg7_1: "bf16[16, 768]", arg8_1: "b8[16, 128, 768]", arg9_1: "b8[16, 128, 768]", arg10_1: "bf16[768, 3072]", arg11_1: "bf16[2048, 3072]", arg12_1: "b8[16, 128, 3072]", arg13_1: "bf16[2048, 3072]", arg14_1: "bf16[3072, 768]", arg15_1: "bf16[2048, 768]", arg16_1: "f32[768]", arg17_1: "f32[16, 128, 768]", arg18_1: "f32[16, 128, 1]", arg19_1: "b8[16, 128, 768]", arg20_1: "bf16[768, 768]", arg21_1: "bf16[2048, 768]", arg22_1: "bf16[192, 128, 128]", arg23_1: "bf16[192, 64, 128]", arg24_1: "b8[16, 12, 128, 128]", arg25_1: "bf16[16, 12, 128, 128]", arg26_1: "f32[16, 12, 128, 1]", arg27_1: "f32[16, 12, 128, 1]", arg28_1: "b8[16, 1, 128, 128]", arg29_1: "bf16[192, 64, 128]", arg30_1: "bf16[192, 128, 64]", arg31_1: "bf16[768, 768]", arg32_1: "bf16[2048, 768]", arg33_1: "bf16[768, 768]", arg34_1: "bf16[768, 768]", arg35_1: "f32[768]", arg36_1: "f32[16, 128, 768]", arg37_1: "f32[16, 128, 1]", arg38_1: "b8[16, 128, 768]", arg39_1: "b8[16, 128, 768]", arg40_1: "bf16[768, 3072]", arg41_1: "bf16[2048, 3072]", arg42_1: "b8[16, 128, 3072]", arg43_1: "bf16[2048, 3072]", arg44_1: "bf16[3072, 768]", arg45_1: "bf16[2048, 768]", arg46_1: "f32[768]", arg47_1: "f32[16, 128, 768]", arg48_1: "f32[16, 128, 1]", arg49_1: "b8[16, 128, 768]", arg50_1: "bf16[768, 768]", arg51_1: "bf16[2048, 768]", arg52_1: "bf16[192, 128, 128]", arg53_1: "bf16[192, 64, 128]", arg54_1: "b8[16, 12, 128, 128]", arg55_1: "bf16[16, 12, 128, 128]", arg56_1: "f32[16, 12, 128, 1]", arg57_1: "f32[16, 12, 128, 1]", arg58_1: "bf16[192, 64, 128]", arg59_1: "bf16[192, 128, 64]", arg60_1: "bf16[768, 768]", arg61_1: "bf16[2048, 768]", arg62_1: "bf16[768, 768]", arg63_1: "bf16[768, 768]", arg64_1: "f32[768]", arg65_1: "f32[16, 128, 768]", arg66_1: "f32[16, 128, 1]", arg67_1: "b8[16, 128, 768]", arg68_1: "b8[16, 128, 768]", arg69_1: "bf16[768, 3072]", arg70_1: "bf16[2048, 3072]", arg71_1: "b8[16, 128, 3072]", arg72_1: "bf16[2048, 3072]", arg73_1: "bf16[3072, 768]", arg74_1: "bf16[2048, 768]", arg75_1: "f32[768]", arg76_1: "f32[16, 128, 768]", arg77_1: "f32[16, 128, 1]", arg78_1: "b8[16, 128, 768]", arg79_1: "bf16[768, 768]", arg80_1: "bf16[2048, 768]", arg81_1: "bf16[192, 128, 128]", arg82_1: "bf16[192, 64, 128]", arg83_1: "b8[16, 12, 128, 128]", arg84_1: "bf16[16, 12, 128, 128]", arg85_1: "f32[16, 12, 128, 1]", arg86_1: "f32[16, 12, 128, 1]", arg87_1: "bf16[192, 64, 128]", arg88_1: "bf16[192, 128, 64]", arg89_1: "bf16[768, 768]", arg90_1: "bf16[2048, 768]", arg91_1: "bf16[768, 768]", arg92_1: "bf16[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[16, 128, 768]", arg95_1: "f32[16, 128, 1]", arg96_1: "b8[16, 128, 768]", arg97_1: "b8[16, 128, 768]", arg98_1: "bf16[768, 3072]", arg99_1: "bf16[2048, 3072]", arg100_1: "b8[16, 128, 3072]", arg101_1: "bf16[2048, 3072]", arg102_1: "bf16[3072, 768]", arg103_1: "bf16[2048, 768]", arg104_1: "f32[768]", arg105_1: "f32[16, 128, 768]", arg106_1: "f32[16, 128, 1]", arg107_1: "b8[16, 128, 768]", arg108_1: "bf16[768, 768]", arg109_1: "bf16[2048, 768]", arg110_1: "bf16[192, 128, 128]", arg111_1: "bf16[192, 64, 128]", arg112_1: "b8[16, 12, 128, 128]", arg113_1: "bf16[16, 12, 128, 128]", arg114_1: "f32[16, 12, 128, 1]", arg115_1: "f32[16, 12, 128, 1]", arg116_1: "bf16[192, 64, 128]", arg117_1: "bf16[192, 128, 64]", arg118_1: "bf16[768, 768]", arg119_1: "bf16[2048, 768]", arg120_1: "bf16[768, 768]", arg121_1: "bf16[768, 768]", arg122_1: "f32[768]", arg123_1: "f32[16, 128, 768]", arg124_1: "f32[16, 128, 1]", arg125_1: "b8[16, 128, 768]", arg126_1: "b8[16, 128, 768]", arg127_1: "bf16[768, 3072]", arg128_1: "bf16[2048, 3072]", arg129_1: "b8[16, 128, 3072]", arg130_1: "bf16[2048, 3072]", arg131_1: "bf16[3072, 768]", arg132_1: "bf16[2048, 768]", arg133_1: "f32[768]", arg134_1: "f32[16, 128, 768]", arg135_1: "f32[16, 128, 1]", arg136_1: "b8[16, 128, 768]", arg137_1: "bf16[768, 768]", arg138_1: "bf16[2048, 768]", arg139_1: "bf16[192, 128, 128]", arg140_1: "bf16[192, 64, 128]", arg141_1: "b8[16, 12, 128, 128]", arg142_1: "bf16[16, 12, 128, 128]", arg143_1: "f32[16, 12, 128, 1]", arg144_1: "f32[16, 12, 128, 1]", arg145_1: "bf16[192, 64, 128]", arg146_1: "bf16[192, 128, 64]", arg147_1: "bf16[768, 768]", arg148_1: "bf16[2048, 768]", arg149_1: "bf16[768, 768]", arg150_1: "bf16[768, 768]", arg151_1: "f32[768]", arg152_1: "f32[16, 128, 768]", arg153_1: "f32[16, 128, 1]", arg154_1: "b8[16, 128, 768]", arg155_1: "b8[16, 128, 768]", arg156_1: "bf16[768, 3072]", arg157_1: "bf16[2048, 3072]", arg158_1: "b8[16, 128, 3072]", arg159_1: "bf16[2048, 3072]", arg160_1: "bf16[3072, 768]", arg161_1: "bf16[2048, 768]", arg162_1: "f32[768]", arg163_1: "f32[16, 128, 768]", arg164_1: "f32[16, 128, 1]", arg165_1: "b8[16, 128, 768]", arg166_1: "bf16[768, 768]", arg167_1: "bf16[2048, 768]", arg168_1: "bf16[192, 128, 128]", arg169_1: "bf16[192, 64, 128]", arg170_1: "b8[16, 12, 128, 128]", arg171_1: "bf16[16, 12, 128, 128]", arg172_1: "f32[16, 12, 128, 1]", arg173_1: "f32[16, 12, 128, 1]", arg174_1: "bf16[192, 64, 128]", arg175_1: "bf16[192, 128, 64]", arg176_1: "bf16[768, 768]", arg177_1: "bf16[2048, 768]", arg178_1: "bf16[768, 768]", arg179_1: "bf16[768, 768]", arg180_1: "f32[768]", arg181_1: "f32[16, 128, 768]", arg182_1: "f32[16, 128, 1]", arg183_1: "b8[16, 128, 768]", arg184_1: "b8[16, 128, 768]", arg185_1: "bf16[768, 3072]", arg186_1: "bf16[2048, 3072]", arg187_1: "b8[16, 128, 3072]", arg188_1: "bf16[2048, 3072]", arg189_1: "bf16[3072, 768]", arg190_1: "bf16[2048, 768]", arg191_1: "f32[768]", arg192_1: "f32[16, 128, 768]", arg193_1: "f32[16, 128, 1]", arg194_1: "b8[16, 128, 768]", arg195_1: "bf16[768, 768]", arg196_1: "bf16[2048, 768]", arg197_1: "bf16[192, 128, 128]", arg198_1: "bf16[192, 64, 128]", arg199_1: "b8[16, 12, 128, 128]", arg200_1: "bf16[16, 12, 128, 128]", arg201_1: "f32[16, 12, 128, 1]", arg202_1: "f32[16, 12, 128, 1]", arg203_1: "bf16[192, 64, 128]", arg204_1: "bf16[192, 128, 64]", arg205_1: "bf16[768, 768]", arg206_1: "bf16[2048, 768]", arg207_1: "bf16[768, 768]", arg208_1: "bf16[768, 768]", arg209_1: "f32[768]", arg210_1: "f32[16, 128, 768]", arg211_1: "f32[16, 128, 1]", arg212_1: "b8[16, 128, 768]", arg213_1: "b8[16, 128, 768]", arg214_1: "bf16[768, 3072]", arg215_1: "bf16[2048, 3072]", arg216_1: "b8[16, 128, 3072]", arg217_1: "bf16[2048, 3072]", arg218_1: "bf16[3072, 768]", arg219_1: "bf16[2048, 768]", arg220_1: "f32[768]", arg221_1: "f32[16, 128, 768]", arg222_1: "f32[16, 128, 1]", arg223_1: "b8[16, 128, 768]", arg224_1: "bf16[768, 768]", arg225_1: "bf16[2048, 768]", arg226_1: "bf16[192, 128, 128]", arg227_1: "bf16[192, 64, 128]", arg228_1: "b8[16, 12, 128, 128]", arg229_1: "bf16[16, 12, 128, 128]", arg230_1: "f32[16, 12, 128, 1]", arg231_1: "f32[16, 12, 128, 1]", arg232_1: "bf16[192, 64, 128]", arg233_1: "bf16[192, 128, 64]", arg234_1: "bf16[768, 768]", arg235_1: "bf16[2048, 768]", arg236_1: "bf16[768, 768]", arg237_1: "bf16[768, 768]", arg238_1: "f32[768]", arg239_1: "f32[16, 128, 768]", arg240_1: "f32[16, 128, 1]", arg241_1: "b8[16, 128, 768]", arg242_1: "b8[16, 128, 768]", arg243_1: "bf16[768, 3072]", arg244_1: "bf16[2048, 3072]", arg245_1: "b8[16, 128, 3072]", arg246_1: "bf16[2048, 3072]", arg247_1: "bf16[3072, 768]", arg248_1: "bf16[2048, 768]", arg249_1: "f32[768]", arg250_1: "f32[16, 128, 768]", arg251_1: "f32[16, 128, 1]", arg252_1: "b8[16, 128, 768]", arg253_1: "bf16[768, 768]", arg254_1: "bf16[2048, 768]", arg255_1: "bf16[192, 128, 128]", arg256_1: "bf16[192, 64, 128]", arg257_1: "b8[16, 12, 128, 128]", arg258_1: "bf16[16, 12, 128, 128]", arg259_1: "f32[16, 12, 128, 1]", arg260_1: "f32[16, 12, 128, 1]", arg261_1: "bf16[192, 64, 128]", arg262_1: "bf16[192, 128, 64]", arg263_1: "bf16[768, 768]", arg264_1: "bf16[2048, 768]", arg265_1: "bf16[768, 768]", arg266_1: "bf16[768, 768]", arg267_1: "f32[768]", arg268_1: "f32[16, 128, 768]", arg269_1: "f32[16, 128, 1]", arg270_1: "b8[16, 128, 768]", arg271_1: "b8[16, 128, 768]", arg272_1: "bf16[768, 3072]", arg273_1: "bf16[2048, 3072]", arg274_1: "b8[16, 128, 3072]", arg275_1: "bf16[2048, 3072]", arg276_1: "bf16[3072, 768]", arg277_1: "bf16[2048, 768]", arg278_1: "f32[768]", arg279_1: "f32[16, 128, 768]", arg280_1: "f32[16, 128, 1]", arg281_1: "b8[16, 128, 768]", arg282_1: "bf16[768, 768]", arg283_1: "bf16[2048, 768]", arg284_1: "bf16[192, 128, 128]", arg285_1: "bf16[192, 64, 128]", arg286_1: "b8[16, 12, 128, 128]", arg287_1: "bf16[16, 12, 128, 128]", arg288_1: "f32[16, 12, 128, 1]", arg289_1: "f32[16, 12, 128, 1]", arg290_1: "bf16[192, 64, 128]", arg291_1: "bf16[192, 128, 64]", arg292_1: "bf16[768, 768]", arg293_1: "bf16[2048, 768]", arg294_1: "bf16[768, 768]", arg295_1: "bf16[768, 768]", arg296_1: "f32[768]", arg297_1: "f32[16, 128, 768]", arg298_1: "f32[16, 128, 1]", arg299_1: "b8[16, 128, 768]", arg300_1: "b8[16, 128, 768]", arg301_1: "bf16[768, 3072]", arg302_1: "bf16[2048, 3072]", arg303_1: "b8[16, 128, 3072]", arg304_1: "bf16[2048, 3072]", arg305_1: "bf16[3072, 768]", arg306_1: "bf16[2048, 768]", arg307_1: "f32[768]", arg308_1: "f32[16, 128, 768]", arg309_1: "f32[16, 128, 1]", arg310_1: "b8[16, 128, 768]", arg311_1: "bf16[768, 768]", arg312_1: "bf16[2048, 768]", arg313_1: "bf16[192, 128, 128]", arg314_1: "bf16[192, 64, 128]", arg315_1: "b8[16, 12, 128, 128]", arg316_1: "bf16[16, 12, 128, 128]", arg317_1: "f32[16, 12, 128, 1]", arg318_1: "f32[16, 12, 128, 1]", arg319_1: "bf16[192, 64, 128]", arg320_1: "bf16[192, 128, 64]", arg321_1: "bf16[768, 768]", arg322_1: "bf16[2048, 768]", arg323_1: "bf16[768, 768]", arg324_1: "bf16[768, 768]", arg325_1: "f32[768]", arg326_1: "f32[16, 128, 768]", arg327_1: "f32[16, 128, 1]", arg328_1: "b8[16, 128, 768]", arg329_1: "b8[16, 128, 768]", arg330_1: "bf16[768, 3072]", arg331_1: "bf16[2048, 3072]", arg332_1: "b8[16, 128, 3072]", arg333_1: "bf16[2048, 3072]", arg334_1: "bf16[3072, 768]", arg335_1: "bf16[2048, 768]", arg336_1: "f32[768]", arg337_1: "f32[16, 128, 768]", arg338_1: "f32[16, 128, 1]", arg339_1: "b8[16, 128, 768]", arg340_1: "bf16[768, 768]", arg341_1: "bf16[2048, 768]", arg342_1: "bf16[192, 128, 128]", arg343_1: "bf16[192, 64, 128]", arg344_1: "b8[16, 12, 128, 128]", arg345_1: "bf16[192, 128, 128]", arg346_1: "f32[16, 12, 128, 1]", arg347_1: "f32[16, 12, 128, 1]", arg348_1: "bf16[192, 64, 128]", arg349_1: "bf16[192, 128, 64]", arg350_1: "bf16[768, 768]", arg351_1: "bf16[2048, 768]", arg352_1: "bf16[768, 768]", arg353_1: "bf16[768, 768]", arg354_1: "f32[768]", arg355_1: "f32[16, 128, 768]", arg356_1: "f32[16, 128, 1]", arg357_1: "b8[16, 128, 768]", arg358_1: "i64[16, 128]", arg359_1: "i64[16, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430):
        # No stacktrace found for following nodes
        sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(arg0_1, [-1], True)
        exp: "f32[16, 128, 20005]" = torch.ops.aten.exp.default(arg1_1);  arg1_1 = None
        mul: "f32[16, 128, 20005]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub: "f32[16, 128, 20005]" = torch.ops.aten.sub.Tensor(arg0_1, mul);  arg0_1 = mul = None
        convert_element_type: "bf16[16, 128, 20005]" = torch.ops.prims.convert_element_type.default(sub, torch.bfloat16);  sub = None
        view: "bf16[2048, 20005]" = torch.ops.aten.view.default(convert_element_type, _shape_param_0);  convert_element_type = _shape_param_0 = None
        constant_pad_nd: "bf16[2048, 20008]" = torch.ops.aten.constant_pad_nd.default(view, _shape_param_1);  _shape_param_1 = None
        constant_pad_nd_1: "bf16[20008, 768]" = torch.ops.aten.constant_pad_nd.default(arg2_1, _shape_param_2);  arg2_1 = _shape_param_2 = None
        mm: "bf16[2048, 768]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        permute: "bf16[20005, 2048]" = torch.ops.aten.permute.default(view, [1, 0])
        constant_pad_nd_2: "bf16[20008, 2048]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_3);  permute = _shape_param_3 = None
        mm_1: "bf16[20008, 768]" = torch.ops.aten.mm.default(constant_pad_nd_2, arg3_1);  constant_pad_nd_2 = arg3_1 = None
        slice_1: "bf16[20005, 768]" = torch.ops.aten.slice.Tensor(mm_1, 0, 0, -3);  mm_1 = None
        sum_2: "f32[1, 20005]" = torch.ops.aten.sum.dim_IntList(view, [0], True, dtype = torch.float32);  view = None
        view_1: "f32[20005]" = torch.ops.aten.view.default(sum_2, _shape_param_4);  sum_2 = _shape_param_4 = None
        convert_element_type_1: "bf16[20005]" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        view_2: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm, _shape_param_5);  mm = _shape_param_5 = None
        convert_element_type_2: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        convert_element_type_3: "f32[20005, 768]" = torch.ops.prims.convert_element_type.default(slice_1, torch.float32);  slice_1 = None
        convert_element_type_4: "f32[20005]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        exp_1: "f32[16, 2]" = torch.ops.aten.exp.default(arg4_1);  arg4_1 = None
        sum_3: "f32[16, 1]" = torch.ops.aten.sum.dim_IntList(arg5_1, [-1], True)
        mul_1: "f32[16, 2]" = torch.ops.aten.mul.Tensor(exp_1, sum_3);  exp_1 = sum_3 = None
        sub_1: "f32[16, 2]" = torch.ops.aten.sub.Tensor(arg5_1, mul_1);  arg5_1 = mul_1 = None
        convert_element_type_5: "bf16[16, 2]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        mm_2: "bf16[16, 768]" = torch.ops.aten.mm.default(convert_element_type_5, arg6_1);  arg6_1 = None
        permute_1: "bf16[2, 16]" = torch.ops.aten.permute.default(convert_element_type_5, [1, 0])
        mm_3: "bf16[2, 768]" = torch.ops.aten.mm.default(permute_1, arg7_1);  permute_1 = arg7_1 = None
        sum_4: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(convert_element_type_5, [0], True, dtype = torch.float32);  convert_element_type_5 = None
        view_3: "f32[2]" = torch.ops.aten.view.default(sum_4, _shape_param_6);  sum_4 = _shape_param_6 = None
        convert_element_type_6: "bf16[2]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        convert_element_type_7: "f32[16, 768]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_8: "f32[2, 768]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_9: "f32[2]" = torch.ops.prims.convert_element_type.default(convert_element_type_6, torch.float32);  convert_element_type_6 = None
        full: "f32[16, 128, 768]" = torch.ops.aten.full.default(_shape_param_7, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_7 = None
        select_scatter: "f32[16, 128, 768]" = torch.ops.aten.select_scatter.default(full, convert_element_type_7, 1, 0);  full = convert_element_type_7 = None
        add: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_2, select_scatter);  convert_element_type_2 = select_scatter = None
        convert_element_type_10: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        mul_2: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_3: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add, mul_2);  add = mul_2 = None
        convert_element_type_11: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_3, torch.bfloat16)
        convert_element_type_12: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        mul_4: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_5: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, mul_4);  convert_element_type_11 = mul_4 = None
        view_4: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_5, _shape_param_8);  mul_5 = _shape_param_8 = None
        mm_4: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_4, arg10_1);  arg10_1 = None
        permute_2: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_5: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_2, arg11_1);  permute_2 = arg11_1 = None
        sum_5: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[768]" = torch.ops.aten.view.default(sum_5, _shape_param_9);  sum_5 = _shape_param_9 = None
        convert_element_type_13: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_4, _shape_param_10);  mm_4 = _shape_param_10 = None
        convert_element_type_14: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_15: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        convert_element_type_16: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.bfloat16);  arg12_1 = None
        mul_6: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_7: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_6, mul_6);  view_6 = mul_6 = None
        convert_element_type_17: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_7, torch.float32);  mul_7 = None
        view_7: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg13_1, _shape_param_11);  arg13_1 = _shape_param_11 = None
        convert_element_type_18: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32);  view_7 = None
        mul_8: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.7071067811865476)
        erf: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_1: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        mul_10: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_18, convert_element_type_18)
        mul_11: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_10, -0.5);  mul_10 = None
        exp_2: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_11);  mul_11 = None
        mul_12: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_13: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_18, mul_12);  convert_element_type_18 = mul_12 = None
        add_2: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        mul_14: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_17, add_2);  convert_element_type_17 = add_2 = None
        convert_element_type_19: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        view_8: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_19, _shape_param_12);  convert_element_type_19 = _shape_param_12 = None
        mm_6: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_8, arg14_1);  arg14_1 = None
        permute_3: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_7: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_3, arg15_1);  permute_3 = arg15_1 = None
        sum_6: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[3072]" = torch.ops.aten.view.default(sum_6, _shape_param_13);  sum_6 = _shape_param_13 = None
        convert_element_type_20: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_6, _shape_param_14);  mm_6 = _shape_param_14 = None
        convert_element_type_21: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_22: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_23: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_20, torch.float32);  convert_element_type_20 = None
        sum_7: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_21, [0, 1], True, dtype = torch.float32)
        view_11: "f32[768]" = torch.ops.aten.view.default(sum_7, _shape_param_15);  sum_7 = _shape_param_15 = None
        mul_15: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg16_1, arg17_1)
        add_3: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg18_1, 1e-06)
        div: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_15, add_3);  mul_15 = None
        div_1: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div, add_3);  div = None
        neg: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_21)
        mul_16: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg, div_1);  neg = div_1 = None
        div_2: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_21, add_3);  convert_element_type_21 = add_3 = None
        sum_8: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_16, [2], True, dtype = torch.float32);  mul_16 = None
        mul_17: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg16_1);  arg16_1 = None
        mul_18: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_2, arg17_1);  div_2 = None
        sum_9: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_18, [0, 1], True, dtype = torch.float32);  mul_18 = None
        view_12: "f32[768]" = torch.ops.aten.view.default(sum_9, _shape_param_16);  sum_9 = _shape_param_16 = None
        neg_1: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_17)
        sum_10: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_1, [2], True, dtype = torch.float32);  neg_1 = None
        add_4: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_3, mul_17);  mul_3 = mul_17 = None
        mul_19: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg18_1, 2)
        div_3: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_8, mul_19);  sum_8 = mul_19 = None
        eq: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg18_1, 0);  arg18_1 = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq, full_1, div_3);  eq = div_3 = None
        mul_20: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where, 0.002607561929595828);  where = None
        mul_21: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_20, arg17_1);  mul_20 = arg17_1 = None
        add_5: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_4, mul_21);  add_4 = mul_21 = None
        expand: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_10, _shape_param_17);  sum_10 = _shape_param_17 = None
        div_4: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand, 768);  expand = None
        add_6: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_5, div_4);  add_5 = div_4 = None
        convert_element_type_24: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16)
        convert_element_type_25: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        mul_22: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_23: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, mul_22);  convert_element_type_24 = mul_22 = None
        view_13: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_23, _shape_param_18);  mul_23 = _shape_param_18 = None
        mm_8: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_13, arg20_1);  arg20_1 = None
        permute_4: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_13, [1, 0])
        mm_9: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_4, arg21_1);  permute_4 = arg21_1 = None
        sum_11: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_13, [0], True, dtype = torch.float32);  view_13 = None
        view_14: "f32[768]" = torch.ops.aten.view.default(sum_11, _shape_param_19);  sum_11 = _shape_param_19 = None
        convert_element_type_26: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16);  view_14 = None
        view_15: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_8, _shape_param_20);  mm_8 = _shape_param_20 = None
        convert_element_type_27: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_28: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_26, torch.float32);  convert_element_type_26 = None
        view_16: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_15, _shape_param_21);  view_15 = _shape_param_21 = None
        permute_5: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_17: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_22);  clone = _shape_param_22 = None
        bmm: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg22_1, view_17);  arg22_1 = None
        bmm_1: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_17, arg23_1);  view_17 = arg23_1 = None
        view_18: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm, _shape_param_23);  bmm = _shape_param_23 = None
        view_19: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_24);  bmm_1 = _shape_param_24 = None
        convert_element_type_29: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        convert_element_type_30: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.float32);  arg24_1 = None
        mul_24: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_25: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_29, mul_24);  convert_element_type_29 = mul_24 = None
        convert_element_type_31: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.float32);  arg25_1 = None
        sub_2: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_31, arg26_1);  convert_element_type_31 = arg26_1 = None
        exp_3: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        div_5: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, arg27_1);  exp_3 = arg27_1 = None
        mul_26: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_25, div_5);  mul_25 = None
        sum_12: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_26, [-1], True)
        neg_2: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_12, mul_26);  neg_2 = sum_12 = mul_26 = None
        convert_element_type_32: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        eq_1: "b8[16, 1, 128, 128]" = torch.ops.aten.eq.Scalar(arg28_1, 0);  arg28_1 = None
        where_1: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_32);  convert_element_type_32 = None
        div_6: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_1, 8.0);  where_1 = None
        view_20: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_6, _shape_param_25);  div_6 = _shape_param_25 = None
        bmm_2: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg29_1, view_20);  arg29_1 = None
        bmm_3: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_20, arg30_1);  view_20 = arg30_1 = None
        view_21: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_26);  bmm_2 = _shape_param_26 = None
        view_22: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_27);  bmm_3 = _shape_param_27 = None
        permute_6: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_21, [0, 1, 3, 2]);  view_21 = None
        permute_7: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_1: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_23: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_1, _shape_param_28);  clone_1 = _shape_param_28 = None
        view_24: "bf16[2048, 768]" = torch.ops.aten.view.default(view_23, _shape_param_29);  view_23 = _shape_param_29 = None
        mm_10: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_24, arg31_1);  arg31_1 = None
        permute_8: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_24, [1, 0])
        mm_11: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_8, arg32_1);  permute_8 = None
        sum_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_24, [0], True, dtype = torch.float32);  view_24 = None
        view_25: "f32[768]" = torch.ops.aten.view.default(sum_13, _shape_param_30);  sum_13 = _shape_param_30 = None
        convert_element_type_33: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_25, torch.bfloat16);  view_25 = None
        view_26: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_10, _shape_param_31);  mm_10 = _shape_param_31 = None
        convert_element_type_34: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_26, torch.float32);  view_26 = None
        convert_element_type_35: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_36: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_33, torch.float32);  convert_element_type_33 = None
        permute_9: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1, 3]);  permute_6 = None
        view_27: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_9, _shape_param_32);  permute_9 = _shape_param_32 = None
        clone_2: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_27, memory_format = torch.contiguous_format);  view_27 = None
        view_28: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_2, _shape_param_33);  clone_2 = _shape_param_33 = None
        mm_12: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_28, arg33_1);  arg33_1 = None
        permute_10: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_13: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_10, arg32_1);  permute_10 = None
        sum_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[768]" = torch.ops.aten.view.default(sum_14, _shape_param_34);  sum_14 = _shape_param_34 = None
        convert_element_type_37: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_12, _shape_param_35);  mm_12 = _shape_param_35 = None
        convert_element_type_38: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        add_7: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_34, convert_element_type_38);  convert_element_type_34 = convert_element_type_38 = None
        convert_element_type_39: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_40: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_37, torch.float32);  convert_element_type_37 = None
        permute_11: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
        clone_3: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_31: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_3, _shape_param_36);  clone_3 = _shape_param_36 = None
        view_32: "bf16[2048, 768]" = torch.ops.aten.view.default(view_31, _shape_param_37);  view_31 = _shape_param_37 = None
        mm_14: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_32, arg34_1);  arg34_1 = None
        permute_12: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_15: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_12, arg32_1);  permute_12 = arg32_1 = None
        sum_15: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_32, [0], True, dtype = torch.float32);  view_32 = None
        view_33: "f32[768]" = torch.ops.aten.view.default(sum_15, _shape_param_38);  sum_15 = _shape_param_38 = None
        convert_element_type_41: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_33, torch.bfloat16);  view_33 = None
        view_34: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_14, _shape_param_39);  mm_14 = _shape_param_39 = None
        convert_element_type_42: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        add_8: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_42);  add_7 = convert_element_type_42 = None
        convert_element_type_43: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_44: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_41, torch.float32);  convert_element_type_41 = None
        sum_16: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_8, [0, 1], True, dtype = torch.float32)
        view_35: "f32[768]" = torch.ops.aten.view.default(sum_16, _shape_param_40);  sum_16 = _shape_param_40 = None
        mul_27: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg35_1, arg36_1)
        add_9: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg37_1, 1e-06)
        div_7: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_27, add_9);  mul_27 = None
        div_8: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_7, add_9);  div_7 = None
        neg_3: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_8)
        mul_28: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_3, div_8);  neg_3 = div_8 = None
        div_9: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_8, add_9);  add_8 = add_9 = None
        sum_17: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_28, [2], True, dtype = torch.float32);  mul_28 = None
        mul_29: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_9, arg35_1);  arg35_1 = None
        mul_30: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_9, arg36_1);  div_9 = None
        sum_18: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_30, [0, 1], True, dtype = torch.float32);  mul_30 = None
        view_36: "f32[768]" = torch.ops.aten.view.default(sum_18, _shape_param_41);  sum_18 = _shape_param_41 = None
        neg_4: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_29)
        sum_19: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_4, [2], True, dtype = torch.float32);  neg_4 = None
        add_10: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_6, mul_29);  add_6 = mul_29 = None
        mul_31: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg37_1, 2)
        div_10: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_17, mul_31);  sum_17 = mul_31 = None
        eq_2: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg37_1, 0);  arg37_1 = None
        where_2: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_2, full_1, div_10);  eq_2 = div_10 = None
        mul_32: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_2, 0.002607561929595828);  where_2 = None
        mul_33: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_32, arg36_1);  mul_32 = arg36_1 = None
        add_11: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_10, mul_33);  add_10 = mul_33 = None
        expand_1: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_19, _shape_param_42);  sum_19 = _shape_param_42 = None
        div_11: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_1, 768);  expand_1 = None
        add_12: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_11, div_11);  add_11 = div_11 = None
        convert_element_type_45: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.float32);  arg38_1 = None
        mul_34: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
        mul_35: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_12, mul_34);  add_12 = mul_34 = None
        convert_element_type_46: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16)
        convert_element_type_47: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg39_1, torch.bfloat16);  arg39_1 = None
        mul_36: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
        mul_37: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_46, mul_36);  convert_element_type_46 = mul_36 = None
        view_37: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_37, _shape_param_43);  mul_37 = _shape_param_43 = None
        mm_16: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_37, arg40_1);  arg40_1 = None
        permute_13: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_37, [1, 0])
        mm_17: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_13, arg41_1);  permute_13 = arg41_1 = None
        sum_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_37, [0], True, dtype = torch.float32);  view_37 = None
        view_38: "f32[768]" = torch.ops.aten.view.default(sum_20, _shape_param_44);  sum_20 = _shape_param_44 = None
        convert_element_type_48: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_38, torch.bfloat16);  view_38 = None
        view_39: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_16, _shape_param_45);  mm_16 = _shape_param_45 = None
        convert_element_type_49: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_50: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_48, torch.float32);  convert_element_type_48 = None
        convert_element_type_51: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        mul_38: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
        mul_39: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_39, mul_38);  view_39 = mul_38 = None
        convert_element_type_52: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_39, torch.float32);  mul_39 = None
        view_40: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg43_1, _shape_param_46);  arg43_1 = _shape_param_46 = None
        convert_element_type_53: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_40, torch.float32);  view_40 = None
        mul_40: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 0.7071067811865476)
        erf_1: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_13: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_41: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
        mul_42: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_53, convert_element_type_53)
        mul_43: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_42, -0.5);  mul_42 = None
        exp_4: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_43);  mul_43 = None
        mul_44: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_45: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_53, mul_44);  convert_element_type_53 = mul_44 = None
        add_14: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_41, mul_45);  mul_41 = mul_45 = None
        mul_46: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_52, add_14);  convert_element_type_52 = add_14 = None
        convert_element_type_54: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None
        view_41: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_54, _shape_param_47);  convert_element_type_54 = _shape_param_47 = None
        mm_18: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_41, arg44_1);  arg44_1 = None
        permute_14: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_41, [1, 0])
        mm_19: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_14, arg45_1);  permute_14 = arg45_1 = None
        sum_21: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True, dtype = torch.float32);  view_41 = None
        view_42: "f32[3072]" = torch.ops.aten.view.default(sum_21, _shape_param_48);  sum_21 = _shape_param_48 = None
        convert_element_type_55: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_42, torch.bfloat16);  view_42 = None
        view_43: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_18, _shape_param_49);  mm_18 = _shape_param_49 = None
        convert_element_type_56: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        convert_element_type_57: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_58: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_55, torch.float32);  convert_element_type_55 = None
        sum_22: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_56, [0, 1], True, dtype = torch.float32)
        view_44: "f32[768]" = torch.ops.aten.view.default(sum_22, _shape_param_50);  sum_22 = _shape_param_50 = None
        mul_47: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg46_1, arg47_1)
        add_15: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg48_1, 1e-06)
        div_12: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_47, add_15);  mul_47 = None
        div_13: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_12, add_15);  div_12 = None
        neg_5: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_56)
        mul_48: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_5, div_13);  neg_5 = div_13 = None
        div_14: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_56, add_15);  convert_element_type_56 = add_15 = None
        sum_23: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_48, [2], True, dtype = torch.float32);  mul_48 = None
        mul_49: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_14, arg46_1);  arg46_1 = None
        mul_50: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_14, arg47_1);  div_14 = None
        sum_24: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_50, [0, 1], True, dtype = torch.float32);  mul_50 = None
        view_45: "f32[768]" = torch.ops.aten.view.default(sum_24, _shape_param_51);  sum_24 = _shape_param_51 = None
        neg_6: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_49)
        sum_25: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_6, [2], True, dtype = torch.float32);  neg_6 = None
        add_16: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_35, mul_49);  mul_35 = mul_49 = None
        mul_51: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg48_1, 2)
        div_15: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_23, mul_51);  sum_23 = mul_51 = None
        eq_3: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg48_1, 0);  arg48_1 = None
        where_3: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_3, full_1, div_15);  eq_3 = div_15 = None
        mul_52: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_3, 0.002607561929595828);  where_3 = None
        mul_53: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_52, arg47_1);  mul_52 = arg47_1 = None
        add_17: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_16, mul_53);  add_16 = mul_53 = None
        expand_2: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_25, _shape_param_52);  sum_25 = _shape_param_52 = None
        div_16: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_2, 768);  expand_2 = None
        add_18: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_17, div_16);  add_17 = div_16 = None
        convert_element_type_59: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16)
        convert_element_type_60: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.bfloat16);  arg49_1 = None
        mul_54: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
        mul_55: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_59, mul_54);  convert_element_type_59 = mul_54 = None
        view_46: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_55, _shape_param_53);  mul_55 = _shape_param_53 = None
        mm_20: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_46, arg50_1);  arg50_1 = None
        permute_15: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_46, [1, 0])
        mm_21: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_15, arg51_1);  permute_15 = arg51_1 = None
        sum_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_46, [0], True, dtype = torch.float32);  view_46 = None
        view_47: "f32[768]" = torch.ops.aten.view.default(sum_26, _shape_param_54);  sum_26 = _shape_param_54 = None
        convert_element_type_61: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_47, torch.bfloat16);  view_47 = None
        view_48: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_20, _shape_param_55);  mm_20 = _shape_param_55 = None
        convert_element_type_62: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_63: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_61, torch.float32);  convert_element_type_61 = None
        view_49: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_48, _shape_param_56);  view_48 = _shape_param_56 = None
        permute_16: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        clone_4: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_50: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_57);  clone_4 = _shape_param_57 = None
        bmm_4: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg52_1, view_50);  arg52_1 = None
        bmm_5: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_50, arg53_1);  view_50 = arg53_1 = None
        view_51: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_4, _shape_param_58);  bmm_4 = _shape_param_58 = None
        view_52: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_5, _shape_param_59);  bmm_5 = _shape_param_59 = None
        convert_element_type_64: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_52, torch.float32);  view_52 = None
        convert_element_type_65: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg54_1, torch.float32);  arg54_1 = None
        mul_56: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
        mul_57: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_64, mul_56);  convert_element_type_64 = mul_56 = None
        convert_element_type_66: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.float32);  arg55_1 = None
        sub_3: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_66, arg56_1);  convert_element_type_66 = arg56_1 = None
        exp_5: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        div_17: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, arg57_1);  exp_5 = arg57_1 = None
        mul_58: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_57, div_17);  mul_57 = None
        sum_27: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_58, [-1], True)
        neg_7: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_1: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_7, sum_27, mul_58);  neg_7 = sum_27 = mul_58 = None
        convert_element_type_67: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        where_4: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_67);  convert_element_type_67 = None
        div_18: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_4, 8.0);  where_4 = None
        view_53: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_18, _shape_param_60);  div_18 = _shape_param_60 = None
        bmm_6: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg58_1, view_53);  arg58_1 = None
        bmm_7: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_53, arg59_1);  view_53 = arg59_1 = None
        view_54: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_61);  bmm_6 = _shape_param_61 = None
        view_55: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_62);  bmm_7 = _shape_param_62 = None
        permute_17: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 1, 3, 2]);  view_54 = None
        permute_18: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        clone_5: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_56: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_5, _shape_param_63);  clone_5 = _shape_param_63 = None
        view_57: "bf16[2048, 768]" = torch.ops.aten.view.default(view_56, _shape_param_64);  view_56 = _shape_param_64 = None
        mm_22: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_57, arg60_1);  arg60_1 = None
        permute_19: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_57, [1, 0])
        mm_23: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_19, arg61_1);  permute_19 = None
        sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_57, [0], True, dtype = torch.float32);  view_57 = None
        view_58: "f32[768]" = torch.ops.aten.view.default(sum_28, _shape_param_65);  sum_28 = _shape_param_65 = None
        convert_element_type_68: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_58, torch.bfloat16);  view_58 = None
        view_59: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_22, _shape_param_66);  mm_22 = _shape_param_66 = None
        convert_element_type_69: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        convert_element_type_70: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_71: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_68, torch.float32);  convert_element_type_68 = None
        permute_20: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_17, [0, 2, 1, 3]);  permute_17 = None
        view_60: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_20, _shape_param_67);  permute_20 = _shape_param_67 = None
        clone_6: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format);  view_60 = None
        view_61: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_6, _shape_param_68);  clone_6 = _shape_param_68 = None
        mm_24: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_61, arg62_1);  arg62_1 = None
        permute_21: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_61, [1, 0])
        mm_25: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_21, arg61_1);  permute_21 = None
        sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_61, [0], True, dtype = torch.float32);  view_61 = None
        view_62: "f32[768]" = torch.ops.aten.view.default(sum_29, _shape_param_69);  sum_29 = _shape_param_69 = None
        convert_element_type_72: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_62, torch.bfloat16);  view_62 = None
        view_63: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_24, _shape_param_70);  mm_24 = _shape_param_70 = None
        convert_element_type_73: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        add_19: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_69, convert_element_type_73);  convert_element_type_69 = convert_element_type_73 = None
        convert_element_type_74: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_75: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_72, torch.float32);  convert_element_type_72 = None
        permute_22: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
        clone_7: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_64: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_7, _shape_param_71);  clone_7 = _shape_param_71 = None
        view_65: "bf16[2048, 768]" = torch.ops.aten.view.default(view_64, _shape_param_72);  view_64 = _shape_param_72 = None
        mm_26: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_65, arg63_1);  arg63_1 = None
        permute_23: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_65, [1, 0])
        mm_27: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_23, arg61_1);  permute_23 = arg61_1 = None
        sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_65, [0], True, dtype = torch.float32);  view_65 = None
        view_66: "f32[768]" = torch.ops.aten.view.default(sum_30, _shape_param_73);  sum_30 = _shape_param_73 = None
        convert_element_type_76: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_66, torch.bfloat16);  view_66 = None
        view_67: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_26, _shape_param_74);  mm_26 = _shape_param_74 = None
        convert_element_type_77: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_67, torch.float32);  view_67 = None
        add_20: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_19, convert_element_type_77);  add_19 = convert_element_type_77 = None
        convert_element_type_78: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_79: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_76, torch.float32);  convert_element_type_76 = None
        sum_31: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_20, [0, 1], True, dtype = torch.float32)
        view_68: "f32[768]" = torch.ops.aten.view.default(sum_31, _shape_param_75);  sum_31 = _shape_param_75 = None
        mul_59: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg64_1, arg65_1)
        add_21: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg66_1, 1e-06)
        div_19: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_59, add_21);  mul_59 = None
        div_20: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_19, add_21);  div_19 = None
        neg_8: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_20)
        mul_60: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_8, div_20);  neg_8 = div_20 = None
        div_21: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_20, add_21);  add_20 = add_21 = None
        sum_32: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_60, [2], True, dtype = torch.float32);  mul_60 = None
        mul_61: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_21, arg64_1);  arg64_1 = None
        mul_62: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_21, arg65_1);  div_21 = None
        sum_33: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_62, [0, 1], True, dtype = torch.float32);  mul_62 = None
        view_69: "f32[768]" = torch.ops.aten.view.default(sum_33, _shape_param_76);  sum_33 = _shape_param_76 = None
        neg_9: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_61)
        sum_34: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_9, [2], True, dtype = torch.float32);  neg_9 = None
        add_22: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_18, mul_61);  add_18 = mul_61 = None
        mul_63: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg66_1, 2)
        div_22: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_32, mul_63);  sum_32 = mul_63 = None
        eq_4: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg66_1, 0);  arg66_1 = None
        where_5: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_4, full_1, div_22);  eq_4 = div_22 = None
        mul_64: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_5, 0.002607561929595828);  where_5 = None
        mul_65: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_64, arg65_1);  mul_64 = arg65_1 = None
        add_23: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_22, mul_65);  add_22 = mul_65 = None
        expand_3: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_34, _shape_param_77);  sum_34 = _shape_param_77 = None
        div_23: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_3, 768);  expand_3 = None
        add_24: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_23, div_23);  add_23 = div_23 = None
        convert_element_type_80: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.float32);  arg67_1 = None
        mul_66: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
        mul_67: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_24, mul_66);  add_24 = mul_66 = None
        convert_element_type_81: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_67, torch.bfloat16)
        convert_element_type_82: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        mul_68: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 1.1111111111111112);  convert_element_type_82 = None
        mul_69: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_81, mul_68);  convert_element_type_81 = mul_68 = None
        view_70: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_69, _shape_param_78);  mul_69 = _shape_param_78 = None
        mm_28: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_70, arg69_1);  arg69_1 = None
        permute_24: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_70, [1, 0])
        mm_29: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_24, arg70_1);  permute_24 = arg70_1 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_70, [0], True, dtype = torch.float32);  view_70 = None
        view_71: "f32[768]" = torch.ops.aten.view.default(sum_35, _shape_param_79);  sum_35 = _shape_param_79 = None
        convert_element_type_83: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_71, torch.bfloat16);  view_71 = None
        view_72: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_28, _shape_param_80);  mm_28 = _shape_param_80 = None
        convert_element_type_84: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_85: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_83, torch.float32);  convert_element_type_83 = None
        convert_element_type_86: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg71_1, torch.bfloat16);  arg71_1 = None
        mul_70: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 1.1111111111111112);  convert_element_type_86 = None
        mul_71: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_72, mul_70);  view_72 = mul_70 = None
        convert_element_type_87: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_71, torch.float32);  mul_71 = None
        view_73: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg72_1, _shape_param_81);  arg72_1 = _shape_param_81 = None
        convert_element_type_88: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        mul_72: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 0.7071067811865476)
        erf_2: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
        add_25: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_73: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_25, 0.5);  add_25 = None
        mul_74: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_88, convert_element_type_88)
        mul_75: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_74, -0.5);  mul_74 = None
        exp_6: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_75);  mul_75 = None
        mul_76: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_77: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_88, mul_76);  convert_element_type_88 = mul_76 = None
        add_26: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_73, mul_77);  mul_73 = mul_77 = None
        mul_78: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_87, add_26);  convert_element_type_87 = add_26 = None
        convert_element_type_89: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None
        view_74: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_89, _shape_param_82);  convert_element_type_89 = _shape_param_82 = None
        mm_30: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_74, arg73_1);  arg73_1 = None
        permute_25: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_74, [1, 0])
        mm_31: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_25, arg74_1);  permute_25 = arg74_1 = None
        sum_36: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_74, [0], True, dtype = torch.float32);  view_74 = None
        view_75: "f32[3072]" = torch.ops.aten.view.default(sum_36, _shape_param_83);  sum_36 = _shape_param_83 = None
        convert_element_type_90: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_75, torch.bfloat16);  view_75 = None
        view_76: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_30, _shape_param_84);  mm_30 = _shape_param_84 = None
        convert_element_type_91: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_76, torch.float32);  view_76 = None
        convert_element_type_92: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_93: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_90, torch.float32);  convert_element_type_90 = None
        sum_37: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_91, [0, 1], True, dtype = torch.float32)
        view_77: "f32[768]" = torch.ops.aten.view.default(sum_37, _shape_param_85);  sum_37 = _shape_param_85 = None
        mul_79: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg75_1, arg76_1)
        add_27: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg77_1, 1e-06)
        div_24: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_79, add_27);  mul_79 = None
        div_25: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_24, add_27);  div_24 = None
        neg_10: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_91)
        mul_80: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_10, div_25);  neg_10 = div_25 = None
        div_26: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_91, add_27);  convert_element_type_91 = add_27 = None
        sum_38: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_80, [2], True, dtype = torch.float32);  mul_80 = None
        mul_81: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_26, arg75_1);  arg75_1 = None
        mul_82: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_26, arg76_1);  div_26 = None
        sum_39: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_82, [0, 1], True, dtype = torch.float32);  mul_82 = None
        view_78: "f32[768]" = torch.ops.aten.view.default(sum_39, _shape_param_86);  sum_39 = _shape_param_86 = None
        neg_11: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_81)
        sum_40: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_11, [2], True, dtype = torch.float32);  neg_11 = None
        add_28: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_67, mul_81);  mul_67 = mul_81 = None
        mul_83: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg77_1, 2)
        div_27: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_38, mul_83);  sum_38 = mul_83 = None
        eq_5: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg77_1, 0);  arg77_1 = None
        where_6: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_5, full_1, div_27);  eq_5 = div_27 = None
        mul_84: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_6, 0.002607561929595828);  where_6 = None
        mul_85: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_84, arg76_1);  mul_84 = arg76_1 = None
        add_29: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_28, mul_85);  add_28 = mul_85 = None
        expand_4: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_40, _shape_param_87);  sum_40 = _shape_param_87 = None
        div_28: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_4, 768);  expand_4 = None
        add_30: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_29, div_28);  add_29 = div_28 = None
        convert_element_type_94: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16)
        convert_element_type_95: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.bfloat16);  arg78_1 = None
        mul_86: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_95, 1.1111111111111112);  convert_element_type_95 = None
        mul_87: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_94, mul_86);  convert_element_type_94 = mul_86 = None
        view_79: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_87, _shape_param_88);  mul_87 = _shape_param_88 = None
        mm_32: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_79, arg79_1);  arg79_1 = None
        permute_26: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_79, [1, 0])
        mm_33: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_26, arg80_1);  permute_26 = arg80_1 = None
        sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_79, [0], True, dtype = torch.float32);  view_79 = None
        view_80: "f32[768]" = torch.ops.aten.view.default(sum_41, _shape_param_89);  sum_41 = _shape_param_89 = None
        convert_element_type_96: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_80, torch.bfloat16);  view_80 = None
        view_81: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_32, _shape_param_90);  mm_32 = _shape_param_90 = None
        convert_element_type_97: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_98: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_96, torch.float32);  convert_element_type_96 = None
        view_82: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_81, _shape_param_91);  view_81 = _shape_param_91 = None
        permute_27: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_8: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_83: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_92);  clone_8 = _shape_param_92 = None
        bmm_8: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg81_1, view_83);  arg81_1 = None
        bmm_9: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_83, arg82_1);  view_83 = arg82_1 = None
        view_84: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_93);  bmm_8 = _shape_param_93 = None
        view_85: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_9, _shape_param_94);  bmm_9 = _shape_param_94 = None
        convert_element_type_99: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        convert_element_type_100: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
        mul_88: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 1.1111111111111112);  convert_element_type_100 = None
        mul_89: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_99, mul_88);  convert_element_type_99 = mul_88 = None
        convert_element_type_101: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg84_1, torch.float32);  arg84_1 = None
        sub_4: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_101, arg85_1);  convert_element_type_101 = arg85_1 = None
        exp_7: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div_29: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, arg86_1);  exp_7 = arg86_1 = None
        mul_90: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_89, div_29);  mul_89 = None
        sum_42: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_90, [-1], True)
        neg_12: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_29);  div_29 = None
        fma_2: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_12, sum_42, mul_90);  neg_12 = sum_42 = mul_90 = None
        convert_element_type_102: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        where_7: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_102);  convert_element_type_102 = None
        div_30: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_7, 8.0);  where_7 = None
        view_86: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_30, _shape_param_95);  div_30 = _shape_param_95 = None
        bmm_10: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg87_1, view_86);  arg87_1 = None
        bmm_11: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_86, arg88_1);  view_86 = arg88_1 = None
        view_87: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_96);  bmm_10 = _shape_param_96 = None
        view_88: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_97);  bmm_11 = _shape_param_97 = None
        permute_28: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_87, [0, 1, 3, 2]);  view_87 = None
        permute_29: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
        clone_9: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_89: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_9, _shape_param_98);  clone_9 = _shape_param_98 = None
        view_90: "bf16[2048, 768]" = torch.ops.aten.view.default(view_89, _shape_param_99);  view_89 = _shape_param_99 = None
        mm_34: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_90, arg89_1);  arg89_1 = None
        permute_30: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_90, [1, 0])
        mm_35: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_30, arg90_1);  permute_30 = None
        sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_90, [0], True, dtype = torch.float32);  view_90 = None
        view_91: "f32[768]" = torch.ops.aten.view.default(sum_43, _shape_param_100);  sum_43 = _shape_param_100 = None
        convert_element_type_103: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_91, torch.bfloat16);  view_91 = None
        view_92: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_34, _shape_param_101);  mm_34 = _shape_param_101 = None
        convert_element_type_104: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_92, torch.float32);  view_92 = None
        convert_element_type_105: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_106: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_103, torch.float32);  convert_element_type_103 = None
        permute_31: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_93: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_31, _shape_param_102);  permute_31 = _shape_param_102 = None
        clone_10: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_93, memory_format = torch.contiguous_format);  view_93 = None
        view_94: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_10, _shape_param_103);  clone_10 = _shape_param_103 = None
        mm_36: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_94, arg91_1);  arg91_1 = None
        permute_32: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_94, [1, 0])
        mm_37: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_32, arg90_1);  permute_32 = None
        sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_94, [0], True, dtype = torch.float32);  view_94 = None
        view_95: "f32[768]" = torch.ops.aten.view.default(sum_44, _shape_param_104);  sum_44 = _shape_param_104 = None
        convert_element_type_107: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_95, torch.bfloat16);  view_95 = None
        view_96: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_36, _shape_param_105);  mm_36 = _shape_param_105 = None
        convert_element_type_108: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_96, torch.float32);  view_96 = None
        add_31: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_104, convert_element_type_108);  convert_element_type_104 = convert_element_type_108 = None
        convert_element_type_109: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_110: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_107, torch.float32);  convert_element_type_107 = None
        permute_33: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        clone_11: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_97: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_11, _shape_param_106);  clone_11 = _shape_param_106 = None
        view_98: "bf16[2048, 768]" = torch.ops.aten.view.default(view_97, _shape_param_107);  view_97 = _shape_param_107 = None
        mm_38: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_98, arg92_1);  arg92_1 = None
        permute_34: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_98, [1, 0])
        mm_39: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_34, arg90_1);  permute_34 = arg90_1 = None
        sum_45: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True, dtype = torch.float32);  view_98 = None
        view_99: "f32[768]" = torch.ops.aten.view.default(sum_45, _shape_param_108);  sum_45 = _shape_param_108 = None
        convert_element_type_111: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_99, torch.bfloat16);  view_99 = None
        view_100: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_38, _shape_param_109);  mm_38 = _shape_param_109 = None
        convert_element_type_112: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_100, torch.float32);  view_100 = None
        add_32: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_31, convert_element_type_112);  add_31 = convert_element_type_112 = None
        convert_element_type_113: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_114: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_111, torch.float32);  convert_element_type_111 = None
        sum_46: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_32, [0, 1], True, dtype = torch.float32)
        view_101: "f32[768]" = torch.ops.aten.view.default(sum_46, _shape_param_110);  sum_46 = _shape_param_110 = None
        mul_91: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg93_1, arg94_1)
        add_33: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg95_1, 1e-06)
        div_31: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_91, add_33);  mul_91 = None
        div_32: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_31, add_33);  div_31 = None
        neg_13: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_32)
        mul_92: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_13, div_32);  neg_13 = div_32 = None
        div_33: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_32, add_33);  add_32 = add_33 = None
        sum_47: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_92, [2], True, dtype = torch.float32);  mul_92 = None
        mul_93: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_33, arg93_1);  arg93_1 = None
        mul_94: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_33, arg94_1);  div_33 = None
        sum_48: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_94, [0, 1], True, dtype = torch.float32);  mul_94 = None
        view_102: "f32[768]" = torch.ops.aten.view.default(sum_48, _shape_param_111);  sum_48 = _shape_param_111 = None
        neg_14: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_93)
        sum_49: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_14, [2], True, dtype = torch.float32);  neg_14 = None
        add_34: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_30, mul_93);  add_30 = mul_93 = None
        mul_95: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg95_1, 2)
        div_34: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_47, mul_95);  sum_47 = mul_95 = None
        eq_6: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg95_1, 0);  arg95_1 = None
        where_8: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_6, full_1, div_34);  eq_6 = div_34 = None
        mul_96: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_8, 0.002607561929595828);  where_8 = None
        mul_97: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_96, arg94_1);  mul_96 = arg94_1 = None
        add_35: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_34, mul_97);  add_34 = mul_97 = None
        expand_5: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_49, _shape_param_112);  sum_49 = _shape_param_112 = None
        div_35: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_5, 768);  expand_5 = None
        add_36: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_35, div_35);  add_35 = div_35 = None
        convert_element_type_115: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg96_1, torch.float32);  arg96_1 = None
        mul_98: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_115, 1.1111111111111112);  convert_element_type_115 = None
        mul_99: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_36, mul_98);  add_36 = mul_98 = None
        convert_element_type_116: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16)
        convert_element_type_117: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg97_1, torch.bfloat16);  arg97_1 = None
        mul_100: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_117, 1.1111111111111112);  convert_element_type_117 = None
        mul_101: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_116, mul_100);  convert_element_type_116 = mul_100 = None
        view_103: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_101, _shape_param_113);  mul_101 = _shape_param_113 = None
        mm_40: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_103, arg98_1);  arg98_1 = None
        permute_35: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_103, [1, 0])
        mm_41: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_35, arg99_1);  permute_35 = arg99_1 = None
        sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_103, [0], True, dtype = torch.float32);  view_103 = None
        view_104: "f32[768]" = torch.ops.aten.view.default(sum_50, _shape_param_114);  sum_50 = _shape_param_114 = None
        convert_element_type_118: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_104, torch.bfloat16);  view_104 = None
        view_105: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_40, _shape_param_115);  mm_40 = _shape_param_115 = None
        convert_element_type_119: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_120: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_118, torch.float32);  convert_element_type_118 = None
        convert_element_type_121: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg100_1, torch.bfloat16);  arg100_1 = None
        mul_102: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_121, 1.1111111111111112);  convert_element_type_121 = None
        mul_103: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_105, mul_102);  view_105 = mul_102 = None
        convert_element_type_122: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_103, torch.float32);  mul_103 = None
        view_106: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg101_1, _shape_param_116);  arg101_1 = _shape_param_116 = None
        convert_element_type_123: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_106, torch.float32);  view_106 = None
        mul_104: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 0.7071067811865476)
        erf_3: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_104);  mul_104 = None
        add_37: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_105: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_37, 0.5);  add_37 = None
        mul_106: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, convert_element_type_123)
        mul_107: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_106, -0.5);  mul_106 = None
        exp_8: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_107);  mul_107 = None
        mul_108: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_109: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_123, mul_108);  convert_element_type_123 = mul_108 = None
        add_38: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_105, mul_109);  mul_105 = mul_109 = None
        mul_110: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_122, add_38);  convert_element_type_122 = add_38 = None
        convert_element_type_124: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        view_107: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_124, _shape_param_117);  convert_element_type_124 = _shape_param_117 = None
        mm_42: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_107, arg102_1);  arg102_1 = None
        permute_36: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_107, [1, 0])
        mm_43: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_36, arg103_1);  permute_36 = arg103_1 = None
        sum_51: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_107, [0], True, dtype = torch.float32);  view_107 = None
        view_108: "f32[3072]" = torch.ops.aten.view.default(sum_51, _shape_param_118);  sum_51 = _shape_param_118 = None
        convert_element_type_125: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_108, torch.bfloat16);  view_108 = None
        view_109: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_42, _shape_param_119);  mm_42 = _shape_param_119 = None
        convert_element_type_126: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        convert_element_type_127: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_128: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_125, torch.float32);  convert_element_type_125 = None
        sum_52: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_126, [0, 1], True, dtype = torch.float32)
        view_110: "f32[768]" = torch.ops.aten.view.default(sum_52, _shape_param_120);  sum_52 = _shape_param_120 = None
        mul_111: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg104_1, arg105_1)
        add_39: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg106_1, 1e-06)
        div_36: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_111, add_39);  mul_111 = None
        div_37: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_36, add_39);  div_36 = None
        neg_15: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_126)
        mul_112: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_15, div_37);  neg_15 = div_37 = None
        div_38: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_126, add_39);  convert_element_type_126 = add_39 = None
        sum_53: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_112, [2], True, dtype = torch.float32);  mul_112 = None
        mul_113: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_38, arg104_1);  arg104_1 = None
        mul_114: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_38, arg105_1);  div_38 = None
        sum_54: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_114, [0, 1], True, dtype = torch.float32);  mul_114 = None
        view_111: "f32[768]" = torch.ops.aten.view.default(sum_54, _shape_param_121);  sum_54 = _shape_param_121 = None
        neg_16: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_113)
        sum_55: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_16, [2], True, dtype = torch.float32);  neg_16 = None
        add_40: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_99, mul_113);  mul_99 = mul_113 = None
        mul_115: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg106_1, 2)
        div_39: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_53, mul_115);  sum_53 = mul_115 = None
        eq_7: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg106_1, 0);  arg106_1 = None
        where_9: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_7, full_1, div_39);  eq_7 = div_39 = None
        mul_116: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_9, 0.002607561929595828);  where_9 = None
        mul_117: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_116, arg105_1);  mul_116 = arg105_1 = None
        add_41: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_40, mul_117);  add_40 = mul_117 = None
        expand_6: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_55, _shape_param_122);  sum_55 = _shape_param_122 = None
        div_40: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_6, 768);  expand_6 = None
        add_42: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_41, div_40);  add_41 = div_40 = None
        convert_element_type_129: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_42, torch.bfloat16)
        convert_element_type_130: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        mul_118: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_130, 1.1111111111111112);  convert_element_type_130 = None
        mul_119: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_129, mul_118);  convert_element_type_129 = mul_118 = None
        view_112: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_119, _shape_param_123);  mul_119 = _shape_param_123 = None
        mm_44: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_112, arg108_1);  arg108_1 = None
        permute_37: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_112, [1, 0])
        mm_45: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_37, arg109_1);  permute_37 = arg109_1 = None
        sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_112, [0], True, dtype = torch.float32);  view_112 = None
        view_113: "f32[768]" = torch.ops.aten.view.default(sum_56, _shape_param_124);  sum_56 = _shape_param_124 = None
        convert_element_type_131: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_113, torch.bfloat16);  view_113 = None
        view_114: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_44, _shape_param_125);  mm_44 = _shape_param_125 = None
        convert_element_type_132: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_133: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_131, torch.float32);  convert_element_type_131 = None
        view_115: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_114, _shape_param_126);  view_114 = _shape_param_126 = None
        permute_38: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        clone_12: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_116: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_127);  clone_12 = _shape_param_127 = None
        bmm_12: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg110_1, view_116);  arg110_1 = None
        bmm_13: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_116, arg111_1);  view_116 = arg111_1 = None
        view_117: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_12, _shape_param_128);  bmm_12 = _shape_param_128 = None
        view_118: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_129);  bmm_13 = _shape_param_129 = None
        convert_element_type_134: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        convert_element_type_135: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.float32);  arg112_1 = None
        mul_120: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_135, 1.1111111111111112);  convert_element_type_135 = None
        mul_121: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_134, mul_120);  convert_element_type_134 = mul_120 = None
        convert_element_type_136: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.float32);  arg113_1 = None
        sub_5: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_136, arg114_1);  convert_element_type_136 = arg114_1 = None
        exp_9: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        div_41: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, arg115_1);  exp_9 = arg115_1 = None
        mul_122: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_121, div_41);  mul_121 = None
        sum_57: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [-1], True)
        neg_17: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_41);  div_41 = None
        fma_3: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_17, sum_57, mul_122);  neg_17 = sum_57 = mul_122 = None
        convert_element_type_137: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        where_10: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_137);  convert_element_type_137 = None
        div_42: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_10, 8.0);  where_10 = None
        view_119: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_42, _shape_param_130);  div_42 = _shape_param_130 = None
        bmm_14: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg116_1, view_119);  arg116_1 = None
        bmm_15: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_119, arg117_1);  view_119 = arg117_1 = None
        view_120: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_131);  bmm_14 = _shape_param_131 = None
        view_121: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_132);  bmm_15 = _shape_param_132 = None
        permute_39: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 1, 3, 2]);  view_120 = None
        permute_40: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_117, [0, 2, 1, 3]);  view_117 = None
        clone_13: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_122: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_13, _shape_param_133);  clone_13 = _shape_param_133 = None
        view_123: "bf16[2048, 768]" = torch.ops.aten.view.default(view_122, _shape_param_134);  view_122 = _shape_param_134 = None
        mm_46: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_123, arg118_1);  arg118_1 = None
        permute_41: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_123, [1, 0])
        mm_47: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_41, arg119_1);  permute_41 = None
        sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True, dtype = torch.float32);  view_123 = None
        view_124: "f32[768]" = torch.ops.aten.view.default(sum_58, _shape_param_135);  sum_58 = _shape_param_135 = None
        convert_element_type_138: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_124, torch.bfloat16);  view_124 = None
        view_125: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_46, _shape_param_136);  mm_46 = _shape_param_136 = None
        convert_element_type_139: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        convert_element_type_140: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_141: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_138, torch.float32);  convert_element_type_138 = None
        permute_42: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_39, [0, 2, 1, 3]);  permute_39 = None
        view_126: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_42, _shape_param_137);  permute_42 = _shape_param_137 = None
        clone_14: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_126, memory_format = torch.contiguous_format);  view_126 = None
        view_127: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_14, _shape_param_138);  clone_14 = _shape_param_138 = None
        mm_48: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_127, arg120_1);  arg120_1 = None
        permute_43: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_127, [1, 0])
        mm_49: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_43, arg119_1);  permute_43 = None
        sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_127, [0], True, dtype = torch.float32);  view_127 = None
        view_128: "f32[768]" = torch.ops.aten.view.default(sum_59, _shape_param_139);  sum_59 = _shape_param_139 = None
        convert_element_type_142: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_128, torch.bfloat16);  view_128 = None
        view_129: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_48, _shape_param_140);  mm_48 = _shape_param_140 = None
        convert_element_type_143: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        add_43: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_139, convert_element_type_143);  convert_element_type_139 = convert_element_type_143 = None
        convert_element_type_144: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_145: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_142, torch.float32);  convert_element_type_142 = None
        permute_44: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
        clone_15: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_130: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_15, _shape_param_141);  clone_15 = _shape_param_141 = None
        view_131: "bf16[2048, 768]" = torch.ops.aten.view.default(view_130, _shape_param_142);  view_130 = _shape_param_142 = None
        mm_50: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_131, arg121_1);  arg121_1 = None
        permute_45: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_131, [1, 0])
        mm_51: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_45, arg119_1);  permute_45 = arg119_1 = None
        sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True, dtype = torch.float32);  view_131 = None
        view_132: "f32[768]" = torch.ops.aten.view.default(sum_60, _shape_param_143);  sum_60 = _shape_param_143 = None
        convert_element_type_146: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_132, torch.bfloat16);  view_132 = None
        view_133: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_50, _shape_param_144);  mm_50 = _shape_param_144 = None
        convert_element_type_147: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        add_44: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_43, convert_element_type_147);  add_43 = convert_element_type_147 = None
        convert_element_type_148: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_149: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_146, torch.float32);  convert_element_type_146 = None
        sum_61: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_44, [0, 1], True, dtype = torch.float32)
        view_134: "f32[768]" = torch.ops.aten.view.default(sum_61, _shape_param_145);  sum_61 = _shape_param_145 = None
        mul_123: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg122_1, arg123_1)
        add_45: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg124_1, 1e-06)
        div_43: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_123, add_45);  mul_123 = None
        div_44: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_43, add_45);  div_43 = None
        neg_18: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_44)
        mul_124: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_18, div_44);  neg_18 = div_44 = None
        div_45: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_44, add_45);  add_44 = add_45 = None
        sum_62: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True, dtype = torch.float32);  mul_124 = None
        mul_125: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_45, arg122_1);  arg122_1 = None
        mul_126: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_45, arg123_1);  div_45 = None
        sum_63: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1], True, dtype = torch.float32);  mul_126 = None
        view_135: "f32[768]" = torch.ops.aten.view.default(sum_63, _shape_param_146);  sum_63 = _shape_param_146 = None
        neg_19: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_125)
        sum_64: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_19, [2], True, dtype = torch.float32);  neg_19 = None
        add_46: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_42, mul_125);  add_42 = mul_125 = None
        mul_127: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg124_1, 2)
        div_46: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_62, mul_127);  sum_62 = mul_127 = None
        eq_8: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg124_1, 0);  arg124_1 = None
        where_11: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_8, full_1, div_46);  eq_8 = div_46 = None
        mul_128: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_11, 0.002607561929595828);  where_11 = None
        mul_129: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_128, arg123_1);  mul_128 = arg123_1 = None
        add_47: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_46, mul_129);  add_46 = mul_129 = None
        expand_7: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_64, _shape_param_147);  sum_64 = _shape_param_147 = None
        div_47: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_7, 768);  expand_7 = None
        add_48: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_47, div_47);  add_47 = div_47 = None
        convert_element_type_150: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.float32);  arg125_1 = None
        mul_130: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_150, 1.1111111111111112);  convert_element_type_150 = None
        mul_131: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_48, mul_130);  add_48 = mul_130 = None
        convert_element_type_151: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_131, torch.bfloat16)
        convert_element_type_152: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.bfloat16);  arg126_1 = None
        mul_132: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_152, 1.1111111111111112);  convert_element_type_152 = None
        mul_133: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_151, mul_132);  convert_element_type_151 = mul_132 = None
        view_136: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_133, _shape_param_148);  mul_133 = _shape_param_148 = None
        mm_52: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_136, arg127_1);  arg127_1 = None
        permute_46: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_136, [1, 0])
        mm_53: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_46, arg128_1);  permute_46 = arg128_1 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_136, [0], True, dtype = torch.float32);  view_136 = None
        view_137: "f32[768]" = torch.ops.aten.view.default(sum_65, _shape_param_149);  sum_65 = _shape_param_149 = None
        convert_element_type_153: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_137, torch.bfloat16);  view_137 = None
        view_138: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_52, _shape_param_150);  mm_52 = _shape_param_150 = None
        convert_element_type_154: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_155: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_153, torch.float32);  convert_element_type_153 = None
        convert_element_type_156: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg129_1, torch.bfloat16);  arg129_1 = None
        mul_134: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_156, 1.1111111111111112);  convert_element_type_156 = None
        mul_135: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_138, mul_134);  view_138 = mul_134 = None
        convert_element_type_157: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_135, torch.float32);  mul_135 = None
        view_139: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg130_1, _shape_param_151);  arg130_1 = _shape_param_151 = None
        convert_element_type_158: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        mul_136: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_158, 0.7071067811865476)
        erf_4: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_136);  mul_136 = None
        add_49: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_137: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
        mul_138: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_158, convert_element_type_158)
        mul_139: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_138, -0.5);  mul_138 = None
        exp_10: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_139);  mul_139 = None
        mul_140: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_141: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_158, mul_140);  convert_element_type_158 = mul_140 = None
        add_50: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_137, mul_141);  mul_137 = mul_141 = None
        mul_142: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_157, add_50);  convert_element_type_157 = add_50 = None
        convert_element_type_159: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        view_140: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_159, _shape_param_152);  convert_element_type_159 = _shape_param_152 = None
        mm_54: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_140, arg131_1);  arg131_1 = None
        permute_47: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_140, [1, 0])
        mm_55: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_47, arg132_1);  permute_47 = arg132_1 = None
        sum_66: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True, dtype = torch.float32);  view_140 = None
        view_141: "f32[3072]" = torch.ops.aten.view.default(sum_66, _shape_param_153);  sum_66 = _shape_param_153 = None
        convert_element_type_160: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_141, torch.bfloat16);  view_141 = None
        view_142: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_54, _shape_param_154);  mm_54 = _shape_param_154 = None
        convert_element_type_161: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_142, torch.float32);  view_142 = None
        convert_element_type_162: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_163: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_160, torch.float32);  convert_element_type_160 = None
        sum_67: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_161, [0, 1], True, dtype = torch.float32)
        view_143: "f32[768]" = torch.ops.aten.view.default(sum_67, _shape_param_155);  sum_67 = _shape_param_155 = None
        mul_143: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg133_1, arg134_1)
        add_51: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg135_1, 1e-06)
        div_48: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_143, add_51);  mul_143 = None
        div_49: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_48, add_51);  div_48 = None
        neg_20: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_161)
        mul_144: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_20, div_49);  neg_20 = div_49 = None
        div_50: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_161, add_51);  convert_element_type_161 = add_51 = None
        sum_68: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [2], True, dtype = torch.float32);  mul_144 = None
        mul_145: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_50, arg133_1);  arg133_1 = None
        mul_146: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_50, arg134_1);  div_50 = None
        sum_69: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_146, [0, 1], True, dtype = torch.float32);  mul_146 = None
        view_144: "f32[768]" = torch.ops.aten.view.default(sum_69, _shape_param_156);  sum_69 = _shape_param_156 = None
        neg_21: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_145)
        sum_70: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_21, [2], True, dtype = torch.float32);  neg_21 = None
        add_52: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_131, mul_145);  mul_131 = mul_145 = None
        mul_147: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg135_1, 2)
        div_51: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_68, mul_147);  sum_68 = mul_147 = None
        eq_9: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg135_1, 0);  arg135_1 = None
        where_12: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_9, full_1, div_51);  eq_9 = div_51 = None
        mul_148: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_12, 0.002607561929595828);  where_12 = None
        mul_149: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_148, arg134_1);  mul_148 = arg134_1 = None
        add_53: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_52, mul_149);  add_52 = mul_149 = None
        expand_8: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_70, _shape_param_157);  sum_70 = _shape_param_157 = None
        div_52: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_8, 768);  expand_8 = None
        add_54: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_53, div_52);  add_53 = div_52 = None
        convert_element_type_164: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16)
        convert_element_type_165: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg136_1, torch.bfloat16);  arg136_1 = None
        mul_150: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_165, 1.1111111111111112);  convert_element_type_165 = None
        mul_151: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_164, mul_150);  convert_element_type_164 = mul_150 = None
        view_145: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_151, _shape_param_158);  mul_151 = _shape_param_158 = None
        mm_56: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_145, arg137_1);  arg137_1 = None
        permute_48: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_145, [1, 0])
        mm_57: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_48, arg138_1);  permute_48 = arg138_1 = None
        sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_145, [0], True, dtype = torch.float32);  view_145 = None
        view_146: "f32[768]" = torch.ops.aten.view.default(sum_71, _shape_param_159);  sum_71 = _shape_param_159 = None
        convert_element_type_166: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_146, torch.bfloat16);  view_146 = None
        view_147: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_56, _shape_param_160);  mm_56 = _shape_param_160 = None
        convert_element_type_167: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_168: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_166, torch.float32);  convert_element_type_166 = None
        view_148: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_147, _shape_param_161);  view_147 = _shape_param_161 = None
        permute_49: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_16: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_149: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_162);  clone_16 = _shape_param_162 = None
        bmm_16: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg139_1, view_149);  arg139_1 = None
        bmm_17: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_149, arg140_1);  view_149 = arg140_1 = None
        view_150: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_16, _shape_param_163);  bmm_16 = _shape_param_163 = None
        view_151: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_17, _shape_param_164);  bmm_17 = _shape_param_164 = None
        convert_element_type_169: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        convert_element_type_170: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float32);  arg141_1 = None
        mul_152: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_170, 1.1111111111111112);  convert_element_type_170 = None
        mul_153: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_169, mul_152);  convert_element_type_169 = mul_152 = None
        convert_element_type_171: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.float32);  arg142_1 = None
        sub_6: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_171, arg143_1);  convert_element_type_171 = arg143_1 = None
        exp_11: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        div_53: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, arg144_1);  exp_11 = arg144_1 = None
        mul_154: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_153, div_53);  mul_153 = None
        sum_72: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [-1], True)
        neg_22: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_53);  div_53 = None
        fma_4: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_22, sum_72, mul_154);  neg_22 = sum_72 = mul_154 = None
        convert_element_type_172: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        where_13: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_172);  convert_element_type_172 = None
        div_54: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_13, 8.0);  where_13 = None
        view_152: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_54, _shape_param_165);  div_54 = _shape_param_165 = None
        bmm_18: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg145_1, view_152);  arg145_1 = None
        bmm_19: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_152, arg146_1);  view_152 = arg146_1 = None
        view_153: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_166);  bmm_18 = _shape_param_166 = None
        view_154: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_167);  bmm_19 = _shape_param_167 = None
        permute_50: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_153, [0, 1, 3, 2]);  view_153 = None
        permute_51: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        clone_17: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_155: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_17, _shape_param_168);  clone_17 = _shape_param_168 = None
        view_156: "bf16[2048, 768]" = torch.ops.aten.view.default(view_155, _shape_param_169);  view_155 = _shape_param_169 = None
        mm_58: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_156, arg147_1);  arg147_1 = None
        permute_52: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_59: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_52, arg148_1);  permute_52 = None
        sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True, dtype = torch.float32);  view_156 = None
        view_157: "f32[768]" = torch.ops.aten.view.default(sum_73, _shape_param_170);  sum_73 = _shape_param_170 = None
        convert_element_type_173: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_157, torch.bfloat16);  view_157 = None
        view_158: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_58, _shape_param_171);  mm_58 = _shape_param_171 = None
        convert_element_type_174: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        convert_element_type_175: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_176: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_173, torch.float32);  convert_element_type_173 = None
        permute_53: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_50, [0, 2, 1, 3]);  permute_50 = None
        view_159: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_53, _shape_param_172);  permute_53 = _shape_param_172 = None
        clone_18: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_159, memory_format = torch.contiguous_format);  view_159 = None
        view_160: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_18, _shape_param_173);  clone_18 = _shape_param_173 = None
        mm_60: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_160, arg149_1);  arg149_1 = None
        permute_54: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_160, [1, 0])
        mm_61: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_54, arg148_1);  permute_54 = None
        sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_160, [0], True, dtype = torch.float32);  view_160 = None
        view_161: "f32[768]" = torch.ops.aten.view.default(sum_74, _shape_param_174);  sum_74 = _shape_param_174 = None
        convert_element_type_177: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_161, torch.bfloat16);  view_161 = None
        view_162: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_60, _shape_param_175);  mm_60 = _shape_param_175 = None
        convert_element_type_178: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32);  view_162 = None
        add_55: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_174, convert_element_type_178);  convert_element_type_174 = convert_element_type_178 = None
        convert_element_type_179: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_180: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_177, torch.float32);  convert_element_type_177 = None
        permute_55: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        clone_19: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_163: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_19, _shape_param_176);  clone_19 = _shape_param_176 = None
        view_164: "bf16[2048, 768]" = torch.ops.aten.view.default(view_163, _shape_param_177);  view_163 = _shape_param_177 = None
        mm_62: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_164, arg150_1);  arg150_1 = None
        permute_56: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_164, [1, 0])
        mm_63: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_56, arg148_1);  permute_56 = arg148_1 = None
        sum_75: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_164, [0], True, dtype = torch.float32);  view_164 = None
        view_165: "f32[768]" = torch.ops.aten.view.default(sum_75, _shape_param_178);  sum_75 = _shape_param_178 = None
        convert_element_type_181: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_165, torch.bfloat16);  view_165 = None
        view_166: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_62, _shape_param_179);  mm_62 = _shape_param_179 = None
        convert_element_type_182: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_166, torch.float32);  view_166 = None
        add_56: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_55, convert_element_type_182);  add_55 = convert_element_type_182 = None
        convert_element_type_183: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_184: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_181, torch.float32);  convert_element_type_181 = None
        sum_76: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_56, [0, 1], True, dtype = torch.float32)
        view_167: "f32[768]" = torch.ops.aten.view.default(sum_76, _shape_param_180);  sum_76 = _shape_param_180 = None
        mul_155: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg151_1, arg152_1)
        add_57: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg153_1, 1e-06)
        div_55: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_155, add_57);  mul_155 = None
        div_56: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_55, add_57);  div_55 = None
        neg_23: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_56)
        mul_156: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_23, div_56);  neg_23 = div_56 = None
        div_57: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_56, add_57);  add_56 = add_57 = None
        sum_77: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True, dtype = torch.float32);  mul_156 = None
        mul_157: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_57, arg151_1);  arg151_1 = None
        mul_158: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_57, arg152_1);  div_57 = None
        sum_78: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True, dtype = torch.float32);  mul_158 = None
        view_168: "f32[768]" = torch.ops.aten.view.default(sum_78, _shape_param_181);  sum_78 = _shape_param_181 = None
        neg_24: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_157)
        sum_79: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_24, [2], True, dtype = torch.float32);  neg_24 = None
        add_58: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_54, mul_157);  add_54 = mul_157 = None
        mul_159: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg153_1, 2)
        div_58: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_77, mul_159);  sum_77 = mul_159 = None
        eq_10: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg153_1, 0);  arg153_1 = None
        where_14: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_10, full_1, div_58);  eq_10 = div_58 = None
        mul_160: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_14, 0.002607561929595828);  where_14 = None
        mul_161: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_160, arg152_1);  mul_160 = arg152_1 = None
        add_59: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_58, mul_161);  add_58 = mul_161 = None
        expand_9: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_79, _shape_param_182);  sum_79 = _shape_param_182 = None
        div_59: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_9, 768);  expand_9 = None
        add_60: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_59, div_59);  add_59 = div_59 = None
        convert_element_type_185: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg154_1, torch.float32);  arg154_1 = None
        mul_162: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_185, 1.1111111111111112);  convert_element_type_185 = None
        mul_163: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_60, mul_162);  add_60 = mul_162 = None
        convert_element_type_186: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_163, torch.bfloat16)
        convert_element_type_187: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.bfloat16);  arg155_1 = None
        mul_164: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_187, 1.1111111111111112);  convert_element_type_187 = None
        mul_165: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_186, mul_164);  convert_element_type_186 = mul_164 = None
        view_169: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_165, _shape_param_183);  mul_165 = _shape_param_183 = None
        mm_64: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_169, arg156_1);  arg156_1 = None
        permute_57: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_169, [1, 0])
        mm_65: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_57, arg157_1);  permute_57 = arg157_1 = None
        sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_169, [0], True, dtype = torch.float32);  view_169 = None
        view_170: "f32[768]" = torch.ops.aten.view.default(sum_80, _shape_param_184);  sum_80 = _shape_param_184 = None
        convert_element_type_188: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_170, torch.bfloat16);  view_170 = None
        view_171: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_64, _shape_param_185);  mm_64 = _shape_param_185 = None
        convert_element_type_189: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_190: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_188, torch.float32);  convert_element_type_188 = None
        convert_element_type_191: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.bfloat16);  arg158_1 = None
        mul_166: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_191, 1.1111111111111112);  convert_element_type_191 = None
        mul_167: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_171, mul_166);  view_171 = mul_166 = None
        convert_element_type_192: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_167, torch.float32);  mul_167 = None
        view_172: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg159_1, _shape_param_186);  arg159_1 = _shape_param_186 = None
        convert_element_type_193: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_172, torch.float32);  view_172 = None
        mul_168: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_193, 0.7071067811865476)
        erf_5: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_61: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_169: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_61, 0.5);  add_61 = None
        mul_170: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_193, convert_element_type_193)
        mul_171: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_170, -0.5);  mul_170 = None
        exp_12: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_171);  mul_171 = None
        mul_172: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_173: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_193, mul_172);  convert_element_type_193 = mul_172 = None
        add_62: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_169, mul_173);  mul_169 = mul_173 = None
        mul_174: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_192, add_62);  convert_element_type_192 = add_62 = None
        convert_element_type_194: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None
        view_173: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_194, _shape_param_187);  convert_element_type_194 = _shape_param_187 = None
        mm_66: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_173, arg160_1);  arg160_1 = None
        permute_58: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_67: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_58, arg161_1);  permute_58 = arg161_1 = None
        sum_81: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True, dtype = torch.float32);  view_173 = None
        view_174: "f32[3072]" = torch.ops.aten.view.default(sum_81, _shape_param_188);  sum_81 = _shape_param_188 = None
        convert_element_type_195: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_174, torch.bfloat16);  view_174 = None
        view_175: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_66, _shape_param_189);  mm_66 = _shape_param_189 = None
        convert_element_type_196: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_175, torch.float32);  view_175 = None
        convert_element_type_197: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_198: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_195, torch.float32);  convert_element_type_195 = None
        sum_82: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_196, [0, 1], True, dtype = torch.float32)
        view_176: "f32[768]" = torch.ops.aten.view.default(sum_82, _shape_param_190);  sum_82 = _shape_param_190 = None
        mul_175: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg162_1, arg163_1)
        add_63: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg164_1, 1e-06)
        div_60: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_175, add_63);  mul_175 = None
        div_61: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_60, add_63);  div_60 = None
        neg_25: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_196)
        mul_176: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_25, div_61);  neg_25 = div_61 = None
        div_62: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_196, add_63);  convert_element_type_196 = add_63 = None
        sum_83: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True, dtype = torch.float32);  mul_176 = None
        mul_177: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_62, arg162_1);  arg162_1 = None
        mul_178: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_62, arg163_1);  div_62 = None
        sum_84: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_178, [0, 1], True, dtype = torch.float32);  mul_178 = None
        view_177: "f32[768]" = torch.ops.aten.view.default(sum_84, _shape_param_191);  sum_84 = _shape_param_191 = None
        neg_26: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_177)
        sum_85: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_26, [2], True, dtype = torch.float32);  neg_26 = None
        add_64: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_163, mul_177);  mul_163 = mul_177 = None
        mul_179: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg164_1, 2)
        div_63: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_83, mul_179);  sum_83 = mul_179 = None
        eq_11: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg164_1, 0);  arg164_1 = None
        where_15: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_11, full_1, div_63);  eq_11 = div_63 = None
        mul_180: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_15, 0.002607561929595828);  where_15 = None
        mul_181: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_180, arg163_1);  mul_180 = arg163_1 = None
        add_65: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_64, mul_181);  add_64 = mul_181 = None
        expand_10: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_85, _shape_param_192);  sum_85 = _shape_param_192 = None
        div_64: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_10, 768);  expand_10 = None
        add_66: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_65, div_64);  add_65 = div_64 = None
        convert_element_type_199: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16)
        convert_element_type_200: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.bfloat16);  arg165_1 = None
        mul_182: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_200, 1.1111111111111112);  convert_element_type_200 = None
        mul_183: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_199, mul_182);  convert_element_type_199 = mul_182 = None
        view_178: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_183, _shape_param_193);  mul_183 = _shape_param_193 = None
        mm_68: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_178, arg166_1);  arg166_1 = None
        permute_59: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_178, [1, 0])
        mm_69: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_59, arg167_1);  permute_59 = arg167_1 = None
        sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_178, [0], True, dtype = torch.float32);  view_178 = None
        view_179: "f32[768]" = torch.ops.aten.view.default(sum_86, _shape_param_194);  sum_86 = _shape_param_194 = None
        convert_element_type_201: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_179, torch.bfloat16);  view_179 = None
        view_180: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_68, _shape_param_195);  mm_68 = _shape_param_195 = None
        convert_element_type_202: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_203: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_201, torch.float32);  convert_element_type_201 = None
        view_181: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_180, _shape_param_196);  view_180 = _shape_param_196 = None
        permute_60: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        clone_20: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_182: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_197);  clone_20 = _shape_param_197 = None
        bmm_20: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg168_1, view_182);  arg168_1 = None
        bmm_21: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_182, arg169_1);  view_182 = arg169_1 = None
        view_183: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_198);  bmm_20 = _shape_param_198 = None
        view_184: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_21, _shape_param_199);  bmm_21 = _shape_param_199 = None
        convert_element_type_204: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_184, torch.float32);  view_184 = None
        convert_element_type_205: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg170_1, torch.float32);  arg170_1 = None
        mul_184: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_205, 1.1111111111111112);  convert_element_type_205 = None
        mul_185: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_204, mul_184);  convert_element_type_204 = mul_184 = None
        convert_element_type_206: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.float32);  arg171_1 = None
        sub_7: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_206, arg172_1);  convert_element_type_206 = arg172_1 = None
        exp_13: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        div_65: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, arg173_1);  exp_13 = arg173_1 = None
        mul_186: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_185, div_65);  mul_185 = None
        sum_87: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [-1], True)
        neg_27: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_65);  div_65 = None
        fma_5: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_27, sum_87, mul_186);  neg_27 = sum_87 = mul_186 = None
        convert_element_type_207: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        where_16: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_207);  convert_element_type_207 = None
        div_66: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_16, 8.0);  where_16 = None
        view_185: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_66, _shape_param_200);  div_66 = _shape_param_200 = None
        bmm_22: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg174_1, view_185);  arg174_1 = None
        bmm_23: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_185, arg175_1);  view_185 = arg175_1 = None
        view_186: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_201);  bmm_22 = _shape_param_201 = None
        view_187: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_202);  bmm_23 = _shape_param_202 = None
        permute_61: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_186, [0, 1, 3, 2]);  view_186 = None
        permute_62: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        clone_21: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_188: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_21, _shape_param_203);  clone_21 = _shape_param_203 = None
        view_189: "bf16[2048, 768]" = torch.ops.aten.view.default(view_188, _shape_param_204);  view_188 = _shape_param_204 = None
        mm_70: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_189, arg176_1);  arg176_1 = None
        permute_63: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_189, [1, 0])
        mm_71: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_63, arg177_1);  permute_63 = None
        sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True, dtype = torch.float32);  view_189 = None
        view_190: "f32[768]" = torch.ops.aten.view.default(sum_88, _shape_param_205);  sum_88 = _shape_param_205 = None
        convert_element_type_208: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_190, torch.bfloat16);  view_190 = None
        view_191: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_70, _shape_param_206);  mm_70 = _shape_param_206 = None
        convert_element_type_209: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_191, torch.float32);  view_191 = None
        convert_element_type_210: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_211: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_208, torch.float32);  convert_element_type_208 = None
        permute_64: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_61, [0, 2, 1, 3]);  permute_61 = None
        view_192: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_64, _shape_param_207);  permute_64 = _shape_param_207 = None
        clone_22: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_192, memory_format = torch.contiguous_format);  view_192 = None
        view_193: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_22, _shape_param_208);  clone_22 = _shape_param_208 = None
        mm_72: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_193, arg178_1);  arg178_1 = None
        permute_65: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_193, [1, 0])
        mm_73: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_65, arg177_1);  permute_65 = None
        sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_193, [0], True, dtype = torch.float32);  view_193 = None
        view_194: "f32[768]" = torch.ops.aten.view.default(sum_89, _shape_param_209);  sum_89 = _shape_param_209 = None
        convert_element_type_212: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_194, torch.bfloat16);  view_194 = None
        view_195: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_72, _shape_param_210);  mm_72 = _shape_param_210 = None
        convert_element_type_213: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        add_67: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_209, convert_element_type_213);  convert_element_type_209 = convert_element_type_213 = None
        convert_element_type_214: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_215: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_212, torch.float32);  convert_element_type_212 = None
        permute_66: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None
        clone_23: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
        view_196: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_23, _shape_param_211);  clone_23 = _shape_param_211 = None
        view_197: "bf16[2048, 768]" = torch.ops.aten.view.default(view_196, _shape_param_212);  view_196 = _shape_param_212 = None
        mm_74: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_197, arg179_1);  arg179_1 = None
        permute_67: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_197, [1, 0])
        mm_75: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_67, arg177_1);  permute_67 = arg177_1 = None
        sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True, dtype = torch.float32);  view_197 = None
        view_198: "f32[768]" = torch.ops.aten.view.default(sum_90, _shape_param_213);  sum_90 = _shape_param_213 = None
        convert_element_type_216: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_198, torch.bfloat16);  view_198 = None
        view_199: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_74, _shape_param_214);  mm_74 = _shape_param_214 = None
        convert_element_type_217: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        add_68: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_67, convert_element_type_217);  add_67 = convert_element_type_217 = None
        convert_element_type_218: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_219: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_216, torch.float32);  convert_element_type_216 = None
        sum_91: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_68, [0, 1], True, dtype = torch.float32)
        view_200: "f32[768]" = torch.ops.aten.view.default(sum_91, _shape_param_215);  sum_91 = _shape_param_215 = None
        mul_187: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg180_1, arg181_1)
        add_69: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg182_1, 1e-06)
        div_67: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_187, add_69);  mul_187 = None
        div_68: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_67, add_69);  div_67 = None
        neg_28: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_68)
        mul_188: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_28, div_68);  neg_28 = div_68 = None
        div_69: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_68, add_69);  add_68 = add_69 = None
        sum_92: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True, dtype = torch.float32);  mul_188 = None
        mul_189: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_69, arg180_1);  arg180_1 = None
        mul_190: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_69, arg181_1);  div_69 = None
        sum_93: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1], True, dtype = torch.float32);  mul_190 = None
        view_201: "f32[768]" = torch.ops.aten.view.default(sum_93, _shape_param_216);  sum_93 = _shape_param_216 = None
        neg_29: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_189)
        sum_94: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_29, [2], True, dtype = torch.float32);  neg_29 = None
        add_70: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_66, mul_189);  add_66 = mul_189 = None
        mul_191: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg182_1, 2)
        div_70: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_92, mul_191);  sum_92 = mul_191 = None
        eq_12: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg182_1, 0);  arg182_1 = None
        where_17: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_12, full_1, div_70);  eq_12 = div_70 = None
        mul_192: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_17, 0.002607561929595828);  where_17 = None
        mul_193: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_192, arg181_1);  mul_192 = arg181_1 = None
        add_71: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_70, mul_193);  add_70 = mul_193 = None
        expand_11: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_94, _shape_param_217);  sum_94 = _shape_param_217 = None
        div_71: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_11, 768);  expand_11 = None
        add_72: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_71, div_71);  add_71 = div_71 = None
        convert_element_type_220: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.float32);  arg183_1 = None
        mul_194: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_220, 1.1111111111111112);  convert_element_type_220 = None
        mul_195: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_72, mul_194);  add_72 = mul_194 = None
        convert_element_type_221: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_195, torch.bfloat16)
        convert_element_type_222: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.bfloat16);  arg184_1 = None
        mul_196: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_222, 1.1111111111111112);  convert_element_type_222 = None
        mul_197: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_221, mul_196);  convert_element_type_221 = mul_196 = None
        view_202: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_197, _shape_param_218);  mul_197 = _shape_param_218 = None
        mm_76: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_202, arg185_1);  arg185_1 = None
        permute_68: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_202, [1, 0])
        mm_77: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_68, arg186_1);  permute_68 = arg186_1 = None
        sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_202, [0], True, dtype = torch.float32);  view_202 = None
        view_203: "f32[768]" = torch.ops.aten.view.default(sum_95, _shape_param_219);  sum_95 = _shape_param_219 = None
        convert_element_type_223: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_203, torch.bfloat16);  view_203 = None
        view_204: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_76, _shape_param_220);  mm_76 = _shape_param_220 = None
        convert_element_type_224: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_225: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_223, torch.float32);  convert_element_type_223 = None
        convert_element_type_226: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.bfloat16);  arg187_1 = None
        mul_198: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_226, 1.1111111111111112);  convert_element_type_226 = None
        mul_199: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_204, mul_198);  view_204 = mul_198 = None
        convert_element_type_227: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_199, torch.float32);  mul_199 = None
        view_205: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg188_1, _shape_param_221);  arg188_1 = _shape_param_221 = None
        convert_element_type_228: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        mul_200: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.7071067811865476)
        erf_6: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_200);  mul_200 = None
        add_73: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_201: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_73, 0.5);  add_73 = None
        mul_202: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_228, convert_element_type_228)
        mul_203: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_202, -0.5);  mul_202 = None
        exp_14: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_203);  mul_203 = None
        mul_204: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_205: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_228, mul_204);  convert_element_type_228 = mul_204 = None
        add_74: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_201, mul_205);  mul_201 = mul_205 = None
        mul_206: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_227, add_74);  convert_element_type_227 = add_74 = None
        convert_element_type_229: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_206, torch.bfloat16);  mul_206 = None
        view_206: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_229, _shape_param_222);  convert_element_type_229 = _shape_param_222 = None
        mm_78: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_206, arg189_1);  arg189_1 = None
        permute_69: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_206, [1, 0])
        mm_79: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_69, arg190_1);  permute_69 = arg190_1 = None
        sum_96: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_206, [0], True, dtype = torch.float32);  view_206 = None
        view_207: "f32[3072]" = torch.ops.aten.view.default(sum_96, _shape_param_223);  sum_96 = _shape_param_223 = None
        convert_element_type_230: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_207, torch.bfloat16);  view_207 = None
        view_208: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_78, _shape_param_224);  mm_78 = _shape_param_224 = None
        convert_element_type_231: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        convert_element_type_232: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_233: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_230, torch.float32);  convert_element_type_230 = None
        sum_97: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_231, [0, 1], True, dtype = torch.float32)
        view_209: "f32[768]" = torch.ops.aten.view.default(sum_97, _shape_param_225);  sum_97 = _shape_param_225 = None
        mul_207: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg191_1, arg192_1)
        add_75: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg193_1, 1e-06)
        div_72: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_207, add_75);  mul_207 = None
        div_73: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_72, add_75);  div_72 = None
        neg_30: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_231)
        mul_208: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_30, div_73);  neg_30 = div_73 = None
        div_74: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_231, add_75);  convert_element_type_231 = add_75 = None
        sum_98: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True, dtype = torch.float32);  mul_208 = None
        mul_209: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_74, arg191_1);  arg191_1 = None
        mul_210: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_74, arg192_1);  div_74 = None
        sum_99: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1], True, dtype = torch.float32);  mul_210 = None
        view_210: "f32[768]" = torch.ops.aten.view.default(sum_99, _shape_param_226);  sum_99 = _shape_param_226 = None
        neg_31: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_209)
        sum_100: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_31, [2], True, dtype = torch.float32);  neg_31 = None
        add_76: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_195, mul_209);  mul_195 = mul_209 = None
        mul_211: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg193_1, 2)
        div_75: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_98, mul_211);  sum_98 = mul_211 = None
        eq_13: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg193_1, 0);  arg193_1 = None
        where_18: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_13, full_1, div_75);  eq_13 = div_75 = None
        mul_212: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_18, 0.002607561929595828);  where_18 = None
        mul_213: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_212, arg192_1);  mul_212 = arg192_1 = None
        add_77: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_76, mul_213);  add_76 = mul_213 = None
        expand_12: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_100, _shape_param_227);  sum_100 = _shape_param_227 = None
        div_76: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_12, 768);  expand_12 = None
        add_78: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_77, div_76);  add_77 = div_76 = None
        convert_element_type_234: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16)
        convert_element_type_235: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.bfloat16);  arg194_1 = None
        mul_214: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_235, 1.1111111111111112);  convert_element_type_235 = None
        mul_215: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_234, mul_214);  convert_element_type_234 = mul_214 = None
        view_211: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_215, _shape_param_228);  mul_215 = _shape_param_228 = None
        mm_80: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_211, arg195_1);  arg195_1 = None
        permute_70: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_211, [1, 0])
        mm_81: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_70, arg196_1);  permute_70 = arg196_1 = None
        sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_211, [0], True, dtype = torch.float32);  view_211 = None
        view_212: "f32[768]" = torch.ops.aten.view.default(sum_101, _shape_param_229);  sum_101 = _shape_param_229 = None
        convert_element_type_236: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_212, torch.bfloat16);  view_212 = None
        view_213: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_80, _shape_param_230);  mm_80 = _shape_param_230 = None
        convert_element_type_237: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_238: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_236, torch.float32);  convert_element_type_236 = None
        view_214: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_213, _shape_param_231);  view_213 = _shape_param_231 = None
        permute_71: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
        clone_24: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_215: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_232);  clone_24 = _shape_param_232 = None
        bmm_24: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg197_1, view_215);  arg197_1 = None
        bmm_25: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_215, arg198_1);  view_215 = arg198_1 = None
        view_216: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_24, _shape_param_233);  bmm_24 = _shape_param_233 = None
        view_217: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_25, _shape_param_234);  bmm_25 = _shape_param_234 = None
        convert_element_type_239: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        convert_element_type_240: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.float32);  arg199_1 = None
        mul_216: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_240, 1.1111111111111112);  convert_element_type_240 = None
        mul_217: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_239, mul_216);  convert_element_type_239 = mul_216 = None
        convert_element_type_241: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg200_1, torch.float32);  arg200_1 = None
        sub_8: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_241, arg201_1);  convert_element_type_241 = arg201_1 = None
        exp_15: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        div_77: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, arg202_1);  exp_15 = arg202_1 = None
        mul_218: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_217, div_77);  mul_217 = None
        sum_102: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_218, [-1], True)
        neg_32: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_77);  div_77 = None
        fma_6: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_32, sum_102, mul_218);  neg_32 = sum_102 = mul_218 = None
        convert_element_type_242: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        where_19: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_242);  convert_element_type_242 = None
        div_78: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_19, 8.0);  where_19 = None
        view_218: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_78, _shape_param_235);  div_78 = _shape_param_235 = None
        bmm_26: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg203_1, view_218);  arg203_1 = None
        bmm_27: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_218, arg204_1);  view_218 = arg204_1 = None
        view_219: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_236);  bmm_26 = _shape_param_236 = None
        view_220: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_237);  bmm_27 = _shape_param_237 = None
        permute_72: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_219, [0, 1, 3, 2]);  view_219 = None
        permute_73: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
        clone_25: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_221: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_25, _shape_param_238);  clone_25 = _shape_param_238 = None
        view_222: "bf16[2048, 768]" = torch.ops.aten.view.default(view_221, _shape_param_239);  view_221 = _shape_param_239 = None
        mm_82: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_222, arg205_1);  arg205_1 = None
        permute_74: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_222, [1, 0])
        mm_83: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_74, arg206_1);  permute_74 = None
        sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True, dtype = torch.float32);  view_222 = None
        view_223: "f32[768]" = torch.ops.aten.view.default(sum_103, _shape_param_240);  sum_103 = _shape_param_240 = None
        convert_element_type_243: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_223, torch.bfloat16);  view_223 = None
        view_224: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_82, _shape_param_241);  mm_82 = _shape_param_241 = None
        convert_element_type_244: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_224, torch.float32);  view_224 = None
        convert_element_type_245: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_246: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_243, torch.float32);  convert_element_type_243 = None
        permute_75: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_72, [0, 2, 1, 3]);  permute_72 = None
        view_225: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_75, _shape_param_242);  permute_75 = _shape_param_242 = None
        clone_26: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_225, memory_format = torch.contiguous_format);  view_225 = None
        view_226: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_26, _shape_param_243);  clone_26 = _shape_param_243 = None
        mm_84: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_226, arg207_1);  arg207_1 = None
        permute_76: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_226, [1, 0])
        mm_85: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_76, arg206_1);  permute_76 = None
        sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True, dtype = torch.float32);  view_226 = None
        view_227: "f32[768]" = torch.ops.aten.view.default(sum_104, _shape_param_244);  sum_104 = _shape_param_244 = None
        convert_element_type_247: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_227, torch.bfloat16);  view_227 = None
        view_228: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_84, _shape_param_245);  mm_84 = _shape_param_245 = None
        convert_element_type_248: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_228, torch.float32);  view_228 = None
        add_79: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_244, convert_element_type_248);  convert_element_type_244 = convert_element_type_248 = None
        convert_element_type_249: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_250: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_247, torch.float32);  convert_element_type_247 = None
        permute_77: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
        clone_27: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_229: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_27, _shape_param_246);  clone_27 = _shape_param_246 = None
        view_230: "bf16[2048, 768]" = torch.ops.aten.view.default(view_229, _shape_param_247);  view_229 = _shape_param_247 = None
        mm_86: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_230, arg208_1);  arg208_1 = None
        permute_78: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_230, [1, 0])
        mm_87: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_78, arg206_1);  permute_78 = arg206_1 = None
        sum_105: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True, dtype = torch.float32);  view_230 = None
        view_231: "f32[768]" = torch.ops.aten.view.default(sum_105, _shape_param_248);  sum_105 = _shape_param_248 = None
        convert_element_type_251: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_231, torch.bfloat16);  view_231 = None
        view_232: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_86, _shape_param_249);  mm_86 = _shape_param_249 = None
        convert_element_type_252: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_232, torch.float32);  view_232 = None
        add_80: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_79, convert_element_type_252);  add_79 = convert_element_type_252 = None
        convert_element_type_253: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_254: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_251, torch.float32);  convert_element_type_251 = None
        sum_106: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_80, [0, 1], True, dtype = torch.float32)
        view_233: "f32[768]" = torch.ops.aten.view.default(sum_106, _shape_param_250);  sum_106 = _shape_param_250 = None
        mul_219: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg209_1, arg210_1)
        add_81: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg211_1, 1e-06)
        div_79: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_219, add_81);  mul_219 = None
        div_80: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_79, add_81);  div_79 = None
        neg_33: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_80)
        mul_220: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_33, div_80);  neg_33 = div_80 = None
        div_81: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_80, add_81);  add_80 = add_81 = None
        sum_107: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True, dtype = torch.float32);  mul_220 = None
        mul_221: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_81, arg209_1);  arg209_1 = None
        mul_222: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_81, arg210_1);  div_81 = None
        sum_108: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 1], True, dtype = torch.float32);  mul_222 = None
        view_234: "f32[768]" = torch.ops.aten.view.default(sum_108, _shape_param_251);  sum_108 = _shape_param_251 = None
        neg_34: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_221)
        sum_109: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_34, [2], True, dtype = torch.float32);  neg_34 = None
        add_82: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_78, mul_221);  add_78 = mul_221 = None
        mul_223: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg211_1, 2)
        div_82: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_107, mul_223);  sum_107 = mul_223 = None
        eq_14: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg211_1, 0);  arg211_1 = None
        where_20: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_14, full_1, div_82);  eq_14 = div_82 = None
        mul_224: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_20, 0.002607561929595828);  where_20 = None
        mul_225: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_224, arg210_1);  mul_224 = arg210_1 = None
        add_83: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_82, mul_225);  add_82 = mul_225 = None
        expand_13: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_109, _shape_param_252);  sum_109 = _shape_param_252 = None
        div_83: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_13, 768);  expand_13 = None
        add_84: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_83, div_83);  add_83 = div_83 = None
        convert_element_type_255: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.float32);  arg212_1 = None
        mul_226: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_255, 1.1111111111111112);  convert_element_type_255 = None
        mul_227: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_84, mul_226);  add_84 = mul_226 = None
        convert_element_type_256: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_227, torch.bfloat16)
        convert_element_type_257: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg213_1, torch.bfloat16);  arg213_1 = None
        mul_228: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_257, 1.1111111111111112);  convert_element_type_257 = None
        mul_229: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_256, mul_228);  convert_element_type_256 = mul_228 = None
        view_235: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_229, _shape_param_253);  mul_229 = _shape_param_253 = None
        mm_88: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_235, arg214_1);  arg214_1 = None
        permute_79: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_235, [1, 0])
        mm_89: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_79, arg215_1);  permute_79 = arg215_1 = None
        sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_235, [0], True, dtype = torch.float32);  view_235 = None
        view_236: "f32[768]" = torch.ops.aten.view.default(sum_110, _shape_param_254);  sum_110 = _shape_param_254 = None
        convert_element_type_258: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_236, torch.bfloat16);  view_236 = None
        view_237: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_88, _shape_param_255);  mm_88 = _shape_param_255 = None
        convert_element_type_259: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_260: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_258, torch.float32);  convert_element_type_258 = None
        convert_element_type_261: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg216_1, torch.bfloat16);  arg216_1 = None
        mul_230: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_261, 1.1111111111111112);  convert_element_type_261 = None
        mul_231: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_237, mul_230);  view_237 = mul_230 = None
        convert_element_type_262: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_231, torch.float32);  mul_231 = None
        view_238: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg217_1, _shape_param_256);  arg217_1 = _shape_param_256 = None
        convert_element_type_263: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_238, torch.float32);  view_238 = None
        mul_232: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_263, 0.7071067811865476)
        erf_7: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_232);  mul_232 = None
        add_85: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_233: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_85, 0.5);  add_85 = None
        mul_234: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_263, convert_element_type_263)
        mul_235: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_234, -0.5);  mul_234 = None
        exp_16: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_235);  mul_235 = None
        mul_236: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_237: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_263, mul_236);  convert_element_type_263 = mul_236 = None
        add_86: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_233, mul_237);  mul_233 = mul_237 = None
        mul_238: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_262, add_86);  convert_element_type_262 = add_86 = None
        convert_element_type_264: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_238, torch.bfloat16);  mul_238 = None
        view_239: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_264, _shape_param_257);  convert_element_type_264 = _shape_param_257 = None
        mm_90: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_239, arg218_1);  arg218_1 = None
        permute_80: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_239, [1, 0])
        mm_91: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_80, arg219_1);  permute_80 = arg219_1 = None
        sum_111: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True, dtype = torch.float32);  view_239 = None
        view_240: "f32[3072]" = torch.ops.aten.view.default(sum_111, _shape_param_258);  sum_111 = _shape_param_258 = None
        convert_element_type_265: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_240, torch.bfloat16);  view_240 = None
        view_241: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_90, _shape_param_259);  mm_90 = _shape_param_259 = None
        convert_element_type_266: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_241, torch.float32);  view_241 = None
        convert_element_type_267: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_268: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_265, torch.float32);  convert_element_type_265 = None
        sum_112: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_266, [0, 1], True, dtype = torch.float32)
        view_242: "f32[768]" = torch.ops.aten.view.default(sum_112, _shape_param_260);  sum_112 = _shape_param_260 = None
        mul_239: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg220_1, arg221_1)
        add_87: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg222_1, 1e-06)
        div_84: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_239, add_87);  mul_239 = None
        div_85: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_84, add_87);  div_84 = None
        neg_35: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_266)
        mul_240: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_35, div_85);  neg_35 = div_85 = None
        div_86: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_266, add_87);  convert_element_type_266 = add_87 = None
        sum_113: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True, dtype = torch.float32);  mul_240 = None
        mul_241: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_86, arg220_1);  arg220_1 = None
        mul_242: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_86, arg221_1);  div_86 = None
        sum_114: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1], True, dtype = torch.float32);  mul_242 = None
        view_243: "f32[768]" = torch.ops.aten.view.default(sum_114, _shape_param_261);  sum_114 = _shape_param_261 = None
        neg_36: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_241)
        sum_115: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_36, [2], True, dtype = torch.float32);  neg_36 = None
        add_88: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_227, mul_241);  mul_227 = mul_241 = None
        mul_243: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg222_1, 2)
        div_87: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_113, mul_243);  sum_113 = mul_243 = None
        eq_15: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg222_1, 0);  arg222_1 = None
        where_21: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_15, full_1, div_87);  eq_15 = div_87 = None
        mul_244: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_21, 0.002607561929595828);  where_21 = None
        mul_245: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_244, arg221_1);  mul_244 = arg221_1 = None
        add_89: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_88, mul_245);  add_88 = mul_245 = None
        expand_14: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_115, _shape_param_262);  sum_115 = _shape_param_262 = None
        div_88: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_14, 768);  expand_14 = None
        add_90: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_89, div_88);  add_89 = div_88 = None
        convert_element_type_269: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16)
        convert_element_type_270: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg223_1, torch.bfloat16);  arg223_1 = None
        mul_246: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_270, 1.1111111111111112);  convert_element_type_270 = None
        mul_247: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_269, mul_246);  convert_element_type_269 = mul_246 = None
        view_244: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_247, _shape_param_263);  mul_247 = _shape_param_263 = None
        mm_92: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_244, arg224_1);  arg224_1 = None
        permute_81: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_244, [1, 0])
        mm_93: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_81, arg225_1);  permute_81 = arg225_1 = None
        sum_116: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True, dtype = torch.float32);  view_244 = None
        view_245: "f32[768]" = torch.ops.aten.view.default(sum_116, _shape_param_264);  sum_116 = _shape_param_264 = None
        convert_element_type_271: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_245, torch.bfloat16);  view_245 = None
        view_246: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_92, _shape_param_265);  mm_92 = _shape_param_265 = None
        convert_element_type_272: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_273: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_271, torch.float32);  convert_element_type_271 = None
        view_247: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_246, _shape_param_266);  view_246 = _shape_param_266 = None
        permute_82: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        clone_28: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_248: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_267);  clone_28 = _shape_param_267 = None
        bmm_28: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg226_1, view_248);  arg226_1 = None
        bmm_29: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_248, arg227_1);  view_248 = arg227_1 = None
        view_249: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_28, _shape_param_268);  bmm_28 = _shape_param_268 = None
        view_250: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_29, _shape_param_269);  bmm_29 = _shape_param_269 = None
        convert_element_type_274: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_250, torch.float32);  view_250 = None
        convert_element_type_275: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.float32);  arg228_1 = None
        mul_248: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_275, 1.1111111111111112);  convert_element_type_275 = None
        mul_249: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_274, mul_248);  convert_element_type_274 = mul_248 = None
        convert_element_type_276: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg229_1, torch.float32);  arg229_1 = None
        sub_9: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_276, arg230_1);  convert_element_type_276 = arg230_1 = None
        exp_17: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        div_89: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, arg231_1);  exp_17 = arg231_1 = None
        mul_250: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_249, div_89);  mul_249 = None
        sum_117: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [-1], True)
        neg_37: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_89);  div_89 = None
        fma_7: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_37, sum_117, mul_250);  neg_37 = sum_117 = mul_250 = None
        convert_element_type_277: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        where_22: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_277);  convert_element_type_277 = None
        div_90: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_22, 8.0);  where_22 = None
        view_251: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_90, _shape_param_270);  div_90 = _shape_param_270 = None
        bmm_30: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg232_1, view_251);  arg232_1 = None
        bmm_31: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_251, arg233_1);  view_251 = arg233_1 = None
        view_252: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_271);  bmm_30 = _shape_param_271 = None
        view_253: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_272);  bmm_31 = _shape_param_272 = None
        permute_83: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_252, [0, 1, 3, 2]);  view_252 = None
        permute_84: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_249, [0, 2, 1, 3]);  view_249 = None
        clone_29: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_254: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_29, _shape_param_273);  clone_29 = _shape_param_273 = None
        view_255: "bf16[2048, 768]" = torch.ops.aten.view.default(view_254, _shape_param_274);  view_254 = _shape_param_274 = None
        mm_94: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_255, arg234_1);  arg234_1 = None
        permute_85: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_255, [1, 0])
        mm_95: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_85, arg235_1);  permute_85 = None
        sum_118: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True, dtype = torch.float32);  view_255 = None
        view_256: "f32[768]" = torch.ops.aten.view.default(sum_118, _shape_param_275);  sum_118 = _shape_param_275 = None
        convert_element_type_278: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_256, torch.bfloat16);  view_256 = None
        view_257: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_94, _shape_param_276);  mm_94 = _shape_param_276 = None
        convert_element_type_279: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None
        convert_element_type_280: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_281: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_278, torch.float32);  convert_element_type_278 = None
        permute_86: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_83, [0, 2, 1, 3]);  permute_83 = None
        view_258: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_86, _shape_param_277);  permute_86 = _shape_param_277 = None
        clone_30: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_258, memory_format = torch.contiguous_format);  view_258 = None
        view_259: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_30, _shape_param_278);  clone_30 = _shape_param_278 = None
        mm_96: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_259, arg236_1);  arg236_1 = None
        permute_87: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_259, [1, 0])
        mm_97: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_87, arg235_1);  permute_87 = None
        sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_259, [0], True, dtype = torch.float32);  view_259 = None
        view_260: "f32[768]" = torch.ops.aten.view.default(sum_119, _shape_param_279);  sum_119 = _shape_param_279 = None
        convert_element_type_282: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_260, torch.bfloat16);  view_260 = None
        view_261: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_96, _shape_param_280);  mm_96 = _shape_param_280 = None
        convert_element_type_283: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        add_91: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_279, convert_element_type_283);  convert_element_type_279 = convert_element_type_283 = None
        convert_element_type_284: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_285: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_282, torch.float32);  convert_element_type_282 = None
        permute_88: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
        clone_31: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
        view_262: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_31, _shape_param_281);  clone_31 = _shape_param_281 = None
        view_263: "bf16[2048, 768]" = torch.ops.aten.view.default(view_262, _shape_param_282);  view_262 = _shape_param_282 = None
        mm_98: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_263, arg237_1);  arg237_1 = None
        permute_89: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_263, [1, 0])
        mm_99: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_89, arg235_1);  permute_89 = arg235_1 = None
        sum_120: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True, dtype = torch.float32);  view_263 = None
        view_264: "f32[768]" = torch.ops.aten.view.default(sum_120, _shape_param_283);  sum_120 = _shape_param_283 = None
        convert_element_type_286: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_264, torch.bfloat16);  view_264 = None
        view_265: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_98, _shape_param_284);  mm_98 = _shape_param_284 = None
        convert_element_type_287: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        add_92: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_91, convert_element_type_287);  add_91 = convert_element_type_287 = None
        convert_element_type_288: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_289: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_286, torch.float32);  convert_element_type_286 = None
        sum_121: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_92, [0, 1], True, dtype = torch.float32)
        view_266: "f32[768]" = torch.ops.aten.view.default(sum_121, _shape_param_285);  sum_121 = _shape_param_285 = None
        mul_251: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg238_1, arg239_1)
        add_93: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg240_1, 1e-06)
        div_91: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_251, add_93);  mul_251 = None
        div_92: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_91, add_93);  div_91 = None
        neg_38: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_92)
        mul_252: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_38, div_92);  neg_38 = div_92 = None
        div_93: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_92, add_93);  add_92 = add_93 = None
        sum_122: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True, dtype = torch.float32);  mul_252 = None
        mul_253: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_93, arg238_1);  arg238_1 = None
        mul_254: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_93, arg239_1);  div_93 = None
        sum_123: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1], True, dtype = torch.float32);  mul_254 = None
        view_267: "f32[768]" = torch.ops.aten.view.default(sum_123, _shape_param_286);  sum_123 = _shape_param_286 = None
        neg_39: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_253)
        sum_124: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_39, [2], True, dtype = torch.float32);  neg_39 = None
        add_94: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_90, mul_253);  add_90 = mul_253 = None
        mul_255: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg240_1, 2)
        div_94: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_122, mul_255);  sum_122 = mul_255 = None
        eq_16: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg240_1, 0);  arg240_1 = None
        where_23: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_16, full_1, div_94);  eq_16 = div_94 = None
        mul_256: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_23, 0.002607561929595828);  where_23 = None
        mul_257: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_256, arg239_1);  mul_256 = arg239_1 = None
        add_95: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_94, mul_257);  add_94 = mul_257 = None
        expand_15: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_124, _shape_param_287);  sum_124 = _shape_param_287 = None
        div_95: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_15, 768);  expand_15 = None
        add_96: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_95, div_95);  add_95 = div_95 = None
        convert_element_type_290: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg241_1, torch.float32);  arg241_1 = None
        mul_258: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_290, 1.1111111111111112);  convert_element_type_290 = None
        mul_259: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_96, mul_258);  add_96 = mul_258 = None
        convert_element_type_291: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_259, torch.bfloat16)
        convert_element_type_292: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.bfloat16);  arg242_1 = None
        mul_260: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_292, 1.1111111111111112);  convert_element_type_292 = None
        mul_261: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_291, mul_260);  convert_element_type_291 = mul_260 = None
        view_268: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_261, _shape_param_288);  mul_261 = _shape_param_288 = None
        mm_100: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_268, arg243_1);  arg243_1 = None
        permute_90: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_268, [1, 0])
        mm_101: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_90, arg244_1);  permute_90 = arg244_1 = None
        sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True, dtype = torch.float32);  view_268 = None
        view_269: "f32[768]" = torch.ops.aten.view.default(sum_125, _shape_param_289);  sum_125 = _shape_param_289 = None
        convert_element_type_293: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_269, torch.bfloat16);  view_269 = None
        view_270: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_100, _shape_param_290);  mm_100 = _shape_param_290 = None
        convert_element_type_294: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_295: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_293, torch.float32);  convert_element_type_293 = None
        convert_element_type_296: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg245_1, torch.bfloat16);  arg245_1 = None
        mul_262: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_296, 1.1111111111111112);  convert_element_type_296 = None
        mul_263: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_270, mul_262);  view_270 = mul_262 = None
        convert_element_type_297: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_263, torch.float32);  mul_263 = None
        view_271: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg246_1, _shape_param_291);  arg246_1 = _shape_param_291 = None
        convert_element_type_298: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_271, torch.float32);  view_271 = None
        mul_264: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_298, 0.7071067811865476)
        erf_8: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_264);  mul_264 = None
        add_97: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_265: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
        mul_266: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_298, convert_element_type_298)
        mul_267: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_266, -0.5);  mul_266 = None
        exp_18: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_267);  mul_267 = None
        mul_268: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_269: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_298, mul_268);  convert_element_type_298 = mul_268 = None
        add_98: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_265, mul_269);  mul_265 = mul_269 = None
        mul_270: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_297, add_98);  convert_element_type_297 = add_98 = None
        convert_element_type_299: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_270, torch.bfloat16);  mul_270 = None
        view_272: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_299, _shape_param_292);  convert_element_type_299 = _shape_param_292 = None
        mm_102: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_272, arg247_1);  arg247_1 = None
        permute_91: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_272, [1, 0])
        mm_103: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_91, arg248_1);  permute_91 = arg248_1 = None
        sum_126: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True, dtype = torch.float32);  view_272 = None
        view_273: "f32[3072]" = torch.ops.aten.view.default(sum_126, _shape_param_293);  sum_126 = _shape_param_293 = None
        convert_element_type_300: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_273, torch.bfloat16);  view_273 = None
        view_274: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_102, _shape_param_294);  mm_102 = _shape_param_294 = None
        convert_element_type_301: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_274, torch.float32);  view_274 = None
        convert_element_type_302: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_303: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_300, torch.float32);  convert_element_type_300 = None
        sum_127: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_301, [0, 1], True, dtype = torch.float32)
        view_275: "f32[768]" = torch.ops.aten.view.default(sum_127, _shape_param_295);  sum_127 = _shape_param_295 = None
        mul_271: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg249_1, arg250_1)
        add_99: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg251_1, 1e-06)
        div_96: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_271, add_99);  mul_271 = None
        div_97: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_96, add_99);  div_96 = None
        neg_40: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_301)
        mul_272: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_40, div_97);  neg_40 = div_97 = None
        div_98: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_301, add_99);  convert_element_type_301 = add_99 = None
        sum_128: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True, dtype = torch.float32);  mul_272 = None
        mul_273: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_98, arg249_1);  arg249_1 = None
        mul_274: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_98, arg250_1);  div_98 = None
        sum_129: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1], True, dtype = torch.float32);  mul_274 = None
        view_276: "f32[768]" = torch.ops.aten.view.default(sum_129, _shape_param_296);  sum_129 = _shape_param_296 = None
        neg_41: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_273)
        sum_130: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_41, [2], True, dtype = torch.float32);  neg_41 = None
        add_100: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_259, mul_273);  mul_259 = mul_273 = None
        mul_275: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg251_1, 2)
        div_99: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_128, mul_275);  sum_128 = mul_275 = None
        eq_17: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg251_1, 0);  arg251_1 = None
        where_24: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_17, full_1, div_99);  eq_17 = div_99 = None
        mul_276: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_24, 0.002607561929595828);  where_24 = None
        mul_277: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_276, arg250_1);  mul_276 = arg250_1 = None
        add_101: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_100, mul_277);  add_100 = mul_277 = None
        expand_16: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_130, _shape_param_297);  sum_130 = _shape_param_297 = None
        div_100: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_16, 768);  expand_16 = None
        add_102: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_101, div_100);  add_101 = div_100 = None
        convert_element_type_304: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16)
        convert_element_type_305: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.bfloat16);  arg252_1 = None
        mul_278: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_305, 1.1111111111111112);  convert_element_type_305 = None
        mul_279: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_304, mul_278);  convert_element_type_304 = mul_278 = None
        view_277: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_279, _shape_param_298);  mul_279 = _shape_param_298 = None
        mm_104: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_277, arg253_1);  arg253_1 = None
        permute_92: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_277, [1, 0])
        mm_105: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_92, arg254_1);  permute_92 = arg254_1 = None
        sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True, dtype = torch.float32);  view_277 = None
        view_278: "f32[768]" = torch.ops.aten.view.default(sum_131, _shape_param_299);  sum_131 = _shape_param_299 = None
        convert_element_type_306: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_278, torch.bfloat16);  view_278 = None
        view_279: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_104, _shape_param_300);  mm_104 = _shape_param_300 = None
        convert_element_type_307: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_308: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_306, torch.float32);  convert_element_type_306 = None
        view_280: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_279, _shape_param_301);  view_279 = _shape_param_301 = None
        permute_93: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
        clone_32: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_281: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_302);  clone_32 = _shape_param_302 = None
        bmm_32: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg255_1, view_281);  arg255_1 = None
        bmm_33: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_281, arg256_1);  view_281 = arg256_1 = None
        view_282: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_32, _shape_param_303);  bmm_32 = _shape_param_303 = None
        view_283: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_33, _shape_param_304);  bmm_33 = _shape_param_304 = None
        convert_element_type_309: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        convert_element_type_310: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.float32);  arg257_1 = None
        mul_280: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_310, 1.1111111111111112);  convert_element_type_310 = None
        mul_281: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_309, mul_280);  convert_element_type_309 = mul_280 = None
        convert_element_type_311: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.float32);  arg258_1 = None
        sub_10: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_311, arg259_1);  convert_element_type_311 = arg259_1 = None
        exp_19: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        div_101: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, arg260_1);  exp_19 = arg260_1 = None
        mul_282: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_281, div_101);  mul_281 = None
        sum_132: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [-1], True)
        neg_42: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_101);  div_101 = None
        fma_8: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_42, sum_132, mul_282);  neg_42 = sum_132 = mul_282 = None
        convert_element_type_312: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        where_25: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_312);  convert_element_type_312 = None
        div_102: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_25, 8.0);  where_25 = None
        view_284: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_102, _shape_param_305);  div_102 = _shape_param_305 = None
        bmm_34: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg261_1, view_284);  arg261_1 = None
        bmm_35: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_284, arg262_1);  view_284 = arg262_1 = None
        view_285: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_306);  bmm_34 = _shape_param_306 = None
        view_286: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_307);  bmm_35 = _shape_param_307 = None
        permute_94: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_285, [0, 1, 3, 2]);  view_285 = None
        permute_95: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
        clone_33: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_287: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_33, _shape_param_308);  clone_33 = _shape_param_308 = None
        view_288: "bf16[2048, 768]" = torch.ops.aten.view.default(view_287, _shape_param_309);  view_287 = _shape_param_309 = None
        mm_106: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_288, arg263_1);  arg263_1 = None
        permute_96: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_107: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_96, arg264_1);  permute_96 = None
        sum_133: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[768]" = torch.ops.aten.view.default(sum_133, _shape_param_310);  sum_133 = _shape_param_310 = None
        convert_element_type_313: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        view_290: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_106, _shape_param_311);  mm_106 = _shape_param_311 = None
        convert_element_type_314: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_290, torch.float32);  view_290 = None
        convert_element_type_315: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_316: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_313, torch.float32);  convert_element_type_313 = None
        permute_97: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_94, [0, 2, 1, 3]);  permute_94 = None
        view_291: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_97, _shape_param_312);  permute_97 = _shape_param_312 = None
        clone_34: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_291, memory_format = torch.contiguous_format);  view_291 = None
        view_292: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_34, _shape_param_313);  clone_34 = _shape_param_313 = None
        mm_108: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_292, arg265_1);  arg265_1 = None
        permute_98: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_292, [1, 0])
        mm_109: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_98, arg264_1);  permute_98 = None
        sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_292, [0], True, dtype = torch.float32);  view_292 = None
        view_293: "f32[768]" = torch.ops.aten.view.default(sum_134, _shape_param_314);  sum_134 = _shape_param_314 = None
        convert_element_type_317: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_293, torch.bfloat16);  view_293 = None
        view_294: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_108, _shape_param_315);  mm_108 = _shape_param_315 = None
        convert_element_type_318: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_294, torch.float32);  view_294 = None
        add_103: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_314, convert_element_type_318);  convert_element_type_314 = convert_element_type_318 = None
        convert_element_type_319: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_320: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_317, torch.float32);  convert_element_type_317 = None
        permute_99: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
        clone_35: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_295: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_35, _shape_param_316);  clone_35 = _shape_param_316 = None
        view_296: "bf16[2048, 768]" = torch.ops.aten.view.default(view_295, _shape_param_317);  view_295 = _shape_param_317 = None
        mm_110: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_296, arg266_1);  arg266_1 = None
        permute_100: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_296, [1, 0])
        mm_111: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_100, arg264_1);  permute_100 = arg264_1 = None
        sum_135: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True, dtype = torch.float32);  view_296 = None
        view_297: "f32[768]" = torch.ops.aten.view.default(sum_135, _shape_param_318);  sum_135 = _shape_param_318 = None
        convert_element_type_321: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_297, torch.bfloat16);  view_297 = None
        view_298: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_110, _shape_param_319);  mm_110 = _shape_param_319 = None
        convert_element_type_322: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_298, torch.float32);  view_298 = None
        add_104: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_103, convert_element_type_322);  add_103 = convert_element_type_322 = None
        convert_element_type_323: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_324: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_321, torch.float32);  convert_element_type_321 = None
        sum_136: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_104, [0, 1], True, dtype = torch.float32)
        view_299: "f32[768]" = torch.ops.aten.view.default(sum_136, _shape_param_320);  sum_136 = _shape_param_320 = None
        mul_283: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg267_1, arg268_1)
        add_105: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg269_1, 1e-06)
        div_103: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_283, add_105);  mul_283 = None
        div_104: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_103, add_105);  div_103 = None
        neg_43: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_104)
        mul_284: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_43, div_104);  neg_43 = div_104 = None
        div_105: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_104, add_105);  add_104 = add_105 = None
        sum_137: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True, dtype = torch.float32);  mul_284 = None
        mul_285: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_105, arg267_1);  arg267_1 = None
        mul_286: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_105, arg268_1);  div_105 = None
        sum_138: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 1], True, dtype = torch.float32);  mul_286 = None
        view_300: "f32[768]" = torch.ops.aten.view.default(sum_138, _shape_param_321);  sum_138 = _shape_param_321 = None
        neg_44: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_285)
        sum_139: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_44, [2], True, dtype = torch.float32);  neg_44 = None
        add_106: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_102, mul_285);  add_102 = mul_285 = None
        mul_287: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg269_1, 2)
        div_106: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_137, mul_287);  sum_137 = mul_287 = None
        eq_18: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg269_1, 0);  arg269_1 = None
        where_26: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_18, full_1, div_106);  eq_18 = div_106 = None
        mul_288: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_26, 0.002607561929595828);  where_26 = None
        mul_289: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_288, arg268_1);  mul_288 = arg268_1 = None
        add_107: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_106, mul_289);  add_106 = mul_289 = None
        expand_17: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_139, _shape_param_322);  sum_139 = _shape_param_322 = None
        div_107: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_17, 768);  expand_17 = None
        add_108: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_107, div_107);  add_107 = div_107 = None
        convert_element_type_325: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg270_1, torch.float32);  arg270_1 = None
        mul_290: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_325, 1.1111111111111112);  convert_element_type_325 = None
        mul_291: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_108, mul_290);  add_108 = mul_290 = None
        convert_element_type_326: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_291, torch.bfloat16)
        convert_element_type_327: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg271_1, torch.bfloat16);  arg271_1 = None
        mul_292: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_327, 1.1111111111111112);  convert_element_type_327 = None
        mul_293: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_326, mul_292);  convert_element_type_326 = mul_292 = None
        view_301: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_293, _shape_param_323);  mul_293 = _shape_param_323 = None
        mm_112: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_301, arg272_1);  arg272_1 = None
        permute_101: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_301, [1, 0])
        mm_113: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_101, arg273_1);  permute_101 = arg273_1 = None
        sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True, dtype = torch.float32);  view_301 = None
        view_302: "f32[768]" = torch.ops.aten.view.default(sum_140, _shape_param_324);  sum_140 = _shape_param_324 = None
        convert_element_type_328: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_302, torch.bfloat16);  view_302 = None
        view_303: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_112, _shape_param_325);  mm_112 = _shape_param_325 = None
        convert_element_type_329: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_330: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_328, torch.float32);  convert_element_type_328 = None
        convert_element_type_331: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg274_1, torch.bfloat16);  arg274_1 = None
        mul_294: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_331, 1.1111111111111112);  convert_element_type_331 = None
        mul_295: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_303, mul_294);  view_303 = mul_294 = None
        convert_element_type_332: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_295, torch.float32);  mul_295 = None
        view_304: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg275_1, _shape_param_326);  arg275_1 = _shape_param_326 = None
        convert_element_type_333: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_304, torch.float32);  view_304 = None
        mul_296: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_333, 0.7071067811865476)
        erf_9: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_296);  mul_296 = None
        add_109: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_297: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_109, 0.5);  add_109 = None
        mul_298: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_333, convert_element_type_333)
        mul_299: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_298, -0.5);  mul_298 = None
        exp_20: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_299);  mul_299 = None
        mul_300: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_301: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_333, mul_300);  convert_element_type_333 = mul_300 = None
        add_110: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_297, mul_301);  mul_297 = mul_301 = None
        mul_302: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_332, add_110);  convert_element_type_332 = add_110 = None
        convert_element_type_334: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16);  mul_302 = None
        view_305: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_334, _shape_param_327);  convert_element_type_334 = _shape_param_327 = None
        mm_114: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_305, arg276_1);  arg276_1 = None
        permute_102: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_115: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_102, arg277_1);  permute_102 = arg277_1 = None
        sum_141: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True, dtype = torch.float32);  view_305 = None
        view_306: "f32[3072]" = torch.ops.aten.view.default(sum_141, _shape_param_328);  sum_141 = _shape_param_328 = None
        convert_element_type_335: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_306, torch.bfloat16);  view_306 = None
        view_307: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_114, _shape_param_329);  mm_114 = _shape_param_329 = None
        convert_element_type_336: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        convert_element_type_337: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_338: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_335, torch.float32);  convert_element_type_335 = None
        sum_142: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_336, [0, 1], True, dtype = torch.float32)
        view_308: "f32[768]" = torch.ops.aten.view.default(sum_142, _shape_param_330);  sum_142 = _shape_param_330 = None
        mul_303: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg278_1, arg279_1)
        add_111: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg280_1, 1e-06)
        div_108: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_303, add_111);  mul_303 = None
        div_109: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_108, add_111);  div_108 = None
        neg_45: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_336)
        mul_304: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_45, div_109);  neg_45 = div_109 = None
        div_110: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_336, add_111);  convert_element_type_336 = add_111 = None
        sum_143: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True, dtype = torch.float32);  mul_304 = None
        mul_305: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_110, arg278_1);  arg278_1 = None
        mul_306: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_110, arg279_1);  div_110 = None
        sum_144: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_306, [0, 1], True, dtype = torch.float32);  mul_306 = None
        view_309: "f32[768]" = torch.ops.aten.view.default(sum_144, _shape_param_331);  sum_144 = _shape_param_331 = None
        neg_46: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_305)
        sum_145: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_46, [2], True, dtype = torch.float32);  neg_46 = None
        add_112: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_291, mul_305);  mul_291 = mul_305 = None
        mul_307: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg280_1, 2)
        div_111: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_143, mul_307);  sum_143 = mul_307 = None
        eq_19: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg280_1, 0);  arg280_1 = None
        where_27: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_19, full_1, div_111);  eq_19 = div_111 = None
        mul_308: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_27, 0.002607561929595828);  where_27 = None
        mul_309: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_308, arg279_1);  mul_308 = arg279_1 = None
        add_113: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_112, mul_309);  add_112 = mul_309 = None
        expand_18: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_145, _shape_param_332);  sum_145 = _shape_param_332 = None
        div_112: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_18, 768);  expand_18 = None
        add_114: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_113, div_112);  add_113 = div_112 = None
        convert_element_type_339: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16)
        convert_element_type_340: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.bfloat16);  arg281_1 = None
        mul_310: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_340, 1.1111111111111112);  convert_element_type_340 = None
        mul_311: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_339, mul_310);  convert_element_type_339 = mul_310 = None
        view_310: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_311, _shape_param_333);  mul_311 = _shape_param_333 = None
        mm_116: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_310, arg282_1);  arg282_1 = None
        permute_103: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_117: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_103, arg283_1);  permute_103 = arg283_1 = None
        sum_146: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True, dtype = torch.float32);  view_310 = None
        view_311: "f32[768]" = torch.ops.aten.view.default(sum_146, _shape_param_334);  sum_146 = _shape_param_334 = None
        convert_element_type_341: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_311, torch.bfloat16);  view_311 = None
        view_312: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_116, _shape_param_335);  mm_116 = _shape_param_335 = None
        convert_element_type_342: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_343: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_341, torch.float32);  convert_element_type_341 = None
        view_313: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_312, _shape_param_336);  view_312 = _shape_param_336 = None
        permute_104: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
        clone_36: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_314: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_337);  clone_36 = _shape_param_337 = None
        bmm_36: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg284_1, view_314);  arg284_1 = None
        bmm_37: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_314, arg285_1);  view_314 = arg285_1 = None
        view_315: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_36, _shape_param_338);  bmm_36 = _shape_param_338 = None
        view_316: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_37, _shape_param_339);  bmm_37 = _shape_param_339 = None
        convert_element_type_344: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_316, torch.float32);  view_316 = None
        convert_element_type_345: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg286_1, torch.float32);  arg286_1 = None
        mul_312: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_345, 1.1111111111111112);  convert_element_type_345 = None
        mul_313: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_344, mul_312);  convert_element_type_344 = mul_312 = None
        convert_element_type_346: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg287_1, torch.float32);  arg287_1 = None
        sub_11: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_346, arg288_1);  convert_element_type_346 = arg288_1 = None
        exp_21: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        div_113: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, arg289_1);  exp_21 = arg289_1 = None
        mul_314: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_313, div_113);  mul_313 = None
        sum_147: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [-1], True)
        neg_47: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_113);  div_113 = None
        fma_9: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_47, sum_147, mul_314);  neg_47 = sum_147 = mul_314 = None
        convert_element_type_347: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        where_28: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_347);  convert_element_type_347 = None
        div_114: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_28, 8.0);  where_28 = None
        view_317: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_114, _shape_param_340);  div_114 = _shape_param_340 = None
        bmm_38: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg290_1, view_317);  arg290_1 = None
        bmm_39: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_317, arg291_1);  view_317 = arg291_1 = None
        view_318: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_341);  bmm_38 = _shape_param_341 = None
        view_319: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_342);  bmm_39 = _shape_param_342 = None
        permute_105: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_318, [0, 1, 3, 2]);  view_318 = None
        permute_106: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_315, [0, 2, 1, 3]);  view_315 = None
        clone_37: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_320: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_37, _shape_param_343);  clone_37 = _shape_param_343 = None
        view_321: "bf16[2048, 768]" = torch.ops.aten.view.default(view_320, _shape_param_344);  view_320 = _shape_param_344 = None
        mm_118: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_321, arg292_1);  arg292_1 = None
        permute_107: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_119: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_107, arg293_1);  permute_107 = None
        sum_148: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True, dtype = torch.float32);  view_321 = None
        view_322: "f32[768]" = torch.ops.aten.view.default(sum_148, _shape_param_345);  sum_148 = _shape_param_345 = None
        convert_element_type_348: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_322, torch.bfloat16);  view_322 = None
        view_323: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_118, _shape_param_346);  mm_118 = _shape_param_346 = None
        convert_element_type_349: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None
        convert_element_type_350: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_351: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_348, torch.float32);  convert_element_type_348 = None
        permute_108: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_105, [0, 2, 1, 3]);  permute_105 = None
        view_324: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_108, _shape_param_347);  permute_108 = _shape_param_347 = None
        clone_38: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_324, memory_format = torch.contiguous_format);  view_324 = None
        view_325: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_38, _shape_param_348);  clone_38 = _shape_param_348 = None
        mm_120: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_325, arg294_1);  arg294_1 = None
        permute_109: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_325, [1, 0])
        mm_121: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_109, arg293_1);  permute_109 = None
        sum_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True, dtype = torch.float32);  view_325 = None
        view_326: "f32[768]" = torch.ops.aten.view.default(sum_149, _shape_param_349);  sum_149 = _shape_param_349 = None
        convert_element_type_352: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_326, torch.bfloat16);  view_326 = None
        view_327: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_120, _shape_param_350);  mm_120 = _shape_param_350 = None
        convert_element_type_353: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        add_115: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_349, convert_element_type_353);  convert_element_type_349 = convert_element_type_353 = None
        convert_element_type_354: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_355: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_352, torch.float32);  convert_element_type_352 = None
        permute_110: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
        clone_39: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_328: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_39, _shape_param_351);  clone_39 = _shape_param_351 = None
        view_329: "bf16[2048, 768]" = torch.ops.aten.view.default(view_328, _shape_param_352);  view_328 = _shape_param_352 = None
        mm_122: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_329, arg295_1);  arg295_1 = None
        permute_111: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_329, [1, 0])
        mm_123: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_111, arg293_1);  permute_111 = arg293_1 = None
        sum_150: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True, dtype = torch.float32);  view_329 = None
        view_330: "f32[768]" = torch.ops.aten.view.default(sum_150, _shape_param_353);  sum_150 = _shape_param_353 = None
        convert_element_type_356: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_330, torch.bfloat16);  view_330 = None
        view_331: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_122, _shape_param_354);  mm_122 = _shape_param_354 = None
        convert_element_type_357: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_331, torch.float32);  view_331 = None
        add_116: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_115, convert_element_type_357);  add_115 = convert_element_type_357 = None
        convert_element_type_358: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_359: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_356, torch.float32);  convert_element_type_356 = None
        sum_151: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1], True, dtype = torch.float32)
        view_332: "f32[768]" = torch.ops.aten.view.default(sum_151, _shape_param_355);  sum_151 = _shape_param_355 = None
        mul_315: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg296_1, arg297_1)
        add_117: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg298_1, 1e-06)
        div_115: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_315, add_117);  mul_315 = None
        div_116: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_115, add_117);  div_115 = None
        neg_48: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_116)
        mul_316: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_48, div_116);  neg_48 = div_116 = None
        div_117: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_116, add_117);  add_116 = add_117 = None
        sum_152: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True, dtype = torch.float32);  mul_316 = None
        mul_317: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_117, arg296_1);  arg296_1 = None
        mul_318: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_117, arg297_1);  div_117 = None
        sum_153: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1], True, dtype = torch.float32);  mul_318 = None
        view_333: "f32[768]" = torch.ops.aten.view.default(sum_153, _shape_param_356);  sum_153 = _shape_param_356 = None
        neg_49: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_317)
        sum_154: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_49, [2], True, dtype = torch.float32);  neg_49 = None
        add_118: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_114, mul_317);  add_114 = mul_317 = None
        mul_319: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg298_1, 2)
        div_118: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_152, mul_319);  sum_152 = mul_319 = None
        eq_20: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg298_1, 0);  arg298_1 = None
        where_29: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_20, full_1, div_118);  eq_20 = div_118 = None
        mul_320: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_29, 0.002607561929595828);  where_29 = None
        mul_321: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_320, arg297_1);  mul_320 = arg297_1 = None
        add_119: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_118, mul_321);  add_118 = mul_321 = None
        expand_19: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_154, _shape_param_357);  sum_154 = _shape_param_357 = None
        div_119: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_19, 768);  expand_19 = None
        add_120: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_119, div_119);  add_119 = div_119 = None
        convert_element_type_360: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg299_1, torch.float32);  arg299_1 = None
        mul_322: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_360, 1.1111111111111112);  convert_element_type_360 = None
        mul_323: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_322);  add_120 = mul_322 = None
        convert_element_type_361: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_323, torch.bfloat16)
        convert_element_type_362: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg300_1, torch.bfloat16);  arg300_1 = None
        mul_324: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_362, 1.1111111111111112);  convert_element_type_362 = None
        mul_325: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_361, mul_324);  convert_element_type_361 = mul_324 = None
        view_334: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_325, _shape_param_358);  mul_325 = _shape_param_358 = None
        mm_124: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_334, arg301_1);  arg301_1 = None
        permute_112: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_125: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_112, arg302_1);  permute_112 = arg302_1 = None
        sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True, dtype = torch.float32);  view_334 = None
        view_335: "f32[768]" = torch.ops.aten.view.default(sum_155, _shape_param_359);  sum_155 = _shape_param_359 = None
        convert_element_type_363: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_335, torch.bfloat16);  view_335 = None
        view_336: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_124, _shape_param_360);  mm_124 = _shape_param_360 = None
        convert_element_type_364: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_365: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_363, torch.float32);  convert_element_type_363 = None
        convert_element_type_366: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.bfloat16);  arg303_1 = None
        mul_326: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_366, 1.1111111111111112);  convert_element_type_366 = None
        mul_327: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_336, mul_326);  view_336 = mul_326 = None
        convert_element_type_367: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_327, torch.float32);  mul_327 = None
        view_337: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg304_1, _shape_param_361);  arg304_1 = _shape_param_361 = None
        convert_element_type_368: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_337, torch.float32);  view_337 = None
        mul_328: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_368, 0.7071067811865476)
        erf_10: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_328);  mul_328 = None
        add_121: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_329: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_121, 0.5);  add_121 = None
        mul_330: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_368, convert_element_type_368)
        mul_331: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_330, -0.5);  mul_330 = None
        exp_22: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_331);  mul_331 = None
        mul_332: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_333: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_368, mul_332);  convert_element_type_368 = mul_332 = None
        add_122: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_329, mul_333);  mul_329 = mul_333 = None
        mul_334: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_367, add_122);  convert_element_type_367 = add_122 = None
        convert_element_type_369: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        view_338: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_369, _shape_param_362);  convert_element_type_369 = _shape_param_362 = None
        mm_126: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_338, arg305_1);  arg305_1 = None
        permute_113: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_127: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_113, arg306_1);  permute_113 = arg306_1 = None
        sum_156: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True, dtype = torch.float32);  view_338 = None
        view_339: "f32[3072]" = torch.ops.aten.view.default(sum_156, _shape_param_363);  sum_156 = _shape_param_363 = None
        convert_element_type_370: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_339, torch.bfloat16);  view_339 = None
        view_340: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_126, _shape_param_364);  mm_126 = _shape_param_364 = None
        convert_element_type_371: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_340, torch.float32);  view_340 = None
        convert_element_type_372: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_373: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_370, torch.float32);  convert_element_type_370 = None
        sum_157: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_371, [0, 1], True, dtype = torch.float32)
        view_341: "f32[768]" = torch.ops.aten.view.default(sum_157, _shape_param_365);  sum_157 = _shape_param_365 = None
        mul_335: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg307_1, arg308_1)
        add_123: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg309_1, 1e-06)
        div_120: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_335, add_123);  mul_335 = None
        div_121: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_120, add_123);  div_120 = None
        neg_50: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_371)
        mul_336: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_50, div_121);  neg_50 = div_121 = None
        div_122: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_371, add_123);  convert_element_type_371 = add_123 = None
        sum_158: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True, dtype = torch.float32);  mul_336 = None
        mul_337: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_122, arg307_1);  arg307_1 = None
        mul_338: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_122, arg308_1);  div_122 = None
        sum_159: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1], True, dtype = torch.float32);  mul_338 = None
        view_342: "f32[768]" = torch.ops.aten.view.default(sum_159, _shape_param_366);  sum_159 = _shape_param_366 = None
        neg_51: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_337)
        sum_160: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_51, [2], True, dtype = torch.float32);  neg_51 = None
        add_124: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_323, mul_337);  mul_323 = mul_337 = None
        mul_339: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg309_1, 2)
        div_123: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_158, mul_339);  sum_158 = mul_339 = None
        eq_21: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg309_1, 0);  arg309_1 = None
        where_30: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_21, full_1, div_123);  eq_21 = div_123 = None
        mul_340: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_30, 0.002607561929595828);  where_30 = None
        mul_341: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_340, arg308_1);  mul_340 = arg308_1 = None
        add_125: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_124, mul_341);  add_124 = mul_341 = None
        expand_20: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_160, _shape_param_367);  sum_160 = _shape_param_367 = None
        div_124: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_20, 768);  expand_20 = None
        add_126: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_125, div_124);  add_125 = div_124 = None
        convert_element_type_374: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_126, torch.bfloat16)
        convert_element_type_375: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.bfloat16);  arg310_1 = None
        mul_342: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_375, 1.1111111111111112);  convert_element_type_375 = None
        mul_343: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_374, mul_342);  convert_element_type_374 = mul_342 = None
        view_343: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_343, _shape_param_368);  mul_343 = _shape_param_368 = None
        mm_128: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_343, arg311_1);  arg311_1 = None
        permute_114: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_343, [1, 0])
        mm_129: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_114, arg312_1);  permute_114 = arg312_1 = None
        sum_161: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True, dtype = torch.float32);  view_343 = None
        view_344: "f32[768]" = torch.ops.aten.view.default(sum_161, _shape_param_369);  sum_161 = _shape_param_369 = None
        convert_element_type_376: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_344, torch.bfloat16);  view_344 = None
        view_345: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_128, _shape_param_370);  mm_128 = _shape_param_370 = None
        convert_element_type_377: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_378: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_376, torch.float32);  convert_element_type_376 = None
        view_346: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_345, _shape_param_371);  view_345 = _shape_param_371 = None
        permute_115: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        clone_40: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_347: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_372);  clone_40 = _shape_param_372 = None
        bmm_40: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg313_1, view_347);  arg313_1 = None
        bmm_41: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_347, arg314_1);  view_347 = arg314_1 = None
        view_348: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_40, _shape_param_373);  bmm_40 = _shape_param_373 = None
        view_349: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_41, _shape_param_374);  bmm_41 = _shape_param_374 = None
        convert_element_type_379: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        convert_element_type_380: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.float32);  arg315_1 = None
        mul_344: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_380, 1.1111111111111112);  convert_element_type_380 = None
        mul_345: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_379, mul_344);  convert_element_type_379 = mul_344 = None
        convert_element_type_381: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.float32);  arg316_1 = None
        sub_12: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_381, arg317_1);  convert_element_type_381 = arg317_1 = None
        exp_23: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        div_125: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, arg318_1);  exp_23 = arg318_1 = None
        mul_346: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_345, div_125);  mul_345 = None
        sum_162: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [-1], True)
        neg_52: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_125);  div_125 = None
        fma_10: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_52, sum_162, mul_346);  neg_52 = sum_162 = mul_346 = None
        convert_element_type_382: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        where_31: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_382);  convert_element_type_382 = None
        div_126: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_31, 8.0);  where_31 = None
        view_350: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_126, _shape_param_375);  div_126 = _shape_param_375 = None
        bmm_42: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg319_1, view_350);  arg319_1 = None
        bmm_43: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_350, arg320_1);  view_350 = arg320_1 = None
        view_351: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_376);  bmm_42 = _shape_param_376 = None
        view_352: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_377);  bmm_43 = _shape_param_377 = None
        permute_116: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_351, [0, 1, 3, 2]);  view_351 = None
        permute_117: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None
        clone_41: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_353: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_41, _shape_param_378);  clone_41 = _shape_param_378 = None
        view_354: "bf16[2048, 768]" = torch.ops.aten.view.default(view_353, _shape_param_379);  view_353 = _shape_param_379 = None
        mm_130: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_354, arg321_1);  arg321_1 = None
        permute_118: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_131: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_118, arg322_1);  permute_118 = None
        sum_163: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_354, [0], True, dtype = torch.float32);  view_354 = None
        view_355: "f32[768]" = torch.ops.aten.view.default(sum_163, _shape_param_380);  sum_163 = _shape_param_380 = None
        convert_element_type_383: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_355, torch.bfloat16);  view_355 = None
        view_356: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_130, _shape_param_381);  mm_130 = _shape_param_381 = None
        convert_element_type_384: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_356, torch.float32);  view_356 = None
        convert_element_type_385: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_386: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_383, torch.float32);  convert_element_type_383 = None
        permute_119: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_116, [0, 2, 1, 3]);  permute_116 = None
        view_357: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_119, _shape_param_382);  permute_119 = _shape_param_382 = None
        clone_42: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_357, memory_format = torch.contiguous_format);  view_357 = None
        view_358: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_42, _shape_param_383);  clone_42 = _shape_param_383 = None
        mm_132: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_358, arg323_1);  arg323_1 = None
        permute_120: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_358, [1, 0])
        mm_133: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_120, arg322_1);  permute_120 = None
        sum_164: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True, dtype = torch.float32);  view_358 = None
        view_359: "f32[768]" = torch.ops.aten.view.default(sum_164, _shape_param_384);  sum_164 = _shape_param_384 = None
        convert_element_type_387: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_359, torch.bfloat16);  view_359 = None
        view_360: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_132, _shape_param_385);  mm_132 = _shape_param_385 = None
        convert_element_type_388: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_360, torch.float32);  view_360 = None
        add_127: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_384, convert_element_type_388);  convert_element_type_384 = convert_element_type_388 = None
        convert_element_type_389: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_390: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_387, torch.float32);  convert_element_type_387 = None
        permute_121: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        clone_43: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_361: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_43, _shape_param_386);  clone_43 = _shape_param_386 = None
        view_362: "bf16[2048, 768]" = torch.ops.aten.view.default(view_361, _shape_param_387);  view_361 = _shape_param_387 = None
        mm_134: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_362, arg324_1);  arg324_1 = None
        permute_122: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_362, [1, 0])
        mm_135: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_122, arg322_1);  permute_122 = arg322_1 = None
        sum_165: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True, dtype = torch.float32);  view_362 = None
        view_363: "f32[768]" = torch.ops.aten.view.default(sum_165, _shape_param_388);  sum_165 = _shape_param_388 = None
        convert_element_type_391: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_363, torch.bfloat16);  view_363 = None
        view_364: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_134, _shape_param_389);  mm_134 = _shape_param_389 = None
        convert_element_type_392: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_364, torch.float32);  view_364 = None
        add_128: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_127, convert_element_type_392);  add_127 = convert_element_type_392 = None
        convert_element_type_393: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_394: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_391, torch.float32);  convert_element_type_391 = None
        sum_166: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1], True, dtype = torch.float32)
        view_365: "f32[768]" = torch.ops.aten.view.default(sum_166, _shape_param_390);  sum_166 = _shape_param_390 = None
        mul_347: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg325_1, arg326_1)
        add_129: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg327_1, 1e-06)
        div_127: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_347, add_129);  mul_347 = None
        div_128: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_127, add_129);  div_127 = None
        neg_53: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_128)
        mul_348: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_53, div_128);  neg_53 = div_128 = None
        div_129: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_128, add_129);  add_128 = add_129 = None
        sum_167: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True, dtype = torch.float32);  mul_348 = None
        mul_349: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_129, arg325_1);  arg325_1 = None
        mul_350: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_129, arg326_1);  div_129 = None
        sum_168: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1], True, dtype = torch.float32);  mul_350 = None
        view_366: "f32[768]" = torch.ops.aten.view.default(sum_168, _shape_param_391);  sum_168 = _shape_param_391 = None
        neg_54: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_349)
        sum_169: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_54, [2], True, dtype = torch.float32);  neg_54 = None
        add_130: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_126, mul_349);  add_126 = mul_349 = None
        mul_351: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg327_1, 2)
        div_130: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_167, mul_351);  sum_167 = mul_351 = None
        eq_22: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg327_1, 0);  arg327_1 = None
        where_32: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_22, full_1, div_130);  eq_22 = div_130 = None
        mul_352: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_32, 0.002607561929595828);  where_32 = None
        mul_353: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_352, arg326_1);  mul_352 = arg326_1 = None
        add_131: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_130, mul_353);  add_130 = mul_353 = None
        expand_21: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_169, _shape_param_392);  sum_169 = _shape_param_392 = None
        div_131: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_21, 768);  expand_21 = None
        add_132: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_131, div_131);  add_131 = div_131 = None
        convert_element_type_395: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.float32);  arg328_1 = None
        mul_354: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_395, 1.1111111111111112);  convert_element_type_395 = None
        mul_355: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_132, mul_354);  add_132 = mul_354 = None
        convert_element_type_396: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(mul_355, torch.bfloat16)
        convert_element_type_397: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg329_1, torch.bfloat16);  arg329_1 = None
        mul_356: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_397, 1.1111111111111112);  convert_element_type_397 = None
        mul_357: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_396, mul_356);  convert_element_type_396 = mul_356 = None
        view_367: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_357, _shape_param_393);  mul_357 = _shape_param_393 = None
        mm_136: "bf16[2048, 3072]" = torch.ops.aten.mm.default(view_367, arg330_1);  arg330_1 = None
        permute_123: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_137: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_123, arg331_1);  permute_123 = arg331_1 = None
        sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True, dtype = torch.float32);  view_367 = None
        view_368: "f32[768]" = torch.ops.aten.view.default(sum_170, _shape_param_394);  sum_170 = _shape_param_394 = None
        convert_element_type_398: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_368, torch.bfloat16);  view_368 = None
        view_369: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(mm_136, _shape_param_395);  mm_136 = _shape_param_395 = None
        convert_element_type_399: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_400: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_398, torch.float32);  convert_element_type_398 = None
        convert_element_type_401: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(arg332_1, torch.bfloat16);  arg332_1 = None
        mul_358: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_401, 1.1111111111111112);  convert_element_type_401 = None
        mul_359: "bf16[16, 128, 3072]" = torch.ops.aten.mul.Tensor(view_369, mul_358);  view_369 = mul_358 = None
        convert_element_type_402: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_359, torch.float32);  mul_359 = None
        view_370: "bf16[16, 128, 3072]" = torch.ops.aten.view.default(arg333_1, _shape_param_396);  arg333_1 = _shape_param_396 = None
        convert_element_type_403: "f32[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(view_370, torch.float32);  view_370 = None
        mul_360: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_403, 0.7071067811865476)
        erf_11: "f32[16, 128, 3072]" = torch.ops.aten.erf.default(mul_360);  mul_360 = None
        add_133: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_361: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(add_133, 0.5);  add_133 = None
        mul_362: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_403, convert_element_type_403)
        mul_363: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_362, -0.5);  mul_362 = None
        exp_24: "f32[16, 128, 3072]" = torch.ops.aten.exp.default(mul_363);  mul_363 = None
        mul_364: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_365: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_403, mul_364);  convert_element_type_403 = mul_364 = None
        add_134: "f32[16, 128, 3072]" = torch.ops.aten.add.Tensor(mul_361, mul_365);  mul_361 = mul_365 = None
        mul_366: "f32[16, 128, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_402, add_134);  convert_element_type_402 = add_134 = None
        convert_element_type_404: "bf16[16, 128, 3072]" = torch.ops.prims.convert_element_type.default(mul_366, torch.bfloat16);  mul_366 = None
        view_371: "bf16[2048, 3072]" = torch.ops.aten.view.default(convert_element_type_404, _shape_param_397);  convert_element_type_404 = _shape_param_397 = None
        mm_138: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_371, arg334_1);  arg334_1 = None
        permute_124: "bf16[3072, 2048]" = torch.ops.aten.permute.default(view_371, [1, 0])
        mm_139: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_124, arg335_1);  permute_124 = arg335_1 = None
        sum_171: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True, dtype = torch.float32);  view_371 = None
        view_372: "f32[3072]" = torch.ops.aten.view.default(sum_171, _shape_param_398);  sum_171 = _shape_param_398 = None
        convert_element_type_405: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_372, torch.bfloat16);  view_372 = None
        view_373: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_138, _shape_param_399);  mm_138 = _shape_param_399 = None
        convert_element_type_406: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_373, torch.float32);  view_373 = None
        convert_element_type_407: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_408: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_405, torch.float32);  convert_element_type_405 = None
        sum_172: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(convert_element_type_406, [0, 1], True, dtype = torch.float32)
        view_374: "f32[768]" = torch.ops.aten.view.default(sum_172, _shape_param_400);  sum_172 = _shape_param_400 = None
        mul_367: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg336_1, arg337_1)
        add_135: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg338_1, 1e-06)
        div_132: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_367, add_135);  mul_367 = None
        div_133: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_132, add_135);  div_132 = None
        neg_55: "f32[16, 128, 768]" = torch.ops.aten.neg.default(convert_element_type_406)
        mul_368: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_55, div_133);  neg_55 = div_133 = None
        div_134: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(convert_element_type_406, add_135);  convert_element_type_406 = add_135 = None
        sum_173: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [2], True, dtype = torch.float32);  mul_368 = None
        mul_369: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_134, arg336_1);  arg336_1 = None
        mul_370: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_134, arg337_1);  div_134 = None
        sum_174: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1], True, dtype = torch.float32);  mul_370 = None
        view_375: "f32[768]" = torch.ops.aten.view.default(sum_174, _shape_param_401);  sum_174 = _shape_param_401 = None
        neg_56: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_369)
        sum_175: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_56, [2], True, dtype = torch.float32);  neg_56 = None
        add_136: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(mul_355, mul_369);  mul_355 = mul_369 = None
        mul_371: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg338_1, 2)
        div_135: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_173, mul_371);  sum_173 = mul_371 = None
        eq_23: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg338_1, 0);  arg338_1 = None
        where_33: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_23, full_1, div_135);  eq_23 = div_135 = None
        mul_372: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_33, 0.002607561929595828);  where_33 = None
        mul_373: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_372, arg337_1);  mul_372 = arg337_1 = None
        add_137: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_136, mul_373);  add_136 = mul_373 = None
        expand_22: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_175, _shape_param_402);  sum_175 = _shape_param_402 = None
        div_136: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_22, 768);  expand_22 = None
        add_138: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_137, div_136);  add_137 = div_136 = None
        convert_element_type_409: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16)
        convert_element_type_410: "bf16[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg339_1, torch.bfloat16);  arg339_1 = None
        mul_374: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_410, 1.1111111111111112);  convert_element_type_410 = None
        mul_375: "bf16[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_409, mul_374);  convert_element_type_409 = mul_374 = None
        view_376: "bf16[2048, 768]" = torch.ops.aten.view.default(mul_375, _shape_param_403);  mul_375 = _shape_param_403 = None
        mm_140: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_376, arg340_1);  arg340_1 = None
        permute_125: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_376, [1, 0])
        mm_141: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_125, arg341_1);  permute_125 = arg341_1 = None
        sum_176: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_376, [0], True, dtype = torch.float32);  view_376 = None
        view_377: "f32[768]" = torch.ops.aten.view.default(sum_176, _shape_param_404);  sum_176 = _shape_param_404 = None
        convert_element_type_411: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_377, torch.bfloat16);  view_377 = None
        view_378: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_140, _shape_param_405);  mm_140 = _shape_param_405 = None
        convert_element_type_412: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_413: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_411, torch.float32);  convert_element_type_411 = None
        view_379: "bf16[16, 128, 12, 64]" = torch.ops.aten.view.default(view_378, _shape_param_406);  view_378 = _shape_param_406 = None
        permute_126: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        clone_44: "bf16[16, 12, 128, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_380: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_407);  clone_44 = _shape_param_407 = None
        bmm_44: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg342_1, view_380);  arg342_1 = None
        bmm_45: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_380, arg343_1);  view_380 = arg343_1 = None
        view_381: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_408);  bmm_44 = _shape_param_408 = None
        view_382: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(bmm_45, _shape_param_409);  bmm_45 = _shape_param_409 = None
        convert_element_type_414: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        convert_element_type_415: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.float32);  arg344_1 = None
        mul_376: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_415, 1.1111111111111112);  convert_element_type_415 = None
        mul_377: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_414, mul_376);  convert_element_type_414 = mul_376 = None
        view_383: "bf16[16, 12, 128, 128]" = torch.ops.aten.view.default(arg345_1, _shape_param_410);  arg345_1 = _shape_param_410 = None
        div_137: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_383, 8.0);  view_383 = None
        full_3: "bf16[]" = torch.ops.aten.full.default([], -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_3, div_137);  full_3 = div_137 = None
        convert_element_type_416: "f32[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(where_34, torch.float32);  where_34 = None
        sub_13: "f32[16, 12, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_416, arg346_1);  convert_element_type_416 = arg346_1 = None
        exp_25: "f32[16, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        div_138: "f32[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_25, arg347_1);  exp_25 = arg347_1 = None
        mul_378: "f32[16, 12, 128, 128]" = torch.ops.aten.mul.Tensor(mul_377, div_138);  mul_377 = None
        sum_177: "f32[16, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [-1], True)
        neg_57: "f32[16, 12, 128, 128]" = torch.ops.aten.neg.default(div_138);  div_138 = None
        fma_11: "f32[16, 12, 128, 128]" = torch.ops.prims.fma.default(neg_57, sum_177, mul_378);  neg_57 = sum_177 = mul_378 = None
        convert_element_type_417: "bf16[16, 12, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        where_35: "bf16[16, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, full_2, convert_element_type_417);  eq_1 = full_2 = convert_element_type_417 = None
        div_139: "bf16[16, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_35, 8.0);  where_35 = None
        view_384: "bf16[192, 128, 128]" = torch.ops.aten.view.default(div_139, _shape_param_411);  div_139 = _shape_param_411 = None
        bmm_46: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg348_1, view_384);  arg348_1 = None
        bmm_47: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_384, arg349_1);  view_384 = arg349_1 = None
        view_385: "bf16[16, 12, 64, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_412);  bmm_46 = _shape_param_412 = None
        view_386: "bf16[16, 12, 128, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_413);  bmm_47 = _shape_param_413 = None
        permute_127: "bf16[16, 12, 128, 64]" = torch.ops.aten.permute.default(view_385, [0, 1, 3, 2]);  view_385 = None
        permute_128: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
        clone_45: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_387: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_45, _shape_param_414);  clone_45 = _shape_param_414 = None
        view_388: "bf16[2048, 768]" = torch.ops.aten.view.default(view_387, _shape_param_415);  view_387 = _shape_param_415 = None
        mm_142: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_388, arg350_1);  arg350_1 = None
        permute_129: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_388, [1, 0])
        mm_143: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_129, arg351_1);  permute_129 = None
        sum_178: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True, dtype = torch.float32);  view_388 = None
        view_389: "f32[768]" = torch.ops.aten.view.default(sum_178, _shape_param_416);  sum_178 = _shape_param_416 = None
        convert_element_type_418: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_389, torch.bfloat16);  view_389 = None
        view_390: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_142, _shape_param_417);  mm_142 = _shape_param_417 = None
        convert_element_type_419: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_390, torch.float32);  view_390 = None
        convert_element_type_420: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_421: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_418, torch.float32);  convert_element_type_418 = None
        permute_130: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(permute_127, [0, 2, 1, 3]);  permute_127 = None
        view_391: "bf16[16, 128, 768]" = torch.ops.aten.view.default(permute_130, _shape_param_418);  permute_130 = _shape_param_418 = None
        clone_46: "bf16[16, 128, 768]" = torch.ops.aten.clone.default(view_391, memory_format = torch.contiguous_format);  view_391 = None
        view_392: "bf16[2048, 768]" = torch.ops.aten.view.default(clone_46, _shape_param_419);  clone_46 = _shape_param_419 = None
        mm_144: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_392, arg352_1);  arg352_1 = None
        permute_131: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_145: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_131, arg351_1);  permute_131 = None
        sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True, dtype = torch.float32);  view_392 = None
        view_393: "f32[768]" = torch.ops.aten.view.default(sum_179, _shape_param_420);  sum_179 = _shape_param_420 = None
        convert_element_type_422: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_393, torch.bfloat16);  view_393 = None
        view_394: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_144, _shape_param_421);  mm_144 = _shape_param_421 = None
        convert_element_type_423: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_394, torch.float32);  view_394 = None
        add_139: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(convert_element_type_419, convert_element_type_423);  convert_element_type_419 = convert_element_type_423 = None
        convert_element_type_424: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_425: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_422, torch.float32);  convert_element_type_422 = None
        permute_132: "bf16[16, 128, 12, 64]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
        clone_47: "bf16[16, 128, 12, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_395: "bf16[16, 128, 768]" = torch.ops.aten.view.default(clone_47, _shape_param_422);  clone_47 = _shape_param_422 = None
        view_396: "bf16[2048, 768]" = torch.ops.aten.view.default(view_395, _shape_param_423);  view_395 = _shape_param_423 = None
        mm_146: "bf16[2048, 768]" = torch.ops.aten.mm.default(view_396, arg353_1);  arg353_1 = None
        permute_133: "bf16[768, 2048]" = torch.ops.aten.permute.default(view_396, [1, 0])
        mm_147: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_133, arg351_1);  permute_133 = arg351_1 = None
        sum_180: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_396, [0], True, dtype = torch.float32);  view_396 = None
        view_397: "f32[768]" = torch.ops.aten.view.default(sum_180, _shape_param_424);  sum_180 = _shape_param_424 = None
        convert_element_type_426: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_397, torch.bfloat16);  view_397 = None
        view_398: "bf16[16, 128, 768]" = torch.ops.aten.view.default(mm_146, _shape_param_425);  mm_146 = _shape_param_425 = None
        convert_element_type_427: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        add_140: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_139, convert_element_type_427);  add_139 = convert_element_type_427 = None
        convert_element_type_428: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_429: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_426, torch.float32);  convert_element_type_426 = None
        sum_181: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1], True, dtype = torch.float32)
        view_399: "f32[768]" = torch.ops.aten.view.default(sum_181, _shape_param_426);  sum_181 = _shape_param_426 = None
        mul_379: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(arg354_1, arg355_1)
        add_141: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(arg356_1, 1e-06)
        div_140: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(mul_379, add_141);  mul_379 = None
        div_141: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(div_140, add_141);  div_140 = None
        neg_58: "f32[16, 128, 768]" = torch.ops.aten.neg.default(add_140)
        mul_380: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(neg_58, div_141);  neg_58 = div_141 = None
        div_142: "f32[16, 128, 768]" = torch.ops.aten.div.Tensor(add_140, add_141);  add_140 = add_141 = None
        sum_182: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True, dtype = torch.float32);  mul_380 = None
        mul_381: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_142, arg354_1);  arg354_1 = None
        mul_382: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(div_142, arg355_1);  div_142 = None
        sum_183: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1], True, dtype = torch.float32);  mul_382 = None
        view_400: "f32[768]" = torch.ops.aten.view.default(sum_183, _shape_param_427);  sum_183 = _shape_param_427 = None
        neg_59: "f32[16, 128, 768]" = torch.ops.aten.neg.default(mul_381)
        sum_184: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_59, [2], True, dtype = torch.float32);  neg_59 = None
        add_142: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_138, mul_381);  add_138 = mul_381 = None
        mul_383: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(arg356_1, 2)
        div_143: "f32[16, 128, 1]" = torch.ops.aten.div.Tensor(sum_182, mul_383);  sum_182 = mul_383 = None
        eq_24: "b8[16, 128, 1]" = torch.ops.aten.eq.Scalar(arg356_1, 0);  arg356_1 = None
        where_36: "f32[16, 128, 1]" = torch.ops.aten.where.self(eq_24, full_1, div_143);  eq_24 = full_1 = div_143 = None
        mul_384: "f32[16, 128, 1]" = torch.ops.aten.mul.Scalar(where_36, 0.002607561929595828);  where_36 = None
        mul_385: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(mul_384, arg355_1);  mul_384 = arg355_1 = None
        add_143: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_142, mul_385);  add_142 = mul_385 = None
        expand_23: "f32[16, 128, 768]" = torch.ops.aten.expand.default(sum_184, _shape_param_428);  sum_184 = _shape_param_428 = None
        div_144: "f32[16, 128, 768]" = torch.ops.aten.div.Scalar(expand_23, 768);  expand_23 = None
        add_144: "f32[16, 128, 768]" = torch.ops.aten.add.Tensor(add_143, div_144);  add_143 = div_144 = None
        convert_element_type_430: "f32[16, 128, 768]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.float32);  arg357_1 = None
        mul_386: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_430, 1.1111111111111112);  convert_element_type_430 = None
        mul_387: "f32[16, 128, 768]" = torch.ops.aten.mul.Tensor(add_144, mul_386);  add_144 = mul_386 = None
        ge: "b8[16, 128]" = torch.ops.aten.ge.Scalar(arg358_1, 0)
        lt: "b8[16, 128]" = torch.ops.aten.lt.Scalar(arg358_1, 3)
        bitwise_and: "b8[16, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[16, 128]" = torch.ops.aten.ne.Scalar(arg358_1, 0)
        bitwise_and_1: "b8[16, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne);  bitwise_and = ne = None
        unsqueeze: "b8[16, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_4: "f32[3, 768]" = torch.ops.aten.full.default(_shape_param_429, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_429 = None
        _unsafe_masked_index_put_accumulate: "f32[3, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze, [arg358_1], mul_387);  full_4 = unsqueeze = arg358_1 = None
        ge_1: "b8[16, 128]" = torch.ops.aten.ge.Scalar(arg359_1, 0)
        lt_1: "b8[16, 128]" = torch.ops.aten.lt.Scalar(arg359_1, 20005)
        bitwise_and_2: "b8[16, 128]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_1: "b8[16, 128]" = torch.ops.aten.ne.Scalar(arg359_1, 0)
        bitwise_and_3: "b8[16, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_1: "b8[16, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_5: "f32[20005, 768]" = torch.ops.aten.full.default(_shape_param_430, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_430 = None
        _unsafe_masked_index_put_accumulate_1: "f32[20005, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_5, unsqueeze_1, [arg359_1], mul_387);  full_5 = unsqueeze_1 = arg359_1 = mul_387 = None
        return (convert_element_type_3, convert_element_type_4, convert_element_type_8, convert_element_type_9, convert_element_type_14, convert_element_type_15, convert_element_type_22, convert_element_type_23, view_11, view_12, convert_element_type_27, convert_element_type_28, convert_element_type_35, convert_element_type_36, convert_element_type_39, convert_element_type_40, convert_element_type_43, convert_element_type_44, view_35, view_36, convert_element_type_49, convert_element_type_50, convert_element_type_57, convert_element_type_58, view_44, view_45, convert_element_type_62, convert_element_type_63, convert_element_type_70, convert_element_type_71, convert_element_type_74, convert_element_type_75, convert_element_type_78, convert_element_type_79, view_68, view_69, convert_element_type_84, convert_element_type_85, convert_element_type_92, convert_element_type_93, view_77, view_78, convert_element_type_97, convert_element_type_98, convert_element_type_105, convert_element_type_106, convert_element_type_109, convert_element_type_110, convert_element_type_113, convert_element_type_114, view_101, view_102, convert_element_type_119, convert_element_type_120, convert_element_type_127, convert_element_type_128, view_110, view_111, convert_element_type_132, convert_element_type_133, convert_element_type_140, convert_element_type_141, convert_element_type_144, convert_element_type_145, convert_element_type_148, convert_element_type_149, view_134, view_135, convert_element_type_154, convert_element_type_155, convert_element_type_162, convert_element_type_163, view_143, view_144, convert_element_type_167, convert_element_type_168, convert_element_type_175, convert_element_type_176, convert_element_type_179, convert_element_type_180, convert_element_type_183, convert_element_type_184, view_167, view_168, convert_element_type_189, convert_element_type_190, convert_element_type_197, convert_element_type_198, view_176, view_177, convert_element_type_202, convert_element_type_203, convert_element_type_210, convert_element_type_211, convert_element_type_214, convert_element_type_215, convert_element_type_218, convert_element_type_219, view_200, view_201, convert_element_type_224, convert_element_type_225, convert_element_type_232, convert_element_type_233, view_209, view_210, convert_element_type_237, convert_element_type_238, convert_element_type_245, convert_element_type_246, convert_element_type_249, convert_element_type_250, convert_element_type_253, convert_element_type_254, view_233, view_234, convert_element_type_259, convert_element_type_260, convert_element_type_267, convert_element_type_268, view_242, view_243, convert_element_type_272, convert_element_type_273, convert_element_type_280, convert_element_type_281, convert_element_type_284, convert_element_type_285, convert_element_type_288, convert_element_type_289, view_266, view_267, convert_element_type_294, convert_element_type_295, convert_element_type_302, convert_element_type_303, view_275, view_276, convert_element_type_307, convert_element_type_308, convert_element_type_315, convert_element_type_316, convert_element_type_319, convert_element_type_320, convert_element_type_323, convert_element_type_324, view_299, view_300, convert_element_type_329, convert_element_type_330, convert_element_type_337, convert_element_type_338, view_308, view_309, convert_element_type_342, convert_element_type_343, convert_element_type_350, convert_element_type_351, convert_element_type_354, convert_element_type_355, convert_element_type_358, convert_element_type_359, view_332, view_333, convert_element_type_364, convert_element_type_365, convert_element_type_372, convert_element_type_373, view_341, view_342, convert_element_type_377, convert_element_type_378, convert_element_type_385, convert_element_type_386, convert_element_type_389, convert_element_type_390, convert_element_type_393, convert_element_type_394, view_365, view_366, convert_element_type_399, convert_element_type_400, convert_element_type_407, convert_element_type_408, view_374, view_375, convert_element_type_412, convert_element_type_413, convert_element_type_420, convert_element_type_421, convert_element_type_424, convert_element_type_425, convert_element_type_428, convert_element_type_429, view_399, view_400, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1)



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
