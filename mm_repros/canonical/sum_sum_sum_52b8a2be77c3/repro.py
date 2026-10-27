"""
Standalone repro captured via capture_hook.
Label: LayoutLMForMaskedLM
Pattern hash: 52b8a2be77c3
Shape hash: 2241007a
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[32, 512]", arg3_1: "bf16[32, 512, 30522]", arg4_1: "f32[16384, 1]", arg5_1: "f32[16384, 1]", arg6_1: "bf16[32, 512, 30522]", arg7_1: "bf16[30522, 768]", arg8_1: "bf16[16384, 768]", arg9_1: "f32[768]", arg10_1: "bf16[16384, 768]", arg11_1: "f32[32, 512, 1]", arg12_1: "f32[32, 512, 1]", arg13_1: "bf16[768, 768]", arg14_1: "bf16[16384, 768]", arg15_1: "f32[768]", arg16_1: "f32[32, 512, 768]", arg17_1: "f32[32, 512, 1]", arg18_1: "b8[32, 512, 768]", arg19_1: "bf16[768, 3072]", arg20_1: "bf16[16384, 3072]", arg21_1: "bf16[16384, 3072]", arg22_1: "bf16[3072, 768]", arg23_1: "bf16[16384, 768]", arg24_1: "f32[768]", arg25_1: "f32[32, 512, 768]", arg26_1: "f32[32, 512, 1]", arg27_1: "b8[32, 512, 768]", arg28_1: "bf16[768, 768]", arg29_1: "bf16[16384, 768]", arg30_1: "bf16[384, 512, 512]", arg31_1: "bf16[384, 64, 512]", arg32_1: "b8[32, 12, 512, 512]", arg33_1: "bf16[384, 512, 512]", arg34_1: "f32[32, 12, 512, 1]", arg35_1: "f32[32, 12, 512, 1]", arg36_1: "b8[32, 12, 512, 1]", arg37_1: "f32[32, 12, 512, 1]", arg38_1: "bf16[384, 64, 512]", arg39_1: "bf16[384, 512, 64]", arg40_1: "bf16[768, 768]", arg41_1: "bf16[16384, 768]", arg42_1: "bf16[768, 768]", arg43_1: "bf16[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[32, 512, 768]", arg46_1: "f32[32, 512, 1]", arg47_1: "b8[32, 512, 768]", arg48_1: "bf16[768, 3072]", arg49_1: "bf16[16384, 3072]", arg50_1: "bf16[16384, 3072]", arg51_1: "bf16[3072, 768]", arg52_1: "bf16[16384, 768]", arg53_1: "f32[768]", arg54_1: "f32[32, 512, 768]", arg55_1: "f32[32, 512, 1]", arg56_1: "b8[32, 512, 768]", arg57_1: "bf16[768, 768]", arg58_1: "bf16[16384, 768]", arg59_1: "bf16[384, 512, 512]", arg60_1: "bf16[384, 64, 512]", arg61_1: "b8[32, 12, 512, 512]", arg62_1: "bf16[384, 512, 512]", arg63_1: "f32[32, 12, 512, 1]", arg64_1: "f32[32, 12, 512, 1]", arg65_1: "b8[32, 12, 512, 1]", arg66_1: "f32[32, 12, 512, 1]", arg67_1: "bf16[384, 64, 512]", arg68_1: "bf16[384, 512, 64]", arg69_1: "bf16[768, 768]", arg70_1: "bf16[16384, 768]", arg71_1: "bf16[768, 768]", arg72_1: "bf16[768, 768]", arg73_1: "f32[768]", arg74_1: "f32[32, 512, 768]", arg75_1: "f32[32, 512, 1]", arg76_1: "b8[32, 512, 768]", arg77_1: "bf16[768, 3072]", arg78_1: "bf16[16384, 3072]", arg79_1: "bf16[16384, 3072]", arg80_1: "bf16[3072, 768]", arg81_1: "bf16[16384, 768]", arg82_1: "f32[768]", arg83_1: "f32[32, 512, 768]", arg84_1: "f32[32, 512, 1]", arg85_1: "b8[32, 512, 768]", arg86_1: "bf16[768, 768]", arg87_1: "bf16[16384, 768]", arg88_1: "bf16[384, 512, 512]", arg89_1: "bf16[384, 64, 512]", arg90_1: "b8[32, 12, 512, 512]", arg91_1: "bf16[384, 512, 512]", arg92_1: "f32[32, 12, 512, 1]", arg93_1: "f32[32, 12, 512, 1]", arg94_1: "b8[32, 12, 512, 1]", arg95_1: "f32[32, 12, 512, 1]", arg96_1: "bf16[384, 64, 512]", arg97_1: "bf16[384, 512, 64]", arg98_1: "bf16[768, 768]", arg99_1: "bf16[16384, 768]", arg100_1: "bf16[768, 768]", arg101_1: "bf16[768, 768]", arg102_1: "f32[768]", arg103_1: "f32[32, 512, 768]", arg104_1: "f32[32, 512, 1]", arg105_1: "b8[32, 512, 768]", arg106_1: "bf16[768, 3072]", arg107_1: "bf16[16384, 3072]", arg108_1: "bf16[16384, 3072]", arg109_1: "bf16[3072, 768]", arg110_1: "bf16[16384, 768]", arg111_1: "f32[768]", arg112_1: "f32[32, 512, 768]", arg113_1: "f32[32, 512, 1]", arg114_1: "b8[32, 512, 768]", arg115_1: "bf16[768, 768]", arg116_1: "bf16[16384, 768]", arg117_1: "bf16[384, 512, 512]", arg118_1: "bf16[384, 64, 512]", arg119_1: "b8[32, 12, 512, 512]", arg120_1: "bf16[384, 512, 512]", arg121_1: "f32[32, 12, 512, 1]", arg122_1: "f32[32, 12, 512, 1]", arg123_1: "b8[32, 12, 512, 1]", arg124_1: "f32[32, 12, 512, 1]", arg125_1: "bf16[384, 64, 512]", arg126_1: "bf16[384, 512, 64]", arg127_1: "bf16[768, 768]", arg128_1: "bf16[16384, 768]", arg129_1: "bf16[768, 768]", arg130_1: "bf16[768, 768]", arg131_1: "f32[768]", arg132_1: "f32[32, 512, 768]", arg133_1: "f32[32, 512, 1]", arg134_1: "b8[32, 512, 768]", arg135_1: "bf16[768, 3072]", arg136_1: "bf16[16384, 3072]", arg137_1: "bf16[16384, 3072]", arg138_1: "bf16[3072, 768]", arg139_1: "bf16[16384, 768]", arg140_1: "f32[768]", arg141_1: "f32[32, 512, 768]", arg142_1: "f32[32, 512, 1]", arg143_1: "b8[32, 512, 768]", arg144_1: "bf16[768, 768]", arg145_1: "bf16[16384, 768]", arg146_1: "bf16[384, 512, 512]", arg147_1: "bf16[384, 64, 512]", arg148_1: "b8[32, 12, 512, 512]", arg149_1: "bf16[384, 512, 512]", arg150_1: "f32[32, 12, 512, 1]", arg151_1: "f32[32, 12, 512, 1]", arg152_1: "b8[32, 12, 512, 1]", arg153_1: "f32[32, 12, 512, 1]", arg154_1: "bf16[384, 64, 512]", arg155_1: "bf16[384, 512, 64]", arg156_1: "bf16[768, 768]", arg157_1: "bf16[16384, 768]", arg158_1: "bf16[768, 768]", arg159_1: "bf16[768, 768]", arg160_1: "f32[768]", arg161_1: "f32[32, 512, 768]", arg162_1: "f32[32, 512, 1]", arg163_1: "b8[32, 512, 768]", arg164_1: "bf16[768, 3072]", arg165_1: "bf16[16384, 3072]", arg166_1: "bf16[16384, 3072]", arg167_1: "bf16[3072, 768]", arg168_1: "bf16[16384, 768]", arg169_1: "f32[768]", arg170_1: "f32[32, 512, 768]", arg171_1: "f32[32, 512, 1]", arg172_1: "b8[32, 512, 768]", arg173_1: "bf16[768, 768]", arg174_1: "bf16[16384, 768]", arg175_1: "bf16[384, 512, 512]", arg176_1: "bf16[384, 64, 512]", arg177_1: "b8[32, 12, 512, 512]", arg178_1: "bf16[384, 512, 512]", arg179_1: "f32[32, 12, 512, 1]", arg180_1: "f32[32, 12, 512, 1]", arg181_1: "b8[32, 12, 512, 1]", arg182_1: "f32[32, 12, 512, 1]", arg183_1: "bf16[384, 64, 512]", arg184_1: "bf16[384, 512, 64]", arg185_1: "bf16[768, 768]", arg186_1: "bf16[16384, 768]", arg187_1: "bf16[768, 768]", arg188_1: "bf16[768, 768]", arg189_1: "f32[768]", arg190_1: "f32[32, 512, 768]", arg191_1: "f32[32, 512, 1]", arg192_1: "b8[32, 512, 768]", arg193_1: "bf16[768, 3072]", arg194_1: "bf16[16384, 3072]", arg195_1: "bf16[16384, 3072]", arg196_1: "bf16[3072, 768]", arg197_1: "bf16[16384, 768]", arg198_1: "f32[768]", arg199_1: "f32[32, 512, 768]", arg200_1: "f32[32, 512, 1]", arg201_1: "b8[32, 512, 768]", arg202_1: "bf16[768, 768]", arg203_1: "bf16[16384, 768]", arg204_1: "bf16[384, 512, 512]", arg205_1: "bf16[384, 64, 512]", arg206_1: "b8[32, 12, 512, 512]", arg207_1: "bf16[384, 512, 512]", arg208_1: "f32[32, 12, 512, 1]", arg209_1: "f32[32, 12, 512, 1]", arg210_1: "b8[32, 12, 512, 1]", arg211_1: "f32[32, 12, 512, 1]", arg212_1: "bf16[384, 64, 512]", arg213_1: "bf16[384, 512, 64]", arg214_1: "bf16[768, 768]", arg215_1: "bf16[16384, 768]", arg216_1: "bf16[768, 768]", arg217_1: "bf16[768, 768]", arg218_1: "f32[768]", arg219_1: "f32[32, 512, 768]", arg220_1: "f32[32, 512, 1]", arg221_1: "b8[32, 512, 768]", arg222_1: "bf16[768, 3072]", arg223_1: "bf16[16384, 3072]", arg224_1: "bf16[16384, 3072]", arg225_1: "bf16[3072, 768]", arg226_1: "bf16[16384, 768]", arg227_1: "f32[768]", arg228_1: "f32[32, 512, 768]", arg229_1: "f32[32, 512, 1]", arg230_1: "b8[32, 512, 768]", arg231_1: "bf16[768, 768]", arg232_1: "bf16[16384, 768]", arg233_1: "bf16[384, 512, 512]", arg234_1: "bf16[384, 64, 512]", arg235_1: "b8[32, 12, 512, 512]", arg236_1: "bf16[384, 512, 512]", arg237_1: "f32[32, 12, 512, 1]", arg238_1: "f32[32, 12, 512, 1]", arg239_1: "b8[32, 12, 512, 1]", arg240_1: "f32[32, 12, 512, 1]", arg241_1: "bf16[384, 64, 512]", arg242_1: "bf16[384, 512, 64]", arg243_1: "bf16[768, 768]", arg244_1: "bf16[16384, 768]", arg245_1: "bf16[768, 768]", arg246_1: "bf16[768, 768]", arg247_1: "f32[768]", arg248_1: "f32[32, 512, 768]", arg249_1: "f32[32, 512, 1]", arg250_1: "b8[32, 512, 768]", arg251_1: "bf16[768, 3072]", arg252_1: "bf16[16384, 3072]", arg253_1: "bf16[16384, 3072]", arg254_1: "bf16[3072, 768]", arg255_1: "bf16[16384, 768]", arg256_1: "f32[768]", arg257_1: "f32[32, 512, 768]", arg258_1: "f32[32, 512, 1]", arg259_1: "b8[32, 512, 768]", arg260_1: "bf16[768, 768]", arg261_1: "bf16[16384, 768]", arg262_1: "bf16[384, 512, 512]", arg263_1: "bf16[384, 64, 512]", arg264_1: "b8[32, 12, 512, 512]", arg265_1: "bf16[384, 512, 512]", arg266_1: "f32[32, 12, 512, 1]", arg267_1: "f32[32, 12, 512, 1]", arg268_1: "b8[32, 12, 512, 1]", arg269_1: "f32[32, 12, 512, 1]", arg270_1: "bf16[384, 64, 512]", arg271_1: "bf16[384, 512, 64]", arg272_1: "bf16[768, 768]", arg273_1: "bf16[16384, 768]", arg274_1: "bf16[768, 768]", arg275_1: "bf16[768, 768]", arg276_1: "f32[768]", arg277_1: "f32[32, 512, 768]", arg278_1: "f32[32, 512, 1]", arg279_1: "b8[32, 512, 768]", arg280_1: "bf16[768, 3072]", arg281_1: "bf16[16384, 3072]", arg282_1: "bf16[16384, 3072]", arg283_1: "bf16[3072, 768]", arg284_1: "bf16[16384, 768]", arg285_1: "f32[768]", arg286_1: "f32[32, 512, 768]", arg287_1: "f32[32, 512, 1]", arg288_1: "b8[32, 512, 768]", arg289_1: "bf16[768, 768]", arg290_1: "bf16[16384, 768]", arg291_1: "bf16[384, 512, 512]", arg292_1: "bf16[384, 64, 512]", arg293_1: "b8[32, 12, 512, 512]", arg294_1: "bf16[384, 512, 512]", arg295_1: "f32[32, 12, 512, 1]", arg296_1: "f32[32, 12, 512, 1]", arg297_1: "b8[32, 12, 512, 1]", arg298_1: "f32[32, 12, 512, 1]", arg299_1: "bf16[384, 64, 512]", arg300_1: "bf16[384, 512, 64]", arg301_1: "bf16[768, 768]", arg302_1: "bf16[16384, 768]", arg303_1: "bf16[768, 768]", arg304_1: "bf16[768, 768]", arg305_1: "f32[768]", arg306_1: "f32[32, 512, 768]", arg307_1: "f32[32, 512, 1]", arg308_1: "b8[32, 512, 768]", arg309_1: "bf16[768, 3072]", arg310_1: "bf16[16384, 3072]", arg311_1: "bf16[16384, 3072]", arg312_1: "bf16[3072, 768]", arg313_1: "bf16[16384, 768]", arg314_1: "f32[768]", arg315_1: "f32[32, 512, 768]", arg316_1: "f32[32, 512, 1]", arg317_1: "b8[32, 512, 768]", arg318_1: "bf16[768, 768]", arg319_1: "bf16[16384, 768]", arg320_1: "bf16[384, 512, 512]", arg321_1: "bf16[384, 64, 512]", arg322_1: "b8[32, 12, 512, 512]", arg323_1: "bf16[384, 512, 512]", arg324_1: "f32[32, 12, 512, 1]", arg325_1: "f32[32, 12, 512, 1]", arg326_1: "b8[32, 12, 512, 1]", arg327_1: "f32[32, 12, 512, 1]", arg328_1: "bf16[384, 64, 512]", arg329_1: "bf16[384, 512, 64]", arg330_1: "bf16[768, 768]", arg331_1: "bf16[16384, 768]", arg332_1: "bf16[768, 768]", arg333_1: "bf16[768, 768]", arg334_1: "f32[768]", arg335_1: "f32[32, 512, 768]", arg336_1: "f32[32, 512, 1]", arg337_1: "b8[32, 512, 768]", arg338_1: "bf16[768, 3072]", arg339_1: "bf16[16384, 3072]", arg340_1: "bf16[16384, 3072]", arg341_1: "bf16[3072, 768]", arg342_1: "bf16[16384, 768]", arg343_1: "f32[768]", arg344_1: "f32[32, 512, 768]", arg345_1: "f32[32, 512, 1]", arg346_1: "b8[32, 512, 768]", arg347_1: "bf16[768, 768]", arg348_1: "bf16[16384, 768]", arg349_1: "bf16[384, 512, 512]", arg350_1: "bf16[384, 64, 512]", arg351_1: "b8[32, 12, 512, 512]", arg352_1: "bf16[384, 512, 512]", arg353_1: "f32[32, 12, 512, 1]", arg354_1: "f32[32, 12, 512, 1]", arg355_1: "b8[32, 12, 512, 1]", arg356_1: "f32[32, 12, 512, 1]", arg357_1: "bf16[384, 64, 512]", arg358_1: "bf16[384, 512, 64]", arg359_1: "bf16[768, 768]", arg360_1: "bf16[16384, 768]", arg361_1: "bf16[768, 768]", arg362_1: "bf16[768, 768]", arg363_1: "b8[32, 512, 768]", arg364_1: "f32[768]", arg365_1: "f32[32, 512, 768]", arg366_1: "f32[32, 512, 1]", arg367_1: "i64[32, 512]", arg368_1: "i64[32, 512]", arg369_1: "i64[32, 512]", arg370_1: "i64[32, 512]", arg371_1: "i64[32, 512]", arg372_1: "i64[32, 512]", arg373_1: "i64[32, 512]", arg374_1: "i64[1, 512]", arg375_1: "i64[32, 512]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377):
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
        view_25: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg33_1, _shape_param_27);  arg33_1 = _shape_param_27 = None
        mul_43: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_25, 0.125)
        convert_element_type_39: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_43, torch.float32);  mul_43 = None
        convert_element_type_40: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_25, torch.float32);  view_25 = None
        mul_44: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1);  convert_element_type_40 = None
        sub_10: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_44, arg34_1);  mul_44 = arg34_1 = None
        mul_45: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_10, 0.125);  sub_10 = None
        sub_11: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_39, arg35_1);  convert_element_type_39 = arg35_1 = None
        where_3: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg36_1, mul_45, sub_11);  arg36_1 = mul_45 = sub_11 = None
        exp_3: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_3);  where_3 = None
        div_2: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_3, arg37_1);  exp_3 = arg37_1 = None
        mul_46: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_38, div_2);  convert_element_type_38 = None
        sum_19: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_46, [-1], True)
        neg: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg, sum_19, mul_46);  neg = sum_19 = mul_46 = None
        convert_element_type_41: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        mul_47: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 0.125);  convert_element_type_41 = None
        view_26: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_47, _shape_param_28);  mul_47 = _shape_param_28 = None
        bmm_2: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg38_1, view_26);  arg38_1 = None
        bmm_3: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_26, arg39_1);  view_26 = arg39_1 = None
        view_27: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_2, _shape_param_29);  bmm_2 = _shape_param_29 = None
        view_28: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_30);  bmm_3 = _shape_param_30 = None
        permute_6: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_27, [0, 1, 3, 2]);  view_27 = None
        permute_7: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
        clone_1: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_29: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_1, _shape_param_31);  clone_1 = _shape_param_31 = None
        view_30: "bf16[16384, 768]" = torch.ops.aten.view.default(view_29, _shape_param_32);  view_29 = _shape_param_32 = None
        mm_10: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_30, arg40_1);  arg40_1 = None
        permute_8: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_30, [1, 0])
        mm_11: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_8, arg41_1);  permute_8 = None
        sum_20: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_30, [0], True, dtype = torch.float32);  view_30 = None
        view_31: "f32[768]" = torch.ops.aten.view.default(sum_20, _shape_param_33);  sum_20 = _shape_param_33 = None
        convert_element_type_42: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_31, torch.bfloat16);  view_31 = None
        view_32: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_10, _shape_param_34);  mm_10 = _shape_param_34 = None
        convert_element_type_43: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_32, torch.float32);  view_32 = None
        add_6: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_37, convert_element_type_43);  mul_37 = convert_element_type_43 = None
        convert_element_type_44: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_45: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_42, torch.float32);  convert_element_type_42 = None
        permute_9: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1, 3]);  permute_6 = None
        view_33: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_9, _shape_param_35);  permute_9 = _shape_param_35 = None
        clone_2: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_33, memory_format = torch.contiguous_format);  view_33 = None
        view_34: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_2, _shape_param_36);  clone_2 = _shape_param_36 = None
        mm_12: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_34, arg42_1);  arg42_1 = None
        permute_10: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_34, [1, 0])
        mm_13: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_10, arg41_1);  permute_10 = None
        sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_34, [0], True, dtype = torch.float32);  view_34 = None
        view_35: "f32[768]" = torch.ops.aten.view.default(sum_21, _shape_param_37);  sum_21 = _shape_param_37 = None
        convert_element_type_46: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_35, torch.bfloat16);  view_35 = None
        view_36: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_12, _shape_param_38);  mm_12 = _shape_param_38 = None
        convert_element_type_47: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_36, torch.float32);  view_36 = None
        add_7: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_47);  add_6 = convert_element_type_47 = None
        convert_element_type_48: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_49: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32);  convert_element_type_46 = None
        permute_11: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
        clone_3: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_37: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_3, _shape_param_39);  clone_3 = _shape_param_39 = None
        view_38: "bf16[16384, 768]" = torch.ops.aten.view.default(view_37, _shape_param_40);  view_37 = _shape_param_40 = None
        mm_14: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_38, arg43_1);  arg43_1 = None
        permute_12: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_38, [1, 0])
        mm_15: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_12, arg41_1);  permute_12 = arg41_1 = None
        sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_38, [0], True, dtype = torch.float32);  view_38 = None
        view_39: "f32[768]" = torch.ops.aten.view.default(sum_22, _shape_param_41);  sum_22 = _shape_param_41 = None
        convert_element_type_50: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_39, torch.bfloat16);  view_39 = None
        view_40: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_14, _shape_param_42);  mm_14 = _shape_param_42 = None
        convert_element_type_51: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_40, torch.float32);  view_40 = None
        add_8: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_7, convert_element_type_51);  add_7 = convert_element_type_51 = None
        convert_element_type_52: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_53: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_50, torch.float32);  convert_element_type_50 = None
        mul_48: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_8, arg44_1);  arg44_1 = None
        mul_49: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, 768)
        sum_23: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_48, [2], True)
        mul_50: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_48, arg45_1);  mul_48 = None
        sum_24: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_50, [2], True);  mul_50 = None
        mul_51: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg45_1, sum_24);  sum_24 = None
        sub_12: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_49, sum_23);  mul_49 = sum_23 = None
        sub_13: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_12, mul_51);  sub_12 = mul_51 = None
        mul_52: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg46_1, sub_13);  arg46_1 = sub_13 = None
        mul_53: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_8, arg45_1);  arg45_1 = None
        sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_53, [0, 1]);  mul_53 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_8, [0, 1]);  add_8 = None
        convert_element_type_54: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16)
        convert_element_type_55: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        mul_54: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_55: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_54, mul_54);  convert_element_type_54 = mul_54 = None
        view_41: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_55, _shape_param_43);  mul_55 = _shape_param_43 = None
        mm_16: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_41, arg48_1);  arg48_1 = None
        permute_13: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_41, [1, 0])
        mm_17: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_13, arg49_1);  permute_13 = arg49_1 = None
        sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_41, [0], True, dtype = torch.float32);  view_41 = None
        view_42: "f32[768]" = torch.ops.aten.view.default(sum_27, _shape_param_44);  sum_27 = _shape_param_44 = None
        convert_element_type_56: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_42, torch.bfloat16);  view_42 = None
        view_43: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_16, _shape_param_45);  mm_16 = _shape_param_45 = None
        convert_element_type_57: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_58: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_56, torch.float32);  convert_element_type_56 = None
        convert_element_type_59: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        view_44: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg50_1, _shape_param_46);  arg50_1 = _shape_param_46 = None
        convert_element_type_60: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_44, torch.float32);  view_44 = None
        mul_56: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.7071067811865476)
        erf_2: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
        add_9: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_57: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
        mul_58: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_60, convert_element_type_60)
        mul_59: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_58, -0.5);  mul_58 = None
        exp_4: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_59);  mul_59 = None
        mul_60: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_61: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_60, mul_60);  convert_element_type_60 = mul_60 = None
        add_10: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_57, mul_61);  mul_57 = mul_61 = None
        mul_62: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_59, add_10);  convert_element_type_59 = add_10 = None
        convert_element_type_61: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        view_45: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_61, _shape_param_47);  convert_element_type_61 = _shape_param_47 = None
        mm_18: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_45, arg51_1);  arg51_1 = None
        permute_14: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_45, [1, 0])
        mm_19: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_14, arg52_1);  permute_14 = arg52_1 = None
        sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_45, [0], True, dtype = torch.float32);  view_45 = None
        view_46: "f32[3072]" = torch.ops.aten.view.default(sum_28, _shape_param_48);  sum_28 = _shape_param_48 = None
        convert_element_type_62: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_46, torch.bfloat16);  view_46 = None
        view_47: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_18, _shape_param_49);  mm_18 = _shape_param_49 = None
        convert_element_type_63: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        add_11: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_52, convert_element_type_63);  mul_52 = convert_element_type_63 = None
        convert_element_type_64: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_65: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_62, torch.float32);  convert_element_type_62 = None
        mul_63: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_11, arg53_1);  arg53_1 = None
        mul_64: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, 768)
        sum_29: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_63, [2], True)
        mul_65: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_63, arg54_1);  mul_63 = None
        sum_30: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_65, [2], True);  mul_65 = None
        mul_66: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg54_1, sum_30);  sum_30 = None
        sub_14: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_64, sum_29);  mul_64 = sum_29 = None
        sub_15: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_14, mul_66);  sub_14 = mul_66 = None
        mul_67: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg55_1, sub_15);  arg55_1 = sub_15 = None
        mul_68: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_11, arg54_1);  arg54_1 = None
        sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_68, [0, 1]);  mul_68 = None
        sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_11, [0, 1]);  add_11 = None
        convert_element_type_66: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_67, torch.bfloat16)
        convert_element_type_67: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        mul_69: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
        mul_70: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_66, mul_69);  convert_element_type_66 = mul_69 = None
        view_48: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_70, _shape_param_50);  mul_70 = _shape_param_50 = None
        mm_20: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_48, arg57_1);  arg57_1 = None
        permute_15: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_48, [1, 0])
        mm_21: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_15, arg58_1);  permute_15 = arg58_1 = None
        sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_48, [0], True, dtype = torch.float32);  view_48 = None
        view_49: "f32[768]" = torch.ops.aten.view.default(sum_33, _shape_param_51);  sum_33 = _shape_param_51 = None
        convert_element_type_68: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_49, torch.bfloat16);  view_49 = None
        view_50: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_20, _shape_param_52);  mm_20 = _shape_param_52 = None
        convert_element_type_69: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_70: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_68, torch.float32);  convert_element_type_68 = None
        view_51: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_50, _shape_param_53);  view_50 = _shape_param_53 = None
        permute_16: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None
        clone_4: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_52: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_54);  clone_4 = _shape_param_54 = None
        bmm_4: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg59_1, view_52);  arg59_1 = None
        bmm_5: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_52, arg60_1);  view_52 = arg60_1 = None
        view_53: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_4, _shape_param_55);  bmm_4 = _shape_param_55 = None
        view_54: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_5, _shape_param_56);  bmm_5 = _shape_param_56 = None
        convert_element_type_71: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        mul_71: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
        mul_72: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_54, mul_71);  view_54 = mul_71 = None
        convert_element_type_72: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_72, torch.float32);  mul_72 = None
        view_55: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg62_1, _shape_param_57);  arg62_1 = _shape_param_57 = None
        mul_73: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_55, 0.125)
        convert_element_type_73: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_73, torch.float32);  mul_73 = None
        convert_element_type_74: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        mul_74: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1);  convert_element_type_74 = None
        sub_16: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_74, arg63_1);  mul_74 = arg63_1 = None
        mul_75: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_16, 0.125);  sub_16 = None
        sub_17: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_73, arg64_1);  convert_element_type_73 = arg64_1 = None
        where_4: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg65_1, mul_75, sub_17);  arg65_1 = mul_75 = sub_17 = None
        exp_5: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_4);  where_4 = None
        div_3: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_5, arg66_1);  exp_5 = arg66_1 = None
        mul_76: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_72, div_3);  convert_element_type_72 = None
        sum_34: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_76, [-1], True)
        neg_1: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_1: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_1, sum_34, mul_76);  neg_1 = sum_34 = mul_76 = None
        convert_element_type_75: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        mul_77: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.125);  convert_element_type_75 = None
        view_56: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_77, _shape_param_58);  mul_77 = _shape_param_58 = None
        bmm_6: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg67_1, view_56);  arg67_1 = None
        bmm_7: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_56, arg68_1);  view_56 = arg68_1 = None
        view_57: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_6, _shape_param_59);  bmm_6 = _shape_param_59 = None
        view_58: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_60);  bmm_7 = _shape_param_60 = None
        permute_17: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_57, [0, 1, 3, 2]);  view_57 = None
        permute_18: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
        clone_5: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_59: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_5, _shape_param_61);  clone_5 = _shape_param_61 = None
        view_60: "bf16[16384, 768]" = torch.ops.aten.view.default(view_59, _shape_param_62);  view_59 = _shape_param_62 = None
        mm_22: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_60, arg69_1);  arg69_1 = None
        permute_19: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_60, [1, 0])
        mm_23: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_19, arg70_1);  permute_19 = None
        sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_60, [0], True, dtype = torch.float32);  view_60 = None
        view_61: "f32[768]" = torch.ops.aten.view.default(sum_35, _shape_param_63);  sum_35 = _shape_param_63 = None
        convert_element_type_76: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_61, torch.bfloat16);  view_61 = None
        view_62: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_22, _shape_param_64);  mm_22 = _shape_param_64 = None
        convert_element_type_77: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None
        add_12: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_67, convert_element_type_77);  mul_67 = convert_element_type_77 = None
        convert_element_type_78: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_79: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_76, torch.float32);  convert_element_type_76 = None
        permute_20: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_17, [0, 2, 1, 3]);  permute_17 = None
        view_63: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_20, _shape_param_65);  permute_20 = _shape_param_65 = None
        clone_6: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_63, memory_format = torch.contiguous_format);  view_63 = None
        view_64: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_6, _shape_param_66);  clone_6 = _shape_param_66 = None
        mm_24: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_64, arg71_1);  arg71_1 = None
        permute_21: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_64, [1, 0])
        mm_25: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_21, arg70_1);  permute_21 = None
        sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_64, [0], True, dtype = torch.float32);  view_64 = None
        view_65: "f32[768]" = torch.ops.aten.view.default(sum_36, _shape_param_67);  sum_36 = _shape_param_67 = None
        convert_element_type_80: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_65, torch.bfloat16);  view_65 = None
        view_66: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_24, _shape_param_68);  mm_24 = _shape_param_68 = None
        convert_element_type_81: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None
        add_13: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_12, convert_element_type_81);  add_12 = convert_element_type_81 = None
        convert_element_type_82: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_83: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_80, torch.float32);  convert_element_type_80 = None
        permute_22: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_7: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_67: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_7, _shape_param_69);  clone_7 = _shape_param_69 = None
        view_68: "bf16[16384, 768]" = torch.ops.aten.view.default(view_67, _shape_param_70);  view_67 = _shape_param_70 = None
        mm_26: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_68, arg72_1);  arg72_1 = None
        permute_23: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_68, [1, 0])
        mm_27: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_23, arg70_1);  permute_23 = arg70_1 = None
        sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_68, [0], True, dtype = torch.float32);  view_68 = None
        view_69: "f32[768]" = torch.ops.aten.view.default(sum_37, _shape_param_71);  sum_37 = _shape_param_71 = None
        convert_element_type_84: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_69, torch.bfloat16);  view_69 = None
        view_70: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_26, _shape_param_72);  mm_26 = _shape_param_72 = None
        convert_element_type_85: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None
        add_14: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_85);  add_13 = convert_element_type_85 = None
        convert_element_type_86: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_87: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_84, torch.float32);  convert_element_type_84 = None
        mul_78: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_14, arg73_1);  arg73_1 = None
        mul_79: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, 768)
        sum_38: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_78, [2], True)
        mul_80: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, arg74_1);  mul_78 = None
        sum_39: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_80, [2], True);  mul_80 = None
        mul_81: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg74_1, sum_39);  sum_39 = None
        sub_18: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_79, sum_38);  mul_79 = sum_38 = None
        sub_19: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_18, mul_81);  sub_18 = mul_81 = None
        mul_82: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg75_1, sub_19);  arg75_1 = sub_19 = None
        mul_83: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_14, arg74_1);  arg74_1 = None
        sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_83, [0, 1]);  mul_83 = None
        sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
        convert_element_type_88: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_82, torch.bfloat16)
        convert_element_type_89: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg76_1, torch.bfloat16);  arg76_1 = None
        mul_84: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.1111111111111112);  convert_element_type_89 = None
        mul_85: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_88, mul_84);  convert_element_type_88 = mul_84 = None
        view_71: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_85, _shape_param_73);  mul_85 = _shape_param_73 = None
        mm_28: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_71, arg77_1);  arg77_1 = None
        permute_24: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_71, [1, 0])
        mm_29: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_24, arg78_1);  permute_24 = arg78_1 = None
        sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_71, [0], True, dtype = torch.float32);  view_71 = None
        view_72: "f32[768]" = torch.ops.aten.view.default(sum_42, _shape_param_74);  sum_42 = _shape_param_74 = None
        convert_element_type_90: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_72, torch.bfloat16);  view_72 = None
        view_73: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_28, _shape_param_75);  mm_28 = _shape_param_75 = None
        convert_element_type_91: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_92: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_90, torch.float32);  convert_element_type_90 = None
        convert_element_type_93: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        view_74: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg79_1, _shape_param_76);  arg79_1 = _shape_param_76 = None
        convert_element_type_94: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_74, torch.float32);  view_74 = None
        mul_86: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.7071067811865476)
        erf_3: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_15: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_87: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_15, 0.5);  add_15 = None
        mul_88: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_94, convert_element_type_94)
        mul_89: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_88, -0.5);  mul_88 = None
        exp_6: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_89);  mul_89 = None
        mul_90: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_91: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_94, mul_90);  convert_element_type_94 = mul_90 = None
        add_16: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_87, mul_91);  mul_87 = mul_91 = None
        mul_92: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_93, add_16);  convert_element_type_93 = add_16 = None
        convert_element_type_95: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_92, torch.bfloat16);  mul_92 = None
        view_75: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_95, _shape_param_77);  convert_element_type_95 = _shape_param_77 = None
        mm_30: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_75, arg80_1);  arg80_1 = None
        permute_25: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_75, [1, 0])
        mm_31: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_25, arg81_1);  permute_25 = arg81_1 = None
        sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_75, [0], True, dtype = torch.float32);  view_75 = None
        view_76: "f32[3072]" = torch.ops.aten.view.default(sum_43, _shape_param_78);  sum_43 = _shape_param_78 = None
        convert_element_type_96: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_76, torch.bfloat16);  view_76 = None
        view_77: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_30, _shape_param_79);  mm_30 = _shape_param_79 = None
        convert_element_type_97: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        add_17: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_82, convert_element_type_97);  mul_82 = convert_element_type_97 = None
        convert_element_type_98: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_99: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_96, torch.float32);  convert_element_type_96 = None
        mul_93: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_17, arg82_1);  arg82_1 = None
        mul_94: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_93, 768)
        sum_44: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_93, [2], True)
        mul_95: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_93, arg83_1);  mul_93 = None
        sum_45: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True);  mul_95 = None
        mul_96: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg83_1, sum_45);  sum_45 = None
        sub_20: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_94, sum_44);  mul_94 = sum_44 = None
        sub_21: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_20, mul_96);  sub_20 = mul_96 = None
        mul_97: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg84_1, sub_21);  arg84_1 = sub_21 = None
        mul_98: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_17, arg83_1);  arg83_1 = None
        sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_98, [0, 1]);  mul_98 = None
        sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_17, [0, 1]);  add_17 = None
        convert_element_type_100: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_97, torch.bfloat16)
        convert_element_type_101: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg85_1, torch.bfloat16);  arg85_1 = None
        mul_99: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_101, 1.1111111111111112);  convert_element_type_101 = None
        mul_100: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_100, mul_99);  convert_element_type_100 = mul_99 = None
        view_78: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_100, _shape_param_80);  mul_100 = _shape_param_80 = None
        mm_32: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_78, arg86_1);  arg86_1 = None
        permute_26: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_78, [1, 0])
        mm_33: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_26, arg87_1);  permute_26 = arg87_1 = None
        sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_78, [0], True, dtype = torch.float32);  view_78 = None
        view_79: "f32[768]" = torch.ops.aten.view.default(sum_48, _shape_param_81);  sum_48 = _shape_param_81 = None
        convert_element_type_102: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_79, torch.bfloat16);  view_79 = None
        view_80: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_32, _shape_param_82);  mm_32 = _shape_param_82 = None
        convert_element_type_103: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_104: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_102, torch.float32);  convert_element_type_102 = None
        view_81: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_80, _shape_param_83);  view_80 = _shape_param_83 = None
        permute_27: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        clone_8: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_82: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_84);  clone_8 = _shape_param_84 = None
        bmm_8: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg88_1, view_82);  arg88_1 = None
        bmm_9: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_82, arg89_1);  view_82 = arg89_1 = None
        view_83: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_85);  bmm_8 = _shape_param_85 = None
        view_84: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_9, _shape_param_86);  bmm_9 = _shape_param_86 = None
        convert_element_type_105: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg90_1, torch.bfloat16);  arg90_1 = None
        mul_101: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_105, 1.1111111111111112);  convert_element_type_105 = None
        mul_102: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_84, mul_101);  view_84 = mul_101 = None
        convert_element_type_106: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_102, torch.float32);  mul_102 = None
        view_85: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg91_1, _shape_param_87);  arg91_1 = _shape_param_87 = None
        mul_103: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_85, 0.125)
        convert_element_type_107: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_103, torch.float32);  mul_103 = None
        convert_element_type_108: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_104: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_108, 1);  convert_element_type_108 = None
        sub_22: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_104, arg92_1);  mul_104 = arg92_1 = None
        mul_105: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_22, 0.125);  sub_22 = None
        sub_23: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_107, arg93_1);  convert_element_type_107 = arg93_1 = None
        where_5: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg94_1, mul_105, sub_23);  arg94_1 = mul_105 = sub_23 = None
        exp_7: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_5);  where_5 = None
        div_4: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_7, arg95_1);  exp_7 = arg95_1 = None
        mul_106: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_106, div_4);  convert_element_type_106 = None
        sum_49: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_106, [-1], True)
        neg_2: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_2: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_2, sum_49, mul_106);  neg_2 = sum_49 = mul_106 = None
        convert_element_type_109: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        mul_107: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_109, 0.125);  convert_element_type_109 = None
        view_86: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_107, _shape_param_88);  mul_107 = _shape_param_88 = None
        bmm_10: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg96_1, view_86);  arg96_1 = None
        bmm_11: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_86, arg97_1);  view_86 = arg97_1 = None
        view_87: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_10, _shape_param_89);  bmm_10 = _shape_param_89 = None
        view_88: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_90);  bmm_11 = _shape_param_90 = None
        permute_28: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_87, [0, 1, 3, 2]);  view_87 = None
        permute_29: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        clone_9: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_89: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_9, _shape_param_91);  clone_9 = _shape_param_91 = None
        view_90: "bf16[16384, 768]" = torch.ops.aten.view.default(view_89, _shape_param_92);  view_89 = _shape_param_92 = None
        mm_34: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_90, arg98_1);  arg98_1 = None
        permute_30: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_90, [1, 0])
        mm_35: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_30, arg99_1);  permute_30 = None
        sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_90, [0], True, dtype = torch.float32);  view_90 = None
        view_91: "f32[768]" = torch.ops.aten.view.default(sum_50, _shape_param_93);  sum_50 = _shape_param_93 = None
        convert_element_type_110: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_91, torch.bfloat16);  view_91 = None
        view_92: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_34, _shape_param_94);  mm_34 = _shape_param_94 = None
        convert_element_type_111: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_92, torch.float32);  view_92 = None
        add_18: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_97, convert_element_type_111);  mul_97 = convert_element_type_111 = None
        convert_element_type_112: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_113: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_110, torch.float32);  convert_element_type_110 = None
        permute_31: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_28, [0, 2, 1, 3]);  permute_28 = None
        view_93: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_31, _shape_param_95);  permute_31 = _shape_param_95 = None
        clone_10: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_93, memory_format = torch.contiguous_format);  view_93 = None
        view_94: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_10, _shape_param_96);  clone_10 = _shape_param_96 = None
        mm_36: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_94, arg100_1);  arg100_1 = None
        permute_32: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_94, [1, 0])
        mm_37: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_32, arg99_1);  permute_32 = None
        sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_94, [0], True, dtype = torch.float32);  view_94 = None
        view_95: "f32[768]" = torch.ops.aten.view.default(sum_51, _shape_param_97);  sum_51 = _shape_param_97 = None
        convert_element_type_114: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_95, torch.bfloat16);  view_95 = None
        view_96: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_36, _shape_param_98);  mm_36 = _shape_param_98 = None
        convert_element_type_115: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_96, torch.float32);  view_96 = None
        add_19: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_18, convert_element_type_115);  add_18 = convert_element_type_115 = None
        convert_element_type_116: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_117: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_114, torch.float32);  convert_element_type_114 = None
        permute_33: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        clone_11: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_97: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_11, _shape_param_99);  clone_11 = _shape_param_99 = None
        view_98: "bf16[16384, 768]" = torch.ops.aten.view.default(view_97, _shape_param_100);  view_97 = _shape_param_100 = None
        mm_38: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_98, arg101_1);  arg101_1 = None
        permute_34: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_98, [1, 0])
        mm_39: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_34, arg99_1);  permute_34 = arg99_1 = None
        sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True, dtype = torch.float32);  view_98 = None
        view_99: "f32[768]" = torch.ops.aten.view.default(sum_52, _shape_param_101);  sum_52 = _shape_param_101 = None
        convert_element_type_118: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_99, torch.bfloat16);  view_99 = None
        view_100: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_38, _shape_param_102);  mm_38 = _shape_param_102 = None
        convert_element_type_119: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_100, torch.float32);  view_100 = None
        add_20: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_19, convert_element_type_119);  add_19 = convert_element_type_119 = None
        convert_element_type_120: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_121: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_118, torch.float32);  convert_element_type_118 = None
        mul_108: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_20, arg102_1);  arg102_1 = None
        mul_109: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, 768)
        sum_53: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
        mul_110: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, arg103_1);  mul_108 = None
        sum_54: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
        mul_111: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg103_1, sum_54);  sum_54 = None
        sub_24: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_109, sum_53);  mul_109 = sum_53 = None
        sub_25: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_24, mul_111);  sub_24 = mul_111 = None
        mul_112: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg104_1, sub_25);  arg104_1 = sub_25 = None
        mul_113: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_20, arg103_1);  arg103_1 = None
        sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
        sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_20, [0, 1]);  add_20 = None
        convert_element_type_122: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16)
        convert_element_type_123: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg105_1, torch.bfloat16);  arg105_1 = None
        mul_114: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 1.1111111111111112);  convert_element_type_123 = None
        mul_115: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_122, mul_114);  convert_element_type_122 = mul_114 = None
        view_101: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_115, _shape_param_103);  mul_115 = _shape_param_103 = None
        mm_40: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_101, arg106_1);  arg106_1 = None
        permute_35: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_101, [1, 0])
        mm_41: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_35, arg107_1);  permute_35 = arg107_1 = None
        sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_101, [0], True, dtype = torch.float32);  view_101 = None
        view_102: "f32[768]" = torch.ops.aten.view.default(sum_57, _shape_param_104);  sum_57 = _shape_param_104 = None
        convert_element_type_124: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_102, torch.bfloat16);  view_102 = None
        view_103: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_40, _shape_param_105);  mm_40 = _shape_param_105 = None
        convert_element_type_125: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_126: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_124, torch.float32);  convert_element_type_124 = None
        convert_element_type_127: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_103, torch.float32);  view_103 = None
        view_104: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg108_1, _shape_param_106);  arg108_1 = _shape_param_106 = None
        convert_element_type_128: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_104, torch.float32);  view_104 = None
        mul_116: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_128, 0.7071067811865476)
        erf_4: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_21: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_117: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
        mul_118: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_128, convert_element_type_128)
        mul_119: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_118, -0.5);  mul_118 = None
        exp_8: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_119);  mul_119 = None
        mul_120: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_121: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_128, mul_120);  convert_element_type_128 = mul_120 = None
        add_22: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_117, mul_121);  mul_117 = mul_121 = None
        mul_122: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_127, add_22);  convert_element_type_127 = add_22 = None
        convert_element_type_129: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_122, torch.bfloat16);  mul_122 = None
        view_105: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_129, _shape_param_107);  convert_element_type_129 = _shape_param_107 = None
        mm_42: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_105, arg109_1);  arg109_1 = None
        permute_36: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_105, [1, 0])
        mm_43: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_36, arg110_1);  permute_36 = arg110_1 = None
        sum_58: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_105, [0], True, dtype = torch.float32);  view_105 = None
        view_106: "f32[3072]" = torch.ops.aten.view.default(sum_58, _shape_param_108);  sum_58 = _shape_param_108 = None
        convert_element_type_130: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_106, torch.bfloat16);  view_106 = None
        view_107: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_42, _shape_param_109);  mm_42 = _shape_param_109 = None
        convert_element_type_131: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        add_23: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_112, convert_element_type_131);  mul_112 = convert_element_type_131 = None
        convert_element_type_132: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_133: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_130, torch.float32);  convert_element_type_130 = None
        mul_123: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_23, arg111_1);  arg111_1 = None
        mul_124: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_123, 768)
        sum_59: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True)
        mul_125: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_123, arg112_1);  mul_123 = None
        sum_60: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True);  mul_125 = None
        mul_126: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg112_1, sum_60);  sum_60 = None
        sub_26: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_124, sum_59);  mul_124 = sum_59 = None
        sub_27: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_126);  sub_26 = mul_126 = None
        mul_127: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg113_1, sub_27);  arg113_1 = sub_27 = None
        mul_128: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_23, arg112_1);  arg112_1 = None
        sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1]);  mul_128 = None
        sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_23, [0, 1]);  add_23 = None
        convert_element_type_134: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_127, torch.bfloat16)
        convert_element_type_135: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.bfloat16);  arg114_1 = None
        mul_129: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_135, 1.1111111111111112);  convert_element_type_135 = None
        mul_130: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_134, mul_129);  convert_element_type_134 = mul_129 = None
        view_108: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_130, _shape_param_110);  mul_130 = _shape_param_110 = None
        mm_44: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_108, arg115_1);  arg115_1 = None
        permute_37: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_108, [1, 0])
        mm_45: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_37, arg116_1);  permute_37 = arg116_1 = None
        sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_108, [0], True, dtype = torch.float32);  view_108 = None
        view_109: "f32[768]" = torch.ops.aten.view.default(sum_63, _shape_param_111);  sum_63 = _shape_param_111 = None
        convert_element_type_136: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_109, torch.bfloat16);  view_109 = None
        view_110: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_44, _shape_param_112);  mm_44 = _shape_param_112 = None
        convert_element_type_137: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_138: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_136, torch.float32);  convert_element_type_136 = None
        view_111: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_110, _shape_param_113);  view_110 = _shape_param_113 = None
        permute_38: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
        clone_12: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        view_112: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_114);  clone_12 = _shape_param_114 = None
        bmm_12: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg117_1, view_112);  arg117_1 = None
        bmm_13: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_112, arg118_1);  view_112 = arg118_1 = None
        view_113: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_12, _shape_param_115);  bmm_12 = _shape_param_115 = None
        view_114: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_13, _shape_param_116);  bmm_13 = _shape_param_116 = None
        convert_element_type_139: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg119_1, torch.bfloat16);  arg119_1 = None
        mul_131: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 1.1111111111111112);  convert_element_type_139 = None
        mul_132: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_114, mul_131);  view_114 = mul_131 = None
        convert_element_type_140: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_132, torch.float32);  mul_132 = None
        view_115: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg120_1, _shape_param_117);  arg120_1 = _shape_param_117 = None
        mul_133: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_115, 0.125)
        convert_element_type_141: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_133, torch.float32);  mul_133 = None
        convert_element_type_142: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_115, torch.float32);  view_115 = None
        mul_134: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_142, 1);  convert_element_type_142 = None
        sub_28: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_134, arg121_1);  mul_134 = arg121_1 = None
        mul_135: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_28, 0.125);  sub_28 = None
        sub_29: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_141, arg122_1);  convert_element_type_141 = arg122_1 = None
        where_6: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg123_1, mul_135, sub_29);  arg123_1 = mul_135 = sub_29 = None
        exp_9: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_6);  where_6 = None
        div_5: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_9, arg124_1);  exp_9 = arg124_1 = None
        mul_136: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_140, div_5);  convert_element_type_140 = None
        sum_64: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_136, [-1], True)
        neg_3: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_3: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_3, sum_64, mul_136);  neg_3 = sum_64 = mul_136 = None
        convert_element_type_143: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        mul_137: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_143, 0.125);  convert_element_type_143 = None
        view_116: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_137, _shape_param_118);  mul_137 = _shape_param_118 = None
        bmm_14: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg125_1, view_116);  arg125_1 = None
        bmm_15: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_116, arg126_1);  view_116 = arg126_1 = None
        view_117: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_14, _shape_param_119);  bmm_14 = _shape_param_119 = None
        view_118: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_120);  bmm_15 = _shape_param_120 = None
        permute_39: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_117, [0, 1, 3, 2]);  view_117 = None
        permute_40: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        clone_13: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_119: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_13, _shape_param_121);  clone_13 = _shape_param_121 = None
        view_120: "bf16[16384, 768]" = torch.ops.aten.view.default(view_119, _shape_param_122);  view_119 = _shape_param_122 = None
        mm_46: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_120, arg127_1);  arg127_1 = None
        permute_41: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_120, [1, 0])
        mm_47: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_41, arg128_1);  permute_41 = None
        sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True, dtype = torch.float32);  view_120 = None
        view_121: "f32[768]" = torch.ops.aten.view.default(sum_65, _shape_param_123);  sum_65 = _shape_param_123 = None
        convert_element_type_144: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_121, torch.bfloat16);  view_121 = None
        view_122: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_46, _shape_param_124);  mm_46 = _shape_param_124 = None
        convert_element_type_145: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_122, torch.float32);  view_122 = None
        add_24: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_127, convert_element_type_145);  mul_127 = convert_element_type_145 = None
        convert_element_type_146: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_147: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_144, torch.float32);  convert_element_type_144 = None
        permute_42: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_39, [0, 2, 1, 3]);  permute_39 = None
        view_123: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_42, _shape_param_125);  permute_42 = _shape_param_125 = None
        clone_14: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_123, memory_format = torch.contiguous_format);  view_123 = None
        view_124: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_14, _shape_param_126);  clone_14 = _shape_param_126 = None
        mm_48: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_124, arg129_1);  arg129_1 = None
        permute_43: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_124, [1, 0])
        mm_49: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_43, arg128_1);  permute_43 = None
        sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_124, [0], True, dtype = torch.float32);  view_124 = None
        view_125: "f32[768]" = torch.ops.aten.view.default(sum_66, _shape_param_127);  sum_66 = _shape_param_127 = None
        convert_element_type_148: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_125, torch.bfloat16);  view_125 = None
        view_126: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_48, _shape_param_128);  mm_48 = _shape_param_128 = None
        convert_element_type_149: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_126, torch.float32);  view_126 = None
        add_25: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_24, convert_element_type_149);  add_24 = convert_element_type_149 = None
        convert_element_type_150: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_151: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_148, torch.float32);  convert_element_type_148 = None
        permute_44: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_15: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_127: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_15, _shape_param_129);  clone_15 = _shape_param_129 = None
        view_128: "bf16[16384, 768]" = torch.ops.aten.view.default(view_127, _shape_param_130);  view_127 = _shape_param_130 = None
        mm_50: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_128, arg130_1);  arg130_1 = None
        permute_45: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_128, [1, 0])
        mm_51: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_45, arg128_1);  permute_45 = arg128_1 = None
        sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_128, [0], True, dtype = torch.float32);  view_128 = None
        view_129: "f32[768]" = torch.ops.aten.view.default(sum_67, _shape_param_131);  sum_67 = _shape_param_131 = None
        convert_element_type_152: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_129, torch.bfloat16);  view_129 = None
        view_130: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_50, _shape_param_132);  mm_50 = _shape_param_132 = None
        convert_element_type_153: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        add_26: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_25, convert_element_type_153);  add_25 = convert_element_type_153 = None
        convert_element_type_154: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_155: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_152, torch.float32);  convert_element_type_152 = None
        mul_138: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_26, arg131_1);  arg131_1 = None
        mul_139: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, 768)
        sum_68: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True)
        mul_140: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, arg132_1);  mul_138 = None
        sum_69: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_140, [2], True);  mul_140 = None
        mul_141: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg132_1, sum_69);  sum_69 = None
        sub_30: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_139, sum_68);  mul_139 = sum_68 = None
        sub_31: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_30, mul_141);  sub_30 = mul_141 = None
        mul_142: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg133_1, sub_31);  arg133_1 = sub_31 = None
        mul_143: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_26, arg132_1);  arg132_1 = None
        sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1]);  mul_143 = None
        sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_26, [0, 1]);  add_26 = None
        convert_element_type_156: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16)
        convert_element_type_157: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.bfloat16);  arg134_1 = None
        mul_144: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_157, 1.1111111111111112);  convert_element_type_157 = None
        mul_145: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_156, mul_144);  convert_element_type_156 = mul_144 = None
        view_131: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_145, _shape_param_133);  mul_145 = _shape_param_133 = None
        mm_52: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_131, arg135_1);  arg135_1 = None
        permute_46: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_131, [1, 0])
        mm_53: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_46, arg136_1);  permute_46 = arg136_1 = None
        sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_131, [0], True, dtype = torch.float32);  view_131 = None
        view_132: "f32[768]" = torch.ops.aten.view.default(sum_72, _shape_param_134);  sum_72 = _shape_param_134 = None
        convert_element_type_158: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_132, torch.bfloat16);  view_132 = None
        view_133: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_52, _shape_param_135);  mm_52 = _shape_param_135 = None
        convert_element_type_159: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_160: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_158, torch.float32);  convert_element_type_158 = None
        convert_element_type_161: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        view_134: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg137_1, _shape_param_136);  arg137_1 = _shape_param_136 = None
        convert_element_type_162: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_134, torch.float32);  view_134 = None
        mul_146: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_162, 0.7071067811865476)
        erf_5: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
        add_27: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_147: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
        mul_148: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_162, convert_element_type_162)
        mul_149: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_148, -0.5);  mul_148 = None
        exp_10: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_149);  mul_149 = None
        mul_150: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_151: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_162, mul_150);  convert_element_type_162 = mul_150 = None
        add_28: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_147, mul_151);  mul_147 = mul_151 = None
        mul_152: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_161, add_28);  convert_element_type_161 = add_28 = None
        convert_element_type_163: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        view_135: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_163, _shape_param_137);  convert_element_type_163 = _shape_param_137 = None
        mm_54: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_135, arg138_1);  arg138_1 = None
        permute_47: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_135, [1, 0])
        mm_55: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_47, arg139_1);  permute_47 = arg139_1 = None
        sum_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True, dtype = torch.float32);  view_135 = None
        view_136: "f32[3072]" = torch.ops.aten.view.default(sum_73, _shape_param_138);  sum_73 = _shape_param_138 = None
        convert_element_type_164: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_136, torch.bfloat16);  view_136 = None
        view_137: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_54, _shape_param_139);  mm_54 = _shape_param_139 = None
        convert_element_type_165: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        add_29: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_142, convert_element_type_165);  mul_142 = convert_element_type_165 = None
        convert_element_type_166: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_167: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_164, torch.float32);  convert_element_type_164 = None
        mul_153: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_29, arg140_1);  arg140_1 = None
        mul_154: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_153, 768)
        sum_74: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [2], True)
        mul_155: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_153, arg141_1);  mul_153 = None
        sum_75: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [2], True);  mul_155 = None
        mul_156: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg141_1, sum_75);  sum_75 = None
        sub_32: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_154, sum_74);  mul_154 = sum_74 = None
        sub_33: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_156);  sub_32 = mul_156 = None
        mul_157: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg142_1, sub_33);  arg142_1 = sub_33 = None
        mul_158: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_29, arg141_1);  arg141_1 = None
        sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1]);  mul_158 = None
        sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_29, [0, 1]);  add_29 = None
        convert_element_type_168: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16)
        convert_element_type_169: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.bfloat16);  arg143_1 = None
        mul_159: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_169, 1.1111111111111112);  convert_element_type_169 = None
        mul_160: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_168, mul_159);  convert_element_type_168 = mul_159 = None
        view_138: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_160, _shape_param_140);  mul_160 = _shape_param_140 = None
        mm_56: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_138, arg144_1);  arg144_1 = None
        permute_48: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_138, [1, 0])
        mm_57: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_48, arg145_1);  permute_48 = arg145_1 = None
        sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_138, [0], True, dtype = torch.float32);  view_138 = None
        view_139: "f32[768]" = torch.ops.aten.view.default(sum_78, _shape_param_141);  sum_78 = _shape_param_141 = None
        convert_element_type_170: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_139, torch.bfloat16);  view_139 = None
        view_140: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_56, _shape_param_142);  mm_56 = _shape_param_142 = None
        convert_element_type_171: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_172: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_170, torch.float32);  convert_element_type_170 = None
        view_141: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_140, _shape_param_143);  view_140 = _shape_param_143 = None
        permute_49: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
        clone_16: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_142: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_144);  clone_16 = _shape_param_144 = None
        bmm_16: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg146_1, view_142);  arg146_1 = None
        bmm_17: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_142, arg147_1);  view_142 = arg147_1 = None
        view_143: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_16, _shape_param_145);  bmm_16 = _shape_param_145 = None
        view_144: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_17, _shape_param_146);  bmm_17 = _shape_param_146 = None
        convert_element_type_173: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.bfloat16);  arg148_1 = None
        mul_161: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_173, 1.1111111111111112);  convert_element_type_173 = None
        mul_162: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_144, mul_161);  view_144 = mul_161 = None
        convert_element_type_174: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_162, torch.float32);  mul_162 = None
        view_145: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg149_1, _shape_param_147);  arg149_1 = _shape_param_147 = None
        mul_163: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_145, 0.125)
        convert_element_type_175: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_163, torch.float32);  mul_163 = None
        convert_element_type_176: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_145, torch.float32);  view_145 = None
        mul_164: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_176, 1);  convert_element_type_176 = None
        sub_34: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_164, arg150_1);  mul_164 = arg150_1 = None
        mul_165: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_34, 0.125);  sub_34 = None
        sub_35: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_175, arg151_1);  convert_element_type_175 = arg151_1 = None
        where_7: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg152_1, mul_165, sub_35);  arg152_1 = mul_165 = sub_35 = None
        exp_11: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_7);  where_7 = None
        div_6: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_11, arg153_1);  exp_11 = arg153_1 = None
        mul_166: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_174, div_6);  convert_element_type_174 = None
        sum_79: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [-1], True)
        neg_4: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_4: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_4, sum_79, mul_166);  neg_4 = sum_79 = mul_166 = None
        convert_element_type_177: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        mul_167: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_177, 0.125);  convert_element_type_177 = None
        view_146: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_167, _shape_param_148);  mul_167 = _shape_param_148 = None
        bmm_18: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg154_1, view_146);  arg154_1 = None
        bmm_19: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_146, arg155_1);  view_146 = arg155_1 = None
        view_147: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_18, _shape_param_149);  bmm_18 = _shape_param_149 = None
        view_148: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_150);  bmm_19 = _shape_param_150 = None
        permute_50: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_147, [0, 1, 3, 2]);  view_147 = None
        permute_51: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_143, [0, 2, 1, 3]);  view_143 = None
        clone_17: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_149: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_17, _shape_param_151);  clone_17 = _shape_param_151 = None
        view_150: "bf16[16384, 768]" = torch.ops.aten.view.default(view_149, _shape_param_152);  view_149 = _shape_param_152 = None
        mm_58: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_150, arg156_1);  arg156_1 = None
        permute_52: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_150, [1, 0])
        mm_59: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_52, arg157_1);  permute_52 = None
        sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True, dtype = torch.float32);  view_150 = None
        view_151: "f32[768]" = torch.ops.aten.view.default(sum_80, _shape_param_153);  sum_80 = _shape_param_153 = None
        convert_element_type_178: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_151, torch.bfloat16);  view_151 = None
        view_152: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_58, _shape_param_154);  mm_58 = _shape_param_154 = None
        convert_element_type_179: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_152, torch.float32);  view_152 = None
        add_30: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_157, convert_element_type_179);  mul_157 = convert_element_type_179 = None
        convert_element_type_180: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_181: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_178, torch.float32);  convert_element_type_178 = None
        permute_53: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_50, [0, 2, 1, 3]);  permute_50 = None
        view_153: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_53, _shape_param_155);  permute_53 = _shape_param_155 = None
        clone_18: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_153, memory_format = torch.contiguous_format);  view_153 = None
        view_154: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_18, _shape_param_156);  clone_18 = _shape_param_156 = None
        mm_60: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_154, arg158_1);  arg158_1 = None
        permute_54: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_154, [1, 0])
        mm_61: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_54, arg157_1);  permute_54 = None
        sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True, dtype = torch.float32);  view_154 = None
        view_155: "f32[768]" = torch.ops.aten.view.default(sum_81, _shape_param_157);  sum_81 = _shape_param_157 = None
        convert_element_type_182: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_155, torch.bfloat16);  view_155 = None
        view_156: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_60, _shape_param_158);  mm_60 = _shape_param_158 = None
        convert_element_type_183: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_156, torch.float32);  view_156 = None
        add_31: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_30, convert_element_type_183);  add_30 = convert_element_type_183 = None
        convert_element_type_184: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_185: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_182, torch.float32);  convert_element_type_182 = None
        permute_55: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_19: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_157: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_19, _shape_param_159);  clone_19 = _shape_param_159 = None
        view_158: "bf16[16384, 768]" = torch.ops.aten.view.default(view_157, _shape_param_160);  view_157 = _shape_param_160 = None
        mm_62: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_158, arg159_1);  arg159_1 = None
        permute_56: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_158, [1, 0])
        mm_63: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_56, arg157_1);  permute_56 = arg157_1 = None
        sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_158, [0], True, dtype = torch.float32);  view_158 = None
        view_159: "f32[768]" = torch.ops.aten.view.default(sum_82, _shape_param_161);  sum_82 = _shape_param_161 = None
        convert_element_type_186: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_159, torch.bfloat16);  view_159 = None
        view_160: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_62, _shape_param_162);  mm_62 = _shape_param_162 = None
        convert_element_type_187: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_160, torch.float32);  view_160 = None
        add_32: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_31, convert_element_type_187);  add_31 = convert_element_type_187 = None
        convert_element_type_188: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_189: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_186, torch.float32);  convert_element_type_186 = None
        mul_168: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_32, arg160_1);  arg160_1 = None
        mul_169: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_168, 768)
        sum_83: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True)
        mul_170: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_168, arg161_1);  mul_168 = None
        sum_84: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True);  mul_170 = None
        mul_171: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg161_1, sum_84);  sum_84 = None
        sub_36: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_169, sum_83);  mul_169 = sum_83 = None
        sub_37: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_36, mul_171);  sub_36 = mul_171 = None
        mul_172: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg162_1, sub_37);  arg162_1 = sub_37 = None
        mul_173: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_32, arg161_1);  arg161_1 = None
        sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_173, [0, 1]);  mul_173 = None
        sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_32, [0, 1]);  add_32 = None
        convert_element_type_190: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_172, torch.bfloat16)
        convert_element_type_191: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.bfloat16);  arg163_1 = None
        mul_174: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_191, 1.1111111111111112);  convert_element_type_191 = None
        mul_175: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_190, mul_174);  convert_element_type_190 = mul_174 = None
        view_161: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_175, _shape_param_163);  mul_175 = _shape_param_163 = None
        mm_64: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_161, arg164_1);  arg164_1 = None
        permute_57: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_161, [1, 0])
        mm_65: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_57, arg165_1);  permute_57 = arg165_1 = None
        sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_161, [0], True, dtype = torch.float32);  view_161 = None
        view_162: "f32[768]" = torch.ops.aten.view.default(sum_87, _shape_param_164);  sum_87 = _shape_param_164 = None
        convert_element_type_192: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_162, torch.bfloat16);  view_162 = None
        view_163: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_64, _shape_param_165);  mm_64 = _shape_param_165 = None
        convert_element_type_193: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_194: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_192, torch.float32);  convert_element_type_192 = None
        convert_element_type_195: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_163, torch.float32);  view_163 = None
        view_164: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg166_1, _shape_param_166);  arg166_1 = _shape_param_166 = None
        convert_element_type_196: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_164, torch.float32);  view_164 = None
        mul_176: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_196, 0.7071067811865476)
        erf_6: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_176);  mul_176 = None
        add_33: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_177: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
        mul_178: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_196, convert_element_type_196)
        mul_179: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_178, -0.5);  mul_178 = None
        exp_12: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_179);  mul_179 = None
        mul_180: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_181: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_196, mul_180);  convert_element_type_196 = mul_180 = None
        add_34: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_177, mul_181);  mul_177 = mul_181 = None
        mul_182: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_195, add_34);  convert_element_type_195 = add_34 = None
        convert_element_type_197: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None
        view_165: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_197, _shape_param_167);  convert_element_type_197 = _shape_param_167 = None
        mm_66: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_165, arg167_1);  arg167_1 = None
        permute_58: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_67: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_58, arg168_1);  permute_58 = arg168_1 = None
        sum_88: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True, dtype = torch.float32);  view_165 = None
        view_166: "f32[3072]" = torch.ops.aten.view.default(sum_88, _shape_param_168);  sum_88 = _shape_param_168 = None
        convert_element_type_198: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_166, torch.bfloat16);  view_166 = None
        view_167: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_66, _shape_param_169);  mm_66 = _shape_param_169 = None
        convert_element_type_199: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_167, torch.float32);  view_167 = None
        add_35: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_172, convert_element_type_199);  mul_172 = convert_element_type_199 = None
        convert_element_type_200: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_201: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_198, torch.float32);  convert_element_type_198 = None
        mul_183: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_35, arg169_1);  arg169_1 = None
        mul_184: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_183, 768)
        sum_89: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True)
        mul_185: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_183, arg170_1);  mul_183 = None
        sum_90: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True);  mul_185 = None
        mul_186: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg170_1, sum_90);  sum_90 = None
        sub_38: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_184, sum_89);  mul_184 = sum_89 = None
        sub_39: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_186);  sub_38 = mul_186 = None
        mul_187: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg171_1, sub_39);  arg171_1 = sub_39 = None
        mul_188: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_35, arg170_1);  arg170_1 = None
        sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 1]);  mul_188 = None
        sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_35, [0, 1]);  add_35 = None
        convert_element_type_202: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_187, torch.bfloat16)
        convert_element_type_203: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.bfloat16);  arg172_1 = None
        mul_189: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_203, 1.1111111111111112);  convert_element_type_203 = None
        mul_190: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_202, mul_189);  convert_element_type_202 = mul_189 = None
        view_168: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_190, _shape_param_170);  mul_190 = _shape_param_170 = None
        mm_68: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_168, arg173_1);  arg173_1 = None
        permute_59: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_168, [1, 0])
        mm_69: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_59, arg174_1);  permute_59 = arg174_1 = None
        sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True, dtype = torch.float32);  view_168 = None
        view_169: "f32[768]" = torch.ops.aten.view.default(sum_93, _shape_param_171);  sum_93 = _shape_param_171 = None
        convert_element_type_204: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_169, torch.bfloat16);  view_169 = None
        view_170: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_68, _shape_param_172);  mm_68 = _shape_param_172 = None
        convert_element_type_205: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_206: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_204, torch.float32);  convert_element_type_204 = None
        view_171: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_170, _shape_param_173);  view_170 = _shape_param_173 = None
        permute_60: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
        clone_20: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        view_172: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_174);  clone_20 = _shape_param_174 = None
        bmm_20: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg175_1, view_172);  arg175_1 = None
        bmm_21: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_172, arg176_1);  view_172 = arg176_1 = None
        view_173: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_175);  bmm_20 = _shape_param_175 = None
        view_174: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_21, _shape_param_176);  bmm_21 = _shape_param_176 = None
        convert_element_type_207: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        mul_191: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_207, 1.1111111111111112);  convert_element_type_207 = None
        mul_192: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_174, mul_191);  view_174 = mul_191 = None
        convert_element_type_208: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_192, torch.float32);  mul_192 = None
        view_175: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg178_1, _shape_param_177);  arg178_1 = _shape_param_177 = None
        mul_193: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_175, 0.125)
        convert_element_type_209: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_193, torch.float32);  mul_193 = None
        convert_element_type_210: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_175, torch.float32);  view_175 = None
        mul_194: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 1);  convert_element_type_210 = None
        sub_40: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_194, arg179_1);  mul_194 = arg179_1 = None
        mul_195: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_40, 0.125);  sub_40 = None
        sub_41: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_209, arg180_1);  convert_element_type_209 = arg180_1 = None
        where_8: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg181_1, mul_195, sub_41);  arg181_1 = mul_195 = sub_41 = None
        exp_13: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_8);  where_8 = None
        div_7: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_13, arg182_1);  exp_13 = arg182_1 = None
        mul_196: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_208, div_7);  convert_element_type_208 = None
        sum_94: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [-1], True)
        neg_5: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_5: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_5, sum_94, mul_196);  neg_5 = sum_94 = mul_196 = None
        convert_element_type_211: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        mul_197: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.125);  convert_element_type_211 = None
        view_176: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_197, _shape_param_178);  mul_197 = _shape_param_178 = None
        bmm_22: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg183_1, view_176);  arg183_1 = None
        bmm_23: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_176, arg184_1);  view_176 = arg184_1 = None
        view_177: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_22, _shape_param_179);  bmm_22 = _shape_param_179 = None
        view_178: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_180);  bmm_23 = _shape_param_180 = None
        permute_61: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_177, [0, 1, 3, 2]);  view_177 = None
        permute_62: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
        clone_21: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_179: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_21, _shape_param_181);  clone_21 = _shape_param_181 = None
        view_180: "bf16[16384, 768]" = torch.ops.aten.view.default(view_179, _shape_param_182);  view_179 = _shape_param_182 = None
        mm_70: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_180, arg185_1);  arg185_1 = None
        permute_63: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_71: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_63, arg186_1);  permute_63 = None
        sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True, dtype = torch.float32);  view_180 = None
        view_181: "f32[768]" = torch.ops.aten.view.default(sum_95, _shape_param_183);  sum_95 = _shape_param_183 = None
        convert_element_type_212: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_181, torch.bfloat16);  view_181 = None
        view_182: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_70, _shape_param_184);  mm_70 = _shape_param_184 = None
        convert_element_type_213: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_182, torch.float32);  view_182 = None
        add_36: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_187, convert_element_type_213);  mul_187 = convert_element_type_213 = None
        convert_element_type_214: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_215: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_212, torch.float32);  convert_element_type_212 = None
        permute_64: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_61, [0, 2, 1, 3]);  permute_61 = None
        view_183: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_64, _shape_param_185);  permute_64 = _shape_param_185 = None
        clone_22: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_183, memory_format = torch.contiguous_format);  view_183 = None
        view_184: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_22, _shape_param_186);  clone_22 = _shape_param_186 = None
        mm_72: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_184, arg187_1);  arg187_1 = None
        permute_65: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_184, [1, 0])
        mm_73: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_65, arg186_1);  permute_65 = None
        sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_184, [0], True, dtype = torch.float32);  view_184 = None
        view_185: "f32[768]" = torch.ops.aten.view.default(sum_96, _shape_param_187);  sum_96 = _shape_param_187 = None
        convert_element_type_216: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_185, torch.bfloat16);  view_185 = None
        view_186: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_72, _shape_param_188);  mm_72 = _shape_param_188 = None
        convert_element_type_217: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_186, torch.float32);  view_186 = None
        add_37: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_36, convert_element_type_217);  add_36 = convert_element_type_217 = None
        convert_element_type_218: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_219: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_216, torch.float32);  convert_element_type_216 = None
        permute_66: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        clone_23: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
        view_187: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_23, _shape_param_189);  clone_23 = _shape_param_189 = None
        view_188: "bf16[16384, 768]" = torch.ops.aten.view.default(view_187, _shape_param_190);  view_187 = _shape_param_190 = None
        mm_74: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_188, arg188_1);  arg188_1 = None
        permute_67: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_188, [1, 0])
        mm_75: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_67, arg186_1);  permute_67 = arg186_1 = None
        sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True, dtype = torch.float32);  view_188 = None
        view_189: "f32[768]" = torch.ops.aten.view.default(sum_97, _shape_param_191);  sum_97 = _shape_param_191 = None
        convert_element_type_220: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_189, torch.bfloat16);  view_189 = None
        view_190: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_74, _shape_param_192);  mm_74 = _shape_param_192 = None
        convert_element_type_221: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_190, torch.float32);  view_190 = None
        add_38: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_37, convert_element_type_221);  add_37 = convert_element_type_221 = None
        convert_element_type_222: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_223: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_220, torch.float32);  convert_element_type_220 = None
        mul_198: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_38, arg189_1);  arg189_1 = None
        mul_199: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_198, 768)
        sum_98: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [2], True)
        mul_200: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_198, arg190_1);  mul_198 = None
        sum_99: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True);  mul_200 = None
        mul_201: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg190_1, sum_99);  sum_99 = None
        sub_42: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_199, sum_98);  mul_199 = sum_98 = None
        sub_43: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_42, mul_201);  sub_42 = mul_201 = None
        mul_202: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg191_1, sub_43);  arg191_1 = sub_43 = None
        mul_203: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_38, arg190_1);  arg190_1 = None
        sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 1]);  mul_203 = None
        sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_38, [0, 1]);  add_38 = None
        convert_element_type_224: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16)
        convert_element_type_225: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.bfloat16);  arg192_1 = None
        mul_204: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_225, 1.1111111111111112);  convert_element_type_225 = None
        mul_205: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_224, mul_204);  convert_element_type_224 = mul_204 = None
        view_191: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_205, _shape_param_193);  mul_205 = _shape_param_193 = None
        mm_76: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_191, arg193_1);  arg193_1 = None
        permute_68: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_191, [1, 0])
        mm_77: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_68, arg194_1);  permute_68 = arg194_1 = None
        sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True, dtype = torch.float32);  view_191 = None
        view_192: "f32[768]" = torch.ops.aten.view.default(sum_102, _shape_param_194);  sum_102 = _shape_param_194 = None
        convert_element_type_226: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_192, torch.bfloat16);  view_192 = None
        view_193: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_76, _shape_param_195);  mm_76 = _shape_param_195 = None
        convert_element_type_227: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_228: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_226, torch.float32);  convert_element_type_226 = None
        convert_element_type_229: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_193, torch.float32);  view_193 = None
        view_194: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg195_1, _shape_param_196);  arg195_1 = _shape_param_196 = None
        convert_element_type_230: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_194, torch.float32);  view_194 = None
        mul_206: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_230, 0.7071067811865476)
        erf_7: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_206);  mul_206 = None
        add_39: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_207: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
        mul_208: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_230, convert_element_type_230)
        mul_209: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_208, -0.5);  mul_208 = None
        exp_14: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_209);  mul_209 = None
        mul_210: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_211: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_230, mul_210);  convert_element_type_230 = mul_210 = None
        add_40: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_207, mul_211);  mul_207 = mul_211 = None
        mul_212: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_229, add_40);  convert_element_type_229 = add_40 = None
        convert_element_type_231: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_212, torch.bfloat16);  mul_212 = None
        view_195: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_231, _shape_param_197);  convert_element_type_231 = _shape_param_197 = None
        mm_78: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_195, arg196_1);  arg196_1 = None
        permute_69: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_195, [1, 0])
        mm_79: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_69, arg197_1);  permute_69 = arg197_1 = None
        sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True, dtype = torch.float32);  view_195 = None
        view_196: "f32[3072]" = torch.ops.aten.view.default(sum_103, _shape_param_198);  sum_103 = _shape_param_198 = None
        convert_element_type_232: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_196, torch.bfloat16);  view_196 = None
        view_197: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_78, _shape_param_199);  mm_78 = _shape_param_199 = None
        convert_element_type_233: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        add_41: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_202, convert_element_type_233);  mul_202 = convert_element_type_233 = None
        convert_element_type_234: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_235: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_232, torch.float32);  convert_element_type_232 = None
        mul_213: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_41, arg198_1);  arg198_1 = None
        mul_214: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
        sum_104: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
        mul_215: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, arg199_1);  mul_213 = None
        sum_105: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
        mul_216: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg199_1, sum_105);  sum_105 = None
        sub_44: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_104);  mul_214 = sum_104 = None
        sub_45: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_216);  sub_44 = mul_216 = None
        mul_217: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg200_1, sub_45);  arg200_1 = sub_45 = None
        mul_218: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_41, arg199_1);  arg199_1 = None
        sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
        sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_41, [0, 1]);  add_41 = None
        convert_element_type_236: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_217, torch.bfloat16)
        convert_element_type_237: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg201_1, torch.bfloat16);  arg201_1 = None
        mul_219: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_237, 1.1111111111111112);  convert_element_type_237 = None
        mul_220: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_236, mul_219);  convert_element_type_236 = mul_219 = None
        view_198: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_220, _shape_param_200);  mul_220 = _shape_param_200 = None
        mm_80: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_198, arg202_1);  arg202_1 = None
        permute_70: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_81: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_70, arg203_1);  permute_70 = arg203_1 = None
        sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True, dtype = torch.float32);  view_198 = None
        view_199: "f32[768]" = torch.ops.aten.view.default(sum_108, _shape_param_201);  sum_108 = _shape_param_201 = None
        convert_element_type_238: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_199, torch.bfloat16);  view_199 = None
        view_200: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_80, _shape_param_202);  mm_80 = _shape_param_202 = None
        convert_element_type_239: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_240: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_238, torch.float32);  convert_element_type_238 = None
        view_201: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_200, _shape_param_203);  view_200 = _shape_param_203 = None
        permute_71: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_201, [0, 2, 1, 3]);  view_201 = None
        clone_24: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_202: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_204);  clone_24 = _shape_param_204 = None
        bmm_24: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg204_1, view_202);  arg204_1 = None
        bmm_25: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_202, arg205_1);  view_202 = arg205_1 = None
        view_203: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, _shape_param_205);  bmm_24 = _shape_param_205 = None
        view_204: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, _shape_param_206);  bmm_25 = _shape_param_206 = None
        convert_element_type_241: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg206_1, torch.bfloat16);  arg206_1 = None
        mul_221: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_241, 1.1111111111111112);  convert_element_type_241 = None
        mul_222: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_204, mul_221);  view_204 = mul_221 = None
        convert_element_type_242: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_222, torch.float32);  mul_222 = None
        view_205: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg207_1, _shape_param_207);  arg207_1 = _shape_param_207 = None
        mul_223: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_205, 0.125)
        convert_element_type_243: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_223, torch.float32);  mul_223 = None
        convert_element_type_244: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        mul_224: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_244, 1);  convert_element_type_244 = None
        sub_46: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_224, arg208_1);  mul_224 = arg208_1 = None
        mul_225: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_46, 0.125);  sub_46 = None
        sub_47: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_243, arg209_1);  convert_element_type_243 = arg209_1 = None
        where_9: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg210_1, mul_225, sub_47);  arg210_1 = mul_225 = sub_47 = None
        exp_15: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_9);  where_9 = None
        div_8: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_15, arg211_1);  exp_15 = arg211_1 = None
        mul_226: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_242, div_8);  convert_element_type_242 = None
        sum_109: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [-1], True)
        neg_6: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_6: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_6, sum_109, mul_226);  neg_6 = sum_109 = mul_226 = None
        convert_element_type_245: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        mul_227: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_245, 0.125);  convert_element_type_245 = None
        view_206: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_227, _shape_param_208);  mul_227 = _shape_param_208 = None
        bmm_26: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg212_1, view_206);  arg212_1 = None
        bmm_27: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_206, arg213_1);  view_206 = arg213_1 = None
        view_207: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, _shape_param_209);  bmm_26 = _shape_param_209 = None
        view_208: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_210);  bmm_27 = _shape_param_210 = None
        permute_72: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_207, [0, 1, 3, 2]);  view_207 = None
        permute_73: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
        clone_25: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_209: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_25, _shape_param_211);  clone_25 = _shape_param_211 = None
        view_210: "bf16[16384, 768]" = torch.ops.aten.view.default(view_209, _shape_param_212);  view_209 = _shape_param_212 = None
        mm_82: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_210, arg214_1);  arg214_1 = None
        permute_74: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_83: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_74, arg215_1);  permute_74 = None
        sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True, dtype = torch.float32);  view_210 = None
        view_211: "f32[768]" = torch.ops.aten.view.default(sum_110, _shape_param_213);  sum_110 = _shape_param_213 = None
        convert_element_type_246: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_211, torch.bfloat16);  view_211 = None
        view_212: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_82, _shape_param_214);  mm_82 = _shape_param_214 = None
        convert_element_type_247: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        add_42: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_217, convert_element_type_247);  mul_217 = convert_element_type_247 = None
        convert_element_type_248: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_249: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_246, torch.float32);  convert_element_type_246 = None
        permute_75: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_72, [0, 2, 1, 3]);  permute_72 = None
        view_213: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_75, _shape_param_215);  permute_75 = _shape_param_215 = None
        clone_26: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_213, memory_format = torch.contiguous_format);  view_213 = None
        view_214: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_26, _shape_param_216);  clone_26 = _shape_param_216 = None
        mm_84: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_214, arg216_1);  arg216_1 = None
        permute_76: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_214, [1, 0])
        mm_85: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_76, arg215_1);  permute_76 = None
        sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True, dtype = torch.float32);  view_214 = None
        view_215: "f32[768]" = torch.ops.aten.view.default(sum_111, _shape_param_217);  sum_111 = _shape_param_217 = None
        convert_element_type_250: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_215, torch.bfloat16);  view_215 = None
        view_216: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_84, _shape_param_218);  mm_84 = _shape_param_218 = None
        convert_element_type_251: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        add_43: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_42, convert_element_type_251);  add_42 = convert_element_type_251 = None
        convert_element_type_252: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_253: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_250, torch.float32);  convert_element_type_250 = None
        permute_77: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        clone_27: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_217: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_27, _shape_param_219);  clone_27 = _shape_param_219 = None
        view_218: "bf16[16384, 768]" = torch.ops.aten.view.default(view_217, _shape_param_220);  view_217 = _shape_param_220 = None
        mm_86: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_218, arg217_1);  arg217_1 = None
        permute_78: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_87: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_78, arg215_1);  permute_78 = arg215_1 = None
        sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True, dtype = torch.float32);  view_218 = None
        view_219: "f32[768]" = torch.ops.aten.view.default(sum_112, _shape_param_221);  sum_112 = _shape_param_221 = None
        convert_element_type_254: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_219, torch.bfloat16);  view_219 = None
        view_220: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_86, _shape_param_222);  mm_86 = _shape_param_222 = None
        convert_element_type_255: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        add_44: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_43, convert_element_type_255);  add_43 = convert_element_type_255 = None
        convert_element_type_256: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_257: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_254, torch.float32);  convert_element_type_254 = None
        mul_228: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_44, arg218_1);  arg218_1 = None
        mul_229: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_228, 768)
        sum_113: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True)
        mul_230: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_228, arg219_1);  mul_228 = None
        sum_114: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_230, [2], True);  mul_230 = None
        mul_231: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg219_1, sum_114);  sum_114 = None
        sub_48: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_229, sum_113);  mul_229 = sum_113 = None
        sub_49: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_231);  sub_48 = mul_231 = None
        mul_232: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg220_1, sub_49);  arg220_1 = sub_49 = None
        mul_233: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_44, arg219_1);  arg219_1 = None
        sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 1]);  mul_233 = None
        sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_44, [0, 1]);  add_44 = None
        convert_element_type_258: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_232, torch.bfloat16)
        convert_element_type_259: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg221_1, torch.bfloat16);  arg221_1 = None
        mul_234: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_259, 1.1111111111111112);  convert_element_type_259 = None
        mul_235: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_258, mul_234);  convert_element_type_258 = mul_234 = None
        view_221: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_235, _shape_param_223);  mul_235 = _shape_param_223 = None
        mm_88: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_221, arg222_1);  arg222_1 = None
        permute_79: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_221, [1, 0])
        mm_89: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_79, arg223_1);  permute_79 = arg223_1 = None
        sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True, dtype = torch.float32);  view_221 = None
        view_222: "f32[768]" = torch.ops.aten.view.default(sum_117, _shape_param_224);  sum_117 = _shape_param_224 = None
        convert_element_type_260: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_222, torch.bfloat16);  view_222 = None
        view_223: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_88, _shape_param_225);  mm_88 = _shape_param_225 = None
        convert_element_type_261: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_262: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_260, torch.float32);  convert_element_type_260 = None
        convert_element_type_263: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None
        view_224: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg224_1, _shape_param_226);  arg224_1 = _shape_param_226 = None
        convert_element_type_264: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_224, torch.float32);  view_224 = None
        mul_236: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_264, 0.7071067811865476)
        erf_8: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_236);  mul_236 = None
        add_45: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_237: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_45, 0.5);  add_45 = None
        mul_238: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_264, convert_element_type_264)
        mul_239: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_238, -0.5);  mul_238 = None
        exp_16: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_239);  mul_239 = None
        mul_240: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_241: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_264, mul_240);  convert_element_type_264 = mul_240 = None
        add_46: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_237, mul_241);  mul_237 = mul_241 = None
        mul_242: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_263, add_46);  convert_element_type_263 = add_46 = None
        convert_element_type_265: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_242, torch.bfloat16);  mul_242 = None
        view_225: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_265, _shape_param_227);  convert_element_type_265 = _shape_param_227 = None
        mm_90: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_225, arg225_1);  arg225_1 = None
        permute_80: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_91: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_80, arg226_1);  permute_80 = arg226_1 = None
        sum_118: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True, dtype = torch.float32);  view_225 = None
        view_226: "f32[3072]" = torch.ops.aten.view.default(sum_118, _shape_param_228);  sum_118 = _shape_param_228 = None
        convert_element_type_266: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_226, torch.bfloat16);  view_226 = None
        view_227: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_90, _shape_param_229);  mm_90 = _shape_param_229 = None
        convert_element_type_267: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        add_47: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_232, convert_element_type_267);  mul_232 = convert_element_type_267 = None
        convert_element_type_268: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_269: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_266, torch.float32);  convert_element_type_266 = None
        mul_243: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_47, arg227_1);  arg227_1 = None
        mul_244: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_243, 768)
        sum_119: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True)
        mul_245: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_243, arg228_1);  mul_243 = None
        sum_120: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True);  mul_245 = None
        mul_246: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg228_1, sum_120);  sum_120 = None
        sub_50: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_244, sum_119);  mul_244 = sum_119 = None
        sub_51: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_246);  sub_50 = mul_246 = None
        mul_247: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg229_1, sub_51);  arg229_1 = sub_51 = None
        mul_248: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_47, arg228_1);  arg228_1 = None
        sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1]);  mul_248 = None
        sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_47, [0, 1]);  add_47 = None
        convert_element_type_270: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_247, torch.bfloat16)
        convert_element_type_271: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg230_1, torch.bfloat16);  arg230_1 = None
        mul_249: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_271, 1.1111111111111112);  convert_element_type_271 = None
        mul_250: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_270, mul_249);  convert_element_type_270 = mul_249 = None
        view_228: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_250, _shape_param_230);  mul_250 = _shape_param_230 = None
        mm_92: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_228, arg231_1);  arg231_1 = None
        permute_81: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_228, [1, 0])
        mm_93: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_81, arg232_1);  permute_81 = arg232_1 = None
        sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True, dtype = torch.float32);  view_228 = None
        view_229: "f32[768]" = torch.ops.aten.view.default(sum_123, _shape_param_231);  sum_123 = _shape_param_231 = None
        convert_element_type_272: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_229, torch.bfloat16);  view_229 = None
        view_230: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_92, _shape_param_232);  mm_92 = _shape_param_232 = None
        convert_element_type_273: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_274: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_272, torch.float32);  convert_element_type_272 = None
        view_231: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_230, _shape_param_233);  view_230 = _shape_param_233 = None
        permute_82: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        clone_28: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_232: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_234);  clone_28 = _shape_param_234 = None
        bmm_28: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg233_1, view_232);  arg233_1 = None
        bmm_29: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_232, arg234_1);  view_232 = arg234_1 = None
        view_233: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, _shape_param_235);  bmm_28 = _shape_param_235 = None
        view_234: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, _shape_param_236);  bmm_29 = _shape_param_236 = None
        convert_element_type_275: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg235_1, torch.bfloat16);  arg235_1 = None
        mul_251: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_275, 1.1111111111111112);  convert_element_type_275 = None
        mul_252: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_234, mul_251);  view_234 = mul_251 = None
        convert_element_type_276: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_252, torch.float32);  mul_252 = None
        view_235: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg236_1, _shape_param_237);  arg236_1 = _shape_param_237 = None
        mul_253: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_235, 0.125)
        convert_element_type_277: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_253, torch.float32);  mul_253 = None
        convert_element_type_278: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_235, torch.float32);  view_235 = None
        mul_254: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_278, 1);  convert_element_type_278 = None
        sub_52: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_254, arg237_1);  mul_254 = arg237_1 = None
        mul_255: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_52, 0.125);  sub_52 = None
        sub_53: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_277, arg238_1);  convert_element_type_277 = arg238_1 = None
        where_10: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg239_1, mul_255, sub_53);  arg239_1 = mul_255 = sub_53 = None
        exp_17: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_10);  where_10 = None
        div_9: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_17, arg240_1);  exp_17 = arg240_1 = None
        mul_256: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_276, div_9);  convert_element_type_276 = None
        sum_124: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [-1], True)
        neg_7: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_7: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_7, sum_124, mul_256);  neg_7 = sum_124 = mul_256 = None
        convert_element_type_279: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        mul_257: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_279, 0.125);  convert_element_type_279 = None
        view_236: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_257, _shape_param_238);  mul_257 = _shape_param_238 = None
        bmm_30: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg241_1, view_236);  arg241_1 = None
        bmm_31: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_236, arg242_1);  view_236 = arg242_1 = None
        view_237: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, _shape_param_239);  bmm_30 = _shape_param_239 = None
        view_238: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_240);  bmm_31 = _shape_param_240 = None
        permute_83: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_237, [0, 1, 3, 2]);  view_237 = None
        permute_84: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
        clone_29: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_239: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_29, _shape_param_241);  clone_29 = _shape_param_241 = None
        view_240: "bf16[16384, 768]" = torch.ops.aten.view.default(view_239, _shape_param_242);  view_239 = _shape_param_242 = None
        mm_94: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_240, arg243_1);  arg243_1 = None
        permute_85: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_240, [1, 0])
        mm_95: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_85, arg244_1);  permute_85 = None
        sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True, dtype = torch.float32);  view_240 = None
        view_241: "f32[768]" = torch.ops.aten.view.default(sum_125, _shape_param_243);  sum_125 = _shape_param_243 = None
        convert_element_type_280: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_241, torch.bfloat16);  view_241 = None
        view_242: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_94, _shape_param_244);  mm_94 = _shape_param_244 = None
        convert_element_type_281: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        add_48: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_247, convert_element_type_281);  mul_247 = convert_element_type_281 = None
        convert_element_type_282: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_283: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_280, torch.float32);  convert_element_type_280 = None
        permute_86: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_83, [0, 2, 1, 3]);  permute_83 = None
        view_243: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_86, _shape_param_245);  permute_86 = _shape_param_245 = None
        clone_30: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_243, memory_format = torch.contiguous_format);  view_243 = None
        view_244: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_30, _shape_param_246);  clone_30 = _shape_param_246 = None
        mm_96: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_244, arg245_1);  arg245_1 = None
        permute_87: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_244, [1, 0])
        mm_97: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_87, arg244_1);  permute_87 = None
        sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True, dtype = torch.float32);  view_244 = None
        view_245: "f32[768]" = torch.ops.aten.view.default(sum_126, _shape_param_247);  sum_126 = _shape_param_247 = None
        convert_element_type_284: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_245, torch.bfloat16);  view_245 = None
        view_246: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_96, _shape_param_248);  mm_96 = _shape_param_248 = None
        convert_element_type_285: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        add_49: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_48, convert_element_type_285);  add_48 = convert_element_type_285 = None
        convert_element_type_286: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_287: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_284, torch.float32);  convert_element_type_284 = None
        permute_88: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
        clone_31: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_88, memory_format = torch.contiguous_format);  permute_88 = None
        view_247: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_31, _shape_param_249);  clone_31 = _shape_param_249 = None
        view_248: "bf16[16384, 768]" = torch.ops.aten.view.default(view_247, _shape_param_250);  view_247 = _shape_param_250 = None
        mm_98: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_248, arg246_1);  arg246_1 = None
        permute_89: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_248, [1, 0])
        mm_99: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_89, arg244_1);  permute_89 = arg244_1 = None
        sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True, dtype = torch.float32);  view_248 = None
        view_249: "f32[768]" = torch.ops.aten.view.default(sum_127, _shape_param_251);  sum_127 = _shape_param_251 = None
        convert_element_type_288: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_249, torch.bfloat16);  view_249 = None
        view_250: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_98, _shape_param_252);  mm_98 = _shape_param_252 = None
        convert_element_type_289: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_250, torch.float32);  view_250 = None
        add_50: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_49, convert_element_type_289);  add_49 = convert_element_type_289 = None
        convert_element_type_290: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_291: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_288, torch.float32);  convert_element_type_288 = None
        mul_258: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_50, arg247_1);  arg247_1 = None
        mul_259: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_258, 768)
        sum_128: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_258, [2], True)
        mul_260: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_258, arg248_1);  mul_258 = None
        sum_129: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_260, [2], True);  mul_260 = None
        mul_261: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg248_1, sum_129);  sum_129 = None
        sub_54: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_259, sum_128);  mul_259 = sum_128 = None
        sub_55: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_54, mul_261);  sub_54 = mul_261 = None
        mul_262: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg249_1, sub_55);  arg249_1 = sub_55 = None
        mul_263: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_50, arg248_1);  arg248_1 = None
        sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 1]);  mul_263 = None
        sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_50, [0, 1]);  add_50 = None
        convert_element_type_292: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_262, torch.bfloat16)
        convert_element_type_293: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg250_1, torch.bfloat16);  arg250_1 = None
        mul_264: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_293, 1.1111111111111112);  convert_element_type_293 = None
        mul_265: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_292, mul_264);  convert_element_type_292 = mul_264 = None
        view_251: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_265, _shape_param_253);  mul_265 = _shape_param_253 = None
        mm_100: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_251, arg251_1);  arg251_1 = None
        permute_90: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_251, [1, 0])
        mm_101: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_90, arg252_1);  permute_90 = arg252_1 = None
        sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True, dtype = torch.float32);  view_251 = None
        view_252: "f32[768]" = torch.ops.aten.view.default(sum_132, _shape_param_254);  sum_132 = _shape_param_254 = None
        convert_element_type_294: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_252, torch.bfloat16);  view_252 = None
        view_253: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_100, _shape_param_255);  mm_100 = _shape_param_255 = None
        convert_element_type_295: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_296: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_294, torch.float32);  convert_element_type_294 = None
        convert_element_type_297: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        view_254: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg253_1, _shape_param_256);  arg253_1 = _shape_param_256 = None
        convert_element_type_298: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_254, torch.float32);  view_254 = None
        mul_266: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_298, 0.7071067811865476)
        erf_9: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_266);  mul_266 = None
        add_51: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_267: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_51, 0.5);  add_51 = None
        mul_268: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_298, convert_element_type_298)
        mul_269: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_268, -0.5);  mul_268 = None
        exp_18: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_269);  mul_269 = None
        mul_270: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_271: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_298, mul_270);  convert_element_type_298 = mul_270 = None
        add_52: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_267, mul_271);  mul_267 = mul_271 = None
        mul_272: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_297, add_52);  convert_element_type_297 = add_52 = None
        convert_element_type_299: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_272, torch.bfloat16);  mul_272 = None
        view_255: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_299, _shape_param_257);  convert_element_type_299 = _shape_param_257 = None
        mm_102: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_255, arg254_1);  arg254_1 = None
        permute_91: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_255, [1, 0])
        mm_103: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_91, arg255_1);  permute_91 = arg255_1 = None
        sum_133: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True, dtype = torch.float32);  view_255 = None
        view_256: "f32[3072]" = torch.ops.aten.view.default(sum_133, _shape_param_258);  sum_133 = _shape_param_258 = None
        convert_element_type_300: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_256, torch.bfloat16);  view_256 = None
        view_257: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_102, _shape_param_259);  mm_102 = _shape_param_259 = None
        convert_element_type_301: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None
        add_53: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_262, convert_element_type_301);  mul_262 = convert_element_type_301 = None
        convert_element_type_302: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_303: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_300, torch.float32);  convert_element_type_300 = None
        mul_273: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_53, arg256_1);  arg256_1 = None
        mul_274: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_273, 768)
        sum_134: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True)
        mul_275: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_273, arg257_1);  mul_273 = None
        sum_135: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [2], True);  mul_275 = None
        mul_276: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg257_1, sum_135);  sum_135 = None
        sub_56: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_274, sum_134);  mul_274 = sum_134 = None
        sub_57: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_276);  sub_56 = mul_276 = None
        mul_277: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg258_1, sub_57);  arg258_1 = sub_57 = None
        mul_278: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_53, arg257_1);  arg257_1 = None
        sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1]);  mul_278 = None
        sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_53, [0, 1]);  add_53 = None
        convert_element_type_304: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_277, torch.bfloat16)
        convert_element_type_305: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.bfloat16);  arg259_1 = None
        mul_279: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_305, 1.1111111111111112);  convert_element_type_305 = None
        mul_280: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_304, mul_279);  convert_element_type_304 = mul_279 = None
        view_258: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_280, _shape_param_260);  mul_280 = _shape_param_260 = None
        mm_104: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_258, arg260_1);  arg260_1 = None
        permute_92: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_105: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_92, arg261_1);  permute_92 = arg261_1 = None
        sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True, dtype = torch.float32);  view_258 = None
        view_259: "f32[768]" = torch.ops.aten.view.default(sum_138, _shape_param_261);  sum_138 = _shape_param_261 = None
        convert_element_type_306: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_259, torch.bfloat16);  view_259 = None
        view_260: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_104, _shape_param_262);  mm_104 = _shape_param_262 = None
        convert_element_type_307: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_308: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_306, torch.float32);  convert_element_type_306 = None
        view_261: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_260, _shape_param_263);  view_260 = _shape_param_263 = None
        permute_93: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
        clone_32: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_262: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_264);  clone_32 = _shape_param_264 = None
        bmm_32: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg262_1, view_262);  arg262_1 = None
        bmm_33: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_262, arg263_1);  view_262 = arg263_1 = None
        view_263: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, _shape_param_265);  bmm_32 = _shape_param_265 = None
        view_264: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, _shape_param_266);  bmm_33 = _shape_param_266 = None
        convert_element_type_309: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg264_1, torch.bfloat16);  arg264_1 = None
        mul_281: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_309, 1.1111111111111112);  convert_element_type_309 = None
        mul_282: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_264, mul_281);  view_264 = mul_281 = None
        convert_element_type_310: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_282, torch.float32);  mul_282 = None
        view_265: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg265_1, _shape_param_267);  arg265_1 = _shape_param_267 = None
        mul_283: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_265, 0.125)
        convert_element_type_311: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_283, torch.float32);  mul_283 = None
        convert_element_type_312: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        mul_284: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_312, 1);  convert_element_type_312 = None
        sub_58: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_284, arg266_1);  mul_284 = arg266_1 = None
        mul_285: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_58, 0.125);  sub_58 = None
        sub_59: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_311, arg267_1);  convert_element_type_311 = arg267_1 = None
        where_11: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg268_1, mul_285, sub_59);  arg268_1 = mul_285 = sub_59 = None
        exp_19: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_11);  where_11 = None
        div_10: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_19, arg269_1);  exp_19 = arg269_1 = None
        mul_286: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_310, div_10);  convert_element_type_310 = None
        sum_139: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [-1], True)
        neg_8: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_8: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_8, sum_139, mul_286);  neg_8 = sum_139 = mul_286 = None
        convert_element_type_313: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        mul_287: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_313, 0.125);  convert_element_type_313 = None
        view_266: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_287, _shape_param_268);  mul_287 = _shape_param_268 = None
        bmm_34: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg270_1, view_266);  arg270_1 = None
        bmm_35: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_266, arg271_1);  view_266 = arg271_1 = None
        view_267: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, _shape_param_269);  bmm_34 = _shape_param_269 = None
        view_268: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_270);  bmm_35 = _shape_param_270 = None
        permute_94: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_267, [0, 1, 3, 2]);  view_267 = None
        permute_95: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
        clone_33: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_269: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_33, _shape_param_271);  clone_33 = _shape_param_271 = None
        view_270: "bf16[16384, 768]" = torch.ops.aten.view.default(view_269, _shape_param_272);  view_269 = _shape_param_272 = None
        mm_106: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_270, arg272_1);  arg272_1 = None
        permute_96: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_270, [1, 0])
        mm_107: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_96, arg273_1);  permute_96 = None
        sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True, dtype = torch.float32);  view_270 = None
        view_271: "f32[768]" = torch.ops.aten.view.default(sum_140, _shape_param_273);  sum_140 = _shape_param_273 = None
        convert_element_type_314: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_271, torch.bfloat16);  view_271 = None
        view_272: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_106, _shape_param_274);  mm_106 = _shape_param_274 = None
        convert_element_type_315: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_272, torch.float32);  view_272 = None
        add_54: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_277, convert_element_type_315);  mul_277 = convert_element_type_315 = None
        convert_element_type_316: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_317: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_314, torch.float32);  convert_element_type_314 = None
        permute_97: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_94, [0, 2, 1, 3]);  permute_94 = None
        view_273: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_97, _shape_param_275);  permute_97 = _shape_param_275 = None
        clone_34: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_273, memory_format = torch.contiguous_format);  view_273 = None
        view_274: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_34, _shape_param_276);  clone_34 = _shape_param_276 = None
        mm_108: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_274, arg274_1);  arg274_1 = None
        permute_98: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_274, [1, 0])
        mm_109: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_98, arg273_1);  permute_98 = None
        sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True, dtype = torch.float32);  view_274 = None
        view_275: "f32[768]" = torch.ops.aten.view.default(sum_141, _shape_param_277);  sum_141 = _shape_param_277 = None
        convert_element_type_318: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_275, torch.bfloat16);  view_275 = None
        view_276: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_108, _shape_param_278);  mm_108 = _shape_param_278 = None
        convert_element_type_319: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_276, torch.float32);  view_276 = None
        add_55: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_54, convert_element_type_319);  add_54 = convert_element_type_319 = None
        convert_element_type_320: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_321: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_318, torch.float32);  convert_element_type_318 = None
        permute_99: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        clone_35: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_277: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_35, _shape_param_279);  clone_35 = _shape_param_279 = None
        view_278: "bf16[16384, 768]" = torch.ops.aten.view.default(view_277, _shape_param_280);  view_277 = _shape_param_280 = None
        mm_110: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_278, arg275_1);  arg275_1 = None
        permute_100: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_278, [1, 0])
        mm_111: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_100, arg273_1);  permute_100 = arg273_1 = None
        sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True, dtype = torch.float32);  view_278 = None
        view_279: "f32[768]" = torch.ops.aten.view.default(sum_142, _shape_param_281);  sum_142 = _shape_param_281 = None
        convert_element_type_322: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_279, torch.bfloat16);  view_279 = None
        view_280: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_110, _shape_param_282);  mm_110 = _shape_param_282 = None
        convert_element_type_323: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_280, torch.float32);  view_280 = None
        add_56: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_55, convert_element_type_323);  add_55 = convert_element_type_323 = None
        convert_element_type_324: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_325: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_322, torch.float32);  convert_element_type_322 = None
        mul_288: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_56, arg276_1);  arg276_1 = None
        mul_289: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_288, 768)
        sum_143: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True)
        mul_290: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_288, arg277_1);  mul_288 = None
        sum_144: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True);  mul_290 = None
        mul_291: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg277_1, sum_144);  sum_144 = None
        sub_60: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_289, sum_143);  mul_289 = sum_143 = None
        sub_61: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_291);  sub_60 = mul_291 = None
        mul_292: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg278_1, sub_61);  arg278_1 = sub_61 = None
        mul_293: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_56, arg277_1);  arg277_1 = None
        sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 1]);  mul_293 = None
        sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_56, [0, 1]);  add_56 = None
        convert_element_type_326: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_292, torch.bfloat16)
        convert_element_type_327: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg279_1, torch.bfloat16);  arg279_1 = None
        mul_294: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_327, 1.1111111111111112);  convert_element_type_327 = None
        mul_295: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_326, mul_294);  convert_element_type_326 = mul_294 = None
        view_281: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_295, _shape_param_283);  mul_295 = _shape_param_283 = None
        mm_112: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_281, arg280_1);  arg280_1 = None
        permute_101: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_281, [1, 0])
        mm_113: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_101, arg281_1);  permute_101 = arg281_1 = None
        sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True, dtype = torch.float32);  view_281 = None
        view_282: "f32[768]" = torch.ops.aten.view.default(sum_147, _shape_param_284);  sum_147 = _shape_param_284 = None
        convert_element_type_328: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_282, torch.bfloat16);  view_282 = None
        view_283: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_112, _shape_param_285);  mm_112 = _shape_param_285 = None
        convert_element_type_329: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_330: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_328, torch.float32);  convert_element_type_328 = None
        convert_element_type_331: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        view_284: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg282_1, _shape_param_286);  arg282_1 = _shape_param_286 = None
        convert_element_type_332: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None
        mul_296: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_332, 0.7071067811865476)
        erf_10: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_296);  mul_296 = None
        add_57: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_297: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_57, 0.5);  add_57 = None
        mul_298: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_332, convert_element_type_332)
        mul_299: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_298, -0.5);  mul_298 = None
        exp_20: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_299);  mul_299 = None
        mul_300: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_301: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_332, mul_300);  convert_element_type_332 = mul_300 = None
        add_58: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_297, mul_301);  mul_297 = mul_301 = None
        mul_302: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_331, add_58);  convert_element_type_331 = add_58 = None
        convert_element_type_333: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_302, torch.bfloat16);  mul_302 = None
        view_285: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_333, _shape_param_287);  convert_element_type_333 = _shape_param_287 = None
        mm_114: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_285, arg283_1);  arg283_1 = None
        permute_102: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_115: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_102, arg284_1);  permute_102 = arg284_1 = None
        sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True, dtype = torch.float32);  view_285 = None
        view_286: "f32[3072]" = torch.ops.aten.view.default(sum_148, _shape_param_288);  sum_148 = _shape_param_288 = None
        convert_element_type_334: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_286, torch.bfloat16);  view_286 = None
        view_287: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_114, _shape_param_289);  mm_114 = _shape_param_289 = None
        convert_element_type_335: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        add_59: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_292, convert_element_type_335);  mul_292 = convert_element_type_335 = None
        convert_element_type_336: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_337: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_334, torch.float32);  convert_element_type_334 = None
        mul_303: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_59, arg285_1);  arg285_1 = None
        mul_304: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_303, 768)
        sum_149: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True)
        mul_305: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_303, arg286_1);  mul_303 = None
        sum_150: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
        mul_306: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg286_1, sum_150);  sum_150 = None
        sub_62: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_304, sum_149);  mul_304 = sum_149 = None
        sub_63: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_306);  sub_62 = mul_306 = None
        mul_307: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg287_1, sub_63);  arg287_1 = sub_63 = None
        mul_308: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_59, arg286_1);  arg286_1 = None
        sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1]);  mul_308 = None
        sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_59, [0, 1]);  add_59 = None
        convert_element_type_338: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_307, torch.bfloat16)
        convert_element_type_339: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.bfloat16);  arg288_1 = None
        mul_309: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_339, 1.1111111111111112);  convert_element_type_339 = None
        mul_310: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_338, mul_309);  convert_element_type_338 = mul_309 = None
        view_288: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_310, _shape_param_290);  mul_310 = _shape_param_290 = None
        mm_116: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_288, arg289_1);  arg289_1 = None
        permute_103: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_117: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_103, arg290_1);  permute_103 = arg290_1 = None
        sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[768]" = torch.ops.aten.view.default(sum_153, _shape_param_291);  sum_153 = _shape_param_291 = None
        convert_element_type_340: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        view_290: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_116, _shape_param_292);  mm_116 = _shape_param_292 = None
        convert_element_type_341: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_342: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_340, torch.float32);  convert_element_type_340 = None
        view_291: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_290, _shape_param_293);  view_290 = _shape_param_293 = None
        permute_104: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        clone_36: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_292: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_294);  clone_36 = _shape_param_294 = None
        bmm_36: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg291_1, view_292);  arg291_1 = None
        bmm_37: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_292, arg292_1);  view_292 = arg292_1 = None
        view_293: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, _shape_param_295);  bmm_36 = _shape_param_295 = None
        view_294: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_37, _shape_param_296);  bmm_37 = _shape_param_296 = None
        convert_element_type_343: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.bfloat16);  arg293_1 = None
        mul_311: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_343, 1.1111111111111112);  convert_element_type_343 = None
        mul_312: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_294, mul_311);  view_294 = mul_311 = None
        convert_element_type_344: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_312, torch.float32);  mul_312 = None
        view_295: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg294_1, _shape_param_297);  arg294_1 = _shape_param_297 = None
        mul_313: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_295, 0.125)
        convert_element_type_345: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_313, torch.float32);  mul_313 = None
        convert_element_type_346: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_295, torch.float32);  view_295 = None
        mul_314: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_346, 1);  convert_element_type_346 = None
        sub_64: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_314, arg295_1);  mul_314 = arg295_1 = None
        mul_315: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_64, 0.125);  sub_64 = None
        sub_65: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_345, arg296_1);  convert_element_type_345 = arg296_1 = None
        where_12: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg297_1, mul_315, sub_65);  arg297_1 = mul_315 = sub_65 = None
        exp_21: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_12);  where_12 = None
        div_11: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_21, arg298_1);  exp_21 = arg298_1 = None
        mul_316: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_344, div_11);  convert_element_type_344 = None
        sum_154: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [-1], True)
        neg_9: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_9: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_9, sum_154, mul_316);  neg_9 = sum_154 = mul_316 = None
        convert_element_type_347: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        mul_317: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.125);  convert_element_type_347 = None
        view_296: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_317, _shape_param_298);  mul_317 = _shape_param_298 = None
        bmm_38: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg299_1, view_296);  arg299_1 = None
        bmm_39: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_296, arg300_1);  view_296 = arg300_1 = None
        view_297: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_38, _shape_param_299);  bmm_38 = _shape_param_299 = None
        view_298: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_300);  bmm_39 = _shape_param_300 = None
        permute_105: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_297, [0, 1, 3, 2]);  view_297 = None
        permute_106: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_293, [0, 2, 1, 3]);  view_293 = None
        clone_37: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_299: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_37, _shape_param_301);  clone_37 = _shape_param_301 = None
        view_300: "bf16[16384, 768]" = torch.ops.aten.view.default(view_299, _shape_param_302);  view_299 = _shape_param_302 = None
        mm_118: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_300, arg301_1);  arg301_1 = None
        permute_107: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_300, [1, 0])
        mm_119: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_107, arg302_1);  permute_107 = None
        sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True, dtype = torch.float32);  view_300 = None
        view_301: "f32[768]" = torch.ops.aten.view.default(sum_155, _shape_param_303);  sum_155 = _shape_param_303 = None
        convert_element_type_348: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_301, torch.bfloat16);  view_301 = None
        view_302: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_118, _shape_param_304);  mm_118 = _shape_param_304 = None
        convert_element_type_349: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_302, torch.float32);  view_302 = None
        add_60: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_307, convert_element_type_349);  mul_307 = convert_element_type_349 = None
        convert_element_type_350: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_351: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_348, torch.float32);  convert_element_type_348 = None
        permute_108: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_105, [0, 2, 1, 3]);  permute_105 = None
        view_303: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_108, _shape_param_305);  permute_108 = _shape_param_305 = None
        clone_38: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_303, memory_format = torch.contiguous_format);  view_303 = None
        view_304: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_38, _shape_param_306);  clone_38 = _shape_param_306 = None
        mm_120: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_304, arg303_1);  arg303_1 = None
        permute_109: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_304, [1, 0])
        mm_121: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_109, arg302_1);  permute_109 = None
        sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True, dtype = torch.float32);  view_304 = None
        view_305: "f32[768]" = torch.ops.aten.view.default(sum_156, _shape_param_307);  sum_156 = _shape_param_307 = None
        convert_element_type_352: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_305, torch.bfloat16);  view_305 = None
        view_306: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_120, _shape_param_308);  mm_120 = _shape_param_308 = None
        convert_element_type_353: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_306, torch.float32);  view_306 = None
        add_61: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_60, convert_element_type_353);  add_60 = convert_element_type_353 = None
        convert_element_type_354: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_355: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_352, torch.float32);  convert_element_type_352 = None
        permute_110: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
        clone_39: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_307: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_39, _shape_param_309);  clone_39 = _shape_param_309 = None
        view_308: "bf16[16384, 768]" = torch.ops.aten.view.default(view_307, _shape_param_310);  view_307 = _shape_param_310 = None
        mm_122: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_308, arg304_1);  arg304_1 = None
        permute_111: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_123: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_111, arg302_1);  permute_111 = arg302_1 = None
        sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True, dtype = torch.float32);  view_308 = None
        view_309: "f32[768]" = torch.ops.aten.view.default(sum_157, _shape_param_311);  sum_157 = _shape_param_311 = None
        convert_element_type_356: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_309, torch.bfloat16);  view_309 = None
        view_310: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_122, _shape_param_312);  mm_122 = _shape_param_312 = None
        convert_element_type_357: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_310, torch.float32);  view_310 = None
        add_62: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_61, convert_element_type_357);  add_61 = convert_element_type_357 = None
        convert_element_type_358: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_359: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_356, torch.float32);  convert_element_type_356 = None
        mul_318: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_62, arg305_1);  arg305_1 = None
        mul_319: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_318, 768)
        sum_158: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
        mul_320: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_318, arg306_1);  mul_318 = None
        sum_159: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
        mul_321: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg306_1, sum_159);  sum_159 = None
        sub_66: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_319, sum_158);  mul_319 = sum_158 = None
        sub_67: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_321);  sub_66 = mul_321 = None
        mul_322: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg307_1, sub_67);  arg307_1 = sub_67 = None
        mul_323: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_62, arg306_1);  arg306_1 = None
        sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
        sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_62, [0, 1]);  add_62 = None
        convert_element_type_360: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_322, torch.bfloat16)
        convert_element_type_361: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg308_1, torch.bfloat16);  arg308_1 = None
        mul_324: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_361, 1.1111111111111112);  convert_element_type_361 = None
        mul_325: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_360, mul_324);  convert_element_type_360 = mul_324 = None
        view_311: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_325, _shape_param_313);  mul_325 = _shape_param_313 = None
        mm_124: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_311, arg309_1);  arg309_1 = None
        permute_112: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_125: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_112, arg310_1);  permute_112 = arg310_1 = None
        sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True, dtype = torch.float32);  view_311 = None
        view_312: "f32[768]" = torch.ops.aten.view.default(sum_162, _shape_param_314);  sum_162 = _shape_param_314 = None
        convert_element_type_362: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_312, torch.bfloat16);  view_312 = None
        view_313: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_124, _shape_param_315);  mm_124 = _shape_param_315 = None
        convert_element_type_363: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_364: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_362, torch.float32);  convert_element_type_362 = None
        convert_element_type_365: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_313, torch.float32);  view_313 = None
        view_314: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg311_1, _shape_param_316);  arg311_1 = _shape_param_316 = None
        convert_element_type_366: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_314, torch.float32);  view_314 = None
        mul_326: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_366, 0.7071067811865476)
        erf_11: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_326);  mul_326 = None
        add_63: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_327: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
        mul_328: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_366, convert_element_type_366)
        mul_329: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_328, -0.5);  mul_328 = None
        exp_22: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_329);  mul_329 = None
        mul_330: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_331: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_366, mul_330);  convert_element_type_366 = mul_330 = None
        add_64: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_327, mul_331);  mul_327 = mul_331 = None
        mul_332: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_365, add_64);  convert_element_type_365 = add_64 = None
        convert_element_type_367: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_332, torch.bfloat16);  mul_332 = None
        view_315: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_367, _shape_param_317);  convert_element_type_367 = _shape_param_317 = None
        mm_126: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_315, arg312_1);  arg312_1 = None
        permute_113: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_315, [1, 0])
        mm_127: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_113, arg313_1);  permute_113 = arg313_1 = None
        sum_163: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True, dtype = torch.float32);  view_315 = None
        view_316: "f32[3072]" = torch.ops.aten.view.default(sum_163, _shape_param_318);  sum_163 = _shape_param_318 = None
        convert_element_type_368: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_316, torch.bfloat16);  view_316 = None
        view_317: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_126, _shape_param_319);  mm_126 = _shape_param_319 = None
        convert_element_type_369: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None
        add_65: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_322, convert_element_type_369);  mul_322 = convert_element_type_369 = None
        convert_element_type_370: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_371: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_368, torch.float32);  convert_element_type_368 = None
        mul_333: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_65, arg314_1);  arg314_1 = None
        mul_334: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_333, 768)
        sum_164: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [2], True)
        mul_335: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_333, arg315_1);  mul_333 = None
        sum_165: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True);  mul_335 = None
        mul_336: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg315_1, sum_165);  sum_165 = None
        sub_68: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_334, sum_164);  mul_334 = sum_164 = None
        sub_69: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_336);  sub_68 = mul_336 = None
        mul_337: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg316_1, sub_69);  arg316_1 = sub_69 = None
        mul_338: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_65, arg315_1);  arg315_1 = None
        sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1]);  mul_338 = None
        sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_65, [0, 1]);  add_65 = None
        convert_element_type_372: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_337, torch.bfloat16)
        convert_element_type_373: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.bfloat16);  arg317_1 = None
        mul_339: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_373, 1.1111111111111112);  convert_element_type_373 = None
        mul_340: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_372, mul_339);  convert_element_type_372 = mul_339 = None
        view_318: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_340, _shape_param_320);  mul_340 = _shape_param_320 = None
        mm_128: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_318, arg318_1);  arg318_1 = None
        permute_114: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_129: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_114, arg319_1);  permute_114 = arg319_1 = None
        sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True, dtype = torch.float32);  view_318 = None
        view_319: "f32[768]" = torch.ops.aten.view.default(sum_168, _shape_param_321);  sum_168 = _shape_param_321 = None
        convert_element_type_374: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_319, torch.bfloat16);  view_319 = None
        view_320: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_128, _shape_param_322);  mm_128 = _shape_param_322 = None
        convert_element_type_375: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_376: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_374, torch.float32);  convert_element_type_374 = None
        view_321: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_320, _shape_param_323);  view_320 = _shape_param_323 = None
        permute_115: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
        clone_40: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        view_322: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_324);  clone_40 = _shape_param_324 = None
        bmm_40: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg320_1, view_322);  arg320_1 = None
        bmm_41: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_322, arg321_1);  view_322 = arg321_1 = None
        view_323: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, _shape_param_325);  bmm_40 = _shape_param_325 = None
        view_324: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_41, _shape_param_326);  bmm_41 = _shape_param_326 = None
        convert_element_type_377: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg322_1, torch.bfloat16);  arg322_1 = None
        mul_341: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_377, 1.1111111111111112);  convert_element_type_377 = None
        mul_342: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_324, mul_341);  view_324 = mul_341 = None
        convert_element_type_378: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_342, torch.float32);  mul_342 = None
        view_325: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg323_1, _shape_param_327);  arg323_1 = _shape_param_327 = None
        mul_343: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_325, 0.125)
        convert_element_type_379: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_343, torch.float32);  mul_343 = None
        convert_element_type_380: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_325, torch.float32);  view_325 = None
        mul_344: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_380, 1);  convert_element_type_380 = None
        sub_70: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_344, arg324_1);  mul_344 = arg324_1 = None
        mul_345: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_70, 0.125);  sub_70 = None
        sub_71: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_379, arg325_1);  convert_element_type_379 = arg325_1 = None
        where_13: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg326_1, mul_345, sub_71);  arg326_1 = mul_345 = sub_71 = None
        exp_23: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_13);  where_13 = None
        div_12: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_23, arg327_1);  exp_23 = arg327_1 = None
        mul_346: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_378, div_12);  convert_element_type_378 = None
        sum_169: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [-1], True)
        neg_10: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_10: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_10, sum_169, mul_346);  neg_10 = sum_169 = mul_346 = None
        convert_element_type_381: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        mul_347: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_381, 0.125);  convert_element_type_381 = None
        view_326: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_347, _shape_param_328);  mul_347 = _shape_param_328 = None
        bmm_42: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg328_1, view_326);  arg328_1 = None
        bmm_43: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_326, arg329_1);  view_326 = arg329_1 = None
        view_327: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_42, _shape_param_329);  bmm_42 = _shape_param_329 = None
        view_328: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_330);  bmm_43 = _shape_param_330 = None
        permute_116: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_327, [0, 1, 3, 2]);  view_327 = None
        permute_117: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
        clone_41: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_329: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_41, _shape_param_331);  clone_41 = _shape_param_331 = None
        view_330: "bf16[16384, 768]" = torch.ops.aten.view.default(view_329, _shape_param_332);  view_329 = _shape_param_332 = None
        mm_130: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_330, arg330_1);  arg330_1 = None
        permute_118: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_131: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_118, arg331_1);  permute_118 = None
        sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True, dtype = torch.float32);  view_330 = None
        view_331: "f32[768]" = torch.ops.aten.view.default(sum_170, _shape_param_333);  sum_170 = _shape_param_333 = None
        convert_element_type_382: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_331, torch.bfloat16);  view_331 = None
        view_332: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_130, _shape_param_334);  mm_130 = _shape_param_334 = None
        convert_element_type_383: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_332, torch.float32);  view_332 = None
        add_66: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_337, convert_element_type_383);  mul_337 = convert_element_type_383 = None
        convert_element_type_384: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_385: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_382, torch.float32);  convert_element_type_382 = None
        permute_119: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_116, [0, 2, 1, 3]);  permute_116 = None
        view_333: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_119, _shape_param_335);  permute_119 = _shape_param_335 = None
        clone_42: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_333, memory_format = torch.contiguous_format);  view_333 = None
        view_334: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_42, _shape_param_336);  clone_42 = _shape_param_336 = None
        mm_132: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_334, arg332_1);  arg332_1 = None
        permute_120: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_133: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_120, arg331_1);  permute_120 = None
        sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True, dtype = torch.float32);  view_334 = None
        view_335: "f32[768]" = torch.ops.aten.view.default(sum_171, _shape_param_337);  sum_171 = _shape_param_337 = None
        convert_element_type_386: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_335, torch.bfloat16);  view_335 = None
        view_336: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_132, _shape_param_338);  mm_132 = _shape_param_338 = None
        convert_element_type_387: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_336, torch.float32);  view_336 = None
        add_67: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_66, convert_element_type_387);  add_66 = convert_element_type_387 = None
        convert_element_type_388: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_389: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_386, torch.float32);  convert_element_type_386 = None
        permute_121: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        clone_43: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_337: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_43, _shape_param_339);  clone_43 = _shape_param_339 = None
        view_338: "bf16[16384, 768]" = torch.ops.aten.view.default(view_337, _shape_param_340);  view_337 = _shape_param_340 = None
        mm_134: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_338, arg333_1);  arg333_1 = None
        permute_122: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_135: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_122, arg331_1);  permute_122 = arg331_1 = None
        sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True, dtype = torch.float32);  view_338 = None
        view_339: "f32[768]" = torch.ops.aten.view.default(sum_172, _shape_param_341);  sum_172 = _shape_param_341 = None
        convert_element_type_390: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_339, torch.bfloat16);  view_339 = None
        view_340: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_134, _shape_param_342);  mm_134 = _shape_param_342 = None
        convert_element_type_391: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_340, torch.float32);  view_340 = None
        add_68: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_67, convert_element_type_391);  add_67 = convert_element_type_391 = None
        convert_element_type_392: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_393: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_390, torch.float32);  convert_element_type_390 = None
        mul_348: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_68, arg334_1);  arg334_1 = None
        mul_349: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_348, 768)
        sum_173: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True)
        mul_350: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_348, arg335_1);  mul_348 = None
        sum_174: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True);  mul_350 = None
        mul_351: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg335_1, sum_174);  sum_174 = None
        sub_72: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_349, sum_173);  mul_349 = sum_173 = None
        sub_73: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_72, mul_351);  sub_72 = mul_351 = None
        mul_352: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg336_1, sub_73);  arg336_1 = sub_73 = None
        mul_353: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_68, arg335_1);  arg335_1 = None
        sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 1]);  mul_353 = None
        sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_68, [0, 1]);  add_68 = None
        convert_element_type_394: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_352, torch.bfloat16)
        convert_element_type_395: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg337_1, torch.bfloat16);  arg337_1 = None
        mul_354: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_395, 1.1111111111111112);  convert_element_type_395 = None
        mul_355: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_394, mul_354);  convert_element_type_394 = mul_354 = None
        view_341: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_355, _shape_param_343);  mul_355 = _shape_param_343 = None
        mm_136: "bf16[16384, 3072]" = torch.ops.aten.mm.default(view_341, arg338_1);  arg338_1 = None
        permute_123: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_341, [1, 0])
        mm_137: "bf16[768, 3072]" = torch.ops.aten.mm.default(permute_123, arg339_1);  permute_123 = arg339_1 = None
        sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True, dtype = torch.float32);  view_341 = None
        view_342: "f32[768]" = torch.ops.aten.view.default(sum_177, _shape_param_344);  sum_177 = _shape_param_344 = None
        convert_element_type_396: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_342, torch.bfloat16);  view_342 = None
        view_343: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(mm_136, _shape_param_345);  mm_136 = _shape_param_345 = None
        convert_element_type_397: "f32[768, 3072]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_398: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_396, torch.float32);  convert_element_type_396 = None
        convert_element_type_399: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_343, torch.float32);  view_343 = None
        view_344: "bf16[32, 512, 3072]" = torch.ops.aten.view.default(arg340_1, _shape_param_346);  arg340_1 = _shape_param_346 = None
        convert_element_type_400: "f32[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(view_344, torch.float32);  view_344 = None
        mul_356: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_400, 0.7071067811865476)
        erf_12: "f32[32, 512, 3072]" = torch.ops.aten.erf.default(mul_356);  mul_356 = None
        add_69: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_357: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
        mul_358: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_400, convert_element_type_400)
        mul_359: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_358, -0.5);  mul_358 = None
        exp_24: "f32[32, 512, 3072]" = torch.ops.aten.exp.default(mul_359);  mul_359 = None
        mul_360: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_361: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_400, mul_360);  convert_element_type_400 = mul_360 = None
        add_70: "f32[32, 512, 3072]" = torch.ops.aten.add.Tensor(mul_357, mul_361);  mul_357 = mul_361 = None
        mul_362: "f32[32, 512, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_399, add_70);  convert_element_type_399 = add_70 = None
        convert_element_type_401: "bf16[32, 512, 3072]" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None
        view_345: "bf16[16384, 3072]" = torch.ops.aten.view.default(convert_element_type_401, _shape_param_347);  convert_element_type_401 = _shape_param_347 = None
        mm_138: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_345, arg341_1);  arg341_1 = None
        permute_124: "bf16[3072, 16384]" = torch.ops.aten.permute.default(view_345, [1, 0])
        mm_139: "bf16[3072, 768]" = torch.ops.aten.mm.default(permute_124, arg342_1);  permute_124 = arg342_1 = None
        sum_178: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True, dtype = torch.float32);  view_345 = None
        view_346: "f32[3072]" = torch.ops.aten.view.default(sum_178, _shape_param_348);  sum_178 = _shape_param_348 = None
        convert_element_type_402: "bf16[3072]" = torch.ops.prims.convert_element_type.default(view_346, torch.bfloat16);  view_346 = None
        view_347: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_138, _shape_param_349);  mm_138 = _shape_param_349 = None
        convert_element_type_403: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_347, torch.float32);  view_347 = None
        add_71: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_352, convert_element_type_403);  mul_352 = convert_element_type_403 = None
        convert_element_type_404: "f32[3072, 768]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_405: "f32[3072]" = torch.ops.prims.convert_element_type.default(convert_element_type_402, torch.float32);  convert_element_type_402 = None
        mul_363: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_71, arg343_1);  arg343_1 = None
        mul_364: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_363, 768)
        sum_179: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True)
        mul_365: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_363, arg344_1);  mul_363 = None
        sum_180: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True);  mul_365 = None
        mul_366: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg344_1, sum_180);  sum_180 = None
        sub_74: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_364, sum_179);  mul_364 = sum_179 = None
        sub_75: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_366);  sub_74 = mul_366 = None
        mul_367: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg345_1, sub_75);  arg345_1 = sub_75 = None
        mul_368: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_71, arg344_1);  arg344_1 = None
        sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1]);  mul_368 = None
        sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_71, [0, 1]);  add_71 = None
        convert_element_type_406: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(mul_367, torch.bfloat16)
        convert_element_type_407: "bf16[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg346_1, torch.bfloat16);  arg346_1 = None
        mul_369: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_407, 1.1111111111111112);  convert_element_type_407 = None
        mul_370: "bf16[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_406, mul_369);  convert_element_type_406 = mul_369 = None
        view_348: "bf16[16384, 768]" = torch.ops.aten.view.default(mul_370, _shape_param_350);  mul_370 = _shape_param_350 = None
        mm_140: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_348, arg347_1);  arg347_1 = None
        permute_125: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_348, [1, 0])
        mm_141: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_125, arg348_1);  permute_125 = arg348_1 = None
        sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True, dtype = torch.float32);  view_348 = None
        view_349: "f32[768]" = torch.ops.aten.view.default(sum_183, _shape_param_351);  sum_183 = _shape_param_351 = None
        convert_element_type_408: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_349, torch.bfloat16);  view_349 = None
        view_350: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_140, _shape_param_352);  mm_140 = _shape_param_352 = None
        convert_element_type_409: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_410: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_408, torch.float32);  convert_element_type_408 = None
        view_351: "bf16[32, 512, 12, 64]" = torch.ops.aten.view.default(view_350, _shape_param_353);  view_350 = _shape_param_353 = None
        permute_126: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        clone_44: "bf16[32, 12, 512, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_352: "bf16[384, 512, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_354);  clone_44 = _shape_param_354 = None
        bmm_44: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(arg349_1, view_352);  arg349_1 = None
        bmm_45: "bf16[384, 512, 512]" = torch.ops.aten.bmm.default(view_352, arg350_1);  view_352 = arg350_1 = None
        view_353: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_355);  bmm_44 = _shape_param_355 = None
        view_354: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(bmm_45, _shape_param_356);  bmm_45 = _shape_param_356 = None
        convert_element_type_411: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.bfloat16);  arg351_1 = None
        mul_371: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_411, 1.1111111111111112);  convert_element_type_411 = None
        mul_372: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_354, mul_371);  view_354 = mul_371 = None
        convert_element_type_412: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_372, torch.float32);  mul_372 = None
        view_355: "bf16[32, 12, 512, 512]" = torch.ops.aten.view.default(arg352_1, _shape_param_357);  arg352_1 = _shape_param_357 = None
        mul_373: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_355, 0.125)
        convert_element_type_413: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(mul_373, torch.float32);  mul_373 = None
        convert_element_type_414: "f32[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(view_355, torch.float32);  view_355 = None
        mul_374: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_414, 1);  convert_element_type_414 = None
        sub_76: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_374, arg353_1);  mul_374 = arg353_1 = None
        mul_375: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(sub_76, 0.125);  sub_76 = None
        sub_77: "f32[32, 12, 512, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_413, arg354_1);  convert_element_type_413 = arg354_1 = None
        where_14: "f32[32, 12, 512, 512]" = torch.ops.aten.where.self(arg355_1, mul_375, sub_77);  arg355_1 = mul_375 = sub_77 = None
        exp_25: "f32[32, 12, 512, 512]" = torch.ops.aten.exp.default(where_14);  where_14 = None
        div_13: "f32[32, 12, 512, 512]" = torch.ops.aten.div.Tensor(exp_25, arg356_1);  exp_25 = arg356_1 = None
        mul_376: "f32[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_412, div_13);  convert_element_type_412 = None
        sum_184: "f32[32, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [-1], True)
        neg_11: "f32[32, 12, 512, 512]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_11: "f32[32, 12, 512, 512]" = torch.ops.prims.fma.default(neg_11, sum_184, mul_376);  neg_11 = sum_184 = mul_376 = None
        convert_element_type_415: "bf16[32, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        mul_377: "bf16[32, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_415, 0.125);  convert_element_type_415 = None
        view_356: "bf16[384, 512, 512]" = torch.ops.aten.view.default(mul_377, _shape_param_358);  mul_377 = _shape_param_358 = None
        bmm_46: "bf16[384, 64, 512]" = torch.ops.aten.bmm.default(arg357_1, view_356);  arg357_1 = None
        bmm_47: "bf16[384, 512, 64]" = torch.ops.aten.bmm.default(view_356, arg358_1);  view_356 = arg358_1 = None
        view_357: "bf16[32, 12, 64, 512]" = torch.ops.aten.view.default(bmm_46, _shape_param_359);  bmm_46 = _shape_param_359 = None
        view_358: "bf16[32, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_360);  bmm_47 = _shape_param_360 = None
        permute_127: "bf16[32, 12, 512, 64]" = torch.ops.aten.permute.default(view_357, [0, 1, 3, 2]);  view_357 = None
        permute_128: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
        clone_45: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_359: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_45, _shape_param_361);  clone_45 = _shape_param_361 = None
        view_360: "bf16[16384, 768]" = torch.ops.aten.view.default(view_359, _shape_param_362);  view_359 = _shape_param_362 = None
        mm_142: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_360, arg359_1);  arg359_1 = None
        permute_129: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_143: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_129, arg360_1);  permute_129 = None
        sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True, dtype = torch.float32);  view_360 = None
        view_361: "f32[768]" = torch.ops.aten.view.default(sum_185, _shape_param_363);  sum_185 = _shape_param_363 = None
        convert_element_type_416: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_361, torch.bfloat16);  view_361 = None
        view_362: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_142, _shape_param_364);  mm_142 = _shape_param_364 = None
        convert_element_type_417: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_362, torch.float32);  view_362 = None
        add_72: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(mul_367, convert_element_type_417);  mul_367 = convert_element_type_417 = None
        convert_element_type_418: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_419: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_416, torch.float32);  convert_element_type_416 = None
        permute_130: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(permute_127, [0, 2, 1, 3]);  permute_127 = None
        view_363: "bf16[32, 512, 768]" = torch.ops.aten.view.default(permute_130, _shape_param_365);  permute_130 = _shape_param_365 = None
        clone_46: "bf16[32, 512, 768]" = torch.ops.aten.clone.default(view_363, memory_format = torch.contiguous_format);  view_363 = None
        view_364: "bf16[16384, 768]" = torch.ops.aten.view.default(clone_46, _shape_param_366);  clone_46 = _shape_param_366 = None
        mm_144: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_364, arg361_1);  arg361_1 = None
        permute_131: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_145: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_131, arg360_1);  permute_131 = None
        sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True, dtype = torch.float32);  view_364 = None
        view_365: "f32[768]" = torch.ops.aten.view.default(sum_186, _shape_param_367);  sum_186 = _shape_param_367 = None
        convert_element_type_420: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_365, torch.bfloat16);  view_365 = None
        view_366: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_144, _shape_param_368);  mm_144 = _shape_param_368 = None
        convert_element_type_421: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None
        add_73: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_72, convert_element_type_421);  add_72 = convert_element_type_421 = None
        convert_element_type_422: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_423: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_420, torch.float32);  convert_element_type_420 = None
        permute_132: "bf16[32, 512, 12, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
        clone_47: "bf16[32, 512, 12, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_367: "bf16[32, 512, 768]" = torch.ops.aten.view.default(clone_47, _shape_param_369);  clone_47 = _shape_param_369 = None
        view_368: "bf16[16384, 768]" = torch.ops.aten.view.default(view_367, _shape_param_370);  view_367 = _shape_param_370 = None
        mm_146: "bf16[16384, 768]" = torch.ops.aten.mm.default(view_368, arg362_1);  arg362_1 = None
        permute_133: "bf16[768, 16384]" = torch.ops.aten.permute.default(view_368, [1, 0])
        mm_147: "bf16[768, 768]" = torch.ops.aten.mm.default(permute_133, arg360_1);  permute_133 = arg360_1 = None
        sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True, dtype = torch.float32);  view_368 = None
        view_369: "f32[768]" = torch.ops.aten.view.default(sum_187, _shape_param_371);  sum_187 = _shape_param_371 = None
        convert_element_type_424: "bf16[768]" = torch.ops.prims.convert_element_type.default(view_369, torch.bfloat16);  view_369 = None
        view_370: "bf16[32, 512, 768]" = torch.ops.aten.view.default(mm_146, _shape_param_372);  mm_146 = _shape_param_372 = None
        convert_element_type_425: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(view_370, torch.float32);  view_370 = None
        add_74: "f32[32, 512, 768]" = torch.ops.aten.add.Tensor(add_73, convert_element_type_425);  add_73 = convert_element_type_425 = None
        convert_element_type_426: "f32[768, 768]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_427: "f32[768]" = torch.ops.prims.convert_element_type.default(convert_element_type_424, torch.float32);  convert_element_type_424 = None
        convert_element_type_428: "f32[32, 512, 768]" = torch.ops.prims.convert_element_type.default(arg363_1, torch.float32);  arg363_1 = None
        mul_378: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_428, 1.1111111111111112);  convert_element_type_428 = None
        mul_379: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(add_74, mul_378);  add_74 = mul_378 = None
        mul_380: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_379, arg364_1);  arg364_1 = None
        mul_381: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_380, 768)
        sum_188: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True)
        mul_382: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_380, arg365_1);  mul_380 = None
        sum_189: "f32[32, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True);  mul_382 = None
        mul_383: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg365_1, sum_189);  sum_189 = None
        sub_78: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(mul_381, sum_188);  mul_381 = sum_188 = None
        sub_79: "f32[32, 512, 768]" = torch.ops.aten.sub.Tensor(sub_78, mul_383);  sub_78 = mul_383 = None
        mul_384: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(arg366_1, sub_79);  arg366_1 = sub_79 = None
        mul_385: "f32[32, 512, 768]" = torch.ops.aten.mul.Tensor(mul_379, arg365_1);  arg365_1 = None
        sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_385, [0, 1]);  mul_385 = None
        sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1]);  mul_379 = None
        sum_192: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_384, [0], True, dtype = torch.float32)
        full_2: "b8[32, 512, 1]" = torch.ops.aten.full.default(_shape_param_373, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_373 = None
        full_3: "f32[2, 768]" = torch.ops.aten.full.default(_shape_param_374, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_374 = None
        _unsafe_masked_index_put_accumulate: "f32[2, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_3, full_2, [arg367_1], mul_384);  full_3 = full_2 = arg367_1 = None
        ge: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg368_1, 0)
        lt: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg368_1, 1024)
        bitwise_and: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_1: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg368_1, -1)
        bitwise_and_1: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_1);  bitwise_and = ne_1 = None
        unsqueeze_1: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_4: "f32[1024, 768]" = torch.ops.aten.full.default(_shape_param_375, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_375 = None
        _unsafe_masked_index_put_accumulate_1: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_1, [arg368_1], mul_384);  unsqueeze_1 = arg368_1 = None
        ge_1: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg369_1, 0)
        lt_1: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg369_1, 1024)
        bitwise_and_2: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_1, lt_1);  ge_1 = lt_1 = None
        ne_2: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg369_1, -1)
        bitwise_and_3: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_2);  bitwise_and_2 = ne_2 = None
        unsqueeze_2: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        _unsafe_masked_index_put_accumulate_2: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_2, [arg369_1], mul_384);  unsqueeze_2 = arg369_1 = None
        ge_2: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg370_1, 0)
        lt_2: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg370_1, 1024)
        bitwise_and_4: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_3: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg370_1, -1)
        bitwise_and_5: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_3);  bitwise_and_4 = ne_3 = None
        unsqueeze_3: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        _unsafe_masked_index_put_accumulate_3: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_3, [arg370_1], mul_384);  unsqueeze_3 = arg370_1 = None
        ge_3: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg371_1, 0)
        lt_3: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg371_1, 1024)
        bitwise_and_6: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_3, lt_3);  ge_3 = lt_3 = None
        ne_4: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg371_1, -1)
        bitwise_and_7: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_6, ne_4);  bitwise_and_6 = ne_4 = None
        unsqueeze_4: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_7, -1);  bitwise_and_7 = None
        _unsafe_masked_index_put_accumulate_4: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_4, [arg371_1], mul_384);  unsqueeze_4 = arg371_1 = None
        ge_4: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg372_1, 0)
        lt_4: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg372_1, 1024)
        bitwise_and_8: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_4, lt_4);  ge_4 = lt_4 = None
        ne_5: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg372_1, -1)
        bitwise_and_9: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_8, ne_5);  bitwise_and_8 = ne_5 = None
        unsqueeze_5: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_9, -1);  bitwise_and_9 = None
        _unsafe_masked_index_put_accumulate_5: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_5, [arg372_1], mul_384);  unsqueeze_5 = arg372_1 = None
        add_75: "f32[1024, 768]" = torch.ops.aten.add.Tensor(_unsafe_masked_index_put_accumulate_3, _unsafe_masked_index_put_accumulate_5);  _unsafe_masked_index_put_accumulate_3 = _unsafe_masked_index_put_accumulate_5 = None
        ge_5: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg373_1, 0)
        lt_5: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg373_1, 1024)
        bitwise_and_10: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_5, lt_5);  ge_5 = lt_5 = None
        ne_6: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg373_1, -1)
        bitwise_and_11: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_10, ne_6);  bitwise_and_10 = ne_6 = None
        unsqueeze_6: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_11, -1);  bitwise_and_11 = None
        _unsafe_masked_index_put_accumulate_6: "f32[1024, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_6, [arg373_1], mul_384);  full_4 = unsqueeze_6 = arg373_1 = None
        add_76: "f32[1024, 768]" = torch.ops.aten.add.Tensor(_unsafe_masked_index_put_accumulate_4, _unsafe_masked_index_put_accumulate_6);  _unsafe_masked_index_put_accumulate_4 = _unsafe_masked_index_put_accumulate_6 = None
        ge_6: "b8[1, 512]" = torch.ops.aten.ge.Scalar(arg374_1, 0)
        lt_6: "b8[1, 512]" = torch.ops.aten.lt.Scalar(arg374_1, 512)
        bitwise_and_12: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_6, lt_6);  ge_6 = lt_6 = None
        ne_7: "b8[1, 512]" = torch.ops.aten.ne.Scalar(arg374_1, -1)
        bitwise_and_13: "b8[1, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_12, ne_7);  bitwise_and_12 = ne_7 = None
        unsqueeze_7: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_13, -1);  bitwise_and_13 = None
        full_5: "f32[512, 768]" = torch.ops.aten.full.default(_shape_param_376, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_376 = None
        _unsafe_masked_index_put_accumulate_7: "f32[512, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_5, unsqueeze_7, [arg374_1], sum_192);  full_5 = unsqueeze_7 = arg374_1 = sum_192 = None
        ge_7: "b8[32, 512]" = torch.ops.aten.ge.Scalar(arg375_1, 0)
        lt_7: "b8[32, 512]" = torch.ops.aten.lt.Scalar(arg375_1, 30522)
        bitwise_and_14: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(ge_7, lt_7);  ge_7 = lt_7 = None
        ne_8: "b8[32, 512]" = torch.ops.aten.ne.Scalar(arg375_1, 0)
        bitwise_and_15: "b8[32, 512]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_14, ne_8);  bitwise_and_14 = ne_8 = None
        unsqueeze_8: "b8[32, 512, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_15, -1);  bitwise_and_15 = None
        full_6: "f32[30522, 768]" = torch.ops.aten.full.default(_shape_param_377, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_377 = None
        _unsafe_masked_index_put_accumulate_8: "f32[30522, 768]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_6, unsqueeze_8, [arg375_1], mul_384);  full_6 = unsqueeze_8 = arg375_1 = mul_384 = None
        add_77: "f32[30522, 768]" = torch.ops.aten.add.Tensor(convert_element_type_8, _unsafe_masked_index_put_accumulate_8);  convert_element_type_8 = _unsafe_masked_index_put_accumulate_8 = None
        return (convert_element_type_9, sum_5, sum_6, convert_element_type_18, convert_element_type_19, sum_10, sum_11, convert_element_type_23, convert_element_type_24, convert_element_type_30, convert_element_type_31, sum_16, sum_17, convert_element_type_35, convert_element_type_36, convert_element_type_44, convert_element_type_45, convert_element_type_48, convert_element_type_49, convert_element_type_52, convert_element_type_53, sum_25, sum_26, convert_element_type_57, convert_element_type_58, convert_element_type_64, convert_element_type_65, sum_31, sum_32, convert_element_type_69, convert_element_type_70, convert_element_type_78, convert_element_type_79, convert_element_type_82, convert_element_type_83, convert_element_type_86, convert_element_type_87, sum_40, sum_41, convert_element_type_91, convert_element_type_92, convert_element_type_98, convert_element_type_99, sum_46, sum_47, convert_element_type_103, convert_element_type_104, convert_element_type_112, convert_element_type_113, convert_element_type_116, convert_element_type_117, convert_element_type_120, convert_element_type_121, sum_55, sum_56, convert_element_type_125, convert_element_type_126, convert_element_type_132, convert_element_type_133, sum_61, sum_62, convert_element_type_137, convert_element_type_138, convert_element_type_146, convert_element_type_147, convert_element_type_150, convert_element_type_151, convert_element_type_154, convert_element_type_155, sum_70, sum_71, convert_element_type_159, convert_element_type_160, convert_element_type_166, convert_element_type_167, sum_76, sum_77, convert_element_type_171, convert_element_type_172, convert_element_type_180, convert_element_type_181, convert_element_type_184, convert_element_type_185, convert_element_type_188, convert_element_type_189, sum_85, sum_86, convert_element_type_193, convert_element_type_194, convert_element_type_200, convert_element_type_201, sum_91, sum_92, convert_element_type_205, convert_element_type_206, convert_element_type_214, convert_element_type_215, convert_element_type_218, convert_element_type_219, convert_element_type_222, convert_element_type_223, sum_100, sum_101, convert_element_type_227, convert_element_type_228, convert_element_type_234, convert_element_type_235, sum_106, sum_107, convert_element_type_239, convert_element_type_240, convert_element_type_248, convert_element_type_249, convert_element_type_252, convert_element_type_253, convert_element_type_256, convert_element_type_257, sum_115, sum_116, convert_element_type_261, convert_element_type_262, convert_element_type_268, convert_element_type_269, sum_121, sum_122, convert_element_type_273, convert_element_type_274, convert_element_type_282, convert_element_type_283, convert_element_type_286, convert_element_type_287, convert_element_type_290, convert_element_type_291, sum_130, sum_131, convert_element_type_295, convert_element_type_296, convert_element_type_302, convert_element_type_303, sum_136, sum_137, convert_element_type_307, convert_element_type_308, convert_element_type_316, convert_element_type_317, convert_element_type_320, convert_element_type_321, convert_element_type_324, convert_element_type_325, sum_145, sum_146, convert_element_type_329, convert_element_type_330, convert_element_type_336, convert_element_type_337, sum_151, sum_152, convert_element_type_341, convert_element_type_342, convert_element_type_350, convert_element_type_351, convert_element_type_354, convert_element_type_355, convert_element_type_358, convert_element_type_359, sum_160, sum_161, convert_element_type_363, convert_element_type_364, convert_element_type_370, convert_element_type_371, sum_166, sum_167, convert_element_type_375, convert_element_type_376, convert_element_type_384, convert_element_type_385, convert_element_type_388, convert_element_type_389, convert_element_type_392, convert_element_type_393, sum_175, sum_176, convert_element_type_397, convert_element_type_398, convert_element_type_404, convert_element_type_405, sum_181, sum_182, convert_element_type_409, convert_element_type_410, convert_element_type_418, convert_element_type_419, convert_element_type_422, convert_element_type_423, convert_element_type_426, convert_element_type_427, sum_190, sum_191, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate_2, add_75, add_76, _unsafe_masked_index_put_accumulate_7, add_77)



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
