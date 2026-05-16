"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s2_g20
Pattern hash: b820cb5c6494
Shape hash: 37ceb0e0
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg3_1: "i64[]", getitem_1: "f32[1, 64, 1, 1]", arg4_1: "f32[64]", getitem: "f32[1, 64, 1, 1]", arg5_1: "f32[64]", arg8_1: "i64[]", getitem_5: "f32[1, 64, 1, 1]", rsqrt_1: "f32[1, 64, 1, 1]", arg9_1: "f32[64]", getitem_4: "f32[1, 64, 1, 1]", arg10_1: "f32[64]", arg14_1: "i64[]", getitem_7: "f32[1, 128, 1, 1]", rsqrt_2: "f32[1, 128, 1, 1]", arg15_1: "f32[128]", getitem_6: "f32[1, 128, 1, 1]", arg16_1: "f32[128]", arg20_1: "i64[]", getitem_9: "f32[1, 96, 1, 1]", rsqrt_3: "f32[1, 96, 1, 1]", arg21_1: "f32[96]", getitem_8: "f32[1, 96, 1, 1]", arg22_1: "f32[96]", arg26_1: "i64[]", getitem_11: "f32[1, 128, 1, 1]", rsqrt_4: "f32[1, 128, 1, 1]", arg27_1: "f32[128]", getitem_10: "f32[1, 128, 1, 1]", arg28_1: "f32[128]", arg32_1: "i64[]", getitem_13: "f32[1, 128, 1, 1]", rsqrt_5: "f32[1, 128, 1, 1]", arg33_1: "f32[128]", getitem_12: "f32[1, 128, 1, 1]", arg34_1: "f32[128]", arg38_1: "i64[]", getitem_15: "f32[1, 128, 1, 1]", rsqrt_6: "f32[1, 128, 1, 1]", arg39_1: "f32[128]", getitem_14: "f32[1, 128, 1, 1]", arg40_1: "f32[128]", arg44_1: "i64[]", getitem_17: "f32[1, 160, 1, 1]", rsqrt_7: "f32[1, 160, 1, 1]", arg45_1: "f32[160]", getitem_16: "f32[1, 160, 1, 1]", arg46_1: "f32[160]", arg50_1: "i64[]", getitem_19: "f32[1, 128, 1, 1]", rsqrt_8: "f32[1, 128, 1, 1]", arg51_1: "f32[128]", getitem_18: "f32[1, 128, 1, 1]", arg52_1: "f32[128]", arg56_1: "i64[]", getitem_21: "f32[1, 192, 1, 1]", rsqrt_9: "f32[1, 192, 1, 1]", arg57_1: "f32[192]", getitem_20: "f32[1, 192, 1, 1]", arg58_1: "f32[192]", arg62_1: "i64[]", getitem_23: "f32[1, 128, 1, 1]", rsqrt_10: "f32[1, 128, 1, 1]", arg63_1: "f32[128]", getitem_22: "f32[1, 128, 1, 1]", arg64_1: "f32[128]", arg68_1: "i64[]", getitem_25: "f32[1, 224, 1, 1]", rsqrt_11: "f32[1, 224, 1, 1]", arg69_1: "f32[224]", getitem_24: "f32[1, 224, 1, 1]", arg70_1: "f32[224]", arg74_1: "i64[]", getitem_27: "f32[1, 128, 1, 1]", rsqrt_12: "f32[1, 128, 1, 1]", arg75_1: "f32[128]", getitem_26: "f32[1, 128, 1, 1]", arg76_1: "f32[128]", arg80_1: "i64[]", getitem_29: "f32[1, 256, 1, 1]", rsqrt_13: "f32[1, 256, 1, 1]", arg81_1: "f32[256]", getitem_28: "f32[1, 256, 1, 1]", arg82_1: "f32[256]", arg86_1: "i64[]", getitem_31: "f32[1, 128, 1, 1]", rsqrt_14: "f32[1, 128, 1, 1]", arg87_1: "f32[128]", getitem_30: "f32[1, 128, 1, 1]", arg88_1: "f32[128]", arg92_1: "i64[]", getitem_33: "f32[1, 128, 1, 1]", rsqrt_15: "f32[1, 128, 1, 1]", arg93_1: "f32[128]", getitem_32: "f32[1, 128, 1, 1]", arg94_1: "f32[128]", arg98_1: "i64[]", getitem_35: "f32[1, 160, 1, 1]", rsqrt_16: "f32[1, 160, 1, 1]", arg99_1: "f32[160]", getitem_34: "f32[1, 160, 1, 1]", arg100_1: "f32[160]", arg104_1: "i64[]", getitem_37: "f32[1, 128, 1, 1]", rsqrt_17: "f32[1, 128, 1, 1]", arg105_1: "f32[128]", getitem_36: "f32[1, 128, 1, 1]", arg106_1: "f32[128]", arg110_1: "i64[]", getitem_39: "f32[1, 192, 1, 1]", rsqrt_18: "f32[1, 192, 1, 1]", arg111_1: "f32[192]", getitem_38: "f32[1, 192, 1, 1]", arg112_1: "f32[192]", arg116_1: "i64[]", getitem_41: "f32[1, 128, 1, 1]", rsqrt_19: "f32[1, 128, 1, 1]", arg117_1: "f32[128]", getitem_40: "f32[1, 128, 1, 1]", arg118_1: "f32[128]", arg122_1: "i64[]", getitem_43: "f32[1, 224, 1, 1]", rsqrt_20: "f32[1, 224, 1, 1]", arg123_1: "f32[224]", getitem_42: "f32[1, 224, 1, 1]", arg124_1: "f32[224]", arg128_1: "i64[]", getitem_45: "f32[1, 128, 1, 1]", rsqrt_21: "f32[1, 128, 1, 1]", arg129_1: "f32[128]", getitem_44: "f32[1, 128, 1, 1]", arg130_1: "f32[128]", arg134_1: "i64[]", getitem_47: "f32[1, 256, 1, 1]", rsqrt_22: "f32[1, 256, 1, 1]", arg135_1: "f32[256]", getitem_46: "f32[1, 256, 1, 1]", arg136_1: "f32[256]", arg140_1: "i64[]", getitem_49: "f32[1, 128, 1, 1]", rsqrt_23: "f32[1, 128, 1, 1]", arg141_1: "f32[128]", getitem_48: "f32[1, 128, 1, 1]", arg142_1: "f32[128]", arg146_1: "i64[]", getitem_51: "f32[1, 288, 1, 1]", rsqrt_24: "f32[1, 288, 1, 1]", arg147_1: "f32[288]", getitem_50: "f32[1, 288, 1, 1]", arg148_1: "f32[288]", arg152_1: "i64[]", getitem_53: "f32[1, 128, 1, 1]", rsqrt_25: "f32[1, 128, 1, 1]", arg153_1: "f32[128]", getitem_52: "f32[1, 128, 1, 1]", arg154_1: "f32[128]", arg158_1: "i64[]", getitem_55: "f32[1, 320, 1, 1]", rsqrt_26: "f32[1, 320, 1, 1]", arg159_1: "f32[320]", getitem_54: "f32[1, 320, 1, 1]", arg160_1: "f32[320]", arg164_1: "i64[]", getitem_57: "f32[1, 128, 1, 1]", rsqrt_27: "f32[1, 128, 1, 1]", arg165_1: "f32[128]", getitem_56: "f32[1, 128, 1, 1]", arg166_1: "f32[128]", arg170_1: "i64[]", getitem_59: "f32[1, 352, 1, 1]", rsqrt_28: "f32[1, 352, 1, 1]", arg171_1: "f32[352]", getitem_58: "f32[1, 352, 1, 1]", arg172_1: "f32[352]", arg176_1: "i64[]", getitem_61: "f32[1, 128, 1, 1]", rsqrt_29: "f32[1, 128, 1, 1]", arg177_1: "f32[128]", getitem_60: "f32[1, 128, 1, 1]", arg178_1: "f32[128]", arg182_1: "i64[]", getitem_63: "f32[1, 384, 1, 1]", rsqrt_30: "f32[1, 384, 1, 1]", arg183_1: "f32[384]", getitem_62: "f32[1, 384, 1, 1]", arg184_1: "f32[384]", arg188_1: "i64[]", getitem_65: "f32[1, 128, 1, 1]", rsqrt_31: "f32[1, 128, 1, 1]", arg189_1: "f32[128]", getitem_64: "f32[1, 128, 1, 1]", arg190_1: "f32[128]", arg194_1: "i64[]", getitem_67: "f32[1, 416, 1, 1]", rsqrt_32: "f32[1, 416, 1, 1]", arg195_1: "f32[416]", getitem_66: "f32[1, 416, 1, 1]", arg196_1: "f32[416]", arg200_1: "i64[]", getitem_69: "f32[1, 128, 1, 1]", rsqrt_33: "f32[1, 128, 1, 1]", arg201_1: "f32[128]", getitem_68: "f32[1, 128, 1, 1]", arg202_1: "f32[128]", arg206_1: "i64[]", getitem_71: "f32[1, 448, 1, 1]", rsqrt_34: "f32[1, 448, 1, 1]", arg207_1: "f32[448]", getitem_70: "f32[1, 448, 1, 1]", arg208_1: "f32[448]", arg212_1: "i64[]", getitem_73: "f32[1, 128, 1, 1]", rsqrt_35: "f32[1, 128, 1, 1]", arg213_1: "f32[128]", getitem_72: "f32[1, 128, 1, 1]", arg214_1: "f32[128]", arg218_1: "i64[]", getitem_75: "f32[1, 480, 1, 1]", rsqrt_36: "f32[1, 480, 1, 1]", arg219_1: "f32[480]", getitem_74: "f32[1, 480, 1, 1]", arg220_1: "f32[480]", arg224_1: "i64[]", getitem_77: "f32[1, 128, 1, 1]", rsqrt_37: "f32[1, 128, 1, 1]", arg225_1: "f32[128]", getitem_76: "f32[1, 128, 1, 1]", arg226_1: "f32[128]", arg230_1: "i64[]", getitem_79: "f32[1, 512, 1, 1]", rsqrt_38: "f32[1, 512, 1, 1]", arg231_1: "f32[512]", getitem_78: "f32[1, 512, 1, 1]", arg232_1: "f32[512]", arg236_1: "i64[]", getitem_81: "f32[1, 256, 1, 1]", rsqrt_39: "f32[1, 256, 1, 1]", arg237_1: "f32[256]", getitem_80: "f32[1, 256, 1, 1]", arg238_1: "f32[256]", arg242_1: "i64[]", getitem_83: "f32[1, 128, 1, 1]", rsqrt_40: "f32[1, 128, 1, 1]", arg243_1: "f32[128]", getitem_82: "f32[1, 128, 1, 1]", arg244_1: "f32[128]", arg248_1: "i64[]", getitem_85: "f32[1, 288, 1, 1]", rsqrt_41: "f32[1, 288, 1, 1]", arg249_1: "f32[288]", getitem_84: "f32[1, 288, 1, 1]", arg250_1: "f32[288]", arg254_1: "i64[]", getitem_87: "f32[1, 128, 1, 1]", rsqrt_42: "f32[1, 128, 1, 1]", arg255_1: "f32[128]", getitem_86: "f32[1, 128, 1, 1]", arg256_1: "f32[128]", arg260_1: "i64[]", getitem_89: "f32[1, 320, 1, 1]", rsqrt_43: "f32[1, 320, 1, 1]", arg261_1: "f32[320]", getitem_88: "f32[1, 320, 1, 1]", arg262_1: "f32[320]", arg266_1: "i64[]", getitem_91: "f32[1, 128, 1, 1]", rsqrt_44: "f32[1, 128, 1, 1]", arg267_1: "f32[128]", getitem_90: "f32[1, 128, 1, 1]", arg268_1: "f32[128]", arg272_1: "i64[]", getitem_93: "f32[1, 352, 1, 1]", rsqrt_45: "f32[1, 352, 1, 1]", arg273_1: "f32[352]", getitem_92: "f32[1, 352, 1, 1]", arg274_1: "f32[352]", arg278_1: "i64[]", getitem_95: "f32[1, 128, 1, 1]", rsqrt_46: "f32[1, 128, 1, 1]", arg279_1: "f32[128]", getitem_94: "f32[1, 128, 1, 1]", arg280_1: "f32[128]", arg284_1: "i64[]", getitem_97: "f32[1, 384, 1, 1]", rsqrt_47: "f32[1, 384, 1, 1]", arg285_1: "f32[384]", getitem_96: "f32[1, 384, 1, 1]", arg286_1: "f32[384]", arg290_1: "i64[]", getitem_99: "f32[1, 128, 1, 1]", rsqrt_48: "f32[1, 128, 1, 1]", arg291_1: "f32[128]", getitem_98: "f32[1, 128, 1, 1]", arg292_1: "f32[128]", arg296_1: "i64[]", getitem_101: "f32[1, 416, 1, 1]", rsqrt_49: "f32[1, 416, 1, 1]", arg297_1: "f32[416]", getitem_100: "f32[1, 416, 1, 1]", arg298_1: "f32[416]", arg302_1: "i64[]", getitem_103: "f32[1, 128, 1, 1]", rsqrt_50: "f32[1, 128, 1, 1]", arg303_1: "f32[128]", getitem_102: "f32[1, 128, 1, 1]", arg304_1: "f32[128]", arg308_1: "i64[]", getitem_105: "f32[1, 448, 1, 1]", rsqrt_51: "f32[1, 448, 1, 1]", arg309_1: "f32[448]", getitem_104: "f32[1, 448, 1, 1]", arg310_1: "f32[448]", arg314_1: "i64[]", getitem_107: "f32[1, 128, 1, 1]", rsqrt_52: "f32[1, 128, 1, 1]", arg315_1: "f32[128]", getitem_106: "f32[1, 128, 1, 1]", arg316_1: "f32[128]", arg320_1: "i64[]", getitem_109: "f32[1, 480, 1, 1]", rsqrt_53: "f32[1, 480, 1, 1]", arg321_1: "f32[480]", getitem_108: "f32[1, 480, 1, 1]", arg322_1: "f32[480]", arg326_1: "i64[]", getitem_111: "f32[1, 128, 1, 1]", rsqrt_54: "f32[1, 128, 1, 1]", arg327_1: "f32[128]", getitem_110: "f32[1, 128, 1, 1]", arg328_1: "f32[128]", arg332_1: "i64[]", getitem_113: "f32[1, 512, 1, 1]", rsqrt_55: "f32[1, 512, 1, 1]", arg333_1: "f32[512]", getitem_112: "f32[1, 512, 1, 1]", arg334_1: "f32[512]", arg338_1: "i64[]", getitem_115: "f32[1, 128, 1, 1]", rsqrt_56: "f32[1, 128, 1, 1]", arg339_1: "f32[128]", getitem_114: "f32[1, 128, 1, 1]", arg340_1: "f32[128]", arg344_1: "i64[]", getitem_117: "f32[1, 544, 1, 1]", rsqrt_57: "f32[1, 544, 1, 1]", arg345_1: "f32[544]", getitem_116: "f32[1, 544, 1, 1]", arg346_1: "f32[544]", arg350_1: "i64[]", getitem_119: "f32[1, 128, 1, 1]", rsqrt_58: "f32[1, 128, 1, 1]", arg351_1: "f32[128]", getitem_118: "f32[1, 128, 1, 1]", arg352_1: "f32[128]", arg356_1: "i64[]", getitem_121: "f32[1, 576, 1, 1]", rsqrt_59: "f32[1, 576, 1, 1]", arg357_1: "f32[576]", getitem_120: "f32[1, 576, 1, 1]", arg358_1: "f32[576]", arg362_1: "i64[]", getitem_123: "f32[1, 128, 1, 1]", rsqrt_60: "f32[1, 128, 1, 1]", arg363_1: "f32[128]", getitem_122: "f32[1, 128, 1, 1]", arg364_1: "f32[128]", arg368_1: "i64[]", getitem_125: "f32[1, 608, 1, 1]", rsqrt_61: "f32[1, 608, 1, 1]", arg369_1: "f32[608]", getitem_124: "f32[1, 608, 1, 1]", arg370_1: "f32[608]", arg374_1: "i64[]", getitem_127: "f32[1, 128, 1, 1]", rsqrt_62: "f32[1, 128, 1, 1]", arg375_1: "f32[128]", getitem_126: "f32[1, 128, 1, 1]", arg376_1: "f32[128]", arg380_1: "i64[]", getitem_129: "f32[1, 640, 1, 1]", rsqrt_63: "f32[1, 640, 1, 1]", arg381_1: "f32[640]", getitem_128: "f32[1, 640, 1, 1]", arg382_1: "f32[640]", arg386_1: "i64[]", getitem_131: "f32[1, 128, 1, 1]", rsqrt_64: "f32[1, 128, 1, 1]", arg387_1: "f32[128]", getitem_130: "f32[1, 128, 1, 1]", arg388_1: "f32[128]", arg392_1: "i64[]", getitem_133: "f32[1, 672, 1, 1]", rsqrt_65: "f32[1, 672, 1, 1]", arg393_1: "f32[672]", getitem_132: "f32[1, 672, 1, 1]", arg394_1: "f32[672]", arg398_1: "i64[]", getitem_135: "f32[1, 128, 1, 1]", rsqrt_66: "f32[1, 128, 1, 1]", arg399_1: "f32[128]", getitem_134: "f32[1, 128, 1, 1]", arg400_1: "f32[128]", arg404_1: "i64[]", getitem_137: "f32[1, 704, 1, 1]", rsqrt_67: "f32[1, 704, 1, 1]", arg405_1: "f32[704]", getitem_136: "f32[1, 704, 1, 1]", arg406_1: "f32[704]", arg410_1: "i64[]", getitem_139: "f32[1, 128, 1, 1]", rsqrt_68: "f32[1, 128, 1, 1]", arg411_1: "f32[128]", getitem_138: "f32[1, 128, 1, 1]", arg412_1: "f32[128]", arg416_1: "i64[]", getitem_141: "f32[1, 736, 1, 1]", rsqrt_69: "f32[1, 736, 1, 1]", arg417_1: "f32[736]", getitem_140: "f32[1, 736, 1, 1]", arg418_1: "f32[736]", arg422_1: "i64[]", getitem_143: "f32[1, 128, 1, 1]", rsqrt_70: "f32[1, 128, 1, 1]", arg423_1: "f32[128]", getitem_142: "f32[1, 128, 1, 1]", arg424_1: "f32[128]", arg428_1: "i64[]", getitem_145: "f32[1, 768, 1, 1]", rsqrt_71: "f32[1, 768, 1, 1]", arg429_1: "f32[768]", getitem_144: "f32[1, 768, 1, 1]", arg430_1: "f32[768]", arg434_1: "i64[]", getitem_147: "f32[1, 128, 1, 1]", rsqrt_72: "f32[1, 128, 1, 1]", arg435_1: "f32[128]", getitem_146: "f32[1, 128, 1, 1]", arg436_1: "f32[128]", arg440_1: "i64[]", getitem_149: "f32[1, 800, 1, 1]", rsqrt_73: "f32[1, 800, 1, 1]", arg441_1: "f32[800]", getitem_148: "f32[1, 800, 1, 1]", arg442_1: "f32[800]", arg446_1: "i64[]", getitem_151: "f32[1, 128, 1, 1]", rsqrt_74: "f32[1, 128, 1, 1]", arg447_1: "f32[128]", getitem_150: "f32[1, 128, 1, 1]", arg448_1: "f32[128]", arg452_1: "i64[]", getitem_153: "f32[1, 832, 1, 1]", rsqrt_75: "f32[1, 832, 1, 1]", arg453_1: "f32[832]", getitem_152: "f32[1, 832, 1, 1]", arg454_1: "f32[832]", arg458_1: "i64[]", getitem_155: "f32[1, 128, 1, 1]", rsqrt_76: "f32[1, 128, 1, 1]", arg459_1: "f32[128]", getitem_154: "f32[1, 128, 1, 1]", arg460_1: "f32[128]", arg464_1: "i64[]", getitem_157: "f32[1, 864, 1, 1]", rsqrt_77: "f32[1, 864, 1, 1]", arg465_1: "f32[864]", getitem_156: "f32[1, 864, 1, 1]", arg466_1: "f32[864]", arg470_1: "i64[]", getitem_159: "f32[1, 128, 1, 1]", rsqrt_78: "f32[1, 128, 1, 1]", arg471_1: "f32[128]", getitem_158: "f32[1, 128, 1, 1]", arg472_1: "f32[128]", arg476_1: "i64[]", getitem_161: "f32[1, 896, 1, 1]", rsqrt_79: "f32[1, 896, 1, 1]", arg477_1: "f32[896]", getitem_160: "f32[1, 896, 1, 1]", arg478_1: "f32[896]", arg482_1: "i64[]", getitem_163: "f32[1, 128, 1, 1]", rsqrt_80: "f32[1, 128, 1, 1]", arg483_1: "f32[128]", getitem_162: "f32[1, 128, 1, 1]", arg484_1: "f32[128]", arg488_1: "i64[]", getitem_165: "f32[1, 928, 1, 1]", rsqrt_81: "f32[1, 928, 1, 1]", arg489_1: "f32[928]", getitem_164: "f32[1, 928, 1, 1]", arg490_1: "f32[928]", arg494_1: "i64[]", getitem_167: "f32[1, 128, 1, 1]", rsqrt_82: "f32[1, 128, 1, 1]", arg495_1: "f32[128]", getitem_166: "f32[1, 128, 1, 1]", arg496_1: "f32[128]", arg500_1: "i64[]", getitem_169: "f32[1, 960, 1, 1]", rsqrt_83: "f32[1, 960, 1, 1]", arg501_1: "f32[960]", getitem_168: "f32[1, 960, 1, 1]", arg502_1: "f32[960]", arg506_1: "i64[]", getitem_171: "f32[1, 128, 1, 1]", rsqrt_84: "f32[1, 128, 1, 1]", arg507_1: "f32[128]", getitem_170: "f32[1, 128, 1, 1]", arg508_1: "f32[128]", arg512_1: "i64[]", getitem_173: "f32[1, 992, 1, 1]", rsqrt_85: "f32[1, 992, 1, 1]", arg513_1: "f32[992]", getitem_172: "f32[1, 992, 1, 1]", arg514_1: "f32[992]", arg518_1: "i64[]", getitem_175: "f32[1, 128, 1, 1]", rsqrt_86: "f32[1, 128, 1, 1]", arg519_1: "f32[128]", getitem_174: "f32[1, 128, 1, 1]", arg520_1: "f32[128]", arg524_1: "i64[]", getitem_177: "f32[1, 1024, 1, 1]", rsqrt_87: "f32[1, 1024, 1, 1]", arg525_1: "f32[1024]", getitem_176: "f32[1, 1024, 1, 1]", arg526_1: "f32[1024]", arg530_1: "i64[]", getitem_179: "f32[1, 512, 1, 1]", rsqrt_88: "f32[1, 512, 1, 1]", arg531_1: "f32[512]", getitem_178: "f32[1, 512, 1, 1]", arg532_1: "f32[512]", arg536_1: "i64[]", getitem_181: "f32[1, 128, 1, 1]", rsqrt_89: "f32[1, 128, 1, 1]", arg537_1: "f32[128]", getitem_180: "f32[1, 128, 1, 1]", arg538_1: "f32[128]", arg542_1: "i64[]", getitem_183: "f32[1, 544, 1, 1]", rsqrt_90: "f32[1, 544, 1, 1]", arg543_1: "f32[544]", getitem_182: "f32[1, 544, 1, 1]", arg544_1: "f32[544]", arg548_1: "i64[]", getitem_185: "f32[1, 128, 1, 1]", rsqrt_91: "f32[1, 128, 1, 1]", arg549_1: "f32[128]", getitem_184: "f32[1, 128, 1, 1]", arg550_1: "f32[128]", arg554_1: "i64[]", getitem_187: "f32[1, 576, 1, 1]", rsqrt_92: "f32[1, 576, 1, 1]", arg555_1: "f32[576]", getitem_186: "f32[1, 576, 1, 1]", arg556_1: "f32[576]", arg560_1: "i64[]", getitem_189: "f32[1, 128, 1, 1]", rsqrt_93: "f32[1, 128, 1, 1]", arg561_1: "f32[128]", getitem_188: "f32[1, 128, 1, 1]", arg562_1: "f32[128]", arg566_1: "i64[]", getitem_191: "f32[1, 608, 1, 1]", rsqrt_94: "f32[1, 608, 1, 1]", arg567_1: "f32[608]", getitem_190: "f32[1, 608, 1, 1]", arg568_1: "f32[608]", arg572_1: "i64[]", getitem_193: "f32[1, 128, 1, 1]", rsqrt_95: "f32[1, 128, 1, 1]", arg573_1: "f32[128]", getitem_192: "f32[1, 128, 1, 1]", arg574_1: "f32[128]", arg578_1: "i64[]", getitem_195: "f32[1, 640, 1, 1]", rsqrt_96: "f32[1, 640, 1, 1]", arg579_1: "f32[640]", getitem_194: "f32[1, 640, 1, 1]", arg580_1: "f32[640]", arg584_1: "i64[]", getitem_197: "f32[1, 128, 1, 1]", rsqrt_97: "f32[1, 128, 1, 1]", arg585_1: "f32[128]", getitem_196: "f32[1, 128, 1, 1]", arg586_1: "f32[128]", arg590_1: "i64[]", getitem_199: "f32[1, 672, 1, 1]", rsqrt_98: "f32[1, 672, 1, 1]", arg591_1: "f32[672]", getitem_198: "f32[1, 672, 1, 1]", arg592_1: "f32[672]", arg596_1: "i64[]", getitem_201: "f32[1, 128, 1, 1]", rsqrt_99: "f32[1, 128, 1, 1]", arg597_1: "f32[128]", getitem_200: "f32[1, 128, 1, 1]", arg598_1: "f32[128]", arg602_1: "i64[]", getitem_203: "f32[1, 704, 1, 1]", rsqrt_100: "f32[1, 704, 1, 1]", arg603_1: "f32[704]", getitem_202: "f32[1, 704, 1, 1]", arg604_1: "f32[704]", arg608_1: "i64[]", getitem_205: "f32[1, 128, 1, 1]", rsqrt_101: "f32[1, 128, 1, 1]", arg609_1: "f32[128]", getitem_204: "f32[1, 128, 1, 1]", arg610_1: "f32[128]", arg614_1: "i64[]", getitem_207: "f32[1, 736, 1, 1]", rsqrt_102: "f32[1, 736, 1, 1]", arg615_1: "f32[736]", getitem_206: "f32[1, 736, 1, 1]", arg616_1: "f32[736]", arg620_1: "i64[]", getitem_209: "f32[1, 128, 1, 1]", rsqrt_103: "f32[1, 128, 1, 1]", arg621_1: "f32[128]", getitem_208: "f32[1, 128, 1, 1]", arg622_1: "f32[128]", arg626_1: "i64[]", getitem_211: "f32[1, 768, 1, 1]", rsqrt_104: "f32[1, 768, 1, 1]", arg627_1: "f32[768]", getitem_210: "f32[1, 768, 1, 1]", arg628_1: "f32[768]", arg632_1: "i64[]", getitem_213: "f32[1, 128, 1, 1]", rsqrt_105: "f32[1, 128, 1, 1]", arg633_1: "f32[128]", getitem_212: "f32[1, 128, 1, 1]", arg634_1: "f32[128]", arg638_1: "i64[]", getitem_215: "f32[1, 800, 1, 1]", rsqrt_106: "f32[1, 800, 1, 1]", arg639_1: "f32[800]", getitem_214: "f32[1, 800, 1, 1]", arg640_1: "f32[800]", arg644_1: "i64[]", getitem_217: "f32[1, 128, 1, 1]", rsqrt_107: "f32[1, 128, 1, 1]", arg645_1: "f32[128]", getitem_216: "f32[1, 128, 1, 1]", arg646_1: "f32[128]", arg650_1: "i64[]", getitem_219: "f32[1, 832, 1, 1]", rsqrt_108: "f32[1, 832, 1, 1]", arg651_1: "f32[832]", getitem_218: "f32[1, 832, 1, 1]", arg652_1: "f32[832]", arg656_1: "i64[]", getitem_221: "f32[1, 128, 1, 1]", rsqrt_109: "f32[1, 128, 1, 1]", arg657_1: "f32[128]", getitem_220: "f32[1, 128, 1, 1]", arg658_1: "f32[128]", arg662_1: "i64[]", getitem_223: "f32[1, 864, 1, 1]", rsqrt_110: "f32[1, 864, 1, 1]", arg663_1: "f32[864]", getitem_222: "f32[1, 864, 1, 1]", arg664_1: "f32[864]", arg668_1: "i64[]", getitem_225: "f32[1, 128, 1, 1]", rsqrt_111: "f32[1, 128, 1, 1]", arg669_1: "f32[128]", getitem_224: "f32[1, 128, 1, 1]", arg670_1: "f32[128]", arg674_1: "i64[]", getitem_227: "f32[1, 896, 1, 1]", rsqrt_112: "f32[1, 896, 1, 1]", arg675_1: "f32[896]", getitem_226: "f32[1, 896, 1, 1]", arg676_1: "f32[896]", arg680_1: "i64[]", getitem_229: "f32[1, 128, 1, 1]", rsqrt_113: "f32[1, 128, 1, 1]", arg681_1: "f32[128]", getitem_228: "f32[1, 128, 1, 1]", arg682_1: "f32[128]", arg686_1: "i64[]", getitem_231: "f32[1, 928, 1, 1]", rsqrt_114: "f32[1, 928, 1, 1]", arg687_1: "f32[928]", getitem_230: "f32[1, 928, 1, 1]", arg688_1: "f32[928]", arg692_1: "i64[]", getitem_233: "f32[1, 128, 1, 1]", rsqrt_115: "f32[1, 128, 1, 1]", arg693_1: "f32[128]", getitem_232: "f32[1, 128, 1, 1]", arg694_1: "f32[128]", arg698_1: "i64[]", getitem_235: "f32[1, 960, 1, 1]", rsqrt_116: "f32[1, 960, 1, 1]", arg699_1: "f32[960]", getitem_234: "f32[1, 960, 1, 1]", arg700_1: "f32[960]", arg704_1: "i64[]", getitem_237: "f32[1, 128, 1, 1]", rsqrt_117: "f32[1, 128, 1, 1]", arg705_1: "f32[128]", getitem_236: "f32[1, 128, 1, 1]", arg706_1: "f32[128]", arg710_1: "i64[]", getitem_239: "f32[1, 992, 1, 1]", rsqrt_118: "f32[1, 992, 1, 1]", arg711_1: "f32[992]", getitem_238: "f32[1, 992, 1, 1]", arg712_1: "f32[992]", arg716_1: "i64[]", getitem_241: "f32[1, 128, 1, 1]", rsqrt_119: "f32[1, 128, 1, 1]", arg717_1: "f32[128]", getitem_240: "f32[1, 128, 1, 1]", arg718_1: "f32[128]", arg722_1: "i64[]", getitem_242: "f32[1, 1024, 1, 1]", cat_57: "f16[4, 1024, 7, 7]", getitem_243: "f32[1, 1024, 1, 1]", arg723_1: "f32[1024]", arg724_1: "f32[1024]", arg725_1: "f32[1024]", arg726_1: "f32[1024]", arg728_1: "f32[1000]", arg727_1: "f32[1000, 1024]"):
        # No stacktrace found for following nodes
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(arg3_1, 1)
        squeeze_dims: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        mul_tensor: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1);  squeeze_dims = None
        mul_tensor_1: "f32[64]" = torch.ops.aten.mul.Tensor(arg4_1, 0.9)
        add_tensor_1: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_1: "f32[64]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_1, 1.0000199302441455);  squeeze_dims_1 = None
        mul_tensor_3: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[64]" = torch.ops.aten.mul.Tensor(arg5_1, 0.9)
        add_tensor_2: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(arg8_1, 1)
        squeeze_dims_2: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_dims_3: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_5: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 0.1)
        mul_tensor_6: "f32[64]" = torch.ops.aten.mul.Tensor(arg9_1, 0.9)
        add_tensor_4: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_4: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_7: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_dims_4, 1.0000797257434426);  squeeze_dims_4 = None
        mul_tensor_8: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[64]" = torch.ops.aten.mul.Tensor(arg10_1, 0.9)
        add_tensor_5: "f32[64]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(arg14_1, 1)
        squeeze_dims_5: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_6: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_tensor_10: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 0.1)
        mul_tensor_11: "f32[128]" = torch.ops.aten.mul.Tensor(arg15_1, 0.9)
        add_tensor_7: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_7: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_12: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, 1.0000797257434426);  squeeze_dims_7 = None
        mul_tensor_13: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[128]" = torch.ops.aten.mul.Tensor(arg16_1, 0.9)
        add_tensor_8: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(arg20_1, 1)
        squeeze_dims_8: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_dims_9: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_tensor_15: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 0.1)
        mul_tensor_16: "f32[96]" = torch.ops.aten.mul.Tensor(arg21_1, 0.9)
        add_tensor_10: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_10: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_17: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_dims_10, 1.0000797257434426);  squeeze_dims_10 = None
        mul_tensor_18: "f32[96]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[96]" = torch.ops.aten.mul.Tensor(arg22_1, 0.9)
        add_tensor_11: "f32[96]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(arg26_1, 1)
        squeeze_dims_11: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_dims_12: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_20: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 0.1)
        mul_tensor_21: "f32[128]" = torch.ops.aten.mul.Tensor(arg27_1, 0.9)
        add_tensor_13: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_13: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_22: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_13, 1.0000797257434426);  squeeze_dims_13 = None
        mul_tensor_23: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[128]" = torch.ops.aten.mul.Tensor(arg28_1, 0.9)
        add_tensor_14: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(arg32_1, 1)
        squeeze_dims_14: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_dims_15: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_25: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 0.1)
        mul_tensor_26: "f32[128]" = torch.ops.aten.mul.Tensor(arg33_1, 0.9)
        add_tensor_16: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_16: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_27: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_16, 1.0000797257434426);  squeeze_dims_16 = None
        mul_tensor_28: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[128]" = torch.ops.aten.mul.Tensor(arg34_1, 0.9)
        add_tensor_17: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(arg38_1, 1)
        squeeze_dims_17: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_dims_18: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_tensor_30: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 0.1)
        mul_tensor_31: "f32[128]" = torch.ops.aten.mul.Tensor(arg39_1, 0.9)
        add_tensor_19: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_19: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_32: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_19, 1.0000797257434426);  squeeze_dims_19 = None
        mul_tensor_33: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[128]" = torch.ops.aten.mul.Tensor(arg40_1, 0.9)
        add_tensor_20: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(arg44_1, 1)
        squeeze_dims_20: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_dims_21: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_tensor_35: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_20, 0.1)
        mul_tensor_36: "f32[160]" = torch.ops.aten.mul.Tensor(arg45_1, 0.9)
        add_tensor_22: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_22: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_37: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_22, 1.0000797257434426);  squeeze_dims_22 = None
        mul_tensor_38: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[160]" = torch.ops.aten.mul.Tensor(arg46_1, 0.9)
        add_tensor_23: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(arg50_1, 1)
        squeeze_dims_23: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_dims_24: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_23, 0.1)
        mul_tensor_41: "f32[128]" = torch.ops.aten.mul.Tensor(arg51_1, 0.9)
        add_tensor_25: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_25: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_42: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_25, 1.0000797257434426);  squeeze_dims_25 = None
        mul_tensor_43: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[128]" = torch.ops.aten.mul.Tensor(arg52_1, 0.9)
        add_tensor_26: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(arg56_1, 1)
        squeeze_dims_26: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_dims_27: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_45: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_26, 0.1)
        mul_tensor_46: "f32[192]" = torch.ops.aten.mul.Tensor(arg57_1, 0.9)
        add_tensor_28: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_28: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_47: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_28, 1.0000797257434426);  squeeze_dims_28 = None
        mul_tensor_48: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[192]" = torch.ops.aten.mul.Tensor(arg58_1, 0.9)
        add_tensor_29: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(arg62_1, 1)
        squeeze_dims_29: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_dims_30: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_50: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_29, 0.1)
        mul_tensor_51: "f32[128]" = torch.ops.aten.mul.Tensor(arg63_1, 0.9)
        add_tensor_31: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_31: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_tensor_52: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_31, 1.0000797257434426);  squeeze_dims_31 = None
        mul_tensor_53: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[128]" = torch.ops.aten.mul.Tensor(arg64_1, 0.9)
        add_tensor_32: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(arg68_1, 1)
        squeeze_dims_32: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_dims_33: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_tensor_55: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_dims_32, 0.1)
        mul_tensor_56: "f32[224]" = torch.ops.aten.mul.Tensor(arg69_1, 0.9)
        add_tensor_34: "f32[224]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_34: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_57: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_dims_34, 1.0000797257434426);  squeeze_dims_34 = None
        mul_tensor_58: "f32[224]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[224]" = torch.ops.aten.mul.Tensor(arg70_1, 0.9)
        add_tensor_35: "f32[224]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(arg74_1, 1)
        squeeze_dims_35: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_dims_36: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_35, 0.1)
        mul_tensor_61: "f32[128]" = torch.ops.aten.mul.Tensor(arg75_1, 0.9)
        add_tensor_37: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_37: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_62: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_37, 1.0000797257434426);  squeeze_dims_37 = None
        mul_tensor_63: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[128]" = torch.ops.aten.mul.Tensor(arg76_1, 0.9)
        add_tensor_38: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(arg80_1, 1)
        squeeze_dims_38: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_dims_39: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_tensor_65: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 0.1)
        mul_tensor_66: "f32[256]" = torch.ops.aten.mul.Tensor(arg81_1, 0.9)
        add_tensor_40: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_40: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_67: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_40, 1.0000797257434426);  squeeze_dims_40 = None
        mul_tensor_68: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[256]" = torch.ops.aten.mul.Tensor(arg82_1, 0.9)
        add_tensor_41: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(arg86_1, 1)
        squeeze_dims_41: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_dims_42: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_41, 0.1)
        mul_tensor_71: "f32[128]" = torch.ops.aten.mul.Tensor(arg87_1, 0.9)
        add_tensor_43: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_43: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_72: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_43, 1.0003189792663476);  squeeze_dims_43 = None
        mul_tensor_73: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[128]" = torch.ops.aten.mul.Tensor(arg88_1, 0.9)
        add_tensor_44: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(arg92_1, 1)
        squeeze_dims_44: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_dims_45: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_tensor_75: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_44, 0.1)
        mul_tensor_76: "f32[128]" = torch.ops.aten.mul.Tensor(arg93_1, 0.9)
        add_tensor_46: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_46: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_77: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_46, 1.0003189792663476);  squeeze_dims_46 = None
        mul_tensor_78: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[128]" = torch.ops.aten.mul.Tensor(arg94_1, 0.9)
        add_tensor_47: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(arg98_1, 1)
        squeeze_dims_47: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_dims_48: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_tensor_80: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_47, 0.1)
        mul_tensor_81: "f32[160]" = torch.ops.aten.mul.Tensor(arg99_1, 0.9)
        add_tensor_49: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_49: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_82: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_49, 1.0003189792663476);  squeeze_dims_49 = None
        mul_tensor_83: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[160]" = torch.ops.aten.mul.Tensor(arg100_1, 0.9)
        add_tensor_50: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(arg104_1, 1)
        squeeze_dims_50: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_dims_51: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_tensor_85: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_50, 0.1)
        mul_tensor_86: "f32[128]" = torch.ops.aten.mul.Tensor(arg105_1, 0.9)
        add_tensor_52: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_52: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_tensor_87: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_52, 1.0003189792663476);  squeeze_dims_52 = None
        mul_tensor_88: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.1);  mul_tensor_87 = None
        mul_tensor_89: "f32[128]" = torch.ops.aten.mul.Tensor(arg106_1, 0.9)
        add_tensor_53: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(arg110_1, 1)
        squeeze_dims_53: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_dims_54: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_tensor_90: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_53, 0.1)
        mul_tensor_91: "f32[192]" = torch.ops.aten.mul.Tensor(arg111_1, 0.9)
        add_tensor_55: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_55: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_tensor_92: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_dims_55, 1.0003189792663476);  squeeze_dims_55 = None
        mul_tensor_93: "f32[192]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.1);  mul_tensor_92 = None
        mul_tensor_94: "f32[192]" = torch.ops.aten.mul.Tensor(arg112_1, 0.9)
        add_tensor_56: "f32[192]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(arg116_1, 1)
        squeeze_dims_56: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_dims_57: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_tensor_95: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_56, 0.1)
        mul_tensor_96: "f32[128]" = torch.ops.aten.mul.Tensor(arg117_1, 0.9)
        add_tensor_58: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_58: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_tensor_97: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_58, 1.0003189792663476);  squeeze_dims_58 = None
        mul_tensor_98: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.1);  mul_tensor_97 = None
        mul_tensor_99: "f32[128]" = torch.ops.aten.mul.Tensor(arg118_1, 0.9)
        add_tensor_59: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(arg122_1, 1)
        squeeze_dims_59: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_dims_60: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_tensor_100: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_dims_59, 0.1)
        mul_tensor_101: "f32[224]" = torch.ops.aten.mul.Tensor(arg123_1, 0.9)
        add_tensor_61: "f32[224]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_61: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_tensor_102: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_dims_61, 1.0003189792663476);  squeeze_dims_61 = None
        mul_tensor_103: "f32[224]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.1);  mul_tensor_102 = None
        mul_tensor_104: "f32[224]" = torch.ops.aten.mul.Tensor(arg124_1, 0.9)
        add_tensor_62: "f32[224]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None
        add_tensor_63: "i64[]" = torch.ops.aten.add.Tensor(arg128_1, 1)
        squeeze_dims_62: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_dims_63: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_tensor_105: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_62, 0.1)
        mul_tensor_106: "f32[128]" = torch.ops.aten.mul.Tensor(arg129_1, 0.9)
        add_tensor_64: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_105, mul_tensor_106);  mul_tensor_105 = mul_tensor_106 = None
        squeeze_dims_64: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_tensor_107: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_64, 1.0003189792663476);  squeeze_dims_64 = None
        mul_tensor_108: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_107, 0.1);  mul_tensor_107 = None
        mul_tensor_109: "f32[128]" = torch.ops.aten.mul.Tensor(arg130_1, 0.9)
        add_tensor_65: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None
        add_tensor_66: "i64[]" = torch.ops.aten.add.Tensor(arg134_1, 1)
        squeeze_dims_65: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_dims_66: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_tensor_110: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_65, 0.1)
        mul_tensor_111: "f32[256]" = torch.ops.aten.mul.Tensor(arg135_1, 0.9)
        add_tensor_67: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        squeeze_dims_67: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_tensor_112: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_67, 1.0003189792663476);  squeeze_dims_67 = None
        mul_tensor_113: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_112, 0.1);  mul_tensor_112 = None
        mul_tensor_114: "f32[256]" = torch.ops.aten.mul.Tensor(arg136_1, 0.9)
        add_tensor_68: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None
        add_tensor_69: "i64[]" = torch.ops.aten.add.Tensor(arg140_1, 1)
        squeeze_dims_68: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_dims_69: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_tensor_115: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_68, 0.1)
        mul_tensor_116: "f32[128]" = torch.ops.aten.mul.Tensor(arg141_1, 0.9)
        add_tensor_70: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        squeeze_dims_70: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_tensor_117: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_70, 1.0003189792663476);  squeeze_dims_70 = None
        mul_tensor_118: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_117, 0.1);  mul_tensor_117 = None
        mul_tensor_119: "f32[128]" = torch.ops.aten.mul.Tensor(arg142_1, 0.9)
        add_tensor_71: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None
        add_tensor_72: "i64[]" = torch.ops.aten.add.Tensor(arg146_1, 1)
        squeeze_dims_71: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_dims_72: "f32[288]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_tensor_120: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_dims_71, 0.1)
        mul_tensor_121: "f32[288]" = torch.ops.aten.mul.Tensor(arg147_1, 0.9)
        add_tensor_73: "f32[288]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        squeeze_dims_73: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_tensor_122: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_dims_73, 1.0003189792663476);  squeeze_dims_73 = None
        mul_tensor_123: "f32[288]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 0.1);  mul_tensor_122 = None
        mul_tensor_124: "f32[288]" = torch.ops.aten.mul.Tensor(arg148_1, 0.9)
        add_tensor_74: "f32[288]" = torch.ops.aten.add.Tensor(mul_tensor_123, mul_tensor_124);  mul_tensor_123 = mul_tensor_124 = None
        add_tensor_75: "i64[]" = torch.ops.aten.add.Tensor(arg152_1, 1)
        squeeze_dims_74: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_dims_75: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_tensor_125: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_74, 0.1)
        mul_tensor_126: "f32[128]" = torch.ops.aten.mul.Tensor(arg153_1, 0.9)
        add_tensor_76: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        squeeze_dims_76: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_tensor_127: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_76, 1.0003189792663476);  squeeze_dims_76 = None
        mul_tensor_128: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_127, 0.1);  mul_tensor_127 = None
        mul_tensor_129: "f32[128]" = torch.ops.aten.mul.Tensor(arg154_1, 0.9)
        add_tensor_77: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None
        add_tensor_78: "i64[]" = torch.ops.aten.add.Tensor(arg158_1, 1)
        squeeze_dims_77: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_dims_78: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_130: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_77, 0.1)
        mul_tensor_131: "f32[320]" = torch.ops.aten.mul.Tensor(arg159_1, 0.9)
        add_tensor_79: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        squeeze_dims_79: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_tensor_132: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_79, 1.0003189792663476);  squeeze_dims_79 = None
        mul_tensor_133: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_132, 0.1);  mul_tensor_132 = None
        mul_tensor_134: "f32[320]" = torch.ops.aten.mul.Tensor(arg160_1, 0.9)
        add_tensor_80: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_133, mul_tensor_134);  mul_tensor_133 = mul_tensor_134 = None
        add_tensor_81: "i64[]" = torch.ops.aten.add.Tensor(arg164_1, 1)
        squeeze_dims_80: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_dims_81: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_tensor_135: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_80, 0.1)
        mul_tensor_136: "f32[128]" = torch.ops.aten.mul.Tensor(arg165_1, 0.9)
        add_tensor_82: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_135, mul_tensor_136);  mul_tensor_135 = mul_tensor_136 = None
        squeeze_dims_82: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_tensor_137: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_82, 1.0003189792663476);  squeeze_dims_82 = None
        mul_tensor_138: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_137, 0.1);  mul_tensor_137 = None
        mul_tensor_139: "f32[128]" = torch.ops.aten.mul.Tensor(arg166_1, 0.9)
        add_tensor_83: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_138, mul_tensor_139);  mul_tensor_138 = mul_tensor_139 = None
        add_tensor_84: "i64[]" = torch.ops.aten.add.Tensor(arg170_1, 1)
        squeeze_dims_83: "f32[352]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_dims_84: "f32[352]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_tensor_140: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_dims_83, 0.1)
        mul_tensor_141: "f32[352]" = torch.ops.aten.mul.Tensor(arg171_1, 0.9)
        add_tensor_85: "f32[352]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        squeeze_dims_85: "f32[352]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_tensor_142: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_dims_85, 1.0003189792663476);  squeeze_dims_85 = None
        mul_tensor_143: "f32[352]" = torch.ops.aten.mul.Tensor(mul_tensor_142, 0.1);  mul_tensor_142 = None
        mul_tensor_144: "f32[352]" = torch.ops.aten.mul.Tensor(arg172_1, 0.9)
        add_tensor_86: "f32[352]" = torch.ops.aten.add.Tensor(mul_tensor_143, mul_tensor_144);  mul_tensor_143 = mul_tensor_144 = None
        add_tensor_87: "i64[]" = torch.ops.aten.add.Tensor(arg176_1, 1)
        squeeze_dims_86: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_dims_87: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_tensor_145: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_86, 0.1)
        mul_tensor_146: "f32[128]" = torch.ops.aten.mul.Tensor(arg177_1, 0.9)
        add_tensor_88: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_145, mul_tensor_146);  mul_tensor_145 = mul_tensor_146 = None
        squeeze_dims_88: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_tensor_147: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_88, 1.0003189792663476);  squeeze_dims_88 = None
        mul_tensor_148: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_147, 0.1);  mul_tensor_147 = None
        mul_tensor_149: "f32[128]" = torch.ops.aten.mul.Tensor(arg178_1, 0.9)
        add_tensor_89: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None
        add_tensor_90: "i64[]" = torch.ops.aten.add.Tensor(arg182_1, 1)
        squeeze_dims_89: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_dims_90: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_tensor_150: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_89, 0.1)
        mul_tensor_151: "f32[384]" = torch.ops.aten.mul.Tensor(arg183_1, 0.9)
        add_tensor_91: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_150, mul_tensor_151);  mul_tensor_150 = mul_tensor_151 = None
        squeeze_dims_91: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_tensor_152: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_91, 1.0003189792663476);  squeeze_dims_91 = None
        mul_tensor_153: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_152, 0.1);  mul_tensor_152 = None
        mul_tensor_154: "f32[384]" = torch.ops.aten.mul.Tensor(arg184_1, 0.9)
        add_tensor_92: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_153, mul_tensor_154);  mul_tensor_153 = mul_tensor_154 = None
        add_tensor_93: "i64[]" = torch.ops.aten.add.Tensor(arg188_1, 1)
        squeeze_dims_92: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_dims_93: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_tensor_155: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_92, 0.1)
        mul_tensor_156: "f32[128]" = torch.ops.aten.mul.Tensor(arg189_1, 0.9)
        add_tensor_94: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_155, mul_tensor_156);  mul_tensor_155 = mul_tensor_156 = None
        squeeze_dims_94: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_tensor_157: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_94, 1.0003189792663476);  squeeze_dims_94 = None
        mul_tensor_158: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_157, 0.1);  mul_tensor_157 = None
        mul_tensor_159: "f32[128]" = torch.ops.aten.mul.Tensor(arg190_1, 0.9)
        add_tensor_95: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_158, mul_tensor_159);  mul_tensor_158 = mul_tensor_159 = None
        add_tensor_96: "i64[]" = torch.ops.aten.add.Tensor(arg194_1, 1)
        squeeze_dims_95: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_dims_96: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_tensor_160: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_dims_95, 0.1)
        mul_tensor_161: "f32[416]" = torch.ops.aten.mul.Tensor(arg195_1, 0.9)
        add_tensor_97: "f32[416]" = torch.ops.aten.add.Tensor(mul_tensor_160, mul_tensor_161);  mul_tensor_160 = mul_tensor_161 = None
        squeeze_dims_97: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_tensor_162: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_dims_97, 1.0003189792663476);  squeeze_dims_97 = None
        mul_tensor_163: "f32[416]" = torch.ops.aten.mul.Tensor(mul_tensor_162, 0.1);  mul_tensor_162 = None
        mul_tensor_164: "f32[416]" = torch.ops.aten.mul.Tensor(arg196_1, 0.9)
        add_tensor_98: "f32[416]" = torch.ops.aten.add.Tensor(mul_tensor_163, mul_tensor_164);  mul_tensor_163 = mul_tensor_164 = None
        add_tensor_99: "i64[]" = torch.ops.aten.add.Tensor(arg200_1, 1)
        squeeze_dims_98: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_dims_99: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_tensor_165: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_98, 0.1)
        mul_tensor_166: "f32[128]" = torch.ops.aten.mul.Tensor(arg201_1, 0.9)
        add_tensor_100: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_165, mul_tensor_166);  mul_tensor_165 = mul_tensor_166 = None
        squeeze_dims_100: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_tensor_167: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_100, 1.0003189792663476);  squeeze_dims_100 = None
        mul_tensor_168: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_167, 0.1);  mul_tensor_167 = None
        mul_tensor_169: "f32[128]" = torch.ops.aten.mul.Tensor(arg202_1, 0.9)
        add_tensor_101: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_168, mul_tensor_169);  mul_tensor_168 = mul_tensor_169 = None
        add_tensor_102: "i64[]" = torch.ops.aten.add.Tensor(arg206_1, 1)
        squeeze_dims_101: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_dims_102: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_tensor_170: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_dims_101, 0.1)
        mul_tensor_171: "f32[448]" = torch.ops.aten.mul.Tensor(arg207_1, 0.9)
        add_tensor_103: "f32[448]" = torch.ops.aten.add.Tensor(mul_tensor_170, mul_tensor_171);  mul_tensor_170 = mul_tensor_171 = None
        squeeze_dims_103: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_tensor_172: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_dims_103, 1.0003189792663476);  squeeze_dims_103 = None
        mul_tensor_173: "f32[448]" = torch.ops.aten.mul.Tensor(mul_tensor_172, 0.1);  mul_tensor_172 = None
        mul_tensor_174: "f32[448]" = torch.ops.aten.mul.Tensor(arg208_1, 0.9)
        add_tensor_104: "f32[448]" = torch.ops.aten.add.Tensor(mul_tensor_173, mul_tensor_174);  mul_tensor_173 = mul_tensor_174 = None
        add_tensor_105: "i64[]" = torch.ops.aten.add.Tensor(arg212_1, 1)
        squeeze_dims_104: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_dims_105: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_tensor_175: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_104, 0.1)
        mul_tensor_176: "f32[128]" = torch.ops.aten.mul.Tensor(arg213_1, 0.9)
        add_tensor_106: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_175, mul_tensor_176);  mul_tensor_175 = mul_tensor_176 = None
        squeeze_dims_106: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_tensor_177: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_106, 1.0003189792663476);  squeeze_dims_106 = None
        mul_tensor_178: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_177, 0.1);  mul_tensor_177 = None
        mul_tensor_179: "f32[128]" = torch.ops.aten.mul.Tensor(arg214_1, 0.9)
        add_tensor_107: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_178, mul_tensor_179);  mul_tensor_178 = mul_tensor_179 = None
        add_tensor_108: "i64[]" = torch.ops.aten.add.Tensor(arg218_1, 1)
        squeeze_dims_107: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_dims_108: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_tensor_180: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_107, 0.1)
        mul_tensor_181: "f32[480]" = torch.ops.aten.mul.Tensor(arg219_1, 0.9)
        add_tensor_109: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_180, mul_tensor_181);  mul_tensor_180 = mul_tensor_181 = None
        squeeze_dims_109: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_tensor_182: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_109, 1.0003189792663476);  squeeze_dims_109 = None
        mul_tensor_183: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_182, 0.1);  mul_tensor_182 = None
        mul_tensor_184: "f32[480]" = torch.ops.aten.mul.Tensor(arg220_1, 0.9)
        add_tensor_110: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_183, mul_tensor_184);  mul_tensor_183 = mul_tensor_184 = None
        add_tensor_111: "i64[]" = torch.ops.aten.add.Tensor(arg224_1, 1)
        squeeze_dims_110: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_dims_111: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_tensor_185: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_110, 0.1)
        mul_tensor_186: "f32[128]" = torch.ops.aten.mul.Tensor(arg225_1, 0.9)
        add_tensor_112: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_185, mul_tensor_186);  mul_tensor_185 = mul_tensor_186 = None
        squeeze_dims_112: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_tensor_187: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_112, 1.0003189792663476);  squeeze_dims_112 = None
        mul_tensor_188: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_187, 0.1);  mul_tensor_187 = None
        mul_tensor_189: "f32[128]" = torch.ops.aten.mul.Tensor(arg226_1, 0.9)
        add_tensor_113: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_188, mul_tensor_189);  mul_tensor_188 = mul_tensor_189 = None
        add_tensor_114: "i64[]" = torch.ops.aten.add.Tensor(arg230_1, 1)
        squeeze_dims_113: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_dims_114: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_tensor_190: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_113, 0.1)
        mul_tensor_191: "f32[512]" = torch.ops.aten.mul.Tensor(arg231_1, 0.9)
        add_tensor_115: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_190, mul_tensor_191);  mul_tensor_190 = mul_tensor_191 = None
        squeeze_dims_115: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_tensor_192: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_115, 1.0003189792663476);  squeeze_dims_115 = None
        mul_tensor_193: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_192, 0.1);  mul_tensor_192 = None
        mul_tensor_194: "f32[512]" = torch.ops.aten.mul.Tensor(arg232_1, 0.9)
        add_tensor_116: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_193, mul_tensor_194);  mul_tensor_193 = mul_tensor_194 = None
        add_tensor_117: "i64[]" = torch.ops.aten.add.Tensor(arg236_1, 1)
        squeeze_dims_116: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_dims_117: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_tensor_195: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_116, 0.1)
        mul_tensor_196: "f32[256]" = torch.ops.aten.mul.Tensor(arg237_1, 0.9)
        add_tensor_118: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_195, mul_tensor_196);  mul_tensor_195 = mul_tensor_196 = None
        squeeze_dims_118: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_tensor_197: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_dims_118, 1.0012771392081736);  squeeze_dims_118 = None
        mul_tensor_198: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_197, 0.1);  mul_tensor_197 = None
        mul_tensor_199: "f32[256]" = torch.ops.aten.mul.Tensor(arg238_1, 0.9)
        add_tensor_119: "f32[256]" = torch.ops.aten.add.Tensor(mul_tensor_198, mul_tensor_199);  mul_tensor_198 = mul_tensor_199 = None
        add_tensor_120: "i64[]" = torch.ops.aten.add.Tensor(arg242_1, 1)
        squeeze_dims_119: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_dims_120: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_tensor_200: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_119, 0.1)
        mul_tensor_201: "f32[128]" = torch.ops.aten.mul.Tensor(arg243_1, 0.9)
        add_tensor_121: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_200, mul_tensor_201);  mul_tensor_200 = mul_tensor_201 = None
        squeeze_dims_121: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_tensor_202: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_121, 1.0012771392081736);  squeeze_dims_121 = None
        mul_tensor_203: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_202, 0.1);  mul_tensor_202 = None
        mul_tensor_204: "f32[128]" = torch.ops.aten.mul.Tensor(arg244_1, 0.9)
        add_tensor_122: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_203, mul_tensor_204);  mul_tensor_203 = mul_tensor_204 = None
        add_tensor_123: "i64[]" = torch.ops.aten.add.Tensor(arg248_1, 1)
        squeeze_dims_122: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_dims_123: "f32[288]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_tensor_205: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_dims_122, 0.1)
        mul_tensor_206: "f32[288]" = torch.ops.aten.mul.Tensor(arg249_1, 0.9)
        add_tensor_124: "f32[288]" = torch.ops.aten.add.Tensor(mul_tensor_205, mul_tensor_206);  mul_tensor_205 = mul_tensor_206 = None
        squeeze_dims_124: "f32[288]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_tensor_207: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_dims_124, 1.0012771392081736);  squeeze_dims_124 = None
        mul_tensor_208: "f32[288]" = torch.ops.aten.mul.Tensor(mul_tensor_207, 0.1);  mul_tensor_207 = None
        mul_tensor_209: "f32[288]" = torch.ops.aten.mul.Tensor(arg250_1, 0.9)
        add_tensor_125: "f32[288]" = torch.ops.aten.add.Tensor(mul_tensor_208, mul_tensor_209);  mul_tensor_208 = mul_tensor_209 = None
        add_tensor_126: "i64[]" = torch.ops.aten.add.Tensor(arg254_1, 1)
        squeeze_dims_125: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_dims_126: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_tensor_210: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_125, 0.1)
        mul_tensor_211: "f32[128]" = torch.ops.aten.mul.Tensor(arg255_1, 0.9)
        add_tensor_127: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_210, mul_tensor_211);  mul_tensor_210 = mul_tensor_211 = None
        squeeze_dims_127: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_tensor_212: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_127, 1.0012771392081736);  squeeze_dims_127 = None
        mul_tensor_213: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_212, 0.1);  mul_tensor_212 = None
        mul_tensor_214: "f32[128]" = torch.ops.aten.mul.Tensor(arg256_1, 0.9)
        add_tensor_128: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_213, mul_tensor_214);  mul_tensor_213 = mul_tensor_214 = None
        add_tensor_129: "i64[]" = torch.ops.aten.add.Tensor(arg260_1, 1)
        squeeze_dims_128: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_dims_129: "f32[320]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_tensor_215: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_128, 0.1)
        mul_tensor_216: "f32[320]" = torch.ops.aten.mul.Tensor(arg261_1, 0.9)
        add_tensor_130: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_215, mul_tensor_216);  mul_tensor_215 = mul_tensor_216 = None
        squeeze_dims_130: "f32[320]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_tensor_217: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_dims_130, 1.0012771392081736);  squeeze_dims_130 = None
        mul_tensor_218: "f32[320]" = torch.ops.aten.mul.Tensor(mul_tensor_217, 0.1);  mul_tensor_217 = None
        mul_tensor_219: "f32[320]" = torch.ops.aten.mul.Tensor(arg262_1, 0.9)
        add_tensor_131: "f32[320]" = torch.ops.aten.add.Tensor(mul_tensor_218, mul_tensor_219);  mul_tensor_218 = mul_tensor_219 = None
        add_tensor_132: "i64[]" = torch.ops.aten.add.Tensor(arg266_1, 1)
        squeeze_dims_131: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_dims_132: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_tensor_220: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_131, 0.1)
        mul_tensor_221: "f32[128]" = torch.ops.aten.mul.Tensor(arg267_1, 0.9)
        add_tensor_133: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_220, mul_tensor_221);  mul_tensor_220 = mul_tensor_221 = None
        squeeze_dims_133: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_tensor_222: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_133, 1.0012771392081736);  squeeze_dims_133 = None
        mul_tensor_223: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_222, 0.1);  mul_tensor_222 = None
        mul_tensor_224: "f32[128]" = torch.ops.aten.mul.Tensor(arg268_1, 0.9)
        add_tensor_134: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_223, mul_tensor_224);  mul_tensor_223 = mul_tensor_224 = None
        add_tensor_135: "i64[]" = torch.ops.aten.add.Tensor(arg272_1, 1)
        squeeze_dims_134: "f32[352]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_dims_135: "f32[352]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_tensor_225: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_dims_134, 0.1)
        mul_tensor_226: "f32[352]" = torch.ops.aten.mul.Tensor(arg273_1, 0.9)
        add_tensor_136: "f32[352]" = torch.ops.aten.add.Tensor(mul_tensor_225, mul_tensor_226);  mul_tensor_225 = mul_tensor_226 = None
        squeeze_dims_136: "f32[352]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_tensor_227: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_dims_136, 1.0012771392081736);  squeeze_dims_136 = None
        mul_tensor_228: "f32[352]" = torch.ops.aten.mul.Tensor(mul_tensor_227, 0.1);  mul_tensor_227 = None
        mul_tensor_229: "f32[352]" = torch.ops.aten.mul.Tensor(arg274_1, 0.9)
        add_tensor_137: "f32[352]" = torch.ops.aten.add.Tensor(mul_tensor_228, mul_tensor_229);  mul_tensor_228 = mul_tensor_229 = None
        add_tensor_138: "i64[]" = torch.ops.aten.add.Tensor(arg278_1, 1)
        squeeze_dims_137: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_dims_138: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_tensor_230: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_137, 0.1)
        mul_tensor_231: "f32[128]" = torch.ops.aten.mul.Tensor(arg279_1, 0.9)
        add_tensor_139: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_230, mul_tensor_231);  mul_tensor_230 = mul_tensor_231 = None
        squeeze_dims_139: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_tensor_232: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_139, 1.0012771392081736);  squeeze_dims_139 = None
        mul_tensor_233: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_232, 0.1);  mul_tensor_232 = None
        mul_tensor_234: "f32[128]" = torch.ops.aten.mul.Tensor(arg280_1, 0.9)
        add_tensor_140: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_233, mul_tensor_234);  mul_tensor_233 = mul_tensor_234 = None
        add_tensor_141: "i64[]" = torch.ops.aten.add.Tensor(arg284_1, 1)
        squeeze_dims_140: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_dims_141: "f32[384]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_tensor_235: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_140, 0.1)
        mul_tensor_236: "f32[384]" = torch.ops.aten.mul.Tensor(arg285_1, 0.9)
        add_tensor_142: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_235, mul_tensor_236);  mul_tensor_235 = mul_tensor_236 = None
        squeeze_dims_142: "f32[384]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_tensor_237: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_dims_142, 1.0012771392081736);  squeeze_dims_142 = None
        mul_tensor_238: "f32[384]" = torch.ops.aten.mul.Tensor(mul_tensor_237, 0.1);  mul_tensor_237 = None
        mul_tensor_239: "f32[384]" = torch.ops.aten.mul.Tensor(arg286_1, 0.9)
        add_tensor_143: "f32[384]" = torch.ops.aten.add.Tensor(mul_tensor_238, mul_tensor_239);  mul_tensor_238 = mul_tensor_239 = None
        add_tensor_144: "i64[]" = torch.ops.aten.add.Tensor(arg290_1, 1)
        squeeze_dims_143: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_dims_144: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_tensor_240: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_143, 0.1)
        mul_tensor_241: "f32[128]" = torch.ops.aten.mul.Tensor(arg291_1, 0.9)
        add_tensor_145: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_240, mul_tensor_241);  mul_tensor_240 = mul_tensor_241 = None
        squeeze_dims_145: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_tensor_242: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_145, 1.0012771392081736);  squeeze_dims_145 = None
        mul_tensor_243: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_242, 0.1);  mul_tensor_242 = None
        mul_tensor_244: "f32[128]" = torch.ops.aten.mul.Tensor(arg292_1, 0.9)
        add_tensor_146: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_243, mul_tensor_244);  mul_tensor_243 = mul_tensor_244 = None
        add_tensor_147: "i64[]" = torch.ops.aten.add.Tensor(arg296_1, 1)
        squeeze_dims_146: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_dims_147: "f32[416]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_tensor_245: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_dims_146, 0.1)
        mul_tensor_246: "f32[416]" = torch.ops.aten.mul.Tensor(arg297_1, 0.9)
        add_tensor_148: "f32[416]" = torch.ops.aten.add.Tensor(mul_tensor_245, mul_tensor_246);  mul_tensor_245 = mul_tensor_246 = None
        squeeze_dims_148: "f32[416]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_tensor_247: "f32[416]" = torch.ops.aten.mul.Tensor(squeeze_dims_148, 1.0012771392081736);  squeeze_dims_148 = None
        mul_tensor_248: "f32[416]" = torch.ops.aten.mul.Tensor(mul_tensor_247, 0.1);  mul_tensor_247 = None
        mul_tensor_249: "f32[416]" = torch.ops.aten.mul.Tensor(arg298_1, 0.9)
        add_tensor_149: "f32[416]" = torch.ops.aten.add.Tensor(mul_tensor_248, mul_tensor_249);  mul_tensor_248 = mul_tensor_249 = None
        add_tensor_150: "i64[]" = torch.ops.aten.add.Tensor(arg302_1, 1)
        squeeze_dims_149: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_dims_150: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_tensor_250: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_149, 0.1)
        mul_tensor_251: "f32[128]" = torch.ops.aten.mul.Tensor(arg303_1, 0.9)
        add_tensor_151: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_250, mul_tensor_251);  mul_tensor_250 = mul_tensor_251 = None
        squeeze_dims_151: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_tensor_252: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_151, 1.0012771392081736);  squeeze_dims_151 = None
        mul_tensor_253: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_252, 0.1);  mul_tensor_252 = None
        mul_tensor_254: "f32[128]" = torch.ops.aten.mul.Tensor(arg304_1, 0.9)
        add_tensor_152: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_253, mul_tensor_254);  mul_tensor_253 = mul_tensor_254 = None
        add_tensor_153: "i64[]" = torch.ops.aten.add.Tensor(arg308_1, 1)
        squeeze_dims_152: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_dims_153: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_tensor_255: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_dims_152, 0.1)
        mul_tensor_256: "f32[448]" = torch.ops.aten.mul.Tensor(arg309_1, 0.9)
        add_tensor_154: "f32[448]" = torch.ops.aten.add.Tensor(mul_tensor_255, mul_tensor_256);  mul_tensor_255 = mul_tensor_256 = None
        squeeze_dims_154: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_tensor_257: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_dims_154, 1.0012771392081736);  squeeze_dims_154 = None
        mul_tensor_258: "f32[448]" = torch.ops.aten.mul.Tensor(mul_tensor_257, 0.1);  mul_tensor_257 = None
        mul_tensor_259: "f32[448]" = torch.ops.aten.mul.Tensor(arg310_1, 0.9)
        add_tensor_155: "f32[448]" = torch.ops.aten.add.Tensor(mul_tensor_258, mul_tensor_259);  mul_tensor_258 = mul_tensor_259 = None
        add_tensor_156: "i64[]" = torch.ops.aten.add.Tensor(arg314_1, 1)
        squeeze_dims_155: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_dims_156: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_tensor_260: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_155, 0.1)
        mul_tensor_261: "f32[128]" = torch.ops.aten.mul.Tensor(arg315_1, 0.9)
        add_tensor_157: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_260, mul_tensor_261);  mul_tensor_260 = mul_tensor_261 = None
        squeeze_dims_157: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_tensor_262: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_157, 1.0012771392081736);  squeeze_dims_157 = None
        mul_tensor_263: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_262, 0.1);  mul_tensor_262 = None
        mul_tensor_264: "f32[128]" = torch.ops.aten.mul.Tensor(arg316_1, 0.9)
        add_tensor_158: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_263, mul_tensor_264);  mul_tensor_263 = mul_tensor_264 = None
        add_tensor_159: "i64[]" = torch.ops.aten.add.Tensor(arg320_1, 1)
        squeeze_dims_158: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_dims_159: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_tensor_265: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_158, 0.1)
        mul_tensor_266: "f32[480]" = torch.ops.aten.mul.Tensor(arg321_1, 0.9)
        add_tensor_160: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_265, mul_tensor_266);  mul_tensor_265 = mul_tensor_266 = None
        squeeze_dims_160: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_tensor_267: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_160, 1.0012771392081736);  squeeze_dims_160 = None
        mul_tensor_268: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_267, 0.1);  mul_tensor_267 = None
        mul_tensor_269: "f32[480]" = torch.ops.aten.mul.Tensor(arg322_1, 0.9)
        add_tensor_161: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_268, mul_tensor_269);  mul_tensor_268 = mul_tensor_269 = None
        add_tensor_162: "i64[]" = torch.ops.aten.add.Tensor(arg326_1, 1)
        squeeze_dims_161: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_dims_162: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_tensor_270: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_161, 0.1)
        mul_tensor_271: "f32[128]" = torch.ops.aten.mul.Tensor(arg327_1, 0.9)
        add_tensor_163: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_270, mul_tensor_271);  mul_tensor_270 = mul_tensor_271 = None
        squeeze_dims_163: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_tensor_272: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_163, 1.0012771392081736);  squeeze_dims_163 = None
        mul_tensor_273: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_272, 0.1);  mul_tensor_272 = None
        mul_tensor_274: "f32[128]" = torch.ops.aten.mul.Tensor(arg328_1, 0.9)
        add_tensor_164: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_273, mul_tensor_274);  mul_tensor_273 = mul_tensor_274 = None
        add_tensor_165: "i64[]" = torch.ops.aten.add.Tensor(arg332_1, 1)
        squeeze_dims_164: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_dims_165: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_tensor_275: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_164, 0.1)
        mul_tensor_276: "f32[512]" = torch.ops.aten.mul.Tensor(arg333_1, 0.9)
        add_tensor_166: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_275, mul_tensor_276);  mul_tensor_275 = mul_tensor_276 = None
        squeeze_dims_166: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_tensor_277: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_166, 1.0012771392081736);  squeeze_dims_166 = None
        mul_tensor_278: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_277, 0.1);  mul_tensor_277 = None
        mul_tensor_279: "f32[512]" = torch.ops.aten.mul.Tensor(arg334_1, 0.9)
        add_tensor_167: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_278, mul_tensor_279);  mul_tensor_278 = mul_tensor_279 = None
        add_tensor_168: "i64[]" = torch.ops.aten.add.Tensor(arg338_1, 1)
        squeeze_dims_167: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        squeeze_dims_168: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_tensor_280: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_167, 0.1)
        mul_tensor_281: "f32[128]" = torch.ops.aten.mul.Tensor(arg339_1, 0.9)
        add_tensor_169: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_280, mul_tensor_281);  mul_tensor_280 = mul_tensor_281 = None
        squeeze_dims_169: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_tensor_282: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_169, 1.0012771392081736);  squeeze_dims_169 = None
        mul_tensor_283: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_282, 0.1);  mul_tensor_282 = None
        mul_tensor_284: "f32[128]" = torch.ops.aten.mul.Tensor(arg340_1, 0.9)
        add_tensor_170: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_283, mul_tensor_284);  mul_tensor_283 = mul_tensor_284 = None
        add_tensor_171: "i64[]" = torch.ops.aten.add.Tensor(arg344_1, 1)
        squeeze_dims_170: "f32[544]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_dims_171: "f32[544]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_tensor_285: "f32[544]" = torch.ops.aten.mul.Tensor(squeeze_dims_170, 0.1)
        mul_tensor_286: "f32[544]" = torch.ops.aten.mul.Tensor(arg345_1, 0.9)
        add_tensor_172: "f32[544]" = torch.ops.aten.add.Tensor(mul_tensor_285, mul_tensor_286);  mul_tensor_285 = mul_tensor_286 = None
        squeeze_dims_172: "f32[544]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_tensor_287: "f32[544]" = torch.ops.aten.mul.Tensor(squeeze_dims_172, 1.0012771392081736);  squeeze_dims_172 = None
        mul_tensor_288: "f32[544]" = torch.ops.aten.mul.Tensor(mul_tensor_287, 0.1);  mul_tensor_287 = None
        mul_tensor_289: "f32[544]" = torch.ops.aten.mul.Tensor(arg346_1, 0.9)
        add_tensor_173: "f32[544]" = torch.ops.aten.add.Tensor(mul_tensor_288, mul_tensor_289);  mul_tensor_288 = mul_tensor_289 = None
        add_tensor_174: "i64[]" = torch.ops.aten.add.Tensor(arg350_1, 1)
        squeeze_dims_173: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        squeeze_dims_174: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_tensor_290: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_173, 0.1)
        mul_tensor_291: "f32[128]" = torch.ops.aten.mul.Tensor(arg351_1, 0.9)
        add_tensor_175: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_290, mul_tensor_291);  mul_tensor_290 = mul_tensor_291 = None
        squeeze_dims_175: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_tensor_292: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_175, 1.0012771392081736);  squeeze_dims_175 = None
        mul_tensor_293: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_292, 0.1);  mul_tensor_292 = None
        mul_tensor_294: "f32[128]" = torch.ops.aten.mul.Tensor(arg352_1, 0.9)
        add_tensor_176: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_293, mul_tensor_294);  mul_tensor_293 = mul_tensor_294 = None
        add_tensor_177: "i64[]" = torch.ops.aten.add.Tensor(arg356_1, 1)
        squeeze_dims_176: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        squeeze_dims_177: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_tensor_295: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_176, 0.1)
        mul_tensor_296: "f32[576]" = torch.ops.aten.mul.Tensor(arg357_1, 0.9)
        add_tensor_178: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_295, mul_tensor_296);  mul_tensor_295 = mul_tensor_296 = None
        squeeze_dims_178: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_tensor_297: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_178, 1.0012771392081736);  squeeze_dims_178 = None
        mul_tensor_298: "f32[576]" = torch.ops.aten.mul.Tensor(mul_tensor_297, 0.1);  mul_tensor_297 = None
        mul_tensor_299: "f32[576]" = torch.ops.aten.mul.Tensor(arg358_1, 0.9)
        add_tensor_179: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_298, mul_tensor_299);  mul_tensor_298 = mul_tensor_299 = None
        add_tensor_180: "i64[]" = torch.ops.aten.add.Tensor(arg362_1, 1)
        squeeze_dims_179: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        squeeze_dims_180: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_tensor_300: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_179, 0.1)
        mul_tensor_301: "f32[128]" = torch.ops.aten.mul.Tensor(arg363_1, 0.9)
        add_tensor_181: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_300, mul_tensor_301);  mul_tensor_300 = mul_tensor_301 = None
        squeeze_dims_181: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_tensor_302: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_181, 1.0012771392081736);  squeeze_dims_181 = None
        mul_tensor_303: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_302, 0.1);  mul_tensor_302 = None
        mul_tensor_304: "f32[128]" = torch.ops.aten.mul.Tensor(arg364_1, 0.9)
        add_tensor_182: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_303, mul_tensor_304);  mul_tensor_303 = mul_tensor_304 = None
        add_tensor_183: "i64[]" = torch.ops.aten.add.Tensor(arg368_1, 1)
        squeeze_dims_182: "f32[608]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_dims_183: "f32[608]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_tensor_305: "f32[608]" = torch.ops.aten.mul.Tensor(squeeze_dims_182, 0.1)
        mul_tensor_306: "f32[608]" = torch.ops.aten.mul.Tensor(arg369_1, 0.9)
        add_tensor_184: "f32[608]" = torch.ops.aten.add.Tensor(mul_tensor_305, mul_tensor_306);  mul_tensor_305 = mul_tensor_306 = None
        squeeze_dims_184: "f32[608]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_tensor_307: "f32[608]" = torch.ops.aten.mul.Tensor(squeeze_dims_184, 1.0012771392081736);  squeeze_dims_184 = None
        mul_tensor_308: "f32[608]" = torch.ops.aten.mul.Tensor(mul_tensor_307, 0.1);  mul_tensor_307 = None
        mul_tensor_309: "f32[608]" = torch.ops.aten.mul.Tensor(arg370_1, 0.9)
        add_tensor_185: "f32[608]" = torch.ops.aten.add.Tensor(mul_tensor_308, mul_tensor_309);  mul_tensor_308 = mul_tensor_309 = None
        add_tensor_186: "i64[]" = torch.ops.aten.add.Tensor(arg374_1, 1)
        squeeze_dims_185: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        squeeze_dims_186: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
        mul_tensor_310: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_185, 0.1)
        mul_tensor_311: "f32[128]" = torch.ops.aten.mul.Tensor(arg375_1, 0.9)
        add_tensor_187: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_310, mul_tensor_311);  mul_tensor_310 = mul_tensor_311 = None
        squeeze_dims_187: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_tensor_312: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_187, 1.0012771392081736);  squeeze_dims_187 = None
        mul_tensor_313: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_312, 0.1);  mul_tensor_312 = None
        mul_tensor_314: "f32[128]" = torch.ops.aten.mul.Tensor(arg376_1, 0.9)
        add_tensor_188: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_313, mul_tensor_314);  mul_tensor_313 = mul_tensor_314 = None
        add_tensor_189: "i64[]" = torch.ops.aten.add.Tensor(arg380_1, 1)
        squeeze_dims_188: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        squeeze_dims_189: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
        mul_tensor_315: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims_188, 0.1)
        mul_tensor_316: "f32[640]" = torch.ops.aten.mul.Tensor(arg381_1, 0.9)
        add_tensor_190: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_315, mul_tensor_316);  mul_tensor_315 = mul_tensor_316 = None
        squeeze_dims_190: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
        mul_tensor_317: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims_190, 1.0012771392081736);  squeeze_dims_190 = None
        mul_tensor_318: "f32[640]" = torch.ops.aten.mul.Tensor(mul_tensor_317, 0.1);  mul_tensor_317 = None
        mul_tensor_319: "f32[640]" = torch.ops.aten.mul.Tensor(arg382_1, 0.9)
        add_tensor_191: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_318, mul_tensor_319);  mul_tensor_318 = mul_tensor_319 = None
        add_tensor_192: "i64[]" = torch.ops.aten.add.Tensor(arg386_1, 1)
        squeeze_dims_191: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
        squeeze_dims_192: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
        mul_tensor_320: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_191, 0.1)
        mul_tensor_321: "f32[128]" = torch.ops.aten.mul.Tensor(arg387_1, 0.9)
        add_tensor_193: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_320, mul_tensor_321);  mul_tensor_320 = mul_tensor_321 = None
        squeeze_dims_193: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
        mul_tensor_322: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_193, 1.0012771392081736);  squeeze_dims_193 = None
        mul_tensor_323: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_322, 0.1);  mul_tensor_322 = None
        mul_tensor_324: "f32[128]" = torch.ops.aten.mul.Tensor(arg388_1, 0.9)
        add_tensor_194: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_323, mul_tensor_324);  mul_tensor_323 = mul_tensor_324 = None
        add_tensor_195: "i64[]" = torch.ops.aten.add.Tensor(arg392_1, 1)
        squeeze_dims_194: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        squeeze_dims_195: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
        mul_tensor_325: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_194, 0.1)
        mul_tensor_326: "f32[672]" = torch.ops.aten.mul.Tensor(arg393_1, 0.9)
        add_tensor_196: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_325, mul_tensor_326);  mul_tensor_325 = mul_tensor_326 = None
        squeeze_dims_196: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
        mul_tensor_327: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_196, 1.0012771392081736);  squeeze_dims_196 = None
        mul_tensor_328: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_327, 0.1);  mul_tensor_327 = None
        mul_tensor_329: "f32[672]" = torch.ops.aten.mul.Tensor(arg394_1, 0.9)
        add_tensor_197: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_328, mul_tensor_329);  mul_tensor_328 = mul_tensor_329 = None
        add_tensor_198: "i64[]" = torch.ops.aten.add.Tensor(arg398_1, 1)
        squeeze_dims_197: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
        squeeze_dims_198: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
        mul_tensor_330: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_197, 0.1)
        mul_tensor_331: "f32[128]" = torch.ops.aten.mul.Tensor(arg399_1, 0.9)
        add_tensor_199: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_330, mul_tensor_331);  mul_tensor_330 = mul_tensor_331 = None
        squeeze_dims_199: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
        mul_tensor_332: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_199, 1.0012771392081736);  squeeze_dims_199 = None
        mul_tensor_333: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_332, 0.1);  mul_tensor_332 = None
        mul_tensor_334: "f32[128]" = torch.ops.aten.mul.Tensor(arg400_1, 0.9)
        add_tensor_200: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_333, mul_tensor_334);  mul_tensor_333 = mul_tensor_334 = None
        add_tensor_201: "i64[]" = torch.ops.aten.add.Tensor(arg404_1, 1)
        squeeze_dims_200: "f32[704]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        squeeze_dims_201: "f32[704]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
        mul_tensor_335: "f32[704]" = torch.ops.aten.mul.Tensor(squeeze_dims_200, 0.1)
        mul_tensor_336: "f32[704]" = torch.ops.aten.mul.Tensor(arg405_1, 0.9)
        add_tensor_202: "f32[704]" = torch.ops.aten.add.Tensor(mul_tensor_335, mul_tensor_336);  mul_tensor_335 = mul_tensor_336 = None
        squeeze_dims_202: "f32[704]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
        mul_tensor_337: "f32[704]" = torch.ops.aten.mul.Tensor(squeeze_dims_202, 1.0012771392081736);  squeeze_dims_202 = None
        mul_tensor_338: "f32[704]" = torch.ops.aten.mul.Tensor(mul_tensor_337, 0.1);  mul_tensor_337 = None
        mul_tensor_339: "f32[704]" = torch.ops.aten.mul.Tensor(arg406_1, 0.9)
        add_tensor_203: "f32[704]" = torch.ops.aten.add.Tensor(mul_tensor_338, mul_tensor_339);  mul_tensor_338 = mul_tensor_339 = None
        add_tensor_204: "i64[]" = torch.ops.aten.add.Tensor(arg410_1, 1)
        squeeze_dims_203: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        squeeze_dims_204: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
        mul_tensor_340: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_203, 0.1)
        mul_tensor_341: "f32[128]" = torch.ops.aten.mul.Tensor(arg411_1, 0.9)
        add_tensor_205: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_340, mul_tensor_341);  mul_tensor_340 = mul_tensor_341 = None
        squeeze_dims_205: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
        mul_tensor_342: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_205, 1.0012771392081736);  squeeze_dims_205 = None
        mul_tensor_343: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_342, 0.1);  mul_tensor_342 = None
        mul_tensor_344: "f32[128]" = torch.ops.aten.mul.Tensor(arg412_1, 0.9)
        add_tensor_206: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_343, mul_tensor_344);  mul_tensor_343 = mul_tensor_344 = None
        add_tensor_207: "i64[]" = torch.ops.aten.add.Tensor(arg416_1, 1)
        squeeze_dims_206: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
        squeeze_dims_207: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
        mul_tensor_345: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_dims_206, 0.1)
        mul_tensor_346: "f32[736]" = torch.ops.aten.mul.Tensor(arg417_1, 0.9)
        add_tensor_208: "f32[736]" = torch.ops.aten.add.Tensor(mul_tensor_345, mul_tensor_346);  mul_tensor_345 = mul_tensor_346 = None
        squeeze_dims_208: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
        mul_tensor_347: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_dims_208, 1.0012771392081736);  squeeze_dims_208 = None
        mul_tensor_348: "f32[736]" = torch.ops.aten.mul.Tensor(mul_tensor_347, 0.1);  mul_tensor_347 = None
        mul_tensor_349: "f32[736]" = torch.ops.aten.mul.Tensor(arg418_1, 0.9)
        add_tensor_209: "f32[736]" = torch.ops.aten.add.Tensor(mul_tensor_348, mul_tensor_349);  mul_tensor_348 = mul_tensor_349 = None
        add_tensor_210: "i64[]" = torch.ops.aten.add.Tensor(arg422_1, 1)
        squeeze_dims_209: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
        squeeze_dims_210: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
        mul_tensor_350: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_209, 0.1)
        mul_tensor_351: "f32[128]" = torch.ops.aten.mul.Tensor(arg423_1, 0.9)
        add_tensor_211: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_350, mul_tensor_351);  mul_tensor_350 = mul_tensor_351 = None
        squeeze_dims_211: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
        mul_tensor_352: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_211, 1.0012771392081736);  squeeze_dims_211 = None
        mul_tensor_353: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_352, 0.1);  mul_tensor_352 = None
        mul_tensor_354: "f32[128]" = torch.ops.aten.mul.Tensor(arg424_1, 0.9)
        add_tensor_212: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_353, mul_tensor_354);  mul_tensor_353 = mul_tensor_354 = None
        add_tensor_213: "i64[]" = torch.ops.aten.add.Tensor(arg428_1, 1)
        squeeze_dims_212: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        squeeze_dims_213: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
        mul_tensor_355: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_212, 0.1)
        mul_tensor_356: "f32[768]" = torch.ops.aten.mul.Tensor(arg429_1, 0.9)
        add_tensor_214: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_355, mul_tensor_356);  mul_tensor_355 = mul_tensor_356 = None
        squeeze_dims_214: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
        mul_tensor_357: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_214, 1.0012771392081736);  squeeze_dims_214 = None
        mul_tensor_358: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_357, 0.1);  mul_tensor_357 = None
        mul_tensor_359: "f32[768]" = torch.ops.aten.mul.Tensor(arg430_1, 0.9)
        add_tensor_215: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_358, mul_tensor_359);  mul_tensor_358 = mul_tensor_359 = None
        add_tensor_216: "i64[]" = torch.ops.aten.add.Tensor(arg434_1, 1)
        squeeze_dims_215: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
        squeeze_dims_216: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
        mul_tensor_360: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_215, 0.1)
        mul_tensor_361: "f32[128]" = torch.ops.aten.mul.Tensor(arg435_1, 0.9)
        add_tensor_217: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_360, mul_tensor_361);  mul_tensor_360 = mul_tensor_361 = None
        squeeze_dims_217: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
        mul_tensor_362: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_217, 1.0012771392081736);  squeeze_dims_217 = None
        mul_tensor_363: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_362, 0.1);  mul_tensor_362 = None
        mul_tensor_364: "f32[128]" = torch.ops.aten.mul.Tensor(arg436_1, 0.9)
        add_tensor_218: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_363, mul_tensor_364);  mul_tensor_363 = mul_tensor_364 = None
        add_tensor_219: "i64[]" = torch.ops.aten.add.Tensor(arg440_1, 1)
        squeeze_dims_218: "f32[800]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
        squeeze_dims_219: "f32[800]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
        mul_tensor_365: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_dims_218, 0.1)
        mul_tensor_366: "f32[800]" = torch.ops.aten.mul.Tensor(arg441_1, 0.9)
        add_tensor_220: "f32[800]" = torch.ops.aten.add.Tensor(mul_tensor_365, mul_tensor_366);  mul_tensor_365 = mul_tensor_366 = None
        squeeze_dims_220: "f32[800]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
        mul_tensor_367: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_dims_220, 1.0012771392081736);  squeeze_dims_220 = None
        mul_tensor_368: "f32[800]" = torch.ops.aten.mul.Tensor(mul_tensor_367, 0.1);  mul_tensor_367 = None
        mul_tensor_369: "f32[800]" = torch.ops.aten.mul.Tensor(arg442_1, 0.9)
        add_tensor_221: "f32[800]" = torch.ops.aten.add.Tensor(mul_tensor_368, mul_tensor_369);  mul_tensor_368 = mul_tensor_369 = None
        add_tensor_222: "i64[]" = torch.ops.aten.add.Tensor(arg446_1, 1)
        squeeze_dims_221: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
        squeeze_dims_222: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
        mul_tensor_370: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_221, 0.1)
        mul_tensor_371: "f32[128]" = torch.ops.aten.mul.Tensor(arg447_1, 0.9)
        add_tensor_223: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_370, mul_tensor_371);  mul_tensor_370 = mul_tensor_371 = None
        squeeze_dims_223: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
        mul_tensor_372: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_223, 1.0012771392081736);  squeeze_dims_223 = None
        mul_tensor_373: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_372, 0.1);  mul_tensor_372 = None
        mul_tensor_374: "f32[128]" = torch.ops.aten.mul.Tensor(arg448_1, 0.9)
        add_tensor_224: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_373, mul_tensor_374);  mul_tensor_373 = mul_tensor_374 = None
        add_tensor_225: "i64[]" = torch.ops.aten.add.Tensor(arg452_1, 1)
        squeeze_dims_224: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        squeeze_dims_225: "f32[832]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
        mul_tensor_375: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_dims_224, 0.1)
        mul_tensor_376: "f32[832]" = torch.ops.aten.mul.Tensor(arg453_1, 0.9)
        add_tensor_226: "f32[832]" = torch.ops.aten.add.Tensor(mul_tensor_375, mul_tensor_376);  mul_tensor_375 = mul_tensor_376 = None
        squeeze_dims_226: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
        mul_tensor_377: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_dims_226, 1.0012771392081736);  squeeze_dims_226 = None
        mul_tensor_378: "f32[832]" = torch.ops.aten.mul.Tensor(mul_tensor_377, 0.1);  mul_tensor_377 = None
        mul_tensor_379: "f32[832]" = torch.ops.aten.mul.Tensor(arg454_1, 0.9)
        add_tensor_227: "f32[832]" = torch.ops.aten.add.Tensor(mul_tensor_378, mul_tensor_379);  mul_tensor_378 = mul_tensor_379 = None
        add_tensor_228: "i64[]" = torch.ops.aten.add.Tensor(arg458_1, 1)
        squeeze_dims_227: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
        squeeze_dims_228: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
        mul_tensor_380: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_227, 0.1)
        mul_tensor_381: "f32[128]" = torch.ops.aten.mul.Tensor(arg459_1, 0.9)
        add_tensor_229: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_380, mul_tensor_381);  mul_tensor_380 = mul_tensor_381 = None
        squeeze_dims_229: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
        mul_tensor_382: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_229, 1.0012771392081736);  squeeze_dims_229 = None
        mul_tensor_383: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_382, 0.1);  mul_tensor_382 = None
        mul_tensor_384: "f32[128]" = torch.ops.aten.mul.Tensor(arg460_1, 0.9)
        add_tensor_230: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_383, mul_tensor_384);  mul_tensor_383 = mul_tensor_384 = None
        add_tensor_231: "i64[]" = torch.ops.aten.add.Tensor(arg464_1, 1)
        squeeze_dims_230: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        squeeze_dims_231: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
        mul_tensor_385: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_dims_230, 0.1)
        mul_tensor_386: "f32[864]" = torch.ops.aten.mul.Tensor(arg465_1, 0.9)
        add_tensor_232: "f32[864]" = torch.ops.aten.add.Tensor(mul_tensor_385, mul_tensor_386);  mul_tensor_385 = mul_tensor_386 = None
        squeeze_dims_232: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
        mul_tensor_387: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_dims_232, 1.0012771392081736);  squeeze_dims_232 = None
        mul_tensor_388: "f32[864]" = torch.ops.aten.mul.Tensor(mul_tensor_387, 0.1);  mul_tensor_387 = None
        mul_tensor_389: "f32[864]" = torch.ops.aten.mul.Tensor(arg466_1, 0.9)
        add_tensor_233: "f32[864]" = torch.ops.aten.add.Tensor(mul_tensor_388, mul_tensor_389);  mul_tensor_388 = mul_tensor_389 = None
        add_tensor_234: "i64[]" = torch.ops.aten.add.Tensor(arg470_1, 1)
        squeeze_dims_233: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
        squeeze_dims_234: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
        mul_tensor_390: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_233, 0.1)
        mul_tensor_391: "f32[128]" = torch.ops.aten.mul.Tensor(arg471_1, 0.9)
        add_tensor_235: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_390, mul_tensor_391);  mul_tensor_390 = mul_tensor_391 = None
        squeeze_dims_235: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
        mul_tensor_392: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_235, 1.0012771392081736);  squeeze_dims_235 = None
        mul_tensor_393: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_392, 0.1);  mul_tensor_392 = None
        mul_tensor_394: "f32[128]" = torch.ops.aten.mul.Tensor(arg472_1, 0.9)
        add_tensor_236: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_393, mul_tensor_394);  mul_tensor_393 = mul_tensor_394 = None
        add_tensor_237: "i64[]" = torch.ops.aten.add.Tensor(arg476_1, 1)
        squeeze_dims_236: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
        squeeze_dims_237: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
        mul_tensor_395: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_dims_236, 0.1)
        mul_tensor_396: "f32[896]" = torch.ops.aten.mul.Tensor(arg477_1, 0.9)
        add_tensor_238: "f32[896]" = torch.ops.aten.add.Tensor(mul_tensor_395, mul_tensor_396);  mul_tensor_395 = mul_tensor_396 = None
        squeeze_dims_238: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
        mul_tensor_397: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_dims_238, 1.0012771392081736);  squeeze_dims_238 = None
        mul_tensor_398: "f32[896]" = torch.ops.aten.mul.Tensor(mul_tensor_397, 0.1);  mul_tensor_397 = None
        mul_tensor_399: "f32[896]" = torch.ops.aten.mul.Tensor(arg478_1, 0.9)
        add_tensor_239: "f32[896]" = torch.ops.aten.add.Tensor(mul_tensor_398, mul_tensor_399);  mul_tensor_398 = mul_tensor_399 = None
        add_tensor_240: "i64[]" = torch.ops.aten.add.Tensor(arg482_1, 1)
        squeeze_dims_239: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
        squeeze_dims_240: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
        mul_tensor_400: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_239, 0.1)
        mul_tensor_401: "f32[128]" = torch.ops.aten.mul.Tensor(arg483_1, 0.9)
        add_tensor_241: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_400, mul_tensor_401);  mul_tensor_400 = mul_tensor_401 = None
        squeeze_dims_241: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
        mul_tensor_402: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_241, 1.0012771392081736);  squeeze_dims_241 = None
        mul_tensor_403: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_402, 0.1);  mul_tensor_402 = None
        mul_tensor_404: "f32[128]" = torch.ops.aten.mul.Tensor(arg484_1, 0.9)
        add_tensor_242: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_403, mul_tensor_404);  mul_tensor_403 = mul_tensor_404 = None
        add_tensor_243: "i64[]" = torch.ops.aten.add.Tensor(arg488_1, 1)
        squeeze_dims_242: "f32[928]" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
        squeeze_dims_243: "f32[928]" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
        mul_tensor_405: "f32[928]" = torch.ops.aten.mul.Tensor(squeeze_dims_242, 0.1)
        mul_tensor_406: "f32[928]" = torch.ops.aten.mul.Tensor(arg489_1, 0.9)
        add_tensor_244: "f32[928]" = torch.ops.aten.add.Tensor(mul_tensor_405, mul_tensor_406);  mul_tensor_405 = mul_tensor_406 = None
        squeeze_dims_244: "f32[928]" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
        mul_tensor_407: "f32[928]" = torch.ops.aten.mul.Tensor(squeeze_dims_244, 1.0012771392081736);  squeeze_dims_244 = None
        mul_tensor_408: "f32[928]" = torch.ops.aten.mul.Tensor(mul_tensor_407, 0.1);  mul_tensor_407 = None
        mul_tensor_409: "f32[928]" = torch.ops.aten.mul.Tensor(arg490_1, 0.9)
        add_tensor_245: "f32[928]" = torch.ops.aten.add.Tensor(mul_tensor_408, mul_tensor_409);  mul_tensor_408 = mul_tensor_409 = None
        add_tensor_246: "i64[]" = torch.ops.aten.add.Tensor(arg494_1, 1)
        squeeze_dims_245: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
        squeeze_dims_246: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
        mul_tensor_410: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_245, 0.1)
        mul_tensor_411: "f32[128]" = torch.ops.aten.mul.Tensor(arg495_1, 0.9)
        add_tensor_247: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_410, mul_tensor_411);  mul_tensor_410 = mul_tensor_411 = None
        squeeze_dims_247: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
        mul_tensor_412: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_247, 1.0012771392081736);  squeeze_dims_247 = None
        mul_tensor_413: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_412, 0.1);  mul_tensor_412 = None
        mul_tensor_414: "f32[128]" = torch.ops.aten.mul.Tensor(arg496_1, 0.9)
        add_tensor_248: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_413, mul_tensor_414);  mul_tensor_413 = mul_tensor_414 = None
        add_tensor_249: "i64[]" = torch.ops.aten.add.Tensor(arg500_1, 1)
        squeeze_dims_248: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
        squeeze_dims_249: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
        mul_tensor_415: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_248, 0.1)
        mul_tensor_416: "f32[960]" = torch.ops.aten.mul.Tensor(arg501_1, 0.9)
        add_tensor_250: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_415, mul_tensor_416);  mul_tensor_415 = mul_tensor_416 = None
        squeeze_dims_250: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
        mul_tensor_417: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_250, 1.0012771392081736);  squeeze_dims_250 = None
        mul_tensor_418: "f32[960]" = torch.ops.aten.mul.Tensor(mul_tensor_417, 0.1);  mul_tensor_417 = None
        mul_tensor_419: "f32[960]" = torch.ops.aten.mul.Tensor(arg502_1, 0.9)
        add_tensor_251: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_418, mul_tensor_419);  mul_tensor_418 = mul_tensor_419 = None
        add_tensor_252: "i64[]" = torch.ops.aten.add.Tensor(arg506_1, 1)
        squeeze_dims_251: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
        squeeze_dims_252: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
        mul_tensor_420: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_251, 0.1)
        mul_tensor_421: "f32[128]" = torch.ops.aten.mul.Tensor(arg507_1, 0.9)
        add_tensor_253: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_420, mul_tensor_421);  mul_tensor_420 = mul_tensor_421 = None
        squeeze_dims_253: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
        mul_tensor_422: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_253, 1.0012771392081736);  squeeze_dims_253 = None
        mul_tensor_423: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_422, 0.1);  mul_tensor_422 = None
        mul_tensor_424: "f32[128]" = torch.ops.aten.mul.Tensor(arg508_1, 0.9)
        add_tensor_254: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_423, mul_tensor_424);  mul_tensor_423 = mul_tensor_424 = None
        add_tensor_255: "i64[]" = torch.ops.aten.add.Tensor(arg512_1, 1)
        squeeze_dims_254: "f32[992]" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
        squeeze_dims_255: "f32[992]" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
        mul_tensor_425: "f32[992]" = torch.ops.aten.mul.Tensor(squeeze_dims_254, 0.1)
        mul_tensor_426: "f32[992]" = torch.ops.aten.mul.Tensor(arg513_1, 0.9)
        add_tensor_256: "f32[992]" = torch.ops.aten.add.Tensor(mul_tensor_425, mul_tensor_426);  mul_tensor_425 = mul_tensor_426 = None
        squeeze_dims_256: "f32[992]" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
        mul_tensor_427: "f32[992]" = torch.ops.aten.mul.Tensor(squeeze_dims_256, 1.0012771392081736);  squeeze_dims_256 = None
        mul_tensor_428: "f32[992]" = torch.ops.aten.mul.Tensor(mul_tensor_427, 0.1);  mul_tensor_427 = None
        mul_tensor_429: "f32[992]" = torch.ops.aten.mul.Tensor(arg514_1, 0.9)
        add_tensor_257: "f32[992]" = torch.ops.aten.add.Tensor(mul_tensor_428, mul_tensor_429);  mul_tensor_428 = mul_tensor_429 = None
        add_tensor_258: "i64[]" = torch.ops.aten.add.Tensor(arg518_1, 1)
        squeeze_dims_257: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
        squeeze_dims_258: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
        mul_tensor_430: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_257, 0.1)
        mul_tensor_431: "f32[128]" = torch.ops.aten.mul.Tensor(arg519_1, 0.9)
        add_tensor_259: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_430, mul_tensor_431);  mul_tensor_430 = mul_tensor_431 = None
        squeeze_dims_259: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
        mul_tensor_432: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_259, 1.0012771392081736);  squeeze_dims_259 = None
        mul_tensor_433: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_432, 0.1);  mul_tensor_432 = None
        mul_tensor_434: "f32[128]" = torch.ops.aten.mul.Tensor(arg520_1, 0.9)
        add_tensor_260: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_433, mul_tensor_434);  mul_tensor_433 = mul_tensor_434 = None
        add_tensor_261: "i64[]" = torch.ops.aten.add.Tensor(arg524_1, 1)
        squeeze_dims_260: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
        squeeze_dims_261: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
        mul_tensor_435: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_260, 0.1)
        mul_tensor_436: "f32[1024]" = torch.ops.aten.mul.Tensor(arg525_1, 0.9)
        add_tensor_262: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_435, mul_tensor_436);  mul_tensor_435 = mul_tensor_436 = None
        squeeze_dims_262: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
        mul_tensor_437: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_262, 1.0012771392081736);  squeeze_dims_262 = None
        mul_tensor_438: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_tensor_437, 0.1);  mul_tensor_437 = None
        mul_tensor_439: "f32[1024]" = torch.ops.aten.mul.Tensor(arg526_1, 0.9)
        add_tensor_263: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_438, mul_tensor_439);  mul_tensor_438 = mul_tensor_439 = None
        add_tensor_264: "i64[]" = torch.ops.aten.add.Tensor(arg530_1, 1)
        squeeze_dims_263: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
        squeeze_dims_264: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
        mul_tensor_440: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_263, 0.1)
        mul_tensor_441: "f32[512]" = torch.ops.aten.mul.Tensor(arg531_1, 0.9)
        add_tensor_265: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_440, mul_tensor_441);  mul_tensor_440 = mul_tensor_441 = None
        squeeze_dims_265: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
        mul_tensor_442: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_dims_265, 1.005128205128205);  squeeze_dims_265 = None
        mul_tensor_443: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_442, 0.1);  mul_tensor_442 = None
        mul_tensor_444: "f32[512]" = torch.ops.aten.mul.Tensor(arg532_1, 0.9)
        add_tensor_266: "f32[512]" = torch.ops.aten.add.Tensor(mul_tensor_443, mul_tensor_444);  mul_tensor_443 = mul_tensor_444 = None
        add_tensor_267: "i64[]" = torch.ops.aten.add.Tensor(arg536_1, 1)
        squeeze_dims_266: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
        squeeze_dims_267: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
        mul_tensor_445: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_266, 0.1)
        mul_tensor_446: "f32[128]" = torch.ops.aten.mul.Tensor(arg537_1, 0.9)
        add_tensor_268: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_445, mul_tensor_446);  mul_tensor_445 = mul_tensor_446 = None
        squeeze_dims_268: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
        mul_tensor_447: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_268, 1.005128205128205);  squeeze_dims_268 = None
        mul_tensor_448: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_447, 0.1);  mul_tensor_447 = None
        mul_tensor_449: "f32[128]" = torch.ops.aten.mul.Tensor(arg538_1, 0.9)
        add_tensor_269: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_448, mul_tensor_449);  mul_tensor_448 = mul_tensor_449 = None
        add_tensor_270: "i64[]" = torch.ops.aten.add.Tensor(arg542_1, 1)
        squeeze_dims_269: "f32[544]" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
        squeeze_dims_270: "f32[544]" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
        mul_tensor_450: "f32[544]" = torch.ops.aten.mul.Tensor(squeeze_dims_269, 0.1)
        mul_tensor_451: "f32[544]" = torch.ops.aten.mul.Tensor(arg543_1, 0.9)
        add_tensor_271: "f32[544]" = torch.ops.aten.add.Tensor(mul_tensor_450, mul_tensor_451);  mul_tensor_450 = mul_tensor_451 = None
        squeeze_dims_271: "f32[544]" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
        mul_tensor_452: "f32[544]" = torch.ops.aten.mul.Tensor(squeeze_dims_271, 1.005128205128205);  squeeze_dims_271 = None
        mul_tensor_453: "f32[544]" = torch.ops.aten.mul.Tensor(mul_tensor_452, 0.1);  mul_tensor_452 = None
        mul_tensor_454: "f32[544]" = torch.ops.aten.mul.Tensor(arg544_1, 0.9)
        add_tensor_272: "f32[544]" = torch.ops.aten.add.Tensor(mul_tensor_453, mul_tensor_454);  mul_tensor_453 = mul_tensor_454 = None
        add_tensor_273: "i64[]" = torch.ops.aten.add.Tensor(arg548_1, 1)
        squeeze_dims_272: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
        squeeze_dims_273: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
        mul_tensor_455: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_272, 0.1)
        mul_tensor_456: "f32[128]" = torch.ops.aten.mul.Tensor(arg549_1, 0.9)
        add_tensor_274: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_455, mul_tensor_456);  mul_tensor_455 = mul_tensor_456 = None
        squeeze_dims_274: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
        mul_tensor_457: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_274, 1.005128205128205);  squeeze_dims_274 = None
        mul_tensor_458: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_457, 0.1);  mul_tensor_457 = None
        mul_tensor_459: "f32[128]" = torch.ops.aten.mul.Tensor(arg550_1, 0.9)
        add_tensor_275: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_458, mul_tensor_459);  mul_tensor_458 = mul_tensor_459 = None
        add_tensor_276: "i64[]" = torch.ops.aten.add.Tensor(arg554_1, 1)
        squeeze_dims_275: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
        squeeze_dims_276: "f32[576]" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
        mul_tensor_460: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_275, 0.1)
        mul_tensor_461: "f32[576]" = torch.ops.aten.mul.Tensor(arg555_1, 0.9)
        add_tensor_277: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_460, mul_tensor_461);  mul_tensor_460 = mul_tensor_461 = None
        squeeze_dims_277: "f32[576]" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
        mul_tensor_462: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_dims_277, 1.005128205128205);  squeeze_dims_277 = None
        mul_tensor_463: "f32[576]" = torch.ops.aten.mul.Tensor(mul_tensor_462, 0.1);  mul_tensor_462 = None
        mul_tensor_464: "f32[576]" = torch.ops.aten.mul.Tensor(arg556_1, 0.9)
        add_tensor_278: "f32[576]" = torch.ops.aten.add.Tensor(mul_tensor_463, mul_tensor_464);  mul_tensor_463 = mul_tensor_464 = None
        add_tensor_279: "i64[]" = torch.ops.aten.add.Tensor(arg560_1, 1)
        squeeze_dims_278: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
        squeeze_dims_279: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
        mul_tensor_465: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_278, 0.1)
        mul_tensor_466: "f32[128]" = torch.ops.aten.mul.Tensor(arg561_1, 0.9)
        add_tensor_280: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_465, mul_tensor_466);  mul_tensor_465 = mul_tensor_466 = None
        squeeze_dims_280: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
        mul_tensor_467: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_280, 1.005128205128205);  squeeze_dims_280 = None
        mul_tensor_468: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_467, 0.1);  mul_tensor_467 = None
        mul_tensor_469: "f32[128]" = torch.ops.aten.mul.Tensor(arg562_1, 0.9)
        add_tensor_281: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_468, mul_tensor_469);  mul_tensor_468 = mul_tensor_469 = None
        add_tensor_282: "i64[]" = torch.ops.aten.add.Tensor(arg566_1, 1)
        squeeze_dims_281: "f32[608]" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
        squeeze_dims_282: "f32[608]" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
        mul_tensor_470: "f32[608]" = torch.ops.aten.mul.Tensor(squeeze_dims_281, 0.1)
        mul_tensor_471: "f32[608]" = torch.ops.aten.mul.Tensor(arg567_1, 0.9)
        add_tensor_283: "f32[608]" = torch.ops.aten.add.Tensor(mul_tensor_470, mul_tensor_471);  mul_tensor_470 = mul_tensor_471 = None
        squeeze_dims_283: "f32[608]" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
        mul_tensor_472: "f32[608]" = torch.ops.aten.mul.Tensor(squeeze_dims_283, 1.005128205128205);  squeeze_dims_283 = None
        mul_tensor_473: "f32[608]" = torch.ops.aten.mul.Tensor(mul_tensor_472, 0.1);  mul_tensor_472 = None
        mul_tensor_474: "f32[608]" = torch.ops.aten.mul.Tensor(arg568_1, 0.9)
        add_tensor_284: "f32[608]" = torch.ops.aten.add.Tensor(mul_tensor_473, mul_tensor_474);  mul_tensor_473 = mul_tensor_474 = None
        add_tensor_285: "i64[]" = torch.ops.aten.add.Tensor(arg572_1, 1)
        squeeze_dims_284: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
        squeeze_dims_285: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
        mul_tensor_475: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_284, 0.1)
        mul_tensor_476: "f32[128]" = torch.ops.aten.mul.Tensor(arg573_1, 0.9)
        add_tensor_286: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_475, mul_tensor_476);  mul_tensor_475 = mul_tensor_476 = None
        squeeze_dims_286: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
        mul_tensor_477: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_286, 1.005128205128205);  squeeze_dims_286 = None
        mul_tensor_478: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_477, 0.1);  mul_tensor_477 = None
        mul_tensor_479: "f32[128]" = torch.ops.aten.mul.Tensor(arg574_1, 0.9)
        add_tensor_287: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_478, mul_tensor_479);  mul_tensor_478 = mul_tensor_479 = None
        add_tensor_288: "i64[]" = torch.ops.aten.add.Tensor(arg578_1, 1)
        squeeze_dims_287: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
        squeeze_dims_288: "f32[640]" = torch.ops.aten.squeeze.dims(rsqrt_96, [0, 2, 3]);  rsqrt_96 = None
        mul_tensor_480: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims_287, 0.1)
        mul_tensor_481: "f32[640]" = torch.ops.aten.mul.Tensor(arg579_1, 0.9)
        add_tensor_289: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_480, mul_tensor_481);  mul_tensor_480 = mul_tensor_481 = None
        squeeze_dims_289: "f32[640]" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
        mul_tensor_482: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_dims_289, 1.005128205128205);  squeeze_dims_289 = None
        mul_tensor_483: "f32[640]" = torch.ops.aten.mul.Tensor(mul_tensor_482, 0.1);  mul_tensor_482 = None
        mul_tensor_484: "f32[640]" = torch.ops.aten.mul.Tensor(arg580_1, 0.9)
        add_tensor_290: "f32[640]" = torch.ops.aten.add.Tensor(mul_tensor_483, mul_tensor_484);  mul_tensor_483 = mul_tensor_484 = None
        add_tensor_291: "i64[]" = torch.ops.aten.add.Tensor(arg584_1, 1)
        squeeze_dims_290: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
        squeeze_dims_291: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
        mul_tensor_485: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_290, 0.1)
        mul_tensor_486: "f32[128]" = torch.ops.aten.mul.Tensor(arg585_1, 0.9)
        add_tensor_292: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_485, mul_tensor_486);  mul_tensor_485 = mul_tensor_486 = None
        squeeze_dims_292: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
        mul_tensor_487: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_292, 1.005128205128205);  squeeze_dims_292 = None
        mul_tensor_488: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_487, 0.1);  mul_tensor_487 = None
        mul_tensor_489: "f32[128]" = torch.ops.aten.mul.Tensor(arg586_1, 0.9)
        add_tensor_293: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_488, mul_tensor_489);  mul_tensor_488 = mul_tensor_489 = None
        add_tensor_294: "i64[]" = torch.ops.aten.add.Tensor(arg590_1, 1)
        squeeze_dims_293: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
        squeeze_dims_294: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_98, [0, 2, 3]);  rsqrt_98 = None
        mul_tensor_490: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_293, 0.1)
        mul_tensor_491: "f32[672]" = torch.ops.aten.mul.Tensor(arg591_1, 0.9)
        add_tensor_295: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_490, mul_tensor_491);  mul_tensor_490 = mul_tensor_491 = None
        squeeze_dims_295: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
        mul_tensor_492: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_295, 1.005128205128205);  squeeze_dims_295 = None
        mul_tensor_493: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_492, 0.1);  mul_tensor_492 = None
        mul_tensor_494: "f32[672]" = torch.ops.aten.mul.Tensor(arg592_1, 0.9)
        add_tensor_296: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_493, mul_tensor_494);  mul_tensor_493 = mul_tensor_494 = None
        add_tensor_297: "i64[]" = torch.ops.aten.add.Tensor(arg596_1, 1)
        squeeze_dims_296: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
        squeeze_dims_297: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_99, [0, 2, 3]);  rsqrt_99 = None
        mul_tensor_495: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_296, 0.1)
        mul_tensor_496: "f32[128]" = torch.ops.aten.mul.Tensor(arg597_1, 0.9)
        add_tensor_298: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_495, mul_tensor_496);  mul_tensor_495 = mul_tensor_496 = None
        squeeze_dims_298: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
        mul_tensor_497: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_298, 1.005128205128205);  squeeze_dims_298 = None
        mul_tensor_498: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_497, 0.1);  mul_tensor_497 = None
        mul_tensor_499: "f32[128]" = torch.ops.aten.mul.Tensor(arg598_1, 0.9)
        add_tensor_299: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_498, mul_tensor_499);  mul_tensor_498 = mul_tensor_499 = None
        add_tensor_300: "i64[]" = torch.ops.aten.add.Tensor(arg602_1, 1)
        squeeze_dims_299: "f32[704]" = torch.ops.aten.squeeze.dims(getitem_203, [0, 2, 3]);  getitem_203 = None
        squeeze_dims_300: "f32[704]" = torch.ops.aten.squeeze.dims(rsqrt_100, [0, 2, 3]);  rsqrt_100 = None
        mul_tensor_500: "f32[704]" = torch.ops.aten.mul.Tensor(squeeze_dims_299, 0.1)
        mul_tensor_501: "f32[704]" = torch.ops.aten.mul.Tensor(arg603_1, 0.9)
        add_tensor_301: "f32[704]" = torch.ops.aten.add.Tensor(mul_tensor_500, mul_tensor_501);  mul_tensor_500 = mul_tensor_501 = None
        squeeze_dims_301: "f32[704]" = torch.ops.aten.squeeze.dims(getitem_202, [0, 2, 3]);  getitem_202 = None
        mul_tensor_502: "f32[704]" = torch.ops.aten.mul.Tensor(squeeze_dims_301, 1.005128205128205);  squeeze_dims_301 = None
        mul_tensor_503: "f32[704]" = torch.ops.aten.mul.Tensor(mul_tensor_502, 0.1);  mul_tensor_502 = None
        mul_tensor_504: "f32[704]" = torch.ops.aten.mul.Tensor(arg604_1, 0.9)
        add_tensor_302: "f32[704]" = torch.ops.aten.add.Tensor(mul_tensor_503, mul_tensor_504);  mul_tensor_503 = mul_tensor_504 = None
        add_tensor_303: "i64[]" = torch.ops.aten.add.Tensor(arg608_1, 1)
        squeeze_dims_302: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_205, [0, 2, 3]);  getitem_205 = None
        squeeze_dims_303: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_101, [0, 2, 3]);  rsqrt_101 = None
        mul_tensor_505: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_302, 0.1)
        mul_tensor_506: "f32[128]" = torch.ops.aten.mul.Tensor(arg609_1, 0.9)
        add_tensor_304: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_505, mul_tensor_506);  mul_tensor_505 = mul_tensor_506 = None
        squeeze_dims_304: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_204, [0, 2, 3]);  getitem_204 = None
        mul_tensor_507: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_304, 1.005128205128205);  squeeze_dims_304 = None
        mul_tensor_508: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_507, 0.1);  mul_tensor_507 = None
        mul_tensor_509: "f32[128]" = torch.ops.aten.mul.Tensor(arg610_1, 0.9)
        add_tensor_305: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_508, mul_tensor_509);  mul_tensor_508 = mul_tensor_509 = None
        add_tensor_306: "i64[]" = torch.ops.aten.add.Tensor(arg614_1, 1)
        squeeze_dims_305: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_207, [0, 2, 3]);  getitem_207 = None
        squeeze_dims_306: "f32[736]" = torch.ops.aten.squeeze.dims(rsqrt_102, [0, 2, 3]);  rsqrt_102 = None
        mul_tensor_510: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_dims_305, 0.1)
        mul_tensor_511: "f32[736]" = torch.ops.aten.mul.Tensor(arg615_1, 0.9)
        add_tensor_307: "f32[736]" = torch.ops.aten.add.Tensor(mul_tensor_510, mul_tensor_511);  mul_tensor_510 = mul_tensor_511 = None
        squeeze_dims_307: "f32[736]" = torch.ops.aten.squeeze.dims(getitem_206, [0, 2, 3]);  getitem_206 = None
        mul_tensor_512: "f32[736]" = torch.ops.aten.mul.Tensor(squeeze_dims_307, 1.005128205128205);  squeeze_dims_307 = None
        mul_tensor_513: "f32[736]" = torch.ops.aten.mul.Tensor(mul_tensor_512, 0.1);  mul_tensor_512 = None
        mul_tensor_514: "f32[736]" = torch.ops.aten.mul.Tensor(arg616_1, 0.9)
        add_tensor_308: "f32[736]" = torch.ops.aten.add.Tensor(mul_tensor_513, mul_tensor_514);  mul_tensor_513 = mul_tensor_514 = None
        add_tensor_309: "i64[]" = torch.ops.aten.add.Tensor(arg620_1, 1)
        squeeze_dims_308: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
        squeeze_dims_309: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_103, [0, 2, 3]);  rsqrt_103 = None
        mul_tensor_515: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_308, 0.1)
        mul_tensor_516: "f32[128]" = torch.ops.aten.mul.Tensor(arg621_1, 0.9)
        add_tensor_310: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_515, mul_tensor_516);  mul_tensor_515 = mul_tensor_516 = None
        squeeze_dims_310: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
        mul_tensor_517: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_310, 1.005128205128205);  squeeze_dims_310 = None
        mul_tensor_518: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_517, 0.1);  mul_tensor_517 = None
        mul_tensor_519: "f32[128]" = torch.ops.aten.mul.Tensor(arg622_1, 0.9)
        add_tensor_311: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_518, mul_tensor_519);  mul_tensor_518 = mul_tensor_519 = None
        add_tensor_312: "i64[]" = torch.ops.aten.add.Tensor(arg626_1, 1)
        squeeze_dims_311: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
        squeeze_dims_312: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_104, [0, 2, 3]);  rsqrt_104 = None
        mul_tensor_520: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_311, 0.1)
        mul_tensor_521: "f32[768]" = torch.ops.aten.mul.Tensor(arg627_1, 0.9)
        add_tensor_313: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_520, mul_tensor_521);  mul_tensor_520 = mul_tensor_521 = None
        squeeze_dims_313: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
        mul_tensor_522: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_dims_313, 1.005128205128205);  squeeze_dims_313 = None
        mul_tensor_523: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_522, 0.1);  mul_tensor_522 = None
        mul_tensor_524: "f32[768]" = torch.ops.aten.mul.Tensor(arg628_1, 0.9)
        add_tensor_314: "f32[768]" = torch.ops.aten.add.Tensor(mul_tensor_523, mul_tensor_524);  mul_tensor_523 = mul_tensor_524 = None
        add_tensor_315: "i64[]" = torch.ops.aten.add.Tensor(arg632_1, 1)
        squeeze_dims_314: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3]);  getitem_213 = None
        squeeze_dims_315: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_105, [0, 2, 3]);  rsqrt_105 = None
        mul_tensor_525: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_314, 0.1)
        mul_tensor_526: "f32[128]" = torch.ops.aten.mul.Tensor(arg633_1, 0.9)
        add_tensor_316: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_525, mul_tensor_526);  mul_tensor_525 = mul_tensor_526 = None
        squeeze_dims_316: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
        mul_tensor_527: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_316, 1.005128205128205);  squeeze_dims_316 = None
        mul_tensor_528: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_527, 0.1);  mul_tensor_527 = None
        mul_tensor_529: "f32[128]" = torch.ops.aten.mul.Tensor(arg634_1, 0.9)
        add_tensor_317: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_528, mul_tensor_529);  mul_tensor_528 = mul_tensor_529 = None
        add_tensor_318: "i64[]" = torch.ops.aten.add.Tensor(arg638_1, 1)
        squeeze_dims_317: "f32[800]" = torch.ops.aten.squeeze.dims(getitem_215, [0, 2, 3]);  getitem_215 = None
        squeeze_dims_318: "f32[800]" = torch.ops.aten.squeeze.dims(rsqrt_106, [0, 2, 3]);  rsqrt_106 = None
        mul_tensor_530: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_dims_317, 0.1)
        mul_tensor_531: "f32[800]" = torch.ops.aten.mul.Tensor(arg639_1, 0.9)
        add_tensor_319: "f32[800]" = torch.ops.aten.add.Tensor(mul_tensor_530, mul_tensor_531);  mul_tensor_530 = mul_tensor_531 = None
        squeeze_dims_319: "f32[800]" = torch.ops.aten.squeeze.dims(getitem_214, [0, 2, 3]);  getitem_214 = None
        mul_tensor_532: "f32[800]" = torch.ops.aten.mul.Tensor(squeeze_dims_319, 1.005128205128205);  squeeze_dims_319 = None
        mul_tensor_533: "f32[800]" = torch.ops.aten.mul.Tensor(mul_tensor_532, 0.1);  mul_tensor_532 = None
        mul_tensor_534: "f32[800]" = torch.ops.aten.mul.Tensor(arg640_1, 0.9)
        add_tensor_320: "f32[800]" = torch.ops.aten.add.Tensor(mul_tensor_533, mul_tensor_534);  mul_tensor_533 = mul_tensor_534 = None
        add_tensor_321: "i64[]" = torch.ops.aten.add.Tensor(arg644_1, 1)
        squeeze_dims_320: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_217, [0, 2, 3]);  getitem_217 = None
        squeeze_dims_321: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_107, [0, 2, 3]);  rsqrt_107 = None
        mul_tensor_535: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_320, 0.1)
        mul_tensor_536: "f32[128]" = torch.ops.aten.mul.Tensor(arg645_1, 0.9)
        add_tensor_322: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_535, mul_tensor_536);  mul_tensor_535 = mul_tensor_536 = None
        squeeze_dims_322: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_216, [0, 2, 3]);  getitem_216 = None
        mul_tensor_537: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_322, 1.005128205128205);  squeeze_dims_322 = None
        mul_tensor_538: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_537, 0.1);  mul_tensor_537 = None
        mul_tensor_539: "f32[128]" = torch.ops.aten.mul.Tensor(arg646_1, 0.9)
        add_tensor_323: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_538, mul_tensor_539);  mul_tensor_538 = mul_tensor_539 = None
        add_tensor_324: "i64[]" = torch.ops.aten.add.Tensor(arg650_1, 1)
        squeeze_dims_323: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_219, [0, 2, 3]);  getitem_219 = None
        squeeze_dims_324: "f32[832]" = torch.ops.aten.squeeze.dims(rsqrt_108, [0, 2, 3]);  rsqrt_108 = None
        mul_tensor_540: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_dims_323, 0.1)
        mul_tensor_541: "f32[832]" = torch.ops.aten.mul.Tensor(arg651_1, 0.9)
        add_tensor_325: "f32[832]" = torch.ops.aten.add.Tensor(mul_tensor_540, mul_tensor_541);  mul_tensor_540 = mul_tensor_541 = None
        squeeze_dims_325: "f32[832]" = torch.ops.aten.squeeze.dims(getitem_218, [0, 2, 3]);  getitem_218 = None
        mul_tensor_542: "f32[832]" = torch.ops.aten.mul.Tensor(squeeze_dims_325, 1.005128205128205);  squeeze_dims_325 = None
        mul_tensor_543: "f32[832]" = torch.ops.aten.mul.Tensor(mul_tensor_542, 0.1);  mul_tensor_542 = None
        mul_tensor_544: "f32[832]" = torch.ops.aten.mul.Tensor(arg652_1, 0.9)
        add_tensor_326: "f32[832]" = torch.ops.aten.add.Tensor(mul_tensor_543, mul_tensor_544);  mul_tensor_543 = mul_tensor_544 = None
        add_tensor_327: "i64[]" = torch.ops.aten.add.Tensor(arg656_1, 1)
        squeeze_dims_326: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_221, [0, 2, 3]);  getitem_221 = None
        squeeze_dims_327: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_109, [0, 2, 3]);  rsqrt_109 = None
        mul_tensor_545: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_326, 0.1)
        mul_tensor_546: "f32[128]" = torch.ops.aten.mul.Tensor(arg657_1, 0.9)
        add_tensor_328: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_545, mul_tensor_546);  mul_tensor_545 = mul_tensor_546 = None
        squeeze_dims_328: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_220, [0, 2, 3]);  getitem_220 = None
        mul_tensor_547: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_328, 1.005128205128205);  squeeze_dims_328 = None
        mul_tensor_548: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_547, 0.1);  mul_tensor_547 = None
        mul_tensor_549: "f32[128]" = torch.ops.aten.mul.Tensor(arg658_1, 0.9)
        add_tensor_329: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_548, mul_tensor_549);  mul_tensor_548 = mul_tensor_549 = None
        add_tensor_330: "i64[]" = torch.ops.aten.add.Tensor(arg662_1, 1)
        squeeze_dims_329: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
        squeeze_dims_330: "f32[864]" = torch.ops.aten.squeeze.dims(rsqrt_110, [0, 2, 3]);  rsqrt_110 = None
        mul_tensor_550: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_dims_329, 0.1)
        mul_tensor_551: "f32[864]" = torch.ops.aten.mul.Tensor(arg663_1, 0.9)
        add_tensor_331: "f32[864]" = torch.ops.aten.add.Tensor(mul_tensor_550, mul_tensor_551);  mul_tensor_550 = mul_tensor_551 = None
        squeeze_dims_331: "f32[864]" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
        mul_tensor_552: "f32[864]" = torch.ops.aten.mul.Tensor(squeeze_dims_331, 1.005128205128205);  squeeze_dims_331 = None
        mul_tensor_553: "f32[864]" = torch.ops.aten.mul.Tensor(mul_tensor_552, 0.1);  mul_tensor_552 = None
        mul_tensor_554: "f32[864]" = torch.ops.aten.mul.Tensor(arg664_1, 0.9)
        add_tensor_332: "f32[864]" = torch.ops.aten.add.Tensor(mul_tensor_553, mul_tensor_554);  mul_tensor_553 = mul_tensor_554 = None
        add_tensor_333: "i64[]" = torch.ops.aten.add.Tensor(arg668_1, 1)
        squeeze_dims_332: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_225, [0, 2, 3]);  getitem_225 = None
        squeeze_dims_333: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_111, [0, 2, 3]);  rsqrt_111 = None
        mul_tensor_555: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_332, 0.1)
        mul_tensor_556: "f32[128]" = torch.ops.aten.mul.Tensor(arg669_1, 0.9)
        add_tensor_334: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_555, mul_tensor_556);  mul_tensor_555 = mul_tensor_556 = None
        squeeze_dims_334: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_224, [0, 2, 3]);  getitem_224 = None
        mul_tensor_557: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_334, 1.005128205128205);  squeeze_dims_334 = None
        mul_tensor_558: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_557, 0.1);  mul_tensor_557 = None
        mul_tensor_559: "f32[128]" = torch.ops.aten.mul.Tensor(arg670_1, 0.9)
        add_tensor_335: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_558, mul_tensor_559);  mul_tensor_558 = mul_tensor_559 = None
        add_tensor_336: "i64[]" = torch.ops.aten.add.Tensor(arg674_1, 1)
        squeeze_dims_335: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
        squeeze_dims_336: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_112, [0, 2, 3]);  rsqrt_112 = None
        mul_tensor_560: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_dims_335, 0.1)
        mul_tensor_561: "f32[896]" = torch.ops.aten.mul.Tensor(arg675_1, 0.9)
        add_tensor_337: "f32[896]" = torch.ops.aten.add.Tensor(mul_tensor_560, mul_tensor_561);  mul_tensor_560 = mul_tensor_561 = None
        squeeze_dims_337: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_226, [0, 2, 3]);  getitem_226 = None
        mul_tensor_562: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_dims_337, 1.005128205128205);  squeeze_dims_337 = None
        mul_tensor_563: "f32[896]" = torch.ops.aten.mul.Tensor(mul_tensor_562, 0.1);  mul_tensor_562 = None
        mul_tensor_564: "f32[896]" = torch.ops.aten.mul.Tensor(arg676_1, 0.9)
        add_tensor_338: "f32[896]" = torch.ops.aten.add.Tensor(mul_tensor_563, mul_tensor_564);  mul_tensor_563 = mul_tensor_564 = None
        add_tensor_339: "i64[]" = torch.ops.aten.add.Tensor(arg680_1, 1)
        squeeze_dims_338: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
        squeeze_dims_339: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_113, [0, 2, 3]);  rsqrt_113 = None
        mul_tensor_565: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_338, 0.1)
        mul_tensor_566: "f32[128]" = torch.ops.aten.mul.Tensor(arg681_1, 0.9)
        add_tensor_340: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_565, mul_tensor_566);  mul_tensor_565 = mul_tensor_566 = None
        squeeze_dims_340: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
        mul_tensor_567: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_340, 1.005128205128205);  squeeze_dims_340 = None
        mul_tensor_568: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_567, 0.1);  mul_tensor_567 = None
        mul_tensor_569: "f32[128]" = torch.ops.aten.mul.Tensor(arg682_1, 0.9)
        add_tensor_341: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_568, mul_tensor_569);  mul_tensor_568 = mul_tensor_569 = None
        add_tensor_342: "i64[]" = torch.ops.aten.add.Tensor(arg686_1, 1)
        squeeze_dims_341: "f32[928]" = torch.ops.aten.squeeze.dims(getitem_231, [0, 2, 3]);  getitem_231 = None
        squeeze_dims_342: "f32[928]" = torch.ops.aten.squeeze.dims(rsqrt_114, [0, 2, 3]);  rsqrt_114 = None
        mul_tensor_570: "f32[928]" = torch.ops.aten.mul.Tensor(squeeze_dims_341, 0.1)
        mul_tensor_571: "f32[928]" = torch.ops.aten.mul.Tensor(arg687_1, 0.9)
        add_tensor_343: "f32[928]" = torch.ops.aten.add.Tensor(mul_tensor_570, mul_tensor_571);  mul_tensor_570 = mul_tensor_571 = None
        squeeze_dims_343: "f32[928]" = torch.ops.aten.squeeze.dims(getitem_230, [0, 2, 3]);  getitem_230 = None
        mul_tensor_572: "f32[928]" = torch.ops.aten.mul.Tensor(squeeze_dims_343, 1.005128205128205);  squeeze_dims_343 = None
        mul_tensor_573: "f32[928]" = torch.ops.aten.mul.Tensor(mul_tensor_572, 0.1);  mul_tensor_572 = None
        mul_tensor_574: "f32[928]" = torch.ops.aten.mul.Tensor(arg688_1, 0.9)
        add_tensor_344: "f32[928]" = torch.ops.aten.add.Tensor(mul_tensor_573, mul_tensor_574);  mul_tensor_573 = mul_tensor_574 = None
        add_tensor_345: "i64[]" = torch.ops.aten.add.Tensor(arg692_1, 1)
        squeeze_dims_344: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_233, [0, 2, 3]);  getitem_233 = None
        squeeze_dims_345: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_115, [0, 2, 3]);  rsqrt_115 = None
        mul_tensor_575: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_344, 0.1)
        mul_tensor_576: "f32[128]" = torch.ops.aten.mul.Tensor(arg693_1, 0.9)
        add_tensor_346: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_575, mul_tensor_576);  mul_tensor_575 = mul_tensor_576 = None
        squeeze_dims_346: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_232, [0, 2, 3]);  getitem_232 = None
        mul_tensor_577: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_346, 1.005128205128205);  squeeze_dims_346 = None
        mul_tensor_578: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_577, 0.1);  mul_tensor_577 = None
        mul_tensor_579: "f32[128]" = torch.ops.aten.mul.Tensor(arg694_1, 0.9)
        add_tensor_347: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_578, mul_tensor_579);  mul_tensor_578 = mul_tensor_579 = None
        add_tensor_348: "i64[]" = torch.ops.aten.add.Tensor(arg698_1, 1)
        squeeze_dims_347: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_235, [0, 2, 3]);  getitem_235 = None
        squeeze_dims_348: "f32[960]" = torch.ops.aten.squeeze.dims(rsqrt_116, [0, 2, 3]);  rsqrt_116 = None
        mul_tensor_580: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_347, 0.1)
        mul_tensor_581: "f32[960]" = torch.ops.aten.mul.Tensor(arg699_1, 0.9)
        add_tensor_349: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_580, mul_tensor_581);  mul_tensor_580 = mul_tensor_581 = None
        squeeze_dims_349: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_234, [0, 2, 3]);  getitem_234 = None
        mul_tensor_582: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_349, 1.005128205128205);  squeeze_dims_349 = None
        mul_tensor_583: "f32[960]" = torch.ops.aten.mul.Tensor(mul_tensor_582, 0.1);  mul_tensor_582 = None
        mul_tensor_584: "f32[960]" = torch.ops.aten.mul.Tensor(arg700_1, 0.9)
        add_tensor_350: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_583, mul_tensor_584);  mul_tensor_583 = mul_tensor_584 = None
        add_tensor_351: "i64[]" = torch.ops.aten.add.Tensor(arg704_1, 1)
        squeeze_dims_350: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_237, [0, 2, 3]);  getitem_237 = None
        squeeze_dims_351: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_117, [0, 2, 3]);  rsqrt_117 = None
        mul_tensor_585: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_350, 0.1)
        mul_tensor_586: "f32[128]" = torch.ops.aten.mul.Tensor(arg705_1, 0.9)
        add_tensor_352: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_585, mul_tensor_586);  mul_tensor_585 = mul_tensor_586 = None
        squeeze_dims_352: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_236, [0, 2, 3]);  getitem_236 = None
        mul_tensor_587: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_352, 1.005128205128205);  squeeze_dims_352 = None
        mul_tensor_588: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_587, 0.1);  mul_tensor_587 = None
        mul_tensor_589: "f32[128]" = torch.ops.aten.mul.Tensor(arg706_1, 0.9)
        add_tensor_353: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_588, mul_tensor_589);  mul_tensor_588 = mul_tensor_589 = None
        add_tensor_354: "i64[]" = torch.ops.aten.add.Tensor(arg710_1, 1)
        squeeze_dims_353: "f32[992]" = torch.ops.aten.squeeze.dims(getitem_239, [0, 2, 3]);  getitem_239 = None
        squeeze_dims_354: "f32[992]" = torch.ops.aten.squeeze.dims(rsqrt_118, [0, 2, 3]);  rsqrt_118 = None
        mul_tensor_590: "f32[992]" = torch.ops.aten.mul.Tensor(squeeze_dims_353, 0.1)
        mul_tensor_591: "f32[992]" = torch.ops.aten.mul.Tensor(arg711_1, 0.9)
        add_tensor_355: "f32[992]" = torch.ops.aten.add.Tensor(mul_tensor_590, mul_tensor_591);  mul_tensor_590 = mul_tensor_591 = None
        squeeze_dims_355: "f32[992]" = torch.ops.aten.squeeze.dims(getitem_238, [0, 2, 3]);  getitem_238 = None
        mul_tensor_592: "f32[992]" = torch.ops.aten.mul.Tensor(squeeze_dims_355, 1.005128205128205);  squeeze_dims_355 = None
        mul_tensor_593: "f32[992]" = torch.ops.aten.mul.Tensor(mul_tensor_592, 0.1);  mul_tensor_592 = None
        mul_tensor_594: "f32[992]" = torch.ops.aten.mul.Tensor(arg712_1, 0.9)
        add_tensor_356: "f32[992]" = torch.ops.aten.add.Tensor(mul_tensor_593, mul_tensor_594);  mul_tensor_593 = mul_tensor_594 = None
        add_tensor_357: "i64[]" = torch.ops.aten.add.Tensor(arg716_1, 1)
        squeeze_dims_356: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_241, [0, 2, 3]);  getitem_241 = None
        squeeze_dims_357: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_119, [0, 2, 3]);  rsqrt_119 = None
        mul_tensor_595: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_356, 0.1)
        mul_tensor_596: "f32[128]" = torch.ops.aten.mul.Tensor(arg717_1, 0.9)
        add_tensor_358: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_595, mul_tensor_596);  mul_tensor_595 = mul_tensor_596 = None
        squeeze_dims_358: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_240, [0, 2, 3]);  getitem_240 = None
        mul_tensor_597: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_dims_358, 1.005128205128205);  squeeze_dims_358 = None
        mul_tensor_598: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_597, 0.1);  mul_tensor_597 = None
        mul_tensor_599: "f32[128]" = torch.ops.aten.mul.Tensor(arg718_1, 0.9)
        add_tensor_359: "f32[128]" = torch.ops.aten.add.Tensor(mul_tensor_598, mul_tensor_599);  mul_tensor_598 = mul_tensor_599 = None
        add_tensor_360: "i64[]" = torch.ops.aten.add.Tensor(arg722_1, 1)
        add_tensor_361: "f32[1, 1024, 1, 1]" = torch.ops.aten.add.Tensor(getitem_242, 1e-05)
        rsqrt_default: "f32[1, 1024, 1, 1]" = torch.ops.aten.rsqrt.default(add_tensor_361);  add_tensor_361 = None
        sub_tensor: "f32[4, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(cat_57, getitem_243);  cat_57 = None
        mul_tensor_600: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        squeeze_dims_359: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_243, [0, 2, 3]);  getitem_243 = None
        mul_tensor_601: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_359, 0.1);  squeeze_dims_359 = None
        mul_tensor_602: "f32[1024]" = torch.ops.aten.mul.Tensor(arg723_1, 0.9)
        add_tensor_362: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_601, mul_tensor_602);  mul_tensor_601 = mul_tensor_602 = None
        squeeze_dims_360: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_242, [0, 2, 3]);  getitem_242 = None
        mul_tensor_603: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_dims_360, 1.005128205128205);  squeeze_dims_360 = None
        mul_tensor_604: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_tensor_603, 0.1);  mul_tensor_603 = None
        mul_tensor_605: "f32[1024]" = torch.ops.aten.mul.Tensor(arg724_1, 0.9)
        add_tensor_363: "f32[1024]" = torch.ops.aten.add.Tensor(mul_tensor_604, mul_tensor_605);  mul_tensor_604 = mul_tensor_605 = None
        unsqueeze_default: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg725_1, -1);  arg725_1 = None
        unsqueeze_default_1: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_606: "f32[4, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_tensor_600, unsqueeze_default_1);  mul_tensor_600 = unsqueeze_default_1 = None
        unsqueeze_default_2: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(arg726_1, -1);  arg726_1 = None
        unsqueeze_default_3: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor_364: "f32[4, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_tensor_606, unsqueeze_default_3);  mul_tensor_606 = unsqueeze_default_3 = None
        convert_element_type_default: "f16[4, 1024, 7, 7]" = torch.ops.prims.convert_element_type.default(add_tensor_364, torch.float16);  add_tensor_364 = None
        relu_default: "f16[4, 1024, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_default);  convert_element_type_default = None
        mean_dim: "f16[4, 1024, 1, 1]" = torch.ops.aten.mean.dim(relu_default, [-1, -2], True);  relu_default = None
        reshape_default: "f16[4, 1024]" = torch.ops.aten.reshape.default(mean_dim, [4, 1024]);  mean_dim = None
        convert_element_type_default_1: "f16[1000]" = torch.ops.prims.convert_element_type.default(arg728_1, torch.float16);  arg728_1 = None
        convert_element_type_default_2: "f16[1000, 1024]" = torch.ops.prims.convert_element_type.default(arg727_1, torch.float16);  arg727_1 = None
        permute_default: "f16[1024, 1000]" = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        permute_default_1: "f16[1000, 1024]" = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        unsqueeze_default_4: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_356, 0);  squeeze_dims_356 = None
        unsqueeze_default_5: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 3);  unsqueeze_default_5 = None
        unsqueeze_default_7: "f32[1, 992]" = torch.ops.aten.unsqueeze.default(squeeze_dims_353, 0);  squeeze_dims_353 = None
        unsqueeze_default_8: "f32[1, 992, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 2);  unsqueeze_default_7 = None
        unsqueeze_default_9: "f32[1, 992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 3);  unsqueeze_default_8 = None
        unsqueeze_default_10: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_350, 0);  squeeze_dims_350 = None
        unsqueeze_default_11: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        unsqueeze_default_12: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_11, 3);  unsqueeze_default_11 = None
        unsqueeze_default_13: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_dims_347, 0);  squeeze_dims_347 = None
        unsqueeze_default_14: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 2);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 3);  unsqueeze_default_14 = None
        unsqueeze_default_16: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_344, 0);  squeeze_dims_344 = None
        unsqueeze_default_17: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        unsqueeze_default_18: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_17, 3);  unsqueeze_default_17 = None
        unsqueeze_default_19: "f32[1, 928]" = torch.ops.aten.unsqueeze.default(squeeze_dims_341, 0);  squeeze_dims_341 = None
        unsqueeze_default_20: "f32[1, 928, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 2);  unsqueeze_default_19 = None
        unsqueeze_default_21: "f32[1, 928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 3);  unsqueeze_default_20 = None
        unsqueeze_default_22: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_338, 0);  squeeze_dims_338 = None
        unsqueeze_default_23: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_23, 3);  unsqueeze_default_23 = None
        unsqueeze_default_25: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_dims_335, 0);  squeeze_dims_335 = None
        unsqueeze_default_26: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 2);  unsqueeze_default_25 = None
        unsqueeze_default_27: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 3);  unsqueeze_default_26 = None
        unsqueeze_default_28: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_332, 0);  squeeze_dims_332 = None
        unsqueeze_default_29: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        unsqueeze_default_30: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_29, 3);  unsqueeze_default_29 = None
        unsqueeze_default_31: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_dims_329, 0);  squeeze_dims_329 = None
        unsqueeze_default_32: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 2);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 3);  unsqueeze_default_32 = None
        unsqueeze_default_34: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_326, 0);  squeeze_dims_326 = None
        unsqueeze_default_35: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 2);  unsqueeze_default_34 = None
        unsqueeze_default_36: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_35, 3);  unsqueeze_default_35 = None
        unsqueeze_default_37: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(squeeze_dims_323, 0);  squeeze_dims_323 = None
        unsqueeze_default_38: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 2);  unsqueeze_default_37 = None
        unsqueeze_default_39: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 3);  unsqueeze_default_38 = None
        unsqueeze_default_40: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_320, 0);  squeeze_dims_320 = None
        unsqueeze_default_41: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 2);  unsqueeze_default_40 = None
        unsqueeze_default_42: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_41, 3);  unsqueeze_default_41 = None
        unsqueeze_default_43: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(squeeze_dims_317, 0);  squeeze_dims_317 = None
        unsqueeze_default_44: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 2);  unsqueeze_default_43 = None
        unsqueeze_default_45: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 3);  unsqueeze_default_44 = None
        unsqueeze_default_46: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_314, 0);  squeeze_dims_314 = None
        unsqueeze_default_47: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 2);  unsqueeze_default_46 = None
        unsqueeze_default_48: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_47, 3);  unsqueeze_default_47 = None
        unsqueeze_default_49: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_311, 0);  squeeze_dims_311 = None
        unsqueeze_default_50: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_49, 2);  unsqueeze_default_49 = None
        unsqueeze_default_51: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, 3);  unsqueeze_default_50 = None
        unsqueeze_default_52: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_308, 0);  squeeze_dims_308 = None
        unsqueeze_default_53: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 2);  unsqueeze_default_52 = None
        unsqueeze_default_54: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_53, 3);  unsqueeze_default_53 = None
        unsqueeze_default_55: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_dims_305, 0);  squeeze_dims_305 = None
        unsqueeze_default_56: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_55, 2);  unsqueeze_default_55 = None
        unsqueeze_default_57: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 3);  unsqueeze_default_56 = None
        unsqueeze_default_58: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_302, 0);  squeeze_dims_302 = None
        unsqueeze_default_59: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 2);  unsqueeze_default_58 = None
        unsqueeze_default_60: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_59, 3);  unsqueeze_default_59 = None
        unsqueeze_default_61: "f32[1, 704]" = torch.ops.aten.unsqueeze.default(squeeze_dims_299, 0);  squeeze_dims_299 = None
        unsqueeze_default_62: "f32[1, 704, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_61, 2);  unsqueeze_default_61 = None
        unsqueeze_default_63: "f32[1, 704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 3);  unsqueeze_default_62 = None
        unsqueeze_default_64: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_296, 0);  squeeze_dims_296 = None
        unsqueeze_default_65: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_64, 2);  unsqueeze_default_64 = None
        unsqueeze_default_66: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_65, 3);  unsqueeze_default_65 = None
        unsqueeze_default_67: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_dims_293, 0);  squeeze_dims_293 = None
        unsqueeze_default_68: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_67, 2);  unsqueeze_default_67 = None
        unsqueeze_default_69: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_68, 3);  unsqueeze_default_68 = None
        unsqueeze_default_70: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_290, 0);  squeeze_dims_290 = None
        unsqueeze_default_71: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_70, 2);  unsqueeze_default_70 = None
        unsqueeze_default_72: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_71, 3);  unsqueeze_default_71 = None
        unsqueeze_default_73: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_dims_287, 0);  squeeze_dims_287 = None
        unsqueeze_default_74: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_73, 2);  unsqueeze_default_73 = None
        unsqueeze_default_75: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_74, 3);  unsqueeze_default_74 = None
        unsqueeze_default_76: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_284, 0);  squeeze_dims_284 = None
        unsqueeze_default_77: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_76, 2);  unsqueeze_default_76 = None
        unsqueeze_default_78: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_77, 3);  unsqueeze_default_77 = None
        unsqueeze_default_79: "f32[1, 608]" = torch.ops.aten.unsqueeze.default(squeeze_dims_281, 0);  squeeze_dims_281 = None
        unsqueeze_default_80: "f32[1, 608, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_79, 2);  unsqueeze_default_79 = None
        unsqueeze_default_81: "f32[1, 608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_80, 3);  unsqueeze_default_80 = None
        unsqueeze_default_82: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_278, 0);  squeeze_dims_278 = None
        unsqueeze_default_83: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_82, 2);  unsqueeze_default_82 = None
        unsqueeze_default_84: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_83, 3);  unsqueeze_default_83 = None
        unsqueeze_default_85: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_dims_275, 0);  squeeze_dims_275 = None
        unsqueeze_default_86: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_85, 2);  unsqueeze_default_85 = None
        unsqueeze_default_87: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_86, 3);  unsqueeze_default_86 = None
        unsqueeze_default_88: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_272, 0);  squeeze_dims_272 = None
        unsqueeze_default_89: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_88, 2);  unsqueeze_default_88 = None
        unsqueeze_default_90: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_89, 3);  unsqueeze_default_89 = None
        unsqueeze_default_91: "f32[1, 544]" = torch.ops.aten.unsqueeze.default(squeeze_dims_269, 0);  squeeze_dims_269 = None
        unsqueeze_default_92: "f32[1, 544, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_91, 2);  unsqueeze_default_91 = None
        unsqueeze_default_93: "f32[1, 544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_92, 3);  unsqueeze_default_92 = None
        unsqueeze_default_94: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_266, 0);  squeeze_dims_266 = None
        unsqueeze_default_95: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_94, 2);  unsqueeze_default_94 = None
        unsqueeze_default_96: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_95, 3);  unsqueeze_default_95 = None
        unsqueeze_default_97: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_263, 0);  squeeze_dims_263 = None
        unsqueeze_default_98: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_97, 2);  unsqueeze_default_97 = None
        unsqueeze_default_99: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_98, 3);  unsqueeze_default_98 = None
        unsqueeze_default_100: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_dims_260, 0);  squeeze_dims_260 = None
        unsqueeze_default_101: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_100, 2);  unsqueeze_default_100 = None
        unsqueeze_default_102: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_101, 3);  unsqueeze_default_101 = None
        unsqueeze_default_103: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_257, 0);  squeeze_dims_257 = None
        unsqueeze_default_104: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_103, 2);  unsqueeze_default_103 = None
        unsqueeze_default_105: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_104, 3);  unsqueeze_default_104 = None
        unsqueeze_default_106: "f32[1, 992]" = torch.ops.aten.unsqueeze.default(squeeze_dims_254, 0);  squeeze_dims_254 = None
        unsqueeze_default_107: "f32[1, 992, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_106, 2);  unsqueeze_default_106 = None
        unsqueeze_default_108: "f32[1, 992, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_107, 3);  unsqueeze_default_107 = None
        unsqueeze_default_109: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_251, 0);  squeeze_dims_251 = None
        unsqueeze_default_110: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_109, 2);  unsqueeze_default_109 = None
        unsqueeze_default_111: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_110, 3);  unsqueeze_default_110 = None
        unsqueeze_default_112: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(squeeze_dims_248, 0);  squeeze_dims_248 = None
        unsqueeze_default_113: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_112, 2);  unsqueeze_default_112 = None
        unsqueeze_default_114: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_113, 3);  unsqueeze_default_113 = None
        unsqueeze_default_115: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_245, 0);  squeeze_dims_245 = None
        unsqueeze_default_116: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_115, 2);  unsqueeze_default_115 = None
        unsqueeze_default_117: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_116, 3);  unsqueeze_default_116 = None
        unsqueeze_default_118: "f32[1, 928]" = torch.ops.aten.unsqueeze.default(squeeze_dims_242, 0);  squeeze_dims_242 = None
        unsqueeze_default_119: "f32[1, 928, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_118, 2);  unsqueeze_default_118 = None
        unsqueeze_default_120: "f32[1, 928, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_119, 3);  unsqueeze_default_119 = None
        unsqueeze_default_121: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_239, 0);  squeeze_dims_239 = None
        unsqueeze_default_122: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_121, 2);  unsqueeze_default_121 = None
        unsqueeze_default_123: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_122, 3);  unsqueeze_default_122 = None
        unsqueeze_default_124: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_dims_236, 0);  squeeze_dims_236 = None
        unsqueeze_default_125: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_124, 2);  unsqueeze_default_124 = None
        unsqueeze_default_126: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_125, 3);  unsqueeze_default_125 = None
        unsqueeze_default_127: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_233, 0);  squeeze_dims_233 = None
        unsqueeze_default_128: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_127, 2);  unsqueeze_default_127 = None
        unsqueeze_default_129: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_128, 3);  unsqueeze_default_128 = None
        unsqueeze_default_130: "f32[1, 864]" = torch.ops.aten.unsqueeze.default(squeeze_dims_230, 0);  squeeze_dims_230 = None
        unsqueeze_default_131: "f32[1, 864, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_130, 2);  unsqueeze_default_130 = None
        unsqueeze_default_132: "f32[1, 864, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_131, 3);  unsqueeze_default_131 = None
        unsqueeze_default_133: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_227, 0);  squeeze_dims_227 = None
        unsqueeze_default_134: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_133, 2);  unsqueeze_default_133 = None
        unsqueeze_default_135: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_134, 3);  unsqueeze_default_134 = None
        unsqueeze_default_136: "f32[1, 832]" = torch.ops.aten.unsqueeze.default(squeeze_dims_224, 0);  squeeze_dims_224 = None
        unsqueeze_default_137: "f32[1, 832, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_136, 2);  unsqueeze_default_136 = None
        unsqueeze_default_138: "f32[1, 832, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_137, 3);  unsqueeze_default_137 = None
        unsqueeze_default_139: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_221, 0);  squeeze_dims_221 = None
        unsqueeze_default_140: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_139, 2);  unsqueeze_default_139 = None
        unsqueeze_default_141: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_140, 3);  unsqueeze_default_140 = None
        unsqueeze_default_142: "f32[1, 800]" = torch.ops.aten.unsqueeze.default(squeeze_dims_218, 0);  squeeze_dims_218 = None
        unsqueeze_default_143: "f32[1, 800, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_142, 2);  unsqueeze_default_142 = None
        unsqueeze_default_144: "f32[1, 800, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_143, 3);  unsqueeze_default_143 = None
        unsqueeze_default_145: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_215, 0);  squeeze_dims_215 = None
        unsqueeze_default_146: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_145, 2);  unsqueeze_default_145 = None
        unsqueeze_default_147: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_146, 3);  unsqueeze_default_146 = None
        unsqueeze_default_148: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_dims_212, 0);  squeeze_dims_212 = None
        unsqueeze_default_149: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_148, 2);  unsqueeze_default_148 = None
        unsqueeze_default_150: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_149, 3);  unsqueeze_default_149 = None
        unsqueeze_default_151: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_209, 0);  squeeze_dims_209 = None
        unsqueeze_default_152: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_151, 2);  unsqueeze_default_151 = None
        unsqueeze_default_153: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_152, 3);  unsqueeze_default_152 = None
        unsqueeze_default_154: "f32[1, 736]" = torch.ops.aten.unsqueeze.default(squeeze_dims_206, 0);  squeeze_dims_206 = None
        unsqueeze_default_155: "f32[1, 736, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_154, 2);  unsqueeze_default_154 = None
        unsqueeze_default_156: "f32[1, 736, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_155, 3);  unsqueeze_default_155 = None
        unsqueeze_default_157: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_203, 0);  squeeze_dims_203 = None
        unsqueeze_default_158: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_157, 2);  unsqueeze_default_157 = None
        unsqueeze_default_159: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_158, 3);  unsqueeze_default_158 = None
        unsqueeze_default_160: "f32[1, 704]" = torch.ops.aten.unsqueeze.default(squeeze_dims_200, 0);  squeeze_dims_200 = None
        unsqueeze_default_161: "f32[1, 704, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_160, 2);  unsqueeze_default_160 = None
        unsqueeze_default_162: "f32[1, 704, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_161, 3);  unsqueeze_default_161 = None
        unsqueeze_default_163: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_197, 0);  squeeze_dims_197 = None
        unsqueeze_default_164: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_163, 2);  unsqueeze_default_163 = None
        unsqueeze_default_165: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_164, 3);  unsqueeze_default_164 = None
        unsqueeze_default_166: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_dims_194, 0);  squeeze_dims_194 = None
        unsqueeze_default_167: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_166, 2);  unsqueeze_default_166 = None
        unsqueeze_default_168: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_167, 3);  unsqueeze_default_167 = None
        unsqueeze_default_169: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_191, 0);  squeeze_dims_191 = None
        unsqueeze_default_170: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_169, 2);  unsqueeze_default_169 = None
        unsqueeze_default_171: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_170, 3);  unsqueeze_default_170 = None
        unsqueeze_default_172: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(squeeze_dims_188, 0);  squeeze_dims_188 = None
        unsqueeze_default_173: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_172, 2);  unsqueeze_default_172 = None
        unsqueeze_default_174: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_173, 3);  unsqueeze_default_173 = None
        unsqueeze_default_175: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_185, 0);  squeeze_dims_185 = None
        unsqueeze_default_176: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_175, 2);  unsqueeze_default_175 = None
        unsqueeze_default_177: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_176, 3);  unsqueeze_default_176 = None
        unsqueeze_default_178: "f32[1, 608]" = torch.ops.aten.unsqueeze.default(squeeze_dims_182, 0);  squeeze_dims_182 = None
        unsqueeze_default_179: "f32[1, 608, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_178, 2);  unsqueeze_default_178 = None
        unsqueeze_default_180: "f32[1, 608, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_179, 3);  unsqueeze_default_179 = None
        unsqueeze_default_181: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_179, 0);  squeeze_dims_179 = None
        unsqueeze_default_182: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_181, 2);  unsqueeze_default_181 = None
        unsqueeze_default_183: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_182, 3);  unsqueeze_default_182 = None
        unsqueeze_default_184: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(squeeze_dims_176, 0);  squeeze_dims_176 = None
        unsqueeze_default_185: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_184, 2);  unsqueeze_default_184 = None
        unsqueeze_default_186: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_185, 3);  unsqueeze_default_185 = None
        unsqueeze_default_187: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_173, 0);  squeeze_dims_173 = None
        unsqueeze_default_188: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_187, 2);  unsqueeze_default_187 = None
        unsqueeze_default_189: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_188, 3);  unsqueeze_default_188 = None
        unsqueeze_default_190: "f32[1, 544]" = torch.ops.aten.unsqueeze.default(squeeze_dims_170, 0);  squeeze_dims_170 = None
        unsqueeze_default_191: "f32[1, 544, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_190, 2);  unsqueeze_default_190 = None
        unsqueeze_default_192: "f32[1, 544, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_191, 3);  unsqueeze_default_191 = None
        unsqueeze_default_193: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_167, 0);  squeeze_dims_167 = None
        unsqueeze_default_194: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_193, 2);  unsqueeze_default_193 = None
        unsqueeze_default_195: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_194, 3);  unsqueeze_default_194 = None
        unsqueeze_default_196: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_164, 0);  squeeze_dims_164 = None
        unsqueeze_default_197: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_196, 2);  unsqueeze_default_196 = None
        unsqueeze_default_198: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_197, 3);  unsqueeze_default_197 = None
        unsqueeze_default_199: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_161, 0);  squeeze_dims_161 = None
        unsqueeze_default_200: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_199, 2);  unsqueeze_default_199 = None
        unsqueeze_default_201: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_200, 3);  unsqueeze_default_200 = None
        unsqueeze_default_202: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_158, 0);  squeeze_dims_158 = None
        unsqueeze_default_203: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_202, 2);  unsqueeze_default_202 = None
        unsqueeze_default_204: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_203, 3);  unsqueeze_default_203 = None
        unsqueeze_default_205: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_155, 0);  squeeze_dims_155 = None
        unsqueeze_default_206: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_205, 2);  unsqueeze_default_205 = None
        unsqueeze_default_207: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_206, 3);  unsqueeze_default_206 = None
        unsqueeze_default_208: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_dims_152, 0);  squeeze_dims_152 = None
        unsqueeze_default_209: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_208, 2);  unsqueeze_default_208 = None
        unsqueeze_default_210: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_209, 3);  unsqueeze_default_209 = None
        unsqueeze_default_211: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_149, 0);  squeeze_dims_149 = None
        unsqueeze_default_212: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_211, 2);  unsqueeze_default_211 = None
        unsqueeze_default_213: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_212, 3);  unsqueeze_default_212 = None
        unsqueeze_default_214: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_dims_146, 0);  squeeze_dims_146 = None
        unsqueeze_default_215: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_214, 2);  unsqueeze_default_214 = None
        unsqueeze_default_216: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_215, 3);  unsqueeze_default_215 = None
        unsqueeze_default_217: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_143, 0);  squeeze_dims_143 = None
        unsqueeze_default_218: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_217, 2);  unsqueeze_default_217 = None
        unsqueeze_default_219: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_218, 3);  unsqueeze_default_218 = None
        unsqueeze_default_220: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_140, 0);  squeeze_dims_140 = None
        unsqueeze_default_221: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_220, 2);  unsqueeze_default_220 = None
        unsqueeze_default_222: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_221, 3);  unsqueeze_default_221 = None
        unsqueeze_default_223: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_137, 0);  squeeze_dims_137 = None
        unsqueeze_default_224: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_223, 2);  unsqueeze_default_223 = None
        unsqueeze_default_225: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_224, 3);  unsqueeze_default_224 = None
        unsqueeze_default_226: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(squeeze_dims_134, 0);  squeeze_dims_134 = None
        unsqueeze_default_227: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_226, 2);  unsqueeze_default_226 = None
        unsqueeze_default_228: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_227, 3);  unsqueeze_default_227 = None
        unsqueeze_default_229: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_131, 0);  squeeze_dims_131 = None
        unsqueeze_default_230: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_229, 2);  unsqueeze_default_229 = None
        unsqueeze_default_231: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_230, 3);  unsqueeze_default_230 = None
        unsqueeze_default_232: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_dims_128, 0);  squeeze_dims_128 = None
        unsqueeze_default_233: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_232, 2);  unsqueeze_default_232 = None
        unsqueeze_default_234: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_233, 3);  unsqueeze_default_233 = None
        unsqueeze_default_235: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_125, 0);  squeeze_dims_125 = None
        unsqueeze_default_236: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_235, 2);  unsqueeze_default_235 = None
        unsqueeze_default_237: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_236, 3);  unsqueeze_default_236 = None
        unsqueeze_default_238: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(squeeze_dims_122, 0);  squeeze_dims_122 = None
        unsqueeze_default_239: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_238, 2);  unsqueeze_default_238 = None
        unsqueeze_default_240: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_239, 3);  unsqueeze_default_239 = None
        unsqueeze_default_241: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_119, 0);  squeeze_dims_119 = None
        unsqueeze_default_242: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_241, 2);  unsqueeze_default_241 = None
        unsqueeze_default_243: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_242, 3);  unsqueeze_default_242 = None
        unsqueeze_default_244: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_116, 0);  squeeze_dims_116 = None
        unsqueeze_default_245: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_244, 2);  unsqueeze_default_244 = None
        unsqueeze_default_246: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_245, 3);  unsqueeze_default_245 = None
        unsqueeze_default_247: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_dims_113, 0);  squeeze_dims_113 = None
        unsqueeze_default_248: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_247, 2);  unsqueeze_default_247 = None
        unsqueeze_default_249: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_248, 3);  unsqueeze_default_248 = None
        unsqueeze_default_250: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_110, 0);  squeeze_dims_110 = None
        unsqueeze_default_251: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_250, 2);  unsqueeze_default_250 = None
        unsqueeze_default_252: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_251, 3);  unsqueeze_default_251 = None
        unsqueeze_default_253: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_107, 0);  squeeze_dims_107 = None
        unsqueeze_default_254: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_253, 2);  unsqueeze_default_253 = None
        unsqueeze_default_255: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_254, 3);  unsqueeze_default_254 = None
        unsqueeze_default_256: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_104, 0);  squeeze_dims_104 = None
        unsqueeze_default_257: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_256, 2);  unsqueeze_default_256 = None
        unsqueeze_default_258: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_257, 3);  unsqueeze_default_257 = None
        unsqueeze_default_259: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_dims_101, 0);  squeeze_dims_101 = None
        unsqueeze_default_260: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_259, 2);  unsqueeze_default_259 = None
        unsqueeze_default_261: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_260, 3);  unsqueeze_default_260 = None
        unsqueeze_default_262: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_98, 0);  squeeze_dims_98 = None
        unsqueeze_default_263: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_262, 2);  unsqueeze_default_262 = None
        unsqueeze_default_264: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_263, 3);  unsqueeze_default_263 = None
        unsqueeze_default_265: "f32[1, 416]" = torch.ops.aten.unsqueeze.default(squeeze_dims_95, 0);  squeeze_dims_95 = None
        unsqueeze_default_266: "f32[1, 416, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_265, 2);  unsqueeze_default_265 = None
        unsqueeze_default_267: "f32[1, 416, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_266, 3);  unsqueeze_default_266 = None
        unsqueeze_default_268: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_92, 0);  squeeze_dims_92 = None
        unsqueeze_default_269: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_268, 2);  unsqueeze_default_268 = None
        unsqueeze_default_270: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_269, 3);  unsqueeze_default_269 = None
        unsqueeze_default_271: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(squeeze_dims_89, 0);  squeeze_dims_89 = None
        unsqueeze_default_272: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_271, 2);  unsqueeze_default_271 = None
        unsqueeze_default_273: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_272, 3);  unsqueeze_default_272 = None
        unsqueeze_default_274: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_86, 0);  squeeze_dims_86 = None
        unsqueeze_default_275: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_274, 2);  unsqueeze_default_274 = None
        unsqueeze_default_276: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_275, 3);  unsqueeze_default_275 = None
        unsqueeze_default_277: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(squeeze_dims_83, 0);  squeeze_dims_83 = None
        unsqueeze_default_278: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_277, 2);  unsqueeze_default_277 = None
        unsqueeze_default_279: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_278, 3);  unsqueeze_default_278 = None
        unsqueeze_default_280: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_80, 0);  squeeze_dims_80 = None
        unsqueeze_default_281: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_280, 2);  unsqueeze_default_280 = None
        unsqueeze_default_282: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_281, 3);  unsqueeze_default_281 = None
        unsqueeze_default_283: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(squeeze_dims_77, 0);  squeeze_dims_77 = None
        unsqueeze_default_284: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_283, 2);  unsqueeze_default_283 = None
        unsqueeze_default_285: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_284, 3);  unsqueeze_default_284 = None
        unsqueeze_default_286: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_74, 0);  squeeze_dims_74 = None
        unsqueeze_default_287: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_286, 2);  unsqueeze_default_286 = None
        unsqueeze_default_288: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_287, 3);  unsqueeze_default_287 = None
        unsqueeze_default_289: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(squeeze_dims_71, 0);  squeeze_dims_71 = None
        unsqueeze_default_290: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_289, 2);  unsqueeze_default_289 = None
        unsqueeze_default_291: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_290, 3);  unsqueeze_default_290 = None
        unsqueeze_default_292: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_68, 0);  squeeze_dims_68 = None
        unsqueeze_default_293: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_292, 2);  unsqueeze_default_292 = None
        unsqueeze_default_294: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_293, 3);  unsqueeze_default_293 = None
        unsqueeze_default_295: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_65, 0);  squeeze_dims_65 = None
        unsqueeze_default_296: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_295, 2);  unsqueeze_default_295 = None
        unsqueeze_default_297: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_296, 3);  unsqueeze_default_296 = None
        unsqueeze_default_298: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_62, 0);  squeeze_dims_62 = None
        unsqueeze_default_299: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_298, 2);  unsqueeze_default_298 = None
        unsqueeze_default_300: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_299, 3);  unsqueeze_default_299 = None
        unsqueeze_default_301: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_dims_59, 0);  squeeze_dims_59 = None
        unsqueeze_default_302: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_301, 2);  unsqueeze_default_301 = None
        unsqueeze_default_303: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_302, 3);  unsqueeze_default_302 = None
        unsqueeze_default_304: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_56, 0);  squeeze_dims_56 = None
        unsqueeze_default_305: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_304, 2);  unsqueeze_default_304 = None
        unsqueeze_default_306: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_305, 3);  unsqueeze_default_305 = None
        unsqueeze_default_307: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_53, 0);  squeeze_dims_53 = None
        unsqueeze_default_308: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_307, 2);  unsqueeze_default_307 = None
        unsqueeze_default_309: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_308, 3);  unsqueeze_default_308 = None
        unsqueeze_default_310: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_50, 0);  squeeze_dims_50 = None
        unsqueeze_default_311: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_310, 2);  unsqueeze_default_310 = None
        unsqueeze_default_312: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_311, 3);  unsqueeze_default_311 = None
        unsqueeze_default_313: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_dims_47, 0);  squeeze_dims_47 = None
        unsqueeze_default_314: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_313, 2);  unsqueeze_default_313 = None
        unsqueeze_default_315: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_314, 3);  unsqueeze_default_314 = None
        unsqueeze_default_316: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_44, 0);  squeeze_dims_44 = None
        unsqueeze_default_317: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_316, 2);  unsqueeze_default_316 = None
        unsqueeze_default_318: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_317, 3);  unsqueeze_default_317 = None
        unsqueeze_default_319: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_41, 0);  squeeze_dims_41 = None
        unsqueeze_default_320: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_319, 2);  unsqueeze_default_319 = None
        unsqueeze_default_321: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_320, 3);  unsqueeze_default_320 = None
        unsqueeze_default_322: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_dims_38, 0);  squeeze_dims_38 = None
        unsqueeze_default_323: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_322, 2);  unsqueeze_default_322 = None
        unsqueeze_default_324: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_323, 3);  unsqueeze_default_323 = None
        unsqueeze_default_325: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_35, 0);  squeeze_dims_35 = None
        unsqueeze_default_326: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_325, 2);  unsqueeze_default_325 = None
        unsqueeze_default_327: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_326, 3);  unsqueeze_default_326 = None
        unsqueeze_default_328: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_dims_32, 0);  squeeze_dims_32 = None
        unsqueeze_default_329: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_328, 2);  unsqueeze_default_328 = None
        unsqueeze_default_330: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_329, 3);  unsqueeze_default_329 = None
        unsqueeze_default_331: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_29, 0);  squeeze_dims_29 = None
        unsqueeze_default_332: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_331, 2);  unsqueeze_default_331 = None
        unsqueeze_default_333: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_332, 3);  unsqueeze_default_332 = None
        unsqueeze_default_334: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_dims_26, 0);  squeeze_dims_26 = None
        unsqueeze_default_335: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_334, 2);  unsqueeze_default_334 = None
        unsqueeze_default_336: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_335, 3);  unsqueeze_default_335 = None
        unsqueeze_default_337: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_23, 0);  squeeze_dims_23 = None
        unsqueeze_default_338: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_337, 2);  unsqueeze_default_337 = None
        unsqueeze_default_339: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_338, 3);  unsqueeze_default_338 = None
        unsqueeze_default_340: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_dims_20, 0);  squeeze_dims_20 = None
        unsqueeze_default_341: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_340, 2);  unsqueeze_default_340 = None
        unsqueeze_default_342: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_341, 3);  unsqueeze_default_341 = None
        unsqueeze_default_343: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_17, 0);  squeeze_dims_17 = None
        unsqueeze_default_344: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_343, 2);  unsqueeze_default_343 = None
        unsqueeze_default_345: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_344, 3);  unsqueeze_default_344 = None
        unsqueeze_default_346: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_14, 0);  squeeze_dims_14 = None
        unsqueeze_default_347: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_346, 2);  unsqueeze_default_346 = None
        unsqueeze_default_348: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_347, 3);  unsqueeze_default_347 = None
        unsqueeze_default_349: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_11, 0);  squeeze_dims_11 = None
        unsqueeze_default_350: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_349, 2);  unsqueeze_default_349 = None
        unsqueeze_default_351: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_350, 3);  unsqueeze_default_350 = None
        unsqueeze_default_352: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_dims_8, 0);  squeeze_dims_8 = None
        unsqueeze_default_353: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_352, 2);  unsqueeze_default_352 = None
        unsqueeze_default_354: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_353, 3);  unsqueeze_default_353 = None
        unsqueeze_default_355: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_dims_5, 0);  squeeze_dims_5 = None
        unsqueeze_default_356: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_355, 2);  unsqueeze_default_355 = None
        unsqueeze_default_357: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_356, 3);  unsqueeze_default_356 = None
        unsqueeze_default_358: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_dims_2, 0);  squeeze_dims_2 = None
        unsqueeze_default_359: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_358, 2);  unsqueeze_default_358 = None
        unsqueeze_default_360: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_359, 3);  unsqueeze_default_359 = None
        copy__default: "i64[]" = torch.ops.aten.copy_.default(arg3_1, add_tensor);  arg3_1 = add_tensor = None
        copy__default_1: "f32[64]" = torch.ops.aten.copy_.default(arg4_1, add_tensor_1);  arg4_1 = add_tensor_1 = None
        copy__default_2: "f32[64]" = torch.ops.aten.copy_.default(arg5_1, add_tensor_2);  arg5_1 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(arg8_1, add_tensor_3);  arg8_1 = add_tensor_3 = None
        copy__default_4: "f32[64]" = torch.ops.aten.copy_.default(arg9_1, add_tensor_4);  arg9_1 = add_tensor_4 = None
        copy__default_5: "f32[64]" = torch.ops.aten.copy_.default(arg10_1, add_tensor_5);  arg10_1 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(arg14_1, add_tensor_6);  arg14_1 = add_tensor_6 = None
        copy__default_7: "f32[128]" = torch.ops.aten.copy_.default(arg15_1, add_tensor_7);  arg15_1 = add_tensor_7 = None
        copy__default_8: "f32[128]" = torch.ops.aten.copy_.default(arg16_1, add_tensor_8);  arg16_1 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(arg20_1, add_tensor_9);  arg20_1 = add_tensor_9 = None
        copy__default_10: "f32[96]" = torch.ops.aten.copy_.default(arg21_1, add_tensor_10);  arg21_1 = add_tensor_10 = None
        copy__default_11: "f32[96]" = torch.ops.aten.copy_.default(arg22_1, add_tensor_11);  arg22_1 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(arg26_1, add_tensor_12);  arg26_1 = add_tensor_12 = None
        copy__default_13: "f32[128]" = torch.ops.aten.copy_.default(arg27_1, add_tensor_13);  arg27_1 = add_tensor_13 = None
        copy__default_14: "f32[128]" = torch.ops.aten.copy_.default(arg28_1, add_tensor_14);  arg28_1 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(arg32_1, add_tensor_15);  arg32_1 = add_tensor_15 = None
        copy__default_16: "f32[128]" = torch.ops.aten.copy_.default(arg33_1, add_tensor_16);  arg33_1 = add_tensor_16 = None
        copy__default_17: "f32[128]" = torch.ops.aten.copy_.default(arg34_1, add_tensor_17);  arg34_1 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(arg38_1, add_tensor_18);  arg38_1 = add_tensor_18 = None
        copy__default_19: "f32[128]" = torch.ops.aten.copy_.default(arg39_1, add_tensor_19);  arg39_1 = add_tensor_19 = None
        copy__default_20: "f32[128]" = torch.ops.aten.copy_.default(arg40_1, add_tensor_20);  arg40_1 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(arg44_1, add_tensor_21);  arg44_1 = add_tensor_21 = None
        copy__default_22: "f32[160]" = torch.ops.aten.copy_.default(arg45_1, add_tensor_22);  arg45_1 = add_tensor_22 = None
        copy__default_23: "f32[160]" = torch.ops.aten.copy_.default(arg46_1, add_tensor_23);  arg46_1 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(arg50_1, add_tensor_24);  arg50_1 = add_tensor_24 = None
        copy__default_25: "f32[128]" = torch.ops.aten.copy_.default(arg51_1, add_tensor_25);  arg51_1 = add_tensor_25 = None
        copy__default_26: "f32[128]" = torch.ops.aten.copy_.default(arg52_1, add_tensor_26);  arg52_1 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(arg56_1, add_tensor_27);  arg56_1 = add_tensor_27 = None
        copy__default_28: "f32[192]" = torch.ops.aten.copy_.default(arg57_1, add_tensor_28);  arg57_1 = add_tensor_28 = None
        copy__default_29: "f32[192]" = torch.ops.aten.copy_.default(arg58_1, add_tensor_29);  arg58_1 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(arg62_1, add_tensor_30);  arg62_1 = add_tensor_30 = None
        copy__default_31: "f32[128]" = torch.ops.aten.copy_.default(arg63_1, add_tensor_31);  arg63_1 = add_tensor_31 = None
        copy__default_32: "f32[128]" = torch.ops.aten.copy_.default(arg64_1, add_tensor_32);  arg64_1 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(arg68_1, add_tensor_33);  arg68_1 = add_tensor_33 = None
        copy__default_34: "f32[224]" = torch.ops.aten.copy_.default(arg69_1, add_tensor_34);  arg69_1 = add_tensor_34 = None
        copy__default_35: "f32[224]" = torch.ops.aten.copy_.default(arg70_1, add_tensor_35);  arg70_1 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(arg74_1, add_tensor_36);  arg74_1 = add_tensor_36 = None
        copy__default_37: "f32[128]" = torch.ops.aten.copy_.default(arg75_1, add_tensor_37);  arg75_1 = add_tensor_37 = None
        copy__default_38: "f32[128]" = torch.ops.aten.copy_.default(arg76_1, add_tensor_38);  arg76_1 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(arg80_1, add_tensor_39);  arg80_1 = add_tensor_39 = None
        copy__default_40: "f32[256]" = torch.ops.aten.copy_.default(arg81_1, add_tensor_40);  arg81_1 = add_tensor_40 = None
        copy__default_41: "f32[256]" = torch.ops.aten.copy_.default(arg82_1, add_tensor_41);  arg82_1 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(arg86_1, add_tensor_42);  arg86_1 = add_tensor_42 = None
        copy__default_43: "f32[128]" = torch.ops.aten.copy_.default(arg87_1, add_tensor_43);  arg87_1 = add_tensor_43 = None
        copy__default_44: "f32[128]" = torch.ops.aten.copy_.default(arg88_1, add_tensor_44);  arg88_1 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(arg92_1, add_tensor_45);  arg92_1 = add_tensor_45 = None
        copy__default_46: "f32[128]" = torch.ops.aten.copy_.default(arg93_1, add_tensor_46);  arg93_1 = add_tensor_46 = None
        copy__default_47: "f32[128]" = torch.ops.aten.copy_.default(arg94_1, add_tensor_47);  arg94_1 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(arg98_1, add_tensor_48);  arg98_1 = add_tensor_48 = None
        copy__default_49: "f32[160]" = torch.ops.aten.copy_.default(arg99_1, add_tensor_49);  arg99_1 = add_tensor_49 = None
        copy__default_50: "f32[160]" = torch.ops.aten.copy_.default(arg100_1, add_tensor_50);  arg100_1 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(arg104_1, add_tensor_51);  arg104_1 = add_tensor_51 = None
        copy__default_52: "f32[128]" = torch.ops.aten.copy_.default(arg105_1, add_tensor_52);  arg105_1 = add_tensor_52 = None
        copy__default_53: "f32[128]" = torch.ops.aten.copy_.default(arg106_1, add_tensor_53);  arg106_1 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(arg110_1, add_tensor_54);  arg110_1 = add_tensor_54 = None
        copy__default_55: "f32[192]" = torch.ops.aten.copy_.default(arg111_1, add_tensor_55);  arg111_1 = add_tensor_55 = None
        copy__default_56: "f32[192]" = torch.ops.aten.copy_.default(arg112_1, add_tensor_56);  arg112_1 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(arg116_1, add_tensor_57);  arg116_1 = add_tensor_57 = None
        copy__default_58: "f32[128]" = torch.ops.aten.copy_.default(arg117_1, add_tensor_58);  arg117_1 = add_tensor_58 = None
        copy__default_59: "f32[128]" = torch.ops.aten.copy_.default(arg118_1, add_tensor_59);  arg118_1 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(arg122_1, add_tensor_60);  arg122_1 = add_tensor_60 = None
        copy__default_61: "f32[224]" = torch.ops.aten.copy_.default(arg123_1, add_tensor_61);  arg123_1 = add_tensor_61 = None
        copy__default_62: "f32[224]" = torch.ops.aten.copy_.default(arg124_1, add_tensor_62);  arg124_1 = add_tensor_62 = None
        copy__default_63: "i64[]" = torch.ops.aten.copy_.default(arg128_1, add_tensor_63);  arg128_1 = add_tensor_63 = None
        copy__default_64: "f32[128]" = torch.ops.aten.copy_.default(arg129_1, add_tensor_64);  arg129_1 = add_tensor_64 = None
        copy__default_65: "f32[128]" = torch.ops.aten.copy_.default(arg130_1, add_tensor_65);  arg130_1 = add_tensor_65 = None
        copy__default_66: "i64[]" = torch.ops.aten.copy_.default(arg134_1, add_tensor_66);  arg134_1 = add_tensor_66 = None
        copy__default_67: "f32[256]" = torch.ops.aten.copy_.default(arg135_1, add_tensor_67);  arg135_1 = add_tensor_67 = None
        copy__default_68: "f32[256]" = torch.ops.aten.copy_.default(arg136_1, add_tensor_68);  arg136_1 = add_tensor_68 = None
        copy__default_69: "i64[]" = torch.ops.aten.copy_.default(arg140_1, add_tensor_69);  arg140_1 = add_tensor_69 = None
        copy__default_70: "f32[128]" = torch.ops.aten.copy_.default(arg141_1, add_tensor_70);  arg141_1 = add_tensor_70 = None
        copy__default_71: "f32[128]" = torch.ops.aten.copy_.default(arg142_1, add_tensor_71);  arg142_1 = add_tensor_71 = None
        copy__default_72: "i64[]" = torch.ops.aten.copy_.default(arg146_1, add_tensor_72);  arg146_1 = add_tensor_72 = None
        copy__default_73: "f32[288]" = torch.ops.aten.copy_.default(arg147_1, add_tensor_73);  arg147_1 = add_tensor_73 = None
        copy__default_74: "f32[288]" = torch.ops.aten.copy_.default(arg148_1, add_tensor_74);  arg148_1 = add_tensor_74 = None
        copy__default_75: "i64[]" = torch.ops.aten.copy_.default(arg152_1, add_tensor_75);  arg152_1 = add_tensor_75 = None
        copy__default_76: "f32[128]" = torch.ops.aten.copy_.default(arg153_1, add_tensor_76);  arg153_1 = add_tensor_76 = None
        copy__default_77: "f32[128]" = torch.ops.aten.copy_.default(arg154_1, add_tensor_77);  arg154_1 = add_tensor_77 = None
        copy__default_78: "i64[]" = torch.ops.aten.copy_.default(arg158_1, add_tensor_78);  arg158_1 = add_tensor_78 = None
        copy__default_79: "f32[320]" = torch.ops.aten.copy_.default(arg159_1, add_tensor_79);  arg159_1 = add_tensor_79 = None
        copy__default_80: "f32[320]" = torch.ops.aten.copy_.default(arg160_1, add_tensor_80);  arg160_1 = add_tensor_80 = None
        copy__default_81: "i64[]" = torch.ops.aten.copy_.default(arg164_1, add_tensor_81);  arg164_1 = add_tensor_81 = None
        copy__default_82: "f32[128]" = torch.ops.aten.copy_.default(arg165_1, add_tensor_82);  arg165_1 = add_tensor_82 = None
        copy__default_83: "f32[128]" = torch.ops.aten.copy_.default(arg166_1, add_tensor_83);  arg166_1 = add_tensor_83 = None
        copy__default_84: "i64[]" = torch.ops.aten.copy_.default(arg170_1, add_tensor_84);  arg170_1 = add_tensor_84 = None
        copy__default_85: "f32[352]" = torch.ops.aten.copy_.default(arg171_1, add_tensor_85);  arg171_1 = add_tensor_85 = None
        copy__default_86: "f32[352]" = torch.ops.aten.copy_.default(arg172_1, add_tensor_86);  arg172_1 = add_tensor_86 = None
        copy__default_87: "i64[]" = torch.ops.aten.copy_.default(arg176_1, add_tensor_87);  arg176_1 = add_tensor_87 = None
        copy__default_88: "f32[128]" = torch.ops.aten.copy_.default(arg177_1, add_tensor_88);  arg177_1 = add_tensor_88 = None
        copy__default_89: "f32[128]" = torch.ops.aten.copy_.default(arg178_1, add_tensor_89);  arg178_1 = add_tensor_89 = None
        copy__default_90: "i64[]" = torch.ops.aten.copy_.default(arg182_1, add_tensor_90);  arg182_1 = add_tensor_90 = None
        copy__default_91: "f32[384]" = torch.ops.aten.copy_.default(arg183_1, add_tensor_91);  arg183_1 = add_tensor_91 = None
        copy__default_92: "f32[384]" = torch.ops.aten.copy_.default(arg184_1, add_tensor_92);  arg184_1 = add_tensor_92 = None
        copy__default_93: "i64[]" = torch.ops.aten.copy_.default(arg188_1, add_tensor_93);  arg188_1 = add_tensor_93 = None
        copy__default_94: "f32[128]" = torch.ops.aten.copy_.default(arg189_1, add_tensor_94);  arg189_1 = add_tensor_94 = None
        copy__default_95: "f32[128]" = torch.ops.aten.copy_.default(arg190_1, add_tensor_95);  arg190_1 = add_tensor_95 = None
        copy__default_96: "i64[]" = torch.ops.aten.copy_.default(arg194_1, add_tensor_96);  arg194_1 = add_tensor_96 = None
        copy__default_97: "f32[416]" = torch.ops.aten.copy_.default(arg195_1, add_tensor_97);  arg195_1 = add_tensor_97 = None
        copy__default_98: "f32[416]" = torch.ops.aten.copy_.default(arg196_1, add_tensor_98);  arg196_1 = add_tensor_98 = None
        copy__default_99: "i64[]" = torch.ops.aten.copy_.default(arg200_1, add_tensor_99);  arg200_1 = add_tensor_99 = None
        copy__default_100: "f32[128]" = torch.ops.aten.copy_.default(arg201_1, add_tensor_100);  arg201_1 = add_tensor_100 = None
        copy__default_101: "f32[128]" = torch.ops.aten.copy_.default(arg202_1, add_tensor_101);  arg202_1 = add_tensor_101 = None
        copy__default_102: "i64[]" = torch.ops.aten.copy_.default(arg206_1, add_tensor_102);  arg206_1 = add_tensor_102 = None
        copy__default_103: "f32[448]" = torch.ops.aten.copy_.default(arg207_1, add_tensor_103);  arg207_1 = add_tensor_103 = None
        copy__default_104: "f32[448]" = torch.ops.aten.copy_.default(arg208_1, add_tensor_104);  arg208_1 = add_tensor_104 = None
        copy__default_105: "i64[]" = torch.ops.aten.copy_.default(arg212_1, add_tensor_105);  arg212_1 = add_tensor_105 = None
        copy__default_106: "f32[128]" = torch.ops.aten.copy_.default(arg213_1, add_tensor_106);  arg213_1 = add_tensor_106 = None
        copy__default_107: "f32[128]" = torch.ops.aten.copy_.default(arg214_1, add_tensor_107);  arg214_1 = add_tensor_107 = None
        copy__default_108: "i64[]" = torch.ops.aten.copy_.default(arg218_1, add_tensor_108);  arg218_1 = add_tensor_108 = None
        copy__default_109: "f32[480]" = torch.ops.aten.copy_.default(arg219_1, add_tensor_109);  arg219_1 = add_tensor_109 = None
        copy__default_110: "f32[480]" = torch.ops.aten.copy_.default(arg220_1, add_tensor_110);  arg220_1 = add_tensor_110 = None
        copy__default_111: "i64[]" = torch.ops.aten.copy_.default(arg224_1, add_tensor_111);  arg224_1 = add_tensor_111 = None
        copy__default_112: "f32[128]" = torch.ops.aten.copy_.default(arg225_1, add_tensor_112);  arg225_1 = add_tensor_112 = None
        copy__default_113: "f32[128]" = torch.ops.aten.copy_.default(arg226_1, add_tensor_113);  arg226_1 = add_tensor_113 = None
        copy__default_114: "i64[]" = torch.ops.aten.copy_.default(arg230_1, add_tensor_114);  arg230_1 = add_tensor_114 = None
        copy__default_115: "f32[512]" = torch.ops.aten.copy_.default(arg231_1, add_tensor_115);  arg231_1 = add_tensor_115 = None
        copy__default_116: "f32[512]" = torch.ops.aten.copy_.default(arg232_1, add_tensor_116);  arg232_1 = add_tensor_116 = None
        copy__default_117: "i64[]" = torch.ops.aten.copy_.default(arg236_1, add_tensor_117);  arg236_1 = add_tensor_117 = None
        copy__default_118: "f32[256]" = torch.ops.aten.copy_.default(arg237_1, add_tensor_118);  arg237_1 = add_tensor_118 = None
        copy__default_119: "f32[256]" = torch.ops.aten.copy_.default(arg238_1, add_tensor_119);  arg238_1 = add_tensor_119 = None
        copy__default_120: "i64[]" = torch.ops.aten.copy_.default(arg242_1, add_tensor_120);  arg242_1 = add_tensor_120 = None
        copy__default_121: "f32[128]" = torch.ops.aten.copy_.default(arg243_1, add_tensor_121);  arg243_1 = add_tensor_121 = None
        copy__default_122: "f32[128]" = torch.ops.aten.copy_.default(arg244_1, add_tensor_122);  arg244_1 = add_tensor_122 = None
        copy__default_123: "i64[]" = torch.ops.aten.copy_.default(arg248_1, add_tensor_123);  arg248_1 = add_tensor_123 = None
        copy__default_124: "f32[288]" = torch.ops.aten.copy_.default(arg249_1, add_tensor_124);  arg249_1 = add_tensor_124 = None
        copy__default_125: "f32[288]" = torch.ops.aten.copy_.default(arg250_1, add_tensor_125);  arg250_1 = add_tensor_125 = None
        copy__default_126: "i64[]" = torch.ops.aten.copy_.default(arg254_1, add_tensor_126);  arg254_1 = add_tensor_126 = None
        copy__default_127: "f32[128]" = torch.ops.aten.copy_.default(arg255_1, add_tensor_127);  arg255_1 = add_tensor_127 = None
        copy__default_128: "f32[128]" = torch.ops.aten.copy_.default(arg256_1, add_tensor_128);  arg256_1 = add_tensor_128 = None
        copy__default_129: "i64[]" = torch.ops.aten.copy_.default(arg260_1, add_tensor_129);  arg260_1 = add_tensor_129 = None
        copy__default_130: "f32[320]" = torch.ops.aten.copy_.default(arg261_1, add_tensor_130);  arg261_1 = add_tensor_130 = None
        copy__default_131: "f32[320]" = torch.ops.aten.copy_.default(arg262_1, add_tensor_131);  arg262_1 = add_tensor_131 = None
        copy__default_132: "i64[]" = torch.ops.aten.copy_.default(arg266_1, add_tensor_132);  arg266_1 = add_tensor_132 = None
        copy__default_133: "f32[128]" = torch.ops.aten.copy_.default(arg267_1, add_tensor_133);  arg267_1 = add_tensor_133 = None
        copy__default_134: "f32[128]" = torch.ops.aten.copy_.default(arg268_1, add_tensor_134);  arg268_1 = add_tensor_134 = None
        copy__default_135: "i64[]" = torch.ops.aten.copy_.default(arg272_1, add_tensor_135);  arg272_1 = add_tensor_135 = None
        copy__default_136: "f32[352]" = torch.ops.aten.copy_.default(arg273_1, add_tensor_136);  arg273_1 = add_tensor_136 = None
        copy__default_137: "f32[352]" = torch.ops.aten.copy_.default(arg274_1, add_tensor_137);  arg274_1 = add_tensor_137 = None
        copy__default_138: "i64[]" = torch.ops.aten.copy_.default(arg278_1, add_tensor_138);  arg278_1 = add_tensor_138 = None
        copy__default_139: "f32[128]" = torch.ops.aten.copy_.default(arg279_1, add_tensor_139);  arg279_1 = add_tensor_139 = None
        copy__default_140: "f32[128]" = torch.ops.aten.copy_.default(arg280_1, add_tensor_140);  arg280_1 = add_tensor_140 = None
        copy__default_141: "i64[]" = torch.ops.aten.copy_.default(arg284_1, add_tensor_141);  arg284_1 = add_tensor_141 = None
        copy__default_142: "f32[384]" = torch.ops.aten.copy_.default(arg285_1, add_tensor_142);  arg285_1 = add_tensor_142 = None
        copy__default_143: "f32[384]" = torch.ops.aten.copy_.default(arg286_1, add_tensor_143);  arg286_1 = add_tensor_143 = None
        copy__default_144: "i64[]" = torch.ops.aten.copy_.default(arg290_1, add_tensor_144);  arg290_1 = add_tensor_144 = None
        copy__default_145: "f32[128]" = torch.ops.aten.copy_.default(arg291_1, add_tensor_145);  arg291_1 = add_tensor_145 = None
        copy__default_146: "f32[128]" = torch.ops.aten.copy_.default(arg292_1, add_tensor_146);  arg292_1 = add_tensor_146 = None
        copy__default_147: "i64[]" = torch.ops.aten.copy_.default(arg296_1, add_tensor_147);  arg296_1 = add_tensor_147 = None
        copy__default_148: "f32[416]" = torch.ops.aten.copy_.default(arg297_1, add_tensor_148);  arg297_1 = add_tensor_148 = None
        copy__default_149: "f32[416]" = torch.ops.aten.copy_.default(arg298_1, add_tensor_149);  arg298_1 = add_tensor_149 = None
        copy__default_150: "i64[]" = torch.ops.aten.copy_.default(arg302_1, add_tensor_150);  arg302_1 = add_tensor_150 = None
        copy__default_151: "f32[128]" = torch.ops.aten.copy_.default(arg303_1, add_tensor_151);  arg303_1 = add_tensor_151 = None
        copy__default_152: "f32[128]" = torch.ops.aten.copy_.default(arg304_1, add_tensor_152);  arg304_1 = add_tensor_152 = None
        copy__default_153: "i64[]" = torch.ops.aten.copy_.default(arg308_1, add_tensor_153);  arg308_1 = add_tensor_153 = None
        copy__default_154: "f32[448]" = torch.ops.aten.copy_.default(arg309_1, add_tensor_154);  arg309_1 = add_tensor_154 = None
        copy__default_155: "f32[448]" = torch.ops.aten.copy_.default(arg310_1, add_tensor_155);  arg310_1 = add_tensor_155 = None
        copy__default_156: "i64[]" = torch.ops.aten.copy_.default(arg314_1, add_tensor_156);  arg314_1 = add_tensor_156 = None
        copy__default_157: "f32[128]" = torch.ops.aten.copy_.default(arg315_1, add_tensor_157);  arg315_1 = add_tensor_157 = None
        copy__default_158: "f32[128]" = torch.ops.aten.copy_.default(arg316_1, add_tensor_158);  arg316_1 = add_tensor_158 = None
        copy__default_159: "i64[]" = torch.ops.aten.copy_.default(arg320_1, add_tensor_159);  arg320_1 = add_tensor_159 = None
        copy__default_160: "f32[480]" = torch.ops.aten.copy_.default(arg321_1, add_tensor_160);  arg321_1 = add_tensor_160 = None
        copy__default_161: "f32[480]" = torch.ops.aten.copy_.default(arg322_1, add_tensor_161);  arg322_1 = add_tensor_161 = None
        copy__default_162: "i64[]" = torch.ops.aten.copy_.default(arg326_1, add_tensor_162);  arg326_1 = add_tensor_162 = None
        copy__default_163: "f32[128]" = torch.ops.aten.copy_.default(arg327_1, add_tensor_163);  arg327_1 = add_tensor_163 = None
        copy__default_164: "f32[128]" = torch.ops.aten.copy_.default(arg328_1, add_tensor_164);  arg328_1 = add_tensor_164 = None
        copy__default_165: "i64[]" = torch.ops.aten.copy_.default(arg332_1, add_tensor_165);  arg332_1 = add_tensor_165 = None
        copy__default_166: "f32[512]" = torch.ops.aten.copy_.default(arg333_1, add_tensor_166);  arg333_1 = add_tensor_166 = None
        copy__default_167: "f32[512]" = torch.ops.aten.copy_.default(arg334_1, add_tensor_167);  arg334_1 = add_tensor_167 = None
        copy__default_168: "i64[]" = torch.ops.aten.copy_.default(arg338_1, add_tensor_168);  arg338_1 = add_tensor_168 = None
        copy__default_169: "f32[128]" = torch.ops.aten.copy_.default(arg339_1, add_tensor_169);  arg339_1 = add_tensor_169 = None
        copy__default_170: "f32[128]" = torch.ops.aten.copy_.default(arg340_1, add_tensor_170);  arg340_1 = add_tensor_170 = None
        copy__default_171: "i64[]" = torch.ops.aten.copy_.default(arg344_1, add_tensor_171);  arg344_1 = add_tensor_171 = None
        copy__default_172: "f32[544]" = torch.ops.aten.copy_.default(arg345_1, add_tensor_172);  arg345_1 = add_tensor_172 = None
        copy__default_173: "f32[544]" = torch.ops.aten.copy_.default(arg346_1, add_tensor_173);  arg346_1 = add_tensor_173 = None
        copy__default_174: "i64[]" = torch.ops.aten.copy_.default(arg350_1, add_tensor_174);  arg350_1 = add_tensor_174 = None
        copy__default_175: "f32[128]" = torch.ops.aten.copy_.default(arg351_1, add_tensor_175);  arg351_1 = add_tensor_175 = None
        copy__default_176: "f32[128]" = torch.ops.aten.copy_.default(arg352_1, add_tensor_176);  arg352_1 = add_tensor_176 = None
        copy__default_177: "i64[]" = torch.ops.aten.copy_.default(arg356_1, add_tensor_177);  arg356_1 = add_tensor_177 = None
        copy__default_178: "f32[576]" = torch.ops.aten.copy_.default(arg357_1, add_tensor_178);  arg357_1 = add_tensor_178 = None
        copy__default_179: "f32[576]" = torch.ops.aten.copy_.default(arg358_1, add_tensor_179);  arg358_1 = add_tensor_179 = None
        copy__default_180: "i64[]" = torch.ops.aten.copy_.default(arg362_1, add_tensor_180);  arg362_1 = add_tensor_180 = None
        copy__default_181: "f32[128]" = torch.ops.aten.copy_.default(arg363_1, add_tensor_181);  arg363_1 = add_tensor_181 = None
        copy__default_182: "f32[128]" = torch.ops.aten.copy_.default(arg364_1, add_tensor_182);  arg364_1 = add_tensor_182 = None
        copy__default_183: "i64[]" = torch.ops.aten.copy_.default(arg368_1, add_tensor_183);  arg368_1 = add_tensor_183 = None
        copy__default_184: "f32[608]" = torch.ops.aten.copy_.default(arg369_1, add_tensor_184);  arg369_1 = add_tensor_184 = None
        copy__default_185: "f32[608]" = torch.ops.aten.copy_.default(arg370_1, add_tensor_185);  arg370_1 = add_tensor_185 = None
        copy__default_186: "i64[]" = torch.ops.aten.copy_.default(arg374_1, add_tensor_186);  arg374_1 = add_tensor_186 = None
        copy__default_187: "f32[128]" = torch.ops.aten.copy_.default(arg375_1, add_tensor_187);  arg375_1 = add_tensor_187 = None
        copy__default_188: "f32[128]" = torch.ops.aten.copy_.default(arg376_1, add_tensor_188);  arg376_1 = add_tensor_188 = None
        copy__default_189: "i64[]" = torch.ops.aten.copy_.default(arg380_1, add_tensor_189);  arg380_1 = add_tensor_189 = None
        copy__default_190: "f32[640]" = torch.ops.aten.copy_.default(arg381_1, add_tensor_190);  arg381_1 = add_tensor_190 = None
        copy__default_191: "f32[640]" = torch.ops.aten.copy_.default(arg382_1, add_tensor_191);  arg382_1 = add_tensor_191 = None
        copy__default_192: "i64[]" = torch.ops.aten.copy_.default(arg386_1, add_tensor_192);  arg386_1 = add_tensor_192 = None
        copy__default_193: "f32[128]" = torch.ops.aten.copy_.default(arg387_1, add_tensor_193);  arg387_1 = add_tensor_193 = None
        copy__default_194: "f32[128]" = torch.ops.aten.copy_.default(arg388_1, add_tensor_194);  arg388_1 = add_tensor_194 = None
        copy__default_195: "i64[]" = torch.ops.aten.copy_.default(arg392_1, add_tensor_195);  arg392_1 = add_tensor_195 = None
        copy__default_196: "f32[672]" = torch.ops.aten.copy_.default(arg393_1, add_tensor_196);  arg393_1 = add_tensor_196 = None
        copy__default_197: "f32[672]" = torch.ops.aten.copy_.default(arg394_1, add_tensor_197);  arg394_1 = add_tensor_197 = None
        copy__default_198: "i64[]" = torch.ops.aten.copy_.default(arg398_1, add_tensor_198);  arg398_1 = add_tensor_198 = None
        copy__default_199: "f32[128]" = torch.ops.aten.copy_.default(arg399_1, add_tensor_199);  arg399_1 = add_tensor_199 = None
        copy__default_200: "f32[128]" = torch.ops.aten.copy_.default(arg400_1, add_tensor_200);  arg400_1 = add_tensor_200 = None
        copy__default_201: "i64[]" = torch.ops.aten.copy_.default(arg404_1, add_tensor_201);  arg404_1 = add_tensor_201 = None
        copy__default_202: "f32[704]" = torch.ops.aten.copy_.default(arg405_1, add_tensor_202);  arg405_1 = add_tensor_202 = None
        copy__default_203: "f32[704]" = torch.ops.aten.copy_.default(arg406_1, add_tensor_203);  arg406_1 = add_tensor_203 = None
        copy__default_204: "i64[]" = torch.ops.aten.copy_.default(arg410_1, add_tensor_204);  arg410_1 = add_tensor_204 = None
        copy__default_205: "f32[128]" = torch.ops.aten.copy_.default(arg411_1, add_tensor_205);  arg411_1 = add_tensor_205 = None
        copy__default_206: "f32[128]" = torch.ops.aten.copy_.default(arg412_1, add_tensor_206);  arg412_1 = add_tensor_206 = None
        copy__default_207: "i64[]" = torch.ops.aten.copy_.default(arg416_1, add_tensor_207);  arg416_1 = add_tensor_207 = None
        copy__default_208: "f32[736]" = torch.ops.aten.copy_.default(arg417_1, add_tensor_208);  arg417_1 = add_tensor_208 = None
        copy__default_209: "f32[736]" = torch.ops.aten.copy_.default(arg418_1, add_tensor_209);  arg418_1 = add_tensor_209 = None
        copy__default_210: "i64[]" = torch.ops.aten.copy_.default(arg422_1, add_tensor_210);  arg422_1 = add_tensor_210 = None
        copy__default_211: "f32[128]" = torch.ops.aten.copy_.default(arg423_1, add_tensor_211);  arg423_1 = add_tensor_211 = None
        copy__default_212: "f32[128]" = torch.ops.aten.copy_.default(arg424_1, add_tensor_212);  arg424_1 = add_tensor_212 = None
        copy__default_213: "i64[]" = torch.ops.aten.copy_.default(arg428_1, add_tensor_213);  arg428_1 = add_tensor_213 = None
        copy__default_214: "f32[768]" = torch.ops.aten.copy_.default(arg429_1, add_tensor_214);  arg429_1 = add_tensor_214 = None
        copy__default_215: "f32[768]" = torch.ops.aten.copy_.default(arg430_1, add_tensor_215);  arg430_1 = add_tensor_215 = None
        copy__default_216: "i64[]" = torch.ops.aten.copy_.default(arg434_1, add_tensor_216);  arg434_1 = add_tensor_216 = None
        copy__default_217: "f32[128]" = torch.ops.aten.copy_.default(arg435_1, add_tensor_217);  arg435_1 = add_tensor_217 = None
        copy__default_218: "f32[128]" = torch.ops.aten.copy_.default(arg436_1, add_tensor_218);  arg436_1 = add_tensor_218 = None
        copy__default_219: "i64[]" = torch.ops.aten.copy_.default(arg440_1, add_tensor_219);  arg440_1 = add_tensor_219 = None
        copy__default_220: "f32[800]" = torch.ops.aten.copy_.default(arg441_1, add_tensor_220);  arg441_1 = add_tensor_220 = None
        copy__default_221: "f32[800]" = torch.ops.aten.copy_.default(arg442_1, add_tensor_221);  arg442_1 = add_tensor_221 = None
        copy__default_222: "i64[]" = torch.ops.aten.copy_.default(arg446_1, add_tensor_222);  arg446_1 = add_tensor_222 = None
        copy__default_223: "f32[128]" = torch.ops.aten.copy_.default(arg447_1, add_tensor_223);  arg447_1 = add_tensor_223 = None
        copy__default_224: "f32[128]" = torch.ops.aten.copy_.default(arg448_1, add_tensor_224);  arg448_1 = add_tensor_224 = None
        copy__default_225: "i64[]" = torch.ops.aten.copy_.default(arg452_1, add_tensor_225);  arg452_1 = add_tensor_225 = None
        copy__default_226: "f32[832]" = torch.ops.aten.copy_.default(arg453_1, add_tensor_226);  arg453_1 = add_tensor_226 = None
        copy__default_227: "f32[832]" = torch.ops.aten.copy_.default(arg454_1, add_tensor_227);  arg454_1 = add_tensor_227 = None
        copy__default_228: "i64[]" = torch.ops.aten.copy_.default(arg458_1, add_tensor_228);  arg458_1 = add_tensor_228 = None
        copy__default_229: "f32[128]" = torch.ops.aten.copy_.default(arg459_1, add_tensor_229);  arg459_1 = add_tensor_229 = None
        copy__default_230: "f32[128]" = torch.ops.aten.copy_.default(arg460_1, add_tensor_230);  arg460_1 = add_tensor_230 = None
        copy__default_231: "i64[]" = torch.ops.aten.copy_.default(arg464_1, add_tensor_231);  arg464_1 = add_tensor_231 = None
        copy__default_232: "f32[864]" = torch.ops.aten.copy_.default(arg465_1, add_tensor_232);  arg465_1 = add_tensor_232 = None
        copy__default_233: "f32[864]" = torch.ops.aten.copy_.default(arg466_1, add_tensor_233);  arg466_1 = add_tensor_233 = None
        copy__default_234: "i64[]" = torch.ops.aten.copy_.default(arg470_1, add_tensor_234);  arg470_1 = add_tensor_234 = None
        copy__default_235: "f32[128]" = torch.ops.aten.copy_.default(arg471_1, add_tensor_235);  arg471_1 = add_tensor_235 = None
        copy__default_236: "f32[128]" = torch.ops.aten.copy_.default(arg472_1, add_tensor_236);  arg472_1 = add_tensor_236 = None
        copy__default_237: "i64[]" = torch.ops.aten.copy_.default(arg476_1, add_tensor_237);  arg476_1 = add_tensor_237 = None
        copy__default_238: "f32[896]" = torch.ops.aten.copy_.default(arg477_1, add_tensor_238);  arg477_1 = add_tensor_238 = None
        copy__default_239: "f32[896]" = torch.ops.aten.copy_.default(arg478_1, add_tensor_239);  arg478_1 = add_tensor_239 = None
        copy__default_240: "i64[]" = torch.ops.aten.copy_.default(arg482_1, add_tensor_240);  arg482_1 = add_tensor_240 = None
        copy__default_241: "f32[128]" = torch.ops.aten.copy_.default(arg483_1, add_tensor_241);  arg483_1 = add_tensor_241 = None
        copy__default_242: "f32[128]" = torch.ops.aten.copy_.default(arg484_1, add_tensor_242);  arg484_1 = add_tensor_242 = None
        copy__default_243: "i64[]" = torch.ops.aten.copy_.default(arg488_1, add_tensor_243);  arg488_1 = add_tensor_243 = None
        copy__default_244: "f32[928]" = torch.ops.aten.copy_.default(arg489_1, add_tensor_244);  arg489_1 = add_tensor_244 = None
        copy__default_245: "f32[928]" = torch.ops.aten.copy_.default(arg490_1, add_tensor_245);  arg490_1 = add_tensor_245 = None
        copy__default_246: "i64[]" = torch.ops.aten.copy_.default(arg494_1, add_tensor_246);  arg494_1 = add_tensor_246 = None
        copy__default_247: "f32[128]" = torch.ops.aten.copy_.default(arg495_1, add_tensor_247);  arg495_1 = add_tensor_247 = None
        copy__default_248: "f32[128]" = torch.ops.aten.copy_.default(arg496_1, add_tensor_248);  arg496_1 = add_tensor_248 = None
        copy__default_249: "i64[]" = torch.ops.aten.copy_.default(arg500_1, add_tensor_249);  arg500_1 = add_tensor_249 = None
        copy__default_250: "f32[960]" = torch.ops.aten.copy_.default(arg501_1, add_tensor_250);  arg501_1 = add_tensor_250 = None
        copy__default_251: "f32[960]" = torch.ops.aten.copy_.default(arg502_1, add_tensor_251);  arg502_1 = add_tensor_251 = None
        copy__default_252: "i64[]" = torch.ops.aten.copy_.default(arg506_1, add_tensor_252);  arg506_1 = add_tensor_252 = None
        copy__default_253: "f32[128]" = torch.ops.aten.copy_.default(arg507_1, add_tensor_253);  arg507_1 = add_tensor_253 = None
        copy__default_254: "f32[128]" = torch.ops.aten.copy_.default(arg508_1, add_tensor_254);  arg508_1 = add_tensor_254 = None
        copy__default_255: "i64[]" = torch.ops.aten.copy_.default(arg512_1, add_tensor_255);  arg512_1 = add_tensor_255 = None
        copy__default_256: "f32[992]" = torch.ops.aten.copy_.default(arg513_1, add_tensor_256);  arg513_1 = add_tensor_256 = None
        copy__default_257: "f32[992]" = torch.ops.aten.copy_.default(arg514_1, add_tensor_257);  arg514_1 = add_tensor_257 = None
        copy__default_258: "i64[]" = torch.ops.aten.copy_.default(arg518_1, add_tensor_258);  arg518_1 = add_tensor_258 = None
        copy__default_259: "f32[128]" = torch.ops.aten.copy_.default(arg519_1, add_tensor_259);  arg519_1 = add_tensor_259 = None
        copy__default_260: "f32[128]" = torch.ops.aten.copy_.default(arg520_1, add_tensor_260);  arg520_1 = add_tensor_260 = None
        copy__default_261: "i64[]" = torch.ops.aten.copy_.default(arg524_1, add_tensor_261);  arg524_1 = add_tensor_261 = None
        copy__default_262: "f32[1024]" = torch.ops.aten.copy_.default(arg525_1, add_tensor_262);  arg525_1 = add_tensor_262 = None
        copy__default_263: "f32[1024]" = torch.ops.aten.copy_.default(arg526_1, add_tensor_263);  arg526_1 = add_tensor_263 = None
        copy__default_264: "i64[]" = torch.ops.aten.copy_.default(arg530_1, add_tensor_264);  arg530_1 = add_tensor_264 = None
        copy__default_265: "f32[512]" = torch.ops.aten.copy_.default(arg531_1, add_tensor_265);  arg531_1 = add_tensor_265 = None
        copy__default_266: "f32[512]" = torch.ops.aten.copy_.default(arg532_1, add_tensor_266);  arg532_1 = add_tensor_266 = None
        copy__default_267: "i64[]" = torch.ops.aten.copy_.default(arg536_1, add_tensor_267);  arg536_1 = add_tensor_267 = None
        copy__default_268: "f32[128]" = torch.ops.aten.copy_.default(arg537_1, add_tensor_268);  arg537_1 = add_tensor_268 = None
        copy__default_269: "f32[128]" = torch.ops.aten.copy_.default(arg538_1, add_tensor_269);  arg538_1 = add_tensor_269 = None
        copy__default_270: "i64[]" = torch.ops.aten.copy_.default(arg542_1, add_tensor_270);  arg542_1 = add_tensor_270 = None
        copy__default_271: "f32[544]" = torch.ops.aten.copy_.default(arg543_1, add_tensor_271);  arg543_1 = add_tensor_271 = None
        copy__default_272: "f32[544]" = torch.ops.aten.copy_.default(arg544_1, add_tensor_272);  arg544_1 = add_tensor_272 = None
        copy__default_273: "i64[]" = torch.ops.aten.copy_.default(arg548_1, add_tensor_273);  arg548_1 = add_tensor_273 = None
        copy__default_274: "f32[128]" = torch.ops.aten.copy_.default(arg549_1, add_tensor_274);  arg549_1 = add_tensor_274 = None
        copy__default_275: "f32[128]" = torch.ops.aten.copy_.default(arg550_1, add_tensor_275);  arg550_1 = add_tensor_275 = None
        copy__default_276: "i64[]" = torch.ops.aten.copy_.default(arg554_1, add_tensor_276);  arg554_1 = add_tensor_276 = None
        copy__default_277: "f32[576]" = torch.ops.aten.copy_.default(arg555_1, add_tensor_277);  arg555_1 = add_tensor_277 = None
        copy__default_278: "f32[576]" = torch.ops.aten.copy_.default(arg556_1, add_tensor_278);  arg556_1 = add_tensor_278 = None
        copy__default_279: "i64[]" = torch.ops.aten.copy_.default(arg560_1, add_tensor_279);  arg560_1 = add_tensor_279 = None
        copy__default_280: "f32[128]" = torch.ops.aten.copy_.default(arg561_1, add_tensor_280);  arg561_1 = add_tensor_280 = None
        copy__default_281: "f32[128]" = torch.ops.aten.copy_.default(arg562_1, add_tensor_281);  arg562_1 = add_tensor_281 = None
        copy__default_282: "i64[]" = torch.ops.aten.copy_.default(arg566_1, add_tensor_282);  arg566_1 = add_tensor_282 = None
        copy__default_283: "f32[608]" = torch.ops.aten.copy_.default(arg567_1, add_tensor_283);  arg567_1 = add_tensor_283 = None
        copy__default_284: "f32[608]" = torch.ops.aten.copy_.default(arg568_1, add_tensor_284);  arg568_1 = add_tensor_284 = None
        copy__default_285: "i64[]" = torch.ops.aten.copy_.default(arg572_1, add_tensor_285);  arg572_1 = add_tensor_285 = None
        copy__default_286: "f32[128]" = torch.ops.aten.copy_.default(arg573_1, add_tensor_286);  arg573_1 = add_tensor_286 = None
        copy__default_287: "f32[128]" = torch.ops.aten.copy_.default(arg574_1, add_tensor_287);  arg574_1 = add_tensor_287 = None
        copy__default_288: "i64[]" = torch.ops.aten.copy_.default(arg578_1, add_tensor_288);  arg578_1 = add_tensor_288 = None
        copy__default_289: "f32[640]" = torch.ops.aten.copy_.default(arg579_1, add_tensor_289);  arg579_1 = add_tensor_289 = None
        copy__default_290: "f32[640]" = torch.ops.aten.copy_.default(arg580_1, add_tensor_290);  arg580_1 = add_tensor_290 = None
        copy__default_291: "i64[]" = torch.ops.aten.copy_.default(arg584_1, add_tensor_291);  arg584_1 = add_tensor_291 = None
        copy__default_292: "f32[128]" = torch.ops.aten.copy_.default(arg585_1, add_tensor_292);  arg585_1 = add_tensor_292 = None
        copy__default_293: "f32[128]" = torch.ops.aten.copy_.default(arg586_1, add_tensor_293);  arg586_1 = add_tensor_293 = None
        copy__default_294: "i64[]" = torch.ops.aten.copy_.default(arg590_1, add_tensor_294);  arg590_1 = add_tensor_294 = None
        copy__default_295: "f32[672]" = torch.ops.aten.copy_.default(arg591_1, add_tensor_295);  arg591_1 = add_tensor_295 = None
        copy__default_296: "f32[672]" = torch.ops.aten.copy_.default(arg592_1, add_tensor_296);  arg592_1 = add_tensor_296 = None
        copy__default_297: "i64[]" = torch.ops.aten.copy_.default(arg596_1, add_tensor_297);  arg596_1 = add_tensor_297 = None
        copy__default_298: "f32[128]" = torch.ops.aten.copy_.default(arg597_1, add_tensor_298);  arg597_1 = add_tensor_298 = None
        copy__default_299: "f32[128]" = torch.ops.aten.copy_.default(arg598_1, add_tensor_299);  arg598_1 = add_tensor_299 = None
        copy__default_300: "i64[]" = torch.ops.aten.copy_.default(arg602_1, add_tensor_300);  arg602_1 = add_tensor_300 = None
        copy__default_301: "f32[704]" = torch.ops.aten.copy_.default(arg603_1, add_tensor_301);  arg603_1 = add_tensor_301 = None
        copy__default_302: "f32[704]" = torch.ops.aten.copy_.default(arg604_1, add_tensor_302);  arg604_1 = add_tensor_302 = None
        copy__default_303: "i64[]" = torch.ops.aten.copy_.default(arg608_1, add_tensor_303);  arg608_1 = add_tensor_303 = None
        copy__default_304: "f32[128]" = torch.ops.aten.copy_.default(arg609_1, add_tensor_304);  arg609_1 = add_tensor_304 = None
        copy__default_305: "f32[128]" = torch.ops.aten.copy_.default(arg610_1, add_tensor_305);  arg610_1 = add_tensor_305 = None
        copy__default_306: "i64[]" = torch.ops.aten.copy_.default(arg614_1, add_tensor_306);  arg614_1 = add_tensor_306 = None
        copy__default_307: "f32[736]" = torch.ops.aten.copy_.default(arg615_1, add_tensor_307);  arg615_1 = add_tensor_307 = None
        copy__default_308: "f32[736]" = torch.ops.aten.copy_.default(arg616_1, add_tensor_308);  arg616_1 = add_tensor_308 = None
        copy__default_309: "i64[]" = torch.ops.aten.copy_.default(arg620_1, add_tensor_309);  arg620_1 = add_tensor_309 = None
        copy__default_310: "f32[128]" = torch.ops.aten.copy_.default(arg621_1, add_tensor_310);  arg621_1 = add_tensor_310 = None
        copy__default_311: "f32[128]" = torch.ops.aten.copy_.default(arg622_1, add_tensor_311);  arg622_1 = add_tensor_311 = None
        copy__default_312: "i64[]" = torch.ops.aten.copy_.default(arg626_1, add_tensor_312);  arg626_1 = add_tensor_312 = None
        copy__default_313: "f32[768]" = torch.ops.aten.copy_.default(arg627_1, add_tensor_313);  arg627_1 = add_tensor_313 = None
        copy__default_314: "f32[768]" = torch.ops.aten.copy_.default(arg628_1, add_tensor_314);  arg628_1 = add_tensor_314 = None
        copy__default_315: "i64[]" = torch.ops.aten.copy_.default(arg632_1, add_tensor_315);  arg632_1 = add_tensor_315 = None
        copy__default_316: "f32[128]" = torch.ops.aten.copy_.default(arg633_1, add_tensor_316);  arg633_1 = add_tensor_316 = None
        copy__default_317: "f32[128]" = torch.ops.aten.copy_.default(arg634_1, add_tensor_317);  arg634_1 = add_tensor_317 = None
        copy__default_318: "i64[]" = torch.ops.aten.copy_.default(arg638_1, add_tensor_318);  arg638_1 = add_tensor_318 = None
        copy__default_319: "f32[800]" = torch.ops.aten.copy_.default(arg639_1, add_tensor_319);  arg639_1 = add_tensor_319 = None
        copy__default_320: "f32[800]" = torch.ops.aten.copy_.default(arg640_1, add_tensor_320);  arg640_1 = add_tensor_320 = None
        copy__default_321: "i64[]" = torch.ops.aten.copy_.default(arg644_1, add_tensor_321);  arg644_1 = add_tensor_321 = None
        copy__default_322: "f32[128]" = torch.ops.aten.copy_.default(arg645_1, add_tensor_322);  arg645_1 = add_tensor_322 = None
        copy__default_323: "f32[128]" = torch.ops.aten.copy_.default(arg646_1, add_tensor_323);  arg646_1 = add_tensor_323 = None
        copy__default_324: "i64[]" = torch.ops.aten.copy_.default(arg650_1, add_tensor_324);  arg650_1 = add_tensor_324 = None
        copy__default_325: "f32[832]" = torch.ops.aten.copy_.default(arg651_1, add_tensor_325);  arg651_1 = add_tensor_325 = None
        copy__default_326: "f32[832]" = torch.ops.aten.copy_.default(arg652_1, add_tensor_326);  arg652_1 = add_tensor_326 = None
        copy__default_327: "i64[]" = torch.ops.aten.copy_.default(arg656_1, add_tensor_327);  arg656_1 = add_tensor_327 = None
        copy__default_328: "f32[128]" = torch.ops.aten.copy_.default(arg657_1, add_tensor_328);  arg657_1 = add_tensor_328 = None
        copy__default_329: "f32[128]" = torch.ops.aten.copy_.default(arg658_1, add_tensor_329);  arg658_1 = add_tensor_329 = None
        copy__default_330: "i64[]" = torch.ops.aten.copy_.default(arg662_1, add_tensor_330);  arg662_1 = add_tensor_330 = None
        copy__default_331: "f32[864]" = torch.ops.aten.copy_.default(arg663_1, add_tensor_331);  arg663_1 = add_tensor_331 = None
        copy__default_332: "f32[864]" = torch.ops.aten.copy_.default(arg664_1, add_tensor_332);  arg664_1 = add_tensor_332 = None
        copy__default_333: "i64[]" = torch.ops.aten.copy_.default(arg668_1, add_tensor_333);  arg668_1 = add_tensor_333 = None
        copy__default_334: "f32[128]" = torch.ops.aten.copy_.default(arg669_1, add_tensor_334);  arg669_1 = add_tensor_334 = None
        copy__default_335: "f32[128]" = torch.ops.aten.copy_.default(arg670_1, add_tensor_335);  arg670_1 = add_tensor_335 = None
        copy__default_336: "i64[]" = torch.ops.aten.copy_.default(arg674_1, add_tensor_336);  arg674_1 = add_tensor_336 = None
        copy__default_337: "f32[896]" = torch.ops.aten.copy_.default(arg675_1, add_tensor_337);  arg675_1 = add_tensor_337 = None
        copy__default_338: "f32[896]" = torch.ops.aten.copy_.default(arg676_1, add_tensor_338);  arg676_1 = add_tensor_338 = None
        copy__default_339: "i64[]" = torch.ops.aten.copy_.default(arg680_1, add_tensor_339);  arg680_1 = add_tensor_339 = None
        copy__default_340: "f32[128]" = torch.ops.aten.copy_.default(arg681_1, add_tensor_340);  arg681_1 = add_tensor_340 = None
        copy__default_341: "f32[128]" = torch.ops.aten.copy_.default(arg682_1, add_tensor_341);  arg682_1 = add_tensor_341 = None
        copy__default_342: "i64[]" = torch.ops.aten.copy_.default(arg686_1, add_tensor_342);  arg686_1 = add_tensor_342 = None
        copy__default_343: "f32[928]" = torch.ops.aten.copy_.default(arg687_1, add_tensor_343);  arg687_1 = add_tensor_343 = None
        copy__default_344: "f32[928]" = torch.ops.aten.copy_.default(arg688_1, add_tensor_344);  arg688_1 = add_tensor_344 = None
        copy__default_345: "i64[]" = torch.ops.aten.copy_.default(arg692_1, add_tensor_345);  arg692_1 = add_tensor_345 = None
        copy__default_346: "f32[128]" = torch.ops.aten.copy_.default(arg693_1, add_tensor_346);  arg693_1 = add_tensor_346 = None
        copy__default_347: "f32[128]" = torch.ops.aten.copy_.default(arg694_1, add_tensor_347);  arg694_1 = add_tensor_347 = None
        copy__default_348: "i64[]" = torch.ops.aten.copy_.default(arg698_1, add_tensor_348);  arg698_1 = add_tensor_348 = None
        copy__default_349: "f32[960]" = torch.ops.aten.copy_.default(arg699_1, add_tensor_349);  arg699_1 = add_tensor_349 = None
        copy__default_350: "f32[960]" = torch.ops.aten.copy_.default(arg700_1, add_tensor_350);  arg700_1 = add_tensor_350 = None
        copy__default_351: "i64[]" = torch.ops.aten.copy_.default(arg704_1, add_tensor_351);  arg704_1 = add_tensor_351 = None
        copy__default_352: "f32[128]" = torch.ops.aten.copy_.default(arg705_1, add_tensor_352);  arg705_1 = add_tensor_352 = None
        copy__default_353: "f32[128]" = torch.ops.aten.copy_.default(arg706_1, add_tensor_353);  arg706_1 = add_tensor_353 = None
        copy__default_354: "i64[]" = torch.ops.aten.copy_.default(arg710_1, add_tensor_354);  arg710_1 = add_tensor_354 = None
        copy__default_355: "f32[992]" = torch.ops.aten.copy_.default(arg711_1, add_tensor_355);  arg711_1 = add_tensor_355 = None
        copy__default_356: "f32[992]" = torch.ops.aten.copy_.default(arg712_1, add_tensor_356);  arg712_1 = add_tensor_356 = None
        copy__default_357: "i64[]" = torch.ops.aten.copy_.default(arg716_1, add_tensor_357);  arg716_1 = add_tensor_357 = None
        copy__default_358: "f32[128]" = torch.ops.aten.copy_.default(arg717_1, add_tensor_358);  arg717_1 = add_tensor_358 = None
        copy__default_359: "f32[128]" = torch.ops.aten.copy_.default(arg718_1, add_tensor_359);  arg718_1 = add_tensor_359 = None
        copy__default_360: "i64[]" = torch.ops.aten.copy_.default(arg722_1, add_tensor_360);  arg722_1 = add_tensor_360 = None
        copy__default_361: "f32[1024]" = torch.ops.aten.copy_.default(arg723_1, add_tensor_362);  arg723_1 = add_tensor_362 = None
        copy__default_362: "f32[1024]" = torch.ops.aten.copy_.default(arg724_1, add_tensor_363);  arg724_1 = add_tensor_363 = None
        return (squeeze_dims_3, squeeze_dims_6, squeeze_dims_9, squeeze_dims_12, squeeze_dims_15, squeeze_dims_18, squeeze_dims_21, squeeze_dims_24, squeeze_dims_27, squeeze_dims_30, squeeze_dims_33, squeeze_dims_36, squeeze_dims_39, squeeze_dims_42, squeeze_dims_45, squeeze_dims_48, squeeze_dims_51, squeeze_dims_54, squeeze_dims_57, squeeze_dims_60, squeeze_dims_63, squeeze_dims_66, squeeze_dims_69, squeeze_dims_72, squeeze_dims_75, squeeze_dims_78, squeeze_dims_81, squeeze_dims_84, squeeze_dims_87, squeeze_dims_90, squeeze_dims_93, squeeze_dims_96, squeeze_dims_99, squeeze_dims_102, squeeze_dims_105, squeeze_dims_108, squeeze_dims_111, squeeze_dims_114, squeeze_dims_117, squeeze_dims_120, squeeze_dims_123, squeeze_dims_126, squeeze_dims_129, squeeze_dims_132, squeeze_dims_135, squeeze_dims_138, squeeze_dims_141, squeeze_dims_144, squeeze_dims_147, squeeze_dims_150, squeeze_dims_153, squeeze_dims_156, squeeze_dims_159, squeeze_dims_162, squeeze_dims_165, squeeze_dims_168, squeeze_dims_171, squeeze_dims_174, squeeze_dims_177, squeeze_dims_180, squeeze_dims_183, squeeze_dims_186, squeeze_dims_189, squeeze_dims_192, squeeze_dims_195, squeeze_dims_198, squeeze_dims_201, squeeze_dims_204, squeeze_dims_207, squeeze_dims_210, squeeze_dims_213, squeeze_dims_216, squeeze_dims_219, squeeze_dims_222, squeeze_dims_225, squeeze_dims_228, squeeze_dims_231, squeeze_dims_234, squeeze_dims_237, squeeze_dims_240, squeeze_dims_243, squeeze_dims_246, squeeze_dims_249, squeeze_dims_252, squeeze_dims_255, squeeze_dims_258, squeeze_dims_261, squeeze_dims_264, squeeze_dims_267, squeeze_dims_270, squeeze_dims_273, squeeze_dims_276, squeeze_dims_279, squeeze_dims_282, squeeze_dims_285, squeeze_dims_288, squeeze_dims_291, squeeze_dims_294, squeeze_dims_297, squeeze_dims_300, squeeze_dims_303, squeeze_dims_306, squeeze_dims_309, squeeze_dims_312, squeeze_dims_315, squeeze_dims_318, squeeze_dims_321, squeeze_dims_324, squeeze_dims_327, squeeze_dims_330, squeeze_dims_333, squeeze_dims_336, squeeze_dims_339, squeeze_dims_342, squeeze_dims_345, squeeze_dims_348, squeeze_dims_351, squeeze_dims_354, squeeze_dims_357, reshape_default, convert_element_type_default_1, permute_default_1, unsqueeze_default_6, unsqueeze_default_9, unsqueeze_default_12, unsqueeze_default_15, unsqueeze_default_18, unsqueeze_default_21, unsqueeze_default_24, unsqueeze_default_27, unsqueeze_default_30, unsqueeze_default_33, unsqueeze_default_36, unsqueeze_default_39, unsqueeze_default_42, unsqueeze_default_45, unsqueeze_default_48, unsqueeze_default_51, unsqueeze_default_54, unsqueeze_default_57, unsqueeze_default_60, unsqueeze_default_63, unsqueeze_default_66, unsqueeze_default_69, unsqueeze_default_72, unsqueeze_default_75, unsqueeze_default_78, unsqueeze_default_81, unsqueeze_default_84, unsqueeze_default_87, unsqueeze_default_90, unsqueeze_default_93, unsqueeze_default_96, unsqueeze_default_99, unsqueeze_default_102, unsqueeze_default_105, unsqueeze_default_108, unsqueeze_default_111, unsqueeze_default_114, unsqueeze_default_117, unsqueeze_default_120, unsqueeze_default_123, unsqueeze_default_126, unsqueeze_default_129, unsqueeze_default_132, unsqueeze_default_135, unsqueeze_default_138, unsqueeze_default_141, unsqueeze_default_144, unsqueeze_default_147, unsqueeze_default_150, unsqueeze_default_153, unsqueeze_default_156, unsqueeze_default_159, unsqueeze_default_162, unsqueeze_default_165, unsqueeze_default_168, unsqueeze_default_171, unsqueeze_default_174, unsqueeze_default_177, unsqueeze_default_180, unsqueeze_default_183, unsqueeze_default_186, unsqueeze_default_189, unsqueeze_default_192, unsqueeze_default_195, unsqueeze_default_198, unsqueeze_default_201, unsqueeze_default_204, unsqueeze_default_207, unsqueeze_default_210, unsqueeze_default_213, unsqueeze_default_216, unsqueeze_default_219, unsqueeze_default_222, unsqueeze_default_225, unsqueeze_default_228, unsqueeze_default_231, unsqueeze_default_234, unsqueeze_default_237, unsqueeze_default_240, unsqueeze_default_243, unsqueeze_default_246, unsqueeze_default_249, unsqueeze_default_252, unsqueeze_default_255, unsqueeze_default_258, unsqueeze_default_261, unsqueeze_default_264, unsqueeze_default_267, unsqueeze_default_270, unsqueeze_default_273, unsqueeze_default_276, unsqueeze_default_279, unsqueeze_default_282, unsqueeze_default_285, unsqueeze_default_288, unsqueeze_default_291, unsqueeze_default_294, unsqueeze_default_297, unsqueeze_default_300, unsqueeze_default_303, unsqueeze_default_306, unsqueeze_default_309, unsqueeze_default_312, unsqueeze_default_315, unsqueeze_default_318, unsqueeze_default_321, unsqueeze_default_324, unsqueeze_default_327, unsqueeze_default_330, unsqueeze_default_333, unsqueeze_default_336, unsqueeze_default_339, unsqueeze_default_342, unsqueeze_default_345, unsqueeze_default_348, unsqueeze_default_351, unsqueeze_default_354, unsqueeze_default_357, unsqueeze_default_360, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146, copy__default_147, copy__default_148, copy__default_149, copy__default_150, copy__default_151, copy__default_152, copy__default_153, copy__default_154, copy__default_155, copy__default_156, copy__default_157, copy__default_158, copy__default_159, copy__default_160, copy__default_161, copy__default_162, copy__default_163, copy__default_164, copy__default_165, copy__default_166, copy__default_167, copy__default_168, copy__default_169, copy__default_170, copy__default_171, copy__default_172, copy__default_173, copy__default_174, copy__default_175, copy__default_176, copy__default_177, copy__default_178, copy__default_179, copy__default_180, copy__default_181, copy__default_182, copy__default_183, copy__default_184, copy__default_185, copy__default_186, copy__default_187, copy__default_188, copy__default_189, copy__default_190, copy__default_191, copy__default_192, copy__default_193, copy__default_194, copy__default_195, copy__default_196, copy__default_197, copy__default_198, copy__default_199, copy__default_200, copy__default_201, copy__default_202, copy__default_203, copy__default_204, copy__default_205, copy__default_206, copy__default_207, copy__default_208, copy__default_209, copy__default_210, copy__default_211, copy__default_212, copy__default_213, copy__default_214, copy__default_215, copy__default_216, copy__default_217, copy__default_218, copy__default_219, copy__default_220, copy__default_221, copy__default_222, copy__default_223, copy__default_224, copy__default_225, copy__default_226, copy__default_227, copy__default_228, copy__default_229, copy__default_230, copy__default_231, copy__default_232, copy__default_233, copy__default_234, copy__default_235, copy__default_236, copy__default_237, copy__default_238, copy__default_239, copy__default_240, copy__default_241, copy__default_242, copy__default_243, copy__default_244, copy__default_245, copy__default_246, copy__default_247, copy__default_248, copy__default_249, copy__default_250, copy__default_251, copy__default_252, copy__default_253, copy__default_254, copy__default_255, copy__default_256, copy__default_257, copy__default_258, copy__default_259, copy__default_260, copy__default_261, copy__default_262, copy__default_263, copy__default_264, copy__default_265, copy__default_266, copy__default_267, copy__default_268, copy__default_269, copy__default_270, copy__default_271, copy__default_272, copy__default_273, copy__default_274, copy__default_275, copy__default_276, copy__default_277, copy__default_278, copy__default_279, copy__default_280, copy__default_281, copy__default_282, copy__default_283, copy__default_284, copy__default_285, copy__default_286, copy__default_287, copy__default_288, copy__default_289, copy__default_290, copy__default_291, copy__default_292, copy__default_293, copy__default_294, copy__default_295, copy__default_296, copy__default_297, copy__default_298, copy__default_299, copy__default_300, copy__default_301, copy__default_302, copy__default_303, copy__default_304, copy__default_305, copy__default_306, copy__default_307, copy__default_308, copy__default_309, copy__default_310, copy__default_311, copy__default_312, copy__default_313, copy__default_314, copy__default_315, copy__default_316, copy__default_317, copy__default_318, copy__default_319, copy__default_320, copy__default_321, copy__default_322, copy__default_323, copy__default_324, copy__default_325, copy__default_326, copy__default_327, copy__default_328, copy__default_329, copy__default_330, copy__default_331, copy__default_332, copy__default_333, copy__default_334, copy__default_335, copy__default_336, copy__default_337, copy__default_338, copy__default_339, copy__default_340, copy__default_341, copy__default_342, copy__default_343, copy__default_344, copy__default_345, copy__default_346, copy__default_347, copy__default_348, copy__default_349, copy__default_350, copy__default_351, copy__default_352, copy__default_353, copy__default_354, copy__default_355, copy__default_356, copy__default_357, copy__default_358, copy__default_359, copy__default_360, copy__default_361, copy__default_362)


def _default_make_inputs():
    return [
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([1, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([1, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 224, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 224, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    torch.randn([1, 224, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randn([1, 160, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([160], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 224, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 224, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    torch.randn([1, 224, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([224], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 288, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 288, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    torch.randn([1, 288, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 352, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 352, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    torch.randn([1, 352, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 416, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 416, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([416], dtype=torch.float32, device='cuda'),
    torch.randn([1, 416, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([416], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 448, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 448, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn([1, 448, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randn([1, 256, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([256], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 288, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 288, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    torch.randn([1, 288, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([288], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randn([1, 320, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([320], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 352, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 352, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    torch.randn([1, 352, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([352], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 416, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 416, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([416], dtype=torch.float32, device='cuda'),
    torch.randn([1, 416, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([416], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 448, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 448, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randn([1, 448, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([448], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randn([1, 480, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([480], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 544, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 544, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randn([1, 544, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 576, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 576, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([1, 576, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 608, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 608, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([608], dtype=torch.float32, device='cuda'),
    torch.randn([1, 608, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([608], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 640, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 640, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([1, 640, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 704, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 704, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([704], dtype=torch.float32, device='cuda'),
    torch.randn([1, 704, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([704], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 736, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 736, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([736], dtype=torch.float32, device='cuda'),
    torch.randn([1, 736, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([736], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 800, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 800, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([800], dtype=torch.float32, device='cuda'),
    torch.randn([1, 800, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([800], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 832, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 832, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([832], dtype=torch.float32, device='cuda'),
    torch.randn([1, 832, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([832], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 864, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 864, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([864], dtype=torch.float32, device='cuda'),
    torch.randn([1, 864, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([864], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 896, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 896, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    torch.randn([1, 896, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 928, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 928, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([928], dtype=torch.float32, device='cuda'),
    torch.randn([1, 928, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([928], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 992, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 992, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([992], dtype=torch.float32, device='cuda'),
    torch.randn([1, 992, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([992], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 1024, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1024, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1024, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1, 512, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 544, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 544, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randn([1, 544, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([544], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 576, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 576, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([1, 576, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 608, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 608, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([608], dtype=torch.float32, device='cuda'),
    torch.randn([1, 608, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([608], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 640, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 640, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randn([1, 640, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([640], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randn([1, 672, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([672], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 704, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 704, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([704], dtype=torch.float32, device='cuda'),
    torch.randn([1, 704, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([704], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 736, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 736, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([736], dtype=torch.float32, device='cuda'),
    torch.randn([1, 736, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([736], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 768, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 800, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 800, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([800], dtype=torch.float32, device='cuda'),
    torch.randn([1, 800, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([800], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 832, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 832, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([832], dtype=torch.float32, device='cuda'),
    torch.randn([1, 832, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([832], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 864, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 864, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([864], dtype=torch.float32, device='cuda'),
    torch.randn([1, 864, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([864], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 896, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 896, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    torch.randn([1, 896, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([896], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 928, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 928, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([928], dtype=torch.float32, device='cuda'),
    torch.randn([1, 928, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([928], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randn([1, 960, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([960], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 992, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 992, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([992], dtype=torch.float32, device='cuda'),
    torch.randn([1, 992, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([992], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randn([1, 128, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([128], dtype=torch.float32, device='cuda'),
    torch.randint(0, 2, [], dtype=torch.int64, device='cuda'),
    torch.randn([1, 1024, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 1024, 7, 7], dtype=torch.float16, device='cuda'),
    torch.randn([1, 1024, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1024], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 1024], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
