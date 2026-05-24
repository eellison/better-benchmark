import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0", primals_2: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_4: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_5: "f32[32, 16, 3, 3][144, 1, 48, 16]cuda:0", primals_6: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_8: "f32[64, 32, 3, 3][288, 1, 96, 32]cuda:0", primals_9: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_11: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_12: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_14: "f32[256, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_15: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_17: "f32[64, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_18: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_20: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_21: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_23: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_24: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_26: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_27: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_29: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_31: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_33: "f32[512, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_34: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_36: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_37: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_39: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_40: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_42: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_43: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_45: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_46: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_48: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_50: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_52: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_53: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_55: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_56: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_58: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_59: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_61: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_62: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_64: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_66: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_68: "f32[1536, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_69: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_71: "f32[384, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_72: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_74: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_75: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_77: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_78: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_80: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_81: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_83: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_85: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_87: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_88: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_90: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_91: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_93: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_94: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_96: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_97: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_99: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_101: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_103: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_104: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_106: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_107: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_109: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_110: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_112: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_113: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_115: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_117: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_119: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_120: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_122: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_123: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_125: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_126: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_128: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_129: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_131: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_133: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_135: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_136: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_138: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_139: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_141: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_142: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_144: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_145: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_147: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_149: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_151: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_152: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_154: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_155: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_157: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_158: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_160: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_161: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_163: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_165: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_167: "f32[1536, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_168: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_170: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_171: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_173: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_174: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_176: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_177: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_179: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_180: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_182: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_184: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_186: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_187: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_189: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_190: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_192: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_193: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_195: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_196: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_198: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_200: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_202: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_203: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_205: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_206: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_208: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_209: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_211: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_212: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_214: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_216: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_218: "f32[2304, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_219: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_221: "f32[1000, 2304][2304, 1]cuda:0", squeeze_1: "f32[16][1]cuda:0", view_2: "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0", convolution: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0", div: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0", squeeze_3: "f32[32][1]cuda:0", view_5: "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0", convolution_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", div_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", squeeze_5: "f32[64][1]cuda:0", view_8: "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0", convolution_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0", div_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0", squeeze_7: "f32[128][1]cuda:0", view_11: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_3: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", mul_12: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", squeeze_9: "f32[256][1]cuda:0", view_14: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_4: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", squeeze_11: "f32[64][1]cuda:0", view_17: "f32[64, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_5: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", div_4: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_13: "f32[64][1]cuda:0", view_20: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_6: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", div_5: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_15: "f32[64][1]cuda:0", view_23: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_7: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", div_6: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_17: "f32[256][1]cuda:0", view_26: "f32[256, 64, 1, 1][64, 1, 1, 1]cuda:0", convolution_8: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", mean: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", relu: "f32[128, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_10: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0", mul_31: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", avg_pool2d: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0", squeeze_19: "f32[512][1]cuda:0", view_29: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_11: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_21: "f32[128][1]cuda:0", view_32: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_12: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", div_8: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", squeeze_23: "f32[128][1]cuda:0", view_35: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_13: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", div_9: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_25: "f32[128][1]cuda:0", view_38: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_14: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", div_10: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_27: "f32[512][1]cuda:0", view_41: "f32[512, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_15: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", mean_1: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0", relu_1: "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_17: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0", mul_50: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_29: "f32[128][1]cuda:0", view_44: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_18: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", div_12: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_31: "f32[128][1]cuda:0", view_47: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_19: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", div_13: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_33: "f32[128][1]cuda:0", view_50: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_20: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", div_14: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_35: "f32[512][1]cuda:0", view_53: "f32[512, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_21: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", mean_2: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0", relu_2: "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_23: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0", add_35: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", mul_66: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", avg_pool2d_1: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0", squeeze_37: "f32[1536][1]cuda:0", view_56: "f32[1536, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_24: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_39: "f32[384][1]cuda:0", view_59: "f32[384, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_25: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", div_16: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", squeeze_41: "f32[384][1]cuda:0", view_62: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_17: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_43: "f32[384][1]cuda:0", view_65: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_27: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_18: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_45: "f32[1536][1]cuda:0", view_68: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_28: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_3: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", relu_3: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_30: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", mul_85: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_47: "f32[384][1]cuda:0", view_71: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_31: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_20: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_49: "f32[384][1]cuda:0", view_74: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_32: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_21: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_51: "f32[384][1]cuda:0", view_77: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_22: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_53: "f32[1536][1]cuda:0", view_80: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_34: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_4: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", relu_4: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_36: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_54: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_101: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_55: "f32[384][1]cuda:0", view_83: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_37: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_24: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_57: "f32[384][1]cuda:0", view_86: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_38: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_25: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_59: "f32[384][1]cuda:0", view_89: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_39: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_61: "f32[1536][1]cuda:0", view_92: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_40: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_5: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", relu_5: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_42: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_63: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_117: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_63: "f32[384][1]cuda:0", view_95: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_43: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_28: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_65: "f32[384][1]cuda:0", view_98: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_44: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_29: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_67: "f32[384][1]cuda:0", view_101: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_45: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_30: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_69: "f32[1536][1]cuda:0", view_104: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_46: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_6: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", relu_6: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_48: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_72: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_133: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_71: "f32[384][1]cuda:0", view_107: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_49: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_32: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_73: "f32[384][1]cuda:0", view_110: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_50: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_75: "f32[384][1]cuda:0", view_113: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_51: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_34: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_77: "f32[1536][1]cuda:0", view_116: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_52: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_7: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", relu_7: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_54: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_81: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_149: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_79: "f32[384][1]cuda:0", view_119: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_55: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_36: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_81: "f32[384][1]cuda:0", view_122: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_56: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_37: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_83: "f32[384][1]cuda:0", view_125: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_57: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_38: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_85: "f32[1536][1]cuda:0", view_128: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_58: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_8: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", relu_8: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_60: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_90: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_165: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", avg_pool2d_2: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_87: "f32[1536][1]cuda:0", view_131: "f32[1536, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_61: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_89: "f32[384][1]cuda:0", view_134: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_62: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", div_40: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_91: "f32[384][1]cuda:0", view_137: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_63: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", div_41: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_93: "f32[384][1]cuda:0", view_140: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_64: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", div_42: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_95: "f32[1536][1]cuda:0", view_143: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_65: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", mean_9: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", relu_9: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_67: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", mul_184: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_97: "f32[384][1]cuda:0", view_146: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_68: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", div_44: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_99: "f32[384][1]cuda:0", view_149: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_69: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", div_45: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_101: "f32[384][1]cuda:0", view_152: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_70: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", div_46: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_103: "f32[1536][1]cuda:0", view_155: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_71: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", mean_10: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", relu_10: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_73: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_109: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", mul_200: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_105: "f32[384][1]cuda:0", view_158: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_74: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", div_48: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_107: "f32[384][1]cuda:0", view_161: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_75: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", div_49: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_109: "f32[384][1]cuda:0", view_164: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_76: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", div_50: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_111: "f32[1536][1]cuda:0", view_167: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_77: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", mean_11: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", relu_11: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_79: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_118: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_113: "f32[2304][1]cuda:0", view_170: "f32[2304, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_80: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0", view_171: "f32[128, 2304][2304, 1]cuda:0", unsqueeze_58: "f32[1, 2304, 1][2304, 1, 1]cuda:0", unsqueeze_66: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_74: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_82: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_90: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_98: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_106: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_114: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_122: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_130: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_138: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_146: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_154: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_162: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_170: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_178: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_186: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_194: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_202: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_210: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_218: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_226: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_234: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_242: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_250: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_258: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_266: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_274: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_282: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_290: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_298: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_306: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_314: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_322: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_330: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_338: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_346: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_354: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_362: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_370: "f32[1, 512, 1][512, 1, 1]cuda:0", unsqueeze_378: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_386: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_394: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_402: "f32[1, 512, 1][512, 1, 1]cuda:0", unsqueeze_410: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_418: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_426: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_434: "f32[1, 512, 1][512, 1, 1]cuda:0", unsqueeze_442: "f32[1, 256, 1][256, 1, 1]cuda:0", unsqueeze_450: "f32[1, 64, 1][64, 1, 1]cuda:0", unsqueeze_458: "f32[1, 64, 1][64, 1, 1]cuda:0", unsqueeze_466: "f32[1, 64, 1][64, 1, 1]cuda:0", unsqueeze_474: "f32[1, 256, 1][256, 1, 1]cuda:0", unsqueeze_482: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_490: "f32[1, 64, 1][64, 1, 1]cuda:0", unsqueeze_498: "f32[1, 32, 1][32, 1, 1]cuda:0", unsqueeze_506: "f32[1, 16, 1][16, 1, 1]cuda:0", tangents_1: "f32[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute: "f32[2304, 1000][1, 2304]cuda:0" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
        permute_1: "f32[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[128, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f32[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view_171);  permute_2 = view_171 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_172: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_173: "f32[128, 2304, 1, 1][2304, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 2304, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_114: "f32[128, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_173, 3);  view_173 = None
        squeeze_115: "f32[128, 2304][2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze_114, 2);  squeeze_114 = None
        full: "f32[294912][1]cuda:0" = torch.ops.aten.full.default([294912], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "f32[294912][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full, squeeze_115, [128, 2304], [2304, 1], 0);  full = squeeze_115 = None
        as_strided_5: "f32[128, 2304, 1, 1][2304, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 2304, 1, 1], [2304, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "f32[128, 2304, 7, 7][2304, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 2304, 7, 7]);  as_strided_5 = None
        div_52: "f32[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:569 in forward_features, code: x = self.final_act(x)
        neg_51: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.neg.default(convolution_80)
        exp_51: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.exp.default(neg_51);  neg_51 = None
        add_120: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.add.Tensor(exp_51, 1);  exp_51 = None
        reciprocal: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.reciprocal.default(add_120);  add_120 = None
        mul_219: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_220: "f32[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, mul_219);  div_52 = None
        sub_57: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_219);  mul_219 = None
        mul_221: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.mul.Tensor(convolution_80, sub_57);  convolution_80 = sub_57 = None
        add_122: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.add.Tensor(mul_221, 1);  mul_221 = None
        mul_222: "f32[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, add_122);  mul_220 = add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_2: "f32[2304][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_222, add_118, view_170, [2304], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_222 = add_118 = view_170 = None
        getitem_114: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = convolution_backward[0]
        getitem_115: "f32[2304, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_174: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_115, [1, 2304, 1536]);  getitem_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_3: "f32[2304][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_174, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_168: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_218, [1, 2304, -1]);  primals_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_58: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_168, unsqueeze_58);  view_168 = unsqueeze_58 = None
        mul_223: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_174, sub_58)
        sum_4: "f32[2304][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_223, [0, 2]);  mul_223 = None
        mul_224: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.0006510416666666666);  sum_3 = None
        unsqueeze_59: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_224, 0);  mul_224 = None
        unsqueeze_60: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 2);  unsqueeze_59 = None
        mul_225: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.0006510416666666666)
        mul_226: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, squeeze_113)
        mul_227: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        unsqueeze_61: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_227, 0);  mul_227 = None
        unsqueeze_62: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_216: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_219, 0.04562504637317021);  primals_219 = None
        view_169: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(mul_216, [-1]);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_228: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, view_169);  view_169 = None
        unsqueeze_63: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_228, 0);  mul_228 = None
        unsqueeze_64: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
        mul_229: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_62);  sub_58 = unsqueeze_62 = None
        sub_60: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_174, mul_229);  view_174 = mul_229 = None
        sub_61: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_60);  sub_60 = unsqueeze_60 = None
        mul_230: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_64);  sub_61 = unsqueeze_64 = None
        mul_231: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, squeeze_113);  sum_4 = squeeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_175: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_231, [2304, 1, 1, 1]);  mul_231 = None
        mul_232: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_175, 0.04562504637317021);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_176: "f32[2304, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_230, [2304, 1536, 1, 1]);  mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_233: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_114, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_234: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, 2.0);  mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_235: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, convolution_77);  convolution_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_11: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_236: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, sigmoid_11);  mul_234 = None
        sum_5: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_235, [2, 3], True);  mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_62: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_11)
        mul_237: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_11, sub_62);  sigmoid_11 = sub_62 = None
        mul_238: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, mul_237);  sum_5 = mul_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_6: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_238, relu_11, primals_216, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_238 = primals_216 = None
        getitem_117: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_1[0]
        getitem_118: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le, full_default, getitem_117);  le = getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_7: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where, mean_11, primals_214, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mean_11 = primals_214 = None
        getitem_120: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_2[0]
        getitem_121: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_2: "f32[128, 1536, 7, 7][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_120, [128, 1536, 7, 7]);  getitem_120 = None
        div_53: "f32[128, 1536, 7, 7][75264, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
        add_123: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_236, div_53);  mul_236 = div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_8: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_123, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_123, div_50, view_167, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_123 = div_50 = view_167 = None
        getitem_123: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_3[0]
        getitem_124: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_177: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_124, [1, 1536, 384]);  getitem_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_9: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_177, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_165: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_211, [1, 1536, -1]);  primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_63: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_165, unsqueeze_66);  view_165 = unsqueeze_66 = None
        mul_239: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_177, sub_63)
        sum_10: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 2]);  mul_239 = None
        mul_240: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.0026041666666666665);  sum_9 = None
        unsqueeze_67: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_240, 0);  mul_240 = None
        unsqueeze_68: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 2);  unsqueeze_67 = None
        mul_241: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.0026041666666666665)
        mul_242: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, squeeze_111)
        mul_243: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
        unsqueeze_69: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_243, 0);  mul_243 = None
        unsqueeze_70: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_210: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_212, 0.09125009274634042);  primals_212 = None
        view_166: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_210, [-1]);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_244: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, view_166);  view_166 = None
        unsqueeze_71: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
        unsqueeze_72: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
        mul_245: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_70);  sub_63 = unsqueeze_70 = None
        sub_65: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_177, mul_245);  view_177 = mul_245 = None
        sub_66: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_68);  sub_65 = unsqueeze_68 = None
        mul_246: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_72);  sub_66 = unsqueeze_72 = None
        mul_247: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, squeeze_111);  sum_10 = squeeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_178: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_247, [1536, 1, 1, 1]);  mul_247 = None
        mul_248: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_178, 0.09125009274634042);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_179: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_246, [1536, 384, 1, 1]);  mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_50: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convolution_76)
        exp_50: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_50);  neg_50 = None
        add_116: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_50, 1);  exp_50 = None
        reciprocal_1: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_116);  add_116 = None
        mul_249: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        mul_250: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_123, mul_249);  getitem_123 = None
        sub_67: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_249);  mul_249 = None
        mul_251: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_76, sub_67);  convolution_76 = sub_67 = None
        add_125: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_251, 1);  mul_251 = None
        mul_252: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, add_125);  mul_250 = add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_11: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_252, div_49, view_164, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_252 = div_49 = view_164 = None
        getitem_126: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_4[0]
        getitem_127: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_57: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_127, memory_format = torch.contiguous_format);  getitem_127 = None
        view_180: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [1, 384, 576]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_12: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_180, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_54: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_208, memory_format = torch.contiguous_format);  primals_208 = None
        view_162: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [1, 384, 576]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_68: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_162, unsqueeze_74);  view_162 = unsqueeze_74 = None
        mul_253: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_180, sub_68)
        sum_13: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 2]);  mul_253 = None
        mul_254: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.001736111111111111);  sum_12 = None
        unsqueeze_75: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_254, 0);  mul_254 = None
        unsqueeze_76: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
        mul_255: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.001736111111111111)
        mul_256: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_257: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, mul_256);  mul_255 = mul_256 = None
        unsqueeze_77: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_257, 0);  mul_257 = None
        unsqueeze_78: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_207: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_209, 0.07450538873672485);  primals_209 = None
        view_163: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_207, [-1]);  mul_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_258: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, view_163);  view_163 = None
        unsqueeze_79: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_258, 0);  mul_258 = None
        unsqueeze_80: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 2);  unsqueeze_79 = None
        mul_259: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_78);  sub_68 = unsqueeze_78 = None
        sub_70: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_180, mul_259);  view_180 = mul_259 = None
        sub_71: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_76);  sub_70 = unsqueeze_76 = None
        mul_260: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_80);  sub_71 = unsqueeze_80 = None
        mul_261: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_109);  sum_13 = squeeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_181: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_261, [384, 1, 1, 1]);  mul_261 = None
        mul_262: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_181, 0.07450538873672485);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_182: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_260, [384, 64, 3, 3]);  mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_49: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convolution_75)
        exp_49: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_49);  neg_49 = None
        add_114: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_49, 1);  exp_49 = None
        reciprocal_2: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_114);  add_114 = None
        mul_263: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        mul_264: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_126, mul_263);  getitem_126 = None
        sub_72: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_263);  mul_263 = None
        mul_265: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_75, sub_72);  convolution_75 = sub_72 = None
        add_127: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_265, 1);  mul_265 = None
        mul_266: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_264, add_127);  mul_264 = add_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_14: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_266, div_48, view_161, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_266 = div_48 = view_161 = None
        getitem_129: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_5[0]
        getitem_130: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_58: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_130, memory_format = torch.contiguous_format);  getitem_130 = None
        view_183: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [1, 384, 576]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_15: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_183, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_52: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_205, memory_format = torch.contiguous_format);  primals_205 = None
        view_159: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [1, 384, 576]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_73: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_159, unsqueeze_82);  view_159 = unsqueeze_82 = None
        mul_267: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_183, sub_73)
        sum_16: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 2]);  mul_267 = None
        mul_268: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.001736111111111111);  sum_15 = None
        unsqueeze_83: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
        unsqueeze_84: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
        mul_269: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.001736111111111111)
        mul_270: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, squeeze_107)
        mul_271: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
        unsqueeze_85: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_86: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 2);  unsqueeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_204: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_206, 0.07450538873672485);  primals_206 = None
        view_160: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_204, [-1]);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_272: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, view_160);  view_160 = None
        unsqueeze_87: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
        unsqueeze_88: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
        mul_273: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_86);  sub_73 = unsqueeze_86 = None
        sub_75: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_183, mul_273);  view_183 = mul_273 = None
        sub_76: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_84);  sub_75 = unsqueeze_84 = None
        mul_274: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_88);  sub_76 = unsqueeze_88 = None
        mul_275: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, squeeze_107);  sum_16 = squeeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_184: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_275, [384, 1, 1, 1]);  mul_275 = None
        mul_276: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_184, 0.07450538873672485);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_185: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_274, [384, 64, 3, 3]);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_48: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convolution_74)
        exp_48: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_48);  neg_48 = None
        add_112: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_48, 1);  exp_48 = None
        reciprocal_3: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_112);  add_112 = None
        mul_277: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        mul_278: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_129, mul_277);  getitem_129 = None
        sub_77: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_277);  mul_277 = None
        mul_279: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_74, sub_77);  convolution_74 = sub_77 = None
        add_129: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_279, 1);  mul_279 = None
        mul_280: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, add_129);  mul_278 = add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_17: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 2, 3])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_280, mul_200, view_158, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_280 = mul_200 = view_158 = None
        getitem_132: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = convolution_backward_6[0]
        getitem_133: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_186: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_133, [1, 384, 1536]);  getitem_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_18: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_186, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_156: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_202, [1, 384, -1]);  primals_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_78: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_156, unsqueeze_90);  view_156 = unsqueeze_90 = None
        mul_281: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_186, sub_78)
        sum_19: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 2]);  mul_281 = None
        mul_282: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.0006510416666666666);  sum_18 = None
        unsqueeze_91: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_282, 0);  mul_282 = None
        unsqueeze_92: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 2);  unsqueeze_91 = None
        mul_283: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.0006510416666666666)
        mul_284: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, squeeze_105)
        mul_285: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
        unsqueeze_93: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
        unsqueeze_94: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_201: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_203, 0.04562504637317021);  primals_203 = None
        view_157: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_201, [-1]);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_286: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, view_157);  view_157 = None
        unsqueeze_95: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
        unsqueeze_96: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        mul_287: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_94);  sub_78 = unsqueeze_94 = None
        sub_80: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_186, mul_287);  view_186 = mul_287 = None
        sub_81: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_92);  sub_80 = unsqueeze_92 = None
        mul_288: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_96);  sub_81 = unsqueeze_96 = None
        mul_289: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_105);  sum_19 = squeeze_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_187: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_289, [384, 1, 1, 1]);  mul_289 = None
        mul_290: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_187, 0.04562504637317021);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_188: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_288, [384, 1536, 1, 1]);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_291: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_132, 0.9622504486493761);  getitem_132 = None
        neg_47: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.neg.default(add_109)
        exp_47: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_110: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_47, 1);  exp_47 = None
        reciprocal_4: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.reciprocal.default(add_110);  add_110 = None
        mul_292: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        mul_293: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_291, mul_292);  mul_291 = None
        sub_82: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_292);  mul_292 = None
        mul_294: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_109, sub_82);  add_109 = sub_82 = None
        add_131: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_294, 1);  mul_294 = None
        mul_295: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, add_131);  mul_293 = add_131 = None
        add_132: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(getitem_114, mul_295);  getitem_114 = mul_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_296: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_132, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_297: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_296, 2.0);  mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_298: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, convolution_71);  convolution_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_10: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_299: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, sigmoid_10);  mul_297 = None
        sum_20: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_298, [2, 3], True);  mul_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_83: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_10)
        mul_300: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_10, sub_83);  sigmoid_10 = sub_83 = None
        mul_301: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, mul_300);  sum_20 = mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_21: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_301, relu_10, primals_200, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_301 = primals_200 = None
        getitem_135: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_7[0]
        getitem_136: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_1: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_1: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_1, full_default, getitem_135);  le_1 = getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_22: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_1, mean_10, primals_198, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_10 = primals_198 = None
        getitem_138: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_8[0]
        getitem_139: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_3: "f32[128, 1536, 7, 7][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_138, [128, 1536, 7, 7]);  getitem_138 = None
        div_54: "f32[128, 1536, 7, 7][75264, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
        add_133: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_299, div_54);  mul_299 = div_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_23: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_133, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(add_133, div_46, view_155, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_133 = div_46 = view_155 = None
        getitem_141: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_9[0]
        getitem_142: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_189: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_142, [1, 1536, 384]);  getitem_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_24: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_189, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_153: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_195, [1, 1536, -1]);  primals_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_84: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_153, unsqueeze_98);  view_153 = unsqueeze_98 = None
        mul_302: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_189, sub_84)
        sum_25: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 2]);  mul_302 = None
        mul_303: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 0.0026041666666666665);  sum_24 = None
        unsqueeze_99: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_303, 0);  mul_303 = None
        unsqueeze_100: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
        mul_304: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 0.0026041666666666665)
        mul_305: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_306: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
        unsqueeze_101: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_306, 0);  mul_306 = None
        unsqueeze_102: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_194: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.09125009274634042);  primals_196 = None
        view_154: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_194, [-1]);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_307: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, view_154);  view_154 = None
        unsqueeze_103: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
        unsqueeze_104: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 2);  unsqueeze_103 = None
        mul_308: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_102);  sub_84 = unsqueeze_102 = None
        sub_86: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_189, mul_308);  view_189 = mul_308 = None
        sub_87: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_100);  sub_86 = unsqueeze_100 = None
        mul_309: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_104);  sub_87 = unsqueeze_104 = None
        mul_310: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_103);  sum_25 = squeeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_190: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_310, [1536, 1, 1, 1]);  mul_310 = None
        mul_311: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_190, 0.09125009274634042);  view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_191: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_309, [1536, 384, 1, 1]);  mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_46: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convolution_70)
        exp_46: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_46);  neg_46 = None
        add_107: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_46, 1);  exp_46 = None
        reciprocal_5: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_107);  add_107 = None
        mul_312: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        mul_313: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_141, mul_312);  getitem_141 = None
        sub_88: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_312);  mul_312 = None
        mul_314: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_70, sub_88);  convolution_70 = sub_88 = None
        add_135: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_314, 1);  mul_314 = None
        mul_315: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_313, add_135);  mul_313 = add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_26: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_315, div_45, view_152, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_315 = div_45 = view_152 = None
        getitem_144: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_10[0]
        getitem_145: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_59: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_145, memory_format = torch.contiguous_format);  getitem_145 = None
        view_192: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [1, 384, 576]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_27: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_192, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_50: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_192, memory_format = torch.contiguous_format);  primals_192 = None
        view_150: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [1, 384, 576]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_89: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_106);  view_150 = unsqueeze_106 = None
        mul_316: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_192, sub_89)
        sum_28: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 2]);  mul_316 = None
        mul_317: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 0.001736111111111111);  sum_27 = None
        unsqueeze_107: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
        unsqueeze_108: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
        mul_318: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.001736111111111111)
        mul_319: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, squeeze_101)
        mul_320: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
        unsqueeze_109: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
        unsqueeze_110: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 2);  unsqueeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_191: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_193, 0.07450538873672485);  primals_193 = None
        view_151: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_191, [-1]);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_321: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, view_151);  view_151 = None
        unsqueeze_111: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_321, 0);  mul_321 = None
        unsqueeze_112: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
        mul_322: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_110);  sub_89 = unsqueeze_110 = None
        sub_91: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_192, mul_322);  view_192 = mul_322 = None
        sub_92: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_108);  sub_91 = unsqueeze_108 = None
        mul_323: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_112);  sub_92 = unsqueeze_112 = None
        mul_324: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, squeeze_101);  sum_28 = squeeze_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_193: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_324, [384, 1, 1, 1]);  mul_324 = None
        mul_325: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_193, 0.07450538873672485);  view_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_194: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_323, [384, 64, 3, 3]);  mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_45: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convolution_69)
        exp_45: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_45);  neg_45 = None
        add_105: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_45, 1);  exp_45 = None
        reciprocal_6: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_105);  add_105 = None
        mul_326: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        mul_327: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_144, mul_326);  getitem_144 = None
        sub_93: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_326);  mul_326 = None
        mul_328: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_69, sub_93);  convolution_69 = sub_93 = None
        add_137: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_328, 1);  mul_328 = None
        mul_329: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, add_137);  mul_327 = add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_29: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2, 3])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_329, div_44, view_149, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_329 = div_44 = view_149 = None
        getitem_147: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_11[0]
        getitem_148: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_60: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_148, memory_format = torch.contiguous_format);  getitem_148 = None
        view_195: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [1, 384, 576]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_30: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_195, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_48: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_189, memory_format = torch.contiguous_format);  primals_189 = None
        view_147: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [1, 384, 576]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_94: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_147, unsqueeze_114);  view_147 = unsqueeze_114 = None
        mul_330: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, sub_94)
        sum_31: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 2]);  mul_330 = None
        mul_331: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.001736111111111111);  sum_30 = None
        unsqueeze_115: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
        unsqueeze_116: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
        mul_332: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.001736111111111111)
        mul_333: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, squeeze_99)
        mul_334: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, mul_333);  mul_332 = mul_333 = None
        unsqueeze_117: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_334, 0);  mul_334 = None
        unsqueeze_118: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_188: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_190, 0.07450538873672485);  primals_190 = None
        view_148: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_188, [-1]);  mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_335: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, view_148);  view_148 = None
        unsqueeze_119: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
        unsqueeze_120: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
        mul_336: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_118);  sub_94 = unsqueeze_118 = None
        sub_96: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_195, mul_336);  view_195 = mul_336 = None
        sub_97: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_116);  sub_96 = unsqueeze_116 = None
        mul_337: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_120);  sub_97 = unsqueeze_120 = None
        mul_338: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_99);  sum_31 = squeeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_196: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_338, [384, 1, 1, 1]);  mul_338 = None
        mul_339: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_196, 0.07450538873672485);  view_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_197: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_337, [384, 64, 3, 3]);  mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_44: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convolution_68)
        exp_44: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_103: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_44, 1);  exp_44 = None
        reciprocal_7: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_103);  add_103 = None
        mul_340: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        mul_341: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_147, mul_340);  getitem_147 = None
        sub_98: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_340);  mul_340 = None
        mul_342: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_68, sub_98);  convolution_68 = sub_98 = None
        add_139: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_342, 1);  mul_342 = None
        mul_343: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_341, add_139);  mul_341 = add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_32: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_343, mul_184, view_146, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_343 = mul_184 = view_146 = None
        getitem_150: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = convolution_backward_12[0]
        getitem_151: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_198: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_151, [1, 384, 1536]);  getitem_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_33: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_198, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_144: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_186, [1, 384, -1]);  primals_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_99: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_122);  view_144 = unsqueeze_122 = None
        mul_344: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_198, sub_99)
        sum_34: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 2]);  mul_344 = None
        mul_345: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 0.0006510416666666666);  sum_33 = None
        unsqueeze_123: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_345, 0);  mul_345 = None
        unsqueeze_124: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
        mul_346: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 0.0006510416666666666)
        mul_347: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_348: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, mul_347);  mul_346 = mul_347 = None
        unsqueeze_125: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_348, 0);  mul_348 = None
        unsqueeze_126: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_185: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_187, 0.04562504637317021);  primals_187 = None
        view_145: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_185, [-1]);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_349: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, view_145);  view_145 = None
        unsqueeze_127: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_349, 0);  mul_349 = None
        unsqueeze_128: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
        mul_350: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_126);  sub_99 = unsqueeze_126 = None
        sub_101: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_198, mul_350);  view_198 = mul_350 = None
        sub_102: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_124);  sub_101 = unsqueeze_124 = None
        mul_351: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_128);  sub_102 = unsqueeze_128 = None
        mul_352: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, squeeze_97);  sum_34 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_199: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_352, [384, 1, 1, 1]);  mul_352 = None
        mul_353: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_199, 0.04562504637317021);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_200: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_351, [384, 1536, 1, 1]);  mul_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_354: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_150, 0.9805806756909201);  getitem_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_9: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_181: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_182: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, 2.0);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_183: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, 0.2);  mul_182 = None
        add_100: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_183, convolution_61);  mul_183 = convolution_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_43: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.neg.default(add_100)
        exp_43: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.exp.default(neg_43);  neg_43 = None
        add_101: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_43, 1);  exp_43 = None
        reciprocal_8: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.reciprocal.default(add_101);  add_101 = None
        mul_355: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        mul_356: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_354, mul_355);  mul_354 = None
        sub_103: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_355);  mul_355 = None
        mul_357: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_100, sub_103);  add_100 = sub_103 = None
        add_141: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_357, 1);  mul_357 = None
        mul_358: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, add_141);  mul_356 = add_141 = None
        add_142: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_132, mul_358);  add_132 = mul_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_359: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_142, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_360: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_359, 2.0);  mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_361: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, convolution_65);  convolution_65 = None
        mul_362: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, sigmoid_9);  mul_360 = None
        sum_35: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [2, 3], True);  mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_104: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_9)
        mul_363: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_9, sub_104);  sigmoid_9 = sub_104 = None
        mul_364: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, mul_363);  sum_35 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_36: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_364, relu_9, primals_184, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_364 = primals_184 = None
        getitem_153: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_13[0]
        getitem_154: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_2: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_2: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_2, full_default, getitem_153);  le_2 = getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_37: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_2, mean_9, primals_182, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = mean_9 = primals_182 = None
        getitem_156: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_14[0]
        getitem_157: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_4: "f32[128, 1536, 7, 7][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_156, [128, 1536, 7, 7]);  getitem_156 = None
        div_55: "f32[128, 1536, 7, 7][75264, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
        add_143: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_362, div_55);  mul_362 = div_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_38: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_143, [0, 2, 3])
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(add_143, div_42, view_143, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_143 = div_42 = view_143 = None
        getitem_159: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_15[0]
        getitem_160: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_201: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_160, [1, 1536, 384]);  getitem_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_39: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_201, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_141: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_179, [1, 1536, -1]);  primals_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_105: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_141, unsqueeze_130);  view_141 = unsqueeze_130 = None
        mul_365: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_201, sub_105)
        sum_40: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 2]);  mul_365 = None
        mul_366: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 0.0026041666666666665);  sum_39 = None
        unsqueeze_131: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_366, 0);  mul_366 = None
        unsqueeze_132: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
        mul_367: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 0.0026041666666666665)
        mul_368: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, squeeze_95)
        mul_369: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
        unsqueeze_133: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
        unsqueeze_134: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_178: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_180, 0.09125009274634042);  primals_180 = None
        view_142: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_178, [-1]);  mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_370: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, view_142);  view_142 = None
        unsqueeze_135: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
        unsqueeze_136: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
        mul_371: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_134);  sub_105 = unsqueeze_134 = None
        sub_107: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_201, mul_371);  view_201 = mul_371 = None
        sub_108: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_132);  sub_107 = unsqueeze_132 = None
        mul_372: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_136);  sub_108 = unsqueeze_136 = None
        mul_373: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, squeeze_95);  sum_40 = squeeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_202: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_373, [1536, 1, 1, 1]);  mul_373 = None
        mul_374: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_202, 0.09125009274634042);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_203: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_372, [1536, 384, 1, 1]);  mul_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_42: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convolution_64)
        exp_42: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_42);  neg_42 = None
        add_98: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_42, 1);  exp_42 = None
        reciprocal_9: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_98);  add_98 = None
        mul_375: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        mul_376: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_159, mul_375);  getitem_159 = None
        sub_109: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_375);  mul_375 = None
        mul_377: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_64, sub_109);  convolution_64 = sub_109 = None
        add_145: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_377, 1);  mul_377 = None
        mul_378: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_376, add_145);  mul_376 = add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_41: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_378, [0, 2, 3])
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_378, div_41, view_140, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_378 = div_41 = view_140 = None
        getitem_162: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_16[0]
        getitem_163: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_61: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_163, memory_format = torch.contiguous_format);  getitem_163 = None
        view_204: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [1, 384, 576]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_42: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_204, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_46: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_176, memory_format = torch.contiguous_format);  primals_176 = None
        view_138: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [1, 384, 576]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_110: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_138, unsqueeze_138);  view_138 = unsqueeze_138 = None
        mul_379: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_204, sub_110)
        sum_43: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 2]);  mul_379 = None
        mul_380: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 0.001736111111111111);  sum_42 = None
        unsqueeze_139: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
        unsqueeze_140: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        mul_381: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 0.001736111111111111)
        mul_382: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, squeeze_93)
        mul_383: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, mul_382);  mul_381 = mul_382 = None
        unsqueeze_141: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
        unsqueeze_142: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_175: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_177, 0.07450538873672485);  primals_177 = None
        view_139: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_175, [-1]);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_384: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, view_139);  view_139 = None
        unsqueeze_143: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_384, 0);  mul_384 = None
        unsqueeze_144: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
        mul_385: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_142);  sub_110 = unsqueeze_142 = None
        sub_112: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_204, mul_385);  view_204 = mul_385 = None
        sub_113: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_140);  sub_112 = unsqueeze_140 = None
        mul_386: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_144);  sub_113 = unsqueeze_144 = None
        mul_387: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_93);  sum_43 = squeeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_205: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_387, [384, 1, 1, 1]);  mul_387 = None
        mul_388: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_205, 0.07450538873672485);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_206: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_386, [384, 64, 3, 3]);  mul_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_41: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.neg.default(convolution_63)
        exp_41: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_96: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_41, 1);  exp_41 = None
        reciprocal_10: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_96);  add_96 = None
        mul_389: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        mul_390: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_162, mul_389);  getitem_162 = None
        sub_114: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_389);  mul_389 = None
        mul_391: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_63, sub_114);  convolution_63 = sub_114 = None
        add_147: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_391, 1);  mul_391 = None
        mul_392: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, add_147);  mul_390 = add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_44: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_392, div_40, view_137, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_392 = div_40 = view_137 = None
        getitem_165: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_17[0]
        getitem_166: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_62: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_166, memory_format = torch.contiguous_format);  getitem_166 = None
        view_207: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [1, 384, 576]);  clone_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_45: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_207, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_44: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_173, memory_format = torch.contiguous_format);  primals_173 = None
        view_135: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [1, 384, 576]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_115: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_135, unsqueeze_146);  view_135 = unsqueeze_146 = None
        mul_393: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_207, sub_115)
        sum_46: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 2]);  mul_393 = None
        mul_394: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 0.001736111111111111);  sum_45 = None
        unsqueeze_147: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_148: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
        mul_395: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 0.001736111111111111)
        mul_396: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_397: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, mul_396);  mul_395 = mul_396 = None
        unsqueeze_149: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
        unsqueeze_150: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_172: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_174, 0.07450538873672485);  primals_174 = None
        view_136: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_172, [-1]);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_398: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, view_136);  view_136 = None
        unsqueeze_151: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
        unsqueeze_152: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        mul_399: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_150);  sub_115 = unsqueeze_150 = None
        sub_117: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_207, mul_399);  view_207 = mul_399 = None
        sub_118: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_148);  sub_117 = unsqueeze_148 = None
        mul_400: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_152);  sub_118 = unsqueeze_152 = None
        mul_401: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, squeeze_91);  sum_46 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_208: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_401, [384, 1, 1, 1]);  mul_401 = None
        mul_402: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_208, 0.07450538873672485);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_209: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_400, [384, 64, 3, 3]);  mul_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_40: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_62)
        exp_40: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_40);  neg_40 = None
        add_94: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_40, 1);  exp_40 = None
        reciprocal_11: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_94);  add_94 = None
        mul_403: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        mul_404: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_165, mul_403);  getitem_165 = None
        sub_119: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_403);  mul_403 = None
        mul_405: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_62, sub_119);  convolution_62 = sub_119 = None
        add_149: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_405, 1);  mul_405 = None
        mul_406: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_404, add_149);  mul_404 = add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_47: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_406, mul_165, view_134, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = view_134 = None
        getitem_168: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_18[0]
        getitem_169: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_210: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_169, [1, 384, 1536]);  getitem_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_48: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_210, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_132: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_170, [1, 384, -1]);  primals_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_120: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_132, unsqueeze_154);  view_132 = unsqueeze_154 = None
        mul_407: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_210, sub_120)
        sum_49: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2]);  mul_407 = None
        mul_408: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 0.0006510416666666666);  sum_48 = None
        unsqueeze_155: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
        unsqueeze_156: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
        mul_409: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 0.0006510416666666666)
        mul_410: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, squeeze_89)
        mul_411: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
        unsqueeze_157: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
        unsqueeze_158: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_169: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_171, 0.04562504637317021);  primals_171 = None
        view_133: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_169, [-1]);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_412: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, view_133);  view_133 = None
        unsqueeze_159: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
        unsqueeze_160: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
        mul_413: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_158);  sub_120 = unsqueeze_158 = None
        sub_122: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_210, mul_413);  view_210 = mul_413 = None
        sub_123: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_156);  sub_122 = unsqueeze_156 = None
        mul_414: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_160);  sub_123 = unsqueeze_160 = None
        mul_415: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_89);  sum_49 = squeeze_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_211: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_415, [384, 1, 1, 1]);  mul_415 = None
        mul_416: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_211, 0.04562504637317021);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_212: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_414, [384, 1536, 1, 1]);  mul_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_50: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_142, [0, 2, 3])
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_142, avg_pool2d_2, view_131, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_142 = avg_pool2d_2 = view_131 = None
        getitem_171: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = convolution_backward_19[0]
        getitem_172: "f32[1536, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_213: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_172, [1, 1536, 1536]);  getitem_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_51: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_213, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_129: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_167, [1, 1536, -1]);  primals_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_124: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_129, unsqueeze_162);  view_129 = unsqueeze_162 = None
        mul_417: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_213, sub_124)
        sum_52: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2]);  mul_417 = None
        mul_418: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 0.0006510416666666666);  sum_51 = None
        unsqueeze_163: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
        unsqueeze_164: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        mul_419: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 0.0006510416666666666)
        mul_420: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, squeeze_87)
        mul_421: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
        unsqueeze_165: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
        unsqueeze_166: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_166: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_168, 0.04562504637317021);  primals_168 = None
        view_130: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_166, [-1]);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_422: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, view_130);  view_130 = None
        unsqueeze_167: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
        unsqueeze_168: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
        mul_423: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_166);  sub_124 = unsqueeze_166 = None
        sub_126: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_213, mul_423);  view_213 = mul_423 = None
        sub_127: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_164);  sub_126 = unsqueeze_164 = None
        mul_424: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_168);  sub_127 = unsqueeze_168 = None
        mul_425: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, squeeze_87);  sum_52 = squeeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_214: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_425, [1536, 1, 1, 1]);  mul_425 = None
        mul_426: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_214, 0.04562504637317021);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_215: "f32[1536, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_424, [1536, 1536, 1, 1]);  mul_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_backward: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_171, mul_165, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_171 = mul_165 = None
        add_150: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(getitem_168, avg_pool2d_backward);  getitem_168 = avg_pool2d_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_427: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_150, 0.8980265101338745);  add_150 = None
        neg_39: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(add_90)
        exp_39: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_39);  neg_39 = None
        add_91: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_39, 1);  exp_39 = None
        reciprocal_12: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.reciprocal.default(add_91);  add_91 = None
        mul_428: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        mul_429: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_427, mul_428);  mul_427 = None
        sub_128: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_428);  mul_428 = None
        mul_430: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_90, sub_128);  add_90 = sub_128 = None
        add_152: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_430, 1);  mul_430 = None
        mul_431: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_429, add_152);  mul_429 = add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_432: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_431, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_433: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, 2.0);  mul_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_434: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_433, convolution_58);  convolution_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_8: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_435: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_433, sigmoid_8);  mul_433 = None
        sum_53: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_434, [2, 3], True);  mul_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_129: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_8)
        mul_436: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_8, sub_129);  sigmoid_8 = sub_129 = None
        mul_437: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, mul_436);  sum_53 = mul_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_54: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3])
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_437, relu_8, primals_165, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_437 = primals_165 = None
        getitem_174: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_20[0]
        getitem_175: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_3: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_3: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_3, full_default, getitem_174);  le_3 = getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_55: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_3, mean_8, primals_163, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = mean_8 = primals_163 = None
        getitem_177: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_21[0]
        getitem_178: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_5: "f32[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_177, [128, 1536, 14, 14]);  getitem_177 = None
        div_56: "f32[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
        add_153: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_435, div_56);  mul_435 = div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_56: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_153, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_153, div_38, view_128, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_153 = div_38 = view_128 = None
        getitem_180: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_22[0]
        getitem_181: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_216: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_181, [1, 1536, 384]);  getitem_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_57: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_216, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_126: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_160, [1, 1536, -1]);  primals_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_130: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_170);  view_126 = unsqueeze_170 = None
        mul_438: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_216, sub_130)
        sum_58: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 2]);  mul_438 = None
        mul_439: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 0.0026041666666666665);  sum_57 = None
        unsqueeze_171: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
        unsqueeze_172: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
        mul_440: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 0.0026041666666666665)
        mul_441: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_442: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, mul_441);  mul_440 = mul_441 = None
        unsqueeze_173: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
        unsqueeze_174: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_159: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_161, 0.09125009274634042);  primals_161 = None
        view_127: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_159, [-1]);  mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_443: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, view_127);  view_127 = None
        unsqueeze_175: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
        unsqueeze_176: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        mul_444: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_174);  sub_130 = unsqueeze_174 = None
        sub_132: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_216, mul_444);  view_216 = mul_444 = None
        sub_133: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_172);  sub_132 = unsqueeze_172 = None
        mul_445: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_176);  sub_133 = unsqueeze_176 = None
        mul_446: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, squeeze_85);  sum_58 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_217: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_446, [1536, 1, 1, 1]);  mul_446 = None
        mul_447: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_217, 0.09125009274634042);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_218: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_445, [1536, 384, 1, 1]);  mul_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_38: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_57)
        exp_38: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_88: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_38, 1);  exp_38 = None
        reciprocal_13: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_88);  add_88 = None
        mul_448: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        mul_449: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_180, mul_448);  getitem_180 = None
        sub_134: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_448);  mul_448 = None
        mul_450: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_57, sub_134);  convolution_57 = sub_134 = None
        add_155: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_450, 1);  mul_450 = None
        mul_451: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_449, add_155);  mul_449 = add_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_59: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_451, div_37, view_125, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_451 = div_37 = view_125 = None
        getitem_183: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_23[0]
        getitem_184: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_63: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_184, memory_format = torch.contiguous_format);  getitem_184 = None
        view_219: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [1, 384, 576]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_60: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_219, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_42: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_157, memory_format = torch.contiguous_format);  primals_157 = None
        view_123: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [1, 384, 576]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_135: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_123, unsqueeze_178);  view_123 = unsqueeze_178 = None
        mul_452: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_219, sub_135)
        sum_61: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 2]);  mul_452 = None
        mul_453: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 0.001736111111111111);  sum_60 = None
        unsqueeze_179: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
        unsqueeze_180: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
        mul_454: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 0.001736111111111111)
        mul_455: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, squeeze_83)
        mul_456: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_454, mul_455);  mul_454 = mul_455 = None
        unsqueeze_181: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
        unsqueeze_182: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_156: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_158, 0.07450538873672485);  primals_158 = None
        view_124: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_156, [-1]);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_457: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, view_124);  view_124 = None
        unsqueeze_183: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
        unsqueeze_184: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
        mul_458: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_182);  sub_135 = unsqueeze_182 = None
        sub_137: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_219, mul_458);  view_219 = mul_458 = None
        sub_138: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_180);  sub_137 = unsqueeze_180 = None
        mul_459: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_184);  sub_138 = unsqueeze_184 = None
        mul_460: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_83);  sum_61 = squeeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_220: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_460, [384, 1, 1, 1]);  mul_460 = None
        mul_461: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_220, 0.07450538873672485);  view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_221: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_459, [384, 64, 3, 3]);  mul_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_37: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_56)
        exp_37: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_37);  neg_37 = None
        add_86: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_37, 1);  exp_37 = None
        reciprocal_14: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_86);  add_86 = None
        mul_462: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        mul_463: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_183, mul_462);  getitem_183 = None
        sub_139: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_462);  mul_462 = None
        mul_464: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_56, sub_139);  convolution_56 = sub_139 = None
        add_157: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_464, 1);  mul_464 = None
        mul_465: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_463, add_157);  mul_463 = add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_62: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 2, 3])
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_465, div_36, view_122, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_465 = div_36 = view_122 = None
        getitem_186: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_24[0]
        getitem_187: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_64: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_187, memory_format = torch.contiguous_format);  getitem_187 = None
        view_222: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [1, 384, 576]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_63: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_222, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_40: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_154, memory_format = torch.contiguous_format);  primals_154 = None
        view_120: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [1, 384, 576]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_140: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_186);  view_120 = unsqueeze_186 = None
        mul_466: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_222, sub_140)
        sum_64: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 2]);  mul_466 = None
        mul_467: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 0.001736111111111111);  sum_63 = None
        unsqueeze_187: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
        unsqueeze_188: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
        mul_468: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 0.001736111111111111)
        mul_469: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, squeeze_81)
        mul_470: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
        unsqueeze_189: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_470, 0);  mul_470 = None
        unsqueeze_190: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_153: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_155, 0.07450538873672485);  primals_155 = None
        view_121: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_153, [-1]);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_471: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, view_121);  view_121 = None
        unsqueeze_191: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
        unsqueeze_192: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
        mul_472: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_190);  sub_140 = unsqueeze_190 = None
        sub_142: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_222, mul_472);  view_222 = mul_472 = None
        sub_143: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_188);  sub_142 = unsqueeze_188 = None
        mul_473: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_192);  sub_143 = unsqueeze_192 = None
        mul_474: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, squeeze_81);  sum_64 = squeeze_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_223: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_474, [384, 1, 1, 1]);  mul_474 = None
        mul_475: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_223, 0.07450538873672485);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_224: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_473, [384, 64, 3, 3]);  mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_36: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_55)
        exp_36: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_36);  neg_36 = None
        add_84: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_36, 1);  exp_36 = None
        reciprocal_15: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_84);  add_84 = None
        mul_476: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        mul_477: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_186, mul_476);  getitem_186 = None
        sub_144: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_476);  mul_476 = None
        mul_478: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_55, sub_144);  convolution_55 = sub_144 = None
        add_159: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_478, 1);  mul_478 = None
        mul_479: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_477, add_159);  mul_477 = add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_65: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3])
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_479, mul_149, view_119, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_479 = mul_149 = view_119 = None
        getitem_189: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_25[0]
        getitem_190: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_225: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_190, [1, 384, 1536]);  getitem_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_66: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_225, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_117: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_151, [1, 384, -1]);  primals_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_145: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_117, unsqueeze_194);  view_117 = unsqueeze_194 = None
        mul_480: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_225, sub_145)
        sum_67: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 2]);  mul_480 = None
        mul_481: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 0.0006510416666666666);  sum_66 = None
        unsqueeze_195: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
        unsqueeze_196: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
        mul_482: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 0.0006510416666666666)
        mul_483: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_484: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_482, mul_483);  mul_482 = mul_483 = None
        unsqueeze_197: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_198: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_150: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_152, 0.04562504637317021);  primals_152 = None
        view_118: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_150, [-1]);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_485: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, view_118);  view_118 = None
        unsqueeze_199: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
        unsqueeze_200: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        mul_486: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_198);  sub_145 = unsqueeze_198 = None
        sub_147: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_225, mul_486);  view_225 = mul_486 = None
        sub_148: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_196);  sub_147 = unsqueeze_196 = None
        mul_487: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_200);  sub_148 = unsqueeze_200 = None
        mul_488: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_79);  sum_67 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_226: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_488, [384, 1, 1, 1]);  mul_488 = None
        mul_489: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_226, 0.04562504637317021);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_227: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_487, [384, 1536, 1, 1]);  mul_487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_490: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_189, 0.9128709291752768);  getitem_189 = None
        neg_35: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(add_81)
        exp_35: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_82: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_35, 1);  exp_35 = None
        reciprocal_16: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.reciprocal.default(add_82);  add_82 = None
        mul_491: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        mul_492: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_490, mul_491);  mul_490 = None
        sub_149: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_491);  mul_491 = None
        mul_493: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_81, sub_149);  add_81 = sub_149 = None
        add_161: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_493, 1);  mul_493 = None
        mul_494: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, add_161);  mul_492 = add_161 = None
        add_162: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_431, mul_494);  mul_431 = mul_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_495: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_162, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_496: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, 2.0);  mul_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_497: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_496, convolution_52);  convolution_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_7: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_54);  convolution_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_498: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_496, sigmoid_7);  mul_496 = None
        sum_68: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [2, 3], True);  mul_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_150: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_7)
        mul_499: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_7, sub_150);  sigmoid_7 = sub_150 = None
        mul_500: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, mul_499);  sum_68 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_69: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3])
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_500, relu_7, primals_149, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_500 = primals_149 = None
        getitem_192: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_26[0]
        getitem_193: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_4: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_4: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_192);  le_4 = getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_70: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_4, mean_7, primals_147, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = mean_7 = primals_147 = None
        getitem_195: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_27[0]
        getitem_196: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_6: "f32[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_195, [128, 1536, 14, 14]);  getitem_195 = None
        div_57: "f32[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
        add_163: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_498, div_57);  mul_498 = div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_71: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_163, [0, 2, 3])
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_163, div_34, view_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_163 = div_34 = view_116 = None
        getitem_198: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_28[0]
        getitem_199: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_228: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_199, [1, 1536, 384]);  getitem_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_72: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_228, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_114: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_144, [1, 1536, -1]);  primals_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_151: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_114, unsqueeze_202);  view_114 = unsqueeze_202 = None
        mul_501: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_228, sub_151)
        sum_73: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 2]);  mul_501 = None
        mul_502: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 0.0026041666666666665);  sum_72 = None
        unsqueeze_203: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
        unsqueeze_204: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        mul_503: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 0.0026041666666666665)
        mul_504: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, squeeze_77)
        mul_505: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_503, mul_504);  mul_503 = mul_504 = None
        unsqueeze_205: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
        unsqueeze_206: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_143: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_145, 0.09125009274634042);  primals_145 = None
        view_115: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_143, [-1]);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_506: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, view_115);  view_115 = None
        unsqueeze_207: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
        unsqueeze_208: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
        mul_507: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_206);  sub_151 = unsqueeze_206 = None
        sub_153: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_228, mul_507);  view_228 = mul_507 = None
        sub_154: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_204);  sub_153 = unsqueeze_204 = None
        mul_508: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_208);  sub_154 = unsqueeze_208 = None
        mul_509: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_77);  sum_73 = squeeze_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_229: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_509, [1536, 1, 1, 1]);  mul_509 = None
        mul_510: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_229, 0.09125009274634042);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_230: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_508, [1536, 384, 1, 1]);  mul_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_34: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_51)
        exp_34: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_34);  neg_34 = None
        add_79: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_34, 1);  exp_34 = None
        reciprocal_17: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_79);  add_79 = None
        mul_511: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        mul_512: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_198, mul_511);  getitem_198 = None
        sub_155: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_511);  mul_511 = None
        mul_513: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_51, sub_155);  convolution_51 = sub_155 = None
        add_165: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_513, 1);  mul_513 = None
        mul_514: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_512, add_165);  mul_512 = add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_74: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_514, [0, 2, 3])
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_514, div_33, view_113, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_514 = div_33 = view_113 = None
        getitem_201: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_29[0]
        getitem_202: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_65: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_202, memory_format = torch.contiguous_format);  getitem_202 = None
        view_231: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1, 384, 576]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_75: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_231, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_38: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_141, memory_format = torch.contiguous_format);  primals_141 = None
        view_111: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [1, 384, 576]);  clone_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_156: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_111, unsqueeze_210);  view_111 = unsqueeze_210 = None
        mul_515: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_231, sub_156)
        sum_76: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2]);  mul_515 = None
        mul_516: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 0.001736111111111111);  sum_75 = None
        unsqueeze_211: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
        unsqueeze_212: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        mul_517: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 0.001736111111111111)
        mul_518: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, squeeze_75)
        mul_519: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_517, mul_518);  mul_517 = mul_518 = None
        unsqueeze_213: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
        unsqueeze_214: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_140: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.07450538873672485);  primals_142 = None
        view_112: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_140, [-1]);  mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_520: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, view_112);  view_112 = None
        unsqueeze_215: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_216: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        mul_521: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_214);  sub_156 = unsqueeze_214 = None
        sub_158: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_231, mul_521);  view_231 = mul_521 = None
        sub_159: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_212);  sub_158 = unsqueeze_212 = None
        mul_522: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_216);  sub_159 = unsqueeze_216 = None
        mul_523: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, squeeze_75);  sum_76 = squeeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_232: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_523, [384, 1, 1, 1]);  mul_523 = None
        mul_524: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_232, 0.07450538873672485);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_233: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_522, [384, 64, 3, 3]);  mul_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_50)
        exp_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_77: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        reciprocal_18: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_77);  add_77 = None
        mul_525: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        mul_526: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_201, mul_525);  getitem_201 = None
        sub_160: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_525);  mul_525 = None
        mul_527: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_50, sub_160);  convolution_50 = sub_160 = None
        add_167: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_527, 1);  mul_527 = None
        mul_528: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_526, add_167);  mul_526 = add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_77: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3])
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_528, div_32, view_110, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_528 = div_32 = view_110 = None
        getitem_204: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_30[0]
        getitem_205: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_66: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_205, memory_format = torch.contiguous_format);  getitem_205 = None
        view_234: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [1, 384, 576]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_78: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_234, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_36: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_138, memory_format = torch.contiguous_format);  primals_138 = None
        view_108: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [1, 384, 576]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_161: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_108, unsqueeze_218);  view_108 = unsqueeze_218 = None
        mul_529: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_234, sub_161)
        sum_79: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 2]);  mul_529 = None
        mul_530: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 0.001736111111111111);  sum_78 = None
        unsqueeze_219: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
        unsqueeze_220: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
        mul_531: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 0.001736111111111111)
        mul_532: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_533: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_531, mul_532);  mul_531 = mul_532 = None
        unsqueeze_221: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
        unsqueeze_222: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_137: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_139, 0.07450538873672485);  primals_139 = None
        view_109: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_137, [-1]);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_534: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, view_109);  view_109 = None
        unsqueeze_223: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
        unsqueeze_224: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        mul_535: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_222);  sub_161 = unsqueeze_222 = None
        sub_163: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_234, mul_535);  view_234 = mul_535 = None
        sub_164: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_220);  sub_163 = unsqueeze_220 = None
        mul_536: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_224);  sub_164 = unsqueeze_224 = None
        mul_537: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_73);  sum_79 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_235: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_537, [384, 1, 1, 1]);  mul_537 = None
        mul_538: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_235, 0.07450538873672485);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_236: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_536, [384, 64, 3, 3]);  mul_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_32: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_49)
        exp_32: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_75: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        reciprocal_19: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_75);  add_75 = None
        mul_539: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        mul_540: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_204, mul_539);  getitem_204 = None
        sub_165: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_539);  mul_539 = None
        mul_541: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_49, sub_165);  convolution_49 = sub_165 = None
        add_169: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_541, 1);  mul_541 = None
        mul_542: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_540, add_169);  mul_540 = add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_80: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 2, 3])
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_542, mul_133, view_107, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_542 = mul_133 = view_107 = None
        getitem_207: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_31[0]
        getitem_208: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_237: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_208, [1, 384, 1536]);  getitem_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_81: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_237, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_105: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_135, [1, 384, -1]);  primals_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_166: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_105, unsqueeze_226);  view_105 = unsqueeze_226 = None
        mul_543: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_237, sub_166)
        sum_82: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 2]);  mul_543 = None
        mul_544: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 0.0006510416666666666);  sum_81 = None
        unsqueeze_227: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
        unsqueeze_228: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
        mul_545: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 0.0006510416666666666)
        mul_546: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, squeeze_71)
        mul_547: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
        unsqueeze_229: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
        unsqueeze_230: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_134: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.04562504637317021);  primals_136 = None
        view_106: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_134, [-1]);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_548: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, view_106);  view_106 = None
        unsqueeze_231: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_232: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
        mul_549: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_230);  sub_166 = unsqueeze_230 = None
        sub_168: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_237, mul_549);  view_237 = mul_549 = None
        sub_169: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_228);  sub_168 = unsqueeze_228 = None
        mul_550: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_232);  sub_169 = unsqueeze_232 = None
        mul_551: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, squeeze_71);  sum_82 = squeeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_238: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_551, [384, 1, 1, 1]);  mul_551 = None
        mul_552: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_238, 0.04562504637317021);  view_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_239: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_550, [384, 1536, 1, 1]);  mul_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_553: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_207, 0.9284766908852592);  getitem_207 = None
        neg_31: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(add_72)
        exp_31: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_73: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        reciprocal_20: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.reciprocal.default(add_73);  add_73 = None
        mul_554: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        mul_555: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_553, mul_554);  mul_553 = None
        sub_170: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_554);  mul_554 = None
        mul_556: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_72, sub_170);  add_72 = sub_170 = None
        add_171: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_556, 1);  mul_556 = None
        mul_557: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_555, add_171);  mul_555 = add_171 = None
        add_172: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_162, mul_557);  add_162 = mul_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_558: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_172, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_559: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_558, 2.0);  mul_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_560: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_559, convolution_46);  convolution_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_6: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_561: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_559, sigmoid_6);  mul_559 = None
        sum_83: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_560, [2, 3], True);  mul_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_171: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_6)
        mul_562: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_6, sub_171);  sigmoid_6 = sub_171 = None
        mul_563: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, mul_562);  sum_83 = mul_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_84: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3])
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_563, relu_6, primals_133, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_563 = primals_133 = None
        getitem_210: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_32[0]
        getitem_211: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_5: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_5: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_5, full_default, getitem_210);  le_5 = getitem_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_85: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_5, mean_6, primals_131, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_6 = primals_131 = None
        getitem_213: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_33[0]
        getitem_214: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_7: "f32[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_213, [128, 1536, 14, 14]);  getitem_213 = None
        div_58: "f32[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
        add_173: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_561, div_58);  mul_561 = div_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_86: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_173, [0, 2, 3])
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(add_173, div_30, view_104, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_173 = div_30 = view_104 = None
        getitem_216: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_34[0]
        getitem_217: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_240: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_217, [1, 1536, 384]);  getitem_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_87: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_240, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_102: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_128, [1, 1536, -1]);  primals_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_172: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_102, unsqueeze_234);  view_102 = unsqueeze_234 = None
        mul_564: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_240, sub_172)
        sum_88: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 2]);  mul_564 = None
        mul_565: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 0.0026041666666666665);  sum_87 = None
        unsqueeze_235: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
        unsqueeze_236: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        mul_566: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 0.0026041666666666665)
        mul_567: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, squeeze_69)
        mul_568: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
        unsqueeze_237: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
        unsqueeze_238: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_127: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_129, 0.09125009274634042);  primals_129 = None
        view_103: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_127, [-1]);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_569: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, view_103);  view_103 = None
        unsqueeze_239: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
        unsqueeze_240: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
        mul_570: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_238);  sub_172 = unsqueeze_238 = None
        sub_174: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_240, mul_570);  view_240 = mul_570 = None
        sub_175: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_236);  sub_174 = unsqueeze_236 = None
        mul_571: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_240);  sub_175 = unsqueeze_240 = None
        mul_572: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, squeeze_69);  sum_88 = squeeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_241: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_572, [1536, 1, 1, 1]);  mul_572 = None
        mul_573: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_241, 0.09125009274634042);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_242: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_571, [1536, 384, 1, 1]);  mul_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_30: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_45)
        exp_30: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_70: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        reciprocal_21: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_70);  add_70 = None
        mul_574: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        mul_575: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_216, mul_574);  getitem_216 = None
        sub_176: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_574);  mul_574 = None
        mul_576: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_45, sub_176);  convolution_45 = sub_176 = None
        add_175: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_576, 1);  mul_576 = None
        mul_577: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_575, add_175);  mul_575 = add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_89: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 2, 3])
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_577, div_29, view_101, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_577 = div_29 = view_101 = None
        getitem_219: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_35[0]
        getitem_220: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_67: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_220, memory_format = torch.contiguous_format);  getitem_220 = None
        view_243: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [1, 384, 576]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_90: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_243, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_34: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_125, memory_format = torch.contiguous_format);  primals_125 = None
        view_99: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [1, 384, 576]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_177: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_242);  view_99 = unsqueeze_242 = None
        mul_578: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_243, sub_177)
        sum_91: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 2]);  mul_578 = None
        mul_579: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 0.001736111111111111);  sum_90 = None
        unsqueeze_243: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
        unsqueeze_244: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
        mul_580: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 0.001736111111111111)
        mul_581: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_582: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_580, mul_581);  mul_580 = mul_581 = None
        unsqueeze_245: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
        unsqueeze_246: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_124: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_126, 0.07450538873672485);  primals_126 = None
        view_100: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_124, [-1]);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_583: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, view_100);  view_100 = None
        unsqueeze_247: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_248: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        mul_584: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_246);  sub_177 = unsqueeze_246 = None
        sub_179: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_243, mul_584);  view_243 = mul_584 = None
        sub_180: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_244);  sub_179 = unsqueeze_244 = None
        mul_585: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_248);  sub_180 = unsqueeze_248 = None
        mul_586: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_67);  sum_91 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_244: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_586, [384, 1, 1, 1]);  mul_586 = None
        mul_587: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_244, 0.07450538873672485);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_245: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_585, [384, 64, 3, 3]);  mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_29: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_44)
        exp_29: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_68: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        reciprocal_22: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_68);  add_68 = None
        mul_588: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        mul_589: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_219, mul_588);  getitem_219 = None
        sub_181: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_588);  mul_588 = None
        mul_590: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_44, sub_181);  convolution_44 = sub_181 = None
        add_177: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_590, 1);  mul_590 = None
        mul_591: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_589, add_177);  mul_589 = add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_92: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 2, 3])
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_591, div_28, view_98, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_591 = div_28 = view_98 = None
        getitem_222: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_36[0]
        getitem_223: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_68: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_223, memory_format = torch.contiguous_format);  getitem_223 = None
        view_246: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [1, 384, 576]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_93: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_246, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_32: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_122, memory_format = torch.contiguous_format);  primals_122 = None
        view_96: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [1, 384, 576]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_182: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_96, unsqueeze_250);  view_96 = unsqueeze_250 = None
        mul_592: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_246, sub_182)
        sum_94: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 2]);  mul_592 = None
        mul_593: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 0.001736111111111111);  sum_93 = None
        unsqueeze_251: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
        unsqueeze_252: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        mul_594: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 0.001736111111111111)
        mul_595: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, squeeze_65)
        mul_596: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_594, mul_595);  mul_594 = mul_595 = None
        unsqueeze_253: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
        unsqueeze_254: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_121: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_123, 0.07450538873672485);  primals_123 = None
        view_97: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_121, [-1]);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_597: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, view_97);  view_97 = None
        unsqueeze_255: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
        unsqueeze_256: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
        mul_598: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_254);  sub_182 = unsqueeze_254 = None
        sub_184: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_246, mul_598);  view_246 = mul_598 = None
        sub_185: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_252);  sub_184 = unsqueeze_252 = None
        mul_599: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_256);  sub_185 = unsqueeze_256 = None
        mul_600: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, squeeze_65);  sum_94 = squeeze_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_247: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_600, [384, 1, 1, 1]);  mul_600 = None
        mul_601: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_247, 0.07450538873672485);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_248: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_599, [384, 64, 3, 3]);  mul_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_28: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_43)
        exp_28: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_66: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        reciprocal_23: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_66);  add_66 = None
        mul_602: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        mul_603: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_222, mul_602);  getitem_222 = None
        sub_186: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_602);  mul_602 = None
        mul_604: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_43, sub_186);  convolution_43 = sub_186 = None
        add_179: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_604, 1);  mul_604 = None
        mul_605: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_603, add_179);  mul_603 = add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_95: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 2, 3])
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_605, mul_117, view_95, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_605 = mul_117 = view_95 = None
        getitem_225: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_37[0]
        getitem_226: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_249: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_226, [1, 384, 1536]);  getitem_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_96: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_249, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_93: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_119, [1, 384, -1]);  primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_187: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_93, unsqueeze_258);  view_93 = unsqueeze_258 = None
        mul_606: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_249, sub_187)
        sum_97: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 2]);  mul_606 = None
        mul_607: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 0.0006510416666666666);  sum_96 = None
        unsqueeze_259: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
        unsqueeze_260: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        mul_608: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 0.0006510416666666666)
        mul_609: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, squeeze_63)
        mul_610: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_608, mul_609);  mul_608 = mul_609 = None
        unsqueeze_261: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
        unsqueeze_262: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_118: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_120, 0.04562504637317021);  primals_120 = None
        view_94: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_118, [-1]);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_611: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, view_94);  view_94 = None
        unsqueeze_263: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
        unsqueeze_264: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        mul_612: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_262);  sub_187 = unsqueeze_262 = None
        sub_189: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_249, mul_612);  view_249 = mul_612 = None
        sub_190: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_260);  sub_189 = unsqueeze_260 = None
        mul_613: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_264);  sub_190 = unsqueeze_264 = None
        mul_614: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_63);  sum_97 = squeeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_250: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_614, [384, 1, 1, 1]);  mul_614 = None
        mul_615: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_250, 0.04562504637317021);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_251: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_613, [384, 1536, 1, 1]);  mul_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_616: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_225, 0.9449111825230679);  getitem_225 = None
        neg_27: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(add_63)
        exp_27: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_64: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        reciprocal_24: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.reciprocal.default(add_64);  add_64 = None
        mul_617: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        mul_618: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_616, mul_617);  mul_616 = None
        sub_191: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_617);  mul_617 = None
        mul_619: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_63, sub_191);  add_63 = sub_191 = None
        add_181: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_619, 1);  mul_619 = None
        mul_620: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_618, add_181);  mul_618 = add_181 = None
        add_182: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_172, mul_620);  add_172 = mul_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_621: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_182, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_622: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_621, 2.0);  mul_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_623: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_622, convolution_40);  convolution_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_5: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_624: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_622, sigmoid_5);  mul_622 = None
        sum_98: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_623, [2, 3], True);  mul_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_192: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_5)
        mul_625: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_5, sub_192);  sigmoid_5 = sub_192 = None
        mul_626: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, mul_625);  sum_98 = mul_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_99: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3])
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_626, relu_5, primals_117, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_626 = primals_117 = None
        getitem_228: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_38[0]
        getitem_229: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_6: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_6: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_6, full_default, getitem_228);  le_6 = getitem_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_100: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_6, mean_5, primals_115, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = mean_5 = primals_115 = None
        getitem_231: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_39[0]
        getitem_232: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_8: "f32[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_231, [128, 1536, 14, 14]);  getitem_231 = None
        div_59: "f32[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
        add_183: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_624, div_59);  mul_624 = div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_101: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_183, [0, 2, 3])
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(add_183, div_26, view_92, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_183 = div_26 = view_92 = None
        getitem_234: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_40[0]
        getitem_235: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_252: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_235, [1, 1536, 384]);  getitem_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_102: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_252, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_90: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_112, [1, 1536, -1]);  primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_193: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_90, unsqueeze_266);  view_90 = unsqueeze_266 = None
        mul_627: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_252, sub_193)
        sum_103: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2]);  mul_627 = None
        mul_628: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 0.0026041666666666665);  sum_102 = None
        unsqueeze_267: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
        unsqueeze_268: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
        mul_629: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 0.0026041666666666665)
        mul_630: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_631: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
        unsqueeze_269: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
        unsqueeze_270: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_111: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_113, 0.09125009274634042);  primals_113 = None
        view_91: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_111, [-1]);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_632: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, view_91);  view_91 = None
        unsqueeze_271: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
        unsqueeze_272: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        mul_633: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_270);  sub_193 = unsqueeze_270 = None
        sub_195: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_252, mul_633);  view_252 = mul_633 = None
        sub_196: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_268);  sub_195 = unsqueeze_268 = None
        mul_634: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_272);  sub_196 = unsqueeze_272 = None
        mul_635: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_61);  sum_103 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_253: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_635, [1536, 1, 1, 1]);  mul_635 = None
        mul_636: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_253, 0.09125009274634042);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_254: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_634, [1536, 384, 1, 1]);  mul_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_39)
        exp_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_61: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        reciprocal_25: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_61);  add_61 = None
        mul_637: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        mul_638: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_234, mul_637);  getitem_234 = None
        sub_197: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_637);  mul_637 = None
        mul_639: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_39, sub_197);  convolution_39 = sub_197 = None
        add_185: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_639, 1);  mul_639 = None
        mul_640: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_638, add_185);  mul_638 = add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_104: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 2, 3])
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_640, div_25, view_89, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_640 = div_25 = view_89 = None
        getitem_237: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_41[0]
        getitem_238: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_69: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_238, memory_format = torch.contiguous_format);  getitem_238 = None
        view_255: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [1, 384, 576]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_105: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_255, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_30: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_109, memory_format = torch.contiguous_format);  primals_109 = None
        view_87: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [1, 384, 576]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_198: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_274);  view_87 = unsqueeze_274 = None
        mul_641: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_255, sub_198)
        sum_106: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_641, [0, 2]);  mul_641 = None
        mul_642: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 0.001736111111111111);  sum_105 = None
        unsqueeze_275: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
        unsqueeze_276: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        mul_643: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 0.001736111111111111)
        mul_644: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, squeeze_59)
        mul_645: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_643, mul_644);  mul_643 = mul_644 = None
        unsqueeze_277: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
        unsqueeze_278: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_108: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_110, 0.07450538873672485);  primals_110 = None
        view_88: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_108, [-1]);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_646: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, view_88);  view_88 = None
        unsqueeze_279: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
        unsqueeze_280: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
        mul_647: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_278);  sub_198 = unsqueeze_278 = None
        sub_200: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_255, mul_647);  view_255 = mul_647 = None
        sub_201: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_276);  sub_200 = unsqueeze_276 = None
        mul_648: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_280);  sub_201 = unsqueeze_280 = None
        mul_649: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, squeeze_59);  sum_106 = squeeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_256: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_649, [384, 1, 1, 1]);  mul_649 = None
        mul_650: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_256, 0.07450538873672485);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_257: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_648, [384, 64, 3, 3]);  mul_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_25: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_38)
        exp_25: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_59: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        reciprocal_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_59);  add_59 = None
        mul_651: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        mul_652: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_237, mul_651);  getitem_237 = None
        sub_202: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_651);  mul_651 = None
        mul_653: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_38, sub_202);  convolution_38 = sub_202 = None
        add_187: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_653, 1);  mul_653 = None
        mul_654: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_652, add_187);  mul_652 = add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_107: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_654, [0, 2, 3])
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_654, div_24, view_86, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_654 = div_24 = view_86 = None
        getitem_240: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_42[0]
        getitem_241: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_70: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_241, memory_format = torch.contiguous_format);  getitem_241 = None
        view_258: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [1, 384, 576]);  clone_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_108: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_258, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_28: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_106, memory_format = torch.contiguous_format);  primals_106 = None
        view_84: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [1, 384, 576]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_203: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_84, unsqueeze_282);  view_84 = unsqueeze_282 = None
        mul_655: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_258, sub_203)
        sum_109: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_655, [0, 2]);  mul_655 = None
        mul_656: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 0.001736111111111111);  sum_108 = None
        unsqueeze_283: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
        unsqueeze_284: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        mul_657: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 0.001736111111111111)
        mul_658: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, squeeze_57)
        mul_659: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_657, mul_658);  mul_657 = mul_658 = None
        unsqueeze_285: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
        unsqueeze_286: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_105: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, 0.07450538873672485);  primals_107 = None
        view_85: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_105, [-1]);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_660: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, view_85);  view_85 = None
        unsqueeze_287: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
        unsqueeze_288: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        mul_661: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_286);  sub_203 = unsqueeze_286 = None
        sub_205: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_258, mul_661);  view_258 = mul_661 = None
        sub_206: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_284);  sub_205 = unsqueeze_284 = None
        mul_662: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_288);  sub_206 = unsqueeze_288 = None
        mul_663: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_57);  sum_109 = squeeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_259: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_663, [384, 1, 1, 1]);  mul_663 = None
        mul_664: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_259, 0.07450538873672485);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_260: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_662, [384, 64, 3, 3]);  mul_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_24: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_37)
        exp_24: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_57: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        reciprocal_27: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_57);  add_57 = None
        mul_665: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        mul_666: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_240, mul_665);  getitem_240 = None
        sub_207: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_665);  mul_665 = None
        mul_667: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_37, sub_207);  convolution_37 = sub_207 = None
        add_189: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_667, 1);  mul_667 = None
        mul_668: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_666, add_189);  mul_666 = add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_110: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 2, 3])
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_668, mul_101, view_83, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_668 = mul_101 = view_83 = None
        getitem_243: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_43[0]
        getitem_244: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_261: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_244, [1, 384, 1536]);  getitem_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_111: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_261, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_81: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_103, [1, 384, -1]);  primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_208: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_290);  view_81 = unsqueeze_290 = None
        mul_669: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_261, sub_208)
        sum_112: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_669, [0, 2]);  mul_669 = None
        mul_670: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 0.0006510416666666666);  sum_111 = None
        unsqueeze_291: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_670, 0);  mul_670 = None
        unsqueeze_292: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
        mul_671: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 0.0006510416666666666)
        mul_672: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_673: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_671, mul_672);  mul_671 = mul_672 = None
        unsqueeze_293: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
        unsqueeze_294: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_102: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_104, 0.04562504637317021);  primals_104 = None
        view_82: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_102, [-1]);  mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_674: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, view_82);  view_82 = None
        unsqueeze_295: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
        unsqueeze_296: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        mul_675: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_294);  sub_208 = unsqueeze_294 = None
        sub_210: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_261, mul_675);  view_261 = mul_675 = None
        sub_211: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_292);  sub_210 = unsqueeze_292 = None
        mul_676: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_296);  sub_211 = unsqueeze_296 = None
        mul_677: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, squeeze_55);  sum_112 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_262: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_677, [384, 1, 1, 1]);  mul_677 = None
        mul_678: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_262, 0.04562504637317021);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_263: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_676, [384, 1536, 1, 1]);  mul_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_679: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_243, 0.9622504486493761);  getitem_243 = None
        neg_23: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(add_54)
        exp_23: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_55: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        reciprocal_28: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.reciprocal.default(add_55);  add_55 = None
        mul_680: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        mul_681: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_679, mul_680);  mul_679 = None
        sub_212: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_680);  mul_680 = None
        mul_682: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_54, sub_212);  add_54 = sub_212 = None
        add_191: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_682, 1);  mul_682 = None
        mul_683: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_681, add_191);  mul_681 = add_191 = None
        add_192: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_182, mul_683);  add_182 = mul_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_684: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_192, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_685: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_684, 2.0);  mul_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_686: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_685, convolution_34);  convolution_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_4: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_687: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_685, sigmoid_4);  mul_685 = None
        sum_113: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_686, [2, 3], True);  mul_686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_213: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_4)
        mul_688: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_4, sub_213);  sigmoid_4 = sub_213 = None
        mul_689: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, mul_688);  sum_113 = mul_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_114: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3])
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_689, relu_4, primals_101, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_689 = primals_101 = None
        getitem_246: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_44[0]
        getitem_247: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_7: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_7: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_7, full_default, getitem_246);  le_7 = getitem_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_115: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_7, mean_4, primals_99, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = mean_4 = primals_99 = None
        getitem_249: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_45[0]
        getitem_250: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_9: "f32[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_249, [128, 1536, 14, 14]);  getitem_249 = None
        div_60: "f32[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
        add_193: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_687, div_60);  mul_687 = div_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_116: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_193, [0, 2, 3])
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(add_193, div_22, view_80, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_193 = div_22 = view_80 = None
        getitem_252: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_46[0]
        getitem_253: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_264: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_253, [1, 1536, 384]);  getitem_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_117: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_264, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_78: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_96, [1, 1536, -1]);  primals_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_214: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_78, unsqueeze_298);  view_78 = unsqueeze_298 = None
        mul_690: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_264, sub_214)
        sum_118: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_690, [0, 2]);  mul_690 = None
        mul_691: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 0.0026041666666666665);  sum_117 = None
        unsqueeze_299: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
        unsqueeze_300: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        mul_692: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 0.0026041666666666665)
        mul_693: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, squeeze_53)
        mul_694: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_692, mul_693);  mul_692 = mul_693 = None
        unsqueeze_301: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
        unsqueeze_302: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_95: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_97, 0.09125009274634042);  primals_97 = None
        view_79: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_95, [-1]);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_695: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, view_79);  view_79 = None
        unsqueeze_303: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
        unsqueeze_304: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        mul_696: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_302);  sub_214 = unsqueeze_302 = None
        sub_216: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_264, mul_696);  view_264 = mul_696 = None
        sub_217: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_300);  sub_216 = unsqueeze_300 = None
        mul_697: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_304);  sub_217 = unsqueeze_304 = None
        mul_698: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, squeeze_53);  sum_118 = squeeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_265: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_698, [1536, 1, 1, 1]);  mul_698 = None
        mul_699: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_265, 0.09125009274634042);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_266: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_697, [1536, 384, 1, 1]);  mul_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_22: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_33)
        exp_22: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_52: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        reciprocal_29: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_52);  add_52 = None
        mul_700: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        mul_701: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_252, mul_700);  getitem_252 = None
        sub_218: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_700);  mul_700 = None
        mul_702: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_33, sub_218);  convolution_33 = sub_218 = None
        add_195: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_702, 1);  mul_702 = None
        mul_703: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_701, add_195);  mul_701 = add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_119: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 2, 3])
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_703, div_21, view_77, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_703 = div_21 = view_77 = None
        getitem_255: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_47[0]
        getitem_256: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_71: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_256, memory_format = torch.contiguous_format);  getitem_256 = None
        view_267: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [1, 384, 576]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_120: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_267, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_26: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_93, memory_format = torch.contiguous_format);  primals_93 = None
        view_75: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [1, 384, 576]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_219: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_306);  view_75 = unsqueeze_306 = None
        mul_704: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_267, sub_219)
        sum_121: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2]);  mul_704 = None
        mul_705: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 0.001736111111111111);  sum_120 = None
        unsqueeze_307: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
        unsqueeze_308: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        mul_706: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 0.001736111111111111)
        mul_707: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, squeeze_51)
        mul_708: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_706, mul_707);  mul_706 = mul_707 = None
        unsqueeze_309: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
        unsqueeze_310: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_92: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.07450538873672485);  primals_94 = None
        view_76: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_92, [-1]);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_709: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, view_76);  view_76 = None
        unsqueeze_311: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
        unsqueeze_312: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        mul_710: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_310);  sub_219 = unsqueeze_310 = None
        sub_221: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_267, mul_710);  view_267 = mul_710 = None
        sub_222: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_308);  sub_221 = unsqueeze_308 = None
        mul_711: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_312);  sub_222 = unsqueeze_312 = None
        mul_712: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_51);  sum_121 = squeeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_268: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_712, [384, 1, 1, 1]);  mul_712 = None
        mul_713: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_268, 0.07450538873672485);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_269: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_711, [384, 64, 3, 3]);  mul_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_21: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_32)
        exp_21: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_50: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        reciprocal_30: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_50);  add_50 = None
        mul_714: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        mul_715: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_255, mul_714);  getitem_255 = None
        sub_223: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_714);  mul_714 = None
        mul_716: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_32, sub_223);  convolution_32 = sub_223 = None
        add_197: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_716, 1);  mul_716 = None
        mul_717: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_715, add_197);  mul_715 = add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_122: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3])
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_717, div_20, view_74, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_717 = div_20 = view_74 = None
        getitem_258: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_48[0]
        getitem_259: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_72: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_259, memory_format = torch.contiguous_format);  getitem_259 = None
        view_270: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [1, 384, 576]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_123: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_270, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_24: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_90, memory_format = torch.contiguous_format);  primals_90 = None
        view_72: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [1, 384, 576]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_224: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_72, unsqueeze_314);  view_72 = unsqueeze_314 = None
        mul_718: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_270, sub_224)
        sum_124: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_718, [0, 2]);  mul_718 = None
        mul_719: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, 0.001736111111111111);  sum_123 = None
        unsqueeze_315: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
        unsqueeze_316: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
        mul_720: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_124, 0.001736111111111111)
        mul_721: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_722: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_720, mul_721);  mul_720 = mul_721 = None
        unsqueeze_317: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
        unsqueeze_318: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_89: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_91, 0.07450538873672485);  primals_91 = None
        view_73: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_89, [-1]);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_723: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, view_73);  view_73 = None
        unsqueeze_319: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
        unsqueeze_320: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        mul_724: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_318);  sub_224 = unsqueeze_318 = None
        sub_226: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_270, mul_724);  view_270 = mul_724 = None
        sub_227: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_316);  sub_226 = unsqueeze_316 = None
        mul_725: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_320);  sub_227 = unsqueeze_320 = None
        mul_726: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_124, squeeze_49);  sum_124 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_271: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_726, [384, 1, 1, 1]);  mul_726 = None
        mul_727: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_271, 0.07450538873672485);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_272: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_725, [384, 64, 3, 3]);  mul_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_20: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_31)
        exp_20: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_48: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        reciprocal_31: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_48);  add_48 = None
        mul_728: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        mul_729: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_258, mul_728);  getitem_258 = None
        sub_228: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_728);  mul_728 = None
        mul_730: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_31, sub_228);  convolution_31 = sub_228 = None
        add_199: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_730, 1);  mul_730 = None
        mul_731: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_729, add_199);  mul_729 = add_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_125: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 2, 3])
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_731, mul_85, view_71, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_731 = mul_85 = view_71 = None
        getitem_261: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_49[0]
        getitem_262: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_273: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_262, [1, 384, 1536]);  getitem_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_126: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_273, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_69: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_87, [1, 384, -1]);  primals_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_229: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_69, unsqueeze_322);  view_69 = unsqueeze_322 = None
        mul_732: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_273, sub_229)
        sum_127: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2]);  mul_732 = None
        mul_733: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_126, 0.0006510416666666666);  sum_126 = None
        unsqueeze_323: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
        unsqueeze_324: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        mul_734: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, 0.0006510416666666666)
        mul_735: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, squeeze_47)
        mul_736: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_734, mul_735);  mul_734 = mul_735 = None
        unsqueeze_325: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
        unsqueeze_326: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_86: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.04562504637317021);  primals_88 = None
        view_70: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_86, [-1]);  mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_737: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, view_70);  view_70 = None
        unsqueeze_327: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
        unsqueeze_328: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
        mul_738: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_326);  sub_229 = unsqueeze_326 = None
        sub_231: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_273, mul_738);  view_273 = mul_738 = None
        sub_232: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_324);  sub_231 = unsqueeze_324 = None
        mul_739: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_328);  sub_232 = unsqueeze_328 = None
        mul_740: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, squeeze_47);  sum_127 = squeeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_274: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_740, [384, 1, 1, 1]);  mul_740 = None
        mul_741: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_274, 0.04562504637317021);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_275: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_739, [384, 1536, 1, 1]);  mul_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_742: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_261, 0.9805806756909201);  getitem_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_3: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_82: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_83: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, 2.0);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_84: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, 0.2);  mul_83 = None
        add_45: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_84, convolution_24);  mul_84 = convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_19: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.neg.default(add_45)
        exp_19: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_46: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        reciprocal_32: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.reciprocal.default(add_46);  add_46 = None
        mul_743: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        mul_744: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_742, mul_743);  mul_742 = None
        sub_233: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_743);  mul_743 = None
        mul_745: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_45, sub_233);  add_45 = sub_233 = None
        add_201: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_745, 1);  mul_745 = None
        mul_746: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_744, add_201);  mul_744 = add_201 = None
        add_202: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_192, mul_746);  add_192 = mul_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_747: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_202, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_748: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_747, 2.0);  mul_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_749: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_748, convolution_28);  convolution_28 = None
        mul_750: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_748, sigmoid_3);  mul_748 = None
        sum_128: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_749, [2, 3], True);  mul_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_234: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_3)
        mul_751: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_3, sub_234);  sigmoid_3 = sub_234 = None
        mul_752: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(sum_128, mul_751);  sum_128 = mul_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_129: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 2, 3])
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_752, relu_3, primals_85, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_752 = primals_85 = None
        getitem_264: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_50[0]
        getitem_265: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_8: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_8: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_8, full_default, getitem_264);  le_8 = getitem_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_130: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(where_8, mean_3, primals_83, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = mean_3 = primals_83 = None
        getitem_267: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_51[0]
        getitem_268: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_10: "f32[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_267, [128, 1536, 14, 14]);  getitem_267 = None
        div_61: "f32[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
        add_203: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_750, div_61);  mul_750 = div_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_131: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_203, [0, 2, 3])
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(add_203, div_18, view_68, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_203 = div_18 = view_68 = None
        getitem_270: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_52[0]
        getitem_271: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_276: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_271, [1, 1536, 384]);  getitem_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_132: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_276, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_66: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_80, [1, 1536, -1]);  primals_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_235: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_66, unsqueeze_330);  view_66 = unsqueeze_330 = None
        mul_753: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_276, sub_235)
        sum_133: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2]);  mul_753 = None
        mul_754: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_132, 0.0026041666666666665);  sum_132 = None
        unsqueeze_331: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
        unsqueeze_332: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        mul_755: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, 0.0026041666666666665)
        mul_756: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, squeeze_45)
        mul_757: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_755, mul_756);  mul_755 = mul_756 = None
        unsqueeze_333: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
        unsqueeze_334: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_79: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_81, 0.09125009274634042);  primals_81 = None
        view_67: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_79, [-1]);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_758: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, view_67);  view_67 = None
        unsqueeze_335: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
        unsqueeze_336: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        mul_759: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_334);  sub_235 = unsqueeze_334 = None
        sub_237: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_276, mul_759);  view_276 = mul_759 = None
        sub_238: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_332);  sub_237 = unsqueeze_332 = None
        mul_760: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_336);  sub_238 = unsqueeze_336 = None
        mul_761: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, squeeze_45);  sum_133 = squeeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_277: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_761, [1536, 1, 1, 1]);  mul_761 = None
        mul_762: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_277, 0.09125009274634042);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_278: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_760, [1536, 384, 1, 1]);  mul_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_18: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_27)
        exp_18: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_43: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        reciprocal_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_43);  add_43 = None
        mul_763: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        mul_764: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_270, mul_763);  getitem_270 = None
        sub_239: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_763);  mul_763 = None
        mul_765: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_27, sub_239);  convolution_27 = sub_239 = None
        add_205: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_765, 1);  mul_765 = None
        mul_766: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_764, add_205);  mul_764 = add_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_134: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 2, 3])
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_766, div_17, view_65, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_766 = div_17 = view_65 = None
        getitem_273: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_53[0]
        getitem_274: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_73: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_274, memory_format = torch.contiguous_format);  getitem_274 = None
        view_279: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [1, 384, 576]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_135: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_279, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_22: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_77, memory_format = torch.contiguous_format);  primals_77 = None
        view_63: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [1, 384, 576]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_240: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_338);  view_63 = unsqueeze_338 = None
        mul_767: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_279, sub_240)
        sum_136: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_767, [0, 2]);  mul_767 = None
        mul_768: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, 0.001736111111111111);  sum_135 = None
        unsqueeze_339: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
        unsqueeze_340: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
        mul_769: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, 0.001736111111111111)
        mul_770: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_771: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_769, mul_770);  mul_769 = mul_770 = None
        unsqueeze_341: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
        unsqueeze_342: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_76: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_78, 0.07450538873672485);  primals_78 = None
        view_64: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_76, [-1]);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_772: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, view_64);  view_64 = None
        unsqueeze_343: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
        unsqueeze_344: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        mul_773: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_342);  sub_240 = unsqueeze_342 = None
        sub_242: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_279, mul_773);  view_279 = mul_773 = None
        sub_243: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_340);  sub_242 = unsqueeze_340 = None
        mul_774: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_344);  sub_243 = unsqueeze_344 = None
        mul_775: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, squeeze_43);  sum_136 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_280: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_775, [384, 1, 1, 1]);  mul_775 = None
        mul_776: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_280, 0.07450538873672485);  view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_281: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_774, [384, 64, 3, 3]);  mul_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_17: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.neg.default(convolution_26)
        exp_17: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_41: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        reciprocal_34: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_41);  add_41 = None
        mul_777: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        mul_778: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_273, mul_777);  getitem_273 = None
        sub_244: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_777);  mul_777 = None
        mul_779: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_26, sub_244);  convolution_26 = sub_244 = None
        add_207: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_779, 1);  mul_779 = None
        mul_780: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_778, add_207);  mul_778 = add_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_137: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 2, 3])
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_780, div_16, view_62, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  mul_780 = div_16 = view_62 = None
        getitem_276: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_54[0]
        getitem_277: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_74: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_277, memory_format = torch.contiguous_format);  getitem_277 = None
        view_282: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [1, 384, 576]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_138: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_282, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_20: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_74, memory_format = torch.contiguous_format);  primals_74 = None
        view_60: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [1, 384, 576]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_245: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_60, unsqueeze_346);  view_60 = unsqueeze_346 = None
        mul_781: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_282, sub_245)
        sum_139: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 2]);  mul_781 = None
        mul_782: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, 0.001736111111111111);  sum_138 = None
        unsqueeze_347: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
        unsqueeze_348: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
        mul_783: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, 0.001736111111111111)
        mul_784: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, squeeze_41)
        mul_785: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_783, mul_784);  mul_783 = mul_784 = None
        unsqueeze_349: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
        unsqueeze_350: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_73: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_75, 0.07450538873672485);  primals_75 = None
        view_61: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_73, [-1]);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_786: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, view_61);  view_61 = None
        unsqueeze_351: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
        unsqueeze_352: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
        mul_787: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_350);  sub_245 = unsqueeze_350 = None
        sub_247: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_282, mul_787);  view_282 = mul_787 = None
        sub_248: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_348);  sub_247 = unsqueeze_348 = None
        mul_788: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_352);  sub_248 = unsqueeze_352 = None
        mul_789: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, squeeze_41);  sum_139 = squeeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_283: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_789, [384, 1, 1, 1]);  mul_789 = None
        mul_790: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_283, 0.07450538873672485);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_284: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_788, [384, 64, 3, 3]);  mul_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_16: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.neg.default(convolution_25)
        exp_16: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_39: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        reciprocal_35: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_39);  add_39 = None
        mul_791: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        mul_792: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_276, mul_791);  getitem_276 = None
        sub_249: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_791);  mul_791 = None
        mul_793: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convolution_25, sub_249);  convolution_25 = sub_249 = None
        add_209: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_793, 1);  mul_793 = None
        mul_794: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_792, add_209);  mul_792 = add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_140: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 2, 3])
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_794, mul_66, view_59, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_794 = view_59 = None
        getitem_279: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_55[0]
        getitem_280: "f32[384, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_285: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_280, [1, 384, 512]);  getitem_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_141: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_285, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_57: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.reshape.default(primals_71, [1, 384, -1]);  primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_250: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_57, unsqueeze_354);  view_57 = unsqueeze_354 = None
        mul_795: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_285, sub_250)
        sum_142: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_795, [0, 2]);  mul_795 = None
        mul_796: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, 0.001953125);  sum_141 = None
        unsqueeze_355: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
        unsqueeze_356: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        mul_797: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, 0.001953125)
        mul_798: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, squeeze_39)
        mul_799: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_797, mul_798);  mul_797 = mul_798 = None
        unsqueeze_357: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
        unsqueeze_358: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_70: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_72, 0.07902489841601695);  primals_72 = None
        view_58: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_70, [-1]);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_800: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, view_58);  view_58 = None
        unsqueeze_359: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
        unsqueeze_360: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
        mul_801: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_358);  sub_250 = unsqueeze_358 = None
        sub_252: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_285, mul_801);  view_285 = mul_801 = None
        sub_253: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_356);  sub_252 = unsqueeze_356 = None
        mul_802: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_360);  sub_253 = unsqueeze_360 = None
        mul_803: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, squeeze_39);  sum_142 = squeeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_286: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_803, [384, 1, 1, 1]);  mul_803 = None
        mul_804: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_286, 0.07902489841601695);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_287: "f32[384, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_802, [384, 512, 1, 1]);  mul_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_143: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_202, [0, 2, 3])
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(add_202, avg_pool2d_1, view_56, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_202 = avg_pool2d_1 = view_56 = None
        getitem_282: "f32[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = convolution_backward_56[0]
        getitem_283: "f32[1536, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_288: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_283, [1, 1536, 512]);  getitem_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_144: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_288, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_54: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.reshape.default(primals_68, [1, 1536, -1]);  primals_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_254: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_54, unsqueeze_362);  view_54 = unsqueeze_362 = None
        mul_805: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_288, sub_254)
        sum_145: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2]);  mul_805 = None
        mul_806: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_144, 0.001953125);  sum_144 = None
        unsqueeze_363: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_364: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
        mul_807: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, 0.001953125)
        mul_808: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_809: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
        unsqueeze_365: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
        unsqueeze_366: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_67: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_69, 0.07902489841601695);  primals_69 = None
        view_55: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [-1]);  mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_810: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, view_55);  view_55 = None
        unsqueeze_367: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
        unsqueeze_368: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        mul_811: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_366);  sub_254 = unsqueeze_366 = None
        sub_256: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_288, mul_811);  view_288 = mul_811 = None
        sub_257: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_256, unsqueeze_364);  sub_256 = unsqueeze_364 = None
        mul_812: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_368);  sub_257 = unsqueeze_368 = None
        mul_813: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, squeeze_37);  sum_145 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_289: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_813, [1536, 1, 1, 1]);  mul_813 = None
        mul_814: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_289, 0.07902489841601695);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_290: "f32[1536, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_812, [1536, 512, 1, 1]);  mul_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_backward_1: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_282, mul_66, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_282 = mul_66 = None
        add_210: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(getitem_279, avg_pool2d_backward_1);  getitem_279 = avg_pool2d_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_815: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(add_210, 0.9622504486493761);  add_210 = None
        neg_15: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.neg.default(add_35)
        exp_15: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_36: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        reciprocal_36: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.reciprocal.default(add_36);  add_36 = None
        mul_816: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        mul_817: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_815, mul_816);  mul_815 = None
        sub_258: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_816);  mul_816 = None
        mul_818: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(add_35, sub_258);  add_35 = sub_258 = None
        add_212: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_818, 1);  mul_818 = None
        mul_819: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_817, add_212);  mul_817 = add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_820: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_819, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_821: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_820, 2.0);  mul_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_822: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_821, convolution_21);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_2: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_823: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_821, sigmoid_2);  mul_821 = None
        sum_146: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_822, [2, 3], True);  mul_822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_259: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_2)
        mul_824: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_2, sub_259);  sigmoid_2 = sub_259 = None
        mul_825: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.mul.Tensor(sum_146, mul_824);  sum_146 = mul_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_147: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_825, [0, 2, 3])
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_825, relu_2, primals_66, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_825 = primals_66 = None
        getitem_285: "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_57[0]
        getitem_286: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_57[1];  convolution_backward_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_9: "b8[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_9: "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.where.self(le_9, full_default, getitem_285);  le_9 = getitem_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_148: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(where_9, mean_2, primals_64, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_2 = primals_64 = None
        getitem_288: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_58[0]
        getitem_289: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_58[1];  convolution_backward_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_11: "f32[128, 512, 28, 28][512, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_288, [128, 512, 28, 28]);  getitem_288 = None
        div_62: "f32[128, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_11, 784);  expand_11 = None
        add_213: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_823, div_62);  mul_823 = div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_149: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_213, [0, 2, 3])
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_213, div_14, view_53, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_213 = div_14 = view_53 = None
        getitem_291: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_59[0]
        getitem_292: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_59[1];  convolution_backward_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_291: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_292, [1, 512, 128]);  getitem_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_150: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_291, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_51: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_61, [1, 512, -1]);  primals_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_260: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_370);  view_51 = unsqueeze_370 = None
        mul_826: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_291, sub_260)
        sum_151: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_826, [0, 2]);  mul_826 = None
        mul_827: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_150, 0.0078125);  sum_150 = None
        unsqueeze_371: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
        unsqueeze_372: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
        mul_828: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, 0.0078125)
        mul_829: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, squeeze_35)
        mul_830: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_828, mul_829);  mul_828 = mul_829 = None
        unsqueeze_373: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
        unsqueeze_374: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_60: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_62, 0.1580497968320339);  primals_62 = None
        view_52: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(mul_60, [-1]);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_831: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, view_52);  view_52 = None
        unsqueeze_375: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
        unsqueeze_376: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
        mul_832: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_374);  sub_260 = unsqueeze_374 = None
        sub_262: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_291, mul_832);  view_291 = mul_832 = None
        sub_263: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_372);  sub_262 = unsqueeze_372 = None
        mul_833: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_376);  sub_263 = unsqueeze_376 = None
        mul_834: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, squeeze_35);  sum_151 = squeeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_292: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_834, [512, 1, 1, 1]);  mul_834 = None
        mul_835: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_292, 0.1580497968320339);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_293: "f32[512, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_833, [512, 128, 1, 1]);  mul_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_14: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convolution_20)
        exp_14: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_33: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        reciprocal_37: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_33);  add_33 = None
        mul_836: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        mul_837: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_291, mul_836);  getitem_291 = None
        sub_264: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_836);  mul_836 = None
        mul_838: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convolution_20, sub_264);  convolution_20 = sub_264 = None
        add_215: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_838, 1);  mul_838 = None
        mul_839: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_837, add_215);  mul_837 = add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_152: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_839, [0, 2, 3])
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_839, div_13, view_50, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_839 = div_13 = view_50 = None
        getitem_294: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_60[0]
        getitem_295: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_60[1];  convolution_backward_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_75: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_295, memory_format = torch.contiguous_format);  getitem_295 = None
        view_294: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [1, 128, 576]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_153: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_294, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_18: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_58, memory_format = torch.contiguous_format);  primals_58 = None
        view_48: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [1, 128, 576]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_265: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_48, unsqueeze_378);  view_48 = unsqueeze_378 = None
        mul_840: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_294, sub_265)
        sum_154: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 2]);  mul_840 = None
        mul_841: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, 0.001736111111111111);  sum_153 = None
        unsqueeze_379: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
        unsqueeze_380: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        mul_842: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, 0.001736111111111111)
        mul_843: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, squeeze_33)
        mul_844: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_842, mul_843);  mul_842 = mul_843 = None
        unsqueeze_381: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
        unsqueeze_382: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_57: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.07450538873672485);  primals_59 = None
        view_49: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_57, [-1]);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_845: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, view_49);  view_49 = None
        unsqueeze_383: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
        unsqueeze_384: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
        mul_846: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_382);  sub_265 = unsqueeze_382 = None
        sub_267: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_294, mul_846);  view_294 = mul_846 = None
        sub_268: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_380);  sub_267 = unsqueeze_380 = None
        mul_847: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_384);  sub_268 = unsqueeze_384 = None
        mul_848: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, squeeze_33);  sum_154 = squeeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_295: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_848, [128, 1, 1, 1]);  mul_848 = None
        mul_849: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_295, 0.07450538873672485);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_296: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_847, [128, 64, 3, 3]);  mul_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_13: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convolution_19)
        exp_13: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_31: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        reciprocal_38: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_31);  add_31 = None
        mul_850: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        mul_851: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_294, mul_850);  getitem_294 = None
        sub_269: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_850);  mul_850 = None
        mul_852: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convolution_19, sub_269);  convolution_19 = sub_269 = None
        add_217: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_852, 1);  mul_852 = None
        mul_853: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_851, add_217);  mul_851 = add_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_155: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_853, [0, 2, 3])
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_853, div_12, view_47, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_853 = div_12 = view_47 = None
        getitem_297: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_61[0]
        getitem_298: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_61[1];  convolution_backward_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_76: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_298, memory_format = torch.contiguous_format);  getitem_298 = None
        view_297: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [1, 128, 576]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_156: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_297, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_16: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_55, memory_format = torch.contiguous_format);  primals_55 = None
        view_45: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [1, 128, 576]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_270: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_45, unsqueeze_386);  view_45 = unsqueeze_386 = None
        mul_854: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_297, sub_270)
        sum_157: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_854, [0, 2]);  mul_854 = None
        mul_855: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_156, 0.001736111111111111);  sum_156 = None
        unsqueeze_387: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
        unsqueeze_388: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
        mul_856: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, 0.001736111111111111)
        mul_857: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_858: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_856, mul_857);  mul_856 = mul_857 = None
        unsqueeze_389: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
        unsqueeze_390: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_54: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_56, 0.07450538873672485);  primals_56 = None
        view_46: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_54, [-1]);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_859: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, view_46);  view_46 = None
        unsqueeze_391: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
        unsqueeze_392: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        mul_860: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_390);  sub_270 = unsqueeze_390 = None
        sub_272: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_297, mul_860);  view_297 = mul_860 = None
        sub_273: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_388);  sub_272 = unsqueeze_388 = None
        mul_861: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_392);  sub_273 = unsqueeze_392 = None
        mul_862: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, squeeze_31);  sum_157 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_298: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_862, [128, 1, 1, 1]);  mul_862 = None
        mul_863: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_298, 0.07450538873672485);  view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_299: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_861, [128, 64, 3, 3]);  mul_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_12: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convolution_18)
        exp_12: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_29: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        reciprocal_39: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_29);  add_29 = None
        mul_864: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        mul_865: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_297, mul_864);  getitem_297 = None
        sub_274: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_864);  mul_864 = None
        mul_866: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convolution_18, sub_274);  convolution_18 = sub_274 = None
        add_219: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_866, 1);  mul_866 = None
        mul_867: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_865, add_219);  mul_865 = add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_158: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2, 3])
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_867, mul_50, view_44, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_867 = mul_50 = view_44 = None
        getitem_300: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_62[0]
        getitem_301: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_62[1];  convolution_backward_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_300: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_301, [1, 128, 512]);  getitem_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_159: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_300, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_42: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(primals_52, [1, 128, -1]);  primals_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_275: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_42, unsqueeze_394);  view_42 = unsqueeze_394 = None
        mul_868: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_300, sub_275)
        sum_160: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_868, [0, 2]);  mul_868 = None
        mul_869: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, 0.001953125);  sum_159 = None
        unsqueeze_395: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
        unsqueeze_396: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
        mul_870: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, 0.001953125)
        mul_871: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, squeeze_29)
        mul_872: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_870, mul_871);  mul_870 = mul_871 = None
        unsqueeze_397: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
        unsqueeze_398: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_51: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.07902489841601695);  primals_53 = None
        view_43: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_51, [-1]);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_873: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, view_43);  view_43 = None
        unsqueeze_399: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
        unsqueeze_400: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
        mul_874: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_398);  sub_275 = unsqueeze_398 = None
        sub_277: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_300, mul_874);  view_300 = mul_874 = None
        sub_278: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_396);  sub_277 = unsqueeze_396 = None
        mul_875: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_400);  sub_278 = unsqueeze_400 = None
        mul_876: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, squeeze_29);  sum_160 = squeeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_301: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_876, [128, 1, 1, 1]);  mul_876 = None
        mul_877: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_301, 0.07902489841601695);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_302: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_875, [128, 512, 1, 1]);  mul_875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_878: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(getitem_300, 0.9805806756909201);  getitem_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_1: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_47: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_48: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, 2.0);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_49: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 0.2);  mul_48 = None
        add_26: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_49, convolution_11);  mul_49 = convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_11: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.neg.default(add_26)
        exp_11: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_27: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        reciprocal_40: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.reciprocal.default(add_27);  add_27 = None
        mul_879: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        mul_880: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_878, mul_879);  mul_878 = None
        sub_279: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_879);  mul_879 = None
        mul_881: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(add_26, sub_279);  add_26 = sub_279 = None
        add_221: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_881, 1);  mul_881 = None
        mul_882: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_880, add_221);  mul_880 = add_221 = None
        add_222: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_819, mul_882);  mul_819 = mul_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_883: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(add_222, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_884: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_883, 2.0);  mul_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_885: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_884, convolution_15);  convolution_15 = None
        mul_886: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_884, sigmoid_1);  mul_884 = None
        sum_161: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_885, [2, 3], True);  mul_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_280: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_1)
        mul_887: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid_1, sub_280);  sigmoid_1 = sub_280 = None
        mul_888: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, mul_887);  sum_161 = mul_887 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_162: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_888, [0, 2, 3])
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_888, relu_1, primals_50, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_888 = primals_50 = None
        getitem_303: "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_63[0]
        getitem_304: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_63[1];  convolution_backward_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_10: "b8[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_10: "f32[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.where.self(le_10, full_default, getitem_303);  le_10 = getitem_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_163: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_10, mean_1, primals_48, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = mean_1 = primals_48 = None
        getitem_306: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_64[0]
        getitem_307: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_64[1];  convolution_backward_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_12: "f32[128, 512, 28, 28][512, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_306, [128, 512, 28, 28]);  getitem_306 = None
        div_63: "f32[128, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_12, 784);  expand_12 = None
        add_223: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_886, div_63);  mul_886 = div_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_164: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_223, [0, 2, 3])
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_223, div_10, view_41, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_223 = div_10 = view_41 = None
        getitem_309: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_65[0]
        getitem_310: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_65[1];  convolution_backward_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_303: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_310, [1, 512, 128]);  getitem_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_165: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_39: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_45, [1, 512, -1]);  primals_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_281: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_402);  view_39 = unsqueeze_402 = None
        mul_889: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_303, sub_281)
        sum_166: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_889, [0, 2]);  mul_889 = None
        mul_890: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, 0.0078125);  sum_165 = None
        unsqueeze_403: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
        unsqueeze_404: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        mul_891: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, 0.0078125)
        mul_892: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, squeeze_27)
        mul_893: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_891, mul_892);  mul_891 = mul_892 = None
        unsqueeze_405: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
        unsqueeze_406: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_44: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.1580497968320339);  primals_46 = None
        view_40: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(mul_44, [-1]);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_894: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, view_40);  view_40 = None
        unsqueeze_407: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_894, 0);  mul_894 = None
        unsqueeze_408: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
        mul_895: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_406);  sub_281 = unsqueeze_406 = None
        sub_283: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_303, mul_895);  view_303 = mul_895 = None
        sub_284: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_404);  sub_283 = unsqueeze_404 = None
        mul_896: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_408);  sub_284 = unsqueeze_408 = None
        mul_897: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, squeeze_27);  sum_166 = squeeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_304: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_897, [512, 1, 1, 1]);  mul_897 = None
        mul_898: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_304, 0.1580497968320339);  view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_305: "f32[512, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_896, [512, 128, 1, 1]);  mul_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_10: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convolution_14)
        exp_10: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_24: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        reciprocal_41: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_24);  add_24 = None
        mul_899: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        mul_900: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_309, mul_899);  getitem_309 = None
        sub_285: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_899);  mul_899 = None
        mul_901: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convolution_14, sub_285);  convolution_14 = sub_285 = None
        add_225: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_901, 1);  mul_901 = None
        mul_902: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_900, add_225);  mul_900 = add_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_167: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_902, [0, 2, 3])
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_902, div_9, view_38, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_902 = div_9 = view_38 = None
        getitem_312: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_66[0]
        getitem_313: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_66[1];  convolution_backward_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_77: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_313, memory_format = torch.contiguous_format);  getitem_313 = None
        view_306: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [1, 128, 576]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_168: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_306, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_14: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_42, memory_format = torch.contiguous_format);  primals_42 = None
        view_36: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [1, 128, 576]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_286: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_36, unsqueeze_410);  view_36 = unsqueeze_410 = None
        mul_903: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_306, sub_286)
        sum_169: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_903, [0, 2]);  mul_903 = None
        mul_904: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_168, 0.001736111111111111);  sum_168 = None
        unsqueeze_411: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
        unsqueeze_412: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
        mul_905: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, 0.001736111111111111)
        mul_906: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_907: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_905, mul_906);  mul_905 = mul_906 = None
        unsqueeze_413: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_907, 0);  mul_907 = None
        unsqueeze_414: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_41: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_43, 0.07450538873672485);  primals_43 = None
        view_37: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_41, [-1]);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_908: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, view_37);  view_37 = None
        unsqueeze_415: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
        unsqueeze_416: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        mul_909: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_414);  sub_286 = unsqueeze_414 = None
        sub_288: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_306, mul_909);  view_306 = mul_909 = None
        sub_289: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_412);  sub_288 = unsqueeze_412 = None
        mul_910: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_416);  sub_289 = unsqueeze_416 = None
        mul_911: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, squeeze_25);  sum_169 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_307: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_911, [128, 1, 1, 1]);  mul_911 = None
        mul_912: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_307, 0.07450538873672485);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_308: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_910, [128, 64, 3, 3]);  mul_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_9: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.neg.default(convolution_13)
        exp_9: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_22: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        reciprocal_42: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_22);  add_22 = None
        mul_913: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        mul_914: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_312, mul_913);  getitem_312 = None
        sub_290: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_913);  mul_913 = None
        mul_915: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convolution_13, sub_290);  convolution_13 = sub_290 = None
        add_227: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_915, 1);  mul_915 = None
        mul_916: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_914, add_227);  mul_914 = add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_170: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_916, [0, 2, 3])
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_916, div_8, view_35, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  mul_916 = div_8 = view_35 = None
        getitem_315: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = convolution_backward_67[0]
        getitem_316: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_67[1];  convolution_backward_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_78: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_316, memory_format = torch.contiguous_format);  getitem_316 = None
        view_309: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [1, 128, 576]);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_171: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_309, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_12: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_39, memory_format = torch.contiguous_format);  primals_39 = None
        view_33: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [1, 128, 576]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_291: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_418);  view_33 = unsqueeze_418 = None
        mul_917: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_309, sub_291)
        sum_172: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 2]);  mul_917 = None
        mul_918: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, 0.001736111111111111);  sum_171 = None
        unsqueeze_419: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
        unsqueeze_420: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
        mul_919: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, 0.001736111111111111)
        mul_920: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, squeeze_23)
        mul_921: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_919, mul_920);  mul_919 = mul_920 = None
        unsqueeze_421: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_921, 0);  mul_921 = None
        unsqueeze_422: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_38: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.07450538873672485);  primals_40 = None
        view_34: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_38, [-1]);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_922: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, view_34);  view_34 = None
        unsqueeze_423: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
        unsqueeze_424: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
        mul_923: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_422);  sub_291 = unsqueeze_422 = None
        sub_293: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_309, mul_923);  view_309 = mul_923 = None
        sub_294: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_420);  sub_293 = unsqueeze_420 = None
        mul_924: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_424);  sub_294 = unsqueeze_424 = None
        mul_925: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, squeeze_23);  sum_172 = squeeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_310: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_925, [128, 1, 1, 1]);  mul_925 = None
        mul_926: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_310, 0.07450538873672485);  view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_311: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_924, [128, 64, 3, 3]);  mul_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_8: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.neg.default(convolution_12)
        exp_8: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_20: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        reciprocal_43: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_20);  add_20 = None
        mul_927: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        mul_928: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_315, mul_927);  getitem_315 = None
        sub_295: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_927);  mul_927 = None
        mul_929: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(convolution_12, sub_295);  convolution_12 = sub_295 = None
        add_229: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_929, 1);  mul_929 = None
        mul_930: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_928, add_229);  mul_928 = add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_173: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_930, [0, 2, 3])
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_930, mul_31, view_32, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_930 = view_32 = None
        getitem_318: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = convolution_backward_68[0]
        getitem_319: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_68[1];  convolution_backward_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_312: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_319, [1, 128, 256]);  getitem_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_174: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_312, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_30: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(primals_36, [1, 128, -1]);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_296: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_30, unsqueeze_426);  view_30 = unsqueeze_426 = None
        mul_931: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_312, sub_296)
        sum_175: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_931, [0, 2]);  mul_931 = None
        mul_932: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_174, 0.00390625);  sum_174 = None
        unsqueeze_427: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_428: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        mul_933: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, 0.00390625)
        mul_934: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, squeeze_21)
        mul_935: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_933, mul_934);  mul_933 = mul_934 = None
        unsqueeze_429: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
        unsqueeze_430: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_35: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_37, 0.11175808310508728);  primals_37 = None
        view_31: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_35, [-1]);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_936: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, view_31);  view_31 = None
        unsqueeze_431: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
        unsqueeze_432: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
        mul_937: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_430);  sub_296 = unsqueeze_430 = None
        sub_298: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_312, mul_937);  view_312 = mul_937 = None
        sub_299: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_428);  sub_298 = unsqueeze_428 = None
        mul_938: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_432);  sub_299 = unsqueeze_432 = None
        mul_939: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, squeeze_21);  sum_175 = squeeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_313: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_939, [128, 1, 1, 1]);  mul_939 = None
        mul_940: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_313, 0.11175808310508728);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_314: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_938, [128, 256, 1, 1]);  mul_938 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_176: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_222, [0, 2, 3])
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(add_222, avg_pool2d, view_29, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_222 = avg_pool2d = view_29 = None
        getitem_321: "f32[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = convolution_backward_69[0]
        getitem_322: "f32[512, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_69[1];  convolution_backward_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_315: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_322, [1, 512, 256]);  getitem_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_177: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_315, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_27: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(primals_33, [1, 512, -1]);  primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_300: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_434);  view_27 = unsqueeze_434 = None
        mul_941: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_315, sub_300)
        sum_178: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_941, [0, 2]);  mul_941 = None
        mul_942: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, 0.00390625);  sum_177 = None
        unsqueeze_435: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
        unsqueeze_436: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
        mul_943: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, 0.00390625)
        mul_944: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_945: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_943, mul_944);  mul_943 = mul_944 = None
        unsqueeze_437: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
        unsqueeze_438: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_32: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.11175808310508728);  primals_34 = None
        view_28: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(mul_32, [-1]);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_946: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, view_28);  view_28 = None
        unsqueeze_439: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_946, 0);  mul_946 = None
        unsqueeze_440: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        mul_947: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_438);  sub_300 = unsqueeze_438 = None
        sub_302: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_315, mul_947);  view_315 = mul_947 = None
        sub_303: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_436);  sub_302 = unsqueeze_436 = None
        mul_948: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_440);  sub_303 = unsqueeze_440 = None
        mul_949: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, squeeze_19);  sum_178 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_316: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_949, [512, 1, 1, 1]);  mul_949 = None
        mul_950: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_316, 0.11175808310508728);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_317: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_948, [512, 256, 1, 1]);  mul_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_backward_2: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_321, mul_31, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_321 = mul_31 = None
        add_230: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(getitem_318, avg_pool2d_backward_2);  getitem_318 = avg_pool2d_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_951: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(add_230, 0.9805806756909201);  add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_28: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_29: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, 2.0);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_30: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, 0.2);  mul_29 = None
        add_16: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_30, convolution_4);  mul_30 = convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        neg_7: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.neg.default(add_16)
        exp_7: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_17: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        reciprocal_44: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.reciprocal.default(add_17);  add_17 = None
        mul_952: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        mul_953: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = None
        sub_304: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_952);  mul_952 = None
        mul_954: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(add_16, sub_304);  add_16 = sub_304 = None
        add_232: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_954, 1);  mul_954 = None
        mul_955: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_953, add_232);  mul_953 = add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_956: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_955, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_957: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_956, 2.0);  mul_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_958: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_957, convolution_8);  convolution_8 = None
        mul_959: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_957, sigmoid);  mul_957 = None
        sum_179: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_958, [2, 3], True);  mul_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sub_305: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_960: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.mul.Tensor(sigmoid, sub_305);  sigmoid = sub_305 = None
        mul_961: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, mul_960);  sum_179 = mul_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_180: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_961, [0, 2, 3])
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_961, relu, primals_31, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_961 = primals_31 = None
        getitem_324: "f32[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_70[0]
        getitem_325: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_70[1];  convolution_backward_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_11: "b8[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_11: "f32[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.aten.where.self(le_11, full_default, getitem_324);  le_11 = full_default = getitem_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_181: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(where_11, mean, primals_29, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean = primals_29 = None
        getitem_327: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_71[0]
        getitem_328: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_71[1];  convolution_backward_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_13: "f32[128, 256, 56, 56][256, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_327, [128, 256, 56, 56]);  getitem_327 = None
        div_64: "f32[128, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_13, 3136);  expand_13 = None
        add_233: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_959, div_64);  mul_959 = div_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_182: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_233, [0, 2, 3])
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(add_233, div_6, view_26, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_233 = div_6 = view_26 = None
        getitem_330: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_72[0]
        getitem_331: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_72[1];  convolution_backward_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_318: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_331, [1, 256, 64]);  getitem_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_183: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_318, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_24: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(primals_26, [1, 256, -1]);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_306: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_24, unsqueeze_442);  view_24 = unsqueeze_442 = None
        mul_962: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_318, sub_306)
        sum_184: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_962, [0, 2]);  mul_962 = None
        mul_963: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, 0.015625);  sum_183 = None
        unsqueeze_443: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
        unsqueeze_444: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
        mul_964: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, 0.015625)
        mul_965: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, squeeze_17)
        mul_966: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_964, mul_965);  mul_964 = mul_965 = None
        unsqueeze_445: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_966, 0);  mul_966 = None
        unsqueeze_446: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_25: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_27, 0.22351616621017456);  primals_27 = None
        view_25: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(mul_25, [-1]);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_967: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, view_25);  view_25 = None
        unsqueeze_447: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_967, 0);  mul_967 = None
        unsqueeze_448: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
        mul_968: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_446);  sub_306 = unsqueeze_446 = None
        sub_308: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_318, mul_968);  view_318 = mul_968 = None
        sub_309: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_444);  sub_308 = unsqueeze_444 = None
        mul_969: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_448);  sub_309 = unsqueeze_448 = None
        mul_970: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, squeeze_17);  sum_184 = squeeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_319: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_970, [256, 1, 1, 1]);  mul_970 = None
        mul_971: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_319, 0.22351616621017456);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_320: "f32[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_969, [256, 64, 1, 1]);  mul_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        neg_6: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.neg.default(convolution_7)
        exp_6: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_14: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        reciprocal_45: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.reciprocal.default(add_14);  add_14 = None
        mul_972: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        mul_973: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(getitem_330, mul_972);  getitem_330 = None
        sub_310: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_972);  mul_972 = None
        mul_974: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convolution_7, sub_310);  convolution_7 = sub_310 = None
        add_235: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_974, 1);  mul_974 = None
        mul_975: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_973, add_235);  mul_973 = add_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_185: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_975, [0, 2, 3])
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_975, div_5, view_23, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_975 = div_5 = view_23 = None
        getitem_333: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_73[0]
        getitem_334: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_73[1];  convolution_backward_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_79: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_334, memory_format = torch.contiguous_format);  getitem_334 = None
        view_321: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [1, 64, 576]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_186: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_321, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_10: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_23, memory_format = torch.contiguous_format);  primals_23 = None
        view_21: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 64, 576]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_311: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_21, unsqueeze_450);  view_21 = unsqueeze_450 = None
        mul_976: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_321, sub_311)
        sum_187: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_976, [0, 2]);  mul_976 = None
        mul_977: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_186, 0.001736111111111111);  sum_186 = None
        unsqueeze_451: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_977, 0);  mul_977 = None
        unsqueeze_452: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        mul_978: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, 0.001736111111111111)
        mul_979: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, squeeze_15)
        mul_980: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_978, mul_979);  mul_978 = mul_979 = None
        unsqueeze_453: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
        unsqueeze_454: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_22: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_24, 0.07450538873672485);  primals_24 = None
        view_22: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_22, [-1]);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_981: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, view_22);  view_22 = None
        unsqueeze_455: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
        unsqueeze_456: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        mul_982: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_454);  sub_311 = unsqueeze_454 = None
        sub_313: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_321, mul_982);  view_321 = mul_982 = None
        sub_314: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_452);  sub_313 = unsqueeze_452 = None
        mul_983: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_456);  sub_314 = unsqueeze_456 = None
        mul_984: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, squeeze_15);  sum_187 = squeeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_322: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_984, [64, 1, 1, 1]);  mul_984 = None
        mul_985: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_322, 0.07450538873672485);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_323: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_983, [64, 64, 3, 3]);  mul_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        neg_5: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.neg.default(convolution_6)
        exp_5: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_12: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        reciprocal_46: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.reciprocal.default(add_12);  add_12 = None
        mul_986: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        mul_987: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(getitem_333, mul_986);  getitem_333 = None
        sub_315: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_986);  mul_986 = None
        mul_988: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convolution_6, sub_315);  convolution_6 = sub_315 = None
        add_237: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_988, 1);  mul_988 = None
        mul_989: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_987, add_237);  mul_987 = add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_188: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_989, [0, 2, 3])
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_989, div_4, view_20, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_989 = div_4 = view_20 = None
        getitem_336: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_74[0]
        getitem_337: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_74[1];  convolution_backward_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_80: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_337, memory_format = torch.contiguous_format);  getitem_337 = None
        view_324: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [1, 64, 576]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_189: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_324, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_8: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_20, memory_format = torch.contiguous_format);  primals_20 = None
        view_18: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1, 64, 576]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_316: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_18, unsqueeze_458);  view_18 = unsqueeze_458 = None
        mul_990: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_324, sub_316)
        sum_190: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_990, [0, 2]);  mul_990 = None
        mul_991: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, 0.001736111111111111);  sum_189 = None
        unsqueeze_459: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
        unsqueeze_460: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
        mul_992: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_190, 0.001736111111111111)
        mul_993: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_994: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_992, mul_993);  mul_992 = mul_993 = None
        unsqueeze_461: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
        unsqueeze_462: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_19: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_21, 0.07450538873672485);  primals_21 = None
        view_19: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [-1]);  mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_995: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, view_19);  view_19 = None
        unsqueeze_463: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_995, 0);  mul_995 = None
        unsqueeze_464: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        mul_996: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_462);  sub_316 = unsqueeze_462 = None
        sub_318: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_324, mul_996);  view_324 = mul_996 = None
        sub_319: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_460);  sub_318 = unsqueeze_460 = None
        mul_997: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_464);  sub_319 = unsqueeze_464 = None
        mul_998: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_190, squeeze_13);  sum_190 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_325: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_998, [64, 1, 1, 1]);  mul_998 = None
        mul_999: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_325, 0.07450538873672485);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_326: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_997, [64, 64, 3, 3]);  mul_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        neg_4: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.neg.default(convolution_5)
        exp_4: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_10: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        reciprocal_47: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.reciprocal.default(add_10);  add_10 = None
        mul_1000: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        mul_1001: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(getitem_336, mul_1000);  getitem_336 = None
        sub_320: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_1000);  mul_1000 = None
        mul_1002: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convolution_5, sub_320);  convolution_5 = sub_320 = None
        add_239: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_1002, 1);  mul_1002 = None
        mul_1003: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_1001, add_239);  mul_1001 = add_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_191: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1003, [0, 2, 3])
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1003, mul_12, view_17, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1003 = view_17 = None
        getitem_339: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = convolution_backward_75[0]
        getitem_340: "f32[64, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_75[1];  convolution_backward_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_327: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_340, [1, 64, 128]);  getitem_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_192: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_327, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_15: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_17, [1, 64, -1]);  primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_321: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_466);  view_15 = unsqueeze_466 = None
        mul_1004: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_327, sub_321)
        sum_193: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1004, [0, 2]);  mul_1004 = None
        mul_1005: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_192, 0.0078125);  sum_192 = None
        unsqueeze_467: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1005, 0);  mul_1005 = None
        unsqueeze_468: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        mul_1006: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, 0.0078125)
        mul_1007: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, squeeze_11)
        mul_1008: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1006, mul_1007);  mul_1006 = mul_1007 = None
        unsqueeze_469: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
        unsqueeze_470: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_16: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_18, 0.1580497968320339);  primals_18 = None
        view_16: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_16, [-1]);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_1009: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, view_16);  view_16 = None
        unsqueeze_471: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
        unsqueeze_472: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
        mul_1010: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_470);  sub_321 = unsqueeze_470 = None
        sub_323: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_327, mul_1010);  view_327 = mul_1010 = None
        sub_324: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_468);  sub_323 = unsqueeze_468 = None
        mul_1011: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_472);  sub_324 = unsqueeze_472 = None
        mul_1012: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, squeeze_11);  sum_193 = squeeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_328: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1012, [64, 1, 1, 1]);  mul_1012 = None
        mul_1013: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_328, 0.1580497968320339);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_329: "f32[64, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1011, [64, 128, 1, 1]);  mul_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_194: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 2, 3])
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_955, mul_12, view_14, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_955 = mul_12 = view_14 = None
        getitem_342: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = convolution_backward_76[0]
        getitem_343: "f32[256, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_76[1];  convolution_backward_76 = None
        add_240: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(getitem_339, getitem_342);  getitem_339 = getitem_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_330: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_343, [1, 256, 128]);  getitem_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_195: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_330, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_12: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_14, [1, 256, -1]);  primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_325: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_12, unsqueeze_474);  view_12 = unsqueeze_474 = None
        mul_1014: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_330, sub_325)
        sum_196: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1014, [0, 2]);  mul_1014 = None
        mul_1015: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_195, 0.0078125);  sum_195 = None
        unsqueeze_475: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1015, 0);  mul_1015 = None
        unsqueeze_476: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        mul_1016: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_196, 0.0078125)
        mul_1017: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, squeeze_9)
        mul_1018: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1016, mul_1017);  mul_1016 = mul_1017 = None
        unsqueeze_477: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
        unsqueeze_478: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_13: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_15, 0.1580497968320339);  primals_15 = None
        view_13: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [-1]);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_1019: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, view_13);  view_13 = None
        unsqueeze_479: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
        unsqueeze_480: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        mul_1020: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_478);  sub_325 = unsqueeze_478 = None
        sub_327: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_330, mul_1020);  view_330 = mul_1020 = None
        sub_328: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_476);  sub_327 = unsqueeze_476 = None
        mul_1021: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_480);  sub_328 = unsqueeze_480 = None
        mul_1022: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_196, squeeze_9);  sum_196 = squeeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_331: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1022, [256, 1, 1, 1]);  mul_1022 = None
        mul_1023: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_331, 0.1580497968320339);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_332: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1021, [256, 128, 1, 1]);  mul_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_1024: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(add_240, 1.0);  add_240 = None
        neg_3: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.neg.default(convolution_3)
        exp_3: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_7: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        reciprocal_48: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_7);  add_7 = None
        mul_1025: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        mul_1026: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_1024, mul_1025);  mul_1024 = None
        sub_329: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_1025);  mul_1025 = None
        mul_1027: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(convolution_3, sub_329);  convolution_3 = sub_329 = None
        add_242: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_1027, 1);  mul_1027 = None
        mul_1028: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_1026, add_242);  mul_1026 = add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_197: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1028, [0, 2, 3])
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1028, div_2, view_11, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1028 = div_2 = view_11 = None
        getitem_345: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = convolution_backward_77[0]
        getitem_346: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_77[1];  convolution_backward_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_81: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_346, memory_format = torch.contiguous_format);  getitem_346 = None
        view_333: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [1, 128, 576]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_198: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_333, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_6: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_11, memory_format = torch.contiguous_format);  primals_11 = None
        view_9: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [1, 128, 576]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_330: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_9, unsqueeze_482);  view_9 = unsqueeze_482 = None
        mul_1029: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_333, sub_330)
        sum_199: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 2]);  mul_1029 = None
        mul_1030: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_198, 0.001736111111111111);  sum_198 = None
        unsqueeze_483: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1030, 0);  mul_1030 = None
        unsqueeze_484: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
        mul_1031: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, 0.001736111111111111)
        mul_1032: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_1033: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1031, mul_1032);  mul_1031 = mul_1032 = None
        unsqueeze_485: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1033, 0);  mul_1033 = None
        unsqueeze_486: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_9: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_12, 0.07450538873672485);  primals_12 = None
        view_10: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_9, [-1]);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_1034: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, view_10);  view_10 = None
        unsqueeze_487: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1034, 0);  mul_1034 = None
        unsqueeze_488: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        mul_1035: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_486);  sub_330 = unsqueeze_486 = None
        sub_332: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_333, mul_1035);  view_333 = mul_1035 = None
        sub_333: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_332, unsqueeze_484);  sub_332 = unsqueeze_484 = None
        mul_1036: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_488);  sub_333 = unsqueeze_488 = None
        mul_1037: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, squeeze_7);  sum_199 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_334: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1037, [128, 1, 1, 1]);  mul_1037 = None
        mul_1038: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_334, 0.07450538873672485);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_335: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1036, [128, 64, 3, 3]);  mul_1036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        neg_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.neg.default(convolution_2)
        exp_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_5: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        reciprocal_49: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.reciprocal.default(add_5);  add_5 = None
        mul_1039: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        mul_1040: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(getitem_345, mul_1039);  getitem_345 = None
        sub_334: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_1039);  mul_1039 = None
        mul_1041: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(convolution_2, sub_334);  convolution_2 = sub_334 = None
        add_244: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_1041, 1);  mul_1041 = None
        mul_1042: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_1040, add_244);  mul_1040 = add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_200: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1042, [0, 2, 3])
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1042, div_1, view_8, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1042 = div_1 = view_8 = None
        getitem_348: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = convolution_backward_78[0]
        getitem_349: "f32[64, 32, 3, 3][288, 1, 96, 32]cuda:0" = convolution_backward_78[1];  convolution_backward_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_82: "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_349, memory_format = torch.contiguous_format);  getitem_349 = None
        view_336: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [1, 64, 288]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_201: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_336, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_4: "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_8, memory_format = torch.contiguous_format);  primals_8 = None
        view_6: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [1, 64, 288]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_335: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_6, unsqueeze_490);  view_6 = unsqueeze_490 = None
        mul_1043: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_336, sub_335)
        sum_202: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1043, [0, 2]);  mul_1043 = None
        mul_1044: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_201, 0.003472222222222222);  sum_201 = None
        unsqueeze_491: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1044, 0);  mul_1044 = None
        unsqueeze_492: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        mul_1045: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_202, 0.003472222222222222)
        mul_1046: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
        mul_1047: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1045, mul_1046);  mul_1045 = mul_1046 = None
        unsqueeze_493: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1047, 0);  mul_1047 = None
        unsqueeze_494: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_6: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_9, 0.10536653122135592);  primals_9 = None
        view_7: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_6, [-1]);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_1048: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, view_7);  view_7 = None
        unsqueeze_495: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
        unsqueeze_496: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
        mul_1049: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_494);  sub_335 = unsqueeze_494 = None
        sub_337: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_336, mul_1049);  view_336 = mul_1049 = None
        sub_338: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_337, unsqueeze_492);  sub_337 = unsqueeze_492 = None
        mul_1050: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_496);  sub_338 = unsqueeze_496 = None
        mul_1051: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_202, squeeze_5);  sum_202 = squeeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_337: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1051, [64, 1, 1, 1]);  mul_1051 = None
        mul_1052: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_337, 0.10536653122135592);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_338: "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1050, [64, 32, 3, 3]);  mul_1050 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        neg_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.neg.default(convolution_1)
        exp_1: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_3: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        reciprocal_50: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.reciprocal.default(add_3);  add_3 = None
        mul_1053: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        mul_1054: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(getitem_348, mul_1053);  getitem_348 = None
        sub_339: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_1053);  mul_1053 = None
        mul_1055: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convolution_1, sub_339);  convolution_1 = sub_339 = None
        add_246: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_1055, 1);  mul_1055 = None
        mul_1056: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_1054, add_246);  mul_1054 = add_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_203: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1056, [0, 2, 3])
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1056, div, view_5, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1056 = div = view_5 = None
        getitem_351: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_79[0]
        getitem_352: "f32[32, 16, 3, 3][144, 1, 48, 16]cuda:0" = convolution_backward_79[1];  convolution_backward_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_83: "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_352, memory_format = torch.contiguous_format);  getitem_352 = None
        view_339: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [1, 32, 144]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_204: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_339, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_2: "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_5, memory_format = torch.contiguous_format);  primals_5 = None
        view_3: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1, 32, 144]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_340: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_498);  view_3 = unsqueeze_498 = None
        mul_1057: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_339, sub_340)
        sum_205: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1057, [0, 2]);  mul_1057 = None
        mul_1058: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_204, 0.006944444444444444);  sum_204 = None
        unsqueeze_499: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
        unsqueeze_500: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        mul_1059: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, 0.006944444444444444)
        mul_1060: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
        mul_1061: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1059, mul_1060);  mul_1059 = mul_1060 = None
        unsqueeze_501: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
        unsqueeze_502: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_3: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_6, 0.1490107774734497);  primals_6 = None
        view_4: "f32[32][1]cuda:0" = torch.ops.aten.reshape.default(mul_3, [-1]);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_1062: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, view_4);  view_4 = None
        unsqueeze_503: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
        unsqueeze_504: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        mul_1063: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_502);  sub_340 = unsqueeze_502 = None
        sub_342: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_339, mul_1063);  view_339 = mul_1063 = None
        sub_343: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_500);  sub_342 = unsqueeze_500 = None
        mul_1064: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_504);  sub_343 = unsqueeze_504 = None
        mul_1065: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, squeeze_3);  sum_205 = squeeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_340: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1065, [32, 1, 1, 1]);  mul_1065 = None
        mul_1066: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_340, 0.1490107774734497);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_341: "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1064, [32, 16, 3, 3]);  mul_1064 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        neg: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.neg.default(convolution)
        exp: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.exp.default(neg);  neg = None
        add_1: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        reciprocal_51: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.reciprocal.default(add_1);  add_1 = None
        mul_1067: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        mul_1068: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(getitem_351, mul_1067);  getitem_351 = None
        sub_344: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_1067);  mul_1067 = None
        mul_1069: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convolution, sub_344);  convolution = sub_344 = None
        add_248: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_1069, 1);  mul_1069 = None
        mul_1070: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_1068, add_248);  mul_1068 = add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_206: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1070, [0, 2, 3])
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1070, primals_4, view_2, [16], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1070 = primals_4 = view_2 = None
        getitem_355: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_80[1];  convolution_backward_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_84: "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(getitem_355, memory_format = torch.contiguous_format);  getitem_355 = None
        view_342: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [1, 16, 27]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_207: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_342, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone: "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_1, memory_format = torch.contiguous_format);  primals_1 = None
        view: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [1, 16, 27]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_345: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.sub.Tensor(view, unsqueeze_506);  view = unsqueeze_506 = None
        mul_1071: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_342, sub_345)
        sum_208: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1071, [0, 2]);  mul_1071 = None
        mul_1072: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_207, 0.037037037037037035);  sum_207 = None
        unsqueeze_507: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
        unsqueeze_508: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
        mul_1073: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_208, 0.037037037037037035)
        mul_1074: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1075: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1073, mul_1074);  mul_1073 = mul_1074 = None
        unsqueeze_509: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
        unsqueeze_510: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_2, 0.34412564994580647);  primals_2 = None
        view_1: "f32[16][1]cuda:0" = torch.ops.aten.reshape.default(mul, [-1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_1076: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, view_1);  view_1 = None
        unsqueeze_511: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1076, 0);  mul_1076 = None
        unsqueeze_512: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        mul_1077: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_510);  sub_345 = unsqueeze_510 = None
        sub_347: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_342, mul_1077);  view_342 = mul_1077 = None
        sub_348: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_508);  sub_347 = unsqueeze_508 = None
        mul_1078: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_512);  sub_348 = unsqueeze_512 = None
        mul_1079: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_208, squeeze_1);  sum_208 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_343: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1079, [16, 1, 1, 1]);  mul_1079 = None
        mul_1080: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_343, 0.34412564994580647);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_344: "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1078, [16, 3, 3, 3]);  mul_1078 = None
        return (view_344, mul_1080, sum_206, None, view_341, mul_1066, sum_203, view_338, mul_1052, sum_200, view_335, mul_1038, sum_197, view_332, mul_1023, sum_194, view_329, mul_1013, sum_191, view_326, mul_999, sum_188, view_323, mul_985, sum_185, view_320, mul_971, sum_182, getitem_328, sum_181, getitem_325, sum_180, view_317, mul_950, sum_176, view_314, mul_940, sum_173, view_311, mul_926, sum_170, view_308, mul_912, sum_167, view_305, mul_898, sum_164, getitem_307, sum_163, getitem_304, sum_162, view_302, mul_877, sum_158, view_299, mul_863, sum_155, view_296, mul_849, sum_152, view_293, mul_835, sum_149, getitem_289, sum_148, getitem_286, sum_147, view_290, mul_814, sum_143, view_287, mul_804, sum_140, view_284, mul_790, sum_137, view_281, mul_776, sum_134, view_278, mul_762, sum_131, getitem_268, sum_130, getitem_265, sum_129, view_275, mul_741, sum_125, view_272, mul_727, sum_122, view_269, mul_713, sum_119, view_266, mul_699, sum_116, getitem_250, sum_115, getitem_247, sum_114, view_263, mul_678, sum_110, view_260, mul_664, sum_107, view_257, mul_650, sum_104, view_254, mul_636, sum_101, getitem_232, sum_100, getitem_229, sum_99, view_251, mul_615, sum_95, view_248, mul_601, sum_92, view_245, mul_587, sum_89, view_242, mul_573, sum_86, getitem_214, sum_85, getitem_211, sum_84, view_239, mul_552, sum_80, view_236, mul_538, sum_77, view_233, mul_524, sum_74, view_230, mul_510, sum_71, getitem_196, sum_70, getitem_193, sum_69, view_227, mul_489, sum_65, view_224, mul_475, sum_62, view_221, mul_461, sum_59, view_218, mul_447, sum_56, getitem_178, sum_55, getitem_175, sum_54, view_215, mul_426, sum_50, view_212, mul_416, sum_47, view_209, mul_402, sum_44, view_206, mul_388, sum_41, view_203, mul_374, sum_38, getitem_157, sum_37, getitem_154, sum_36, view_200, mul_353, sum_32, view_197, mul_339, sum_29, view_194, mul_325, sum_26, view_191, mul_311, sum_23, getitem_139, sum_22, getitem_136, sum_21, view_188, mul_290, sum_17, view_185, mul_276, sum_14, view_182, mul_262, sum_11, view_179, mul_248, sum_8, getitem_121, sum_7, getitem_118, sum_6, view_176, mul_232, sum_2, mm_1, view_172)
