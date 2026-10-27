class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0", primals_2: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_5: "f32[32, 16, 3, 3][144, 1, 48, 16]cuda:0", primals_6: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_8: "f32[64, 32, 3, 3][288, 1, 96, 32]cuda:0", primals_9: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_11: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_12: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_14: "f32[256, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_15: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_17: "f32[64, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_18: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_20: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_21: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_23: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_24: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_26: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_27: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_33: "f32[512, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_34: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_36: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_37: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_39: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_40: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_42: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_43: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_45: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_46: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_52: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_53: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_55: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_56: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_58: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_59: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_61: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_62: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_68: "f32[1536, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_69: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_71: "f32[384, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_72: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_74: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_75: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_77: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_78: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_80: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_81: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_87: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_88: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_90: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_91: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_93: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_94: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_96: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_97: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_103: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_104: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_106: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_107: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_109: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_110: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_112: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_113: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_119: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_120: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_122: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_123: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_125: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_126: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_128: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_129: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_135: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_136: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_138: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_139: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_141: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_142: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_144: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_145: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_151: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_152: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_154: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_155: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_157: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_158: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_160: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_161: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_167: "f32[1536, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_168: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_170: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_171: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_173: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_174: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_176: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_177: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_179: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_180: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_186: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_187: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_189: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_190: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_192: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_193: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_195: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_196: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_202: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_203: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_205: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_206: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_208: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_209: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_211: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_212: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0", primals_218: "f32[2304, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", primals_219: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0", squeeze_1: "f32[16][1]cuda:0", convert_element_type_1: "bf16[16, 3, 3, 3][27, 9, 3, 1]cuda:0", convert_element_type_2: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0", convert_element_type_4: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0", squeeze_3: "f32[32][1]cuda:0", convert_element_type_6: "bf16[32, 16, 3, 3][144, 9, 3, 1]cuda:0", convolution_1: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", convert_element_type_8: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", squeeze_5: "f32[64][1]cuda:0", convert_element_type_10: "bf16[64, 32, 3, 3][288, 9, 3, 1]cuda:0", convolution_2: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0", convert_element_type_12: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0", squeeze_7: "f32[128][1]cuda:0", convert_element_type_14: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_3: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", mul_12: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", squeeze_9: "f32[256][1]cuda:0", convert_element_type_18: "bf16[256, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_4: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", squeeze_11: "f32[64][1]cuda:0", convert_element_type_20: "bf16[64, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_5: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_22: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_13: "f32[64][1]cuda:0", convert_element_type_24: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_6: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_26: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_15: "f32[64][1]cuda:0", convert_element_type_28: "bf16[64, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_7: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_30: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_17: "f32[256][1]cuda:0", convert_element_type_32: "bf16[256, 64, 1, 1][64, 1, 1, 1]cuda:0", convolution_8: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", mean: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convert_element_type_34: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0", relu: "bf16[128, 64, 1, 1][64, 1, 64, 64]cuda:0", convert_element_type_36: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_10: "bf16[128, 256, 1, 1][256, 1, 256, 256]cuda:0", mul_31: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0", avg_pool2d: "bf16[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0", squeeze_19: "f32[512][1]cuda:0", convert_element_type_40: "bf16[512, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_11: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_21: "f32[128][1]cuda:0", convert_element_type_42: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0", convolution_12: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", convert_element_type_44: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0", squeeze_23: "f32[128][1]cuda:0", convert_element_type_46: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_13: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_48: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_25: "f32[128][1]cuda:0", convert_element_type_50: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_14: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_52: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_27: "f32[512][1]cuda:0", convert_element_type_54: "bf16[512, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_15: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", mean_1: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0", convert_element_type_56: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", relu_1: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0", convert_element_type_58: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_17: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", mul_50: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", squeeze_29: "f32[128][1]cuda:0", convert_element_type_62: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_18: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_64: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_31: "f32[128][1]cuda:0", convert_element_type_66: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_19: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_68: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_33: "f32[128][1]cuda:0", convert_element_type_70: "bf16[128, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_20: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_72: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_35: "f32[512][1]cuda:0", convert_element_type_74: "bf16[512, 128, 1, 1][128, 1, 1, 1]cuda:0", convolution_21: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", mean_2: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0", convert_element_type_76: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", relu_2: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0", convert_element_type_78: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_23: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0", add_35: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", mul_66: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0", avg_pool2d_1: "bf16[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0", squeeze_37: "f32[1536][1]cuda:0", convert_element_type_82: "bf16[1536, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_24: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_39: "f32[384][1]cuda:0", convert_element_type_84: "bf16[384, 512, 1, 1][512, 1, 1, 1]cuda:0", convolution_25: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_86: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", squeeze_41: "f32[384][1]cuda:0", convert_element_type_88: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_26: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_90: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_43: "f32[384][1]cuda:0", convert_element_type_92: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_27: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_94: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_45: "f32[1536][1]cuda:0", convert_element_type_96: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_28: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_3: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convert_element_type_98: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", relu_3: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_100: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_30: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", mul_85: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_47: "f32[384][1]cuda:0", convert_element_type_104: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_31: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_106: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_49: "f32[384][1]cuda:0", convert_element_type_108: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_32: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_110: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_51: "f32[384][1]cuda:0", convert_element_type_112: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_33: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_114: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_53: "f32[1536][1]cuda:0", convert_element_type_116: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_34: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_4: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convert_element_type_118: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", relu_4: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_120: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_36: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_54: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_101: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_55: "f32[384][1]cuda:0", convert_element_type_124: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_37: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_126: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_57: "f32[384][1]cuda:0", convert_element_type_128: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_38: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_130: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_59: "f32[384][1]cuda:0", convert_element_type_132: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_39: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_134: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_61: "f32[1536][1]cuda:0", convert_element_type_136: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_40: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_5: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convert_element_type_138: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", relu_5: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_140: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_42: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_63: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_117: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_63: "f32[384][1]cuda:0", convert_element_type_144: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_43: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_146: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_65: "f32[384][1]cuda:0", convert_element_type_148: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_44: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_150: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_67: "f32[384][1]cuda:0", convert_element_type_152: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_45: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_154: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_69: "f32[1536][1]cuda:0", convert_element_type_156: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_46: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_6: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convert_element_type_158: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", relu_6: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_160: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_48: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_72: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_133: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_71: "f32[384][1]cuda:0", convert_element_type_164: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_49: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_166: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_73: "f32[384][1]cuda:0", convert_element_type_168: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_50: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_170: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_75: "f32[384][1]cuda:0", convert_element_type_172: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_51: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_174: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_77: "f32[1536][1]cuda:0", convert_element_type_176: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_52: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_7: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convert_element_type_178: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", relu_7: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_180: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_54: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_81: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_149: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", squeeze_79: "f32[384][1]cuda:0", convert_element_type_184: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_55: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_186: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_81: "f32[384][1]cuda:0", convert_element_type_188: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_56: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_190: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_83: "f32[384][1]cuda:0", convert_element_type_192: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_57: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_194: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_85: "f32[1536][1]cuda:0", convert_element_type_196: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_58: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mean_8: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convert_element_type_198: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", relu_8: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_200: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_60: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_90: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", mul_165: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", avg_pool2d_2: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_87: "f32[1536][1]cuda:0", convert_element_type_204: "bf16[1536, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_61: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_89: "f32[384][1]cuda:0", convert_element_type_206: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_62: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_208: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_91: "f32[384][1]cuda:0", convert_element_type_210: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_63: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", convert_element_type_212: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_93: "f32[384][1]cuda:0", convert_element_type_214: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_64: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", convert_element_type_216: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_95: "f32[1536][1]cuda:0", convert_element_type_218: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_65: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", mean_9: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convert_element_type_220: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", relu_9: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_222: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_67: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", mul_184: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_97: "f32[384][1]cuda:0", convert_element_type_226: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_68: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", convert_element_type_228: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_99: "f32[384][1]cuda:0", convert_element_type_230: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_69: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", convert_element_type_232: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_101: "f32[384][1]cuda:0", convert_element_type_234: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_70: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", convert_element_type_236: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_103: "f32[1536][1]cuda:0", convert_element_type_238: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_71: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", mean_10: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convert_element_type_240: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", relu_10: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_242: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_73: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_109: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", mul_200: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_105: "f32[384][1]cuda:0", convert_element_type_246: "bf16[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_74: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", convert_element_type_248: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_107: "f32[384][1]cuda:0", convert_element_type_250: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_75: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", convert_element_type_252: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_109: "f32[384][1]cuda:0", convert_element_type_254: "bf16[384, 64, 3, 3][576, 9, 3, 1]cuda:0", convolution_76: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", convert_element_type_256: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0", squeeze_111: "f32[1536][1]cuda:0", convert_element_type_258: "bf16[1536, 384, 1, 1][384, 1, 1, 1]cuda:0", convolution_77: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", mean_11: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convert_element_type_260: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", relu_11: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_262: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_79: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", add_118: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0", squeeze_113: "f32[2304][1]cuda:0", convert_element_type_264: "bf16[2304, 1536, 1, 1][1536, 1, 1, 1]cuda:0", convolution_80: "bf16[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0", view_171: "bf16[128, 2304][2304, 1]cuda:0", permute_1: "bf16[1000, 2304][2304, 1]cuda:0", unsqueeze_58: "f32[1, 2304, 1][2304, 1, 1]cuda:0", unsqueeze_66: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_74: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_82: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_90: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_98: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_106: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_114: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_122: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_130: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_138: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_146: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_154: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_162: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_170: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_178: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_186: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_194: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_202: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_210: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_218: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_226: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_234: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_242: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_250: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_258: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_266: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_274: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_282: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_290: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_298: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_306: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_314: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_322: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_330: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_338: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_346: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_354: "f32[1, 384, 1][384, 1, 1]cuda:0", unsqueeze_362: "f32[1, 1536, 1][1536, 1, 1]cuda:0", unsqueeze_370: "f32[1, 512, 1][512, 1, 1]cuda:0", unsqueeze_378: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_386: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_394: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_402: "f32[1, 512, 1][512, 1, 1]cuda:0", unsqueeze_410: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_418: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_426: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_434: "f32[1, 512, 1][512, 1, 1]cuda:0", unsqueeze_442: "f32[1, 256, 1][256, 1, 1]cuda:0", unsqueeze_450: "f32[1, 64, 1][64, 1, 1]cuda:0", unsqueeze_458: "f32[1, 64, 1][64, 1, 1]cuda:0", unsqueeze_466: "f32[1, 64, 1][64, 1, 1]cuda:0", unsqueeze_474: "f32[1, 256, 1][256, 1, 1]cuda:0", unsqueeze_482: "f32[1, 128, 1][128, 1, 1]cuda:0", unsqueeze_490: "f32[1, 64, 1][64, 1, 1]cuda:0", unsqueeze_498: "f32[1, 32, 1][32, 1, 1]cuda:0", unsqueeze_506: "f32[1, 16, 1][16, 1, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        mm: "bf16[128, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 2304][2304, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view_171);  permute_2 = view_171 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_172: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_276: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_172, torch.bfloat16);  view_172 = None
        convert_element_type_277: "f32[1000, 2304][2304, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_278: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_276, torch.float32);  convert_element_type_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_173: "bf16[128, 2304, 1, 1][2304, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 2304, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_114: "bf16[128, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_173, 3);  view_173 = None
        squeeze_115: "bf16[128, 2304][2304, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze_114, 2);  squeeze_114 = None
        full: "bf16[294912][1]cuda:0" = torch.ops.aten.full.default([294912], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "bf16[294912][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full, squeeze_115, [128, 2304], [2304, 1], 0);  full = squeeze_115 = None
        as_strided_5: "bf16[128, 2304, 1, 1][2304, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 2304, 1, 1], [2304, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "bf16[128, 2304, 7, 7][2304, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 2304, 7, 7]);  as_strided_5 = None
        div_52: "bf16[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:569 in forward_features, code: x = self.final_act(x)
        convert_element_type_279: "f32[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_52, torch.float32);  div_52 = None
        convert_element_type_265: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32);  convolution_80 = None
        sigmoid_12: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_265)
        mul_219: "f32[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_279, sigmoid_12);  convert_element_type_279 = None
        sub_57: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_12);  sigmoid_12 = None
        mul_220: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, sub_57);  convert_element_type_265 = sub_57 = None
        add_121: "f32[128, 2304, 7, 7][112896, 1, 16128, 2304]cuda:0" = torch.ops.aten.add.Tensor(mul_220, 1);  mul_220 = None
        mul_221: "f32[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_219, add_121);  mul_219 = add_121 = None
        convert_element_type_281: "bf16[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_2: "bf16[2304][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_281, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_281, add_118, convert_element_type_264, [2304], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_281 = add_118 = convert_element_type_264 = None
        getitem_114: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = convolution_backward[0]
        getitem_115: "bf16[2304, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_282: "f32[2304, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_115, torch.float32);  getitem_115 = None
        convert_element_type_283: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_2, torch.float32);  sum_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_174: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_282, [1, 2304, 1536]);  convert_element_type_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_3: "f32[2304][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_174, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_168: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_218, [1, 2304, -1]);  primals_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_58: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_168, unsqueeze_58);  view_168 = unsqueeze_58 = None
        mul_222: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_174, sub_58)
        sum_4: "f32[2304][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_222, [0, 2]);  mul_222 = None
        mul_223: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.0006510416666666666);  sum_3 = None
        unsqueeze_59: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_223, 0);  mul_223 = None
        unsqueeze_60: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 2);  unsqueeze_59 = None
        mul_224: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.0006510416666666666)
        mul_225: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, squeeze_113)
        mul_226: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
        unsqueeze_61: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
        unsqueeze_62: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_61, 2);  unsqueeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_216: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_219, 0.04562504637317021);  primals_219 = None
        view_169: "f32[2304][1]cuda:0" = torch.ops.aten.reshape.default(mul_216, [-1]);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_227: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, view_169);  view_169 = None
        unsqueeze_63: "f32[1, 2304][2304, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_227, 0);  mul_227 = None
        unsqueeze_64: "f32[1, 2304, 1][2304, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 2);  unsqueeze_63 = None
        mul_228: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_62);  sub_58 = unsqueeze_62 = None
        sub_60: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_174, mul_228);  view_174 = mul_228 = None
        sub_61: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_60);  sub_60 = unsqueeze_60 = None
        mul_229: "f32[1, 2304, 1536][3538944, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_64);  sub_61 = unsqueeze_64 = None
        mul_230: "f32[2304][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, squeeze_113);  sum_4 = squeeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_175: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_230, [2304, 1, 1, 1]);  mul_230 = None
        mul_231: "f32[2304, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_175, 0.04562504637317021);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_176: "f32[2304, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_229, [2304, 1536, 1, 1]);  mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_232: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_114, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_233: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_232, 2.0);  mul_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_234: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, convolution_77);  convolution_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_11: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_79);  convolution_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_235: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, sigmoid_11);  mul_233 = None
        sum_5: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_234, [2, 3], True, dtype = torch.float32);  mul_234 = None
        convert_element_type_284: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_5, torch.bfloat16);  sum_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_285: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_284, torch.float32);  convert_element_type_284 = None
        convert_element_type_286: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_11, torch.float32);  sigmoid_11 = None
        sub_62: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_286)
        mul_236: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_286, sub_62);  convert_element_type_286 = sub_62 = None
        mul_237: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_285, mul_236);  convert_element_type_285 = mul_236 = None
        convert_element_type_287: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_6: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_287, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_287, relu_11, convert_element_type_262, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_287 = convert_element_type_262 = None
        getitem_117: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_1[0]
        getitem_118: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_288: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_118, torch.float32);  getitem_118 = None
        convert_element_type_289: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_6, torch.float32);  sum_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le, full_default, getitem_117);  le = getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_7: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(where, mean_11, convert_element_type_260, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where = mean_11 = convert_element_type_260 = None
        getitem_120: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_2[0]
        getitem_121: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_290: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
        convert_element_type_291: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_7, torch.float32);  sum_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_2: "bf16[128, 1536, 7, 7][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_120, [128, 1536, 7, 7]);  getitem_120 = None
        div_53: "bf16[128, 1536, 7, 7][75264, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
        add_122: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_235, div_53);  mul_235 = div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_8: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_122, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_122, convert_element_type_256, convert_element_type_258, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_122 = convert_element_type_256 = convert_element_type_258 = None
        getitem_123: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_3[0]
        getitem_124: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_292: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_124, torch.float32);  getitem_124 = None
        convert_element_type_293: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_8, torch.float32);  sum_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_177: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_292, [1, 1536, 384]);  convert_element_type_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_9: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_177, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_165: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_211, [1, 1536, -1]);  primals_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_63: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_165, unsqueeze_66);  view_165 = unsqueeze_66 = None
        mul_238: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_177, sub_63)
        sum_10: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [0, 2]);  mul_238 = None
        mul_239: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.0026041666666666665);  sum_9 = None
        unsqueeze_67: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_239, 0);  mul_239 = None
        unsqueeze_68: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 2);  unsqueeze_67 = None
        mul_240: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.0026041666666666665)
        mul_241: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, squeeze_111)
        mul_242: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
        unsqueeze_69: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_242, 0);  mul_242 = None
        unsqueeze_70: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 2);  unsqueeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_210: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_212, 0.09125009274634042);  primals_212 = None
        view_166: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_210, [-1]);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_243: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, view_166);  view_166 = None
        unsqueeze_71: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_243, 0);  mul_243 = None
        unsqueeze_72: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_71, 2);  unsqueeze_71 = None
        mul_244: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_70);  sub_63 = unsqueeze_70 = None
        sub_65: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_177, mul_244);  view_177 = mul_244 = None
        sub_66: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_68);  sub_65 = unsqueeze_68 = None
        mul_245: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_72);  sub_66 = unsqueeze_72 = None
        mul_246: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, squeeze_111);  sum_10 = squeeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_178: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_246, [1536, 1, 1, 1]);  mul_246 = None
        mul_247: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_178, 0.09125009274634042);  view_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_179: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_245, [1536, 384, 1, 1]);  mul_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_294: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None
        convert_element_type_255: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32);  convolution_76 = None
        sigmoid_13: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_255)
        mul_248: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_294, sigmoid_13);  convert_element_type_294 = None
        sub_67: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_13);  sigmoid_13 = None
        mul_249: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_255, sub_67);  convert_element_type_255 = sub_67 = None
        add_123: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_249, 1);  mul_249 = None
        mul_250: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, add_123);  mul_248 = add_123 = None
        convert_element_type_296: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_11: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_296, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_296, convert_element_type_252, convert_element_type_254, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_296 = convert_element_type_252 = convert_element_type_254 = None
        getitem_126: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_4[0]
        getitem_127: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_297: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
        convert_element_type_298: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_11, torch.float32);  sum_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_57: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_297, memory_format = torch.contiguous_format);  convert_element_type_297 = None
        view_180: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_57, [1, 384, 576]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_12: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_180, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_54: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_208, memory_format = torch.contiguous_format);  primals_208 = None
        view_162: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_54, [1, 384, 576]);  clone_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_68: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_162, unsqueeze_74);  view_162 = unsqueeze_74 = None
        mul_251: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_180, sub_68)
        sum_13: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_251, [0, 2]);  mul_251 = None
        mul_252: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.001736111111111111);  sum_12 = None
        unsqueeze_75: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_252, 0);  mul_252 = None
        unsqueeze_76: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 2);  unsqueeze_75 = None
        mul_253: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.001736111111111111)
        mul_254: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_255: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        unsqueeze_77: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_255, 0);  mul_255 = None
        unsqueeze_78: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_77, 2);  unsqueeze_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_207: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_209, 0.07450538873672485);  primals_209 = None
        view_163: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_207, [-1]);  mul_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_256: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, view_163);  view_163 = None
        unsqueeze_79: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_256, 0);  mul_256 = None
        unsqueeze_80: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_79, 2);  unsqueeze_79 = None
        mul_257: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_78);  sub_68 = unsqueeze_78 = None
        sub_70: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_180, mul_257);  view_180 = mul_257 = None
        sub_71: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_76);  sub_70 = unsqueeze_76 = None
        mul_258: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_80);  sub_71 = unsqueeze_80 = None
        mul_259: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_109);  sum_13 = squeeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_181: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_259, [384, 1, 1, 1]);  mul_259 = None
        mul_260: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_181, 0.07450538873672485);  view_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_182: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_258, [384, 64, 3, 3]);  mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_299: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_126, torch.float32);  getitem_126 = None
        convert_element_type_251: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32);  convolution_75 = None
        sigmoid_14: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_251)
        mul_261: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_299, sigmoid_14);  convert_element_type_299 = None
        sub_72: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_14);  sigmoid_14 = None
        mul_262: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_251, sub_72);  convert_element_type_251 = sub_72 = None
        add_124: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_262, 1);  mul_262 = None
        mul_263: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, add_124);  mul_261 = add_124 = None
        convert_element_type_301: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_263, torch.bfloat16);  mul_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_14: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_301, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_301, convert_element_type_248, convert_element_type_250, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_301 = convert_element_type_248 = convert_element_type_250 = None
        getitem_129: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_5[0]
        getitem_130: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_302: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_130, torch.float32);  getitem_130 = None
        convert_element_type_303: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_58: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_302, memory_format = torch.contiguous_format);  convert_element_type_302 = None
        view_183: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_58, [1, 384, 576]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_15: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_183, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_52: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_205, memory_format = torch.contiguous_format);  primals_205 = None
        view_159: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_52, [1, 384, 576]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_73: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_159, unsqueeze_82);  view_159 = unsqueeze_82 = None
        mul_264: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_183, sub_73)
        sum_16: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 2]);  mul_264 = None
        mul_265: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.001736111111111111);  sum_15 = None
        unsqueeze_83: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_265, 0);  mul_265 = None
        unsqueeze_84: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
        mul_266: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.001736111111111111)
        mul_267: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, squeeze_107)
        mul_268: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
        unsqueeze_85: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_268, 0);  mul_268 = None
        unsqueeze_86: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_85, 2);  unsqueeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_204: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_206, 0.07450538873672485);  primals_206 = None
        view_160: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_204, [-1]);  mul_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_269: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, view_160);  view_160 = None
        unsqueeze_87: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
        unsqueeze_88: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 2);  unsqueeze_87 = None
        mul_270: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_86);  sub_73 = unsqueeze_86 = None
        sub_75: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_183, mul_270);  view_183 = mul_270 = None
        sub_76: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_84);  sub_75 = unsqueeze_84 = None
        mul_271: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_88);  sub_76 = unsqueeze_88 = None
        mul_272: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, squeeze_107);  sum_16 = squeeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_184: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_272, [384, 1, 1, 1]);  mul_272 = None
        mul_273: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_184, 0.07450538873672485);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_185: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_271, [384, 64, 3, 3]);  mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_304: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None
        convert_element_type_247: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_74, torch.float32);  convolution_74 = None
        sigmoid_15: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_247)
        mul_274: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_304, sigmoid_15);  convert_element_type_304 = None
        sub_77: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_15);  sigmoid_15 = None
        mul_275: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_247, sub_77);  convert_element_type_247 = sub_77 = None
        add_125: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_275, 1);  mul_275 = None
        mul_276: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_274, add_125);  mul_274 = add_125 = None
        convert_element_type_306: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_276, torch.bfloat16);  mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_17: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_306, [0, 2, 3])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_306, mul_200, convert_element_type_246, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_306 = mul_200 = convert_element_type_246 = None
        getitem_132: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = convolution_backward_6[0]
        getitem_133: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_307: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_133, torch.float32);  getitem_133 = None
        convert_element_type_308: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_186: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_307, [1, 384, 1536]);  convert_element_type_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_18: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_186, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_156: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_202, [1, 384, -1]);  primals_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_78: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_156, unsqueeze_90);  view_156 = unsqueeze_90 = None
        mul_277: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_186, sub_78)
        sum_19: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 2]);  mul_277 = None
        mul_278: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.0006510416666666666);  sum_18 = None
        unsqueeze_91: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
        unsqueeze_92: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_91, 2);  unsqueeze_91 = None
        mul_279: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.0006510416666666666)
        mul_280: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, squeeze_105)
        mul_281: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_279, mul_280);  mul_279 = mul_280 = None
        unsqueeze_93: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
        unsqueeze_94: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 2);  unsqueeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_201: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_203, 0.04562504637317021);  primals_203 = None
        view_157: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_201, [-1]);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_282: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, view_157);  view_157 = None
        unsqueeze_95: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_282, 0);  mul_282 = None
        unsqueeze_96: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        mul_283: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_94);  sub_78 = unsqueeze_94 = None
        sub_80: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_186, mul_283);  view_186 = mul_283 = None
        sub_81: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, unsqueeze_92);  sub_80 = unsqueeze_92 = None
        mul_284: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_96);  sub_81 = unsqueeze_96 = None
        mul_285: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_105);  sum_19 = squeeze_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_187: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_285, [384, 1, 1, 1]);  mul_285 = None
        mul_286: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_187, 0.04562504637317021);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_188: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_284, [384, 1536, 1, 1]);  mul_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_287: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_132, 0.9622504486493761);  getitem_132 = None
        convert_element_type_309: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_287, torch.float32);  mul_287 = None
        convert_element_type_243: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.float32);  add_109 = None
        sigmoid_16: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_243)
        mul_288: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_309, sigmoid_16);  convert_element_type_309 = None
        sub_82: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_16);  sigmoid_16 = None
        mul_289: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_243, sub_82);  convert_element_type_243 = sub_82 = None
        add_126: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_289, 1);  mul_289 = None
        mul_290: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_288, add_126);  mul_288 = add_126 = None
        convert_element_type_311: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16);  mul_290 = None
        add_127: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(getitem_114, convert_element_type_311);  getitem_114 = convert_element_type_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_291: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_127, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_292: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_291, 2.0);  mul_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_293: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_292, convolution_71);  convolution_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_10: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_294: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_292, sigmoid_10);  mul_292 = None
        sum_20: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_293, [2, 3], True, dtype = torch.float32);  mul_293 = None
        convert_element_type_312: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_20, torch.bfloat16);  sum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_313: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_312, torch.float32);  convert_element_type_312 = None
        convert_element_type_314: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_10, torch.float32);  sigmoid_10 = None
        sub_83: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_314)
        mul_295: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_314, sub_83);  convert_element_type_314 = sub_83 = None
        mul_296: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_313, mul_295);  convert_element_type_313 = mul_295 = None
        convert_element_type_315: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_21: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_315, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_315, relu_10, convert_element_type_242, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_315 = convert_element_type_242 = None
        getitem_135: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_7[0]
        getitem_136: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_316: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_136, torch.float32);  getitem_136 = None
        convert_element_type_317: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_21, torch.float32);  sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_1: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_1: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_1, full_default, getitem_135);  le_1 = getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_22: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_1, mean_10, convert_element_type_240, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_10 = convert_element_type_240 = None
        getitem_138: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_8[0]
        getitem_139: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_318: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_139, torch.float32);  getitem_139 = None
        convert_element_type_319: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_22, torch.float32);  sum_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_3: "bf16[128, 1536, 7, 7][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_138, [128, 1536, 7, 7]);  getitem_138 = None
        div_54: "bf16[128, 1536, 7, 7][75264, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
        add_128: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_294, div_54);  mul_294 = div_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_23: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_128, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(add_128, convert_element_type_236, convert_element_type_238, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_128 = convert_element_type_236 = convert_element_type_238 = None
        getitem_141: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_9[0]
        getitem_142: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_320: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_142, torch.float32);  getitem_142 = None
        convert_element_type_321: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_23, torch.float32);  sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_189: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_320, [1, 1536, 384]);  convert_element_type_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_24: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_189, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_153: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_195, [1, 1536, -1]);  primals_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_84: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_153, unsqueeze_98);  view_153 = unsqueeze_98 = None
        mul_297: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_189, sub_84)
        sum_25: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 2]);  mul_297 = None
        mul_298: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 0.0026041666666666665);  sum_24 = None
        unsqueeze_99: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_100: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 2);  unsqueeze_99 = None
        mul_299: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 0.0026041666666666665)
        mul_300: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_301: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
        unsqueeze_101: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
        unsqueeze_102: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_194: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.09125009274634042);  primals_196 = None
        view_154: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_194, [-1]);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_302: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, view_154);  view_154 = None
        unsqueeze_103: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
        unsqueeze_104: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 2);  unsqueeze_103 = None
        mul_303: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_102);  sub_84 = unsqueeze_102 = None
        sub_86: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_189, mul_303);  view_189 = mul_303 = None
        sub_87: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_100);  sub_86 = unsqueeze_100 = None
        mul_304: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_104);  sub_87 = unsqueeze_104 = None
        mul_305: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_103);  sum_25 = squeeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_190: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_305, [1536, 1, 1, 1]);  mul_305 = None
        mul_306: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_190, 0.09125009274634042);  view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_191: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_304, [1536, 384, 1, 1]);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_322: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None
        convert_element_type_235: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_70, torch.float32);  convolution_70 = None
        sigmoid_17: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_235)
        mul_307: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_322, sigmoid_17);  convert_element_type_322 = None
        sub_88: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_17);  sigmoid_17 = None
        mul_308: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, sub_88);  convert_element_type_235 = sub_88 = None
        add_129: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_308, 1);  mul_308 = None
        mul_309: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_307, add_129);  mul_307 = add_129 = None
        convert_element_type_324: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_309, torch.bfloat16);  mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_26: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_324, [0, 2, 3])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_324, convert_element_type_232, convert_element_type_234, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_324 = convert_element_type_232 = convert_element_type_234 = None
        getitem_144: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_10[0]
        getitem_145: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_325: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_145, torch.float32);  getitem_145 = None
        convert_element_type_326: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_59: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_325, memory_format = torch.contiguous_format);  convert_element_type_325 = None
        view_192: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [1, 384, 576]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_27: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_192, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_50: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_192, memory_format = torch.contiguous_format);  primals_192 = None
        view_150: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_50, [1, 384, 576]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_89: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_150, unsqueeze_106);  view_150 = unsqueeze_106 = None
        mul_310: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_192, sub_89)
        sum_28: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_310, [0, 2]);  mul_310 = None
        mul_311: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 0.001736111111111111);  sum_27 = None
        unsqueeze_107: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_311, 0);  mul_311 = None
        unsqueeze_108: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
        mul_312: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.001736111111111111)
        mul_313: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, squeeze_101)
        mul_314: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_109: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
        unsqueeze_110: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 2);  unsqueeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_191: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_193, 0.07450538873672485);  primals_193 = None
        view_151: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_191, [-1]);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_315: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, view_151);  view_151 = None
        unsqueeze_111: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
        unsqueeze_112: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 2);  unsqueeze_111 = None
        mul_316: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_110);  sub_89 = unsqueeze_110 = None
        sub_91: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_192, mul_316);  view_192 = mul_316 = None
        sub_92: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_108);  sub_91 = unsqueeze_108 = None
        mul_317: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_112);  sub_92 = unsqueeze_112 = None
        mul_318: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, squeeze_101);  sum_28 = squeeze_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_193: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_318, [384, 1, 1, 1]);  mul_318 = None
        mul_319: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_193, 0.07450538873672485);  view_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_194: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_317, [384, 64, 3, 3]);  mul_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_327: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None
        convert_element_type_231: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32);  convolution_69 = None
        sigmoid_18: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_231)
        mul_320: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_327, sigmoid_18);  convert_element_type_327 = None
        sub_93: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_18);  sigmoid_18 = None
        mul_321: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_231, sub_93);  convert_element_type_231 = sub_93 = None
        add_130: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_321, 1);  mul_321 = None
        mul_322: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_320, add_130);  mul_320 = add_130 = None
        convert_element_type_329: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_322, torch.bfloat16);  mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_29: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_329, [0, 2, 3])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_329, convert_element_type_228, convert_element_type_230, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_329 = convert_element_type_228 = convert_element_type_230 = None
        getitem_147: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_11[0]
        getitem_148: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_330: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_148, torch.float32);  getitem_148 = None
        convert_element_type_331: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_29, torch.float32);  sum_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_60: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_330, memory_format = torch.contiguous_format);  convert_element_type_330 = None
        view_195: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_60, [1, 384, 576]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_30: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_195, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_48: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_189, memory_format = torch.contiguous_format);  primals_189 = None
        view_147: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [1, 384, 576]);  clone_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_94: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_147, unsqueeze_114);  view_147 = unsqueeze_114 = None
        mul_323: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, sub_94)
        sum_31: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 2]);  mul_323 = None
        mul_324: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.001736111111111111);  sum_30 = None
        unsqueeze_115: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_324, 0);  mul_324 = None
        unsqueeze_116: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
        mul_325: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.001736111111111111)
        mul_326: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, squeeze_99)
        mul_327: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
        unsqueeze_117: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_327, 0);  mul_327 = None
        unsqueeze_118: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 2);  unsqueeze_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_188: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_190, 0.07450538873672485);  primals_190 = None
        view_148: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_188, [-1]);  mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_328: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, view_148);  view_148 = None
        unsqueeze_119: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_328, 0);  mul_328 = None
        unsqueeze_120: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
        mul_329: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_118);  sub_94 = unsqueeze_118 = None
        sub_96: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_195, mul_329);  view_195 = mul_329 = None
        sub_97: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_116);  sub_96 = unsqueeze_116 = None
        mul_330: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_120);  sub_97 = unsqueeze_120 = None
        mul_331: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_99);  sum_31 = squeeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_196: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_331, [384, 1, 1, 1]);  mul_331 = None
        mul_332: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_196, 0.07450538873672485);  view_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_197: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_330, [384, 64, 3, 3]);  mul_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_332: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
        convert_element_type_227: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_68, torch.float32);  convolution_68 = None
        sigmoid_19: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_227)
        mul_333: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_332, sigmoid_19);  convert_element_type_332 = None
        sub_98: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_19);  sigmoid_19 = None
        mul_334: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_227, sub_98);  convert_element_type_227 = sub_98 = None
        add_131: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_334, 1);  mul_334 = None
        mul_335: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, add_131);  mul_333 = add_131 = None
        convert_element_type_334: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_335, torch.bfloat16);  mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_32: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_334, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_334, mul_184, convert_element_type_226, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_334 = mul_184 = convert_element_type_226 = None
        getitem_150: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = convolution_backward_12[0]
        getitem_151: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_335: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_151, torch.float32);  getitem_151 = None
        convert_element_type_336: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_32, torch.float32);  sum_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_198: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_335, [1, 384, 1536]);  convert_element_type_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_33: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_198, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_144: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_186, [1, 384, -1]);  primals_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_99: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_144, unsqueeze_122);  view_144 = unsqueeze_122 = None
        mul_336: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_198, sub_99)
        sum_34: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2]);  mul_336 = None
        mul_337: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 0.0006510416666666666);  sum_33 = None
        unsqueeze_123: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_337, 0);  mul_337 = None
        unsqueeze_124: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 2);  unsqueeze_123 = None
        mul_338: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 0.0006510416666666666)
        mul_339: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_340: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
        unsqueeze_125: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_340, 0);  mul_340 = None
        unsqueeze_126: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_185: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_187, 0.04562504637317021);  primals_187 = None
        view_145: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_185, [-1]);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_341: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, view_145);  view_145 = None
        unsqueeze_127: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
        unsqueeze_128: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
        mul_342: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_126);  sub_99 = unsqueeze_126 = None
        sub_101: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_198, mul_342);  view_198 = mul_342 = None
        sub_102: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_124);  sub_101 = unsqueeze_124 = None
        mul_343: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_128);  sub_102 = unsqueeze_128 = None
        mul_344: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, squeeze_97);  sum_34 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_199: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_344, [384, 1, 1, 1]);  mul_344 = None
        mul_345: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_199, 0.04562504637317021);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_200: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_343, [384, 1536, 1, 1]);  mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_346: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_150, 0.9805806756909201);  getitem_150 = None
        convert_element_type_337: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_346, torch.float32);  mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_9: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_181: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_65, sigmoid_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_182: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, 2.0);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_183: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, 0.2);  mul_182 = None
        add_100: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_183, convolution_61);  mul_183 = convolution_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_223: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.float32);  add_100 = None
        sigmoid_20: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_223)
        mul_347: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_337, sigmoid_20);  convert_element_type_337 = None
        sub_103: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_20);  sigmoid_20 = None
        mul_348: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_223, sub_103);  convert_element_type_223 = sub_103 = None
        add_132: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_348, 1);  mul_348 = None
        mul_349: "f32[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_347, add_132);  mul_347 = add_132 = None
        convert_element_type_339: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_349, torch.bfloat16);  mul_349 = None
        add_133: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_127, convert_element_type_339);  add_127 = convert_element_type_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_350: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_133, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_351: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, 2.0);  mul_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_352: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, convolution_65);  convolution_65 = None
        mul_353: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, sigmoid_9);  mul_351 = None
        sum_35: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_352, [2, 3], True, dtype = torch.float32);  mul_352 = None
        convert_element_type_340: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_35, torch.bfloat16);  sum_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_341: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_340, torch.float32);  convert_element_type_340 = None
        convert_element_type_342: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_9, torch.float32);  sigmoid_9 = None
        sub_104: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_342)
        mul_354: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_342, sub_104);  convert_element_type_342 = sub_104 = None
        mul_355: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_341, mul_354);  convert_element_type_341 = mul_354 = None
        convert_element_type_343: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_355, torch.bfloat16);  mul_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_36: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_343, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_343, relu_9, convert_element_type_222, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_343 = convert_element_type_222 = None
        getitem_153: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_13[0]
        getitem_154: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_344: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_154, torch.float32);  getitem_154 = None
        convert_element_type_345: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_36, torch.float32);  sum_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_2: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_2: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_2, full_default, getitem_153);  le_2 = getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_37: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(where_2, mean_9, convert_element_type_220, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_2 = mean_9 = convert_element_type_220 = None
        getitem_156: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_14[0]
        getitem_157: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_346: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_157, torch.float32);  getitem_157 = None
        convert_element_type_347: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_37, torch.float32);  sum_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_4: "bf16[128, 1536, 7, 7][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_156, [128, 1536, 7, 7]);  getitem_156 = None
        div_55: "bf16[128, 1536, 7, 7][75264, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
        add_134: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_353, div_55);  mul_353 = div_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_38: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_134, [0, 2, 3])
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(add_134, convert_element_type_216, convert_element_type_218, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_134 = convert_element_type_216 = convert_element_type_218 = None
        getitem_159: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_15[0]
        getitem_160: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_348: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_160, torch.float32);  getitem_160 = None
        convert_element_type_349: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_38, torch.float32);  sum_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_201: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_348, [1, 1536, 384]);  convert_element_type_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_39: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_201, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_141: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_179, [1, 1536, -1]);  primals_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_105: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_141, unsqueeze_130);  view_141 = unsqueeze_130 = None
        mul_356: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_201, sub_105)
        sum_40: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 2]);  mul_356 = None
        mul_357: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 0.0026041666666666665);  sum_39 = None
        unsqueeze_131: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_357, 0);  mul_357 = None
        unsqueeze_132: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
        mul_358: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 0.0026041666666666665)
        mul_359: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, squeeze_95)
        mul_360: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        unsqueeze_133: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_360, 0);  mul_360 = None
        unsqueeze_134: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_178: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_180, 0.09125009274634042);  primals_180 = None
        view_142: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_178, [-1]);  mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_361: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, view_142);  view_142 = None
        unsqueeze_135: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_361, 0);  mul_361 = None
        unsqueeze_136: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 2);  unsqueeze_135 = None
        mul_362: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_134);  sub_105 = unsqueeze_134 = None
        sub_107: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_201, mul_362);  view_201 = mul_362 = None
        sub_108: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_132);  sub_107 = unsqueeze_132 = None
        mul_363: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_136);  sub_108 = unsqueeze_136 = None
        mul_364: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, squeeze_95);  sum_40 = squeeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_202: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_364, [1536, 1, 1, 1]);  mul_364 = None
        mul_365: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_202, 0.09125009274634042);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_203: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_363, [1536, 384, 1, 1]);  mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_350: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None
        convert_element_type_215: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_64, torch.float32);  convolution_64 = None
        sigmoid_21: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_215)
        mul_366: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_350, sigmoid_21);  convert_element_type_350 = None
        sub_109: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_21);  sigmoid_21 = None
        mul_367: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_215, sub_109);  convert_element_type_215 = sub_109 = None
        add_135: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_367, 1);  mul_367 = None
        mul_368: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_366, add_135);  mul_366 = add_135 = None
        convert_element_type_352: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_368, torch.bfloat16);  mul_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_41: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_352, [0, 2, 3])
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_352, convert_element_type_212, convert_element_type_214, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_352 = convert_element_type_212 = convert_element_type_214 = None
        getitem_162: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = convolution_backward_16[0]
        getitem_163: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_353: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_163, torch.float32);  getitem_163 = None
        convert_element_type_354: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_41, torch.float32);  sum_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_61: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_353, memory_format = torch.contiguous_format);  convert_element_type_353 = None
        view_204: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_61, [1, 384, 576]);  clone_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_42: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_204, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_46: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_176, memory_format = torch.contiguous_format);  primals_176 = None
        view_138: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [1, 384, 576]);  clone_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_110: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_138, unsqueeze_138);  view_138 = unsqueeze_138 = None
        mul_369: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_204, sub_110)
        sum_43: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 2]);  mul_369 = None
        mul_370: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 0.001736111111111111);  sum_42 = None
        unsqueeze_139: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
        unsqueeze_140: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        mul_371: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 0.001736111111111111)
        mul_372: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, squeeze_93)
        mul_373: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
        unsqueeze_141: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
        unsqueeze_142: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 2);  unsqueeze_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_175: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_177, 0.07450538873672485);  primals_177 = None
        view_139: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_175, [-1]);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_374: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, view_139);  view_139 = None
        unsqueeze_143: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
        unsqueeze_144: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
        mul_375: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_142);  sub_110 = unsqueeze_142 = None
        sub_112: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_204, mul_375);  view_204 = mul_375 = None
        sub_113: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, unsqueeze_140);  sub_112 = unsqueeze_140 = None
        mul_376: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_144);  sub_113 = unsqueeze_144 = None
        mul_377: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_93);  sum_43 = squeeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_205: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_377, [384, 1, 1, 1]);  mul_377 = None
        mul_378: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_205, 0.07450538873672485);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_206: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_376, [384, 64, 3, 3]);  mul_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_355: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_162, torch.float32);  getitem_162 = None
        convert_element_type_211: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32);  convolution_63 = None
        sigmoid_22: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_211)
        mul_379: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_355, sigmoid_22);  convert_element_type_355 = None
        sub_114: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_22);  sigmoid_22 = None
        mul_380: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_211, sub_114);  convert_element_type_211 = sub_114 = None
        add_136: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_380, 1);  mul_380 = None
        mul_381: "f32[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_379, add_136);  mul_379 = add_136 = None
        convert_element_type_357: "bf16[128, 384, 7, 7][18816, 1, 2688, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_381, torch.bfloat16);  mul_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_44: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_357, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_357, convert_element_type_208, convert_element_type_210, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_357 = convert_element_type_208 = convert_element_type_210 = None
        getitem_165: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_17[0]
        getitem_166: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_358: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_166, torch.float32);  getitem_166 = None
        convert_element_type_359: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_44, torch.float32);  sum_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_62: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_358, memory_format = torch.contiguous_format);  convert_element_type_358 = None
        view_207: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_62, [1, 384, 576]);  clone_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_45: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_207, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_44: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_173, memory_format = torch.contiguous_format);  primals_173 = None
        view_135: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [1, 384, 576]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_115: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_135, unsqueeze_146);  view_135 = unsqueeze_146 = None
        mul_382: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_207, sub_115)
        sum_46: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 2]);  mul_382 = None
        mul_383: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 0.001736111111111111);  sum_45 = None
        unsqueeze_147: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
        unsqueeze_148: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 2);  unsqueeze_147 = None
        mul_384: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 0.001736111111111111)
        mul_385: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_386: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
        unsqueeze_149: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
        unsqueeze_150: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_172: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_174, 0.07450538873672485);  primals_174 = None
        view_136: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_172, [-1]);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_387: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, view_136);  view_136 = None
        unsqueeze_151: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
        unsqueeze_152: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        mul_388: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_150);  sub_115 = unsqueeze_150 = None
        sub_117: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_207, mul_388);  view_207 = mul_388 = None
        sub_118: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_148);  sub_117 = unsqueeze_148 = None
        mul_389: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_152);  sub_118 = unsqueeze_152 = None
        mul_390: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, squeeze_91);  sum_46 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_208: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_390, [384, 1, 1, 1]);  mul_390 = None
        mul_391: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_208, 0.07450538873672485);  view_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_209: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_389, [384, 64, 3, 3]);  mul_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_360: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None
        convert_element_type_207: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32);  convolution_62 = None
        sigmoid_23: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_207)
        mul_392: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_360, sigmoid_23);  convert_element_type_360 = None
        sub_119: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_23);  sigmoid_23 = None
        mul_393: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_207, sub_119);  convert_element_type_207 = sub_119 = None
        add_137: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_393, 1);  mul_393 = None
        mul_394: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, add_137);  mul_392 = add_137 = None
        convert_element_type_362: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_394, torch.bfloat16);  mul_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_47: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_362, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_362, mul_165, convert_element_type_206, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_362 = convert_element_type_206 = None
        getitem_168: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_18[0]
        getitem_169: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_363: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_169, torch.float32);  getitem_169 = None
        convert_element_type_364: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_47, torch.float32);  sum_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_210: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_363, [1, 384, 1536]);  convert_element_type_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_48: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_210, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_132: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_170, [1, 384, -1]);  primals_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_120: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_132, unsqueeze_154);  view_132 = unsqueeze_154 = None
        mul_395: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_210, sub_120)
        sum_49: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 2]);  mul_395 = None
        mul_396: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 0.0006510416666666666);  sum_48 = None
        unsqueeze_155: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
        unsqueeze_156: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
        mul_397: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 0.0006510416666666666)
        mul_398: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, squeeze_89)
        mul_399: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
        unsqueeze_157: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_399, 0);  mul_399 = None
        unsqueeze_158: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_169: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_171, 0.04562504637317021);  primals_171 = None
        view_133: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_169, [-1]);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_400: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, view_133);  view_133 = None
        unsqueeze_159: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
        unsqueeze_160: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
        mul_401: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_158);  sub_120 = unsqueeze_158 = None
        sub_122: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_210, mul_401);  view_210 = mul_401 = None
        sub_123: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_156);  sub_122 = unsqueeze_156 = None
        mul_402: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_160);  sub_123 = unsqueeze_160 = None
        mul_403: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_89);  sum_49 = squeeze_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_211: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_403, [384, 1, 1, 1]);  mul_403 = None
        mul_404: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_211, 0.04562504637317021);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_212: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_402, [384, 1536, 1, 1]);  mul_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_50: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_133, [0, 2, 3])
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_133, avg_pool2d_2, convert_element_type_204, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_133 = avg_pool2d_2 = convert_element_type_204 = None
        getitem_171: "bf16[128, 1536, 7, 7][75264, 1, 10752, 1536]cuda:0" = convolution_backward_19[0]
        getitem_172: "bf16[1536, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_365: "f32[1536, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_172, torch.float32);  getitem_172 = None
        convert_element_type_366: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_50, torch.float32);  sum_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_213: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_365, [1, 1536, 1536]);  convert_element_type_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_51: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_213, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_129: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_167, [1, 1536, -1]);  primals_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_124: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_129, unsqueeze_162);  view_129 = unsqueeze_162 = None
        mul_405: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_213, sub_124)
        sum_52: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2]);  mul_405 = None
        mul_406: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 0.0006510416666666666);  sum_51 = None
        unsqueeze_163: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
        unsqueeze_164: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        mul_407: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 0.0006510416666666666)
        mul_408: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, squeeze_87)
        mul_409: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
        unsqueeze_165: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
        unsqueeze_166: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_166: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_168, 0.04562504637317021);  primals_168 = None
        view_130: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_166, [-1]);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_410: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, view_130);  view_130 = None
        unsqueeze_167: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
        unsqueeze_168: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
        mul_411: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_166);  sub_124 = unsqueeze_166 = None
        sub_126: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_213, mul_411);  view_213 = mul_411 = None
        sub_127: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_164);  sub_126 = unsqueeze_164 = None
        mul_412: "f32[1, 1536, 1536][2359296, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_168);  sub_127 = unsqueeze_168 = None
        mul_413: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, squeeze_87);  sum_52 = squeeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_214: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_413, [1536, 1, 1, 1]);  mul_413 = None
        mul_414: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_214, 0.04562504637317021);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_215: "f32[1536, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_412, [1536, 1536, 1, 1]);  mul_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_backward: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_171, mul_165, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_171 = mul_165 = None
        add_138: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(getitem_168, avg_pool2d_backward);  getitem_168 = avg_pool2d_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_415: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_138, 0.8980265101338745);  add_138 = None
        convert_element_type_367: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_415, torch.float32);  mul_415 = None
        convert_element_type_201: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_90, torch.float32);  add_90 = None
        sigmoid_24: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_201)
        mul_416: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_367, sigmoid_24);  convert_element_type_367 = None
        sub_128: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_24);  sigmoid_24 = None
        mul_417: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_201, sub_128);  convert_element_type_201 = sub_128 = None
        add_139: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_417, 1);  mul_417 = None
        mul_418: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, add_139);  mul_416 = add_139 = None
        convert_element_type_369: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_418, torch.bfloat16);  mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_419: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_369, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_420: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_419, 2.0);  mul_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_421: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, convolution_58);  convolution_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_8: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_60);  convolution_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_422: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, sigmoid_8);  mul_420 = None
        sum_53: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_421, [2, 3], True, dtype = torch.float32);  mul_421 = None
        convert_element_type_370: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_53, torch.bfloat16);  sum_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_371: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_370, torch.float32);  convert_element_type_370 = None
        convert_element_type_372: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_8, torch.float32);  sigmoid_8 = None
        sub_129: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_372)
        mul_423: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_372, sub_129);  convert_element_type_372 = sub_129 = None
        mul_424: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_371, mul_423);  convert_element_type_371 = mul_423 = None
        convert_element_type_373: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_424, torch.bfloat16);  mul_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_54: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_373, [0, 2, 3])
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_373, relu_8, convert_element_type_200, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_373 = convert_element_type_200 = None
        getitem_174: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_20[0]
        getitem_175: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_374: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_175, torch.float32);  getitem_175 = None
        convert_element_type_375: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_54, torch.float32);  sum_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_3: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_3: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_3, full_default, getitem_174);  le_3 = getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_55: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(where_3, mean_8, convert_element_type_198, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_3 = mean_8 = convert_element_type_198 = None
        getitem_177: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_21[0]
        getitem_178: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_376: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_178, torch.float32);  getitem_178 = None
        convert_element_type_377: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_55, torch.float32);  sum_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_5: "bf16[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_177, [128, 1536, 14, 14]);  getitem_177 = None
        div_56: "bf16[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
        add_140: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_422, div_56);  mul_422 = div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_56: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_140, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_140, convert_element_type_194, convert_element_type_196, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_140 = convert_element_type_194 = convert_element_type_196 = None
        getitem_180: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_22[0]
        getitem_181: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_378: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_181, torch.float32);  getitem_181 = None
        convert_element_type_379: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_56, torch.float32);  sum_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_216: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_378, [1, 1536, 384]);  convert_element_type_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_57: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_216, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_126: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_160, [1, 1536, -1]);  primals_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_130: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_126, unsqueeze_170);  view_126 = unsqueeze_170 = None
        mul_425: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_216, sub_130)
        sum_58: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 2]);  mul_425 = None
        mul_426: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 0.0026041666666666665);  sum_57 = None
        unsqueeze_171: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
        unsqueeze_172: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
        mul_427: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 0.0026041666666666665)
        mul_428: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_429: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_427, mul_428);  mul_427 = mul_428 = None
        unsqueeze_173: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_429, 0);  mul_429 = None
        unsqueeze_174: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_159: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_161, 0.09125009274634042);  primals_161 = None
        view_127: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_159, [-1]);  mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_430: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, view_127);  view_127 = None
        unsqueeze_175: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_430, 0);  mul_430 = None
        unsqueeze_176: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        mul_431: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_174);  sub_130 = unsqueeze_174 = None
        sub_132: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_216, mul_431);  view_216 = mul_431 = None
        sub_133: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_172);  sub_132 = unsqueeze_172 = None
        mul_432: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_176);  sub_133 = unsqueeze_176 = None
        mul_433: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, squeeze_85);  sum_58 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_217: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_433, [1536, 1, 1, 1]);  mul_433 = None
        mul_434: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_217, 0.09125009274634042);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_218: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_432, [1536, 384, 1, 1]);  mul_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_380: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_180, torch.float32);  getitem_180 = None
        convert_element_type_193: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32);  convolution_57 = None
        sigmoid_25: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_193)
        mul_435: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, sigmoid_25);  convert_element_type_380 = None
        sub_134: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_25);  sigmoid_25 = None
        mul_436: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_193, sub_134);  convert_element_type_193 = sub_134 = None
        add_141: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_436, 1);  mul_436 = None
        mul_437: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_435, add_141);  mul_435 = add_141 = None
        convert_element_type_382: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_437, torch.bfloat16);  mul_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_59: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_382, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_382, convert_element_type_190, convert_element_type_192, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_382 = convert_element_type_190 = convert_element_type_192 = None
        getitem_183: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_23[0]
        getitem_184: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_383: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_184, torch.float32);  getitem_184 = None
        convert_element_type_384: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_59, torch.float32);  sum_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_63: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_383, memory_format = torch.contiguous_format);  convert_element_type_383 = None
        view_219: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [1, 384, 576]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_60: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_219, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_42: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_157, memory_format = torch.contiguous_format);  primals_157 = None
        view_123: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [1, 384, 576]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_135: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_123, unsqueeze_178);  view_123 = unsqueeze_178 = None
        mul_438: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_219, sub_135)
        sum_61: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 2]);  mul_438 = None
        mul_439: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 0.001736111111111111);  sum_60 = None
        unsqueeze_179: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
        unsqueeze_180: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
        mul_440: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 0.001736111111111111)
        mul_441: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, squeeze_83)
        mul_442: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, mul_441);  mul_440 = mul_441 = None
        unsqueeze_181: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
        unsqueeze_182: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_156: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_158, 0.07450538873672485);  primals_158 = None
        view_124: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_156, [-1]);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_443: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, view_124);  view_124 = None
        unsqueeze_183: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
        unsqueeze_184: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
        mul_444: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_182);  sub_135 = unsqueeze_182 = None
        sub_137: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_219, mul_444);  view_219 = mul_444 = None
        sub_138: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_180);  sub_137 = unsqueeze_180 = None
        mul_445: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_184);  sub_138 = unsqueeze_184 = None
        mul_446: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_83);  sum_61 = squeeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_220: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_446, [384, 1, 1, 1]);  mul_446 = None
        mul_447: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_220, 0.07450538873672485);  view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_221: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_445, [384, 64, 3, 3]);  mul_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_385: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_183, torch.float32);  getitem_183 = None
        convert_element_type_189: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32);  convolution_56 = None
        sigmoid_26: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_189)
        mul_448: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_385, sigmoid_26);  convert_element_type_385 = None
        sub_139: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_26);  sigmoid_26 = None
        mul_449: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_189, sub_139);  convert_element_type_189 = sub_139 = None
        add_142: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_449, 1);  mul_449 = None
        mul_450: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, add_142);  mul_448 = add_142 = None
        convert_element_type_387: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_450, torch.bfloat16);  mul_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_62: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_387, [0, 2, 3])
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_387, convert_element_type_186, convert_element_type_188, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_387 = convert_element_type_186 = convert_element_type_188 = None
        getitem_186: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_24[0]
        getitem_187: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_388: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_187, torch.float32);  getitem_187 = None
        convert_element_type_389: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_62, torch.float32);  sum_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_64: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_388, memory_format = torch.contiguous_format);  convert_element_type_388 = None
        view_222: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_64, [1, 384, 576]);  clone_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_63: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_222, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_40: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_154, memory_format = torch.contiguous_format);  primals_154 = None
        view_120: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [1, 384, 576]);  clone_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_140: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_120, unsqueeze_186);  view_120 = unsqueeze_186 = None
        mul_451: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_222, sub_140)
        sum_64: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [0, 2]);  mul_451 = None
        mul_452: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 0.001736111111111111);  sum_63 = None
        unsqueeze_187: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
        unsqueeze_188: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
        mul_453: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 0.001736111111111111)
        mul_454: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, squeeze_81)
        mul_455: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
        unsqueeze_189: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
        unsqueeze_190: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_153: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_155, 0.07450538873672485);  primals_155 = None
        view_121: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_153, [-1]);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_456: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, view_121);  view_121 = None
        unsqueeze_191: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
        unsqueeze_192: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
        mul_457: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_190);  sub_140 = unsqueeze_190 = None
        sub_142: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_222, mul_457);  view_222 = mul_457 = None
        sub_143: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_188);  sub_142 = unsqueeze_188 = None
        mul_458: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_192);  sub_143 = unsqueeze_192 = None
        mul_459: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, squeeze_81);  sum_64 = squeeze_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_223: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_459, [384, 1, 1, 1]);  mul_459 = None
        mul_460: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_223, 0.07450538873672485);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_224: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_458, [384, 64, 3, 3]);  mul_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_390: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_186, torch.float32);  getitem_186 = None
        convert_element_type_185: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        sigmoid_27: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_185)
        mul_461: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_390, sigmoid_27);  convert_element_type_390 = None
        sub_144: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_27);  sigmoid_27 = None
        mul_462: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_185, sub_144);  convert_element_type_185 = sub_144 = None
        add_143: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_462, 1);  mul_462 = None
        mul_463: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_461, add_143);  mul_461 = add_143 = None
        convert_element_type_392: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_463, torch.bfloat16);  mul_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_65: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_392, [0, 2, 3])
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_392, mul_149, convert_element_type_184, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_392 = mul_149 = convert_element_type_184 = None
        getitem_189: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_25[0]
        getitem_190: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_393: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_190, torch.float32);  getitem_190 = None
        convert_element_type_394: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_65, torch.float32);  sum_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_225: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_393, [1, 384, 1536]);  convert_element_type_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_66: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_225, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_117: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_151, [1, 384, -1]);  primals_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_145: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_117, unsqueeze_194);  view_117 = unsqueeze_194 = None
        mul_464: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_225, sub_145)
        sum_67: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2]);  mul_464 = None
        mul_465: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 0.0006510416666666666);  sum_66 = None
        unsqueeze_195: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
        unsqueeze_196: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
        mul_466: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 0.0006510416666666666)
        mul_467: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_468: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        unsqueeze_197: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
        unsqueeze_198: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_150: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_152, 0.04562504637317021);  primals_152 = None
        view_118: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_150, [-1]);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_469: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, view_118);  view_118 = None
        unsqueeze_199: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
        unsqueeze_200: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        mul_470: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_198);  sub_145 = unsqueeze_198 = None
        sub_147: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_225, mul_470);  view_225 = mul_470 = None
        sub_148: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_196);  sub_147 = unsqueeze_196 = None
        mul_471: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_200);  sub_148 = unsqueeze_200 = None
        mul_472: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_79);  sum_67 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_226: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_472, [384, 1, 1, 1]);  mul_472 = None
        mul_473: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_226, 0.04562504637317021);  view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_227: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_471, [384, 1536, 1, 1]);  mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_474: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_189, 0.9128709291752768);  getitem_189 = None
        convert_element_type_395: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_474, torch.float32);  mul_474 = None
        convert_element_type_181: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.float32);  add_81 = None
        sigmoid_28: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_181)
        mul_475: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_395, sigmoid_28);  convert_element_type_395 = None
        sub_149: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_28);  sigmoid_28 = None
        mul_476: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_181, sub_149);  convert_element_type_181 = sub_149 = None
        add_144: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_476, 1);  mul_476 = None
        mul_477: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_475, add_144);  mul_475 = add_144 = None
        convert_element_type_397: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_477, torch.bfloat16);  mul_477 = None
        add_145: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_369, convert_element_type_397);  convert_element_type_369 = convert_element_type_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_478: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_145, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_479: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_478, 2.0);  mul_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_480: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_479, convolution_52);  convolution_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_7: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_54);  convolution_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_481: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_479, sigmoid_7);  mul_479 = None
        sum_68: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_480, [2, 3], True, dtype = torch.float32);  mul_480 = None
        convert_element_type_398: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_68, torch.bfloat16);  sum_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_399: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_398, torch.float32);  convert_element_type_398 = None
        convert_element_type_400: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_7, torch.float32);  sigmoid_7 = None
        sub_150: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_400)
        mul_482: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_400, sub_150);  convert_element_type_400 = sub_150 = None
        mul_483: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_399, mul_482);  convert_element_type_399 = mul_482 = None
        convert_element_type_401: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_483, torch.bfloat16);  mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_69: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_401, [0, 2, 3])
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_401, relu_7, convert_element_type_180, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_401 = convert_element_type_180 = None
        getitem_192: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_26[0]
        getitem_193: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_402: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_193, torch.float32);  getitem_193 = None
        convert_element_type_403: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_69, torch.float32);  sum_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_4: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_4: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_192);  le_4 = getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_70: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_4, mean_7, convert_element_type_178, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = mean_7 = convert_element_type_178 = None
        getitem_195: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_27[0]
        getitem_196: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_404: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_196, torch.float32);  getitem_196 = None
        convert_element_type_405: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_70, torch.float32);  sum_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_6: "bf16[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_195, [128, 1536, 14, 14]);  getitem_195 = None
        div_57: "bf16[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
        add_146: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_481, div_57);  mul_481 = div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_71: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_146, [0, 2, 3])
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_146, convert_element_type_174, convert_element_type_176, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_146 = convert_element_type_174 = convert_element_type_176 = None
        getitem_198: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_28[0]
        getitem_199: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_406: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_199, torch.float32);  getitem_199 = None
        convert_element_type_407: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_71, torch.float32);  sum_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_228: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_406, [1, 1536, 384]);  convert_element_type_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_72: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_228, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_114: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_144, [1, 1536, -1]);  primals_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_151: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_114, unsqueeze_202);  view_114 = unsqueeze_202 = None
        mul_484: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_228, sub_151)
        sum_73: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_484, [0, 2]);  mul_484 = None
        mul_485: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 0.0026041666666666665);  sum_72 = None
        unsqueeze_203: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
        unsqueeze_204: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        mul_486: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 0.0026041666666666665)
        mul_487: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, squeeze_77)
        mul_488: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, mul_487);  mul_486 = mul_487 = None
        unsqueeze_205: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
        unsqueeze_206: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_143: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_145, 0.09125009274634042);  primals_145 = None
        view_115: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_143, [-1]);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_489: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, view_115);  view_115 = None
        unsqueeze_207: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
        unsqueeze_208: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
        mul_490: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_206);  sub_151 = unsqueeze_206 = None
        sub_153: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_228, mul_490);  view_228 = mul_490 = None
        sub_154: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_204);  sub_153 = unsqueeze_204 = None
        mul_491: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_208);  sub_154 = unsqueeze_208 = None
        mul_492: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_77);  sum_73 = squeeze_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_229: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_492, [1536, 1, 1, 1]);  mul_492 = None
        mul_493: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_229, 0.09125009274634042);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_230: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_491, [1536, 384, 1, 1]);  mul_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_408: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_198, torch.float32);  getitem_198 = None
        convert_element_type_173: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sigmoid_29: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_173)
        mul_494: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_408, sigmoid_29);  convert_element_type_408 = None
        sub_155: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_29);  sigmoid_29 = None
        mul_495: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_173, sub_155);  convert_element_type_173 = sub_155 = None
        add_147: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_495, 1);  mul_495 = None
        mul_496: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_494, add_147);  mul_494 = add_147 = None
        convert_element_type_410: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_496, torch.bfloat16);  mul_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_74: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_410, [0, 2, 3])
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_410, convert_element_type_170, convert_element_type_172, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_410 = convert_element_type_170 = convert_element_type_172 = None
        getitem_201: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_29[0]
        getitem_202: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_411: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_202, torch.float32);  getitem_202 = None
        convert_element_type_412: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_74, torch.float32);  sum_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_65: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_411, memory_format = torch.contiguous_format);  convert_element_type_411 = None
        view_231: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_65, [1, 384, 576]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_75: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_231, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_38: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_141, memory_format = torch.contiguous_format);  primals_141 = None
        view_111: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [1, 384, 576]);  clone_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_156: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_111, unsqueeze_210);  view_111 = unsqueeze_210 = None
        mul_497: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_231, sub_156)
        sum_76: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 2]);  mul_497 = None
        mul_498: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 0.001736111111111111);  sum_75 = None
        unsqueeze_211: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
        unsqueeze_212: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        mul_499: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 0.001736111111111111)
        mul_500: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, squeeze_75)
        mul_501: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_499, mul_500);  mul_499 = mul_500 = None
        unsqueeze_213: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
        unsqueeze_214: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_140: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.07450538873672485);  primals_142 = None
        view_112: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_140, [-1]);  mul_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_502: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, view_112);  view_112 = None
        unsqueeze_215: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
        unsqueeze_216: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        mul_503: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_214);  sub_156 = unsqueeze_214 = None
        sub_158: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_231, mul_503);  view_231 = mul_503 = None
        sub_159: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_212);  sub_158 = unsqueeze_212 = None
        mul_504: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_216);  sub_159 = unsqueeze_216 = None
        mul_505: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, squeeze_75);  sum_76 = squeeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_232: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_505, [384, 1, 1, 1]);  mul_505 = None
        mul_506: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_232, 0.07450538873672485);  view_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_233: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_504, [384, 64, 3, 3]);  mul_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_413: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None
        convert_element_type_169: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        sigmoid_30: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_169)
        mul_507: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, sigmoid_30);  convert_element_type_413 = None
        sub_160: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_30);  sigmoid_30 = None
        mul_508: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, sub_160);  convert_element_type_169 = sub_160 = None
        add_148: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_508, 1);  mul_508 = None
        mul_509: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_507, add_148);  mul_507 = add_148 = None
        convert_element_type_415: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_509, torch.bfloat16);  mul_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_77: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_415, [0, 2, 3])
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_415, convert_element_type_166, convert_element_type_168, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_415 = convert_element_type_166 = convert_element_type_168 = None
        getitem_204: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_30[0]
        getitem_205: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_416: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_205, torch.float32);  getitem_205 = None
        convert_element_type_417: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_77, torch.float32);  sum_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_66: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_416, memory_format = torch.contiguous_format);  convert_element_type_416 = None
        view_234: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_66, [1, 384, 576]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_78: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_234, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_36: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_138, memory_format = torch.contiguous_format);  primals_138 = None
        view_108: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [1, 384, 576]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_161: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_108, unsqueeze_218);  view_108 = unsqueeze_218 = None
        mul_510: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_234, sub_161)
        sum_79: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 2]);  mul_510 = None
        mul_511: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 0.001736111111111111);  sum_78 = None
        unsqueeze_219: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_220: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
        mul_512: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 0.001736111111111111)
        mul_513: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_514: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
        unsqueeze_221: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
        unsqueeze_222: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_137: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_139, 0.07450538873672485);  primals_139 = None
        view_109: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_137, [-1]);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_515: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, view_109);  view_109 = None
        unsqueeze_223: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
        unsqueeze_224: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        mul_516: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_222);  sub_161 = unsqueeze_222 = None
        sub_163: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_234, mul_516);  view_234 = mul_516 = None
        sub_164: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_220);  sub_163 = unsqueeze_220 = None
        mul_517: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_224);  sub_164 = unsqueeze_224 = None
        mul_518: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_73);  sum_79 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_235: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_518, [384, 1, 1, 1]);  mul_518 = None
        mul_519: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_235, 0.07450538873672485);  view_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_236: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_517, [384, 64, 3, 3]);  mul_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_418: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_204, torch.float32);  getitem_204 = None
        convert_element_type_165: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        sigmoid_31: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_165)
        mul_520: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_418, sigmoid_31);  convert_element_type_418 = None
        sub_165: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_31);  sigmoid_31 = None
        mul_521: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, sub_165);  convert_element_type_165 = sub_165 = None
        add_149: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_521, 1);  mul_521 = None
        mul_522: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_520, add_149);  mul_520 = add_149 = None
        convert_element_type_420: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_522, torch.bfloat16);  mul_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_80: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_420, [0, 2, 3])
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_420, mul_133, convert_element_type_164, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_420 = mul_133 = convert_element_type_164 = None
        getitem_207: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_31[0]
        getitem_208: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_421: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_208, torch.float32);  getitem_208 = None
        convert_element_type_422: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_80, torch.float32);  sum_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_237: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_421, [1, 384, 1536]);  convert_element_type_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_81: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_237, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_105: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_135, [1, 384, -1]);  primals_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_166: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_105, unsqueeze_226);  view_105 = unsqueeze_226 = None
        mul_523: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_237, sub_166)
        sum_82: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_523, [0, 2]);  mul_523 = None
        mul_524: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 0.0006510416666666666);  sum_81 = None
        unsqueeze_227: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
        unsqueeze_228: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
        mul_525: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 0.0006510416666666666)
        mul_526: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, squeeze_71)
        mul_527: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, mul_526);  mul_525 = mul_526 = None
        unsqueeze_229: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
        unsqueeze_230: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_134: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.04562504637317021);  primals_136 = None
        view_106: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_134, [-1]);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_528: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, view_106);  view_106 = None
        unsqueeze_231: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
        unsqueeze_232: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
        mul_529: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_230);  sub_166 = unsqueeze_230 = None
        sub_168: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_237, mul_529);  view_237 = mul_529 = None
        sub_169: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_228);  sub_168 = unsqueeze_228 = None
        mul_530: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_232);  sub_169 = unsqueeze_232 = None
        mul_531: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, squeeze_71);  sum_82 = squeeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_238: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_531, [384, 1, 1, 1]);  mul_531 = None
        mul_532: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_238, 0.04562504637317021);  view_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_239: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_530, [384, 1536, 1, 1]);  mul_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_533: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_207, 0.9284766908852592);  getitem_207 = None
        convert_element_type_423: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_533, torch.float32);  mul_533 = None
        convert_element_type_161: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.float32);  add_72 = None
        sigmoid_32: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_161)
        mul_534: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_423, sigmoid_32);  convert_element_type_423 = None
        sub_170: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_32);  sigmoid_32 = None
        mul_535: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_161, sub_170);  convert_element_type_161 = sub_170 = None
        add_150: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_535, 1);  mul_535 = None
        mul_536: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_534, add_150);  mul_534 = add_150 = None
        convert_element_type_425: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_536, torch.bfloat16);  mul_536 = None
        add_151: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_145, convert_element_type_425);  add_145 = convert_element_type_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_537: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_151, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_538: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_537, 2.0);  mul_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_539: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_538, convolution_46);  convolution_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_6: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_540: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_538, sigmoid_6);  mul_538 = None
        sum_83: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_539, [2, 3], True, dtype = torch.float32);  mul_539 = None
        convert_element_type_426: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_83, torch.bfloat16);  sum_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_427: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_426, torch.float32);  convert_element_type_426 = None
        convert_element_type_428: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_6, torch.float32);  sigmoid_6 = None
        sub_171: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_428)
        mul_541: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, sub_171);  convert_element_type_428 = sub_171 = None
        mul_542: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_427, mul_541);  convert_element_type_427 = mul_541 = None
        convert_element_type_429: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_542, torch.bfloat16);  mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_84: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_429, [0, 2, 3])
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_429, relu_6, convert_element_type_160, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_429 = convert_element_type_160 = None
        getitem_210: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_32[0]
        getitem_211: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_430: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_211, torch.float32);  getitem_211 = None
        convert_element_type_431: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_84, torch.float32);  sum_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_5: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_5: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_5, full_default, getitem_210);  le_5 = getitem_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_85: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_5, mean_6, convert_element_type_158, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_6 = convert_element_type_158 = None
        getitem_213: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_33[0]
        getitem_214: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_432: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_214, torch.float32);  getitem_214 = None
        convert_element_type_433: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_85, torch.float32);  sum_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_7: "bf16[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_213, [128, 1536, 14, 14]);  getitem_213 = None
        div_58: "bf16[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
        add_152: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_540, div_58);  mul_540 = div_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_86: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_152, [0, 2, 3])
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(add_152, convert_element_type_154, convert_element_type_156, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_152 = convert_element_type_154 = convert_element_type_156 = None
        getitem_216: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_34[0]
        getitem_217: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_434: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_217, torch.float32);  getitem_217 = None
        convert_element_type_435: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_86, torch.float32);  sum_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_240: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_434, [1, 1536, 384]);  convert_element_type_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_87: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_240, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_102: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_128, [1, 1536, -1]);  primals_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_172: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_102, unsqueeze_234);  view_102 = unsqueeze_234 = None
        mul_543: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_240, sub_172)
        sum_88: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 2]);  mul_543 = None
        mul_544: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 0.0026041666666666665);  sum_87 = None
        unsqueeze_235: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
        unsqueeze_236: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        mul_545: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 0.0026041666666666665)
        mul_546: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, squeeze_69)
        mul_547: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
        unsqueeze_237: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
        unsqueeze_238: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_127: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_129, 0.09125009274634042);  primals_129 = None
        view_103: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_127, [-1]);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_548: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, view_103);  view_103 = None
        unsqueeze_239: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_240: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
        mul_549: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_238);  sub_172 = unsqueeze_238 = None
        sub_174: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_240, mul_549);  view_240 = mul_549 = None
        sub_175: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_236);  sub_174 = unsqueeze_236 = None
        mul_550: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_240);  sub_175 = unsqueeze_240 = None
        mul_551: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, squeeze_69);  sum_88 = squeeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_241: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_551, [1536, 1, 1, 1]);  mul_551 = None
        mul_552: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_241, 0.09125009274634042);  view_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_242: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_550, [1536, 384, 1, 1]);  mul_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_436: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_216, torch.float32);  getitem_216 = None
        convert_element_type_153: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sigmoid_33: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_153)
        mul_553: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_436, sigmoid_33);  convert_element_type_436 = None
        sub_176: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_33);  sigmoid_33 = None
        mul_554: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_153, sub_176);  convert_element_type_153 = sub_176 = None
        add_153: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_554, 1);  mul_554 = None
        mul_555: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_553, add_153);  mul_553 = add_153 = None
        convert_element_type_438: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_555, torch.bfloat16);  mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_89: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_438, [0, 2, 3])
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_438, convert_element_type_150, convert_element_type_152, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_438 = convert_element_type_150 = convert_element_type_152 = None
        getitem_219: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_35[0]
        getitem_220: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_439: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_220, torch.float32);  getitem_220 = None
        convert_element_type_440: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_89, torch.float32);  sum_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_67: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_439, memory_format = torch.contiguous_format);  convert_element_type_439 = None
        view_243: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [1, 384, 576]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_90: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_243, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_34: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_125, memory_format = torch.contiguous_format);  primals_125 = None
        view_99: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [1, 384, 576]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_177: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_99, unsqueeze_242);  view_99 = unsqueeze_242 = None
        mul_556: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_243, sub_177)
        sum_91: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 2]);  mul_556 = None
        mul_557: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 0.001736111111111111);  sum_90 = None
        unsqueeze_243: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
        unsqueeze_244: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
        mul_558: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 0.001736111111111111)
        mul_559: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_560: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_558, mul_559);  mul_558 = mul_559 = None
        unsqueeze_245: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
        unsqueeze_246: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_124: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_126, 0.07450538873672485);  primals_126 = None
        view_100: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_124, [-1]);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_561: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, view_100);  view_100 = None
        unsqueeze_247: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
        unsqueeze_248: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        mul_562: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_246);  sub_177 = unsqueeze_246 = None
        sub_179: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_243, mul_562);  view_243 = mul_562 = None
        sub_180: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_244);  sub_179 = unsqueeze_244 = None
        mul_563: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_248);  sub_180 = unsqueeze_248 = None
        mul_564: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_67);  sum_91 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_244: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_564, [384, 1, 1, 1]);  mul_564 = None
        mul_565: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_244, 0.07450538873672485);  view_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_245: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_563, [384, 64, 3, 3]);  mul_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_441: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_219, torch.float32);  getitem_219 = None
        convert_element_type_149: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        sigmoid_34: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_149)
        mul_566: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_441, sigmoid_34);  convert_element_type_441 = None
        sub_181: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_34);  sigmoid_34 = None
        mul_567: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_149, sub_181);  convert_element_type_149 = sub_181 = None
        add_154: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_567, 1);  mul_567 = None
        mul_568: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, add_154);  mul_566 = add_154 = None
        convert_element_type_443: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_568, torch.bfloat16);  mul_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_92: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_443, [0, 2, 3])
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_443, convert_element_type_146, convert_element_type_148, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_443 = convert_element_type_146 = convert_element_type_148 = None
        getitem_222: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_36[0]
        getitem_223: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_444: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_223, torch.float32);  getitem_223 = None
        convert_element_type_445: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_92, torch.float32);  sum_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_68: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_444, memory_format = torch.contiguous_format);  convert_element_type_444 = None
        view_246: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [1, 384, 576]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_93: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_246, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_32: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_122, memory_format = torch.contiguous_format);  primals_122 = None
        view_96: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [1, 384, 576]);  clone_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_182: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_96, unsqueeze_250);  view_96 = unsqueeze_250 = None
        mul_569: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_246, sub_182)
        sum_94: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_569, [0, 2]);  mul_569 = None
        mul_570: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 0.001736111111111111);  sum_93 = None
        unsqueeze_251: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
        unsqueeze_252: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        mul_571: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 0.001736111111111111)
        mul_572: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, squeeze_65)
        mul_573: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
        unsqueeze_253: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
        unsqueeze_254: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_121: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_123, 0.07450538873672485);  primals_123 = None
        view_97: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_121, [-1]);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_574: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, view_97);  view_97 = None
        unsqueeze_255: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
        unsqueeze_256: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
        mul_575: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_254);  sub_182 = unsqueeze_254 = None
        sub_184: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_246, mul_575);  view_246 = mul_575 = None
        sub_185: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_252);  sub_184 = unsqueeze_252 = None
        mul_576: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_256);  sub_185 = unsqueeze_256 = None
        mul_577: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, squeeze_65);  sum_94 = squeeze_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_247: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_577, [384, 1, 1, 1]);  mul_577 = None
        mul_578: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_247, 0.07450538873672485);  view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_248: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_576, [384, 64, 3, 3]);  mul_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_446: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_222, torch.float32);  getitem_222 = None
        convert_element_type_145: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        sigmoid_35: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_145)
        mul_579: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_446, sigmoid_35);  convert_element_type_446 = None
        sub_186: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_35);  sigmoid_35 = None
        mul_580: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, sub_186);  convert_element_type_145 = sub_186 = None
        add_155: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_580, 1);  mul_580 = None
        mul_581: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_579, add_155);  mul_579 = add_155 = None
        convert_element_type_448: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_581, torch.bfloat16);  mul_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_95: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_448, [0, 2, 3])
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_448, mul_117, convert_element_type_144, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_448 = mul_117 = convert_element_type_144 = None
        getitem_225: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_37[0]
        getitem_226: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_449: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_226, torch.float32);  getitem_226 = None
        convert_element_type_450: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_95, torch.float32);  sum_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_249: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_449, [1, 384, 1536]);  convert_element_type_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_96: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_249, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_93: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_119, [1, 384, -1]);  primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_187: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_93, unsqueeze_258);  view_93 = unsqueeze_258 = None
        mul_582: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_249, sub_187)
        sum_97: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 2]);  mul_582 = None
        mul_583: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 0.0006510416666666666);  sum_96 = None
        unsqueeze_259: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_260: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        mul_584: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 0.0006510416666666666)
        mul_585: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, squeeze_63)
        mul_586: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_584, mul_585);  mul_584 = mul_585 = None
        unsqueeze_261: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
        unsqueeze_262: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_118: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_120, 0.04562504637317021);  primals_120 = None
        view_94: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_118, [-1]);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_587: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, view_94);  view_94 = None
        unsqueeze_263: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
        unsqueeze_264: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        mul_588: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_262);  sub_187 = unsqueeze_262 = None
        sub_189: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_249, mul_588);  view_249 = mul_588 = None
        sub_190: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_260);  sub_189 = unsqueeze_260 = None
        mul_589: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_264);  sub_190 = unsqueeze_264 = None
        mul_590: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_63);  sum_97 = squeeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_250: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_590, [384, 1, 1, 1]);  mul_590 = None
        mul_591: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_250, 0.04562504637317021);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_251: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_589, [384, 1536, 1, 1]);  mul_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_592: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_225, 0.9449111825230679);  getitem_225 = None
        convert_element_type_451: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_592, torch.float32);  mul_592 = None
        convert_element_type_141: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.float32);  add_63 = None
        sigmoid_36: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_141)
        mul_593: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_451, sigmoid_36);  convert_element_type_451 = None
        sub_191: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_36);  sigmoid_36 = None
        mul_594: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, sub_191);  convert_element_type_141 = sub_191 = None
        add_156: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_594, 1);  mul_594 = None
        mul_595: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_593, add_156);  mul_593 = add_156 = None
        convert_element_type_453: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_595, torch.bfloat16);  mul_595 = None
        add_157: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_151, convert_element_type_453);  add_151 = convert_element_type_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_596: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_157, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_597: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_596, 2.0);  mul_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_598: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_597, convolution_40);  convolution_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_5: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_599: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_597, sigmoid_5);  mul_597 = None
        sum_98: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_598, [2, 3], True, dtype = torch.float32);  mul_598 = None
        convert_element_type_454: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_98, torch.bfloat16);  sum_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_455: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_454, torch.float32);  convert_element_type_454 = None
        convert_element_type_456: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_5, torch.float32);  sigmoid_5 = None
        sub_192: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_456)
        mul_600: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_456, sub_192);  convert_element_type_456 = sub_192 = None
        mul_601: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_455, mul_600);  convert_element_type_455 = mul_600 = None
        convert_element_type_457: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_601, torch.bfloat16);  mul_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_99: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_457, [0, 2, 3])
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_457, relu_5, convert_element_type_140, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_457 = convert_element_type_140 = None
        getitem_228: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_38[0]
        getitem_229: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_458: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_229, torch.float32);  getitem_229 = None
        convert_element_type_459: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_99, torch.float32);  sum_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_6: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_6: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_6, full_default, getitem_228);  le_6 = getitem_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_100: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_6, mean_5, convert_element_type_138, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_6 = mean_5 = convert_element_type_138 = None
        getitem_231: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_39[0]
        getitem_232: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_460: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_232, torch.float32);  getitem_232 = None
        convert_element_type_461: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_100, torch.float32);  sum_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_8: "bf16[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_231, [128, 1536, 14, 14]);  getitem_231 = None
        div_59: "bf16[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
        add_158: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_599, div_59);  mul_599 = div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_101: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_158, [0, 2, 3])
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(add_158, convert_element_type_134, convert_element_type_136, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_158 = convert_element_type_134 = convert_element_type_136 = None
        getitem_234: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_40[0]
        getitem_235: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_462: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_235, torch.float32);  getitem_235 = None
        convert_element_type_463: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_101, torch.float32);  sum_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_252: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_462, [1, 1536, 384]);  convert_element_type_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_102: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_252, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_90: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_112, [1, 1536, -1]);  primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_193: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_90, unsqueeze_266);  view_90 = unsqueeze_266 = None
        mul_602: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_252, sub_193)
        sum_103: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 2]);  mul_602 = None
        mul_603: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 0.0026041666666666665);  sum_102 = None
        unsqueeze_267: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
        unsqueeze_268: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
        mul_604: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 0.0026041666666666665)
        mul_605: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_606: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
        unsqueeze_269: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
        unsqueeze_270: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_111: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_113, 0.09125009274634042);  primals_113 = None
        view_91: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_111, [-1]);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_607: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, view_91);  view_91 = None
        unsqueeze_271: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
        unsqueeze_272: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        mul_608: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_270);  sub_193 = unsqueeze_270 = None
        sub_195: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_252, mul_608);  view_252 = mul_608 = None
        sub_196: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_268);  sub_195 = unsqueeze_268 = None
        mul_609: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_272);  sub_196 = unsqueeze_272 = None
        mul_610: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_61);  sum_103 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_253: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_610, [1536, 1, 1, 1]);  mul_610 = None
        mul_611: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_253, 0.09125009274634042);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_254: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_609, [1536, 384, 1, 1]);  mul_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_464: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_234, torch.float32);  getitem_234 = None
        convert_element_type_133: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sigmoid_37: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_133)
        mul_612: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_464, sigmoid_37);  convert_element_type_464 = None
        sub_197: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_37);  sigmoid_37 = None
        mul_613: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_133, sub_197);  convert_element_type_133 = sub_197 = None
        add_159: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_613, 1);  mul_613 = None
        mul_614: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_612, add_159);  mul_612 = add_159 = None
        convert_element_type_466: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_614, torch.bfloat16);  mul_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_104: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_466, [0, 2, 3])
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_466, convert_element_type_130, convert_element_type_132, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_466 = convert_element_type_130 = convert_element_type_132 = None
        getitem_237: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_41[0]
        getitem_238: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_467: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_238, torch.float32);  getitem_238 = None
        convert_element_type_468: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_104, torch.float32);  sum_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_69: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_467, memory_format = torch.contiguous_format);  convert_element_type_467 = None
        view_255: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [1, 384, 576]);  clone_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_105: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_255, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_30: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_109, memory_format = torch.contiguous_format);  primals_109 = None
        view_87: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [1, 384, 576]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_198: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_87, unsqueeze_274);  view_87 = unsqueeze_274 = None
        mul_615: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_255, sub_198)
        sum_106: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_615, [0, 2]);  mul_615 = None
        mul_616: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 0.001736111111111111);  sum_105 = None
        unsqueeze_275: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
        unsqueeze_276: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        mul_617: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 0.001736111111111111)
        mul_618: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, squeeze_59)
        mul_619: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
        unsqueeze_277: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
        unsqueeze_278: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_108: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_110, 0.07450538873672485);  primals_110 = None
        view_88: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_108, [-1]);  mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_620: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, view_88);  view_88 = None
        unsqueeze_279: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
        unsqueeze_280: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
        mul_621: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_278);  sub_198 = unsqueeze_278 = None
        sub_200: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_255, mul_621);  view_255 = mul_621 = None
        sub_201: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_276);  sub_200 = unsqueeze_276 = None
        mul_622: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_280);  sub_201 = unsqueeze_280 = None
        mul_623: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, squeeze_59);  sum_106 = squeeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_256: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_623, [384, 1, 1, 1]);  mul_623 = None
        mul_624: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_256, 0.07450538873672485);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_257: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_622, [384, 64, 3, 3]);  mul_622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_469: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None
        convert_element_type_129: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32);  convolution_38 = None
        sigmoid_38: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_129)
        mul_625: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_469, sigmoid_38);  convert_element_type_469 = None
        sub_202: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_38);  sigmoid_38 = None
        mul_626: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_129, sub_202);  convert_element_type_129 = sub_202 = None
        add_160: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_626, 1);  mul_626 = None
        mul_627: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_625, add_160);  mul_625 = add_160 = None
        convert_element_type_471: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_627, torch.bfloat16);  mul_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_107: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_471, [0, 2, 3])
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_471, convert_element_type_126, convert_element_type_128, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_471 = convert_element_type_126 = convert_element_type_128 = None
        getitem_240: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_42[0]
        getitem_241: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_472: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_241, torch.float32);  getitem_241 = None
        convert_element_type_473: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_107, torch.float32);  sum_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_70: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_472, memory_format = torch.contiguous_format);  convert_element_type_472 = None
        view_258: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [1, 384, 576]);  clone_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_108: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_258, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_28: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_106, memory_format = torch.contiguous_format);  primals_106 = None
        view_84: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [1, 384, 576]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_203: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_84, unsqueeze_282);  view_84 = unsqueeze_282 = None
        mul_628: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_258, sub_203)
        sum_109: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_628, [0, 2]);  mul_628 = None
        mul_629: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 0.001736111111111111);  sum_108 = None
        unsqueeze_283: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
        unsqueeze_284: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        mul_630: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 0.001736111111111111)
        mul_631: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, squeeze_57)
        mul_632: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
        unsqueeze_285: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
        unsqueeze_286: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_105: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, 0.07450538873672485);  primals_107 = None
        view_85: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_105, [-1]);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_633: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, view_85);  view_85 = None
        unsqueeze_287: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
        unsqueeze_288: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        mul_634: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_286);  sub_203 = unsqueeze_286 = None
        sub_205: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_258, mul_634);  view_258 = mul_634 = None
        sub_206: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_284);  sub_205 = unsqueeze_284 = None
        mul_635: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_288);  sub_206 = unsqueeze_288 = None
        mul_636: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_57);  sum_109 = squeeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_259: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_636, [384, 1, 1, 1]);  mul_636 = None
        mul_637: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_259, 0.07450538873672485);  view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_260: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_635, [384, 64, 3, 3]);  mul_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_474: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_240, torch.float32);  getitem_240 = None
        convert_element_type_125: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sigmoid_39: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_125)
        mul_638: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_474, sigmoid_39);  convert_element_type_474 = None
        sub_207: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_39);  sigmoid_39 = None
        mul_639: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_125, sub_207);  convert_element_type_125 = sub_207 = None
        add_161: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_639, 1);  mul_639 = None
        mul_640: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_638, add_161);  mul_638 = add_161 = None
        convert_element_type_476: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_640, torch.bfloat16);  mul_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_110: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_476, [0, 2, 3])
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_476, mul_101, convert_element_type_124, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_476 = mul_101 = convert_element_type_124 = None
        getitem_243: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_43[0]
        getitem_244: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_477: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_244, torch.float32);  getitem_244 = None
        convert_element_type_478: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_110, torch.float32);  sum_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_261: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_477, [1, 384, 1536]);  convert_element_type_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_111: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_261, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_81: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_103, [1, 384, -1]);  primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_208: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_81, unsqueeze_290);  view_81 = unsqueeze_290 = None
        mul_641: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_261, sub_208)
        sum_112: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_641, [0, 2]);  mul_641 = None
        mul_642: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 0.0006510416666666666);  sum_111 = None
        unsqueeze_291: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
        unsqueeze_292: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
        mul_643: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 0.0006510416666666666)
        mul_644: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_645: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_643, mul_644);  mul_643 = mul_644 = None
        unsqueeze_293: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
        unsqueeze_294: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_102: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_104, 0.04562504637317021);  primals_104 = None
        view_82: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_102, [-1]);  mul_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_646: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, view_82);  view_82 = None
        unsqueeze_295: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
        unsqueeze_296: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        mul_647: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_294);  sub_208 = unsqueeze_294 = None
        sub_210: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_261, mul_647);  view_261 = mul_647 = None
        sub_211: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_292);  sub_210 = unsqueeze_292 = None
        mul_648: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_296);  sub_211 = unsqueeze_296 = None
        mul_649: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, squeeze_55);  sum_112 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_262: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_649, [384, 1, 1, 1]);  mul_649 = None
        mul_650: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_262, 0.04562504637317021);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_263: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_648, [384, 1536, 1, 1]);  mul_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_651: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_243, 0.9622504486493761);  getitem_243 = None
        convert_element_type_479: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_651, torch.float32);  mul_651 = None
        convert_element_type_121: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.float32);  add_54 = None
        sigmoid_40: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_121)
        mul_652: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_479, sigmoid_40);  convert_element_type_479 = None
        sub_212: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_40);  sigmoid_40 = None
        mul_653: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_121, sub_212);  convert_element_type_121 = sub_212 = None
        add_162: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_653, 1);  mul_653 = None
        mul_654: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_652, add_162);  mul_652 = add_162 = None
        convert_element_type_481: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_654, torch.bfloat16);  mul_654 = None
        add_163: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_157, convert_element_type_481);  add_157 = convert_element_type_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_655: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_163, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_656: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_655, 2.0);  mul_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_657: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_656, convolution_34);  convolution_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_4: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_36);  convolution_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_658: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_656, sigmoid_4);  mul_656 = None
        sum_113: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_657, [2, 3], True, dtype = torch.float32);  mul_657 = None
        convert_element_type_482: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_113, torch.bfloat16);  sum_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_483: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_482, torch.float32);  convert_element_type_482 = None
        convert_element_type_484: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_4, torch.float32);  sigmoid_4 = None
        sub_213: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_484)
        mul_659: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_484, sub_213);  convert_element_type_484 = sub_213 = None
        mul_660: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_483, mul_659);  convert_element_type_483 = mul_659 = None
        convert_element_type_485: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_660, torch.bfloat16);  mul_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_114: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_485, [0, 2, 3])
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_485, relu_4, convert_element_type_120, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_485 = convert_element_type_120 = None
        getitem_246: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_44[0]
        getitem_247: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_486: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_247, torch.float32);  getitem_247 = None
        convert_element_type_487: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_114, torch.float32);  sum_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_7: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_7: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_7, full_default, getitem_246);  le_7 = getitem_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_115: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_7, mean_4, convert_element_type_118, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_7 = mean_4 = convert_element_type_118 = None
        getitem_249: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_45[0]
        getitem_250: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_488: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_250, torch.float32);  getitem_250 = None
        convert_element_type_489: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_115, torch.float32);  sum_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_9: "bf16[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_249, [128, 1536, 14, 14]);  getitem_249 = None
        div_60: "bf16[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
        add_164: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_658, div_60);  mul_658 = div_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_116: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_164, [0, 2, 3])
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(add_164, convert_element_type_114, convert_element_type_116, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_164 = convert_element_type_114 = convert_element_type_116 = None
        getitem_252: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_46[0]
        getitem_253: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_490: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_253, torch.float32);  getitem_253 = None
        convert_element_type_491: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_116, torch.float32);  sum_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_264: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_490, [1, 1536, 384]);  convert_element_type_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_117: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_264, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_78: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_96, [1, 1536, -1]);  primals_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_214: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_78, unsqueeze_298);  view_78 = unsqueeze_298 = None
        mul_661: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_264, sub_214)
        sum_118: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2]);  mul_661 = None
        mul_662: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 0.0026041666666666665);  sum_117 = None
        unsqueeze_299: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
        unsqueeze_300: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        mul_663: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 0.0026041666666666665)
        mul_664: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, squeeze_53)
        mul_665: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
        unsqueeze_301: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
        unsqueeze_302: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_95: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_97, 0.09125009274634042);  primals_97 = None
        view_79: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_95, [-1]);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_666: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, view_79);  view_79 = None
        unsqueeze_303: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_304: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        mul_667: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_302);  sub_214 = unsqueeze_302 = None
        sub_216: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_264, mul_667);  view_264 = mul_667 = None
        sub_217: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_300);  sub_216 = unsqueeze_300 = None
        mul_668: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_304);  sub_217 = unsqueeze_304 = None
        mul_669: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, squeeze_53);  sum_118 = squeeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_265: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_669, [1536, 1, 1, 1]);  mul_669 = None
        mul_670: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_265, 0.09125009274634042);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_266: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_668, [1536, 384, 1, 1]);  mul_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_492: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_252, torch.float32);  getitem_252 = None
        convert_element_type_113: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sigmoid_41: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_113)
        mul_671: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_492, sigmoid_41);  convert_element_type_492 = None
        sub_218: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_41);  sigmoid_41 = None
        mul_672: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_113, sub_218);  convert_element_type_113 = sub_218 = None
        add_165: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_672, 1);  mul_672 = None
        mul_673: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_671, add_165);  mul_671 = add_165 = None
        convert_element_type_494: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_673, torch.bfloat16);  mul_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_119: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_494, [0, 2, 3])
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(convert_element_type_494, convert_element_type_110, convert_element_type_112, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_494 = convert_element_type_110 = convert_element_type_112 = None
        getitem_255: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_47[0]
        getitem_256: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_495: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_256, torch.float32);  getitem_256 = None
        convert_element_type_496: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_119, torch.float32);  sum_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_71: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_495, memory_format = torch.contiguous_format);  convert_element_type_495 = None
        view_267: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [1, 384, 576]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_120: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_267, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_26: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_93, memory_format = torch.contiguous_format);  primals_93 = None
        view_75: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [1, 384, 576]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_219: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_75, unsqueeze_306);  view_75 = unsqueeze_306 = None
        mul_674: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_267, sub_219)
        sum_121: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_674, [0, 2]);  mul_674 = None
        mul_675: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 0.001736111111111111);  sum_120 = None
        unsqueeze_307: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_308: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        mul_676: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 0.001736111111111111)
        mul_677: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, squeeze_51)
        mul_678: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
        unsqueeze_309: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
        unsqueeze_310: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_92: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.07450538873672485);  primals_94 = None
        view_76: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_92, [-1]);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_679: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, view_76);  view_76 = None
        unsqueeze_311: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_679, 0);  mul_679 = None
        unsqueeze_312: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        mul_680: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_310);  sub_219 = unsqueeze_310 = None
        sub_221: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_267, mul_680);  view_267 = mul_680 = None
        sub_222: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_308);  sub_221 = unsqueeze_308 = None
        mul_681: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_312);  sub_222 = unsqueeze_312 = None
        mul_682: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_51);  sum_121 = squeeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_268: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_682, [384, 1, 1, 1]);  mul_682 = None
        mul_683: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_268, 0.07450538873672485);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_269: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_681, [384, 64, 3, 3]);  mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_497: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_255, torch.float32);  getitem_255 = None
        convert_element_type_109: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sigmoid_42: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_109)
        mul_684: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_497, sigmoid_42);  convert_element_type_497 = None
        sub_223: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_42);  sigmoid_42 = None
        mul_685: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_109, sub_223);  convert_element_type_109 = sub_223 = None
        add_166: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_685, 1);  mul_685 = None
        mul_686: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_684, add_166);  mul_684 = add_166 = None
        convert_element_type_499: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_686, torch.bfloat16);  mul_686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_122: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_499, [0, 2, 3])
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_499, convert_element_type_106, convert_element_type_108, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_499 = convert_element_type_106 = convert_element_type_108 = None
        getitem_258: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_48[0]
        getitem_259: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_500: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_259, torch.float32);  getitem_259 = None
        convert_element_type_501: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_122, torch.float32);  sum_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_72: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_500, memory_format = torch.contiguous_format);  convert_element_type_500 = None
        view_270: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [1, 384, 576]);  clone_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_123: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_270, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_24: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_90, memory_format = torch.contiguous_format);  primals_90 = None
        view_72: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [1, 384, 576]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_224: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_72, unsqueeze_314);  view_72 = unsqueeze_314 = None
        mul_687: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_270, sub_224)
        sum_124: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_687, [0, 2]);  mul_687 = None
        mul_688: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, 0.001736111111111111);  sum_123 = None
        unsqueeze_315: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
        unsqueeze_316: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
        mul_689: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_124, 0.001736111111111111)
        mul_690: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_691: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_689, mul_690);  mul_689 = mul_690 = None
        unsqueeze_317: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
        unsqueeze_318: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_89: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_91, 0.07450538873672485);  primals_91 = None
        view_73: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_89, [-1]);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_692: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, view_73);  view_73 = None
        unsqueeze_319: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
        unsqueeze_320: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        mul_693: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_318);  sub_224 = unsqueeze_318 = None
        sub_226: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_270, mul_693);  view_270 = mul_693 = None
        sub_227: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_316);  sub_226 = unsqueeze_316 = None
        mul_694: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_320);  sub_227 = unsqueeze_320 = None
        mul_695: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_124, squeeze_49);  sum_124 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_271: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_695, [384, 1, 1, 1]);  mul_695 = None
        mul_696: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_271, 0.07450538873672485);  view_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_272: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_694, [384, 64, 3, 3]);  mul_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_502: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_258, torch.float32);  getitem_258 = None
        convert_element_type_105: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sigmoid_43: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_105)
        mul_697: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_502, sigmoid_43);  convert_element_type_502 = None
        sub_228: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_43);  sigmoid_43 = None
        mul_698: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_105, sub_228);  convert_element_type_105 = sub_228 = None
        add_167: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_698, 1);  mul_698 = None
        mul_699: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_697, add_167);  mul_697 = add_167 = None
        convert_element_type_504: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_699, torch.bfloat16);  mul_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_125: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_504, [0, 2, 3])
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(convert_element_type_504, mul_85, convert_element_type_104, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_504 = mul_85 = convert_element_type_104 = None
        getitem_261: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_49[0]
        getitem_262: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_505: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_262, torch.float32);  getitem_262 = None
        convert_element_type_506: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_125, torch.float32);  sum_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_273: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_505, [1, 384, 1536]);  convert_element_type_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_126: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_273, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_69: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(primals_87, [1, 384, -1]);  primals_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_229: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_69, unsqueeze_322);  view_69 = unsqueeze_322 = None
        mul_700: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_273, sub_229)
        sum_127: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_700, [0, 2]);  mul_700 = None
        mul_701: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_126, 0.0006510416666666666);  sum_126 = None
        unsqueeze_323: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
        unsqueeze_324: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        mul_702: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, 0.0006510416666666666)
        mul_703: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, squeeze_47)
        mul_704: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_702, mul_703);  mul_702 = mul_703 = None
        unsqueeze_325: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
        unsqueeze_326: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_86: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.04562504637317021);  primals_88 = None
        view_70: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_86, [-1]);  mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_705: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, view_70);  view_70 = None
        unsqueeze_327: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
        unsqueeze_328: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
        mul_706: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_326);  sub_229 = unsqueeze_326 = None
        sub_231: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_273, mul_706);  view_273 = mul_706 = None
        sub_232: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_324);  sub_231 = unsqueeze_324 = None
        mul_707: "f32[1, 384, 1536][589824, 1536, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_328);  sub_232 = unsqueeze_328 = None
        mul_708: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_127, squeeze_47);  sum_127 = squeeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_274: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_708, [384, 1, 1, 1]);  mul_708 = None
        mul_709: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_274, 0.04562504637317021);  view_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_275: "f32[384, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_707, [384, 1536, 1, 1]);  mul_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_710: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(getitem_261, 0.9805806756909201);  getitem_261 = None
        convert_element_type_507: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_710, torch.float32);  mul_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_3: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convolution_30);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_82: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convolution_28, sigmoid_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_83: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, 2.0);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_84: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_83, 0.2);  mul_83 = None
        add_45: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_84, convolution_24);  mul_84 = convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_101: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.float32);  add_45 = None
        sigmoid_44: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_101)
        mul_711: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_507, sigmoid_44);  convert_element_type_507 = None
        sub_233: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_44);  sigmoid_44 = None
        mul_712: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_101, sub_233);  convert_element_type_101 = sub_233 = None
        add_168: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_712, 1);  mul_712 = None
        mul_713: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_711, add_168);  mul_711 = add_168 = None
        convert_element_type_509: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_713, torch.bfloat16);  mul_713 = None
        add_169: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(add_163, convert_element_type_509);  add_163 = convert_element_type_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_714: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_169, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_715: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_714, 2.0);  mul_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_716: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_715, convolution_28);  convolution_28 = None
        mul_717: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_715, sigmoid_3);  mul_715 = None
        sum_128: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_716, [2, 3], True, dtype = torch.float32);  mul_716 = None
        convert_element_type_510: "bf16[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_128, torch.bfloat16);  sum_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_511: "f32[128, 1536, 1, 1][1536, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_510, torch.float32);  convert_element_type_510 = None
        convert_element_type_512: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_3, torch.float32);  sigmoid_3 = None
        sub_234: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_512)
        mul_718: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_512, sub_234);  convert_element_type_512 = sub_234 = None
        mul_719: "f32[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_511, mul_718);  convert_element_type_511 = mul_718 = None
        convert_element_type_513: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_719, torch.bfloat16);  mul_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_129: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_513, [0, 2, 3])
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_513, relu_3, convert_element_type_100, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_513 = convert_element_type_100 = None
        getitem_264: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_50[0]
        getitem_265: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_514: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_265, torch.float32);  getitem_265 = None
        convert_element_type_515: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_129, torch.float32);  sum_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_8: "b8[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_8: "bf16[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.aten.where.self(le_8, full_default, getitem_264);  le_8 = getitem_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_130: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(where_8, mean_3, convert_element_type_98, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_8 = mean_3 = convert_element_type_98 = None
        getitem_267: "bf16[128, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_51[0]
        getitem_268: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_516: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_268, torch.float32);  getitem_268 = None
        convert_element_type_517: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_130, torch.float32);  sum_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_10: "bf16[128, 1536, 14, 14][1536, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_267, [128, 1536, 14, 14]);  getitem_267 = None
        div_61: "bf16[128, 1536, 14, 14][301056, 196, 14, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
        add_170: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_717, div_61);  mul_717 = div_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_131: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_170, [0, 2, 3])
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(add_170, convert_element_type_94, convert_element_type_96, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_170 = convert_element_type_94 = convert_element_type_96 = None
        getitem_270: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_52[0]
        getitem_271: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        convert_element_type_518: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_271, torch.float32);  getitem_271 = None
        convert_element_type_519: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_131, torch.float32);  sum_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_276: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_518, [1, 1536, 384]);  convert_element_type_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_132: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_276, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_66: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.reshape.default(primals_80, [1, 1536, -1]);  primals_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_235: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_66, unsqueeze_330);  view_66 = unsqueeze_330 = None
        mul_720: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_276, sub_235)
        sum_133: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_720, [0, 2]);  mul_720 = None
        mul_721: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_132, 0.0026041666666666665);  sum_132 = None
        unsqueeze_331: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
        unsqueeze_332: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        mul_722: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, 0.0026041666666666665)
        mul_723: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, squeeze_45)
        mul_724: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
        unsqueeze_333: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
        unsqueeze_334: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_79: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_81, 0.09125009274634042);  primals_81 = None
        view_67: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_79, [-1]);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_725: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, view_67);  view_67 = None
        unsqueeze_335: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
        unsqueeze_336: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        mul_726: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_334);  sub_235 = unsqueeze_334 = None
        sub_237: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_276, mul_726);  view_276 = mul_726 = None
        sub_238: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_332);  sub_237 = unsqueeze_332 = None
        mul_727: "f32[1, 1536, 384][589824, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_336);  sub_238 = unsqueeze_336 = None
        mul_728: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_133, squeeze_45);  sum_133 = squeeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_277: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_728, [1536, 1, 1, 1]);  mul_728 = None
        mul_729: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_277, 0.09125009274634042);  view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_278: "f32[1536, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_727, [1536, 384, 1, 1]);  mul_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_520: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_270, torch.float32);  getitem_270 = None
        convert_element_type_93: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sigmoid_45: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_93)
        mul_730: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_520, sigmoid_45);  convert_element_type_520 = None
        sub_239: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_45);  sigmoid_45 = None
        mul_731: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, sub_239);  convert_element_type_93 = sub_239 = None
        add_171: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_731, 1);  mul_731 = None
        mul_732: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_730, add_171);  mul_730 = add_171 = None
        convert_element_type_522: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_732, torch.bfloat16);  mul_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_134: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_522, [0, 2, 3])
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(convert_element_type_522, convert_element_type_90, convert_element_type_92, [384], [1, 1], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_522 = convert_element_type_90 = convert_element_type_92 = None
        getitem_273: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_53[0]
        getitem_274: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_523: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_274, torch.float32);  getitem_274 = None
        convert_element_type_524: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_134, torch.float32);  sum_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_73: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_523, memory_format = torch.contiguous_format);  convert_element_type_523 = None
        view_279: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [1, 384, 576]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_135: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_279, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_22: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_77, memory_format = torch.contiguous_format);  primals_77 = None
        view_63: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [1, 384, 576]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_240: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_63, unsqueeze_338);  view_63 = unsqueeze_338 = None
        mul_733: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_279, sub_240)
        sum_136: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2]);  mul_733 = None
        mul_734: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_135, 0.001736111111111111);  sum_135 = None
        unsqueeze_339: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
        unsqueeze_340: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
        mul_735: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, 0.001736111111111111)
        mul_736: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_737: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
        unsqueeze_341: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
        unsqueeze_342: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_76: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_78, 0.07450538873672485);  primals_78 = None
        view_64: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_76, [-1]);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_738: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, view_64);  view_64 = None
        unsqueeze_343: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_344: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        mul_739: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_342);  sub_240 = unsqueeze_342 = None
        sub_242: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_279, mul_739);  view_279 = mul_739 = None
        sub_243: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_340);  sub_242 = unsqueeze_340 = None
        mul_740: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_344);  sub_243 = unsqueeze_344 = None
        mul_741: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_136, squeeze_43);  sum_136 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_280: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_741, [384, 1, 1, 1]);  mul_741 = None
        mul_742: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_280, 0.07450538873672485);  view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_281: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_740, [384, 64, 3, 3]);  mul_740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_525: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_273, torch.float32);  getitem_273 = None
        convert_element_type_89: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sigmoid_46: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_89)
        mul_743: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_525, sigmoid_46);  convert_element_type_525 = None
        sub_244: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_46);  sigmoid_46 = None
        mul_744: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_89, sub_244);  convert_element_type_89 = sub_244 = None
        add_172: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_744, 1);  mul_744 = None
        mul_745: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_743, add_172);  mul_743 = add_172 = None
        convert_element_type_527: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_745, torch.bfloat16);  mul_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_137: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_527, [0, 2, 3])
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(convert_element_type_527, convert_element_type_86, convert_element_type_88, [384], [2, 2], [1, 1], [1, 1], False, [0, 0], 6, [True, True, False]);  convert_element_type_527 = convert_element_type_86 = convert_element_type_88 = None
        getitem_276: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_54[0]
        getitem_277: "bf16[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None
        convert_element_type_528: "f32[384, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_277, torch.float32);  getitem_277 = None
        convert_element_type_529: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_137, torch.float32);  sum_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_74: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_528, memory_format = torch.contiguous_format);  convert_element_type_528 = None
        view_282: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [1, 384, 576]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_138: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_282, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_20: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_74, memory_format = torch.contiguous_format);  primals_74 = None
        view_60: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [1, 384, 576]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_245: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_60, unsqueeze_346);  view_60 = unsqueeze_346 = None
        mul_746: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_282, sub_245)
        sum_139: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_746, [0, 2]);  mul_746 = None
        mul_747: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_138, 0.001736111111111111);  sum_138 = None
        unsqueeze_347: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
        unsqueeze_348: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
        mul_748: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, 0.001736111111111111)
        mul_749: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, squeeze_41)
        mul_750: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_748, mul_749);  mul_748 = mul_749 = None
        unsqueeze_349: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
        unsqueeze_350: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_73: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_75, 0.07450538873672485);  primals_75 = None
        view_61: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_73, [-1]);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_751: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, view_61);  view_61 = None
        unsqueeze_351: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
        unsqueeze_352: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
        mul_752: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_350);  sub_245 = unsqueeze_350 = None
        sub_247: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_282, mul_752);  view_282 = mul_752 = None
        sub_248: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_348);  sub_247 = unsqueeze_348 = None
        mul_753: "f32[1, 384, 576][221184, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_352);  sub_248 = unsqueeze_352 = None
        mul_754: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_139, squeeze_41);  sum_139 = squeeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_283: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_754, [384, 1, 1, 1]);  mul_754 = None
        mul_755: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_283, 0.07450538873672485);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_284: "f32[384, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_753, [384, 64, 3, 3]);  mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_530: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_276, torch.float32);  getitem_276 = None
        convert_element_type_85: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sigmoid_47: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_85)
        mul_756: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_530, sigmoid_47);  convert_element_type_530 = None
        sub_249: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_47);  sigmoid_47 = None
        mul_757: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_85, sub_249);  convert_element_type_85 = sub_249 = None
        add_173: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_757, 1);  mul_757 = None
        mul_758: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_756, add_173);  mul_756 = add_173 = None
        convert_element_type_532: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_758, torch.bfloat16);  mul_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_140: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_532, [0, 2, 3])
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(convert_element_type_532, mul_66, convert_element_type_84, [384], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_532 = convert_element_type_84 = None
        getitem_279: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_55[0]
        getitem_280: "bf16[384, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_533: "f32[384, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_280, torch.float32);  getitem_280 = None
        convert_element_type_534: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_140, torch.float32);  sum_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_285: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_533, [1, 384, 512]);  convert_element_type_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_141: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_285, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_57: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.reshape.default(primals_71, [1, 384, -1]);  primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_250: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_57, unsqueeze_354);  view_57 = unsqueeze_354 = None
        mul_759: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_285, sub_250)
        sum_142: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2]);  mul_759 = None
        mul_760: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_141, 0.001953125);  sum_141 = None
        unsqueeze_355: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
        unsqueeze_356: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        mul_761: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, 0.001953125)
        mul_762: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, squeeze_39)
        mul_763: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
        unsqueeze_357: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
        unsqueeze_358: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_70: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_72, 0.07902489841601695);  primals_72 = None
        view_58: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(mul_70, [-1]);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_764: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, view_58);  view_58 = None
        unsqueeze_359: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
        unsqueeze_360: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
        mul_765: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_358);  sub_250 = unsqueeze_358 = None
        sub_252: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_285, mul_765);  view_285 = mul_765 = None
        sub_253: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_356);  sub_252 = unsqueeze_356 = None
        mul_766: "f32[1, 384, 512][196608, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_360);  sub_253 = unsqueeze_360 = None
        mul_767: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_142, squeeze_39);  sum_142 = squeeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_286: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_767, [384, 1, 1, 1]);  mul_767 = None
        mul_768: "f32[384, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_286, 0.07902489841601695);  view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_287: "f32[384, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_766, [384, 512, 1, 1]);  mul_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_143: "bf16[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_169, [0, 2, 3])
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(add_169, avg_pool2d_1, convert_element_type_82, [1536], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_169 = avg_pool2d_1 = convert_element_type_82 = None
        getitem_282: "bf16[128, 512, 14, 14][100352, 1, 7168, 512]cuda:0" = convolution_backward_56[0]
        getitem_283: "bf16[1536, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None
        convert_element_type_535: "f32[1536, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_283, torch.float32);  getitem_283 = None
        convert_element_type_536: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_143, torch.float32);  sum_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_288: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_535, [1, 1536, 512]);  convert_element_type_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_144: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_288, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_54: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.reshape.default(primals_68, [1, 1536, -1]);  primals_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_254: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_54, unsqueeze_362);  view_54 = unsqueeze_362 = None
        mul_769: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_288, sub_254)
        sum_145: "f32[1536][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2]);  mul_769 = None
        mul_770: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_144, 0.001953125);  sum_144 = None
        unsqueeze_363: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
        unsqueeze_364: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
        mul_771: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, 0.001953125)
        mul_772: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_773: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
        unsqueeze_365: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
        unsqueeze_366: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_67: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_69, 0.07902489841601695);  primals_69 = None
        view_55: "f32[1536][1]cuda:0" = torch.ops.aten.reshape.default(mul_67, [-1]);  mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_774: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, view_55);  view_55 = None
        unsqueeze_367: "f32[1, 1536][1536, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_368: "f32[1, 1536, 1][1536, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        mul_775: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_366);  sub_254 = unsqueeze_366 = None
        sub_256: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_288, mul_775);  view_288 = mul_775 = None
        sub_257: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_256, unsqueeze_364);  sub_256 = unsqueeze_364 = None
        mul_776: "f32[1, 1536, 512][786432, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_368);  sub_257 = unsqueeze_368 = None
        mul_777: "f32[1536][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_145, squeeze_37);  sum_145 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_289: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_777, [1536, 1, 1, 1]);  mul_777 = None
        mul_778: "f32[1536, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_289, 0.07902489841601695);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_290: "f32[1536, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_776, [1536, 512, 1, 1]);  mul_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_backward_1: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_282, mul_66, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_282 = mul_66 = None
        add_174: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(getitem_279, avg_pool2d_backward_1);  getitem_279 = avg_pool2d_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_779: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(add_174, 0.9622504486493761);  add_174 = None
        convert_element_type_537: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_779, torch.float32);  mul_779 = None
        convert_element_type_79: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.float32);  add_35 = None
        sigmoid_48: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_79)
        mul_780: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_537, sigmoid_48);  convert_element_type_537 = None
        sub_258: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_48);  sigmoid_48 = None
        mul_781: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, sub_258);  convert_element_type_79 = sub_258 = None
        add_175: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_781, 1);  mul_781 = None
        mul_782: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_780, add_175);  mul_780 = add_175 = None
        convert_element_type_539: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_782, torch.bfloat16);  mul_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_783: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_539, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_784: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_783, 2.0);  mul_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_785: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_784, convolution_21);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_2: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_786: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_784, sigmoid_2);  mul_784 = None
        sum_146: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_785, [2, 3], True, dtype = torch.float32);  mul_785 = None
        convert_element_type_540: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_146, torch.bfloat16);  sum_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_541: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_540, torch.float32);  convert_element_type_540 = None
        convert_element_type_542: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_2, torch.float32);  sigmoid_2 = None
        sub_259: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_542)
        mul_787: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_542, sub_259);  convert_element_type_542 = sub_259 = None
        mul_788: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_541, mul_787);  convert_element_type_541 = mul_787 = None
        convert_element_type_543: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_788, torch.bfloat16);  mul_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_147: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_543, [0, 2, 3])
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(convert_element_type_543, relu_2, convert_element_type_78, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_543 = convert_element_type_78 = None
        getitem_285: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_57[0]
        getitem_286: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_57[1];  convolution_backward_57 = None
        convert_element_type_544: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_286, torch.float32);  getitem_286 = None
        convert_element_type_545: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_147, torch.float32);  sum_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_9: "b8[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_9: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.where.self(le_9, full_default, getitem_285);  le_9 = getitem_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_148: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(where_9, mean_2, convert_element_type_76, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_2 = convert_element_type_76 = None
        getitem_288: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_58[0]
        getitem_289: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_58[1];  convolution_backward_58 = None
        convert_element_type_546: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_289, torch.float32);  getitem_289 = None
        convert_element_type_547: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_148, torch.float32);  sum_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_11: "bf16[128, 512, 28, 28][512, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_288, [128, 512, 28, 28]);  getitem_288 = None
        div_62: "bf16[128, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_11, 784);  expand_11 = None
        add_176: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_786, div_62);  mul_786 = div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_149: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_176, [0, 2, 3])
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(add_176, convert_element_type_72, convert_element_type_74, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_176 = convert_element_type_72 = convert_element_type_74 = None
        getitem_291: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_59[0]
        getitem_292: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_59[1];  convolution_backward_59 = None
        convert_element_type_548: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_292, torch.float32);  getitem_292 = None
        convert_element_type_549: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_149, torch.float32);  sum_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_291: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_548, [1, 512, 128]);  convert_element_type_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_150: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_291, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_51: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_61, [1, 512, -1]);  primals_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_260: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_51, unsqueeze_370);  view_51 = unsqueeze_370 = None
        mul_789: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_291, sub_260)
        sum_151: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 2]);  mul_789 = None
        mul_790: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_150, 0.0078125);  sum_150 = None
        unsqueeze_371: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
        unsqueeze_372: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
        mul_791: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, 0.0078125)
        mul_792: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, squeeze_35)
        mul_793: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_791, mul_792);  mul_791 = mul_792 = None
        unsqueeze_373: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
        unsqueeze_374: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_60: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_62, 0.1580497968320339);  primals_62 = None
        view_52: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(mul_60, [-1]);  mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_794: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, view_52);  view_52 = None
        unsqueeze_375: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
        unsqueeze_376: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
        mul_795: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_374);  sub_260 = unsqueeze_374 = None
        sub_262: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_291, mul_795);  view_291 = mul_795 = None
        sub_263: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_372);  sub_262 = unsqueeze_372 = None
        mul_796: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_376);  sub_263 = unsqueeze_376 = None
        mul_797: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_151, squeeze_35);  sum_151 = squeeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_292: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_797, [512, 1, 1, 1]);  mul_797 = None
        mul_798: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_292, 0.1580497968320339);  view_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_293: "f32[512, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_796, [512, 128, 1, 1]);  mul_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_550: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_291, torch.float32);  getitem_291 = None
        convert_element_type_71: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sigmoid_49: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_71)
        mul_799: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_550, sigmoid_49);  convert_element_type_550 = None
        sub_264: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_49);  sigmoid_49 = None
        mul_800: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_71, sub_264);  convert_element_type_71 = sub_264 = None
        add_177: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_800, 1);  mul_800 = None
        mul_801: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_799, add_177);  mul_799 = add_177 = None
        convert_element_type_552: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_801, torch.bfloat16);  mul_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_152: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_552, [0, 2, 3])
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(convert_element_type_552, convert_element_type_68, convert_element_type_70, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  convert_element_type_552 = convert_element_type_68 = convert_element_type_70 = None
        getitem_294: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_60[0]
        getitem_295: "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_60[1];  convolution_backward_60 = None
        convert_element_type_553: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_295, torch.float32);  getitem_295 = None
        convert_element_type_554: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_152, torch.float32);  sum_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_75: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_553, memory_format = torch.contiguous_format);  convert_element_type_553 = None
        view_294: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [1, 128, 576]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_153: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_294, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_18: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_58, memory_format = torch.contiguous_format);  primals_58 = None
        view_48: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [1, 128, 576]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_265: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_48, unsqueeze_378);  view_48 = unsqueeze_378 = None
        mul_802: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_294, sub_265)
        sum_154: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 2]);  mul_802 = None
        mul_803: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, 0.001736111111111111);  sum_153 = None
        unsqueeze_379: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
        unsqueeze_380: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        mul_804: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, 0.001736111111111111)
        mul_805: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, squeeze_33)
        mul_806: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
        unsqueeze_381: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_382: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_57: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.07450538873672485);  primals_59 = None
        view_49: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_57, [-1]);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_807: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, view_49);  view_49 = None
        unsqueeze_383: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_384: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
        mul_808: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_382);  sub_265 = unsqueeze_382 = None
        sub_267: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_294, mul_808);  view_294 = mul_808 = None
        sub_268: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_380);  sub_267 = unsqueeze_380 = None
        mul_809: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_384);  sub_268 = unsqueeze_384 = None
        mul_810: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, squeeze_33);  sum_154 = squeeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_295: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_810, [128, 1, 1, 1]);  mul_810 = None
        mul_811: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_295, 0.07450538873672485);  view_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_296: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_809, [128, 64, 3, 3]);  mul_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_555: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_294, torch.float32);  getitem_294 = None
        convert_element_type_67: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sigmoid_50: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_67)
        mul_812: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_555, sigmoid_50);  convert_element_type_555 = None
        sub_269: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_50);  sigmoid_50 = None
        mul_813: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, sub_269);  convert_element_type_67 = sub_269 = None
        add_178: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_813, 1);  mul_813 = None
        mul_814: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_812, add_178);  mul_812 = add_178 = None
        convert_element_type_557: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_814, torch.bfloat16);  mul_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_155: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_557, [0, 2, 3])
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(convert_element_type_557, convert_element_type_64, convert_element_type_66, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  convert_element_type_557 = convert_element_type_64 = convert_element_type_66 = None
        getitem_297: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_61[0]
        getitem_298: "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_61[1];  convolution_backward_61 = None
        convert_element_type_558: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_298, torch.float32);  getitem_298 = None
        convert_element_type_559: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_155, torch.float32);  sum_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_76: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_558, memory_format = torch.contiguous_format);  convert_element_type_558 = None
        view_297: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [1, 128, 576]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_156: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_297, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_16: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_55, memory_format = torch.contiguous_format);  primals_55 = None
        view_45: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [1, 128, 576]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_270: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_45, unsqueeze_386);  view_45 = unsqueeze_386 = None
        mul_815: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_297, sub_270)
        sum_157: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2]);  mul_815 = None
        mul_816: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_156, 0.001736111111111111);  sum_156 = None
        unsqueeze_387: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
        unsqueeze_388: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
        mul_817: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, 0.001736111111111111)
        mul_818: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_819: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_817, mul_818);  mul_817 = mul_818 = None
        unsqueeze_389: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
        unsqueeze_390: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_54: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_56, 0.07450538873672485);  primals_56 = None
        view_46: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_54, [-1]);  mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_820: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, view_46);  view_46 = None
        unsqueeze_391: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
        unsqueeze_392: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        mul_821: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_390);  sub_270 = unsqueeze_390 = None
        sub_272: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_297, mul_821);  view_297 = mul_821 = None
        sub_273: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_388);  sub_272 = unsqueeze_388 = None
        mul_822: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_392);  sub_273 = unsqueeze_392 = None
        mul_823: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, squeeze_31);  sum_157 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_298: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_823, [128, 1, 1, 1]);  mul_823 = None
        mul_824: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_298, 0.07450538873672485);  view_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_299: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_822, [128, 64, 3, 3]);  mul_822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_560: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_297, torch.float32);  getitem_297 = None
        convert_element_type_63: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sigmoid_51: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_63)
        mul_825: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_560, sigmoid_51);  convert_element_type_560 = None
        sub_274: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_51);  sigmoid_51 = None
        mul_826: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_63, sub_274);  convert_element_type_63 = sub_274 = None
        add_179: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_826, 1);  mul_826 = None
        mul_827: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_825, add_179);  mul_825 = add_179 = None
        convert_element_type_562: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_827, torch.bfloat16);  mul_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_158: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_562, [0, 2, 3])
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(convert_element_type_562, mul_50, convert_element_type_62, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_562 = mul_50 = convert_element_type_62 = None
        getitem_300: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = convolution_backward_62[0]
        getitem_301: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_62[1];  convolution_backward_62 = None
        convert_element_type_563: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_301, torch.float32);  getitem_301 = None
        convert_element_type_564: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_158, torch.float32);  sum_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_300: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_563, [1, 128, 512]);  convert_element_type_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_159: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_300, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_42: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.reshape.default(primals_52, [1, 128, -1]);  primals_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_275: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_42, unsqueeze_394);  view_42 = unsqueeze_394 = None
        mul_828: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_300, sub_275)
        sum_160: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_828, [0, 2]);  mul_828 = None
        mul_829: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, 0.001953125);  sum_159 = None
        unsqueeze_395: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
        unsqueeze_396: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
        mul_830: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, 0.001953125)
        mul_831: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, squeeze_29)
        mul_832: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
        unsqueeze_397: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
        unsqueeze_398: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_51: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.07902489841601695);  primals_53 = None
        view_43: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_51, [-1]);  mul_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_833: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, view_43);  view_43 = None
        unsqueeze_399: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
        unsqueeze_400: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
        mul_834: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_398);  sub_275 = unsqueeze_398 = None
        sub_277: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_300, mul_834);  view_300 = mul_834 = None
        sub_278: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_396);  sub_277 = unsqueeze_396 = None
        mul_835: "f32[1, 128, 512][65536, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_400);  sub_278 = unsqueeze_400 = None
        mul_836: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, squeeze_29);  sum_160 = squeeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_301: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_836, [128, 1, 1, 1]);  mul_836 = None
        mul_837: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_301, 0.07902489841601695);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_302: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_835, [128, 512, 1, 1]);  mul_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_838: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(getitem_300, 0.9805806756909201);  getitem_300 = None
        convert_element_type_565: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_838, torch.float32);  mul_838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_1: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_47: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convolution_15, sigmoid_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_48: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, 2.0);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_49: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 0.2);  mul_48 = None
        add_26: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_49, convolution_11);  mul_49 = convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_59: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(add_26, torch.float32);  add_26 = None
        sigmoid_52: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_59)
        mul_839: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_565, sigmoid_52);  convert_element_type_565 = None
        sub_279: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_52);  sigmoid_52 = None
        mul_840: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_59, sub_279);  convert_element_type_59 = sub_279 = None
        add_180: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_840, 1);  mul_840 = None
        mul_841: "f32[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_839, add_180);  mul_839 = add_180 = None
        convert_element_type_567: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_841, torch.bfloat16);  mul_841 = None
        add_181: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_539, convert_element_type_567);  convert_element_type_539 = convert_element_type_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_842: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(add_181, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_843: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_842, 2.0);  mul_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_844: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_843, convolution_15);  convolution_15 = None
        mul_845: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_843, sigmoid_1);  mul_843 = None
        sum_161: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_844, [2, 3], True, dtype = torch.float32);  mul_844 = None
        convert_element_type_568: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_161, torch.bfloat16);  sum_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_569: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_568, torch.float32);  convert_element_type_568 = None
        convert_element_type_570: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_1, torch.float32);  sigmoid_1 = None
        sub_280: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_570)
        mul_846: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_570, sub_280);  convert_element_type_570 = sub_280 = None
        mul_847: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_569, mul_846);  convert_element_type_569 = mul_846 = None
        convert_element_type_571: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_847, torch.bfloat16);  mul_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_162: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_571, [0, 2, 3])
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(convert_element_type_571, relu_1, convert_element_type_58, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_571 = convert_element_type_58 = None
        getitem_303: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_63[0]
        getitem_304: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_63[1];  convolution_backward_63 = None
        convert_element_type_572: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_304, torch.float32);  getitem_304 = None
        convert_element_type_573: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_162, torch.float32);  sum_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_10: "b8[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_10: "bf16[128, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.aten.where.self(le_10, full_default, getitem_303);  le_10 = getitem_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_163: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(where_10, mean_1, convert_element_type_56, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = mean_1 = convert_element_type_56 = None
        getitem_306: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_64[0]
        getitem_307: "bf16[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_64[1];  convolution_backward_64 = None
        convert_element_type_574: "f32[128, 512, 1, 1][512, 1, 512, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_307, torch.float32);  getitem_307 = None
        convert_element_type_575: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_163, torch.float32);  sum_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_12: "bf16[128, 512, 28, 28][512, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_306, [128, 512, 28, 28]);  getitem_306 = None
        div_63: "bf16[128, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_12, 784);  expand_12 = None
        add_182: "bf16[128, 512, 28, 28][401408, 1, 14336, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_845, div_63);  mul_845 = div_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_164: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_182, [0, 2, 3])
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(add_182, convert_element_type_52, convert_element_type_54, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_182 = convert_element_type_52 = convert_element_type_54 = None
        getitem_309: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_65[0]
        getitem_310: "bf16[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_65[1];  convolution_backward_65 = None
        convert_element_type_576: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_310, torch.float32);  getitem_310 = None
        convert_element_type_577: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_164, torch.float32);  sum_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_303: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_576, [1, 512, 128]);  convert_element_type_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_165: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_39: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_45, [1, 512, -1]);  primals_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_281: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_39, unsqueeze_402);  view_39 = unsqueeze_402 = None
        mul_848: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_303, sub_281)
        sum_166: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_848, [0, 2]);  mul_848 = None
        mul_849: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, 0.0078125);  sum_165 = None
        unsqueeze_403: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_849, 0);  mul_849 = None
        unsqueeze_404: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        mul_850: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, 0.0078125)
        mul_851: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, squeeze_27)
        mul_852: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_850, mul_851);  mul_850 = mul_851 = None
        unsqueeze_405: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
        unsqueeze_406: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_44: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.1580497968320339);  primals_46 = None
        view_40: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(mul_44, [-1]);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_853: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, view_40);  view_40 = None
        unsqueeze_407: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_853, 0);  mul_853 = None
        unsqueeze_408: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
        mul_854: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_406);  sub_281 = unsqueeze_406 = None
        sub_283: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_303, mul_854);  view_303 = mul_854 = None
        sub_284: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_404);  sub_283 = unsqueeze_404 = None
        mul_855: "f32[1, 512, 128][65536, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_408);  sub_284 = unsqueeze_408 = None
        mul_856: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, squeeze_27);  sum_166 = squeeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_304: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_856, [512, 1, 1, 1]);  mul_856 = None
        mul_857: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_304, 0.1580497968320339);  view_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_305: "f32[512, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_855, [512, 128, 1, 1]);  mul_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_578: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_309, torch.float32);  getitem_309 = None
        convert_element_type_51: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sigmoid_53: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_51)
        mul_858: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_578, sigmoid_53);  convert_element_type_578 = None
        sub_285: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_53);  sigmoid_53 = None
        mul_859: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, sub_285);  convert_element_type_51 = sub_285 = None
        add_183: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_859, 1);  mul_859 = None
        mul_860: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_858, add_183);  mul_858 = add_183 = None
        convert_element_type_580: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_860, torch.bfloat16);  mul_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_167: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_580, [0, 2, 3])
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(convert_element_type_580, convert_element_type_48, convert_element_type_50, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  convert_element_type_580 = convert_element_type_48 = convert_element_type_50 = None
        getitem_312: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_66[0]
        getitem_313: "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_66[1];  convolution_backward_66 = None
        convert_element_type_581: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_313, torch.float32);  getitem_313 = None
        convert_element_type_582: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_167, torch.float32);  sum_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_77: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_581, memory_format = torch.contiguous_format);  convert_element_type_581 = None
        view_306: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [1, 128, 576]);  clone_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_168: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_306, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_14: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_42, memory_format = torch.contiguous_format);  primals_42 = None
        view_36: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [1, 128, 576]);  clone_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_286: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_36, unsqueeze_410);  view_36 = unsqueeze_410 = None
        mul_861: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_306, sub_286)
        sum_169: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_861, [0, 2]);  mul_861 = None
        mul_862: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_168, 0.001736111111111111);  sum_168 = None
        unsqueeze_411: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
        unsqueeze_412: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
        mul_863: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, 0.001736111111111111)
        mul_864: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_865: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_863, mul_864);  mul_863 = mul_864 = None
        unsqueeze_413: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
        unsqueeze_414: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_41: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_43, 0.07450538873672485);  primals_43 = None
        view_37: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_41, [-1]);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_866: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, view_37);  view_37 = None
        unsqueeze_415: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
        unsqueeze_416: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        mul_867: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_414);  sub_286 = unsqueeze_414 = None
        sub_288: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_306, mul_867);  view_306 = mul_867 = None
        sub_289: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_288, unsqueeze_412);  sub_288 = unsqueeze_412 = None
        mul_868: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_416);  sub_289 = unsqueeze_416 = None
        mul_869: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, squeeze_25);  sum_169 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_307: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_869, [128, 1, 1, 1]);  mul_869 = None
        mul_870: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_307, 0.07450538873672485);  view_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_308: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_868, [128, 64, 3, 3]);  mul_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_583: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_312, torch.float32);  getitem_312 = None
        convert_element_type_47: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sigmoid_54: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_47)
        mul_871: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, sigmoid_54);  convert_element_type_583 = None
        sub_290: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_54);  sigmoid_54 = None
        mul_872: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_47, sub_290);  convert_element_type_47 = sub_290 = None
        add_184: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_872, 1);  mul_872 = None
        mul_873: "f32[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_871, add_184);  mul_871 = add_184 = None
        convert_element_type_585: "bf16[128, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_873, torch.bfloat16);  mul_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_170: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_585, [0, 2, 3])
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(convert_element_type_585, convert_element_type_44, convert_element_type_46, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False]);  convert_element_type_585 = convert_element_type_44 = convert_element_type_46 = None
        getitem_315: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = convolution_backward_67[0]
        getitem_316: "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_67[1];  convolution_backward_67 = None
        convert_element_type_586: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_316, torch.float32);  getitem_316 = None
        convert_element_type_587: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_170, torch.float32);  sum_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_78: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_586, memory_format = torch.contiguous_format);  convert_element_type_586 = None
        view_309: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [1, 128, 576]);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_171: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_309, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_12: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_39, memory_format = torch.contiguous_format);  primals_39 = None
        view_33: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [1, 128, 576]);  clone_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_291: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_33, unsqueeze_418);  view_33 = unsqueeze_418 = None
        mul_874: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_309, sub_291)
        sum_172: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 2]);  mul_874 = None
        mul_875: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, 0.001736111111111111);  sum_171 = None
        unsqueeze_419: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
        unsqueeze_420: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
        mul_876: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, 0.001736111111111111)
        mul_877: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, squeeze_23)
        mul_878: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_876, mul_877);  mul_876 = mul_877 = None
        unsqueeze_421: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
        unsqueeze_422: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_38: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.07450538873672485);  primals_40 = None
        view_34: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_38, [-1]);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_879: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, view_34);  view_34 = None
        unsqueeze_423: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
        unsqueeze_424: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
        mul_880: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_422);  sub_291 = unsqueeze_422 = None
        sub_293: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_309, mul_880);  view_309 = mul_880 = None
        sub_294: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_420);  sub_293 = unsqueeze_420 = None
        mul_881: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_424);  sub_294 = unsqueeze_424 = None
        mul_882: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, squeeze_23);  sum_172 = squeeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_310: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_882, [128, 1, 1, 1]);  mul_882 = None
        mul_883: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_310, 0.07450538873672485);  view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_311: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_881, [128, 64, 3, 3]);  mul_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_588: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_315, torch.float32);  getitem_315 = None
        convert_element_type_43: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sigmoid_55: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_43)
        mul_884: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_588, sigmoid_55);  convert_element_type_588 = None
        sub_295: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_55);  sigmoid_55 = None
        mul_885: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_43, sub_295);  convert_element_type_43 = sub_295 = None
        add_185: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_885, 1);  mul_885 = None
        mul_886: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_884, add_185);  mul_884 = add_185 = None
        convert_element_type_590: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_886, torch.bfloat16);  mul_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_173: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_590, [0, 2, 3])
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(convert_element_type_590, mul_31, convert_element_type_42, [128], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_590 = convert_element_type_42 = None
        getitem_318: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = convolution_backward_68[0]
        getitem_319: "bf16[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_68[1];  convolution_backward_68 = None
        convert_element_type_591: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_319, torch.float32);  getitem_319 = None
        convert_element_type_592: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_173, torch.float32);  sum_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_312: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_591, [1, 128, 256]);  convert_element_type_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_174: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_312, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_30: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.reshape.default(primals_36, [1, 128, -1]);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_296: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_30, unsqueeze_426);  view_30 = unsqueeze_426 = None
        mul_887: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_312, sub_296)
        sum_175: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 2]);  mul_887 = None
        mul_888: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_174, 0.00390625);  sum_174 = None
        unsqueeze_427: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
        unsqueeze_428: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        mul_889: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, 0.00390625)
        mul_890: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, squeeze_21)
        mul_891: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
        unsqueeze_429: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
        unsqueeze_430: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_35: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_37, 0.11175808310508728);  primals_37 = None
        view_31: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_35, [-1]);  mul_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_892: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, view_31);  view_31 = None
        unsqueeze_431: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
        unsqueeze_432: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
        mul_893: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_430);  sub_296 = unsqueeze_430 = None
        sub_298: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_312, mul_893);  view_312 = mul_893 = None
        sub_299: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_428);  sub_298 = unsqueeze_428 = None
        mul_894: "f32[1, 128, 256][32768, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_432);  sub_299 = unsqueeze_432 = None
        mul_895: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, squeeze_21);  sum_175 = squeeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_313: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_895, [128, 1, 1, 1]);  mul_895 = None
        mul_896: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_313, 0.11175808310508728);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_314: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_894, [128, 256, 1, 1]);  mul_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_176: "bf16[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_181, [0, 2, 3])
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(add_181, avg_pool2d, convert_element_type_40, [512], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_181 = avg_pool2d = convert_element_type_40 = None
        getitem_321: "bf16[128, 256, 28, 28][200704, 1, 7168, 256]cuda:0" = convolution_backward_69[0]
        getitem_322: "bf16[512, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_69[1];  convolution_backward_69 = None
        convert_element_type_593: "f32[512, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_322, torch.float32);  getitem_322 = None
        convert_element_type_594: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_176, torch.float32);  sum_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_315: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_593, [1, 512, 256]);  convert_element_type_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_177: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_315, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_27: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.reshape.default(primals_33, [1, 512, -1]);  primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_300: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_27, unsqueeze_434);  view_27 = unsqueeze_434 = None
        mul_897: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_315, sub_300)
        sum_178: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2]);  mul_897 = None
        mul_898: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, 0.00390625);  sum_177 = None
        unsqueeze_435: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
        unsqueeze_436: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
        mul_899: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, 0.00390625)
        mul_900: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_901: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_899, mul_900);  mul_899 = mul_900 = None
        unsqueeze_437: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
        unsqueeze_438: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_32: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.11175808310508728);  primals_34 = None
        view_28: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(mul_32, [-1]);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_902: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, view_28);  view_28 = None
        unsqueeze_439: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
        unsqueeze_440: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        mul_903: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_438);  sub_300 = unsqueeze_438 = None
        sub_302: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_315, mul_903);  view_315 = mul_903 = None
        sub_303: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_436);  sub_302 = unsqueeze_436 = None
        mul_904: "f32[1, 512, 256][131072, 256, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_440);  sub_303 = unsqueeze_440 = None
        mul_905: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, squeeze_19);  sum_178 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_316: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_905, [512, 1, 1, 1]);  mul_905 = None
        mul_906: "f32[512, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_316, 0.11175808310508728);  view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_317: "f32[512, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_904, [512, 256, 1, 1]);  mul_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:149 in forward, code: return self.conv(self.pool(x))
        avg_pool2d_backward_2: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.avg_pool2d_backward.default(getitem_321, mul_31, [2, 2], [2, 2], [0, 0], True, False, None);  getitem_321 = mul_31 = None
        add_186: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(getitem_318, avg_pool2d_backward_2);  getitem_318 = avg_pool2d_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_907: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(add_186, 0.9805806756909201);  add_186 = None
        convert_element_type_595: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_907, torch.float32);  mul_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid: "bf16[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_28: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convolution_8, sigmoid)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_29: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, 2.0);  mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_30: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, 0.2);  mul_29 = None
        add_16: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_30, convolution_4);  mul_30 = convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        convert_element_type_37: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(add_16, torch.float32);  add_16 = None
        sigmoid_56: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_37)
        mul_908: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_595, sigmoid_56);  convert_element_type_595 = None
        sub_304: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_56);  sigmoid_56 = None
        mul_909: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, sub_304);  convert_element_type_37 = sub_304 = None
        add_187: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_909, 1);  mul_909 = None
        mul_910: "f32[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_908, add_187);  mul_908 = add_187 = None
        convert_element_type_597: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_910, torch.bfloat16);  mul_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:281 in forward, code: out = out * self.alpha + shortcut
        mul_911: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_597, 0.2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_912: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_911, 2.0);  mul_911 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_913: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_912, convolution_8);  convolution_8 = None
        mul_914: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_912, sigmoid);  mul_912 = None
        sum_179: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_913, [2, 3], True, dtype = torch.float32);  mul_913 = None
        convert_element_type_598: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_179, torch.bfloat16);  sum_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        convert_element_type_599: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_598, torch.float32);  convert_element_type_598 = None
        convert_element_type_600: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32);  sigmoid = None
        sub_305: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_600)
        mul_915: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_600, sub_305);  convert_element_type_600 = sub_305 = None
        mul_916: "f32[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_599, mul_915);  convert_element_type_599 = mul_915 = None
        convert_element_type_601: "bf16[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_916, torch.bfloat16);  mul_916 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_180: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_601, [0, 2, 3])
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(convert_element_type_601, relu, convert_element_type_36, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_601 = convert_element_type_36 = None
        getitem_324: "bf16[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_70[0]
        getitem_325: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_70[1];  convolution_backward_70 = None
        convert_element_type_602: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_325, torch.float32);  getitem_325 = None
        convert_element_type_603: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_180, torch.float32);  sum_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        le_11: "b8[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_11: "bf16[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.aten.where.self(le_11, full_default, getitem_324);  le_11 = full_default = getitem_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_181: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(where_11, mean, convert_element_type_34, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean = convert_element_type_34 = None
        getitem_327: "bf16[128, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_71[0]
        getitem_328: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_71[1];  convolution_backward_71 = None
        convert_element_type_604: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_328, torch.float32);  getitem_328 = None
        convert_element_type_605: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_181, torch.float32);  sum_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_13: "bf16[128, 256, 56, 56][256, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(getitem_327, [128, 256, 56, 56]);  getitem_327 = None
        div_64: "bf16[128, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_13, 3136);  expand_13 = None
        add_188: "bf16[128, 256, 56, 56][802816, 1, 14336, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_914, div_64);  mul_914 = div_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_182: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_188, [0, 2, 3])
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(add_188, convert_element_type_30, convert_element_type_32, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_188 = convert_element_type_30 = convert_element_type_32 = None
        getitem_330: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_72[0]
        getitem_331: "bf16[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_72[1];  convolution_backward_72 = None
        convert_element_type_606: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_331, torch.float32);  getitem_331 = None
        convert_element_type_607: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_182, torch.float32);  sum_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_318: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_606, [1, 256, 64]);  convert_element_type_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_183: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_318, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_24: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(primals_26, [1, 256, -1]);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_306: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_24, unsqueeze_442);  view_24 = unsqueeze_442 = None
        mul_917: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_318, sub_306)
        sum_184: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 2]);  mul_917 = None
        mul_918: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, 0.015625);  sum_183 = None
        unsqueeze_443: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
        unsqueeze_444: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
        mul_919: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, 0.015625)
        mul_920: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, squeeze_17)
        mul_921: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_919, mul_920);  mul_919 = mul_920 = None
        unsqueeze_445: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_921, 0);  mul_921 = None
        unsqueeze_446: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_25: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_27, 0.22351616621017456);  primals_27 = None
        view_25: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(mul_25, [-1]);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_922: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, view_25);  view_25 = None
        unsqueeze_447: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
        unsqueeze_448: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
        mul_923: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_446);  sub_306 = unsqueeze_446 = None
        sub_308: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_318, mul_923);  view_318 = mul_923 = None
        sub_309: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_444);  sub_308 = unsqueeze_444 = None
        mul_924: "f32[1, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_448);  sub_309 = unsqueeze_448 = None
        mul_925: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, squeeze_17);  sum_184 = squeeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_319: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_925, [256, 1, 1, 1]);  mul_925 = None
        mul_926: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_319, 0.22351616621017456);  view_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_320: "f32[256, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_924, [256, 64, 1, 1]);  mul_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:274 in forward, code: out = self.conv3(self.act3(out))
        convert_element_type_608: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_330, torch.float32);  getitem_330 = None
        convert_element_type_29: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sigmoid_57: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_29)
        mul_927: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_608, sigmoid_57);  convert_element_type_608 = None
        sub_310: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_57);  sigmoid_57 = None
        mul_928: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, sub_310);  convert_element_type_29 = sub_310 = None
        add_189: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_928, 1);  mul_928 = None
        mul_929: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_927, add_189);  mul_927 = add_189 = None
        convert_element_type_610: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_929, torch.bfloat16);  mul_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_185: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_610, [0, 2, 3])
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(convert_element_type_610, convert_element_type_26, convert_element_type_28, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_610 = convert_element_type_26 = convert_element_type_28 = None
        getitem_333: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_73[0]
        getitem_334: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_73[1];  convolution_backward_73 = None
        convert_element_type_611: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_334, torch.float32);  getitem_334 = None
        convert_element_type_612: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_185, torch.float32);  sum_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_79: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_611, memory_format = torch.contiguous_format);  convert_element_type_611 = None
        view_321: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [1, 64, 576]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_186: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_321, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_10: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_23, memory_format = torch.contiguous_format);  primals_23 = None
        view_21: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 64, 576]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_311: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_21, unsqueeze_450);  view_21 = unsqueeze_450 = None
        mul_930: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_321, sub_311)
        sum_187: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_930, [0, 2]);  mul_930 = None
        mul_931: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_186, 0.001736111111111111);  sum_186 = None
        unsqueeze_451: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
        unsqueeze_452: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        mul_932: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, 0.001736111111111111)
        mul_933: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, squeeze_15)
        mul_934: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_932, mul_933);  mul_932 = mul_933 = None
        unsqueeze_453: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_934, 0);  mul_934 = None
        unsqueeze_454: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_22: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_24, 0.07450538873672485);  primals_24 = None
        view_22: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_22, [-1]);  mul_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_935: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, view_22);  view_22 = None
        unsqueeze_455: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
        unsqueeze_456: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        mul_936: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_454);  sub_311 = unsqueeze_454 = None
        sub_313: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_321, mul_936);  view_321 = mul_936 = None
        sub_314: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_452);  sub_313 = unsqueeze_452 = None
        mul_937: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_456);  sub_314 = unsqueeze_456 = None
        mul_938: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_187, squeeze_15);  sum_187 = squeeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_322: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_938, [64, 1, 1, 1]);  mul_938 = None
        mul_939: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_322, 0.07450538873672485);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_323: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_937, [64, 64, 3, 3]);  mul_937 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:271 in forward, code: out = self.conv2b(self.act2b(out))
        convert_element_type_613: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_333, torch.float32);  getitem_333 = None
        convert_element_type_25: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sigmoid_58: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_25)
        mul_940: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_613, sigmoid_58);  convert_element_type_613 = None
        sub_315: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_58);  sigmoid_58 = None
        mul_941: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, sub_315);  convert_element_type_25 = sub_315 = None
        add_190: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_941, 1);  mul_941 = None
        mul_942: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_940, add_190);  mul_940 = add_190 = None
        convert_element_type_615: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_942, torch.bfloat16);  mul_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_188: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_615, [0, 2, 3])
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(convert_element_type_615, convert_element_type_22, convert_element_type_24, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_615 = convert_element_type_22 = convert_element_type_24 = None
        getitem_336: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_74[0]
        getitem_337: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_74[1];  convolution_backward_74 = None
        convert_element_type_616: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_337, torch.float32);  getitem_337 = None
        convert_element_type_617: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_188, torch.float32);  sum_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_80: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_616, memory_format = torch.contiguous_format);  convert_element_type_616 = None
        view_324: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [1, 64, 576]);  clone_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_189: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_324, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_8: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_20, memory_format = torch.contiguous_format);  primals_20 = None
        view_18: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1, 64, 576]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_316: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_18, unsqueeze_458);  view_18 = unsqueeze_458 = None
        mul_943: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_324, sub_316)
        sum_190: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_943, [0, 2]);  mul_943 = None
        mul_944: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_189, 0.001736111111111111);  sum_189 = None
        unsqueeze_459: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
        unsqueeze_460: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
        mul_945: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_190, 0.001736111111111111)
        mul_946: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_947: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_945, mul_946);  mul_945 = mul_946 = None
        unsqueeze_461: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
        unsqueeze_462: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_19: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_21, 0.07450538873672485);  primals_21 = None
        view_19: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_19, [-1]);  mul_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_948: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, view_19);  view_19 = None
        unsqueeze_463: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_948, 0);  mul_948 = None
        unsqueeze_464: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        mul_949: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_462);  sub_316 = unsqueeze_462 = None
        sub_318: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_324, mul_949);  view_324 = mul_949 = None
        sub_319: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_460);  sub_318 = unsqueeze_460 = None
        mul_950: "f32[1, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_464);  sub_319 = unsqueeze_464 = None
        mul_951: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_190, squeeze_13);  sum_190 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_325: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_951, [64, 1, 1, 1]);  mul_951 = None
        mul_952: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_325, 0.07450538873672485);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_326: "f32[64, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_950, [64, 64, 3, 3]);  mul_950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:269 in forward, code: out = self.conv2(self.act2(out))
        convert_element_type_618: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_336, torch.float32);  getitem_336 = None
        convert_element_type_21: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sigmoid_59: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_21)
        mul_953: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_618, sigmoid_59);  convert_element_type_618 = None
        sub_320: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_59);  sigmoid_59 = None
        mul_954: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_21, sub_320);  convert_element_type_21 = sub_320 = None
        add_191: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_954, 1);  mul_954 = None
        mul_955: "f32[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_953, add_191);  mul_953 = add_191 = None
        convert_element_type_620: "bf16[128, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_955, torch.bfloat16);  mul_955 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_191: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_620, [0, 2, 3])
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(convert_element_type_620, mul_12, convert_element_type_20, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_620 = convert_element_type_20 = None
        getitem_339: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = convolution_backward_75[0]
        getitem_340: "bf16[64, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_75[1];  convolution_backward_75 = None
        convert_element_type_621: "f32[64, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_340, torch.float32);  getitem_340 = None
        convert_element_type_622: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_191, torch.float32);  sum_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_327: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_621, [1, 64, 128]);  convert_element_type_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_192: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_327, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_15: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_17, [1, 64, -1]);  primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_321: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_15, unsqueeze_466);  view_15 = unsqueeze_466 = None
        mul_956: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_327, sub_321)
        sum_193: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_956, [0, 2]);  mul_956 = None
        mul_957: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_192, 0.0078125);  sum_192 = None
        unsqueeze_467: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_957, 0);  mul_957 = None
        unsqueeze_468: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        mul_958: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, 0.0078125)
        mul_959: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, squeeze_11)
        mul_960: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_958, mul_959);  mul_958 = mul_959 = None
        unsqueeze_469: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
        unsqueeze_470: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_16: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_18, 0.1580497968320339);  primals_18 = None
        view_16: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_16, [-1]);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_961: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, view_16);  view_16 = None
        unsqueeze_471: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_961, 0);  mul_961 = None
        unsqueeze_472: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
        mul_962: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_470);  sub_321 = unsqueeze_470 = None
        sub_323: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_327, mul_962);  view_327 = mul_962 = None
        sub_324: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_468);  sub_323 = unsqueeze_468 = None
        mul_963: "f32[1, 64, 128][8192, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_472);  sub_324 = unsqueeze_472 = None
        mul_964: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_193, squeeze_11);  sum_193 = squeeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_328: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_964, [64, 1, 1, 1]);  mul_964 = None
        mul_965: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_328, 0.1580497968320339);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_329: "f32[64, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_963, [64, 128, 1, 1]);  mul_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_194: "bf16[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_597, [0, 2, 3])
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(convert_element_type_597, mul_12, convert_element_type_18, [256], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_597 = mul_12 = convert_element_type_18 = None
        getitem_342: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = convolution_backward_76[0]
        getitem_343: "bf16[256, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_76[1];  convolution_backward_76 = None
        add_192: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(getitem_339, getitem_342);  getitem_339 = getitem_342 = None
        convert_element_type_623: "f32[256, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_343, torch.float32);  getitem_343 = None
        convert_element_type_624: "f32[256][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_194, torch.float32);  sum_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        view_330: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_623, [1, 256, 128]);  convert_element_type_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_195: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_330, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_12: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.reshape.default(primals_14, [1, 256, -1]);  primals_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_325: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_12, unsqueeze_474);  view_12 = unsqueeze_474 = None
        mul_966: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_330, sub_325)
        sum_196: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_966, [0, 2]);  mul_966 = None
        mul_967: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_195, 0.0078125);  sum_195 = None
        unsqueeze_475: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_967, 0);  mul_967 = None
        unsqueeze_476: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        mul_968: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_196, 0.0078125)
        mul_969: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, squeeze_9)
        mul_970: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_968, mul_969);  mul_968 = mul_969 = None
        unsqueeze_477: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_970, 0);  mul_970 = None
        unsqueeze_478: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_13: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_15, 0.1580497968320339);  primals_15 = None
        view_13: "f32[256][1]cuda:0" = torch.ops.aten.reshape.default(mul_13, [-1]);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_971: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, view_13);  view_13 = None
        unsqueeze_479: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
        unsqueeze_480: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        mul_972: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_325, unsqueeze_478);  sub_325 = unsqueeze_478 = None
        sub_327: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_330, mul_972);  view_330 = mul_972 = None
        sub_328: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_327, unsqueeze_476);  sub_327 = unsqueeze_476 = None
        mul_973: "f32[1, 256, 128][32768, 128, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_328, unsqueeze_480);  sub_328 = unsqueeze_480 = None
        mul_974: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_196, squeeze_9);  sum_196 = squeeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_331: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_974, [256, 1, 1, 1]);  mul_974 = None
        mul_975: "f32[256, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_331, 0.1580497968320339);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_332: "f32[256, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_973, [256, 128, 1, 1]);  mul_973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:260 in forward, code: out = self.act1(x) * self.beta
        mul_976: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(add_192, 1.0);  add_192 = None
        convert_element_type_625: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_976, torch.float32);  mul_976 = None
        convert_element_type_15: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sigmoid_60: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_15)
        mul_977: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_625, sigmoid_60);  convert_element_type_625 = None
        sub_329: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_60);  sigmoid_60 = None
        mul_978: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_15, sub_329);  convert_element_type_15 = sub_329 = None
        add_193: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_978, 1);  mul_978 = None
        mul_979: "f32[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_977, add_193);  mul_977 = add_193 = None
        convert_element_type_627: "bf16[128, 128, 56, 56][401408, 1, 7168, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_979, torch.bfloat16);  mul_979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_197: "bf16[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_627, [0, 2, 3])
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(convert_element_type_627, convert_element_type_12, convert_element_type_14, [128], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_627 = convert_element_type_12 = convert_element_type_14 = None
        getitem_345: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = convolution_backward_77[0]
        getitem_346: "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_77[1];  convolution_backward_77 = None
        convert_element_type_628: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_346, torch.float32);  getitem_346 = None
        convert_element_type_629: "f32[128][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_197, torch.float32);  sum_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_81: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_628, memory_format = torch.contiguous_format);  convert_element_type_628 = None
        view_333: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [1, 128, 576]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_198: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_333, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_6: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_11, memory_format = torch.contiguous_format);  primals_11 = None
        view_9: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [1, 128, 576]);  clone_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_330: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_9, unsqueeze_482);  view_9 = unsqueeze_482 = None
        mul_980: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_333, sub_330)
        sum_199: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_980, [0, 2]);  mul_980 = None
        mul_981: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_198, 0.001736111111111111);  sum_198 = None
        unsqueeze_483: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
        unsqueeze_484: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
        mul_982: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, 0.001736111111111111)
        mul_983: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_984: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_982, mul_983);  mul_982 = mul_983 = None
        unsqueeze_485: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
        unsqueeze_486: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_9: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_12, 0.07450538873672485);  primals_12 = None
        view_10: "f32[128][1]cuda:0" = torch.ops.aten.reshape.default(mul_9, [-1]);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_985: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, view_10);  view_10 = None
        unsqueeze_487: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_985, 0);  mul_985 = None
        unsqueeze_488: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        mul_986: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_486);  sub_330 = unsqueeze_486 = None
        sub_332: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_333, mul_986);  view_333 = mul_986 = None
        sub_333: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_332, unsqueeze_484);  sub_332 = unsqueeze_484 = None
        mul_987: "f32[1, 128, 576][73728, 576, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_333, unsqueeze_488);  sub_333 = unsqueeze_488 = None
        mul_988: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_199, squeeze_7);  sum_199 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_334: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_988, [128, 1, 1, 1]);  mul_988 = None
        mul_989: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_334, 0.07450538873672485);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_335: "f32[128, 64, 3, 3][576, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_987, [128, 64, 3, 3]);  mul_987 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        convert_element_type_630: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_345, torch.float32);  getitem_345 = None
        convert_element_type_11: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sigmoid_61: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_11)
        mul_990: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_630, sigmoid_61);  convert_element_type_630 = None
        sub_334: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_61);  sigmoid_61 = None
        mul_991: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, sub_334);  convert_element_type_11 = sub_334 = None
        add_194: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_991, 1);  mul_991 = None
        mul_992: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_990, add_194);  mul_990 = add_194 = None
        convert_element_type_632: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_992, torch.bfloat16);  mul_992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_200: "bf16[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_632, [0, 2, 3])
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(convert_element_type_632, convert_element_type_8, convert_element_type_10, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_632 = convert_element_type_8 = convert_element_type_10 = None
        getitem_348: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = convolution_backward_78[0]
        getitem_349: "bf16[64, 32, 3, 3][288, 1, 96, 32]cuda:0" = convolution_backward_78[1];  convolution_backward_78 = None
        convert_element_type_633: "f32[64, 32, 3, 3][288, 1, 96, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_349, torch.float32);  getitem_349 = None
        convert_element_type_634: "f32[64][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_200, torch.float32);  sum_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_82: "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_633, memory_format = torch.contiguous_format);  convert_element_type_633 = None
        view_336: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [1, 64, 288]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_201: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_336, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_4: "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_8, memory_format = torch.contiguous_format);  primals_8 = None
        view_6: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [1, 64, 288]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_335: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_6, unsqueeze_490);  view_6 = unsqueeze_490 = None
        mul_993: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_336, sub_335)
        sum_202: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_993, [0, 2]);  mul_993 = None
        mul_994: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_201, 0.003472222222222222);  sum_201 = None
        unsqueeze_491: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_994, 0);  mul_994 = None
        unsqueeze_492: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        mul_995: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_202, 0.003472222222222222)
        mul_996: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
        mul_997: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_995, mul_996);  mul_995 = mul_996 = None
        unsqueeze_493: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_997, 0);  mul_997 = None
        unsqueeze_494: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_6: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_9, 0.10536653122135592);  primals_9 = None
        view_7: "f32[64][1]cuda:0" = torch.ops.aten.reshape.default(mul_6, [-1]);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_998: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, view_7);  view_7 = None
        unsqueeze_495: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
        unsqueeze_496: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
        mul_999: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_335, unsqueeze_494);  sub_335 = unsqueeze_494 = None
        sub_337: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_336, mul_999);  view_336 = mul_999 = None
        sub_338: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_337, unsqueeze_492);  sub_337 = unsqueeze_492 = None
        mul_1000: "f32[1, 64, 288][18432, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_338, unsqueeze_496);  sub_338 = unsqueeze_496 = None
        mul_1001: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_202, squeeze_5);  sum_202 = squeeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_337: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1001, [64, 1, 1, 1]);  mul_1001 = None
        mul_1002: "f32[64, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_337, 0.10536653122135592);  view_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_338: "f32[64, 32, 3, 3][288, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1000, [64, 32, 3, 3]);  mul_1000 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        convert_element_type_635: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_348, torch.float32);  getitem_348 = None
        convert_element_type_7: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sigmoid_62: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_7)
        mul_1003: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_635, sigmoid_62);  convert_element_type_635 = None
        sub_339: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_62);  sigmoid_62 = None
        mul_1004: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_7, sub_339);  convert_element_type_7 = sub_339 = None
        add_195: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_1004, 1);  mul_1004 = None
        mul_1005: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_1003, add_195);  mul_1003 = add_195 = None
        convert_element_type_637: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1005, torch.bfloat16);  mul_1005 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_203: "bf16[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_637, [0, 2, 3])
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(convert_element_type_637, convert_element_type_4, convert_element_type_6, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_637 = convert_element_type_4 = convert_element_type_6 = None
        getitem_351: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_79[0]
        getitem_352: "bf16[32, 16, 3, 3][144, 1, 48, 16]cuda:0" = convolution_backward_79[1];  convolution_backward_79 = None
        convert_element_type_638: "f32[32, 16, 3, 3][144, 1, 48, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_352, torch.float32);  getitem_352 = None
        convert_element_type_639: "f32[32][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_203, torch.float32);  sum_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_83: "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_638, memory_format = torch.contiguous_format);  convert_element_type_638 = None
        view_339: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [1, 32, 144]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_204: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_339, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_2: "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_5, memory_format = torch.contiguous_format);  primals_5 = None
        view_3: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1, 32, 144]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_340: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_3, unsqueeze_498);  view_3 = unsqueeze_498 = None
        mul_1006: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_339, sub_340)
        sum_205: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1006, [0, 2]);  mul_1006 = None
        mul_1007: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_204, 0.006944444444444444);  sum_204 = None
        unsqueeze_499: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
        unsqueeze_500: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        mul_1008: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, 0.006944444444444444)
        mul_1009: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
        mul_1010: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1008, mul_1009);  mul_1008 = mul_1009 = None
        unsqueeze_501: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
        unsqueeze_502: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_3: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_6, 0.1490107774734497);  primals_6 = None
        view_4: "f32[32][1]cuda:0" = torch.ops.aten.reshape.default(mul_3, [-1]);  mul_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_1011: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, view_4);  view_4 = None
        unsqueeze_503: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1011, 0);  mul_1011 = None
        unsqueeze_504: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        mul_1012: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_340, unsqueeze_502);  sub_340 = unsqueeze_502 = None
        sub_342: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_339, mul_1012);  view_339 = mul_1012 = None
        sub_343: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_342, unsqueeze_500);  sub_342 = unsqueeze_500 = None
        mul_1013: "f32[1, 32, 144][4608, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_343, unsqueeze_504);  sub_343 = unsqueeze_504 = None
        mul_1014: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_205, squeeze_3);  sum_205 = squeeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_340: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1014, [32, 1, 1, 1]);  mul_1014 = None
        mul_1015: "f32[32, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_340, 0.1490107774734497);  view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_341: "f32[32, 16, 3, 3][144, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1013, [32, 16, 3, 3]);  mul_1013 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:563 in forward_features, code: x = self.stem(x)
        convert_element_type_640: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_351, torch.float32);  getitem_351 = None
        convert_element_type_3: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sigmoid_63: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sigmoid.default(convert_element_type_3)
        mul_1016: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_640, sigmoid_63);  convert_element_type_640 = None
        sub_344: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(1, sigmoid_63);  sigmoid_63 = None
        mul_1017: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_3, sub_344);  convert_element_type_3 = sub_344 = None
        add_196: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_1017, 1);  mul_1017 = None
        mul_1018: "f32[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_1016, add_196);  mul_1016 = add_196 = None
        convert_element_type_642: "bf16[128, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1018, torch.bfloat16);  mul_1018 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:169 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_206: "bf16[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_642, [0, 2, 3])
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(convert_element_type_642, convert_element_type_2, convert_element_type_1, [16], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_642 = convert_element_type_2 = convert_element_type_1 = None
        getitem_355: "bf16[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_80[1];  convolution_backward_80 = None
        convert_element_type_643: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_355, torch.float32);  getitem_355 = None
        convert_element_type_644: "f32[16][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_206, torch.float32);  sum_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:168 in forward, code: ).reshape_as(self.weight)
        clone_84: "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_643, memory_format = torch.contiguous_format);  convert_element_type_643 = None
        view_342: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [1, 16, 27]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sum_207: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_342, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone: "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.aten.clone.default(primals_1, memory_format = torch.contiguous_format);  primals_1 = None
        view: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.reshape.default(clone, [1, 16, 27]);  clone = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        sub_345: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.sub.Tensor(view, unsqueeze_506);  view = unsqueeze_506 = None
        mul_1019: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_342, sub_345)
        sum_208: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1019, [0, 2]);  mul_1019 = None
        mul_1020: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_207, 0.037037037037037035);  sum_207 = None
        unsqueeze_507: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1020, 0);  mul_1020 = None
        unsqueeze_508: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
        mul_1021: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_208, 0.037037037037037035)
        mul_1022: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1023: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1021, mul_1022);  mul_1021 = mul_1022 = None
        unsqueeze_509: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
        unsqueeze_510: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(primals_2, 0.34412564994580647);  primals_2 = None
        view_1: "f32[16][1]cuda:0" = torch.ops.aten.reshape.default(mul, [-1]);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:160 in forward, code: weight = F.batch_norm(
        mul_1024: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, view_1);  view_1 = None
        unsqueeze_511: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_1024, 0);  mul_1024 = None
        unsqueeze_512: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        mul_1025: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_345, unsqueeze_510);  sub_345 = unsqueeze_510 = None
        sub_347: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_342, mul_1025);  view_342 = mul_1025 = None
        sub_348: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_347, unsqueeze_508);  sub_347 = unsqueeze_508 = None
        mul_1026: "f32[1, 16, 27][432, 27, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_348, unsqueeze_512);  sub_348 = unsqueeze_512 = None
        mul_1027: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_208, squeeze_1);  sum_208 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:164 in forward, code: weight=(self.gain * self.scale).view(-1),
        view_343: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1027, [16, 1, 1, 1]);  mul_1027 = None
        mul_1028: "f32[16, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_343, 0.34412564994580647);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:161 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        view_344: "f32[16, 3, 3, 3][27, 9, 3, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1026, [16, 3, 3, 3]);  mul_1026 = None
        return (view_344, mul_1028, convert_element_type_644, None, view_341, mul_1015, convert_element_type_639, view_338, mul_1002, convert_element_type_634, view_335, mul_989, convert_element_type_629, view_332, mul_975, convert_element_type_624, view_329, mul_965, convert_element_type_622, view_326, mul_952, convert_element_type_617, view_323, mul_939, convert_element_type_612, view_320, mul_926, convert_element_type_607, convert_element_type_604, convert_element_type_605, convert_element_type_602, convert_element_type_603, view_317, mul_906, convert_element_type_594, view_314, mul_896, convert_element_type_592, view_311, mul_883, convert_element_type_587, view_308, mul_870, convert_element_type_582, view_305, mul_857, convert_element_type_577, convert_element_type_574, convert_element_type_575, convert_element_type_572, convert_element_type_573, view_302, mul_837, convert_element_type_564, view_299, mul_824, convert_element_type_559, view_296, mul_811, convert_element_type_554, view_293, mul_798, convert_element_type_549, convert_element_type_546, convert_element_type_547, convert_element_type_544, convert_element_type_545, view_290, mul_778, convert_element_type_536, view_287, mul_768, convert_element_type_534, view_284, mul_755, convert_element_type_529, view_281, mul_742, convert_element_type_524, view_278, mul_729, convert_element_type_519, convert_element_type_516, convert_element_type_517, convert_element_type_514, convert_element_type_515, view_275, mul_709, convert_element_type_506, view_272, mul_696, convert_element_type_501, view_269, mul_683, convert_element_type_496, view_266, mul_670, convert_element_type_491, convert_element_type_488, convert_element_type_489, convert_element_type_486, convert_element_type_487, view_263, mul_650, convert_element_type_478, view_260, mul_637, convert_element_type_473, view_257, mul_624, convert_element_type_468, view_254, mul_611, convert_element_type_463, convert_element_type_460, convert_element_type_461, convert_element_type_458, convert_element_type_459, view_251, mul_591, convert_element_type_450, view_248, mul_578, convert_element_type_445, view_245, mul_565, convert_element_type_440, view_242, mul_552, convert_element_type_435, convert_element_type_432, convert_element_type_433, convert_element_type_430, convert_element_type_431, view_239, mul_532, convert_element_type_422, view_236, mul_519, convert_element_type_417, view_233, mul_506, convert_element_type_412, view_230, mul_493, convert_element_type_407, convert_element_type_404, convert_element_type_405, convert_element_type_402, convert_element_type_403, view_227, mul_473, convert_element_type_394, view_224, mul_460, convert_element_type_389, view_221, mul_447, convert_element_type_384, view_218, mul_434, convert_element_type_379, convert_element_type_376, convert_element_type_377, convert_element_type_374, convert_element_type_375, view_215, mul_414, convert_element_type_366, view_212, mul_404, convert_element_type_364, view_209, mul_391, convert_element_type_359, view_206, mul_378, convert_element_type_354, view_203, mul_365, convert_element_type_349, convert_element_type_346, convert_element_type_347, convert_element_type_344, convert_element_type_345, view_200, mul_345, convert_element_type_336, view_197, mul_332, convert_element_type_331, view_194, mul_319, convert_element_type_326, view_191, mul_306, convert_element_type_321, convert_element_type_318, convert_element_type_319, convert_element_type_316, convert_element_type_317, view_188, mul_286, convert_element_type_308, view_185, mul_273, convert_element_type_303, view_182, mul_260, convert_element_type_298, view_179, mul_247, convert_element_type_293, convert_element_type_290, convert_element_type_291, convert_element_type_288, convert_element_type_289, view_176, mul_231, convert_element_type_283, convert_element_type_277, convert_element_type_278)
