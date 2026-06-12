class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[32][1]cuda:0", primals_7: "f32[32][1]cuda:0", primals_12: "f32[32][1]cuda:0", primals_13: "f32[32][1]cuda:0", primals_18: "f32[16][1]cuda:0", primals_24: "f32[96][1]cuda:0", primals_25: "f32[96][1]cuda:0", primals_30: "f32[96][1]cuda:0", primals_31: "f32[96][1]cuda:0", primals_36: "f32[24][1]cuda:0", primals_42: "f32[144][1]cuda:0", primals_43: "f32[144][1]cuda:0", primals_48: "f32[144][1]cuda:0", primals_49: "f32[144][1]cuda:0", primals_54: "f32[24][1]cuda:0", primals_60: "f32[144][1]cuda:0", primals_61: "f32[144][1]cuda:0", primals_66: "f32[144][1]cuda:0", primals_67: "f32[144][1]cuda:0", primals_72: "f32[32][1]cuda:0", primals_78: "f32[192][1]cuda:0", primals_79: "f32[192][1]cuda:0", primals_84: "f32[192][1]cuda:0", primals_85: "f32[192][1]cuda:0", primals_90: "f32[32][1]cuda:0", primals_96: "f32[192][1]cuda:0", primals_97: "f32[192][1]cuda:0", primals_102: "f32[192][1]cuda:0", primals_103: "f32[192][1]cuda:0", primals_108: "f32[32][1]cuda:0", primals_114: "f32[192][1]cuda:0", primals_115: "f32[192][1]cuda:0", primals_120: "f32[192][1]cuda:0", primals_121: "f32[192][1]cuda:0", primals_126: "f32[64][1]cuda:0", primals_132: "f32[384][1]cuda:0", primals_133: "f32[384][1]cuda:0", primals_138: "f32[384][1]cuda:0", primals_139: "f32[384][1]cuda:0", primals_144: "f32[64][1]cuda:0", primals_150: "f32[384][1]cuda:0", primals_151: "f32[384][1]cuda:0", primals_156: "f32[384][1]cuda:0", primals_157: "f32[384][1]cuda:0", primals_162: "f32[64][1]cuda:0", primals_168: "f32[384][1]cuda:0", primals_169: "f32[384][1]cuda:0", primals_174: "f32[384][1]cuda:0", primals_175: "f32[384][1]cuda:0", primals_180: "f32[64][1]cuda:0", primals_186: "f32[384][1]cuda:0", primals_187: "f32[384][1]cuda:0", primals_192: "f32[384][1]cuda:0", primals_193: "f32[384][1]cuda:0", primals_198: "f32[96][1]cuda:0", primals_204: "f32[576][1]cuda:0", primals_205: "f32[576][1]cuda:0", primals_210: "f32[576][1]cuda:0", primals_211: "f32[576][1]cuda:0", primals_216: "f32[96][1]cuda:0", primals_222: "f32[576][1]cuda:0", primals_223: "f32[576][1]cuda:0", primals_228: "f32[576][1]cuda:0", primals_229: "f32[576][1]cuda:0", primals_234: "f32[96][1]cuda:0", primals_240: "f32[576][1]cuda:0", primals_241: "f32[576][1]cuda:0", primals_246: "f32[576][1]cuda:0", primals_247: "f32[576][1]cuda:0", primals_252: "f32[160][1]cuda:0", primals_258: "f32[960][1]cuda:0", primals_259: "f32[960][1]cuda:0", primals_264: "f32[960][1]cuda:0", primals_265: "f32[960][1]cuda:0", primals_270: "f32[160][1]cuda:0", primals_276: "f32[960][1]cuda:0", primals_277: "f32[960][1]cuda:0", primals_282: "f32[960][1]cuda:0", primals_283: "f32[960][1]cuda:0", primals_288: "f32[160][1]cuda:0", primals_294: "f32[960][1]cuda:0", primals_295: "f32[960][1]cuda:0", primals_300: "f32[960][1]cuda:0", primals_301: "f32[960][1]cuda:0", primals_306: "f32[320][1]cuda:0", primals_312: "f32[1280][1]cuda:0", primals_313: "f32[1280][1]cuda:0", convert_element_type: "bf16[32, 3, 3, 3][27, 1, 9, 3]cuda:0", convert_element_type_1: "bf16[96, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0", getitem_1: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", rsqrt: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", convert_element_type_5: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0", convert_element_type_6: "bf16[32, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_1: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0", getitem_3: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", rsqrt_1: "f32[1, 32, 1, 1][32, 1, 32, 32]cuda:0", convert_element_type_10: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0", convert_element_type_11: "bf16[16, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_2: "bf16[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0", squeeze_7: "f32[16][1]cuda:0", convert_element_type_13: "bf16[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0", convert_element_type_14: "bf16[96, 16, 1, 1][16, 1, 16, 16]cuda:0", convolution_3: "bf16[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0", getitem_7: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", rsqrt_3: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", convert_element_type_18: "bf16[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0", convert_element_type_19: "bf16[96, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_4: "bf16[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0", getitem_9: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", rsqrt_4: "f32[1, 96, 1, 1][96, 1, 96, 96]cuda:0", convert_element_type_23: "bf16[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0", convert_element_type_24: "bf16[24, 96, 1, 1][96, 1, 96, 96]cuda:0", convolution_5: "bf16[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0", squeeze_16: "f32[24][1]cuda:0", convert_element_type_26: "bf16[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0", convert_element_type_27: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0", convolution_6: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0", getitem_13: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", rsqrt_6: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", convert_element_type_31: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0", convert_element_type_32: "bf16[144, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_7: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0", getitem_15: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", rsqrt_7: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", convert_element_type_36: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0", convert_element_type_37: "bf16[24, 144, 1, 1][144, 1, 144, 144]cuda:0", convolution_8: "bf16[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0", squeeze_25: "f32[24][1]cuda:0", add_45: "bf16[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0", convert_element_type_40: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0", convolution_9: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0", getitem_19: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", rsqrt_9: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", convert_element_type_44: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0", convert_element_type_45: "bf16[144, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_10: "bf16[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0", getitem_21: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", rsqrt_10: "f32[1, 144, 1, 1][144, 1, 144, 144]cuda:0", convert_element_type_49: "bf16[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0", convert_element_type_50: "bf16[32, 144, 1, 1][144, 1, 144, 144]cuda:0", convolution_11: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0", squeeze_34: "f32[32][1]cuda:0", convert_element_type_52: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0", convert_element_type_53: "bf16[192, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_12: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", getitem_25: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_12: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_57: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convert_element_type_58: "bf16[192, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_13: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", getitem_27: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_13: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_62: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convert_element_type_63: "bf16[32, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_14: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0", squeeze_43: "f32[32][1]cuda:0", add_76: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0", convert_element_type_66: "bf16[192, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_15: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", getitem_31: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_15: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_70: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convert_element_type_71: "bf16[192, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_16: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", getitem_33: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_16: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_75: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convert_element_type_76: "bf16[32, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_17: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0", squeeze_52: "f32[32][1]cuda:0", add_92: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0", convert_element_type_79: "bf16[192, 32, 1, 1][32, 1, 32, 32]cuda:0", convolution_18: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", getitem_37: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_18: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_83: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convert_element_type_84: "bf16[192, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_19: "bf16[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0", getitem_39: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", rsqrt_19: "f32[1, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_88: "bf16[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0", convert_element_type_89: "bf16[64, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_20: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0", squeeze_61: "f32[64][1]cuda:0", convert_element_type_91: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0", convert_element_type_92: "bf16[384, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_21: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", getitem_43: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_21: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_96: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_97: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_22: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", getitem_45: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_22: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_101: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_102: "bf16[64, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_23: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0", squeeze_70: "f32[64][1]cuda:0", add_123: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0", convert_element_type_105: "bf16[384, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_24: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", getitem_49: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_24: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_109: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_110: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_25: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", getitem_51: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_25: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_114: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_115: "bf16[64, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_26: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0", squeeze_79: "f32[64][1]cuda:0", add_139: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0", convert_element_type_118: "bf16[384, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_27: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", getitem_55: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_27: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_122: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_123: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_28: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", getitem_57: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_28: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_127: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_128: "bf16[64, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_29: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0", squeeze_88: "f32[64][1]cuda:0", add_155: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0", convert_element_type_131: "bf16[384, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_30: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", getitem_61: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_30: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_135: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_136: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_31: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", getitem_63: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", rsqrt_31: "f32[1, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_140: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_141: "bf16[96, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_32: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0", squeeze_97: "f32[96][1]cuda:0", convert_element_type_143: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0", convert_element_type_144: "bf16[576, 96, 1, 1][96, 1, 96, 96]cuda:0", convolution_33: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", getitem_67: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", rsqrt_33: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", convert_element_type_148: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", convert_element_type_149: "bf16[576, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_34: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", getitem_69: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", rsqrt_34: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", convert_element_type_153: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", convert_element_type_154: "bf16[96, 576, 1, 1][576, 1, 576, 576]cuda:0", convolution_35: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0", squeeze_106: "f32[96][1]cuda:0", add_186: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0", convert_element_type_157: "bf16[576, 96, 1, 1][96, 1, 96, 96]cuda:0", convolution_36: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", getitem_73: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", rsqrt_36: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", convert_element_type_161: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", convert_element_type_162: "bf16[576, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_37: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", getitem_75: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", rsqrt_37: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", convert_element_type_166: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", convert_element_type_167: "bf16[96, 576, 1, 1][576, 1, 576, 576]cuda:0", convolution_38: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0", squeeze_115: "f32[96][1]cuda:0", add_202: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0", convert_element_type_170: "bf16[576, 96, 1, 1][96, 1, 96, 96]cuda:0", convolution_39: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", getitem_79: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", rsqrt_39: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", convert_element_type_174: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0", convert_element_type_175: "bf16[576, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_40: "bf16[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0", getitem_81: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", rsqrt_40: "f32[1, 576, 1, 1][576, 1, 576, 576]cuda:0", convert_element_type_179: "bf16[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0", convert_element_type_180: "bf16[160, 576, 1, 1][576, 1, 576, 576]cuda:0", convolution_41: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0", squeeze_124: "f32[160][1]cuda:0", convert_element_type_182: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convert_element_type_183: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0", convolution_42: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_85: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_42: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_187: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_188: "bf16[960, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_43: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_87: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_43: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_192: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_193: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0", convolution_44: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0", squeeze_133: "f32[160][1]cuda:0", add_233: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convert_element_type_196: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0", convolution_45: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_91: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_45: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_200: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_201: "bf16[960, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_46: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_93: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_46: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_205: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_206: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0", convolution_47: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0", squeeze_142: "f32[160][1]cuda:0", add_249: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0", convert_element_type_209: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0", convolution_48: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_97: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_48: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_213: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_214: "bf16[960, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_49: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", getitem_99: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", rsqrt_49: "f32[1, 960, 1, 1][960, 1, 960, 960]cuda:0", convert_element_type_218: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0", convert_element_type_219: "bf16[320, 960, 1, 1][960, 1, 960, 960]cuda:0", convolution_50: "bf16[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0", squeeze_151: "f32[320][1]cuda:0", convert_element_type_221: "bf16[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0", convert_element_type_222: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convolution_51: "bf16[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0", getitem_103: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0", rsqrt_51: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0", gt: "b8[96, 1280][1280, 1]cuda:0", mul_365: "bf16[96, 1280][1280, 1]cuda:0", permute_1: "bf16[1000, 1280][1280, 1]cuda:0", unsqueeze_222: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0", unsqueeze_258: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_294: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_330: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", unsqueeze_366: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_402: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_438: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_474: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_510: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_546: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_582: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_618: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_654: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_690: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", unsqueeze_726: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", unsqueeze_762: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", unsqueeze_798: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", tangents_1: "bf16[96, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:170 in _forward_impl, code: x = self.classifier(x)
        mm: "bf16[96, 1280][1280, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[1000, 96][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 1280][1280, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, mul_365);  permute_2 = mul_365 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_236: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_237: "f32[1000, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_238: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_236, torch.float32);  convert_element_type_236 = None
        convert_element_type_239: "bf16[96, 1280][1280, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_366: "bf16[96, 1280][1280, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, 1.25);  convert_element_type_239 = None
        mul_367: "bf16[96, 1280][1280, 1]cuda:0" = torch.ops.aten.mul.Tensor(mm, mul_366);  mm = mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:169 in _forward_impl, code: x = torch.flatten(x, 1)
        view_2: "bf16[96, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mul_367, [96, 1280, 1, 1]);  mul_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:168 in _forward_impl, code: x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        expand: "bf16[96, 1280, 7, 7][1280, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(view_2, [96, 1280, 7, 7]);  view_2 = None
        div: "bf16[96, 1280, 7, 7][62720, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:166 in _forward_impl, code: x = self.features(x)
        sub_51: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_103)
        mul_357: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        unsqueeze_204: "f32[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_205: "f32[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[1280, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_207: "f32[1280, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_269: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        convert_element_type_224: "bf16[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_269, torch.bfloat16);  add_269 = None
        le: "b8[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_224, 0.0)
        ge: "b8[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_224, 6.0);  convert_element_type_224 = None
        bitwise_or: "b8[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le, ge);  le = ge = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.where.self(bitwise_or, full_default, div);  bitwise_or = div = None
        convert_element_type_240: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_153: "f32[1280][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        unsqueeze_208: "f32[1, 1280][1280, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_209: "f32[1, 1280, 1][1280, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 2);  unsqueeze_208 = None
        unsqueeze_210: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 3);  unsqueeze_209 = None
        sum_2: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_240, [0, 2, 3])
        convert_element_type_223: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sub_52: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_223, unsqueeze_210);  convert_element_type_223 = unsqueeze_210 = None
        mul_368: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_240, sub_52)
        sum_3: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3]);  mul_368 = None
        mul_369: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.00021258503401360543)
        unsqueeze_211: "f32[1, 1280][1280, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
        unsqueeze_212: "f32[1, 1280, 1][1280, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_370: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.00021258503401360543)
        squeeze_154: "f32[1280][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_371: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_372: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_370, mul_371);  mul_370 = mul_371 = None
        unsqueeze_214: "f32[1, 1280][1280, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
        unsqueeze_215: "f32[1, 1280, 1][1280, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_373: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_312);  primals_312 = None
        unsqueeze_217: "f32[1, 1280][1280, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
        unsqueeze_218: "f32[1, 1280, 1][1280, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_374: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_216);  sub_52 = unsqueeze_216 = None
        sub_54: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, mul_374);  convert_element_type_240 = mul_374 = None
        sub_55: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.sub.Tensor(sub_54, unsqueeze_213);  sub_54 = unsqueeze_213 = None
        mul_375: "f32[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_219);  sub_55 = unsqueeze_219 = None
        mul_376: "f32[1280][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_154);  sum_3 = squeeze_154 = None
        convert_element_type_242: "bf16[96, 1280, 7, 7][62720, 1, 8960, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_375, torch.bfloat16);  mul_375 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_242, convert_element_type_221, convert_element_type_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_242 = convert_element_type_221 = convert_element_type_222 = None
        getitem_104: "bf16[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = convolution_backward[0]
        getitem_105: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_243: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:64 in forward, code: return self.conv(x)
        convert_element_type_244: "f32[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_104, torch.float32);  getitem_104 = None
        sum_4: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_244, [0, 2, 3])
        convert_element_type_220: "f32[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        sub_56: "f32[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_220, unsqueeze_222);  convert_element_type_220 = unsqueeze_222 = None
        mul_377: "f32[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_244, sub_56)
        sum_5: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 2, 3]);  mul_377 = None
        mul_378: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.00021258503401360543)
        unsqueeze_223: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
        unsqueeze_224: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_379: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.00021258503401360543)
        mul_380: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_381: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        unsqueeze_226: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
        unsqueeze_227: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_382: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_306);  primals_306 = None
        unsqueeze_229: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
        unsqueeze_230: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_383: "f32[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_228);  sub_56 = unsqueeze_228 = None
        sub_58: "f32[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_244, mul_383);  convert_element_type_244 = mul_383 = None
        sub_59: "f32[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_225);  sub_58 = unsqueeze_225 = None
        mul_384: "f32[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_231);  sub_59 = unsqueeze_231 = None
        mul_385: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_151);  sum_5 = squeeze_151 = None
        convert_element_type_246: "bf16[96, 320, 7, 7][15680, 1, 2240, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_384, torch.bfloat16);  mul_384 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_246, convert_element_type_218, convert_element_type_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_246 = convert_element_type_218 = convert_element_type_219 = None
        getitem_107: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_1[0]
        getitem_108: "bf16[320, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_247: "f32[320, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_108, torch.float32);  getitem_108 = None
        sub_49: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, getitem_99)
        mul_343: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        unsqueeze_196: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_197: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_349: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        unsqueeze_198: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_301, -1);  primals_301 = None
        unsqueeze_199: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_259: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        convert_element_type_216: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None
        le_1: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_216, 0.0)
        ge_1: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_216, 6.0);  convert_element_type_216 = None
        bitwise_or_1: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_1, ge_1);  le_1 = ge_1 = None
        where_1: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(bitwise_or_1, full_default, getitem_107);  bitwise_or_1 = getitem_107 = None
        convert_element_type_248: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        squeeze_147: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        unsqueeze_232: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_233: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        sum_6: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_248, [0, 2, 3])
        convert_element_type_215: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        sub_60: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_215, unsqueeze_234);  convert_element_type_215 = unsqueeze_234 = None
        mul_386: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_248, sub_60)
        sum_7: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 2, 3]);  mul_386 = None
        mul_387: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.00021258503401360543)
        unsqueeze_235: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
        unsqueeze_236: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_388: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.00021258503401360543)
        squeeze_148: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_389: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_390: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_388, mul_389);  mul_388 = mul_389 = None
        unsqueeze_238: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_390, 0);  mul_390 = None
        unsqueeze_239: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_391: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_300);  primals_300 = None
        unsqueeze_241: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
        unsqueeze_242: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_392: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_240);  sub_60 = unsqueeze_240 = None
        sub_62: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_248, mul_392);  convert_element_type_248 = mul_392 = None
        sub_63: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_237);  sub_62 = unsqueeze_237 = None
        mul_393: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_243);  sub_63 = unsqueeze_243 = None
        mul_394: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_148);  sum_7 = squeeze_148 = None
        convert_element_type_250: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_393, torch.bfloat16);  mul_393 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_250, convert_element_type_213, convert_element_type_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False]);  convert_element_type_250 = convert_element_type_213 = convert_element_type_214 = None
        getitem_110: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_2[0]
        getitem_111: "bf16[960, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_251: "f32[960, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
        sub_48: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_48, getitem_97)
        mul_336: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        unsqueeze_192: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_193: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_295, -1);  primals_295 = None
        unsqueeze_195: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_254: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        convert_element_type_211: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16);  add_254 = None
        le_2: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_211, 0.0)
        ge_2: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_211, 6.0);  convert_element_type_211 = None
        bitwise_or_2: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_2, ge_2);  le_2 = ge_2 = None
        where_2: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(bitwise_or_2, full_default, getitem_110);  bitwise_or_2 = getitem_110 = None
        convert_element_type_252: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        squeeze_144: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        unsqueeze_244: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_245: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        sum_8: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_252, [0, 2, 3])
        convert_element_type_210: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_48, torch.float32);  convolution_48 = None
        sub_64: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_210, unsqueeze_246);  convert_element_type_210 = unsqueeze_246 = None
        mul_395: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_252, sub_64)
        sum_9: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 2, 3]);  mul_395 = None
        mul_396: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 0.00021258503401360543)
        unsqueeze_247: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
        unsqueeze_248: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_397: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.00021258503401360543)
        squeeze_145: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_398: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_399: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
        unsqueeze_250: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_399, 0);  mul_399 = None
        unsqueeze_251: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_400: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_294);  primals_294 = None
        unsqueeze_253: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
        unsqueeze_254: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_401: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_252);  sub_64 = unsqueeze_252 = None
        sub_66: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_252, mul_401);  convert_element_type_252 = mul_401 = None
        sub_67: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_249);  sub_66 = unsqueeze_249 = None
        mul_402: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_255);  sub_67 = unsqueeze_255 = None
        mul_403: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_145);  sum_9 = squeeze_145 = None
        convert_element_type_254: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_402, torch.bfloat16);  mul_402 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_254, add_249, convert_element_type_209, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_254 = add_249 = convert_element_type_209 = None
        getitem_113: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_3[0]
        getitem_114: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_255: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_114, torch.float32);  getitem_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:62 in forward, code: return x + self.conv(x)
        convert_element_type_256: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32)
        sum_10: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_256, [0, 2, 3])
        convert_element_type_207: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        sub_68: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_207, unsqueeze_258);  convert_element_type_207 = unsqueeze_258 = None
        mul_404: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_256, sub_68)
        sum_11: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 2, 3]);  mul_404 = None
        mul_405: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.00021258503401360543)
        unsqueeze_259: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
        unsqueeze_260: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_406: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.00021258503401360543)
        mul_407: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_408: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
        unsqueeze_262: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
        unsqueeze_263: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_409: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_288);  primals_288 = None
        unsqueeze_265: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
        unsqueeze_266: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_410: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_264);  sub_68 = unsqueeze_264 = None
        sub_70: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_256, mul_410);  convert_element_type_256 = mul_410 = None
        sub_71: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_261);  sub_70 = unsqueeze_261 = None
        mul_411: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_267);  sub_71 = unsqueeze_267 = None
        mul_412: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_142);  sum_11 = squeeze_142 = None
        convert_element_type_258: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_411, torch.bfloat16);  mul_411 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_258, convert_element_type_205, convert_element_type_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_258 = convert_element_type_205 = convert_element_type_206 = None
        getitem_116: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_4[0]
        getitem_117: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_259: "f32[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
        sub_46: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, getitem_93)
        mul_322: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        unsqueeze_184: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_282, -1)
        unsqueeze_185: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_328: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        unsqueeze_186: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_283, -1);  primals_283 = None
        unsqueeze_187: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_243: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
        convert_element_type_203: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_243, torch.bfloat16);  add_243 = None
        le_3: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_203, 0.0)
        ge_3: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_203, 6.0);  convert_element_type_203 = None
        bitwise_or_3: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_3, ge_3);  le_3 = ge_3 = None
        where_3: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(bitwise_or_3, full_default, getitem_116);  bitwise_or_3 = getitem_116 = None
        convert_element_type_260: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        squeeze_138: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        unsqueeze_268: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_269: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
        unsqueeze_270: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
        sum_12: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_260, [0, 2, 3])
        convert_element_type_202: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32);  convolution_46 = None
        sub_72: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_202, unsqueeze_270);  convert_element_type_202 = unsqueeze_270 = None
        mul_413: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_260, sub_72)
        sum_13: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 2, 3]);  mul_413 = None
        mul_414: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.00021258503401360543)
        unsqueeze_271: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
        unsqueeze_272: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_415: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.00021258503401360543)
        squeeze_139: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_416: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_417: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_415, mul_416);  mul_415 = mul_416 = None
        unsqueeze_274: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_417, 0);  mul_417 = None
        unsqueeze_275: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        mul_418: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_282);  primals_282 = None
        unsqueeze_277: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
        unsqueeze_278: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_419: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_276);  sub_72 = unsqueeze_276 = None
        sub_74: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_260, mul_419);  convert_element_type_260 = mul_419 = None
        sub_75: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_273);  sub_74 = unsqueeze_273 = None
        mul_420: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_279);  sub_75 = unsqueeze_279 = None
        mul_421: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_139);  sum_13 = squeeze_139 = None
        convert_element_type_262: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_420, torch.bfloat16);  mul_420 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_262, convert_element_type_200, convert_element_type_201, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False]);  convert_element_type_262 = convert_element_type_200 = convert_element_type_201 = None
        getitem_119: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_5[0]
        getitem_120: "bf16[960, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_263: "f32[960, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_120, torch.float32);  getitem_120 = None
        sub_45: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_91)
        mul_315: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        unsqueeze_180: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_181: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_321: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        unsqueeze_182: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_183: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_238: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        convert_element_type_198: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_238, torch.bfloat16);  add_238 = None
        le_4: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_198, 0.0)
        ge_4: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_198, 6.0);  convert_element_type_198 = None
        bitwise_or_4: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_4, ge_4);  le_4 = ge_4 = None
        where_4: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(bitwise_or_4, full_default, getitem_119);  bitwise_or_4 = getitem_119 = None
        convert_element_type_264: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        squeeze_135: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        unsqueeze_280: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_281: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
        unsqueeze_282: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
        sum_14: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_264, [0, 2, 3])
        convert_element_type_197: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sub_76: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_197, unsqueeze_282);  convert_element_type_197 = unsqueeze_282 = None
        mul_422: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_264, sub_76)
        sum_15: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 2, 3]);  mul_422 = None
        mul_423: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.00021258503401360543)
        unsqueeze_283: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
        unsqueeze_284: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_424: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.00021258503401360543)
        squeeze_136: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_425: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_426: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        unsqueeze_286: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
        unsqueeze_287: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        mul_427: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_276);  primals_276 = None
        unsqueeze_289: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_427, 0);  mul_427 = None
        unsqueeze_290: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_428: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_288);  sub_76 = unsqueeze_288 = None
        sub_78: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_264, mul_428);  convert_element_type_264 = mul_428 = None
        sub_79: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_285);  sub_78 = unsqueeze_285 = None
        mul_429: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_291);  sub_79 = unsqueeze_291 = None
        mul_430: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_136);  sum_15 = squeeze_136 = None
        convert_element_type_266: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_429, torch.bfloat16);  mul_429 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_266, add_233, convert_element_type_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_266 = add_233 = convert_element_type_196 = None
        getitem_122: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_6[0]
        getitem_123: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        add_270: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.add.Tensor(getitem_113, getitem_122);  getitem_113 = getitem_122 = None
        convert_element_type_267: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None
        convert_element_type_268: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_270, torch.float32)
        sum_16: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_268, [0, 2, 3])
        convert_element_type_194: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        sub_80: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_194, unsqueeze_294);  convert_element_type_194 = unsqueeze_294 = None
        mul_431: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_268, sub_80)
        sum_17: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 2, 3]);  mul_431 = None
        mul_432: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.00021258503401360543)
        unsqueeze_295: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
        unsqueeze_296: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_433: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 0.00021258503401360543)
        mul_434: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_435: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
        unsqueeze_298: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
        unsqueeze_299: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        mul_436: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_270);  primals_270 = None
        unsqueeze_301: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
        unsqueeze_302: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_437: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_300);  sub_80 = unsqueeze_300 = None
        sub_82: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_268, mul_437);  convert_element_type_268 = mul_437 = None
        sub_83: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_297);  sub_82 = unsqueeze_297 = None
        mul_438: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_303);  sub_83 = unsqueeze_303 = None
        mul_439: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_133);  sum_17 = squeeze_133 = None
        convert_element_type_270: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_438, torch.bfloat16);  mul_438 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_270, convert_element_type_192, convert_element_type_193, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_270 = convert_element_type_192 = convert_element_type_193 = None
        getitem_125: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_7[0]
        getitem_126: "bf16[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_271: "f32[160, 960, 1, 1][960, 1, 960, 960]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_126, torch.float32);  getitem_126 = None
        sub_43: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, getitem_87)
        mul_301: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        unsqueeze_172: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_173: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_307: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        unsqueeze_174: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
        unsqueeze_175: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_227: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        convert_element_type_190: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_227, torch.bfloat16);  add_227 = None
        le_5: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_190, 0.0)
        ge_5: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_190, 6.0);  convert_element_type_190 = None
        bitwise_or_5: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_5, ge_5);  le_5 = ge_5 = None
        where_5: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(bitwise_or_5, full_default, getitem_125);  bitwise_or_5 = getitem_125 = None
        convert_element_type_272: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        squeeze_129: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        unsqueeze_304: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_305: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
        unsqueeze_306: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
        sum_18: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_272, [0, 2, 3])
        convert_element_type_189: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        sub_84: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_189, unsqueeze_306);  convert_element_type_189 = unsqueeze_306 = None
        mul_440: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_272, sub_84)
        sum_19: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_440, [0, 2, 3]);  mul_440 = None
        mul_441: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.00021258503401360543)
        unsqueeze_307: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
        unsqueeze_308: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_442: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.00021258503401360543)
        squeeze_130: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_443: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_444: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
        unsqueeze_310: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
        unsqueeze_311: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        mul_445: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_264);  primals_264 = None
        unsqueeze_313: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
        unsqueeze_314: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_446: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_312);  sub_84 = unsqueeze_312 = None
        sub_86: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_272, mul_446);  convert_element_type_272 = mul_446 = None
        sub_87: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_309);  sub_86 = unsqueeze_309 = None
        mul_447: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_315);  sub_87 = unsqueeze_315 = None
        mul_448: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_130);  sum_19 = squeeze_130 = None
        convert_element_type_274: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_447, torch.bfloat16);  mul_447 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_274, convert_element_type_187, convert_element_type_188, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 960, [True, True, False]);  convert_element_type_274 = convert_element_type_187 = convert_element_type_188 = None
        getitem_128: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = convolution_backward_8[0]
        getitem_129: "bf16[960, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_275: "f32[960, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None
        sub_42: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, getitem_85)
        mul_294: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        unsqueeze_168: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_258, -1)
        unsqueeze_169: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_300: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        unsqueeze_170: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
        unsqueeze_171: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_222: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        convert_element_type_185: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(add_222, torch.bfloat16);  add_222 = None
        le_6: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_185, 0.0)
        ge_6: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_185, 6.0);  convert_element_type_185 = None
        bitwise_or_6: "b8[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_6, ge_6);  le_6 = ge_6 = None
        where_6: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.where.self(bitwise_or_6, full_default, getitem_128);  bitwise_or_6 = getitem_128 = None
        convert_element_type_276: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        squeeze_126: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        unsqueeze_316: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_317: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
        unsqueeze_318: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
        sum_20: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_276, [0, 2, 3])
        convert_element_type_184: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        sub_88: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_184, unsqueeze_318);  convert_element_type_184 = unsqueeze_318 = None
        mul_449: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_276, sub_88)
        sum_21: "f32[960][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 2, 3]);  mul_449 = None
        mul_450: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.00021258503401360543)
        unsqueeze_319: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
        unsqueeze_320: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_451: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.00021258503401360543)
        squeeze_127: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_452: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_453: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_451, mul_452);  mul_451 = mul_452 = None
        unsqueeze_322: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
        unsqueeze_323: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        mul_454: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_258);  primals_258 = None
        unsqueeze_325: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
        unsqueeze_326: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_455: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_324);  sub_88 = unsqueeze_324 = None
        sub_90: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_276, mul_455);  convert_element_type_276 = mul_455 = None
        sub_91: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_321);  sub_90 = unsqueeze_321 = None
        mul_456: "f32[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_327);  sub_91 = unsqueeze_327 = None
        mul_457: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_127);  sum_21 = squeeze_127 = None
        convert_element_type_278: "bf16[96, 960, 7, 7][47040, 1, 6720, 960]cuda:0" = torch.ops.prims.convert_element_type.default(mul_456, torch.bfloat16);  mul_456 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_278, convert_element_type_182, convert_element_type_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_278 = convert_element_type_182 = convert_element_type_183 = None
        getitem_131: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = convolution_backward_9[0]
        getitem_132: "bf16[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        add_271: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.add.Tensor(add_270, getitem_131);  add_270 = getitem_131 = None
        convert_element_type_279: "f32[960, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_132, torch.float32);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:64 in forward, code: return self.conv(x)
        convert_element_type_280: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_271, torch.float32);  add_271 = None
        sum_22: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_280, [0, 2, 3])
        convert_element_type_181: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_92: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_181, unsqueeze_330);  convert_element_type_181 = unsqueeze_330 = None
        mul_458: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, sub_92)
        sum_23: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
        mul_459: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 0.00021258503401360543)
        unsqueeze_331: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
        unsqueeze_332: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_460: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 0.00021258503401360543)
        mul_461: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_462: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
        unsqueeze_334: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
        unsqueeze_335: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
        mul_463: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_252);  primals_252 = None
        unsqueeze_337: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
        unsqueeze_338: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_464: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_336);  sub_92 = unsqueeze_336 = None
        sub_94: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_280, mul_464);  convert_element_type_280 = mul_464 = None
        sub_95: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_333);  sub_94 = unsqueeze_333 = None
        mul_465: "f32[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_339);  sub_95 = unsqueeze_339 = None
        mul_466: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_124);  sum_23 = squeeze_124 = None
        convert_element_type_282: "bf16[96, 160, 7, 7][7840, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(mul_465, torch.bfloat16);  mul_465 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_282, convert_element_type_179, convert_element_type_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_282 = convert_element_type_179 = convert_element_type_180 = None
        getitem_134: "bf16[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = convolution_backward_10[0]
        getitem_135: "bf16[160, 576, 1, 1][576, 1, 576, 576]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_283: "f32[160, 576, 1, 1][576, 1, 576, 576]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None
        sub_40: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_81)
        mul_280: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        unsqueeze_160: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_161: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_163: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_212: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        convert_element_type_177: "bf16[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.prims.convert_element_type.default(add_212, torch.bfloat16);  add_212 = None
        le_7: "b8[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_177, 0.0)
        ge_7: "b8[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_177, 6.0);  convert_element_type_177 = None
        bitwise_or_7: "b8[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_7, ge_7);  le_7 = ge_7 = None
        where_7: "bf16[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.where.self(bitwise_or_7, full_default, getitem_134);  bitwise_or_7 = getitem_134 = None
        convert_element_type_284: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        squeeze_120: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        unsqueeze_340: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_341: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
        unsqueeze_342: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
        sum_24: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_284, [0, 2, 3])
        convert_element_type_176: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_96: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_176, unsqueeze_342);  convert_element_type_176 = unsqueeze_342 = None
        mul_467: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_284, sub_96)
        sum_25: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
        mul_468: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 0.00021258503401360543)
        unsqueeze_343: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
        unsqueeze_344: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_469: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 0.00021258503401360543)
        squeeze_121: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_470: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_471: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
        unsqueeze_346: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
        unsqueeze_347: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        mul_472: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_246);  primals_246 = None
        unsqueeze_349: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
        unsqueeze_350: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_473: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_348);  sub_96 = unsqueeze_348 = None
        sub_98: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_284, mul_473);  convert_element_type_284 = mul_473 = None
        sub_99: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_345);  sub_98 = unsqueeze_345 = None
        mul_474: "f32[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_351);  sub_99 = unsqueeze_351 = None
        mul_475: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_121);  sum_25 = squeeze_121 = None
        convert_element_type_286: "bf16[96, 576, 7, 7][28224, 1, 4032, 576]cuda:0" = torch.ops.prims.convert_element_type.default(mul_474, torch.bfloat16);  mul_474 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_286, convert_element_type_174, convert_element_type_175, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  convert_element_type_286 = convert_element_type_174 = convert_element_type_175 = None
        getitem_137: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = convolution_backward_11[0]
        getitem_138: "bf16[576, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_287: "f32[576, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_138, torch.float32);  getitem_138 = None
        sub_39: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, getitem_79)
        mul_273: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        unsqueeze_156: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_157: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_279: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        unsqueeze_158: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
        unsqueeze_159: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_207: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        convert_element_type_172: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(add_207, torch.bfloat16);  add_207 = None
        le_8: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_172, 0.0)
        ge_8: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_172, 6.0);  convert_element_type_172 = None
        bitwise_or_8: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_8, ge_8);  le_8 = ge_8 = None
        where_8: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.where.self(bitwise_or_8, full_default, getitem_137);  bitwise_or_8 = getitem_137 = None
        convert_element_type_288: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        squeeze_117: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        unsqueeze_352: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_353: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
        unsqueeze_354: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
        sum_26: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_288, [0, 2, 3])
        convert_element_type_171: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sub_100: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_171, unsqueeze_354);  convert_element_type_171 = unsqueeze_354 = None
        mul_476: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_288, sub_100)
        sum_27: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3]);  mul_476 = None
        mul_477: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 5.314625850340136e-05)
        unsqueeze_355: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
        unsqueeze_356: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_478: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 5.314625850340136e-05)
        squeeze_118: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_479: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_480: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
        unsqueeze_358: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
        unsqueeze_359: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        mul_481: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_240);  primals_240 = None
        unsqueeze_361: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
        unsqueeze_362: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_482: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_360);  sub_100 = unsqueeze_360 = None
        sub_102: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_288, mul_482);  convert_element_type_288 = mul_482 = None
        sub_103: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_357);  sub_102 = unsqueeze_357 = None
        mul_483: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_363);  sub_103 = unsqueeze_363 = None
        mul_484: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_118);  sum_27 = squeeze_118 = None
        convert_element_type_290: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(mul_483, torch.bfloat16);  mul_483 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_290, add_202, convert_element_type_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_290 = add_202 = convert_element_type_170 = None
        getitem_140: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = convolution_backward_12[0]
        getitem_141: "bf16[576, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_291: "f32[576, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:62 in forward, code: return x + self.conv(x)
        convert_element_type_292: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_140, torch.float32)
        sum_28: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_292, [0, 2, 3])
        convert_element_type_168: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32);  convolution_38 = None
        sub_104: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_168, unsqueeze_366);  convert_element_type_168 = unsqueeze_366 = None
        mul_485: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_292, sub_104)
        sum_29: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_485, [0, 2, 3]);  mul_485 = None
        mul_486: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 5.314625850340136e-05)
        unsqueeze_367: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
        unsqueeze_368: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_487: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 5.314625850340136e-05)
        mul_488: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_489: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
        unsqueeze_370: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
        unsqueeze_371: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        mul_490: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_234);  primals_234 = None
        unsqueeze_373: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
        unsqueeze_374: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_491: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_372);  sub_104 = unsqueeze_372 = None
        sub_106: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_292, mul_491);  convert_element_type_292 = mul_491 = None
        sub_107: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_369);  sub_106 = unsqueeze_369 = None
        mul_492: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_375);  sub_107 = unsqueeze_375 = None
        mul_493: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_115);  sum_29 = squeeze_115 = None
        convert_element_type_294: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_492, torch.bfloat16);  mul_492 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_294, convert_element_type_166, convert_element_type_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_294 = convert_element_type_166 = convert_element_type_167 = None
        getitem_143: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = convolution_backward_13[0]
        getitem_144: "bf16[96, 576, 1, 1][576, 1, 576, 576]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_295: "f32[96, 576, 1, 1][576, 1, 576, 576]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None
        sub_37: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, getitem_75)
        mul_259: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        unsqueeze_148: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_151: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_196: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        convert_element_type_164: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None
        le_9: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_164, 0.0)
        ge_9: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_164, 6.0);  convert_element_type_164 = None
        bitwise_or_9: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_9, ge_9);  le_9 = ge_9 = None
        where_9: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.where.self(bitwise_or_9, full_default, getitem_143);  bitwise_or_9 = getitem_143 = None
        convert_element_type_296: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        squeeze_111: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        unsqueeze_376: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_377: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        sum_30: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_296, [0, 2, 3])
        convert_element_type_163: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sub_108: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, unsqueeze_378);  convert_element_type_163 = unsqueeze_378 = None
        mul_494: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, sub_108)
        sum_31: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_494, [0, 2, 3]);  mul_494 = None
        mul_495: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 5.314625850340136e-05)
        unsqueeze_379: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
        unsqueeze_380: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_496: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 5.314625850340136e-05)
        squeeze_112: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_497: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_498: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_496, mul_497);  mul_496 = mul_497 = None
        unsqueeze_382: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
        unsqueeze_383: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        mul_499: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  primals_228 = None
        unsqueeze_385: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
        unsqueeze_386: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_500: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_384);  sub_108 = unsqueeze_384 = None
        sub_110: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_296, mul_500);  convert_element_type_296 = mul_500 = None
        sub_111: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_381);  sub_110 = unsqueeze_381 = None
        mul_501: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_387);  sub_111 = unsqueeze_387 = None
        mul_502: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_112);  sum_31 = squeeze_112 = None
        convert_element_type_298: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(mul_501, torch.bfloat16);  mul_501 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_298, convert_element_type_161, convert_element_type_162, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  convert_element_type_298 = convert_element_type_161 = convert_element_type_162 = None
        getitem_146: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = convolution_backward_14[0]
        getitem_147: "bf16[576, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_299: "f32[576, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
        sub_36: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_73)
        mul_252: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        unsqueeze_144: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_145: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_258: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        unsqueeze_146: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_147: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_191: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
        convert_element_type_159: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16);  add_191 = None
        le_10: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_159, 0.0)
        ge_10: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_159, 6.0);  convert_element_type_159 = None
        bitwise_or_10: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_10, ge_10);  le_10 = ge_10 = None
        where_10: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.where.self(bitwise_or_10, full_default, getitem_146);  bitwise_or_10 = getitem_146 = None
        convert_element_type_300: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        squeeze_108: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        unsqueeze_388: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_389: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        sum_32: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_300, [0, 2, 3])
        convert_element_type_158: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_112: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_158, unsqueeze_390);  convert_element_type_158 = unsqueeze_390 = None
        mul_503: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_300, sub_112)
        sum_33: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2, 3]);  mul_503 = None
        mul_504: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 5.314625850340136e-05)
        unsqueeze_391: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
        unsqueeze_392: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_505: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 5.314625850340136e-05)
        squeeze_109: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_506: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_507: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
        unsqueeze_394: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
        unsqueeze_395: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_508: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_222);  primals_222 = None
        unsqueeze_397: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
        unsqueeze_398: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_509: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_396);  sub_112 = unsqueeze_396 = None
        sub_114: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_300, mul_509);  convert_element_type_300 = mul_509 = None
        sub_115: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_393);  sub_114 = unsqueeze_393 = None
        mul_510: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_399);  sub_115 = unsqueeze_399 = None
        mul_511: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_109);  sum_33 = squeeze_109 = None
        convert_element_type_302: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(mul_510, torch.bfloat16);  mul_510 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_302, add_186, convert_element_type_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_302 = add_186 = convert_element_type_157 = None
        getitem_149: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = convolution_backward_15[0]
        getitem_150: "bf16[576, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        add_272: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.add.Tensor(getitem_140, getitem_149);  getitem_140 = getitem_149 = None
        convert_element_type_303: "f32[576, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_150, torch.float32);  getitem_150 = None
        convert_element_type_304: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_272, torch.float32)
        sum_34: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_304, [0, 2, 3])
        convert_element_type_155: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32);  convolution_35 = None
        sub_116: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_155, unsqueeze_402);  convert_element_type_155 = unsqueeze_402 = None
        mul_512: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_304, sub_116)
        sum_35: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 2, 3]);  mul_512 = None
        mul_513: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 5.314625850340136e-05)
        unsqueeze_403: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
        unsqueeze_404: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_514: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 5.314625850340136e-05)
        mul_515: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_516: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_514, mul_515);  mul_514 = mul_515 = None
        unsqueeze_406: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
        unsqueeze_407: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        mul_517: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  primals_216 = None
        unsqueeze_409: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
        unsqueeze_410: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_518: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_408);  sub_116 = unsqueeze_408 = None
        sub_118: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_304, mul_518);  convert_element_type_304 = mul_518 = None
        sub_119: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_405);  sub_118 = unsqueeze_405 = None
        mul_519: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_411);  sub_119 = unsqueeze_411 = None
        mul_520: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_106);  sum_35 = squeeze_106 = None
        convert_element_type_306: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_519, torch.bfloat16);  mul_519 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_306, convert_element_type_153, convert_element_type_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_306 = convert_element_type_153 = convert_element_type_154 = None
        getitem_152: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = convolution_backward_16[0]
        getitem_153: "bf16[96, 576, 1, 1][576, 1, 576, 576]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_307: "f32[96, 576, 1, 1][576, 1, 576, 576]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None
        sub_34: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_69)
        mul_238: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        unsqueeze_136: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_137: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_139: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_180: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        convert_element_type_151: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(add_180, torch.bfloat16);  add_180 = None
        le_11: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_151, 0.0)
        ge_11: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_151, 6.0);  convert_element_type_151 = None
        bitwise_or_11: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_11, ge_11);  le_11 = ge_11 = None
        where_11: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.where.self(bitwise_or_11, full_default, getitem_152);  bitwise_or_11 = getitem_152 = None
        convert_element_type_308: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        squeeze_102: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        unsqueeze_412: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_413: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
        sum_36: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_308, [0, 2, 3])
        convert_element_type_150: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_120: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_150, unsqueeze_414);  convert_element_type_150 = unsqueeze_414 = None
        mul_521: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_308, sub_120)
        sum_37: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 2, 3]);  mul_521 = None
        mul_522: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 5.314625850340136e-05)
        unsqueeze_415: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
        unsqueeze_416: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_523: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 5.314625850340136e-05)
        squeeze_103: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_524: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_525: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_523, mul_524);  mul_523 = mul_524 = None
        unsqueeze_418: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
        unsqueeze_419: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        mul_526: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_210);  primals_210 = None
        unsqueeze_421: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
        unsqueeze_422: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_527: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_420);  sub_120 = unsqueeze_420 = None
        sub_122: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_308, mul_527);  convert_element_type_308 = mul_527 = None
        sub_123: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_417);  sub_122 = unsqueeze_417 = None
        mul_528: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_423);  sub_123 = unsqueeze_423 = None
        mul_529: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_103);  sum_37 = squeeze_103 = None
        convert_element_type_310: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(mul_528, torch.bfloat16);  mul_528 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_310, convert_element_type_148, convert_element_type_149, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  convert_element_type_310 = convert_element_type_148 = convert_element_type_149 = None
        getitem_155: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = convolution_backward_17[0]
        getitem_156: "bf16[576, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_311: "f32[576, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_156, torch.float32);  getitem_156 = None
        sub_33: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_67)
        mul_231: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        unsqueeze_132: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_133: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_237: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        unsqueeze_134: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_135: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_175: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        convert_element_type_146: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None
        le_12: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_146, 0.0)
        ge_12: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_146, 6.0);  convert_element_type_146 = None
        bitwise_or_12: "b8[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_12, ge_12);  le_12 = ge_12 = None
        where_12: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.where.self(bitwise_or_12, full_default, getitem_155);  bitwise_or_12 = getitem_155 = None
        convert_element_type_312: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32);  where_12 = None
        squeeze_99: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        unsqueeze_424: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_425: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
        unsqueeze_426: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
        sum_38: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_312, [0, 2, 3])
        convert_element_type_145: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sub_124: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_145, unsqueeze_426);  convert_element_type_145 = unsqueeze_426 = None
        mul_530: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_312, sub_124)
        sum_39: "f32[576][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 2, 3]);  mul_530 = None
        mul_531: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 5.314625850340136e-05)
        unsqueeze_427: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
        unsqueeze_428: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_532: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 5.314625850340136e-05)
        squeeze_100: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_533: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_534: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
        unsqueeze_430: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
        unsqueeze_431: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        mul_535: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_204);  primals_204 = None
        unsqueeze_433: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
        unsqueeze_434: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_536: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_432);  sub_124 = unsqueeze_432 = None
        sub_126: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_312, mul_536);  convert_element_type_312 = mul_536 = None
        sub_127: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_429);  sub_126 = unsqueeze_429 = None
        mul_537: "f32[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_435);  sub_127 = unsqueeze_435 = None
        mul_538: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_100);  sum_39 = squeeze_100 = None
        convert_element_type_314: "bf16[96, 576, 14, 14][112896, 1, 8064, 576]cuda:0" = torch.ops.prims.convert_element_type.default(mul_537, torch.bfloat16);  mul_537 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_314, convert_element_type_143, convert_element_type_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_314 = convert_element_type_143 = convert_element_type_144 = None
        getitem_158: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = convolution_backward_18[0]
        getitem_159: "bf16[576, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        add_273: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.add.Tensor(add_272, getitem_158);  add_272 = getitem_158 = None
        convert_element_type_315: "f32[576, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:64 in forward, code: return self.conv(x)
        convert_element_type_316: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_273, torch.float32);  add_273 = None
        sum_40: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_316, [0, 2, 3])
        convert_element_type_142: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_128: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_142, unsqueeze_438);  convert_element_type_142 = unsqueeze_438 = None
        mul_539: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_316, sub_128)
        sum_41: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 2, 3]);  mul_539 = None
        mul_540: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 5.314625850340136e-05)
        unsqueeze_439: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
        unsqueeze_440: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_541: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 5.314625850340136e-05)
        mul_542: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_543: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_541, mul_542);  mul_541 = mul_542 = None
        unsqueeze_442: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
        unsqueeze_443: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        mul_544: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  primals_198 = None
        unsqueeze_445: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
        unsqueeze_446: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_545: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_444);  sub_128 = unsqueeze_444 = None
        sub_130: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_316, mul_545);  convert_element_type_316 = mul_545 = None
        sub_131: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_441);  sub_130 = unsqueeze_441 = None
        mul_546: "f32[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_447);  sub_131 = unsqueeze_447 = None
        mul_547: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_97);  sum_41 = squeeze_97 = None
        convert_element_type_318: "bf16[96, 96, 14, 14][18816, 1, 1344, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_546, torch.bfloat16);  mul_546 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_318, convert_element_type_140, convert_element_type_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_318 = convert_element_type_140 = convert_element_type_141 = None
        getitem_161: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_19[0]
        getitem_162: "bf16[96, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_319: "f32[96, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_162, torch.float32);  getitem_162 = None
        sub_31: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, getitem_63)
        mul_217: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        unsqueeze_124: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
        unsqueeze_127: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_165: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        convert_element_type_138: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16);  add_165 = None
        le_13: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_138, 0.0)
        ge_13: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_138, 6.0);  convert_element_type_138 = None
        bitwise_or_13: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_13, ge_13);  le_13 = ge_13 = None
        where_13: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(bitwise_or_13, full_default, getitem_161);  bitwise_or_13 = getitem_161 = None
        convert_element_type_320: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        squeeze_93: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        unsqueeze_448: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_449: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
        unsqueeze_450: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
        sum_42: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_320, [0, 2, 3])
        convert_element_type_137: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sub_132: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_137, unsqueeze_450);  convert_element_type_137 = unsqueeze_450 = None
        mul_548: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_320, sub_132)
        sum_43: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3]);  mul_548 = None
        mul_549: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 5.314625850340136e-05)
        unsqueeze_451: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
        unsqueeze_452: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_550: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 5.314625850340136e-05)
        squeeze_94: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_551: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_552: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
        unsqueeze_454: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
        unsqueeze_455: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
        unsqueeze_456: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
        mul_553: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_192);  primals_192 = None
        unsqueeze_457: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
        unsqueeze_458: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_554: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_456);  sub_132 = unsqueeze_456 = None
        sub_134: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_320, mul_554);  convert_element_type_320 = mul_554 = None
        sub_135: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_453);  sub_134 = unsqueeze_453 = None
        mul_555: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_459);  sub_135 = unsqueeze_459 = None
        mul_556: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_94);  sum_43 = squeeze_94 = None
        convert_element_type_322: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_555, torch.bfloat16);  mul_555 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_322, convert_element_type_135, convert_element_type_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  convert_element_type_322 = convert_element_type_135 = convert_element_type_136 = None
        getitem_164: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_20[0]
        getitem_165: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_323: "f32[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None
        sub_30: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_61)
        mul_210: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        unsqueeze_120: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_121: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_216: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        unsqueeze_122: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_187, -1);  primals_187 = None
        unsqueeze_123: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_160: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        convert_element_type_133: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16);  add_160 = None
        le_14: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_133, 0.0)
        ge_14: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_133, 6.0);  convert_element_type_133 = None
        bitwise_or_14: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_14, ge_14);  le_14 = ge_14 = None
        where_14: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(bitwise_or_14, full_default, getitem_164);  bitwise_or_14 = getitem_164 = None
        convert_element_type_324: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        squeeze_90: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        unsqueeze_460: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_461: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
        unsqueeze_462: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
        sum_44: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_324, [0, 2, 3])
        convert_element_type_132: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_136: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_132, unsqueeze_462);  convert_element_type_132 = unsqueeze_462 = None
        mul_557: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_324, sub_136)
        sum_45: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 2, 3]);  mul_557 = None
        mul_558: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 5.314625850340136e-05)
        unsqueeze_463: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
        unsqueeze_464: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_559: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 5.314625850340136e-05)
        squeeze_91: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_560: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_561: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_559, mul_560);  mul_559 = mul_560 = None
        unsqueeze_466: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
        unsqueeze_467: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        mul_562: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_186);  primals_186 = None
        unsqueeze_469: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
        unsqueeze_470: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_563: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_468);  sub_136 = unsqueeze_468 = None
        sub_138: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_324, mul_563);  convert_element_type_324 = mul_563 = None
        sub_139: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_465);  sub_138 = unsqueeze_465 = None
        mul_564: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_471);  sub_139 = unsqueeze_471 = None
        mul_565: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_91);  sum_45 = squeeze_91 = None
        convert_element_type_326: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_564, torch.bfloat16);  mul_564 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_326, add_155, convert_element_type_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_326 = add_155 = convert_element_type_131 = None
        getitem_167: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = convolution_backward_21[0]
        getitem_168: "bf16[384, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_327: "f32[384, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_168, torch.float32);  getitem_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:62 in forward, code: return x + self.conv(x)
        convert_element_type_328: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_167, torch.float32)
        sum_46: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_328, [0, 2, 3])
        convert_element_type_129: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_140: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_129, unsqueeze_474);  convert_element_type_129 = unsqueeze_474 = None
        mul_566: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_328, sub_140)
        sum_47: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 2, 3]);  mul_566 = None
        mul_567: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 5.314625850340136e-05)
        unsqueeze_475: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
        unsqueeze_476: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_568: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 5.314625850340136e-05)
        mul_569: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_570: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
        unsqueeze_478: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
        unsqueeze_479: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        mul_571: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_180);  primals_180 = None
        unsqueeze_481: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
        unsqueeze_482: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_572: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_480);  sub_140 = unsqueeze_480 = None
        sub_142: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_328, mul_572);  convert_element_type_328 = mul_572 = None
        sub_143: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_477);  sub_142 = unsqueeze_477 = None
        mul_573: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_483);  sub_143 = unsqueeze_483 = None
        mul_574: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_88);  sum_47 = squeeze_88 = None
        convert_element_type_330: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_573, torch.bfloat16);  mul_573 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_330, convert_element_type_127, convert_element_type_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_330 = convert_element_type_127 = convert_element_type_128 = None
        getitem_170: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_22[0]
        getitem_171: "bf16[64, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_331: "f32[64, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None
        sub_28: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_57)
        mul_196: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        unsqueeze_112: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_113: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_115: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_149: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        convert_element_type_125: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None
        le_15: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_125, 0.0)
        ge_15: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_125, 6.0);  convert_element_type_125 = None
        bitwise_or_15: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_15, ge_15);  le_15 = ge_15 = None
        where_15: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(bitwise_or_15, full_default, getitem_170);  bitwise_or_15 = getitem_170 = None
        convert_element_type_332: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        squeeze_84: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        unsqueeze_484: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_485: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        sum_48: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_332, [0, 2, 3])
        convert_element_type_124: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_144: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_124, unsqueeze_486);  convert_element_type_124 = unsqueeze_486 = None
        mul_575: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_332, sub_144)
        sum_49: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 2, 3]);  mul_575 = None
        mul_576: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 5.314625850340136e-05)
        unsqueeze_487: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
        unsqueeze_488: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_577: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 5.314625850340136e-05)
        squeeze_85: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_578: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_579: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_577, mul_578);  mul_577 = mul_578 = None
        unsqueeze_490: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
        unsqueeze_491: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_580: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_174);  primals_174 = None
        unsqueeze_493: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
        unsqueeze_494: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_581: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_492);  sub_144 = unsqueeze_492 = None
        sub_146: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_332, mul_581);  convert_element_type_332 = mul_581 = None
        sub_147: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_489);  sub_146 = unsqueeze_489 = None
        mul_582: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_495);  sub_147 = unsqueeze_495 = None
        mul_583: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None
        convert_element_type_334: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_582, torch.bfloat16);  mul_582 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_334, convert_element_type_122, convert_element_type_123, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  convert_element_type_334 = convert_element_type_122 = convert_element_type_123 = None
        getitem_173: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_23[0]
        getitem_174: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_335: "f32[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_174, torch.float32);  getitem_174 = None
        sub_27: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, getitem_55)
        mul_189: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        unsqueeze_108: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_109: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_195: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        unsqueeze_110: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
        unsqueeze_111: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_144: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
        convert_element_type_120: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None
        le_16: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_120, 0.0)
        ge_16: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_120, 6.0);  convert_element_type_120 = None
        bitwise_or_16: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_16, ge_16);  le_16 = ge_16 = None
        where_16: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(bitwise_or_16, full_default, getitem_173);  bitwise_or_16 = getitem_173 = None
        convert_element_type_336: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        squeeze_81: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        unsqueeze_496: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_497: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
        sum_50: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_336, [0, 2, 3])
        convert_element_type_119: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sub_148: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_119, unsqueeze_498);  convert_element_type_119 = unsqueeze_498 = None
        mul_584: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_336, sub_148)
        sum_51: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
        mul_585: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 5.314625850340136e-05)
        unsqueeze_499: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
        unsqueeze_500: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_586: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 5.314625850340136e-05)
        squeeze_82: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_587: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_588: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
        unsqueeze_502: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
        unsqueeze_503: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_589: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  primals_168 = None
        unsqueeze_505: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
        unsqueeze_506: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_590: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_504);  sub_148 = unsqueeze_504 = None
        sub_150: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_336, mul_590);  convert_element_type_336 = mul_590 = None
        sub_151: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_501);  sub_150 = unsqueeze_501 = None
        mul_591: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_507);  sub_151 = unsqueeze_507 = None
        mul_592: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None
        convert_element_type_338: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_591, torch.bfloat16);  mul_591 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_338, add_139, convert_element_type_118, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_338 = add_139 = convert_element_type_118 = None
        getitem_176: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = convolution_backward_24[0]
        getitem_177: "bf16[384, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        add_274: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.add.Tensor(getitem_167, getitem_176);  getitem_167 = getitem_176 = None
        convert_element_type_339: "f32[384, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None
        convert_element_type_340: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_274, torch.float32)
        sum_52: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_340, [0, 2, 3])
        convert_element_type_116: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_152: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_116, unsqueeze_510);  convert_element_type_116 = unsqueeze_510 = None
        mul_593: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_340, sub_152)
        sum_53: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 2, 3]);  mul_593 = None
        mul_594: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 5.314625850340136e-05)
        unsqueeze_511: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
        unsqueeze_512: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_595: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 5.314625850340136e-05)
        mul_596: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_597: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_595, mul_596);  mul_595 = mul_596 = None
        unsqueeze_514: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
        unsqueeze_515: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_598: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_162);  primals_162 = None
        unsqueeze_517: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
        unsqueeze_518: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_599: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_516);  sub_152 = unsqueeze_516 = None
        sub_154: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_340, mul_599);  convert_element_type_340 = mul_599 = None
        sub_155: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_513);  sub_154 = unsqueeze_513 = None
        mul_600: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_519);  sub_155 = unsqueeze_519 = None
        mul_601: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None
        convert_element_type_342: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_600, torch.bfloat16);  mul_600 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_342, convert_element_type_114, convert_element_type_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_342 = convert_element_type_114 = convert_element_type_115 = None
        getitem_179: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_25[0]
        getitem_180: "bf16[64, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_343: "f32[64, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_180, torch.float32);  getitem_180 = None
        sub_25: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_51)
        mul_175: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_100: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_103: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_133: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        convert_element_type_112: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None
        le_17: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_112, 0.0)
        ge_17: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_112, 6.0);  convert_element_type_112 = None
        bitwise_or_17: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_17, ge_17);  le_17 = ge_17 = None
        where_17: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(bitwise_or_17, full_default, getitem_179);  bitwise_or_17 = getitem_179 = None
        convert_element_type_344: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        squeeze_75: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        unsqueeze_520: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_521: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
        sum_54: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_344, [0, 2, 3])
        convert_element_type_111: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sub_156: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_111, unsqueeze_522);  convert_element_type_111 = unsqueeze_522 = None
        mul_602: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_344, sub_156)
        sum_55: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 2, 3]);  mul_602 = None
        mul_603: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 5.314625850340136e-05)
        unsqueeze_523: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
        unsqueeze_524: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_604: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 5.314625850340136e-05)
        squeeze_76: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_605: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_606: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
        unsqueeze_526: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
        unsqueeze_527: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_607: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_156);  primals_156 = None
        unsqueeze_529: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
        unsqueeze_530: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_608: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_528);  sub_156 = unsqueeze_528 = None
        sub_158: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_344, mul_608);  convert_element_type_344 = mul_608 = None
        sub_159: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_525);  sub_158 = unsqueeze_525 = None
        mul_609: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_531);  sub_159 = unsqueeze_531 = None
        mul_610: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None
        convert_element_type_346: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_609, torch.bfloat16);  mul_609 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_346, convert_element_type_109, convert_element_type_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  convert_element_type_346 = convert_element_type_109 = convert_element_type_110 = None
        getitem_182: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_26[0]
        getitem_183: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_347: "f32[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_183, torch.float32);  getitem_183 = None
        sub_24: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_49)
        mul_168: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        unsqueeze_96: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_97: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_174: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        unsqueeze_98: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_99: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_128: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        convert_element_type_107: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16);  add_128 = None
        le_18: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_107, 0.0)
        ge_18: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_107, 6.0);  convert_element_type_107 = None
        bitwise_or_18: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_18, ge_18);  le_18 = ge_18 = None
        where_18: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(bitwise_or_18, full_default, getitem_182);  bitwise_or_18 = getitem_182 = None
        convert_element_type_348: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32);  where_18 = None
        squeeze_72: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        unsqueeze_532: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_533: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
        unsqueeze_534: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
        sum_56: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_348, [0, 2, 3])
        convert_element_type_106: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_160: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_106, unsqueeze_534);  convert_element_type_106 = unsqueeze_534 = None
        mul_611: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_348, sub_160)
        sum_57: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 2, 3]);  mul_611 = None
        mul_612: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 5.314625850340136e-05)
        unsqueeze_535: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_536: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_613: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 5.314625850340136e-05)
        squeeze_73: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_614: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_615: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
        unsqueeze_538: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
        unsqueeze_539: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_616: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  primals_150 = None
        unsqueeze_541: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
        unsqueeze_542: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_617: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_540);  sub_160 = unsqueeze_540 = None
        sub_162: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_348, mul_617);  convert_element_type_348 = mul_617 = None
        sub_163: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_537);  sub_162 = unsqueeze_537 = None
        mul_618: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_543);  sub_163 = unsqueeze_543 = None
        mul_619: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None
        convert_element_type_350: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_618, torch.bfloat16);  mul_618 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_350, add_123, convert_element_type_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_350 = add_123 = convert_element_type_105 = None
        getitem_185: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = convolution_backward_27[0]
        getitem_186: "bf16[384, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        add_275: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.add.Tensor(add_274, getitem_185);  add_274 = getitem_185 = None
        convert_element_type_351: "f32[384, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_186, torch.float32);  getitem_186 = None
        convert_element_type_352: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_275, torch.float32)
        sum_58: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_352, [0, 2, 3])
        convert_element_type_103: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32);  convolution_23 = None
        sub_164: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_103, unsqueeze_546);  convert_element_type_103 = unsqueeze_546 = None
        mul_620: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_352, sub_164)
        sum_59: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3]);  mul_620 = None
        mul_621: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 5.314625850340136e-05)
        unsqueeze_547: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
        unsqueeze_548: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_622: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 5.314625850340136e-05)
        mul_623: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_624: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
        unsqueeze_550: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
        unsqueeze_551: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        mul_625: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_144);  primals_144 = None
        unsqueeze_553: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
        unsqueeze_554: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_626: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_552);  sub_164 = unsqueeze_552 = None
        sub_166: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_352, mul_626);  convert_element_type_352 = mul_626 = None
        sub_167: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_549);  sub_166 = unsqueeze_549 = None
        mul_627: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_555);  sub_167 = unsqueeze_555 = None
        mul_628: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None
        convert_element_type_354: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_627, torch.bfloat16);  mul_627 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(convert_element_type_354, convert_element_type_101, convert_element_type_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_354 = convert_element_type_101 = convert_element_type_102 = None
        getitem_188: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_28[0]
        getitem_189: "bf16[64, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_355: "f32[64, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_189, torch.float32);  getitem_189 = None
        sub_22: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, getitem_45)
        mul_154: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        unsqueeze_88: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_89: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_91: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_117: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        convert_element_type_99: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16);  add_117 = None
        le_19: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_99, 0.0)
        ge_19: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_99, 6.0);  convert_element_type_99 = None
        bitwise_or_19: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_19, ge_19);  le_19 = ge_19 = None
        where_19: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(bitwise_or_19, full_default, getitem_188);  bitwise_or_19 = getitem_188 = None
        convert_element_type_356: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        squeeze_66: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        unsqueeze_556: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_557: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
        unsqueeze_558: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
        sum_60: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_356, [0, 2, 3])
        convert_element_type_98: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_168: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_98, unsqueeze_558);  convert_element_type_98 = unsqueeze_558 = None
        mul_629: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_356, sub_168)
        sum_61: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 2, 3]);  mul_629 = None
        mul_630: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 5.314625850340136e-05)
        unsqueeze_559: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
        unsqueeze_560: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_631: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 5.314625850340136e-05)
        squeeze_67: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_632: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_633: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
        unsqueeze_562: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
        unsqueeze_563: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        mul_634: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_138);  primals_138 = None
        unsqueeze_565: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_634, 0);  mul_634 = None
        unsqueeze_566: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_635: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_564);  sub_168 = unsqueeze_564 = None
        sub_170: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_356, mul_635);  convert_element_type_356 = mul_635 = None
        sub_171: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_561);  sub_170 = unsqueeze_561 = None
        mul_636: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_567);  sub_171 = unsqueeze_567 = None
        mul_637: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_67);  sum_61 = squeeze_67 = None
        convert_element_type_358: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_636, torch.bfloat16);  mul_636 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_358, convert_element_type_96, convert_element_type_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  convert_element_type_358 = convert_element_type_96 = convert_element_type_97 = None
        getitem_191: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_29[0]
        getitem_192: "bf16[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_359: "f32[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_192, torch.float32);  getitem_192 = None
        sub_21: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_43)
        mul_147: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        unsqueeze_84: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_85: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_153: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        unsqueeze_86: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_87: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_112: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        convert_element_type_94: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None
        le_20: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_94, 0.0)
        ge_20: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_94, 6.0);  convert_element_type_94 = None
        bitwise_or_20: "b8[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_20, ge_20);  le_20 = ge_20 = None
        where_20: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(bitwise_or_20, full_default, getitem_191);  bitwise_or_20 = getitem_191 = None
        convert_element_type_360: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        squeeze_63: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        unsqueeze_568: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_569: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
        unsqueeze_570: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
        sum_62: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_360, [0, 2, 3])
        convert_element_type_93: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32);  convolution_21 = None
        sub_172: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_93, unsqueeze_570);  convert_element_type_93 = unsqueeze_570 = None
        mul_638: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_360, sub_172)
        sum_63: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_638, [0, 2, 3]);  mul_638 = None
        mul_639: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 5.314625850340136e-05)
        unsqueeze_571: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
        unsqueeze_572: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_640: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 5.314625850340136e-05)
        squeeze_64: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_641: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_642: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_640, mul_641);  mul_640 = mul_641 = None
        unsqueeze_574: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
        unsqueeze_575: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        mul_643: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_132);  primals_132 = None
        unsqueeze_577: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
        unsqueeze_578: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_644: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_576);  sub_172 = unsqueeze_576 = None
        sub_174: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_360, mul_644);  convert_element_type_360 = mul_644 = None
        sub_175: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_573);  sub_174 = unsqueeze_573 = None
        mul_645: "f32[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_579);  sub_175 = unsqueeze_579 = None
        mul_646: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_64);  sum_63 = squeeze_64 = None
        convert_element_type_362: "bf16[96, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_645, torch.bfloat16);  mul_645 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_362, convert_element_type_91, convert_element_type_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_362 = convert_element_type_91 = convert_element_type_92 = None
        getitem_194: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = convolution_backward_30[0]
        getitem_195: "bf16[384, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        add_276: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.add.Tensor(add_275, getitem_194);  add_275 = getitem_194 = None
        convert_element_type_363: "f32[384, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_195, torch.float32);  getitem_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:64 in forward, code: return self.conv(x)
        convert_element_type_364: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_276, torch.float32);  add_276 = None
        sum_64: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_364, [0, 2, 3])
        convert_element_type_90: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_176: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_90, unsqueeze_582);  convert_element_type_90 = unsqueeze_582 = None
        mul_647: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_364, sub_176)
        sum_65: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 2, 3]);  mul_647 = None
        mul_648: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 5.314625850340136e-05)
        unsqueeze_583: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
        unsqueeze_584: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_649: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 5.314625850340136e-05)
        mul_650: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_651: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_649, mul_650);  mul_649 = mul_650 = None
        unsqueeze_586: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
        unsqueeze_587: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        mul_652: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_589: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
        unsqueeze_590: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_653: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_588);  sub_176 = unsqueeze_588 = None
        sub_178: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_364, mul_653);  convert_element_type_364 = mul_653 = None
        sub_179: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_585);  sub_178 = unsqueeze_585 = None
        mul_654: "f32[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_591);  sub_179 = unsqueeze_591 = None
        mul_655: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_61);  sum_65 = squeeze_61 = None
        convert_element_type_366: "bf16[96, 64, 14, 14][12544, 1, 896, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_654, torch.bfloat16);  mul_654 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_366, convert_element_type_88, convert_element_type_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_366 = convert_element_type_88 = convert_element_type_89 = None
        getitem_197: "bf16[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = convolution_backward_31[0]
        getitem_198: "bf16[64, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_367: "f32[64, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_198, torch.float32);  getitem_198 = None
        sub_19: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_39)
        mul_133: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        unsqueeze_76: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_102: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        convert_element_type_86: "bf16[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None
        le_21: "b8[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_86, 0.0)
        ge_21: "b8[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_86, 6.0);  convert_element_type_86 = None
        bitwise_or_21: "b8[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_21, ge_21);  le_21 = ge_21 = None
        where_21: "bf16[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.where.self(bitwise_or_21, full_default, getitem_197);  bitwise_or_21 = getitem_197 = None
        convert_element_type_368: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        squeeze_57: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        unsqueeze_592: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_593: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
        unsqueeze_594: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
        sum_66: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_368, [0, 2, 3])
        convert_element_type_85: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_180: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_85, unsqueeze_594);  convert_element_type_85 = unsqueeze_594 = None
        mul_656: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_368, sub_180)
        sum_67: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_656, [0, 2, 3]);  mul_656 = None
        mul_657: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 5.314625850340136e-05)
        unsqueeze_595: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
        unsqueeze_596: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_658: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 5.314625850340136e-05)
        squeeze_58: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_659: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_660: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_658, mul_659);  mul_658 = mul_659 = None
        unsqueeze_598: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
        unsqueeze_599: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        mul_661: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_601: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
        unsqueeze_602: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_662: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_600);  sub_180 = unsqueeze_600 = None
        sub_182: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_368, mul_662);  convert_element_type_368 = mul_662 = None
        sub_183: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_597);  sub_182 = unsqueeze_597 = None
        mul_663: "f32[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_603);  sub_183 = unsqueeze_603 = None
        mul_664: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_58);  sum_67 = squeeze_58 = None
        convert_element_type_370: "bf16[96, 192, 14, 14][37632, 1, 2688, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_663, torch.bfloat16);  mul_663 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_370, convert_element_type_83, convert_element_type_84, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  convert_element_type_370 = convert_element_type_83 = convert_element_type_84 = None
        getitem_200: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_32[0]
        getitem_201: "bf16[192, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_371: "f32[192, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None
        sub_18: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_37)
        mul_126: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        unsqueeze_72: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_97: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        convert_element_type_81: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16);  add_97 = None
        le_22: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_81, 0.0)
        ge_22: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_81, 6.0);  convert_element_type_81 = None
        bitwise_or_22: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_22, ge_22);  le_22 = ge_22 = None
        where_22: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.where.self(bitwise_or_22, full_default, getitem_200);  bitwise_or_22 = getitem_200 = None
        convert_element_type_372: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        squeeze_54: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        unsqueeze_604: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_605: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
        unsqueeze_606: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
        sum_68: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_372, [0, 2, 3])
        convert_element_type_80: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_184: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, unsqueeze_606);  convert_element_type_80 = unsqueeze_606 = None
        mul_665: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_372, sub_184)
        sum_69: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_665, [0, 2, 3]);  mul_665 = None
        mul_666: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 1.328656462585034e-05)
        unsqueeze_607: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_608: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_667: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 1.328656462585034e-05)
        squeeze_55: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_668: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_669: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_667, mul_668);  mul_667 = mul_668 = None
        unsqueeze_610: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
        unsqueeze_611: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        mul_670: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_613: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_670, 0);  mul_670 = None
        unsqueeze_614: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_671: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_612);  sub_184 = unsqueeze_612 = None
        sub_186: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_372, mul_671);  convert_element_type_372 = mul_671 = None
        sub_187: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_609);  sub_186 = unsqueeze_609 = None
        mul_672: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_615);  sub_187 = unsqueeze_615 = None
        mul_673: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_55);  sum_69 = squeeze_55 = None
        convert_element_type_374: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_672, torch.bfloat16);  mul_672 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_374, add_92, convert_element_type_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_374 = add_92 = convert_element_type_79 = None
        getitem_203: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = convolution_backward_33[0]
        getitem_204: "bf16[192, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_375: "f32[192, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_204, torch.float32);  getitem_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:62 in forward, code: return x + self.conv(x)
        convert_element_type_376: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_203, torch.float32)
        sum_70: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_376, [0, 2, 3])
        convert_element_type_77: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        sub_188: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_77, unsqueeze_618);  convert_element_type_77 = unsqueeze_618 = None
        mul_674: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_376, sub_188)
        sum_71: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_674, [0, 2, 3]);  mul_674 = None
        mul_675: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 1.328656462585034e-05)
        unsqueeze_619: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_620: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_676: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 1.328656462585034e-05)
        mul_677: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_678: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
        unsqueeze_622: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
        unsqueeze_623: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        mul_679: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_625: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_679, 0);  mul_679 = None
        unsqueeze_626: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_680: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_624);  sub_188 = unsqueeze_624 = None
        sub_190: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_376, mul_680);  convert_element_type_376 = mul_680 = None
        sub_191: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_621);  sub_190 = unsqueeze_621 = None
        mul_681: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_627);  sub_191 = unsqueeze_627 = None
        mul_682: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_52);  sum_71 = squeeze_52 = None
        convert_element_type_378: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_681, torch.bfloat16);  mul_681 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_378, convert_element_type_75, convert_element_type_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_378 = convert_element_type_75 = convert_element_type_76 = None
        getitem_206: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_34[0]
        getitem_207: "bf16[32, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_379: "f32[32, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None
        sub_16: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
        mul_112: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        unsqueeze_64: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_86: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_73: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16);  add_86 = None
        le_23: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_73, 0.0)
        ge_23: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_73, 6.0);  convert_element_type_73 = None
        bitwise_or_23: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_23, ge_23);  le_23 = ge_23 = None
        where_23: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.where.self(bitwise_or_23, full_default, getitem_206);  bitwise_or_23 = getitem_206 = None
        convert_element_type_380: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        squeeze_48: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        unsqueeze_628: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_629: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
        unsqueeze_630: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
        sum_72: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_380, [0, 2, 3])
        convert_element_type_72: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_192: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, unsqueeze_630);  convert_element_type_72 = unsqueeze_630 = None
        mul_683: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, sub_192)
        sum_73: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 2, 3]);  mul_683 = None
        mul_684: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 1.328656462585034e-05)
        unsqueeze_631: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_632: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_685: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 1.328656462585034e-05)
        squeeze_49: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_686: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_687: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_685, mul_686);  mul_685 = mul_686 = None
        unsqueeze_634: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
        unsqueeze_635: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        mul_688: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_637: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
        unsqueeze_638: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_689: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_636);  sub_192 = unsqueeze_636 = None
        sub_194: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_380, mul_689);  convert_element_type_380 = mul_689 = None
        sub_195: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_633);  sub_194 = unsqueeze_633 = None
        mul_690: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_639);  sub_195 = unsqueeze_639 = None
        mul_691: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_49);  sum_73 = squeeze_49 = None
        convert_element_type_382: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_690, torch.bfloat16);  mul_690 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_382, convert_element_type_70, convert_element_type_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  convert_element_type_382 = convert_element_type_70 = convert_element_type_71 = None
        getitem_209: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_35[0]
        getitem_210: "bf16[192, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_383: "f32[192, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_210, torch.float32);  getitem_210 = None
        sub_15: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_31)
        mul_105: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        unsqueeze_60: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_81: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        convert_element_type_68: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None
        le_24: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_68, 0.0)
        ge_24: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_68, 6.0);  convert_element_type_68 = None
        bitwise_or_24: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_24, ge_24);  le_24 = ge_24 = None
        where_24: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.where.self(bitwise_or_24, full_default, getitem_209);  bitwise_or_24 = getitem_209 = None
        convert_element_type_384: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_24, torch.float32);  where_24 = None
        squeeze_45: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        unsqueeze_640: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_641: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
        unsqueeze_642: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
        sum_74: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_384, [0, 2, 3])
        convert_element_type_67: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_196: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_67, unsqueeze_642);  convert_element_type_67 = unsqueeze_642 = None
        mul_692: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_384, sub_196)
        sum_75: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 2, 3]);  mul_692 = None
        mul_693: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 1.328656462585034e-05)
        unsqueeze_643: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
        unsqueeze_644: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_694: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 1.328656462585034e-05)
        squeeze_46: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_695: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_696: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
        unsqueeze_646: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
        unsqueeze_647: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
        mul_697: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_649: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
        unsqueeze_650: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_698: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_648);  sub_196 = unsqueeze_648 = None
        sub_198: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_384, mul_698);  convert_element_type_384 = mul_698 = None
        sub_199: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_645);  sub_198 = unsqueeze_645 = None
        mul_699: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_651);  sub_199 = unsqueeze_651 = None
        mul_700: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_46);  sum_75 = squeeze_46 = None
        convert_element_type_386: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_699, torch.bfloat16);  mul_699 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_386, add_76, convert_element_type_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_386 = add_76 = convert_element_type_66 = None
        getitem_212: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = convolution_backward_36[0]
        getitem_213: "bf16[192, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        add_277: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.add.Tensor(getitem_203, getitem_212);  getitem_203 = getitem_212 = None
        convert_element_type_387: "f32[192, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_213, torch.float32);  getitem_213 = None
        convert_element_type_388: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_277, torch.float32)
        sum_76: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_388, [0, 2, 3])
        convert_element_type_64: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_200: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_64, unsqueeze_654);  convert_element_type_64 = unsqueeze_654 = None
        mul_701: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_388, sub_200)
        sum_77: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_701, [0, 2, 3]);  mul_701 = None
        mul_702: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 1.328656462585034e-05)
        unsqueeze_655: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
        unsqueeze_656: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_703: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 1.328656462585034e-05)
        mul_704: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_705: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_703, mul_704);  mul_703 = mul_704 = None
        unsqueeze_658: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
        unsqueeze_659: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        mul_706: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_661: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_706, 0);  mul_706 = None
        unsqueeze_662: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_707: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_660);  sub_200 = unsqueeze_660 = None
        sub_202: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_388, mul_707);  convert_element_type_388 = mul_707 = None
        sub_203: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_657);  sub_202 = unsqueeze_657 = None
        mul_708: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_663);  sub_203 = unsqueeze_663 = None
        mul_709: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_43);  sum_77 = squeeze_43 = None
        convert_element_type_390: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_708, torch.bfloat16);  mul_708 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_390, convert_element_type_62, convert_element_type_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_390 = convert_element_type_62 = convert_element_type_63 = None
        getitem_215: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_37[0]
        getitem_216: "bf16[32, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_391: "f32[32, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_216, torch.float32);  getitem_216 = None
        sub_13: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
        mul_91: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        unsqueeze_52: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_70: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convert_element_type_60: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None
        le_25: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_60, 0.0)
        ge_25: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_60, 6.0);  convert_element_type_60 = None
        bitwise_or_25: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_25, ge_25);  le_25 = ge_25 = None
        where_25: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.where.self(bitwise_or_25, full_default, getitem_215);  bitwise_or_25 = getitem_215 = None
        convert_element_type_392: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_25, torch.float32);  where_25 = None
        squeeze_39: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_664: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_665: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
        unsqueeze_666: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
        sum_78: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_392, [0, 2, 3])
        convert_element_type_59: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sub_204: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_59, unsqueeze_666);  convert_element_type_59 = unsqueeze_666 = None
        mul_710: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_392, sub_204)
        sum_79: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 2, 3]);  mul_710 = None
        mul_711: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 1.328656462585034e-05)
        unsqueeze_667: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_668: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_712: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 1.328656462585034e-05)
        squeeze_40: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_713: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_714: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_712, mul_713);  mul_712 = mul_713 = None
        unsqueeze_670: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
        unsqueeze_671: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        mul_715: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_673: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
        unsqueeze_674: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_716: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_672);  sub_204 = unsqueeze_672 = None
        sub_206: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_392, mul_716);  convert_element_type_392 = mul_716 = None
        sub_207: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_669);  sub_206 = unsqueeze_669 = None
        mul_717: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_675);  sub_207 = unsqueeze_675 = None
        mul_718: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_40);  sum_79 = squeeze_40 = None
        convert_element_type_394: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_717, torch.bfloat16);  mul_717 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_394, convert_element_type_57, convert_element_type_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  convert_element_type_394 = convert_element_type_57 = convert_element_type_58 = None
        getitem_218: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_38[0]
        getitem_219: "bf16[192, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_395: "f32[192, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_219, torch.float32);  getitem_219 = None
        sub_12: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, getitem_25)
        mul_84: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        unsqueeze_48: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_65: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        convert_element_type_55: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16);  add_65 = None
        le_26: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_55, 0.0)
        ge_26: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_55, 6.0);  convert_element_type_55 = None
        bitwise_or_26: "b8[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_26, ge_26);  le_26 = ge_26 = None
        where_26: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.where.self(bitwise_or_26, full_default, getitem_218);  bitwise_or_26 = getitem_218 = None
        convert_element_type_396: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_26, torch.float32);  where_26 = None
        squeeze_36: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        unsqueeze_676: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_677: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
        unsqueeze_678: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
        sum_80: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_396, [0, 2, 3])
        convert_element_type_54: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sub_208: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_54, unsqueeze_678);  convert_element_type_54 = unsqueeze_678 = None
        mul_719: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_396, sub_208)
        sum_81: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 2, 3]);  mul_719 = None
        mul_720: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, 1.328656462585034e-05)
        unsqueeze_679: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_680: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_721: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 1.328656462585034e-05)
        squeeze_37: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_722: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_723: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_721, mul_722);  mul_721 = mul_722 = None
        unsqueeze_682: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
        unsqueeze_683: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        mul_724: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_685: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
        unsqueeze_686: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_725: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_684);  sub_208 = unsqueeze_684 = None
        sub_210: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_396, mul_725);  convert_element_type_396 = mul_725 = None
        sub_211: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_681);  sub_210 = unsqueeze_681 = None
        mul_726: "f32[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_687);  sub_211 = unsqueeze_687 = None
        mul_727: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, squeeze_37);  sum_81 = squeeze_37 = None
        convert_element_type_398: "bf16[96, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_726, torch.bfloat16);  mul_726 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_398, convert_element_type_52, convert_element_type_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_398 = convert_element_type_52 = convert_element_type_53 = None
        getitem_221: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = convolution_backward_39[0]
        getitem_222: "bf16[192, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        add_278: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.add.Tensor(add_277, getitem_221);  add_277 = getitem_221 = None
        convert_element_type_399: "f32[192, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_222, torch.float32);  getitem_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:64 in forward, code: return self.conv(x)
        convert_element_type_400: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_278, torch.float32);  add_278 = None
        sum_82: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_400, [0, 2, 3])
        convert_element_type_51: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_212: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_51, unsqueeze_690);  convert_element_type_51 = unsqueeze_690 = None
        mul_728: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_400, sub_212)
        sum_83: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_728, [0, 2, 3]);  mul_728 = None
        mul_729: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 1.328656462585034e-05)
        unsqueeze_691: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
        unsqueeze_692: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_730: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 1.328656462585034e-05)
        mul_731: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_732: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_730, mul_731);  mul_730 = mul_731 = None
        unsqueeze_694: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
        unsqueeze_695: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        mul_733: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_697: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
        unsqueeze_698: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_734: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_696);  sub_212 = unsqueeze_696 = None
        sub_214: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_400, mul_734);  convert_element_type_400 = mul_734 = None
        sub_215: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_693);  sub_214 = unsqueeze_693 = None
        mul_735: "f32[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_699);  sub_215 = unsqueeze_699 = None
        mul_736: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_34);  sum_83 = squeeze_34 = None
        convert_element_type_402: "bf16[96, 32, 28, 28][25088, 1, 896, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_735, torch.bfloat16);  mul_735 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_402, convert_element_type_49, convert_element_type_50, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_402 = convert_element_type_49 = convert_element_type_50 = None
        getitem_224: "bf16[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = convolution_backward_40[0]
        getitem_225: "bf16[32, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_403: "f32[32, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_225, torch.float32);  getitem_225 = None
        sub_10: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_70: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        unsqueeze_40: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_55: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_47: "bf16[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        le_27: "b8[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_47, 0.0)
        ge_27: "b8[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_47, 6.0);  convert_element_type_47 = None
        bitwise_or_27: "b8[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_27, ge_27);  le_27 = ge_27 = None
        where_27: "bf16[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.where.self(bitwise_or_27, full_default, getitem_224);  bitwise_or_27 = getitem_224 = None
        convert_element_type_404: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(where_27, torch.float32);  where_27 = None
        squeeze_30: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_700: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_701: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
        unsqueeze_702: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
        sum_84: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_404, [0, 2, 3])
        convert_element_type_46: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_216: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_46, unsqueeze_702);  convert_element_type_46 = unsqueeze_702 = None
        mul_737: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_404, sub_216)
        sum_85: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3]);  mul_737 = None
        mul_738: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 1.328656462585034e-05)
        unsqueeze_703: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_704: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_739: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 1.328656462585034e-05)
        squeeze_31: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_740: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_741: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
        unsqueeze_706: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
        unsqueeze_707: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
        mul_742: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_709: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
        unsqueeze_710: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_743: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_708);  sub_216 = unsqueeze_708 = None
        sub_218: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_404, mul_743);  convert_element_type_404 = mul_743 = None
        sub_219: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_705);  sub_218 = unsqueeze_705 = None
        mul_744: "f32[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_711);  sub_219 = unsqueeze_711 = None
        mul_745: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_31);  sum_85 = squeeze_31 = None
        convert_element_type_406: "bf16[96, 144, 28, 28][112896, 1, 4032, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_744, torch.bfloat16);  mul_744 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_406, convert_element_type_44, convert_element_type_45, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  convert_element_type_406 = convert_element_type_44 = convert_element_type_45 = None
        getitem_227: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = convolution_backward_41[0]
        getitem_228: "bf16[144, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_407: "f32[144, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_228, torch.float32);  getitem_228 = None
        sub_9: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_19)
        mul_63: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        unsqueeze_36: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_50: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        convert_element_type_42: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None
        le_28: "b8[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_42, 0.0)
        ge_28: "b8[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_42, 6.0);  convert_element_type_42 = None
        bitwise_or_28: "b8[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_28, ge_28);  le_28 = ge_28 = None
        where_28: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.where.self(bitwise_or_28, full_default, getitem_227);  bitwise_or_28 = getitem_227 = None
        convert_element_type_408: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(where_28, torch.float32);  where_28 = None
        squeeze_27: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        unsqueeze_712: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_713: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
        unsqueeze_714: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
        sum_86: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_408, [0, 2, 3])
        convert_element_type_41: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_220: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_41, unsqueeze_714);  convert_element_type_41 = unsqueeze_714 = None
        mul_746: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_408, sub_220)
        sum_87: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_746, [0, 2, 3]);  mul_746 = None
        mul_747: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 3.321641156462585e-06)
        unsqueeze_715: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
        unsqueeze_716: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_748: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 3.321641156462585e-06)
        squeeze_28: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_749: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_750: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_748, mul_749);  mul_748 = mul_749 = None
        unsqueeze_718: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
        unsqueeze_719: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        mul_751: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_721: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
        unsqueeze_722: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_752: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_720);  sub_220 = unsqueeze_720 = None
        sub_222: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_408, mul_752);  convert_element_type_408 = mul_752 = None
        sub_223: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_717);  sub_222 = unsqueeze_717 = None
        mul_753: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_723);  sub_223 = unsqueeze_723 = None
        mul_754: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_28);  sum_87 = squeeze_28 = None
        convert_element_type_410: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_753, torch.bfloat16);  mul_753 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_410, add_45, convert_element_type_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_410 = add_45 = convert_element_type_40 = None
        getitem_230: "bf16[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_42[0]
        getitem_231: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_411: "f32[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:62 in forward, code: return x + self.conv(x)
        convert_element_type_412: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_230, torch.float32)
        sum_88: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_412, [0, 2, 3])
        convert_element_type_38: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        sub_224: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, unsqueeze_726);  convert_element_type_38 = unsqueeze_726 = None
        mul_755: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_412, sub_224)
        sum_89: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_755, [0, 2, 3]);  mul_755 = None
        mul_756: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 3.321641156462585e-06)
        unsqueeze_727: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
        unsqueeze_728: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_757: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 3.321641156462585e-06)
        mul_758: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_759: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_757, mul_758);  mul_757 = mul_758 = None
        unsqueeze_730: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
        unsqueeze_731: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        mul_760: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_733: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
        unsqueeze_734: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_761: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_732);  sub_224 = unsqueeze_732 = None
        sub_226: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_412, mul_761);  convert_element_type_412 = mul_761 = None
        sub_227: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_729);  sub_226 = unsqueeze_729 = None
        mul_762: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_735);  sub_227 = unsqueeze_735 = None
        mul_763: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_25);  sum_89 = squeeze_25 = None
        convert_element_type_414: "bf16[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(mul_762, torch.bfloat16);  mul_762 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_414, convert_element_type_36, convert_element_type_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_414 = convert_element_type_36 = convert_element_type_37 = None
        getitem_233: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = convolution_backward_43[0]
        getitem_234: "bf16[24, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_415: "f32[24, 144, 1, 1][144, 1, 144, 144]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_234, torch.float32);  getitem_234 = None
        sub_7: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        unsqueeze_28: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_34: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        le_29: "b8[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_34, 0.0)
        ge_29: "b8[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_34, 6.0);  convert_element_type_34 = None
        bitwise_or_29: "b8[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_29, ge_29);  le_29 = ge_29 = None
        where_29: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.where.self(bitwise_or_29, full_default, getitem_233);  bitwise_or_29 = getitem_233 = None
        convert_element_type_416: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(where_29, torch.float32);  where_29 = None
        squeeze_21: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        unsqueeze_736: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_737: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
        unsqueeze_738: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
        sum_90: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_416, [0, 2, 3])
        convert_element_type_33: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_228: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_33, unsqueeze_738);  convert_element_type_33 = unsqueeze_738 = None
        mul_764: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_416, sub_228)
        sum_91: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_764, [0, 2, 3]);  mul_764 = None
        mul_765: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 3.321641156462585e-06)
        unsqueeze_739: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
        unsqueeze_740: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_766: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 3.321641156462585e-06)
        squeeze_22: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_767: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_768: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_766, mul_767);  mul_766 = mul_767 = None
        unsqueeze_742: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
        unsqueeze_743: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
        unsqueeze_744: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
        mul_769: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_745: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_769, 0);  mul_769 = None
        unsqueeze_746: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_770: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_744);  sub_228 = unsqueeze_744 = None
        sub_230: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_416, mul_770);  convert_element_type_416 = mul_770 = None
        sub_231: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_741);  sub_230 = unsqueeze_741 = None
        mul_771: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_747);  sub_231 = unsqueeze_747 = None
        mul_772: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_22);  sum_91 = squeeze_22 = None
        convert_element_type_418: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_771, torch.bfloat16);  mul_771 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_418, convert_element_type_31, convert_element_type_32, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  convert_element_type_418 = convert_element_type_31 = convert_element_type_32 = None
        getitem_236: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = convolution_backward_44[0]
        getitem_237: "bf16[144, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_419: "f32[144, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None
        sub_6: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_13)
        mul_42: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        unsqueeze_24: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[144, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[144, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_34: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        convert_element_type_29: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        le_30: "b8[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_29, 0.0)
        ge_30: "b8[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_29, 6.0);  convert_element_type_29 = None
        bitwise_or_30: "b8[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_30, ge_30);  le_30 = ge_30 = None
        where_30: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.where.self(bitwise_or_30, full_default, getitem_236);  bitwise_or_30 = getitem_236 = None
        convert_element_type_420: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(where_30, torch.float32);  where_30 = None
        squeeze_18: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        unsqueeze_748: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_749: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
        unsqueeze_750: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
        sum_92: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_420, [0, 2, 3])
        convert_element_type_28: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_232: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_28, unsqueeze_750);  convert_element_type_28 = unsqueeze_750 = None
        mul_773: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_420, sub_232)
        sum_93: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_773, [0, 2, 3]);  mul_773 = None
        mul_774: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 3.321641156462585e-06)
        unsqueeze_751: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_752: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_775: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 3.321641156462585e-06)
        squeeze_19: "f32[144][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_776: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_777: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_775, mul_776);  mul_775 = mul_776 = None
        unsqueeze_754: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
        unsqueeze_755: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        mul_778: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_757: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
        unsqueeze_758: "f32[1, 144, 1][144, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 144, 1, 1][144, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_779: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_756);  sub_232 = unsqueeze_756 = None
        sub_234: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_420, mul_779);  convert_element_type_420 = mul_779 = None
        sub_235: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_753);  sub_234 = unsqueeze_753 = None
        mul_780: "f32[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_759);  sub_235 = unsqueeze_759 = None
        mul_781: "f32[144][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_19);  sum_93 = squeeze_19 = None
        convert_element_type_422: "bf16[96, 144, 56, 56][451584, 1, 8064, 144]cuda:0" = torch.ops.prims.convert_element_type.default(mul_780, torch.bfloat16);  mul_780 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(convert_element_type_422, convert_element_type_26, convert_element_type_27, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_422 = convert_element_type_26 = convert_element_type_27 = None
        getitem_239: "bf16[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_45[0]
        getitem_240: "bf16[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        add_279: "bf16[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(getitem_230, getitem_239);  getitem_230 = getitem_239 = None
        convert_element_type_423: "f32[144, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_240, torch.float32);  getitem_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:64 in forward, code: return self.conv(x)
        convert_element_type_424: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.float32);  add_279 = None
        sum_94: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_424, [0, 2, 3])
        convert_element_type_25: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_236: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_25, unsqueeze_762);  convert_element_type_25 = unsqueeze_762 = None
        mul_782: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_424, sub_236)
        sum_95: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 2, 3]);  mul_782 = None
        mul_783: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 3.321641156462585e-06)
        unsqueeze_763: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
        unsqueeze_764: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_784: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 3.321641156462585e-06)
        mul_785: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_786: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_784, mul_785);  mul_784 = mul_785 = None
        unsqueeze_766: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
        unsqueeze_767: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
        unsqueeze_768: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
        mul_787: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_769: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
        unsqueeze_770: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_788: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_768);  sub_236 = unsqueeze_768 = None
        sub_238: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_424, mul_788);  convert_element_type_424 = mul_788 = None
        sub_239: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_765);  sub_238 = unsqueeze_765 = None
        mul_789: "f32[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_771);  sub_239 = unsqueeze_771 = None
        mul_790: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_16);  sum_95 = squeeze_16 = None
        convert_element_type_426: "bf16[96, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims.convert_element_type.default(mul_789, torch.bfloat16);  mul_789 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_426, convert_element_type_23, convert_element_type_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_426 = convert_element_type_23 = convert_element_type_24 = None
        getitem_242: "bf16[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = convolution_backward_46[0]
        getitem_243: "bf16[24, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_427: "f32[24, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_243, torch.float32);  getitem_243 = None
        sub_4: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        unsqueeze_16: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_21: "bf16[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        le_31: "b8[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_21, 0.0)
        ge_31: "b8[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_21, 6.0);  convert_element_type_21 = None
        bitwise_or_31: "b8[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_31, ge_31);  le_31 = ge_31 = None
        where_31: "bf16[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.where.self(bitwise_or_31, full_default, getitem_242);  bitwise_or_31 = getitem_242 = None
        convert_element_type_428: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.float32);  where_31 = None
        squeeze_12: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        unsqueeze_772: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_773: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
        unsqueeze_774: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
        sum_96: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_428, [0, 2, 3])
        convert_element_type_20: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_240: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, unsqueeze_774);  convert_element_type_20 = unsqueeze_774 = None
        mul_791: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, sub_240)
        sum_97: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_791, [0, 2, 3]);  mul_791 = None
        mul_792: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 3.321641156462585e-06)
        unsqueeze_775: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
        unsqueeze_776: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_793: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 3.321641156462585e-06)
        squeeze_13: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_794: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_795: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_793, mul_794);  mul_793 = mul_794 = None
        unsqueeze_778: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
        unsqueeze_779: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
        unsqueeze_780: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
        mul_796: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_781: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
        unsqueeze_782: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_797: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_780);  sub_240 = unsqueeze_780 = None
        sub_242: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_428, mul_797);  convert_element_type_428 = mul_797 = None
        sub_243: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_777);  sub_242 = unsqueeze_777 = None
        mul_798: "f32[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_783);  sub_243 = unsqueeze_783 = None
        mul_799: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_13);  sum_97 = squeeze_13 = None
        convert_element_type_430: "bf16[96, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_798, torch.bfloat16);  mul_798 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(convert_element_type_430, convert_element_type_18, convert_element_type_19, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False]);  convert_element_type_430 = convert_element_type_18 = convert_element_type_19 = None
        getitem_245: "bf16[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = convolution_backward_47[0]
        getitem_246: "bf16[96, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_431: "f32[96, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_246, torch.float32);  getitem_246 = None
        sub_3: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_7)
        mul_21: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        unsqueeze_12: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_16: "bf16[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        le_32: "b8[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_16, 0.0)
        ge_32: "b8[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_16, 6.0);  convert_element_type_16 = None
        bitwise_or_32: "b8[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_32, ge_32);  le_32 = ge_32 = None
        where_32: "bf16[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.where.self(bitwise_or_32, full_default, getitem_245);  bitwise_or_32 = getitem_245 = None
        convert_element_type_432: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_32, torch.float32);  where_32 = None
        squeeze_9: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        unsqueeze_784: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_785: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
        unsqueeze_786: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
        sum_98: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_432, [0, 2, 3])
        convert_element_type_15: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_244: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_15, unsqueeze_786);  convert_element_type_15 = unsqueeze_786 = None
        mul_800: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_432, sub_244)
        sum_99: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_800, [0, 2, 3]);  mul_800 = None
        mul_801: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 8.304102891156462e-07)
        unsqueeze_787: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
        unsqueeze_788: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
        unsqueeze_789: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
        mul_802: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 8.304102891156462e-07)
        squeeze_10: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_803: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_804: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
        unsqueeze_790: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
        unsqueeze_791: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
        unsqueeze_792: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
        mul_805: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_793: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_805, 0);  mul_805 = None
        unsqueeze_794: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
        unsqueeze_795: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
        mul_806: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_792);  sub_244 = unsqueeze_792 = None
        sub_246: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_432, mul_806);  convert_element_type_432 = mul_806 = None
        sub_247: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_789);  sub_246 = unsqueeze_789 = None
        mul_807: "f32[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_795);  sub_247 = unsqueeze_795 = None
        mul_808: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_10);  sum_99 = squeeze_10 = None
        convert_element_type_434: "bf16[96, 96, 112, 112][1204224, 1, 10752, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_807, torch.bfloat16);  mul_807 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_434, convert_element_type_13, convert_element_type_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_434 = convert_element_type_13 = convert_element_type_14 = None
        getitem_248: "bf16[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = convolution_backward_48[0]
        getitem_249: "bf16[96, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_435: "f32[96, 16, 1, 1][16, 1, 16, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_249, torch.float32);  getitem_249 = None
        convert_element_type_436: "f32[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_248, torch.float32);  getitem_248 = None
        sum_100: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_436, [0, 2, 3])
        convert_element_type_12: "f32[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_248: "f32[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_12, unsqueeze_798);  convert_element_type_12 = unsqueeze_798 = None
        mul_809: "f32[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_436, sub_248)
        sum_101: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_809, [0, 2, 3]);  mul_809 = None
        mul_810: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 8.304102891156462e-07)
        unsqueeze_799: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
        unsqueeze_800: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
        unsqueeze_801: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
        mul_811: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 8.304102891156462e-07)
        mul_812: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_813: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_811, mul_812);  mul_811 = mul_812 = None
        unsqueeze_802: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_813, 0);  mul_813 = None
        unsqueeze_803: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
        unsqueeze_804: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
        mul_814: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_805: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_814, 0);  mul_814 = None
        unsqueeze_806: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
        unsqueeze_807: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
        mul_815: "f32[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_804);  sub_248 = unsqueeze_804 = None
        sub_250: "f32[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_436, mul_815);  convert_element_type_436 = mul_815 = None
        sub_251: "f32[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_801);  sub_250 = unsqueeze_801 = None
        mul_816: "f32[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_807);  sub_251 = unsqueeze_807 = None
        mul_817: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_7);  sum_101 = squeeze_7 = None
        convert_element_type_438: "bf16[96, 16, 112, 112][200704, 1, 1792, 16]cuda:0" = torch.ops.prims.convert_element_type.default(mul_816, torch.bfloat16);  mul_816 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(convert_element_type_438, convert_element_type_10, convert_element_type_11, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_438 = convert_element_type_10 = convert_element_type_11 = None
        getitem_251: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = convolution_backward_49[0]
        getitem_252: "bf16[16, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_439: "f32[16, 32, 1, 1][32, 1, 32, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_252, torch.float32);  getitem_252 = None
        sub_1: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        unsqueeze_4: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_8: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        le_33: "b8[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_8, 0.0)
        ge_33: "b8[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_8, 6.0);  convert_element_type_8 = None
        bitwise_or_33: "b8[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_33, ge_33);  le_33 = ge_33 = None
        where_33: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.where.self(bitwise_or_33, full_default, getitem_251);  bitwise_or_33 = getitem_251 = None
        convert_element_type_440: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(where_33, torch.float32);  where_33 = None
        squeeze_3: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        unsqueeze_808: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_809: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
        unsqueeze_810: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
        sum_102: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_440, [0, 2, 3])
        convert_element_type_7: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_252: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_7, unsqueeze_810);  convert_element_type_7 = unsqueeze_810 = None
        mul_818: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_440, sub_252)
        sum_103: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 2, 3]);  mul_818 = None
        mul_819: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 8.304102891156462e-07)
        unsqueeze_811: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
        unsqueeze_812: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
        unsqueeze_813: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
        mul_820: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 8.304102891156462e-07)
        squeeze_4: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_821: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_822: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_820, mul_821);  mul_820 = mul_821 = None
        unsqueeze_814: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
        unsqueeze_815: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
        unsqueeze_816: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
        mul_823: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_817: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_823, 0);  mul_823 = None
        unsqueeze_818: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
        unsqueeze_819: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
        mul_824: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_816);  sub_252 = unsqueeze_816 = None
        sub_254: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_440, mul_824);  convert_element_type_440 = mul_824 = None
        sub_255: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_813);  sub_254 = unsqueeze_813 = None
        mul_825: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_819);  sub_255 = unsqueeze_819 = None
        mul_826: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_4);  sum_103 = squeeze_4 = None
        convert_element_type_442: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_825, torch.bfloat16);  mul_825 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_442, convert_element_type_5, convert_element_type_6, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  convert_element_type_442 = convert_element_type_5 = convert_element_type_6 = None
        getitem_254: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = convolution_backward_50[0]
        getitem_255: "bf16[32, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_443: "f32[32, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_255, torch.float32);  getitem_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mobilenetv2.py:166 in _forward_impl, code: x = self.features(x)
        sub: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        le_34: "b8[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.le.Scalar(convert_element_type_3, 0.0)
        ge_34: "b8[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.ge.Scalar(convert_element_type_3, 6.0);  convert_element_type_3 = None
        bitwise_or_34: "b8[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.bitwise_or.Tensor(le_34, ge_34);  le_34 = ge_34 = None
        where_34: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.where.self(bitwise_or_34, full_default, getitem_254);  bitwise_or_34 = full_default = getitem_254 = None
        convert_element_type_444: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(where_34, torch.float32);  where_34 = None
        squeeze: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_820: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_821: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
        unsqueeze_822: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
        sum_104: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_444, [0, 2, 3])
        convert_element_type_2: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_256: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_822);  convert_element_type_2 = unsqueeze_822 = None
        mul_827: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, sub_256)
        sum_105: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3]);  mul_827 = None
        mul_828: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 8.304102891156462e-07)
        unsqueeze_823: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
        unsqueeze_824: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
        unsqueeze_825: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
        mul_829: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 8.304102891156462e-07)
        squeeze_1: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_830: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_831: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_829, mul_830);  mul_829 = mul_830 = None
        unsqueeze_826: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
        unsqueeze_827: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
        unsqueeze_828: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
        mul_832: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_829: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
        unsqueeze_830: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
        unsqueeze_831: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
        mul_833: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_828);  sub_256 = unsqueeze_828 = None
        sub_258: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_444, mul_833);  convert_element_type_444 = mul_833 = None
        sub_259: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_825);  sub_258 = unsqueeze_825 = None
        mul_834: "f32[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_831);  sub_259 = unsqueeze_831 = None
        mul_835: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_1);  sum_105 = squeeze_1 = None
        convert_element_type_446: "bf16[96, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_834, torch.bfloat16);  mul_834 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(convert_element_type_446, convert_element_type_1, convert_element_type, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_446 = convert_element_type_1 = convert_element_type = None
        getitem_258: "bf16[32, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_447: "f32[32, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_258, torch.float32);  getitem_258 = None
        return (convert_element_type_447, None, None, None, None, mul_835, sum_104, convert_element_type_443, None, None, None, mul_826, sum_102, convert_element_type_439, None, None, None, mul_817, sum_100, convert_element_type_435, None, None, None, mul_808, sum_98, convert_element_type_431, None, None, None, mul_799, sum_96, convert_element_type_427, None, None, None, mul_790, sum_94, convert_element_type_423, None, None, None, mul_781, sum_92, convert_element_type_419, None, None, None, mul_772, sum_90, convert_element_type_415, None, None, None, mul_763, sum_88, convert_element_type_411, None, None, None, mul_754, sum_86, convert_element_type_407, None, None, None, mul_745, sum_84, convert_element_type_403, None, None, None, mul_736, sum_82, convert_element_type_399, None, None, None, mul_727, sum_80, convert_element_type_395, None, None, None, mul_718, sum_78, convert_element_type_391, None, None, None, mul_709, sum_76, convert_element_type_387, None, None, None, mul_700, sum_74, convert_element_type_383, None, None, None, mul_691, sum_72, convert_element_type_379, None, None, None, mul_682, sum_70, convert_element_type_375, None, None, None, mul_673, sum_68, convert_element_type_371, None, None, None, mul_664, sum_66, convert_element_type_367, None, None, None, mul_655, sum_64, convert_element_type_363, None, None, None, mul_646, sum_62, convert_element_type_359, None, None, None, mul_637, sum_60, convert_element_type_355, None, None, None, mul_628, sum_58, convert_element_type_351, None, None, None, mul_619, sum_56, convert_element_type_347, None, None, None, mul_610, sum_54, convert_element_type_343, None, None, None, mul_601, sum_52, convert_element_type_339, None, None, None, mul_592, sum_50, convert_element_type_335, None, None, None, mul_583, sum_48, convert_element_type_331, None, None, None, mul_574, sum_46, convert_element_type_327, None, None, None, mul_565, sum_44, convert_element_type_323, None, None, None, mul_556, sum_42, convert_element_type_319, None, None, None, mul_547, sum_40, convert_element_type_315, None, None, None, mul_538, sum_38, convert_element_type_311, None, None, None, mul_529, sum_36, convert_element_type_307, None, None, None, mul_520, sum_34, convert_element_type_303, None, None, None, mul_511, sum_32, convert_element_type_299, None, None, None, mul_502, sum_30, convert_element_type_295, None, None, None, mul_493, sum_28, convert_element_type_291, None, None, None, mul_484, sum_26, convert_element_type_287, None, None, None, mul_475, sum_24, convert_element_type_283, None, None, None, mul_466, sum_22, convert_element_type_279, None, None, None, mul_457, sum_20, convert_element_type_275, None, None, None, mul_448, sum_18, convert_element_type_271, None, None, None, mul_439, sum_16, convert_element_type_267, None, None, None, mul_430, sum_14, convert_element_type_263, None, None, None, mul_421, sum_12, convert_element_type_259, None, None, None, mul_412, sum_10, convert_element_type_255, None, None, None, mul_403, sum_8, convert_element_type_251, None, None, None, mul_394, sum_6, convert_element_type_247, None, None, None, mul_385, sum_4, convert_element_type_243, None, None, None, mul_376, sum_2, convert_element_type_237, convert_element_type_238)
