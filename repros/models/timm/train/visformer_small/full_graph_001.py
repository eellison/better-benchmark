class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[32][1]cuda:0", primals_13: "f32[192][1]cuda:0", primals_14: "f32[192][1]cuda:0", primals_15: "f32[1, 192, 28, 28][150528, 1, 5376, 192]cuda:0", primals_19: "f32[192][1]cuda:0", primals_27: "f32[192][1]cuda:0", primals_35: "f32[192][1]cuda:0", primals_43: "f32[192][1]cuda:0", primals_51: "f32[192][1]cuda:0", primals_59: "f32[192][1]cuda:0", primals_67: "f32[192][1]cuda:0", primals_77: "f32[384][1]cuda:0", primals_78: "f32[384][1]cuda:0", primals_79: "f32[1, 384, 14, 14][75264, 1, 5376, 384]cuda:0", primals_83: "f32[384][1]cuda:0", primals_90: "f32[384][1]cuda:0", primals_97: "f32[384][1]cuda:0", primals_104: "f32[384][1]cuda:0", primals_111: "f32[384][1]cuda:0", primals_118: "f32[384][1]cuda:0", primals_125: "f32[384][1]cuda:0", primals_132: "f32[384][1]cuda:0", primals_141: "f32[768][1]cuda:0", primals_142: "f32[768][1]cuda:0", primals_143: "f32[1, 768, 7, 7][37632, 1, 5376, 768]cuda:0", primals_147: "f32[768][1]cuda:0", primals_154: "f32[768][1]cuda:0", primals_161: "f32[768][1]cuda:0", primals_168: "f32[768][1]cuda:0", primals_175: "f32[768][1]cuda:0", primals_182: "f32[768][1]cuda:0", primals_189: "f32[768][1]cuda:0", primals_196: "f32[768][1]cuda:0", primals_203: "f32[768][1]cuda:0", convert_element_type: "bf16[32, 3, 7, 7][147, 1, 21, 3]cuda:0", convert_element_type_1: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", squeeze_1: "f32[32][1]cuda:0", relu: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0", convert_element_type_5: "bf16[192, 32, 4, 4][512, 1, 128, 32]cuda:0", convolution_1: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", getitem_3: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", rsqrt_1: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", squeeze_7: "f32[192][1]cuda:0", convert_element_type_8: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_9: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convolution_2: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_11: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_12: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0", convolution_3: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_14: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_15: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_10: "f32[192][1]cuda:0", convert_element_type_16: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_17: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convolution_5: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_19: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_20: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0", convolution_6: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_22: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_23: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_13: "f32[192][1]cuda:0", convert_element_type_24: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_25: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convolution_8: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_27: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_28: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0", convolution_9: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_30: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_31: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_16: "f32[192][1]cuda:0", convert_element_type_32: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_33: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convolution_11: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_35: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_36: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0", convolution_12: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_38: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_39: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_19: "f32[192][1]cuda:0", convert_element_type_40: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_41: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convolution_14: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_43: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_44: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0", convolution_15: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_46: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_47: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_22: "f32[192][1]cuda:0", convert_element_type_48: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_49: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convolution_17: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_51: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_52: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0", convolution_18: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_54: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_55: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_25: "f32[192][1]cuda:0", convert_element_type_56: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0", convert_element_type_57: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convolution_20: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_59: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_60: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0", convolution_21: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_62: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0", convert_element_type_63: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_65: "bf16[384, 192, 2, 2][768, 1, 384, 192]cuda:0", convert_element_type_66: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convolution_23: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", getitem_19: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", rsqrt_9: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", squeeze_31: "f32[384][1]cuda:0", convert_element_type_69: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_70: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", bmm_default_11: "bf16[768, 200, 200][40000, 200, 1]cuda:0", amax_default_14: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", amax_default_15: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", logical_not_default_15: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0", sum_1: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", view_7: "bf16[128, 384, 14, 14][75264, 196, 14, 1]cuda:0", convert_element_type_77: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_34: "f32[384][1]cuda:0", convert_element_type_78: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_79: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convolution_26: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", convert_element_type_81: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", convert_element_type_82: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", squeeze_37: "f32[384][1]cuda:0", convert_element_type_83: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_84: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", bmm_default_9: "bf16[768, 200, 200][40000, 200, 1]cuda:0", amax_default_12: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", amax_default_13: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", logical_not_default_13: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0", sum_2: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", view_15: "bf16[128, 384, 14, 14][75264, 196, 14, 1]cuda:0", convert_element_type_91: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_40: "f32[384][1]cuda:0", convert_element_type_92: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_93: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convolution_30: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", convert_element_type_95: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", convert_element_type_96: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", squeeze_43: "f32[384][1]cuda:0", convert_element_type_97: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_98: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", bmm_default_7: "bf16[768, 200, 200][40000, 200, 1]cuda:0", amax_default_10: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", amax_default_11: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", logical_not_default_11: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0", sum_3: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", view_23: "bf16[128, 384, 14, 14][75264, 196, 14, 1]cuda:0", convert_element_type_105: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_46: "f32[384][1]cuda:0", convert_element_type_106: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_107: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convolution_34: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", convert_element_type_109: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", convert_element_type_110: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", squeeze_49: "f32[384][1]cuda:0", convert_element_type_111: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_112: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", bmm_default_5: "bf16[768, 200, 200][40000, 200, 1]cuda:0", amax_default_8: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", amax_default_9: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", logical_not_default_9: "b8[128, 6, 196, 1][1176, 196, 1, 1]cuda:0", sum_4: "f32[128, 6, 196, 1][1184, 196, 1, 1]cuda:0", view_31: "bf16[128, 384, 14, 14][75264, 196, 14, 1]cuda:0", convert_element_type_119: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", squeeze_52: "f32[384][1]cuda:0", convert_element_type_120: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0", convert_element_type_121: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convolution_38: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", convert_element_type_123: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0", convert_element_type_124: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0", convert_element_type_126: "bf16[768, 384, 2, 2][1536, 1, 768, 384]cuda:0", convert_element_type_127: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convolution_40: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", getitem_49: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0", rsqrt_18: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0", squeeze_58: "f32[768][1]cuda:0", convert_element_type_130: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0", convert_element_type_131: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", bmm_default_3: "bf16[768, 56, 56][3136, 56, 1]cuda:0", amax_default_6: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", amax_default_7: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", logical_not_default_7: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0", sum_5: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", view_39: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0", convert_element_type_138: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0", squeeze_61: "f32[768][1]cuda:0", convert_element_type_139: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0", convert_element_type_140: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", convolution_43: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0", convert_element_type_142: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0", convert_element_type_143: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0", squeeze_64: "f32[768][1]cuda:0", convert_element_type_144: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0", convert_element_type_145: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", bmm_default_2: "bf16[768, 56, 56][3136, 56, 1]cuda:0", amax_default_4: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", amax_default_5: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", logical_not_default_5: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0", sum_6: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", view_47: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0", convert_element_type_152: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0", squeeze_67: "f32[768][1]cuda:0", convert_element_type_153: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0", convert_element_type_154: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", convolution_47: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0", convert_element_type_156: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0", convert_element_type_157: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0", squeeze_70: "f32[768][1]cuda:0", convert_element_type_158: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0", convert_element_type_159: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", bmm_default_1: "bf16[768, 56, 56][3136, 56, 1]cuda:0", amax_default_2: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", amax_default_3: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", logical_not_default_3: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0", sum_7: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", view_55: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0", convert_element_type_166: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0", squeeze_73: "f32[768][1]cuda:0", convert_element_type_167: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0", convert_element_type_168: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", convolution_51: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0", convert_element_type_170: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0", convert_element_type_171: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0", squeeze_76: "f32[768][1]cuda:0", convert_element_type_172: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0", convert_element_type_173: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", bmm_default: "bf16[768, 56, 56][3136, 56, 1]cuda:0", amax_default: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", amax_default_1: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", logical_not_default_1: "b8[128, 6, 49, 1][294, 49, 1, 1]cuda:0", sum_8: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0", view_63: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0", convert_element_type_180: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0", squeeze_79: "f32[768][1]cuda:0", convert_element_type_181: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0", convert_element_type_182: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", convolution_55: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0", convert_element_type_184: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0", convert_element_type_185: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0", squeeze_82: "f32[768][1]cuda:0", convert_element_type_188: "bf16[128, 768][768, 1]cuda:0", permute_25: "bf16[1000, 768][768, 1]cuda:0", sub_36: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", sub_40: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", permute_30: "bf16[768, 49, 49][2401, 1, 49]cuda:0", permute_31: "bf16[768, 128, 49][6272, 1, 128]cuda:0", permute_32: "bf16[768, 128, 49][6272, 1, 128]cuda:0", permute_33: "bf16[768, 49, 128][6272, 1, 49]cuda:0", sub_44: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", sub_48: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", permute_37: "bf16[768, 49, 49][2401, 1, 49]cuda:0", permute_38: "bf16[768, 128, 49][6272, 1, 128]cuda:0", permute_39: "bf16[768, 128, 49][6272, 1, 128]cuda:0", permute_40: "bf16[768, 49, 128][6272, 1, 49]cuda:0", sub_52: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", sub_56: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", permute_44: "bf16[768, 49, 49][2401, 1, 49]cuda:0", permute_45: "bf16[768, 128, 49][6272, 1, 128]cuda:0", permute_46: "bf16[768, 128, 49][6272, 1, 128]cuda:0", permute_47: "bf16[768, 49, 128][6272, 1, 49]cuda:0", sub_60: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", sub_64: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0", permute_51: "bf16[768, 49, 49][2401, 1, 49]cuda:0", permute_52: "bf16[768, 128, 49][6272, 1, 128]cuda:0", permute_53: "bf16[768, 128, 49][6272, 1, 128]cuda:0", permute_54: "bf16[768, 49, 128][6272, 1, 49]cuda:0", unsqueeze_210: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0", sub_76: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", permute_58: "bf16[768, 196, 196][38464, 1, 196]cuda:0", permute_59: "bf16[768, 64, 196][12544, 1, 64]cuda:0", permute_60: "bf16[768, 64, 196][12544, 1, 64]cuda:0", permute_61: "bf16[768, 196, 64][12544, 1, 196]cuda:0", sub_80: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", sub_84: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", permute_65: "bf16[768, 196, 196][38464, 1, 196]cuda:0", permute_66: "bf16[768, 64, 196][12544, 1, 64]cuda:0", permute_67: "bf16[768, 64, 196][12544, 1, 64]cuda:0", permute_68: "bf16[768, 196, 64][12544, 1, 196]cuda:0", sub_88: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", sub_92: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", permute_72: "bf16[768, 196, 196][38464, 1, 196]cuda:0", permute_73: "bf16[768, 64, 196][12544, 1, 64]cuda:0", permute_74: "bf16[768, 64, 196][12544, 1, 64]cuda:0", permute_75: "bf16[768, 196, 64][12544, 1, 196]cuda:0", sub_96: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", sub_100: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", permute_79: "bf16[768, 196, 196][38464, 1, 196]cuda:0", permute_80: "bf16[768, 64, 196][12544, 1, 64]cuda:0", permute_81: "bf16[768, 64, 196][12544, 1, 64]cuda:0", permute_82: "bf16[768, 196, 64][12544, 1, 196]cuda:0", unsqueeze_318: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", sub_112: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", sub_116: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", sub_120: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", sub_124: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", sub_128: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", sub_132: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", unsqueeze_414: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_438: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:474 in forward_head, code: return x if pre_logits else self.head(x)
        mm: "bf16[128, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_25);  permute_25 = None
        permute_26: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_26, convert_element_type_188);  permute_26 = convert_element_type_188 = None
        sum_9: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_65: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_9, [1000]);  sum_9 = None
        convert_element_type_196: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_65, torch.bfloat16);  view_65 = None
        convert_element_type_197: "f32[128, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_198: "f32[1000, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_199: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_196, torch.float32);  convert_element_type_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_66: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_197, [128, 768, 1, 1]);  convert_element_type_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_84: "f32[128, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_66, 3);  view_66 = None
        squeeze_85: "f32[128, 768][768, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze_84, 2);  squeeze_84 = None
        full_24: "f32[98304][1]cuda:0" = torch.ops.aten.full.default([98304], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "f32[98304][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full_24, squeeze_85, [128, 768], [768, 1], 0);  full_24 = squeeze_85 = None
        as_strided_5: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 768, 1, 1], [768, 1, 1, 1], 0);  as_strided_scatter = None
        expand_33: "f32[128, 768, 7, 7][768, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 768, 7, 7]);  as_strided_5 = None
        div_8: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_33, 49);  expand_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:468 in forward_features, code: x = self.norm(x)
        sum_10: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_8, [0, 2, 3])
        mul_270: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_8, sub_36)
        sum_11: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 2, 3]);  mul_270 = None
        mul_271: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.00015943877551020407)
        unsqueeze_115: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_116: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
        unsqueeze_117: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 3);  unsqueeze_116 = None
        mul_272: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.00015943877551020407)
        mul_273: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_274: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
        unsqueeze_118: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
        unsqueeze_119: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 2);  unsqueeze_118 = None
        unsqueeze_120: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 3);  unsqueeze_119 = None
        mul_275: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_203);  primals_203 = None
        unsqueeze_121: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_275, 0);  mul_275 = None
        unsqueeze_122: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 2);  unsqueeze_121 = None
        unsqueeze_123: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 3);  unsqueeze_122 = None
        mul_276: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_120);  sub_36 = unsqueeze_120 = None
        sub_38: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(div_8, mul_276);  div_8 = mul_276 = None
        sub_39: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, unsqueeze_117);  sub_38 = unsqueeze_117 = None
        mul_277: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_123);  sub_39 = unsqueeze_123 = None
        mul_278: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_82);  sum_11 = squeeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        convert_element_type_200: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_277, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_200, convert_element_type_184, convert_element_type_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_200 = convert_element_type_184 = convert_element_type_185 = None
        getitem_80: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = convolution_backward[0]
        getitem_81: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_201: "f32[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_202: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_80, torch.float32);  getitem_80 = None
        convert_element_type_183: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        mul_261: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_183, 0.7071067811865476)
        erf_21: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.erf.default(mul_261);  mul_261 = None
        add_181: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_280: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(add_181, 0.5);  add_181 = None
        mul_281: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_183, convert_element_type_183)
        mul_282: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, -0.5);  mul_281 = None
        exp_8: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.exp.default(mul_282);  mul_282 = None
        mul_283: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_284: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_183, mul_283);  convert_element_type_183 = mul_283 = None
        add_189: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(mul_280, mul_284);  mul_280 = mul_284 = None
        mul_285: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_202, add_189);  convert_element_type_202 = add_189 = None
        convert_element_type_204: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(mul_285, torch.bfloat16);  mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_204, convert_element_type_182, convert_element_type_181, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_204 = convert_element_type_182 = convert_element_type_181 = None
        getitem_83: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_1[0]
        getitem_84: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_205: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_83, torch.float32);  getitem_83 = None
        convert_element_type_206: "f32[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_84, torch.float32);  getitem_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_12: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_205, [0, 2, 3])
        mul_286: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_205, sub_40)
        sum_13: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 2, 3]);  mul_286 = None
        mul_287: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.00015943877551020407)
        unsqueeze_127: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
        unsqueeze_128: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
        unsqueeze_129: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
        mul_288: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.00015943877551020407)
        mul_289: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_290: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        unsqueeze_130: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_290, 0);  mul_290 = None
        unsqueeze_131: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
        unsqueeze_132: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
        mul_291: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_196);  primals_196 = None
        unsqueeze_133: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
        unsqueeze_134: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
        unsqueeze_135: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
        mul_292: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_132);  sub_40 = unsqueeze_132 = None
        sub_42: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_205, mul_292);  convert_element_type_205 = mul_292 = None
        sub_43: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(sub_42, unsqueeze_129);  sub_42 = unsqueeze_129 = None
        mul_293: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_135);  sub_43 = unsqueeze_135 = None
        mul_294: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_79);  sum_13 = squeeze_79 = None
        add_190: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_277, mul_293);  mul_277 = mul_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        convert_element_type_207: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_190, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_207, view_63, convert_element_type_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_207 = view_63 = convert_element_type_180 = None
        getitem_86: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_2[0]
        getitem_87: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_208: "f32[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        view_67: "bf16[128, 6, 128, 49][37632, 128, 1, 768]cuda:0" = torch.ops.aten.reshape.default(getitem_86, [128, 6, 128, 49]);  getitem_86 = None
        permute_29: "bf16[128, 6, 49, 128][37632, 128, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_67, [0, 1, 3, 2]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_82: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_68: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [768, 49, 128]);  clone_82 = None
        bmm_16: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_30, view_68);  permute_30 = None
        bmm_17: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_68, permute_31);  view_68 = permute_31 = None
        view_69: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_16, [128, 6, 49, 128]);  bmm_16 = None
        view_70: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_17, [128, 6, 49, 49]);  bmm_17 = None
        convert_element_type_213: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        slice_tensor: "bf16[768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default, 1, 0, -7);  bmm_default = None
        slice_tensor_1: "bf16[768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor, 2, 0, -7);  slice_tensor = None
        view_59: "bf16[128, 6, 49, 49][18816, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_1, [128, 6, 49, 49]);  slice_tensor_1 = None

        # No stacktrace found for following nodes
        mul_tensor: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_59, 0.08838834764831845)
        convert_element_type_default: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        mul_tensor_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        sub_tensor: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = amax_default = None
        mul_tensor_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor, 0.08838834764831845);  sub_tensor = None
        sub_tensor_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default_1);  convert_element_type_default = amax_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  logical_not_default_1 = mul_tensor_2 = sub_tensor_1 = None
        exp_7: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(where_self);  where_self = None
        div_7: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        mul_295: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_213, div_7);  convert_element_type_213 = None
        sum_14: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_295, [-1], True)
        neg: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_14, mul_295);  neg = sum_14 = mul_295 = None
        convert_element_type_214: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_296: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_214, 0.08838834764831845);  convert_element_type_214 = None
        view_71: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(mul_296, [768, 49, 49]);  mul_296 = None
        bmm_18: "bf16[768, 128, 49][6272, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_32, view_71);  permute_32 = None
        bmm_19: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_71, permute_33);  view_71 = permute_33 = None
        view_72: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [128, 6, 128, 49]);  bmm_18 = None
        view_73: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_19, [128, 6, 49, 128]);  bmm_19 = None
        permute_34: "bf16[128, 6, 49, 128][37632, 6272, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_72, [0, 1, 3, 2]);  view_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        cat: "bf16[384, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.cat.default([view_73, permute_34, view_69]);  view_73 = permute_34 = view_69 = None
        view_74: "bf16[3, 128, 6, 49, 128][4816896, 37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(cat, [3, 128, 6, 49, 128]);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        permute_35: "bf16[128, 3, 6, 128, 49][37632, 4816896, 6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_74, [1, 0, 2, 4, 3]);  view_74 = None
        clone_83: "bf16[128, 3, 6, 128, 49][112896, 37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_75: "bf16[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [128, 2304, 7, 7]);  clone_83 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_75, convert_element_type_173, convert_element_type_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_75 = convert_element_type_173 = convert_element_type_172 = None
        getitem_89: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_3[0]
        getitem_90: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_219: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_89, torch.float32);  getitem_89 = None
        convert_element_type_220: "f32[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_90, torch.float32);  getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sum_15: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_219, [0, 2, 3])
        mul_297: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_219, sub_44)
        sum_16: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_297, [0, 2, 3]);  mul_297 = None
        mul_298: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.00015943877551020407)
        unsqueeze_139: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_140: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
        unsqueeze_141: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
        mul_299: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.00015943877551020407)
        mul_300: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_301: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
        unsqueeze_142: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
        unsqueeze_143: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
        unsqueeze_144: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
        mul_302: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_189);  primals_189 = None
        unsqueeze_145: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
        unsqueeze_146: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
        unsqueeze_147: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
        mul_303: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_144);  sub_44 = unsqueeze_144 = None
        sub_46: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_219, mul_303);  convert_element_type_219 = mul_303 = None
        sub_47: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(sub_46, unsqueeze_141);  sub_46 = unsqueeze_141 = None
        mul_304: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_147);  sub_47 = unsqueeze_147 = None
        mul_305: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, squeeze_76);  sum_16 = squeeze_76 = None
        add_191: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_190, mul_304);  add_190 = mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        convert_element_type_221: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_221, convert_element_type_170, convert_element_type_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_221 = convert_element_type_170 = convert_element_type_171 = None
        getitem_92: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = convolution_backward_4[0]
        getitem_93: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_222: "f32[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_93, torch.float32);  getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_223: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_92, torch.float32);  getitem_92 = None
        convert_element_type_169: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        mul_243: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, 0.7071067811865476)
        erf_20: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_168: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_307: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(add_168, 0.5);  add_168 = None
        mul_308: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, convert_element_type_169)
        mul_309: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, -0.5);  mul_308 = None
        exp_9: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.exp.default(mul_309);  mul_309 = None
        mul_310: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_311: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, mul_310);  convert_element_type_169 = mul_310 = None
        add_193: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(mul_307, mul_311);  mul_307 = mul_311 = None
        mul_312: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_223, add_193);  convert_element_type_223 = add_193 = None
        convert_element_type_225: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_225, convert_element_type_168, convert_element_type_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_225 = convert_element_type_168 = convert_element_type_167 = None
        getitem_95: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_5[0]
        getitem_96: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_226: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
        convert_element_type_227: "f32[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_96, torch.float32);  getitem_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_17: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_226, [0, 2, 3])
        mul_313: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_226, sub_48)
        sum_18: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_313, [0, 2, 3]);  mul_313 = None
        mul_314: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 0.00015943877551020407)
        unsqueeze_151: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
        unsqueeze_152: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
        unsqueeze_153: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
        mul_315: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.00015943877551020407)
        mul_316: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_317: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
        unsqueeze_154: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_317, 0);  mul_317 = None
        unsqueeze_155: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
        unsqueeze_156: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
        mul_318: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_182);  primals_182 = None
        unsqueeze_157: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
        unsqueeze_158: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
        unsqueeze_159: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
        mul_319: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_156);  sub_48 = unsqueeze_156 = None
        sub_50: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_226, mul_319);  convert_element_type_226 = mul_319 = None
        sub_51: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(sub_50, unsqueeze_153);  sub_50 = unsqueeze_153 = None
        mul_320: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_159);  sub_51 = unsqueeze_159 = None
        mul_321: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, squeeze_73);  sum_18 = squeeze_73 = None
        add_194: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_191, mul_320);  add_191 = mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        convert_element_type_228: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_194, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_228, view_55, convert_element_type_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_228 = view_55 = convert_element_type_166 = None
        getitem_98: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_6[0]
        getitem_99: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_229: "f32[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_99, torch.float32);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        view_76: "bf16[128, 6, 128, 49][37632, 128, 1, 768]cuda:0" = torch.ops.aten.reshape.default(getitem_98, [128, 6, 128, 49]);  getitem_98 = None
        permute_36: "bf16[128, 6, 49, 128][37632, 128, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 1, 3, 2]);  view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_84: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_77: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [768, 49, 128]);  clone_84 = None
        bmm_20: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_37, view_77);  permute_37 = None
        bmm_21: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_77, permute_38);  view_77 = permute_38 = None
        view_78: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [128, 6, 49, 128]);  bmm_20 = None
        view_79: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [128, 6, 49, 49]);  bmm_21 = None
        convert_element_type_234: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        slice_tensor_2: "bf16[768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_1, 1, 0, -7);  bmm_default_1 = None
        slice_tensor_3: "bf16[768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_2, 2, 0, -7);  slice_tensor_2 = None
        view_51: "bf16[128, 6, 49, 49][18816, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_3, [128, 6, 49, 49]);  slice_tensor_3 = None

        # No stacktrace found for following nodes
        mul_tensor_4: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_51, 0.08838834764831845)
        convert_element_type_default_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float32);  mul_tensor_4 = None
        convert_element_type_default_3: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_51, torch.float32);  view_51 = None
        mul_tensor_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1);  convert_element_type_default_3 = None
        sub_tensor_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_5, amax_default_2);  mul_tensor_5 = amax_default_2 = None
        mul_tensor_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.08838834764831845);  sub_tensor_2 = None
        sub_tensor_3: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default_3);  convert_element_type_default_2 = amax_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_3, mul_tensor_6, sub_tensor_3);  logical_not_default_3 = mul_tensor_6 = sub_tensor_3 = None
        exp_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(where_self_1);  where_self_1 = None
        div_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        mul_322: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_234, div_6);  convert_element_type_234 = None
        sum_19: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_322, [-1], True)
        neg_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_1: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_19, mul_322);  neg_1 = sum_19 = mul_322 = None
        convert_element_type_235: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_323: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, 0.08838834764831845);  convert_element_type_235 = None
        view_80: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(mul_323, [768, 49, 49]);  mul_323 = None
        bmm_22: "bf16[768, 128, 49][6272, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_39, view_80);  permute_39 = None
        bmm_23: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_80, permute_40);  view_80 = permute_40 = None
        view_81: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [128, 6, 128, 49]);  bmm_22 = None
        view_82: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_23, [128, 6, 49, 128]);  bmm_23 = None
        permute_41: "bf16[128, 6, 49, 128][37632, 6272, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_81, [0, 1, 3, 2]);  view_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        cat_1: "bf16[384, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.cat.default([view_82, permute_41, view_78]);  view_82 = permute_41 = view_78 = None
        view_83: "bf16[3, 128, 6, 49, 128][4816896, 37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(cat_1, [3, 128, 6, 49, 128]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        permute_42: "bf16[128, 3, 6, 128, 49][37632, 4816896, 6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_83, [1, 0, 2, 4, 3]);  view_83 = None
        clone_85: "bf16[128, 3, 6, 128, 49][112896, 37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
        view_84: "bf16[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [128, 2304, 7, 7]);  clone_85 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_84, convert_element_type_159, convert_element_type_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_84 = convert_element_type_159 = convert_element_type_158 = None
        getitem_101: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_7[0]
        getitem_102: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_240: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
        convert_element_type_241: "f32[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_102, torch.float32);  getitem_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sum_20: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_240, [0, 2, 3])
        mul_324: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_240, sub_52)
        sum_21: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 2, 3]);  mul_324 = None
        mul_325: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.00015943877551020407)
        unsqueeze_163: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_325, 0);  mul_325 = None
        unsqueeze_164: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
        unsqueeze_165: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
        mul_326: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.00015943877551020407)
        mul_327: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_328: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_166: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_328, 0);  mul_328 = None
        unsqueeze_167: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
        unsqueeze_168: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
        mul_329: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_175);  primals_175 = None
        unsqueeze_169: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_329, 0);  mul_329 = None
        unsqueeze_170: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
        unsqueeze_171: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
        mul_330: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_168);  sub_52 = unsqueeze_168 = None
        sub_54: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_240, mul_330);  convert_element_type_240 = mul_330 = None
        sub_55: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(sub_54, unsqueeze_165);  sub_54 = unsqueeze_165 = None
        mul_331: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_171);  sub_55 = unsqueeze_171 = None
        mul_332: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_70);  sum_21 = squeeze_70 = None
        add_195: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_194, mul_331);  add_194 = mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        convert_element_type_242: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_195, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_242, convert_element_type_156, convert_element_type_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_242 = convert_element_type_156 = convert_element_type_157 = None
        getitem_104: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = convolution_backward_8[0]
        getitem_105: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_243: "f32[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_244: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_104, torch.float32);  getitem_104 = None
        convert_element_type_155: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        mul_225: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_155, 0.7071067811865476)
        erf_19: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.erf.default(mul_225);  mul_225 = None
        add_155: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_334: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(add_155, 0.5);  add_155 = None
        mul_335: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_155, convert_element_type_155)
        mul_336: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(mul_335, -0.5);  mul_335 = None
        exp_10: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.exp.default(mul_336);  mul_336 = None
        mul_337: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_338: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_155, mul_337);  convert_element_type_155 = mul_337 = None
        add_197: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(mul_334, mul_338);  mul_334 = mul_338 = None
        mul_339: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_244, add_197);  convert_element_type_244 = add_197 = None
        convert_element_type_246: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(mul_339, torch.bfloat16);  mul_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_246, convert_element_type_154, convert_element_type_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_246 = convert_element_type_154 = convert_element_type_153 = None
        getitem_107: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_9[0]
        getitem_108: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_247: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
        convert_element_type_248: "f32[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_108, torch.float32);  getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_22: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_247, [0, 2, 3])
        mul_340: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_247, sub_56)
        sum_23: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 2, 3]);  mul_340 = None
        mul_341: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 0.00015943877551020407)
        unsqueeze_175: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
        unsqueeze_176: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
        unsqueeze_177: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
        mul_342: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 0.00015943877551020407)
        mul_343: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_344: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, mul_343);  mul_342 = mul_343 = None
        unsqueeze_178: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_344, 0);  mul_344 = None
        unsqueeze_179: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
        unsqueeze_180: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
        mul_345: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_168);  primals_168 = None
        unsqueeze_181: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_345, 0);  mul_345 = None
        unsqueeze_182: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
        unsqueeze_183: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
        mul_346: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_180);  sub_56 = unsqueeze_180 = None
        sub_58: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_247, mul_346);  convert_element_type_247 = mul_346 = None
        sub_59: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_177);  sub_58 = unsqueeze_177 = None
        mul_347: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_183);  sub_59 = unsqueeze_183 = None
        mul_348: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_67);  sum_23 = squeeze_67 = None
        add_198: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_195, mul_347);  add_195 = mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        convert_element_type_249: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_249, view_47, convert_element_type_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_249 = view_47 = convert_element_type_152 = None
        getitem_110: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_10[0]
        getitem_111: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_250: "f32[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        view_85: "bf16[128, 6, 128, 49][37632, 128, 1, 768]cuda:0" = torch.ops.aten.reshape.default(getitem_110, [128, 6, 128, 49]);  getitem_110 = None
        permute_43: "bf16[128, 6, 49, 128][37632, 128, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_85, [0, 1, 3, 2]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_86: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
        view_86: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [768, 49, 128]);  clone_86 = None
        bmm_24: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_44, view_86);  permute_44 = None
        bmm_25: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_86, permute_45);  view_86 = permute_45 = None
        view_87: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [128, 6, 49, 128]);  bmm_24 = None
        view_88: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [128, 6, 49, 49]);  bmm_25 = None
        convert_element_type_255: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_88, torch.float32);  view_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        slice_tensor_4: "bf16[768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_2, 1, 0, -7);  bmm_default_2 = None
        slice_tensor_5: "bf16[768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_4, 2, 0, -7);  slice_tensor_4 = None
        view_43: "bf16[128, 6, 49, 49][18816, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_5, [128, 6, 49, 49]);  slice_tensor_5 = None

        # No stacktrace found for following nodes
        mul_tensor_8: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_43, 0.08838834764831845)
        convert_element_type_default_4: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.float32);  mul_tensor_8 = None
        convert_element_type_default_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        mul_tensor_9: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1);  convert_element_type_default_5 = None
        sub_tensor_4: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_9, amax_default_4);  mul_tensor_9 = amax_default_4 = None
        mul_tensor_10: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.08838834764831845);  sub_tensor_4 = None
        sub_tensor_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_4, amax_default_5);  convert_element_type_default_4 = amax_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_5, mul_tensor_10, sub_tensor_5);  logical_not_default_5 = mul_tensor_10 = sub_tensor_5 = None
        exp_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(where_self_2);  where_self_2 = None
        div_5: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        mul_349: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_255, div_5);  convert_element_type_255 = None
        sum_24: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_349, [-1], True)
        neg_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_2: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_24, mul_349);  neg_2 = sum_24 = mul_349 = None
        convert_element_type_256: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_350: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_256, 0.08838834764831845);  convert_element_type_256 = None
        view_89: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(mul_350, [768, 49, 49]);  mul_350 = None
        bmm_26: "bf16[768, 128, 49][6272, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_46, view_89);  permute_46 = None
        bmm_27: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_89, permute_47);  view_89 = permute_47 = None
        view_90: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [128, 6, 128, 49]);  bmm_26 = None
        view_91: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [128, 6, 49, 128]);  bmm_27 = None
        permute_48: "bf16[128, 6, 49, 128][37632, 6272, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 1, 3, 2]);  view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        cat_2: "bf16[384, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.cat.default([view_91, permute_48, view_87]);  view_91 = permute_48 = view_87 = None
        view_92: "bf16[3, 128, 6, 49, 128][4816896, 37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(cat_2, [3, 128, 6, 49, 128]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        permute_49: "bf16[128, 3, 6, 128, 49][37632, 4816896, 6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_92, [1, 0, 2, 4, 3]);  view_92 = None
        clone_87: "bf16[128, 3, 6, 128, 49][112896, 37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_93: "bf16[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [128, 2304, 7, 7]);  clone_87 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(view_93, convert_element_type_145, convert_element_type_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_93 = convert_element_type_145 = convert_element_type_144 = None
        getitem_113: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_11[0]
        getitem_114: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_261: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None
        convert_element_type_262: "f32[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_114, torch.float32);  getitem_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sum_25: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_261, [0, 2, 3])
        mul_351: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_261, sub_60)
        sum_26: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 2, 3]);  mul_351 = None
        mul_352: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 0.00015943877551020407)
        unsqueeze_187: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_352, 0);  mul_352 = None
        unsqueeze_188: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
        unsqueeze_189: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
        mul_353: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 0.00015943877551020407)
        mul_354: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_355: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, mul_354);  mul_353 = mul_354 = None
        unsqueeze_190: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
        unsqueeze_191: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
        unsqueeze_192: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
        mul_356: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_161);  primals_161 = None
        unsqueeze_193: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
        unsqueeze_194: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
        unsqueeze_195: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
        mul_357: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_192);  sub_60 = unsqueeze_192 = None
        sub_62: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_261, mul_357);  convert_element_type_261 = mul_357 = None
        sub_63: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_189);  sub_62 = unsqueeze_189 = None
        mul_358: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_195);  sub_63 = unsqueeze_195 = None
        mul_359: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, squeeze_64);  sum_26 = squeeze_64 = None
        add_199: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_198, mul_358);  add_198 = mul_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        convert_element_type_263: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_263, convert_element_type_142, convert_element_type_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_263 = convert_element_type_142 = convert_element_type_143 = None
        getitem_116: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = convolution_backward_12[0]
        getitem_117: "bf16[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_264: "f32[768, 3072, 1, 1][3072, 1, 3072, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_265: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_116, torch.float32);  getitem_116 = None
        convert_element_type_141: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        mul_207: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, 0.7071067811865476)
        erf_18: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_142: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_361: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(add_142, 0.5);  add_142 = None
        mul_362: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, convert_element_type_141)
        mul_363: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(mul_362, -0.5);  mul_362 = None
        exp_11: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.exp.default(mul_363);  mul_363 = None
        mul_364: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_365: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, mul_364);  convert_element_type_141 = mul_364 = None
        add_201: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.add.Tensor(mul_361, mul_365);  mul_361 = mul_365 = None
        mul_366: "f32[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, add_201);  convert_element_type_265 = add_201 = None
        convert_element_type_267: "bf16[128, 3072, 7, 7][150528, 1, 21504, 3072]cuda:0" = torch.ops.prims.convert_element_type.default(mul_366, torch.bfloat16);  mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_267, convert_element_type_140, convert_element_type_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_267 = convert_element_type_140 = convert_element_type_139 = None
        getitem_119: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_13[0]
        getitem_120: "bf16[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_268: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_119, torch.float32);  getitem_119 = None
        convert_element_type_269: "f32[3072, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_120, torch.float32);  getitem_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_27: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_268, [0, 2, 3])
        mul_367: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_268, sub_64)
        sum_28: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 2, 3]);  mul_367 = None
        mul_368: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 0.00015943877551020407)
        unsqueeze_199: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
        unsqueeze_200: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        unsqueeze_201: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
        mul_369: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
        mul_370: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_371: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
        unsqueeze_202: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
        unsqueeze_203: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
        unsqueeze_204: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
        mul_372: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_154);  primals_154 = None
        unsqueeze_205: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
        unsqueeze_206: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
        unsqueeze_207: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        mul_373: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_204);  sub_64 = unsqueeze_204 = None
        sub_66: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_268, mul_373);  convert_element_type_268 = mul_373 = None
        sub_67: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_201);  sub_66 = unsqueeze_201 = None
        mul_374: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_207);  sub_67 = unsqueeze_207 = None
        mul_375: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, squeeze_61);  sum_28 = squeeze_61 = None
        add_202: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_199, mul_374);  add_199 = mul_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        convert_element_type_270: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_270, view_39, convert_element_type_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_270 = view_39 = convert_element_type_138 = None
        getitem_122: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_14[0]
        getitem_123: "bf16[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_271: "f32[768, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        view_94: "bf16[128, 6, 128, 49][37632, 128, 1, 768]cuda:0" = torch.ops.aten.reshape.default(getitem_122, [128, 6, 128, 49]);  getitem_122 = None
        permute_50: "bf16[128, 6, 49, 128][37632, 128, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_94, [0, 1, 3, 2]);  view_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_88: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_95: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [768, 49, 128]);  clone_88 = None
        bmm_28: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(permute_51, view_95);  permute_51 = None
        bmm_29: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.bmm.default(view_95, permute_52);  view_95 = permute_52 = None
        view_96: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [128, 6, 49, 128]);  bmm_28 = None
        view_97: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [128, 6, 49, 49]);  bmm_29 = None
        convert_element_type_276: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        slice_tensor_6: "bf16[768, 49, 56][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_3, 1, 0, -7);  bmm_default_3 = None
        slice_tensor_7: "bf16[768, 49, 49][3136, 56, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_6, 2, 0, -7);  slice_tensor_6 = None
        view_35: "bf16[128, 6, 49, 49][18816, 3136, 56, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_7, [128, 6, 49, 49]);  slice_tensor_7 = None

        # No stacktrace found for following nodes
        mul_tensor_12: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_35, 0.08838834764831845)
        convert_element_type_default_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.float32);  mul_tensor_12 = None
        convert_element_type_default_7: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        mul_tensor_13: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1);  convert_element_type_default_7 = None
        sub_tensor_6: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_13, amax_default_6);  mul_tensor_13 = amax_default_6 = None
        mul_tensor_14: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.08838834764831845);  sub_tensor_6 = None
        sub_tensor_7: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_6, amax_default_7);  convert_element_type_default_6 = amax_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_3: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_7, mul_tensor_14, sub_tensor_7);  logical_not_default_7 = mul_tensor_14 = sub_tensor_7 = None
        exp_4: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.exp.default(where_self_3);  where_self_3 = None
        div_4: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        mul_376: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_276, div_4);  convert_element_type_276 = None
        sum_29: "f32[128, 6, 49, 1][294, 49, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_376, [-1], True)
        neg_3: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_3: "f32[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_29, mul_376);  neg_3 = sum_29 = mul_376 = None
        convert_element_type_277: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_377: "bf16[128, 6, 49, 49][14406, 2401, 49, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_277, 0.08838834764831845);  convert_element_type_277 = None
        view_98: "bf16[768, 49, 49][2401, 49, 1]cuda:0" = torch.ops.aten.reshape.default(mul_377, [768, 49, 49]);  mul_377 = None
        bmm_30: "bf16[768, 128, 49][6272, 49, 1]cuda:0" = torch.ops.aten.bmm.default(permute_53, view_98);  permute_53 = None
        bmm_31: "bf16[768, 49, 128][6272, 128, 1]cuda:0" = torch.ops.aten.bmm.default(view_98, permute_54);  view_98 = permute_54 = None
        view_99: "bf16[128, 6, 128, 49][37632, 6272, 49, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [128, 6, 128, 49]);  bmm_30 = None
        view_100: "bf16[128, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [128, 6, 49, 128]);  bmm_31 = None
        permute_55: "bf16[128, 6, 49, 128][37632, 6272, 1, 49]cuda:0" = torch.ops.aten.permute.default(view_99, [0, 1, 3, 2]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        cat_3: "bf16[384, 6, 49, 128][37632, 6272, 128, 1]cuda:0" = torch.ops.aten.cat.default([view_100, permute_55, view_96]);  view_100 = permute_55 = view_96 = None
        view_101: "bf16[3, 128, 6, 49, 128][4816896, 37632, 6272, 128, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [3, 128, 6, 49, 128]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        permute_56: "bf16[128, 3, 6, 128, 49][37632, 4816896, 6272, 1, 128]cuda:0" = torch.ops.aten.permute.default(view_101, [1, 0, 2, 4, 3]);  view_101 = None
        clone_89: "bf16[128, 3, 6, 128, 49][112896, 37632, 6272, 49, 1]cuda:0" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_102: "bf16[128, 2304, 7, 7][112896, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [128, 2304, 7, 7]);  clone_89 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(view_102, convert_element_type_131, convert_element_type_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_102 = convert_element_type_131 = convert_element_type_130 = None
        getitem_125: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = convolution_backward_15[0]
        getitem_126: "bf16[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_282: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
        convert_element_type_283: "f32[2304, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_126, torch.float32);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sum_30: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_282, [0, 2, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        sub_22: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_49)
        mul_184: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_18);  sub_22 = None
        unsqueeze_72: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_141, -1)
        unsqueeze_73: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_190: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_73);  mul_184 = unsqueeze_73 = None
        unsqueeze_74: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_142, -1);  primals_142 = None
        unsqueeze_75: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_129: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(mul_190, unsqueeze_75);  mul_190 = unsqueeze_75 = None
        convert_element_type_129: "bf16[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:462 in forward_features, code: x = self.pos_drop(x + self.pos_embed3)
        add_130: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_129, primals_143);  convert_element_type_129 = primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sub_68: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(add_130, unsqueeze_210);  add_130 = unsqueeze_210 = None
        mul_378: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_282, sub_68)
        sum_31: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_378, [0, 2, 3]);  mul_378 = None
        mul_379: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
        unsqueeze_211: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
        unsqueeze_212: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_380: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
        mul_381: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_382: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
        unsqueeze_214: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
        unsqueeze_215: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_383: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_147);  primals_147 = None
        unsqueeze_217: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
        unsqueeze_218: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_384: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_216);  sub_68 = unsqueeze_216 = None
        sub_70: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_282, mul_384);  convert_element_type_282 = mul_384 = None
        sub_71: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_213);  sub_70 = unsqueeze_213 = None
        mul_385: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_219);  sub_71 = unsqueeze_219 = None
        mul_386: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_58);  sum_31 = squeeze_58 = None
        add_203: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_202, mul_385);  add_202 = mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:462 in forward_features, code: x = self.pos_drop(x + self.pos_embed3)
        convert_element_type_284: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_203, torch.bfloat16)
        sum_32: "f32[1, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_203, [0], True, dtype = torch.float32);  add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        convert_element_type_285: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_284, torch.float32);  convert_element_type_284 = None
        squeeze_54: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        unsqueeze_220: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_221: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
        unsqueeze_222: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
        sum_33: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_285, [0, 2, 3])
        convert_element_type_128: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_72: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_128, unsqueeze_222);  convert_element_type_128 = unsqueeze_222 = None
        mul_387: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_285, sub_72)
        sum_34: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 2, 3]);  mul_387 = None
        mul_388: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 0.00015943877551020407)
        unsqueeze_223: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
        unsqueeze_224: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_389: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 0.00015943877551020407)
        squeeze_55: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_390: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_391: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
        unsqueeze_226: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
        unsqueeze_227: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_392: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_141);  primals_141 = None
        unsqueeze_229: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
        unsqueeze_230: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_393: "f32[128, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_228);  sub_72 = unsqueeze_228 = None
        sub_74: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_285, mul_393);  convert_element_type_285 = mul_393 = None
        sub_75: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_225);  sub_74 = unsqueeze_225 = None
        mul_394: "f32[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_231);  sub_75 = unsqueeze_231 = None
        mul_395: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, squeeze_55);  sum_34 = squeeze_55 = None
        convert_element_type_287: "bf16[128, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_394, torch.bfloat16);  mul_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_35: "bf16[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_287, [0, 2, 3])
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_287, convert_element_type_127, convert_element_type_126, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_287 = convert_element_type_127 = convert_element_type_126 = None
        getitem_128: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_16[0]
        getitem_129: "bf16[768, 384, 2, 2][1536, 1, 768, 384]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_288: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_128, torch.float32);  getitem_128 = None
        convert_element_type_289: "f32[768, 384, 2, 2][1536, 1, 768, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None
        convert_element_type_290: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_35, torch.float32);  sum_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        convert_element_type_291: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_288, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_291, convert_element_type_123, convert_element_type_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_291 = convert_element_type_123 = convert_element_type_124 = None
        getitem_131: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_17[0]
        getitem_132: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_292: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_132, torch.float32);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_293: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
        convert_element_type_122: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32);  convolution_38 = None
        mul_182: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, 0.7071067811865476)
        erf_17: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
        add_123: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_397: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_123, 0.5);  add_123 = None
        mul_398: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, convert_element_type_122)
        mul_399: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_398, -0.5);  mul_398 = None
        exp_12: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(mul_399);  mul_399 = None
        mul_400: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_401: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, mul_400);  convert_element_type_122 = mul_400 = None
        add_205: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_397, mul_401);  mul_397 = mul_401 = None
        mul_402: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_293, add_205);  convert_element_type_293 = add_205 = None
        convert_element_type_295: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_402, torch.bfloat16);  mul_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_295, convert_element_type_121, convert_element_type_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_295 = convert_element_type_121 = convert_element_type_120 = None
        getitem_134: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_18[0]
        getitem_135: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_296: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_134, torch.float32);  getitem_134 = None
        convert_element_type_297: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_36: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_296, [0, 2, 3])
        mul_403: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_296, sub_76)
        sum_37: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 2, 3]);  mul_403 = None
        mul_404: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 3.985969387755102e-05)
        unsqueeze_235: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
        unsqueeze_236: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_405: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 3.985969387755102e-05)
        mul_406: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_407: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_405, mul_406);  mul_405 = mul_406 = None
        unsqueeze_238: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
        unsqueeze_239: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_408: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_132);  primals_132 = None
        unsqueeze_241: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
        unsqueeze_242: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_409: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_240);  sub_76 = unsqueeze_240 = None
        sub_78: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_296, mul_409);  convert_element_type_296 = mul_409 = None
        sub_79: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_237);  sub_78 = unsqueeze_237 = None
        mul_410: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_243);  sub_79 = unsqueeze_243 = None
        mul_411: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_52);  sum_37 = squeeze_52 = None
        add_206: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_288, mul_410);  convert_element_type_288 = mul_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        convert_element_type_298: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_206, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_298, view_31, convert_element_type_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_298 = view_31 = convert_element_type_119 = None
        getitem_137: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_19[0]
        getitem_138: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_299: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_138, torch.float32);  getitem_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        view_103: "bf16[128, 6, 64, 196][75264, 64, 1, 384]cuda:0" = torch.ops.aten.reshape.default(getitem_137, [128, 6, 64, 196]);  getitem_137 = None
        permute_57: "bf16[128, 6, 196, 64][75264, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_103, [0, 1, 3, 2]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_90: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_104: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [768, 196, 64]);  clone_90 = None
        bmm_32: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_58, view_104);  permute_58 = None
        bmm_33: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.bmm.default(view_104, permute_59);  view_104 = permute_59 = None
        view_105: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [128, 6, 196, 64]);  bmm_32 = None
        view_106: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [128, 6, 196, 196]);  bmm_33 = None
        convert_element_type_304: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_106, torch.float32);  view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        slice_tensor_9: "bf16[768, 196, 200][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_5, 1, 0, -4);  bmm_default_5 = None
        slice_tensor_10: "bf16[768, 196, 196][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_9, 2, 0, -4);  slice_tensor_9 = None
        view_27: "bf16[128, 6, 196, 196][240000, 40000, 200, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_10, [128, 6, 196, 196]);  slice_tensor_10 = None

        # No stacktrace found for following nodes
        mul_tensor_16: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_27, 0.125)
        convert_element_type_default_8: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.float32);  mul_tensor_16 = None
        convert_element_type_default_9: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        mul_tensor_17: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1);  convert_element_type_default_9 = None
        sub_tensor_8: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_17, amax_default_8);  mul_tensor_17 = amax_default_8 = None
        mul_tensor_18: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_8, 0.125);  sub_tensor_8 = None
        sub_tensor_9: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_8, amax_default_9);  convert_element_type_default_8 = amax_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_4: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_9, mul_tensor_18, sub_tensor_9);  logical_not_default_9 = mul_tensor_18 = sub_tensor_9 = None
        exp_3: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.exp.default(where_self_4);  where_self_4 = None
        div_3: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        mul_412: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_304, div_3);  convert_element_type_304 = None
        sum_38: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_412, [-1], True)
        neg_4: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_4: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_38, mul_412);  neg_4 = sum_38 = mul_412 = None
        convert_element_type_305: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_413: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_305, 0.125);  convert_element_type_305 = None
        view_107: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(mul_413, [768, 196, 196]);  mul_413 = None
        bmm_34: "bf16[768, 64, 196][12544, 196, 1]cuda:0" = torch.ops.aten.bmm.default(permute_60, view_107);  permute_60 = None
        bmm_35: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_107, permute_61);  view_107 = permute_61 = None
        view_108: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [128, 6, 64, 196]);  bmm_34 = None
        view_109: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [128, 6, 196, 64]);  bmm_35 = None
        permute_62: "bf16[128, 6, 196, 64][75264, 12544, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_108, [0, 1, 3, 2]);  view_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        cat_4: "bf16[384, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.cat.default([view_109, permute_62, view_105]);  view_109 = permute_62 = view_105 = None
        view_110: "bf16[3, 128, 6, 196, 64][9633792, 75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [3, 128, 6, 196, 64]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        permute_63: "bf16[128, 3, 6, 64, 196][75264, 9633792, 12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_110, [1, 0, 2, 4, 3]);  view_110 = None
        clone_91: "bf16[128, 3, 6, 64, 196][225792, 75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        view_111: "bf16[128, 1152, 14, 14][225792, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [128, 1152, 14, 14]);  clone_91 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_111, convert_element_type_112, convert_element_type_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_111 = convert_element_type_112 = convert_element_type_111 = None
        getitem_140: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_20[0]
        getitem_141: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_310: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_140, torch.float32);  getitem_140 = None
        convert_element_type_311: "f32[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sum_39: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_310, [0, 2, 3])
        mul_414: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_310, sub_80)
        sum_40: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 2, 3]);  mul_414 = None
        mul_415: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 3.985969387755102e-05)
        unsqueeze_247: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_415, 0);  mul_415 = None
        unsqueeze_248: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_416: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 3.985969387755102e-05)
        mul_417: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_418: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, mul_417);  mul_416 = mul_417 = None
        unsqueeze_250: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
        unsqueeze_251: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_419: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_125);  primals_125 = None
        unsqueeze_253: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
        unsqueeze_254: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_420: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_252);  sub_80 = unsqueeze_252 = None
        sub_82: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_310, mul_420);  convert_element_type_310 = mul_420 = None
        sub_83: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_249);  sub_82 = unsqueeze_249 = None
        mul_421: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_255);  sub_83 = unsqueeze_255 = None
        mul_422: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, squeeze_49);  sum_40 = squeeze_49 = None
        add_207: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_206, mul_421);  add_206 = mul_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        convert_element_type_312: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_207, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_312, convert_element_type_109, convert_element_type_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_312 = convert_element_type_109 = convert_element_type_110 = None
        getitem_143: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_21[0]
        getitem_144: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_313: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_314: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_143, torch.float32);  getitem_143 = None
        convert_element_type_108: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        mul_164: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_108, 0.7071067811865476)
        erf_16: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
        add_110: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_424: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_110, 0.5);  add_110 = None
        mul_425: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_108, convert_element_type_108)
        mul_426: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_425, -0.5);  mul_425 = None
        exp_13: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(mul_426);  mul_426 = None
        mul_427: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_428: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_108, mul_427);  convert_element_type_108 = mul_427 = None
        add_209: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_424, mul_428);  mul_424 = mul_428 = None
        mul_429: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_314, add_209);  convert_element_type_314 = add_209 = None
        convert_element_type_316: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_429, torch.bfloat16);  mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_316, convert_element_type_107, convert_element_type_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_316 = convert_element_type_107 = convert_element_type_106 = None
        getitem_146: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_22[0]
        getitem_147: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_317: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_146, torch.float32);  getitem_146 = None
        convert_element_type_318: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_41: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_317, [0, 2, 3])
        mul_430: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_317, sub_84)
        sum_42: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 2, 3]);  mul_430 = None
        mul_431: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 3.985969387755102e-05)
        unsqueeze_259: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
        unsqueeze_260: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_432: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 3.985969387755102e-05)
        mul_433: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_434: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
        unsqueeze_262: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
        unsqueeze_263: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_435: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_118);  primals_118 = None
        unsqueeze_265: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
        unsqueeze_266: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_436: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_264);  sub_84 = unsqueeze_264 = None
        sub_86: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_317, mul_436);  convert_element_type_317 = mul_436 = None
        sub_87: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_261);  sub_86 = unsqueeze_261 = None
        mul_437: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_267);  sub_87 = unsqueeze_267 = None
        mul_438: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, squeeze_46);  sum_42 = squeeze_46 = None
        add_210: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_207, mul_437);  add_207 = mul_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        convert_element_type_319: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_319, view_23, convert_element_type_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_319 = view_23 = convert_element_type_105 = None
        getitem_149: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_23[0]
        getitem_150: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_320: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_150, torch.float32);  getitem_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        view_112: "bf16[128, 6, 64, 196][75264, 64, 1, 384]cuda:0" = torch.ops.aten.reshape.default(getitem_149, [128, 6, 64, 196]);  getitem_149 = None
        permute_64: "bf16[128, 6, 196, 64][75264, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 2]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_92: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_113: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [768, 196, 64]);  clone_92 = None
        bmm_36: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_65, view_113);  permute_65 = None
        bmm_37: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.bmm.default(view_113, permute_66);  view_113 = permute_66 = None
        view_114: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [128, 6, 196, 64]);  bmm_36 = None
        view_115: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [128, 6, 196, 196]);  bmm_37 = None
        convert_element_type_325: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_115, torch.float32);  view_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        slice_tensor_12: "bf16[768, 196, 200][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_7, 1, 0, -4);  bmm_default_7 = None
        slice_tensor_13: "bf16[768, 196, 196][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_12, 2, 0, -4);  slice_tensor_12 = None
        view_19: "bf16[128, 6, 196, 196][240000, 40000, 200, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_13, [128, 6, 196, 196]);  slice_tensor_13 = None

        # No stacktrace found for following nodes
        mul_tensor_20: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_19, 0.125)
        convert_element_type_default_10: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.float32);  mul_tensor_20 = None
        convert_element_type_default_11: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_tensor_21: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1);  convert_element_type_default_11 = None
        sub_tensor_10: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_21, amax_default_10);  mul_tensor_21 = amax_default_10 = None
        mul_tensor_22: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_10, 0.125);  sub_tensor_10 = None
        sub_tensor_11: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_10, amax_default_11);  convert_element_type_default_10 = amax_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_5: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_11, mul_tensor_22, sub_tensor_11);  logical_not_default_11 = mul_tensor_22 = sub_tensor_11 = None
        exp_2: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.exp.default(where_self_5);  where_self_5 = None
        div_2: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        mul_439: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_325, div_2);  convert_element_type_325 = None
        sum_43: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_439, [-1], True)
        neg_5: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_5: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_43, mul_439);  neg_5 = sum_43 = mul_439 = None
        convert_element_type_326: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_440: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_326, 0.125);  convert_element_type_326 = None
        view_116: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(mul_440, [768, 196, 196]);  mul_440 = None
        bmm_38: "bf16[768, 64, 196][12544, 196, 1]cuda:0" = torch.ops.aten.bmm.default(permute_67, view_116);  permute_67 = None
        bmm_39: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_116, permute_68);  view_116 = permute_68 = None
        view_117: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [128, 6, 64, 196]);  bmm_38 = None
        view_118: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [128, 6, 196, 64]);  bmm_39 = None
        permute_69: "bf16[128, 6, 196, 64][75264, 12544, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_117, [0, 1, 3, 2]);  view_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        cat_5: "bf16[384, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.cat.default([view_118, permute_69, view_114]);  view_118 = permute_69 = view_114 = None
        view_119: "bf16[3, 128, 6, 196, 64][9633792, 75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [3, 128, 6, 196, 64]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        permute_70: "bf16[128, 3, 6, 64, 196][75264, 9633792, 12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_119, [1, 0, 2, 4, 3]);  view_119 = None
        clone_93: "bf16[128, 3, 6, 64, 196][225792, 75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_120: "bf16[128, 1152, 14, 14][225792, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [128, 1152, 14, 14]);  clone_93 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(view_120, convert_element_type_98, convert_element_type_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_120 = convert_element_type_98 = convert_element_type_97 = None
        getitem_152: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_24[0]
        getitem_153: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_331: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_152, torch.float32);  getitem_152 = None
        convert_element_type_332: "f32[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sum_44: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_331, [0, 2, 3])
        mul_441: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_331, sub_88)
        sum_45: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 2, 3]);  mul_441 = None
        mul_442: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
        unsqueeze_271: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
        unsqueeze_272: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_443: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
        mul_444: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_445: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None
        unsqueeze_274: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
        unsqueeze_275: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        mul_446: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_111);  primals_111 = None
        unsqueeze_277: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
        unsqueeze_278: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_447: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_276);  sub_88 = unsqueeze_276 = None
        sub_90: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, mul_447);  convert_element_type_331 = mul_447 = None
        sub_91: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_273);  sub_90 = unsqueeze_273 = None
        mul_448: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_279);  sub_91 = unsqueeze_279 = None
        mul_449: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_43);  sum_45 = squeeze_43 = None
        add_211: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_210, mul_448);  add_210 = mul_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        convert_element_type_333: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_211, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_333, convert_element_type_95, convert_element_type_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_333 = convert_element_type_95 = convert_element_type_96 = None
        getitem_155: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_25[0]
        getitem_156: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_334: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_156, torch.float32);  getitem_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_335: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_155, torch.float32);  getitem_155 = None
        convert_element_type_94: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        mul_146: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.7071067811865476)
        erf_15: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
        add_97: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_451: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
        mul_452: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_94, convert_element_type_94)
        mul_453: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_452, -0.5);  mul_452 = None
        exp_14: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(mul_453);  mul_453 = None
        mul_454: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_455: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_94, mul_454);  convert_element_type_94 = mul_454 = None
        add_213: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_451, mul_455);  mul_451 = mul_455 = None
        mul_456: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_335, add_213);  convert_element_type_335 = add_213 = None
        convert_element_type_337: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_456, torch.bfloat16);  mul_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_337, convert_element_type_93, convert_element_type_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_337 = convert_element_type_93 = convert_element_type_92 = None
        getitem_158: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_26[0]
        getitem_159: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_338: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_158, torch.float32);  getitem_158 = None
        convert_element_type_339: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_46: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_338, [0, 2, 3])
        mul_457: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_338, sub_92)
        sum_47: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3]);  mul_457 = None
        mul_458: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 3.985969387755102e-05)
        unsqueeze_283: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
        unsqueeze_284: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_459: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
        mul_460: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_461: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
        unsqueeze_286: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
        unsqueeze_287: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        mul_462: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_104);  primals_104 = None
        unsqueeze_289: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
        unsqueeze_290: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_463: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_288);  sub_92 = unsqueeze_288 = None
        sub_94: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_338, mul_463);  convert_element_type_338 = mul_463 = None
        sub_95: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_285);  sub_94 = unsqueeze_285 = None
        mul_464: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_291);  sub_95 = unsqueeze_291 = None
        mul_465: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_40);  sum_47 = squeeze_40 = None
        add_214: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_211, mul_464);  add_211 = mul_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        convert_element_type_340: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_340, view_15, convert_element_type_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_340 = view_15 = convert_element_type_91 = None
        getitem_161: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_27[0]
        getitem_162: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_341: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_162, torch.float32);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        view_121: "bf16[128, 6, 64, 196][75264, 64, 1, 384]cuda:0" = torch.ops.aten.reshape.default(getitem_161, [128, 6, 64, 196]);  getitem_161 = None
        permute_71: "bf16[128, 6, 196, 64][75264, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_121, [0, 1, 3, 2]);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_94: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_122: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [768, 196, 64]);  clone_94 = None
        bmm_40: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_72, view_122);  permute_72 = None
        bmm_41: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.bmm.default(view_122, permute_73);  view_122 = permute_73 = None
        view_123: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [128, 6, 196, 64]);  bmm_40 = None
        view_124: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [128, 6, 196, 196]);  bmm_41 = None
        convert_element_type_346: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_124, torch.float32);  view_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        slice_tensor_15: "bf16[768, 196, 200][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_9, 1, 0, -4);  bmm_default_9 = None
        slice_tensor_16: "bf16[768, 196, 196][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_15, 2, 0, -4);  slice_tensor_15 = None
        view_11: "bf16[128, 6, 196, 196][240000, 40000, 200, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_16, [128, 6, 196, 196]);  slice_tensor_16 = None

        # No stacktrace found for following nodes
        mul_tensor_24: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_11, 0.125)
        convert_element_type_default_12: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.float32);  mul_tensor_24 = None
        convert_element_type_default_13: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_11, torch.float32);  view_11 = None
        mul_tensor_25: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_13, 1);  convert_element_type_default_13 = None
        sub_tensor_12: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_25, amax_default_12);  mul_tensor_25 = amax_default_12 = None
        mul_tensor_26: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_12, 0.125);  sub_tensor_12 = None
        sub_tensor_13: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_12, amax_default_13);  convert_element_type_default_12 = amax_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_6: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_13, mul_tensor_26, sub_tensor_13);  logical_not_default_13 = mul_tensor_26 = sub_tensor_13 = None
        exp_1: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.exp.default(where_self_6);  where_self_6 = None
        div_1: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        mul_466: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_346, div_1);  convert_element_type_346 = None
        sum_48: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_466, [-1], True)
        neg_6: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_6: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_48, mul_466);  neg_6 = sum_48 = mul_466 = None
        convert_element_type_347: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_467: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_347, 0.125);  convert_element_type_347 = None
        view_125: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(mul_467, [768, 196, 196]);  mul_467 = None
        bmm_42: "bf16[768, 64, 196][12544, 196, 1]cuda:0" = torch.ops.aten.bmm.default(permute_74, view_125);  permute_74 = None
        bmm_43: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_125, permute_75);  view_125 = permute_75 = None
        view_126: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [128, 6, 64, 196]);  bmm_42 = None
        view_127: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [128, 6, 196, 64]);  bmm_43 = None
        permute_76: "bf16[128, 6, 196, 64][75264, 12544, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_126, [0, 1, 3, 2]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        cat_6: "bf16[384, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.cat.default([view_127, permute_76, view_123]);  view_127 = permute_76 = view_123 = None
        view_128: "bf16[3, 128, 6, 196, 64][9633792, 75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [3, 128, 6, 196, 64]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        permute_77: "bf16[128, 3, 6, 64, 196][75264, 9633792, 12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_128, [1, 0, 2, 4, 3]);  view_128 = None
        clone_95: "bf16[128, 3, 6, 64, 196][225792, 75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_129: "bf16[128, 1152, 14, 14][225792, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [128, 1152, 14, 14]);  clone_95 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(view_129, convert_element_type_84, convert_element_type_83, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_129 = convert_element_type_84 = convert_element_type_83 = None
        getitem_164: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_28[0]
        getitem_165: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_352: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_164, torch.float32);  getitem_164 = None
        convert_element_type_353: "f32[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sum_49: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_352, [0, 2, 3])
        mul_468: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_352, sub_96)
        sum_50: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_468, [0, 2, 3]);  mul_468 = None
        mul_469: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
        unsqueeze_295: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
        unsqueeze_296: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_470: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
        mul_471: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_472: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
        unsqueeze_298: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
        unsqueeze_299: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        mul_473: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_97);  primals_97 = None
        unsqueeze_301: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
        unsqueeze_302: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_474: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_300);  sub_96 = unsqueeze_300 = None
        sub_98: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_352, mul_474);  convert_element_type_352 = mul_474 = None
        sub_99: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_297);  sub_98 = unsqueeze_297 = None
        mul_475: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_303);  sub_99 = unsqueeze_303 = None
        mul_476: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, squeeze_37);  sum_50 = squeeze_37 = None
        add_215: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_214, mul_475);  add_214 = mul_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        convert_element_type_354: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_354, convert_element_type_81, convert_element_type_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_354 = convert_element_type_81 = convert_element_type_82 = None
        getitem_167: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = convolution_backward_29[0]
        getitem_168: "bf16[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_355: "f32[384, 1536, 1, 1][1536, 1, 1536, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_168, torch.float32);  getitem_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_356: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_167, torch.float32);  getitem_167 = None
        convert_element_type_80: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        mul_128: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, 0.7071067811865476)
        erf_14: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
        add_84: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_478: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
        mul_479: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, convert_element_type_80)
        mul_480: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(mul_479, -0.5);  mul_479 = None
        exp_15: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.exp.default(mul_480);  mul_480 = None
        mul_481: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_482: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, mul_481);  convert_element_type_80 = mul_481 = None
        add_217: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.add.Tensor(mul_478, mul_482);  mul_478 = mul_482 = None
        mul_483: "f32[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_356, add_217);  convert_element_type_356 = add_217 = None
        convert_element_type_358: "bf16[128, 1536, 14, 14][301056, 1, 21504, 1536]cuda:0" = torch.ops.prims.convert_element_type.default(mul_483, torch.bfloat16);  mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_358, convert_element_type_79, convert_element_type_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_358 = convert_element_type_79 = convert_element_type_78 = None
        getitem_170: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_30[0]
        getitem_171: "bf16[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_359: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_170, torch.float32);  getitem_170 = None
        convert_element_type_360: "f32[1536, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_51: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_359, [0, 2, 3])
        mul_484: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_359, sub_100)
        sum_52: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_484, [0, 2, 3]);  mul_484 = None
        mul_485: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 3.985969387755102e-05)
        unsqueeze_307: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
        unsqueeze_308: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_486: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 3.985969387755102e-05)
        mul_487: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_488: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, mul_487);  mul_486 = mul_487 = None
        unsqueeze_310: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
        unsqueeze_311: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        mul_489: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_90);  primals_90 = None
        unsqueeze_313: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
        unsqueeze_314: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_490: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_312);  sub_100 = unsqueeze_312 = None
        sub_102: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_359, mul_490);  convert_element_type_359 = mul_490 = None
        sub_103: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_309);  sub_102 = unsqueeze_309 = None
        mul_491: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_315);  sub_103 = unsqueeze_315 = None
        mul_492: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, squeeze_34);  sum_52 = squeeze_34 = None
        add_218: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_215, mul_491);  add_215 = mul_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        convert_element_type_361: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_218, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:122 in forward, code: x = self.proj(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_361, view_7, convert_element_type_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_361 = view_7 = convert_element_type_77 = None
        getitem_173: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_31[0]
        getitem_174: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_362: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_174, torch.float32);  getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:121 in forward, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
        view_130: "bf16[128, 6, 64, 196][75264, 64, 1, 384]cuda:0" = torch.ops.aten.reshape.default(getitem_173, [128, 6, 64, 196]);  getitem_173 = None
        permute_78: "bf16[128, 6, 196, 64][75264, 64, 384, 1]cuda:0" = torch.ops.aten.permute.default(view_130, [0, 1, 3, 2]);  view_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:119 in forward, code: x = attn @ v
        clone_96: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_78, memory_format = torch.contiguous_format);  permute_78 = None
        view_131: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [768, 196, 64]);  clone_96 = None
        bmm_44: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_79, view_131);  permute_79 = None
        bmm_45: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.bmm.default(view_131, permute_80);  view_131 = permute_80 = None
        view_132: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [128, 6, 196, 64]);  bmm_44 = None
        view_133: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [128, 6, 196, 196]);  bmm_45 = None
        convert_element_type_367: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        slice_tensor_18: "bf16[768, 196, 200][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(bmm_default_11, 1, 0, -4);  bmm_default_11 = None
        slice_tensor_19: "bf16[768, 196, 196][40000, 200, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_tensor_18, 2, 0, -4);  slice_tensor_18 = None
        view_3: "bf16[128, 6, 196, 196][240000, 40000, 200, 1]cuda:0" = torch.ops.aten.reshape.default(slice_tensor_19, [128, 6, 196, 196]);  slice_tensor_19 = None

        # No stacktrace found for following nodes
        mul_tensor_28: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_3, 0.125)
        convert_element_type_default_14: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.float32);  mul_tensor_28 = None
        convert_element_type_default_15: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        mul_tensor_29: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_15, 1);  convert_element_type_default_15 = None
        sub_tensor_14: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_29, amax_default_14);  mul_tensor_29 = amax_default_14 = None
        mul_tensor_30: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_14, 0.125);  sub_tensor_14 = None
        sub_tensor_15: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_14, amax_default_15);  convert_element_type_default_14 = amax_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:117 in forward, code: attn = attn.softmax(dim=-1)
        where_self_7: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_15, mul_tensor_30, sub_tensor_15);  logical_not_default_15 = mul_tensor_30 = sub_tensor_15 = None
        exp: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.exp.default(where_self_7);  where_self_7 = None
        div: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_493: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_367, div);  convert_element_type_367 = None
        sum_53: "f32[128, 6, 196, 1][1176, 196, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_493, [-1], True)
        neg_7: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.neg.default(div);  div = None
        fma_7: "f32[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_53, mul_493);  neg_7 = sum_53 = mul_493 = None
        convert_element_type_368: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:116 in forward, code: attn = (q @ k.transpose(-2, -1)) * self.scale
        mul_494: "bf16[128, 6, 196, 196][230496, 38416, 196, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_368, 0.125);  convert_element_type_368 = None
        view_134: "bf16[768, 196, 196][38416, 196, 1]cuda:0" = torch.ops.aten.reshape.default(mul_494, [768, 196, 196]);  mul_494 = None
        bmm_46: "bf16[768, 64, 196][12544, 196, 1]cuda:0" = torch.ops.aten.bmm.default(permute_81, view_134);  permute_81 = None
        bmm_47: "bf16[768, 196, 64][12544, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_134, permute_82);  view_134 = permute_82 = None
        view_135: "bf16[128, 6, 64, 196][75264, 12544, 196, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [128, 6, 64, 196]);  bmm_46 = None
        view_136: "bf16[128, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [128, 6, 196, 64]);  bmm_47 = None
        permute_83: "bf16[128, 6, 196, 64][75264, 12544, 1, 196]cuda:0" = torch.ops.aten.permute.default(view_135, [0, 1, 3, 2]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:108 in forward, code: q, k, v = x.unbind(0)
        cat_7: "bf16[384, 6, 196, 64][75264, 12544, 64, 1]cuda:0" = torch.ops.aten.cat.default([view_136, permute_83, view_132]);  view_136 = permute_83 = view_132 = None
        view_137: "bf16[3, 128, 6, 196, 64][9633792, 75264, 12544, 64, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [3, 128, 6, 196, 64]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:107 in forward, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
        permute_84: "bf16[128, 3, 6, 64, 196][75264, 9633792, 12544, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_137, [1, 0, 2, 4, 3]);  view_137 = None
        clone_97: "bf16[128, 3, 6, 64, 196][225792, 75264, 12544, 196, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_138: "bf16[128, 1152, 14, 14][225792, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [128, 1152, 14, 14]);  clone_97 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_138, convert_element_type_70, convert_element_type_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_138 = convert_element_type_70 = convert_element_type_69 = None
        getitem_176: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_32[0]
        getitem_177: "bf16[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_373: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_176, torch.float32);  getitem_176 = None
        convert_element_type_374: "f32[1152, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sum_54: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_373, [0, 2, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        sub_9: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, getitem_19)
        mul_105: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        unsqueeze_36: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_77, -1)
        unsqueeze_37: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_111: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_37);  mul_105 = unsqueeze_37 = None
        unsqueeze_38: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
        unsqueeze_39: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_71: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_39);  mul_111 = unsqueeze_39 = None
        convert_element_type_68: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:452 in forward_features, code: x = self.pos_drop(x + self.pos_embed2)
        add_72: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_68, primals_79);  convert_element_type_68 = primals_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:176 in forward, code: x = x + self.drop_path(self.attn(self.norm1(x)))
        sub_104: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(add_72, unsqueeze_318);  add_72 = unsqueeze_318 = None
        mul_495: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_373, sub_104)
        sum_55: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_495, [0, 2, 3]);  mul_495 = None
        mul_496: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 3.985969387755102e-05)
        unsqueeze_319: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
        unsqueeze_320: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_497: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 3.985969387755102e-05)
        mul_498: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_499: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_497, mul_498);  mul_497 = mul_498 = None
        unsqueeze_322: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
        unsqueeze_323: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        mul_500: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_83);  primals_83 = None
        unsqueeze_325: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
        unsqueeze_326: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_501: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_324);  sub_104 = unsqueeze_324 = None
        sub_106: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_373, mul_501);  convert_element_type_373 = mul_501 = None
        sub_107: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_321);  sub_106 = unsqueeze_321 = None
        mul_502: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_327);  sub_107 = unsqueeze_327 = None
        mul_503: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_31);  sum_55 = squeeze_31 = None
        add_219: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_218, mul_502);  add_218 = mul_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:452 in forward_features, code: x = self.pos_drop(x + self.pos_embed2)
        convert_element_type_375: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16)
        sum_56: "f32[1, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_219, [0], True, dtype = torch.float32);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        convert_element_type_376: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_375, torch.float32);  convert_element_type_375 = None
        squeeze_27: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        unsqueeze_328: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_329: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
        unsqueeze_330: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
        sum_57: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_376, [0, 2, 3])
        convert_element_type_67: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32);  convolution_23 = None
        sub_108: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_67, unsqueeze_330);  convert_element_type_67 = unsqueeze_330 = None
        mul_504: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_376, sub_108)
        sum_58: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 2, 3]);  mul_504 = None
        mul_505: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 3.985969387755102e-05)
        unsqueeze_331: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
        unsqueeze_332: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_506: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 3.985969387755102e-05)
        squeeze_28: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_507: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_508: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, mul_507);  mul_506 = mul_507 = None
        unsqueeze_334: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
        unsqueeze_335: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
        mul_509: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_77);  primals_77 = None
        unsqueeze_337: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_509, 0);  mul_509 = None
        unsqueeze_338: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_510: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_336);  sub_108 = unsqueeze_336 = None
        sub_110: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_376, mul_510);  convert_element_type_376 = mul_510 = None
        sub_111: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_333);  sub_110 = unsqueeze_333 = None
        mul_511: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_339);  sub_111 = unsqueeze_339 = None
        mul_512: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, squeeze_28);  sum_58 = squeeze_28 = None
        convert_element_type_378: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_511, torch.bfloat16);  mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_59: "bf16[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_378, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_378, convert_element_type_66, convert_element_type_65, [384], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_378 = convert_element_type_66 = convert_element_type_65 = None
        getitem_179: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_33[0]
        getitem_180: "bf16[384, 192, 2, 2][768, 1, 384, 192]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_379: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_179, torch.float32);  getitem_179 = None
        convert_element_type_380: "f32[384, 192, 2, 2][768, 1, 384, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_180, torch.float32);  getitem_180 = None
        convert_element_type_381: "f32[384][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_59, torch.float32);  sum_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        convert_element_type_382: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_379, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_382, convert_element_type_62, convert_element_type_63, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_382 = convert_element_type_62 = convert_element_type_63 = None
        getitem_182: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_34[0]
        getitem_183: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_383: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_183, torch.float32);  getitem_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_384: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_182, torch.float32);  getitem_182 = None
        convert_element_type_61: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32);  convolution_21 = None
        mul_103: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, 0.7071067811865476)
        erf_13: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_65: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_514: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_65, 0.5);  add_65 = None
        mul_515: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, convert_element_type_61)
        mul_516: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_515, -0.5);  mul_515 = None
        exp_16: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_516);  mul_516 = None
        mul_517: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_518: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_61, mul_517);  convert_element_type_61 = mul_517 = None
        add_221: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_514, mul_518);  mul_514 = mul_518 = None
        mul_519: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_384, add_221);  convert_element_type_384 = add_221 = None
        convert_element_type_386: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_519, torch.bfloat16);  mul_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_386, convert_element_type_59, convert_element_type_60, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  convert_element_type_386 = convert_element_type_59 = convert_element_type_60 = None
        getitem_185: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_35[0]
        getitem_186: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_387: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_186, torch.float32);  getitem_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_388: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_185, torch.float32);  getitem_185 = None
        convert_element_type_58: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        mul_100: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, 0.7071067811865476)
        erf_12: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
        add_64: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_521: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
        mul_522: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, convert_element_type_58)
        mul_523: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_522, -0.5);  mul_522 = None
        exp_17: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_523);  mul_523 = None
        mul_524: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_525: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_58, mul_524);  convert_element_type_58 = mul_524 = None
        add_223: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_521, mul_525);  mul_521 = mul_525 = None
        mul_526: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_388, add_223);  convert_element_type_388 = add_223 = None
        convert_element_type_390: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_526, torch.bfloat16);  mul_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_390, convert_element_type_57, convert_element_type_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_390 = convert_element_type_57 = convert_element_type_56 = None
        getitem_188: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_36[0]
        getitem_189: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_391: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_188, torch.float32);  getitem_188 = None
        convert_element_type_392: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_189, torch.float32);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_60: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_391, [0, 2, 3])
        mul_527: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_391, sub_112)
        sum_61: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3]);  mul_527 = None
        mul_528: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 9.964923469387754e-06)
        unsqueeze_343: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
        unsqueeze_344: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_529: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 9.964923469387754e-06)
        mul_530: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_531: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
        unsqueeze_346: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
        unsqueeze_347: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        mul_532: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_67);  primals_67 = None
        unsqueeze_349: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
        unsqueeze_350: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_533: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_348);  sub_112 = unsqueeze_348 = None
        sub_114: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_391, mul_533);  convert_element_type_391 = mul_533 = None
        sub_115: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_345);  sub_114 = unsqueeze_345 = None
        mul_534: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_351);  sub_115 = unsqueeze_351 = None
        mul_535: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_25);  sum_61 = squeeze_25 = None
        add_224: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_379, mul_534);  convert_element_type_379 = mul_534 = None
        convert_element_type_393: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_224, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_393, convert_element_type_54, convert_element_type_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_393 = convert_element_type_54 = convert_element_type_55 = None
        getitem_191: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_37[0]
        getitem_192: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_394: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_192, torch.float32);  getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_395: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_191, torch.float32);  getitem_191 = None
        convert_element_type_53: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        mul_90: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, 0.7071067811865476)
        erf_11: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_57: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_537: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_57, 0.5);  add_57 = None
        mul_538: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, convert_element_type_53)
        mul_539: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_538, -0.5);  mul_538 = None
        exp_18: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_539);  mul_539 = None
        mul_540: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_541: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_53, mul_540);  convert_element_type_53 = mul_540 = None
        add_226: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_537, mul_541);  mul_537 = mul_541 = None
        mul_542: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_395, add_226);  convert_element_type_395 = add_226 = None
        convert_element_type_397: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_542, torch.bfloat16);  mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_397, convert_element_type_51, convert_element_type_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  convert_element_type_397 = convert_element_type_51 = convert_element_type_52 = None
        getitem_194: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_38[0]
        getitem_195: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_398: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_195, torch.float32);  getitem_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_399: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_194, torch.float32);  getitem_194 = None
        convert_element_type_50: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        mul_87: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_50, 0.7071067811865476)
        erf_10: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
        add_56: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_544: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_545: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_50, convert_element_type_50)
        mul_546: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_545, -0.5);  mul_545 = None
        exp_19: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_546);  mul_546 = None
        mul_547: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_548: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_50, mul_547);  convert_element_type_50 = mul_547 = None
        add_228: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_544, mul_548);  mul_544 = mul_548 = None
        mul_549: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_399, add_228);  convert_element_type_399 = add_228 = None
        convert_element_type_401: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_549, torch.bfloat16);  mul_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_401, convert_element_type_49, convert_element_type_48, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_401 = convert_element_type_49 = convert_element_type_48 = None
        getitem_197: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_39[0]
        getitem_198: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_402: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_197, torch.float32);  getitem_197 = None
        convert_element_type_403: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_198, torch.float32);  getitem_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_62: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_402, [0, 2, 3])
        mul_550: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_402, sub_116)
        sum_63: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 2, 3]);  mul_550 = None
        mul_551: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 9.964923469387754e-06)
        unsqueeze_355: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
        unsqueeze_356: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_552: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 9.964923469387754e-06)
        mul_553: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_554: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_552, mul_553);  mul_552 = mul_553 = None
        unsqueeze_358: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
        unsqueeze_359: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        mul_555: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_59);  primals_59 = None
        unsqueeze_361: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
        unsqueeze_362: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_556: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_360);  sub_116 = unsqueeze_360 = None
        sub_118: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_402, mul_556);  convert_element_type_402 = mul_556 = None
        sub_119: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_357);  sub_118 = unsqueeze_357 = None
        mul_557: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_363);  sub_119 = unsqueeze_363 = None
        mul_558: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_22);  sum_63 = squeeze_22 = None
        add_229: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_224, mul_557);  add_224 = mul_557 = None
        convert_element_type_404: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_229, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_404, convert_element_type_46, convert_element_type_47, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_404 = convert_element_type_46 = convert_element_type_47 = None
        getitem_200: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_40[0]
        getitem_201: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_405: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_406: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_200, torch.float32);  getitem_200 = None
        convert_element_type_45: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        mul_77: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, 0.7071067811865476)
        erf_9: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_49: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_560: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
        mul_561: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, convert_element_type_45)
        mul_562: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_561, -0.5);  mul_561 = None
        exp_20: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_562);  mul_562 = None
        mul_563: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_564: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_45, mul_563);  convert_element_type_45 = mul_563 = None
        add_231: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_560, mul_564);  mul_560 = mul_564 = None
        mul_565: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_406, add_231);  convert_element_type_406 = add_231 = None
        convert_element_type_408: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_565, torch.bfloat16);  mul_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_408, convert_element_type_43, convert_element_type_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  convert_element_type_408 = convert_element_type_43 = convert_element_type_44 = None
        getitem_203: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_41[0]
        getitem_204: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_409: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_204, torch.float32);  getitem_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_410: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_203, torch.float32);  getitem_203 = None
        convert_element_type_42: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        mul_74: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.7071067811865476)
        erf_8: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
        add_48: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_567: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_568: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, convert_element_type_42)
        mul_569: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, -0.5);  mul_568 = None
        exp_21: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_569);  mul_569 = None
        mul_570: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_571: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_42, mul_570);  convert_element_type_42 = mul_570 = None
        add_233: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_567, mul_571);  mul_567 = mul_571 = None
        mul_572: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_410, add_233);  convert_element_type_410 = add_233 = None
        convert_element_type_412: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_572, torch.bfloat16);  mul_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_412, convert_element_type_41, convert_element_type_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_412 = convert_element_type_41 = convert_element_type_40 = None
        getitem_206: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_42[0]
        getitem_207: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_413: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_206, torch.float32);  getitem_206 = None
        convert_element_type_414: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_64: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_413, [0, 2, 3])
        mul_573: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_413, sub_120)
        sum_65: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 2, 3]);  mul_573 = None
        mul_574: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 9.964923469387754e-06)
        unsqueeze_367: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
        unsqueeze_368: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_575: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 9.964923469387754e-06)
        mul_576: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_577: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
        unsqueeze_370: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
        unsqueeze_371: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        mul_578: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_51);  primals_51 = None
        unsqueeze_373: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
        unsqueeze_374: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_579: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_372);  sub_120 = unsqueeze_372 = None
        sub_122: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_413, mul_579);  convert_element_type_413 = mul_579 = None
        sub_123: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_369);  sub_122 = unsqueeze_369 = None
        mul_580: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_375);  sub_123 = unsqueeze_375 = None
        mul_581: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_19);  sum_65 = squeeze_19 = None
        add_234: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_229, mul_580);  add_229 = mul_580 = None
        convert_element_type_415: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_415, convert_element_type_38, convert_element_type_39, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_415 = convert_element_type_38 = convert_element_type_39 = None
        getitem_209: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_43[0]
        getitem_210: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_416: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_210, torch.float32);  getitem_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_417: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_209, torch.float32);  getitem_209 = None
        convert_element_type_37: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        mul_64: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, 0.7071067811865476)
        erf_7: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_41: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_583: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_584: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, convert_element_type_37)
        mul_585: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_584, -0.5);  mul_584 = None
        exp_22: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_585);  mul_585 = None
        mul_586: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_587: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_37, mul_586);  convert_element_type_37 = mul_586 = None
        add_236: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_583, mul_587);  mul_583 = mul_587 = None
        mul_588: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_417, add_236);  convert_element_type_417 = add_236 = None
        convert_element_type_419: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_588, torch.bfloat16);  mul_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_419, convert_element_type_35, convert_element_type_36, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  convert_element_type_419 = convert_element_type_35 = convert_element_type_36 = None
        getitem_212: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_44[0]
        getitem_213: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_420: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_213, torch.float32);  getitem_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_421: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_212, torch.float32);  getitem_212 = None
        convert_element_type_34: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        mul_61: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476)
        erf_6: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_40: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_590: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
        mul_591: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, convert_element_type_34)
        mul_592: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, -0.5);  mul_591 = None
        exp_23: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_592);  mul_592 = None
        mul_593: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_594: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, mul_593);  convert_element_type_34 = mul_593 = None
        add_238: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_590, mul_594);  mul_590 = mul_594 = None
        mul_595: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_421, add_238);  convert_element_type_421 = add_238 = None
        convert_element_type_423: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_595, torch.bfloat16);  mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(convert_element_type_423, convert_element_type_33, convert_element_type_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_423 = convert_element_type_33 = convert_element_type_32 = None
        getitem_215: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_45[0]
        getitem_216: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_424: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_215, torch.float32);  getitem_215 = None
        convert_element_type_425: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_216, torch.float32);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_66: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_424, [0, 2, 3])
        mul_596: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_424, sub_124)
        sum_67: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 2, 3]);  mul_596 = None
        mul_597: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 9.964923469387754e-06)
        unsqueeze_379: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
        unsqueeze_380: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_598: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 9.964923469387754e-06)
        mul_599: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_600: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_598, mul_599);  mul_598 = mul_599 = None
        unsqueeze_382: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
        unsqueeze_383: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        mul_601: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_43);  primals_43 = None
        unsqueeze_385: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
        unsqueeze_386: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_602: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_384);  sub_124 = unsqueeze_384 = None
        sub_126: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_424, mul_602);  convert_element_type_424 = mul_602 = None
        sub_127: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_381);  sub_126 = unsqueeze_381 = None
        mul_603: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_387);  sub_127 = unsqueeze_387 = None
        mul_604: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_16);  sum_67 = squeeze_16 = None
        add_239: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_234, mul_603);  add_234 = mul_603 = None
        convert_element_type_426: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_239, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_426, convert_element_type_30, convert_element_type_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_426 = convert_element_type_30 = convert_element_type_31 = None
        getitem_218: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_46[0]
        getitem_219: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_427: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_219, torch.float32);  getitem_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_428: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_218, torch.float32);  getitem_218 = None
        convert_element_type_29: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        mul_51: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, 0.7071067811865476)
        erf_5: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_33: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_606: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
        mul_607: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, convert_element_type_29)
        mul_608: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_607, -0.5);  mul_607 = None
        exp_24: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_608);  mul_608 = None
        mul_609: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
        mul_610: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_29, mul_609);  convert_element_type_29 = mul_609 = None
        add_241: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_606, mul_610);  mul_606 = mul_610 = None
        mul_611: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_428, add_241);  convert_element_type_428 = add_241 = None
        convert_element_type_430: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_611, torch.bfloat16);  mul_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(convert_element_type_430, convert_element_type_27, convert_element_type_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  convert_element_type_430 = convert_element_type_27 = convert_element_type_28 = None
        getitem_221: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_47[0]
        getitem_222: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_431: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_222, torch.float32);  getitem_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_432: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_221, torch.float32);  getitem_221 = None
        convert_element_type_26: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        mul_48: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, 0.7071067811865476)
        erf_4: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
        add_32: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_613: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
        mul_614: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, convert_element_type_26)
        mul_615: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_614, -0.5);  mul_614 = None
        exp_25: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_615);  mul_615 = None
        mul_616: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
        mul_617: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_26, mul_616);  convert_element_type_26 = mul_616 = None
        add_243: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_613, mul_617);  mul_613 = mul_617 = None
        mul_618: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_432, add_243);  convert_element_type_432 = add_243 = None
        convert_element_type_434: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_618, torch.bfloat16);  mul_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_434, convert_element_type_25, convert_element_type_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_434 = convert_element_type_25 = convert_element_type_24 = None
        getitem_224: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_48[0]
        getitem_225: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_435: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_224, torch.float32);  getitem_224 = None
        convert_element_type_436: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_225, torch.float32);  getitem_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_68: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_435, [0, 2, 3])
        mul_619: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_435, sub_128)
        sum_69: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3]);  mul_619 = None
        mul_620: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 9.964923469387754e-06)
        unsqueeze_391: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
        unsqueeze_392: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_621: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 9.964923469387754e-06)
        mul_622: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_623: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_621, mul_622);  mul_621 = mul_622 = None
        unsqueeze_394: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
        unsqueeze_395: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_624: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_35);  primals_35 = None
        unsqueeze_397: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
        unsqueeze_398: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_625: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_396);  sub_128 = unsqueeze_396 = None
        sub_130: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_435, mul_625);  convert_element_type_435 = mul_625 = None
        sub_131: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_393);  sub_130 = unsqueeze_393 = None
        mul_626: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_399);  sub_131 = unsqueeze_399 = None
        mul_627: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_13);  sum_69 = squeeze_13 = None
        add_244: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_239, mul_626);  add_239 = mul_626 = None
        convert_element_type_437: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(convert_element_type_437, convert_element_type_22, convert_element_type_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_437 = convert_element_type_22 = convert_element_type_23 = None
        getitem_227: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_49[0]
        getitem_228: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_438: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_228, torch.float32);  getitem_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_439: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_227, torch.float32);  getitem_227 = None
        convert_element_type_21: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        mul_38: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_21, 0.7071067811865476)
        erf_3: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_25: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_629: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_25, 0.5);  add_25 = None
        mul_630: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_21, convert_element_type_21)
        mul_631: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, -0.5);  mul_630 = None
        exp_26: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_631);  mul_631 = None
        mul_632: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_633: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_21, mul_632);  convert_element_type_21 = mul_632 = None
        add_246: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_629, mul_633);  mul_629 = mul_633 = None
        mul_634: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_439, add_246);  convert_element_type_439 = add_246 = None
        convert_element_type_441: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_634, torch.bfloat16);  mul_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_441, convert_element_type_19, convert_element_type_20, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  convert_element_type_441 = convert_element_type_19 = convert_element_type_20 = None
        getitem_230: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_50[0]
        getitem_231: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_442: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_443: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_230, torch.float32);  getitem_230 = None
        convert_element_type_18: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        mul_35: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.7071067811865476)
        erf_2: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
        add_24: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_636: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_637: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, convert_element_type_18)
        mul_638: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_637, -0.5);  mul_637 = None
        exp_27: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_638);  mul_638 = None
        mul_639: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_640: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_18, mul_639);  convert_element_type_18 = mul_639 = None
        add_248: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_636, mul_640);  mul_636 = mul_640 = None
        mul_641: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_443, add_248);  convert_element_type_443 = add_248 = None
        convert_element_type_445: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_641, torch.bfloat16);  mul_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(convert_element_type_445, convert_element_type_17, convert_element_type_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_445 = convert_element_type_17 = convert_element_type_16 = None
        getitem_233: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_51[0]
        getitem_234: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_446: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_233, torch.float32);  getitem_233 = None
        convert_element_type_447: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_234, torch.float32);  getitem_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_70: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_446, [0, 2, 3])
        mul_642: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_446, sub_132)
        sum_71: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_642, [0, 2, 3]);  mul_642 = None
        mul_643: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 9.964923469387754e-06)
        unsqueeze_403: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
        unsqueeze_404: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_644: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 9.964923469387754e-06)
        mul_645: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_646: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, mul_645);  mul_644 = mul_645 = None
        unsqueeze_406: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
        unsqueeze_407: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        mul_647: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_27);  primals_27 = None
        unsqueeze_409: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
        unsqueeze_410: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_648: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_408);  sub_132 = unsqueeze_408 = None
        sub_134: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_446, mul_648);  convert_element_type_446 = mul_648 = None
        sub_135: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_405);  sub_134 = unsqueeze_405 = None
        mul_649: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_411);  sub_135 = unsqueeze_411 = None
        mul_650: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_10);  sum_71 = squeeze_10 = None
        add_249: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_244, mul_649);  add_244 = mul_649 = None
        convert_element_type_448: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:73 in forward, code: x = self.conv3(x)
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(convert_element_type_448, convert_element_type_14, convert_element_type_15, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_448 = convert_element_type_14 = convert_element_type_15 = None
        getitem_236: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_52[0]
        getitem_237: "bf16[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        convert_element_type_449: "f32[192, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:72 in forward, code: x = self.act2(x)
        convert_element_type_450: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_236, torch.float32);  getitem_236 = None
        convert_element_type_13: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        mul_25: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.7071067811865476)
        erf_1: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_17: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_652: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_17, 0.5);  add_17 = None
        mul_653: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, convert_element_type_13)
        mul_654: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_653, -0.5);  mul_653 = None
        exp_28: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_654);  mul_654 = None
        mul_655: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_656: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_13, mul_655);  convert_element_type_13 = mul_655 = None
        add_251: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_652, mul_656);  mul_652 = mul_656 = None
        mul_657: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_450, add_251);  convert_element_type_450 = add_251 = None
        convert_element_type_452: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_657, torch.bfloat16);  mul_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:71 in forward, code: x = self.conv2(x)
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(convert_element_type_452, convert_element_type_11, convert_element_type_12, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  convert_element_type_452 = convert_element_type_11 = convert_element_type_12 = None
        getitem_239: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = convolution_backward_53[0]
        getitem_240: "bf16[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_453: "f32[384, 48, 3, 3][432, 1, 144, 48]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_240, torch.float32);  getitem_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:68 in forward, code: x = self.act1(x)
        convert_element_type_454: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_239, torch.float32);  getitem_239 = None
        convert_element_type_10: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        mul_22: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.7071067811865476)
        erf: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
        add_16: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_659: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
        mul_660: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, convert_element_type_10)
        mul_661: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_660, -0.5);  mul_660 = None
        exp_29: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.exp.default(mul_661);  mul_661 = None
        mul_662: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
        mul_663: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_662);  convert_element_type_10 = mul_662 = None
        add_253: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_659, mul_663);  mul_659 = mul_663 = None
        mul_664: "f32[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_454, add_253);  convert_element_type_454 = add_253 = None
        convert_element_type_456: "bf16[128, 384, 28, 28][301056, 1, 10752, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_664, torch.bfloat16);  mul_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:67 in forward, code: x = self.conv1(x)
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(convert_element_type_456, convert_element_type_9, convert_element_type_8, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_456 = convert_element_type_9 = convert_element_type_8 = None
        getitem_242: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_54[0]
        getitem_243: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None
        convert_element_type_457: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_242, torch.float32);  getitem_242 = None
        convert_element_type_458: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_243, torch.float32);  getitem_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sum_72: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_457, [0, 2, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        sub_1: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        unsqueeze_4: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1)
        unsqueeze_5: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_14, -1);  primals_14 = None
        unsqueeze_7: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_7: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:442 in forward_features, code: x = self.pos_drop(x + self.pos_embed1)
        add_10: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_7, primals_15);  convert_element_type_7 = primals_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:177 in forward, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
        sub_136: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(add_10, unsqueeze_414);  add_10 = unsqueeze_414 = None
        mul_665: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_457, sub_136)
        sum_73: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_665, [0, 2, 3]);  mul_665 = None
        mul_666: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 9.964923469387754e-06)
        unsqueeze_415: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_416: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_667: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 9.964923469387754e-06)
        mul_668: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_669: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_667, mul_668);  mul_667 = mul_668 = None
        unsqueeze_418: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
        unsqueeze_419: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        mul_670: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_19);  primals_19 = None
        unsqueeze_421: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_670, 0);  mul_670 = None
        unsqueeze_422: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_671: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_420);  sub_136 = unsqueeze_420 = None
        sub_138: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_457, mul_671);  convert_element_type_457 = mul_671 = None
        sub_139: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_417);  sub_138 = unsqueeze_417 = None
        mul_672: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_423);  sub_139 = unsqueeze_423 = None
        mul_673: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_7);  sum_73 = squeeze_7 = None
        add_254: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_249, mul_672);  add_249 = mul_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:442 in forward_features, code: x = self.pos_drop(x + self.pos_embed1)
        convert_element_type_459: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16)
        sum_74: "f32[1, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_254, [0], True, dtype = torch.float32);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        convert_element_type_460: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_459, torch.float32);  convert_element_type_459 = None
        squeeze_3: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        unsqueeze_424: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_425: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
        unsqueeze_426: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
        sum_75: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_460, [0, 2, 3])
        convert_element_type_6: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_140: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_6, unsqueeze_426);  convert_element_type_6 = unsqueeze_426 = None
        mul_674: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_460, sub_140)
        sum_76: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_674, [0, 2, 3]);  mul_674 = None
        mul_675: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 9.964923469387754e-06)
        unsqueeze_427: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_428: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_676: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 9.964923469387754e-06)
        squeeze_4: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_677: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_678: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
        unsqueeze_430: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
        unsqueeze_431: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        mul_679: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_13);  primals_13 = None
        unsqueeze_433: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_679, 0);  mul_679 = None
        unsqueeze_434: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_680: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_432);  sub_140 = unsqueeze_432 = None
        sub_142: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_460, mul_680);  convert_element_type_460 = mul_680 = None
        sub_143: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_429);  sub_142 = unsqueeze_429 = None
        mul_681: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_435);  sub_143 = unsqueeze_435 = None
        mul_682: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, squeeze_4);  sum_76 = squeeze_4 = None
        convert_element_type_462: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_681, torch.bfloat16);  mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_77: "bf16[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_462, [0, 2, 3])
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(convert_element_type_462, relu, convert_element_type_5, [192], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_462 = convert_element_type_5 = None
        getitem_245: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = convolution_backward_55[0]
        getitem_246: "bf16[192, 32, 4, 4][512, 1, 128, 32]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_463: "f32[192, 32, 4, 4][512, 1, 128, 32]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_246, torch.float32);  getitem_246 = None
        convert_element_type_464: "f32[192][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_77, torch.float32);  sum_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/visformer.py:437 in forward_features, code: x = self.stem(x)
        le: "b8[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.where.self(le, full_default, getitem_245);  le = full_default = getitem_245 = None
        convert_element_type_465: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        sum_78: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_465, [0, 2, 3])
        convert_element_type_2: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_144: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_438);  convert_element_type_2 = unsqueeze_438 = None
        mul_683: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_465, sub_144)
        sum_79: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 2, 3]);  mul_683 = None
        mul_684: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 6.228077168367346e-07)
        unsqueeze_439: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_440: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_685: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 6.228077168367346e-07)
        mul_686: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_687: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_685, mul_686);  mul_685 = mul_686 = None
        unsqueeze_442: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
        unsqueeze_443: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        mul_688: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_445: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
        unsqueeze_446: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_689: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_444);  sub_144 = unsqueeze_444 = None
        sub_146: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_465, mul_689);  convert_element_type_465 = mul_689 = None
        sub_147: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_441);  sub_146 = unsqueeze_441 = None
        mul_690: "f32[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_447);  sub_147 = unsqueeze_447 = None
        mul_691: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_1);  sum_79 = squeeze_1 = None
        convert_element_type_467: "bf16[128, 32, 112, 112][401408, 1, 3584, 32]cuda:0" = torch.ops.prims.convert_element_type.default(mul_690, torch.bfloat16);  mul_690 = None
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(convert_element_type_467, convert_element_type_1, convert_element_type, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_467 = convert_element_type_1 = convert_element_type = None
        getitem_249: "bf16[32, 3, 7, 7][147, 1, 21, 3]cuda:0" = convolution_backward_56[1];  convolution_backward_56 = None
        convert_element_type_468: "f32[32, 3, 7, 7][147, 1, 21, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_249, torch.float32);  getitem_249 = None
        return (convert_element_type_468, None, None, None, None, mul_691, sum_78, convert_element_type_463, convert_element_type_464, None, None, None, mul_682, sum_75, sum_74, None, None, None, mul_673, sum_72, convert_element_type_458, convert_element_type_453, convert_element_type_449, None, None, None, mul_650, sum_70, convert_element_type_447, convert_element_type_442, convert_element_type_438, None, None, None, mul_627, sum_68, convert_element_type_436, convert_element_type_431, convert_element_type_427, None, None, None, mul_604, sum_66, convert_element_type_425, convert_element_type_420, convert_element_type_416, None, None, None, mul_581, sum_64, convert_element_type_414, convert_element_type_409, convert_element_type_405, None, None, None, mul_558, sum_62, convert_element_type_403, convert_element_type_398, convert_element_type_394, None, None, None, mul_535, sum_60, convert_element_type_392, convert_element_type_387, convert_element_type_383, convert_element_type_380, convert_element_type_381, None, None, None, mul_512, sum_57, sum_56, None, None, None, mul_503, sum_54, convert_element_type_374, convert_element_type_362, None, None, None, mul_492, sum_51, convert_element_type_360, convert_element_type_355, None, None, None, mul_476, sum_49, convert_element_type_353, convert_element_type_341, None, None, None, mul_465, sum_46, convert_element_type_339, convert_element_type_334, None, None, None, mul_449, sum_44, convert_element_type_332, convert_element_type_320, None, None, None, mul_438, sum_41, convert_element_type_318, convert_element_type_313, None, None, None, mul_422, sum_39, convert_element_type_311, convert_element_type_299, None, None, None, mul_411, sum_36, convert_element_type_297, convert_element_type_292, convert_element_type_289, convert_element_type_290, None, None, None, mul_395, sum_33, sum_32, None, None, None, mul_386, sum_30, convert_element_type_283, convert_element_type_271, None, None, None, mul_375, sum_27, convert_element_type_269, convert_element_type_264, None, None, None, mul_359, sum_25, convert_element_type_262, convert_element_type_250, None, None, None, mul_348, sum_22, convert_element_type_248, convert_element_type_243, None, None, None, mul_332, sum_20, convert_element_type_241, convert_element_type_229, None, None, None, mul_321, sum_17, convert_element_type_227, convert_element_type_222, None, None, None, mul_305, sum_15, convert_element_type_220, convert_element_type_208, None, None, None, mul_294, sum_12, convert_element_type_206, convert_element_type_201, None, None, None, mul_278, sum_10, convert_element_type_198, convert_element_type_199)
