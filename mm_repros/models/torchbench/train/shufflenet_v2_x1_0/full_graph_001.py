class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[24][1]cuda:0", primals_7: "f32[24][1]cuda:0", primals_12: "f32[24][1]cuda:0", primals_18: "f32[58][1]cuda:0", primals_19: "f32[58][1]cuda:0", primals_24: "f32[58][1]cuda:0", primals_30: "f32[58][1]cuda:0", primals_36: "f32[58][1]cuda:0", primals_37: "f32[58][1]cuda:0", primals_42: "f32[58][1]cuda:0", primals_48: "f32[58][1]cuda:0", primals_54: "f32[58][1]cuda:0", primals_55: "f32[58][1]cuda:0", primals_60: "f32[58][1]cuda:0", primals_66: "f32[58][1]cuda:0", primals_72: "f32[58][1]cuda:0", primals_73: "f32[58][1]cuda:0", primals_78: "f32[58][1]cuda:0", primals_84: "f32[58][1]cuda:0", primals_90: "f32[58][1]cuda:0", primals_91: "f32[58][1]cuda:0", primals_96: "f32[116][1]cuda:0", primals_102: "f32[116][1]cuda:0", primals_103: "f32[116][1]cuda:0", primals_108: "f32[116][1]cuda:0", primals_114: "f32[116][1]cuda:0", primals_120: "f32[116][1]cuda:0", primals_121: "f32[116][1]cuda:0", primals_126: "f32[116][1]cuda:0", primals_132: "f32[116][1]cuda:0", primals_138: "f32[116][1]cuda:0", primals_139: "f32[116][1]cuda:0", primals_144: "f32[116][1]cuda:0", primals_150: "f32[116][1]cuda:0", primals_156: "f32[116][1]cuda:0", primals_157: "f32[116][1]cuda:0", primals_162: "f32[116][1]cuda:0", primals_168: "f32[116][1]cuda:0", primals_174: "f32[116][1]cuda:0", primals_175: "f32[116][1]cuda:0", primals_180: "f32[116][1]cuda:0", primals_186: "f32[116][1]cuda:0", primals_192: "f32[116][1]cuda:0", primals_193: "f32[116][1]cuda:0", primals_198: "f32[116][1]cuda:0", primals_204: "f32[116][1]cuda:0", primals_210: "f32[116][1]cuda:0", primals_211: "f32[116][1]cuda:0", primals_216: "f32[116][1]cuda:0", primals_222: "f32[116][1]cuda:0", primals_228: "f32[116][1]cuda:0", primals_229: "f32[116][1]cuda:0", primals_234: "f32[116][1]cuda:0", primals_240: "f32[116][1]cuda:0", primals_246: "f32[116][1]cuda:0", primals_247: "f32[116][1]cuda:0", primals_252: "f32[232][1]cuda:0", primals_258: "f32[232][1]cuda:0", primals_259: "f32[232][1]cuda:0", primals_264: "f32[232][1]cuda:0", primals_270: "f32[232][1]cuda:0", primals_276: "f32[232][1]cuda:0", primals_277: "f32[232][1]cuda:0", primals_282: "f32[232][1]cuda:0", primals_288: "f32[232][1]cuda:0", primals_294: "f32[232][1]cuda:0", primals_295: "f32[232][1]cuda:0", primals_300: "f32[232][1]cuda:0", primals_306: "f32[232][1]cuda:0", primals_312: "f32[232][1]cuda:0", primals_313: "f32[232][1]cuda:0", primals_318: "f32[232][1]cuda:0", primals_324: "f32[232][1]cuda:0", primals_330: "f32[232][1]cuda:0", primals_331: "f32[232][1]cuda:0", primals_336: "f32[1024][1]cuda:0", primals_337: "f32[1024][1]cuda:0", convert_element_type: "bf16[24, 3, 3, 3][27, 1, 9, 3]cuda:0", convert_element_type_1: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0", getitem_1: "f32[1, 24, 1, 1][24, 1, 24, 24]cuda:0", rsqrt: "f32[1, 24, 1, 1][24, 1, 24, 24]cuda:0", getitem_2: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0", getitem_3: "i8[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0", convert_element_type_4: "bf16[24, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_1: "bf16[128, 24, 28, 28][18816, 1, 672, 24]cuda:0", squeeze_4: "f32[24][1]cuda:0", convert_element_type_6: "bf16[128, 24, 28, 28][18816, 1, 672, 24]cuda:0", convert_element_type_7: "bf16[58, 24, 1, 1][24, 1, 24, 24]cuda:0", convolution_2: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", getitem_7: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", rsqrt_2: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", convert_element_type_10: "bf16[58, 24, 1, 1][24, 1, 24, 24]cuda:0", convolution_3: "bf16[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0", squeeze_10: "f32[58][1]cuda:0", relu_2: "bf16[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0", convert_element_type_13: "bf16[58, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_4: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", squeeze_13: "f32[58][1]cuda:0", convert_element_type_15: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", convert_element_type_16: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0", convolution_5: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", getitem_13: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", rsqrt_5: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", getitem_15: "bf16[128, 58, 28, 28][45504, 784, 28, 1]cuda:0", convert_element_type_19: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0", convolution_6: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", squeeze_19: "f32[58][1]cuda:0", relu_4: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", convert_element_type_22: "bf16[58, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_7: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", squeeze_22: "f32[58][1]cuda:0", convert_element_type_24: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", convert_element_type_25: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0", convolution_8: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", getitem_21: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", rsqrt_8: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", getitem_23: "bf16[128, 58, 28, 28][45504, 784, 28, 1]cuda:0", convert_element_type_28: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0", convolution_9: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", squeeze_28: "f32[58][1]cuda:0", relu_6: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", convert_element_type_31: "bf16[58, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_10: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", squeeze_31: "f32[58][1]cuda:0", convert_element_type_33: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", convert_element_type_34: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0", convolution_11: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", getitem_29: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", rsqrt_11: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", getitem_31: "bf16[128, 58, 28, 28][45504, 784, 28, 1]cuda:0", convert_element_type_37: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0", convolution_12: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", squeeze_37: "f32[58][1]cuda:0", relu_8: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", convert_element_type_40: "bf16[58, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_13: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", squeeze_40: "f32[58][1]cuda:0", convert_element_type_42: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", convert_element_type_43: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0", convolution_14: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0", getitem_37: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", rsqrt_14: "f32[1, 58, 1, 1][58, 1, 58, 58]cuda:0", view_7: "bf16[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0", convert_element_type_46: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_15: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_46: "f32[116][1]cuda:0", convert_element_type_48: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_49: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_16: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", getitem_41: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", rsqrt_16: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", convert_element_type_52: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_17: "bf16[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0", squeeze_52: "f32[116][1]cuda:0", relu_11: "bf16[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0", convert_element_type_55: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_18: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_55: "f32[116][1]cuda:0", convert_element_type_57: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_58: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_19: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", getitem_47: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", rsqrt_19: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", getitem_49: "bf16[128, 116, 14, 14][45504, 196, 14, 1]cuda:0", convert_element_type_61: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_20: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_61: "f32[116][1]cuda:0", relu_13: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_64: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_21: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_64: "f32[116][1]cuda:0", convert_element_type_66: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_67: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_22: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", getitem_55: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", rsqrt_22: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", getitem_57: "bf16[128, 116, 14, 14][45504, 196, 14, 1]cuda:0", convert_element_type_70: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_23: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_70: "f32[116][1]cuda:0", relu_15: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_73: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_24: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_73: "f32[116][1]cuda:0", convert_element_type_75: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_76: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_25: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", getitem_63: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", rsqrt_25: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", getitem_65: "bf16[128, 116, 14, 14][45504, 196, 14, 1]cuda:0", convert_element_type_79: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_26: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_79: "f32[116][1]cuda:0", relu_17: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_82: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_27: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_82: "f32[116][1]cuda:0", convert_element_type_84: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_85: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_28: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", getitem_71: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", rsqrt_28: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", getitem_73: "bf16[128, 116, 14, 14][45504, 196, 14, 1]cuda:0", convert_element_type_88: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_29: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_88: "f32[116][1]cuda:0", relu_19: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_91: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_30: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_91: "f32[116][1]cuda:0", convert_element_type_93: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_94: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_31: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", getitem_79: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", rsqrt_31: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", getitem_81: "bf16[128, 116, 14, 14][45504, 196, 14, 1]cuda:0", convert_element_type_97: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_32: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_97: "f32[116][1]cuda:0", relu_21: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_100: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_33: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_100: "f32[116][1]cuda:0", convert_element_type_102: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_103: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_34: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", getitem_87: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", rsqrt_34: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", getitem_89: "bf16[128, 116, 14, 14][45504, 196, 14, 1]cuda:0", convert_element_type_106: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_35: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_106: "f32[116][1]cuda:0", relu_23: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_109: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_36: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_109: "f32[116][1]cuda:0", convert_element_type_111: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_112: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_37: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", getitem_95: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", rsqrt_37: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", getitem_97: "bf16[128, 116, 14, 14][45504, 196, 14, 1]cuda:0", convert_element_type_115: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_38: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_115: "f32[116][1]cuda:0", relu_25: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_118: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_39: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", squeeze_118: "f32[116][1]cuda:0", convert_element_type_120: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", convert_element_type_121: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0", convolution_40: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0", getitem_103: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", rsqrt_40: "f32[1, 116, 1, 1][116, 1, 116, 116]cuda:0", view_23: "bf16[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0", convert_element_type_124: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_41: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", squeeze_124: "f32[232][1]cuda:0", convert_element_type_126: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", convert_element_type_127: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0", convolution_42: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", getitem_107: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", rsqrt_42: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", convert_element_type_130: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0", convolution_43: "bf16[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0", squeeze_130: "f32[232][1]cuda:0", relu_28: "bf16[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0", convert_element_type_133: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_44: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", squeeze_133: "f32[232][1]cuda:0", convert_element_type_135: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", convert_element_type_136: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0", convolution_45: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", getitem_113: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", rsqrt_45: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", getitem_115: "bf16[128, 232, 7, 7][22784, 49, 7, 1]cuda:0", convert_element_type_139: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0", convolution_46: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", squeeze_139: "f32[232][1]cuda:0", relu_30: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", convert_element_type_142: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_47: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", squeeze_142: "f32[232][1]cuda:0", convert_element_type_144: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", convert_element_type_145: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0", convolution_48: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", getitem_121: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", rsqrt_48: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", getitem_123: "bf16[128, 232, 7, 7][22784, 49, 7, 1]cuda:0", convert_element_type_148: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0", convolution_49: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", squeeze_148: "f32[232][1]cuda:0", relu_32: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", convert_element_type_151: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_50: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", squeeze_151: "f32[232][1]cuda:0", convert_element_type_153: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", convert_element_type_154: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0", convolution_51: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", getitem_129: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", rsqrt_51: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", getitem_131: "bf16[128, 232, 7, 7][22784, 49, 7, 1]cuda:0", convert_element_type_157: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0", convolution_52: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", squeeze_157: "f32[232][1]cuda:0", relu_34: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", convert_element_type_160: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0", convolution_53: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", squeeze_160: "f32[232][1]cuda:0", convert_element_type_162: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", convert_element_type_163: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0", convolution_54: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0", getitem_137: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", rsqrt_54: "f32[1, 232, 1, 1][232, 1, 232, 232]cuda:0", view_31: "bf16[128, 464, 7, 7][22736, 1, 3248, 464]cuda:0", convert_element_type_166: "bf16[1024, 464, 1, 1][464, 1, 464, 464]cuda:0", convolution_55: "bf16[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0", getitem_139: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", rsqrt_55: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0", mean: "bf16[128, 1024][1024, 1]cuda:0", permute_17: "bf16[1000, 1024][1024, 1]cuda:0", unsqueeze_252: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0", unsqueeze_264: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0", unsqueeze_288: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0", unsqueeze_300: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0", unsqueeze_324: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0", unsqueeze_336: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0", unsqueeze_360: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0", unsqueeze_372: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0", unsqueeze_396: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0", unsqueeze_420: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_432: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_456: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_468: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_492: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_504: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_528: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_540: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_564: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_576: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_600: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_612: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_636: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_648: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_672: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_684: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_708: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0", unsqueeze_732: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0", unsqueeze_744: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0", unsqueeze_768: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0", unsqueeze_780: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0", unsqueeze_804: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0", unsqueeze_816: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0", unsqueeze_840: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0", unsqueeze_852: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0", unsqueeze_876: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:162 in _forward_impl, code: x = self.fc(x)
        mm: "bf16[128, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_17);  permute_17 = None
        permute_18: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_18, mean);  permute_18 = mean = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_32: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_178: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_32, torch.bfloat16);  view_32 = None
        convert_element_type_179: "f32[1000, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_180: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_178, torch.float32);  convert_element_type_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:161 in _forward_impl, code: x = x.mean([2, 3])  # globalpool
        unsqueeze_224: "bf16[128, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm, 2);  mm = None
        unsqueeze_225: "bf16[128, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        expand: "bf16[128, 1024, 7, 7][1024, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(unsqueeze_225, [128, 1024, 7, 7]);  unsqueeze_225 = None
        div: "bf16[128, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:160 in _forward_impl, code: x = self.conv5(x)
        sub_55: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, getitem_139)
        mul_385: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        unsqueeze_220: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_221: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_391: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
        unsqueeze_222: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_223: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_279: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
        convert_element_type_168: "bf16[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16);  add_279 = None
        relu_36: "bf16[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.relu.default(convert_element_type_168);  convert_element_type_168 = None
        le: "b8[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
        convert_element_type_181: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_165: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        unsqueeze_226: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
        unsqueeze_227: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        sum_2: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_181, [0, 2, 3])
        convert_element_type_167: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32);  convolution_55 = None
        sub_56: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_167, unsqueeze_228);  convert_element_type_167 = unsqueeze_228 = None
        mul_392: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_181, sub_56)
        sum_3: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 2, 3]);  mul_392 = None
        mul_393: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.00015943877551020407)
        unsqueeze_229: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_393, 0);  mul_393 = None
        unsqueeze_230: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_394: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.00015943877551020407)
        squeeze_166: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_395: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_396: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
        unsqueeze_232: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
        unsqueeze_233: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        mul_397: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, primals_336);  primals_336 = None
        unsqueeze_235: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
        unsqueeze_236: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_398: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_234);  sub_56 = unsqueeze_234 = None
        sub_58: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_181, mul_398);  convert_element_type_181 = mul_398 = None
        sub_59: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_231);  sub_58 = unsqueeze_231 = None
        mul_399: "f32[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_237);  sub_59 = unsqueeze_237 = None
        mul_400: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_166);  sum_3 = squeeze_166 = None
        convert_element_type_183: "bf16[128, 1024, 7, 7][50176, 1, 7168, 1024]cuda:0" = torch.ops.prims.convert_element_type.default(mul_399, torch.bfloat16);  mul_399 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_183, view_31, convert_element_type_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_183 = view_31 = convert_element_type_166 = None
        getitem_140: "bf16[128, 464, 7, 7][22736, 1, 3248, 464]cuda:0" = convolution_backward[0]
        getitem_141: "bf16[1024, 464, 1, 1][464, 1, 464, 464]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_184: "f32[1024, 464, 1, 1][464, 1, 464, 464]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_33: "bf16[128, 232, 2, 7, 7][22736, 2, 1, 3248, 464]cuda:0" = torch.ops.aten.reshape.default(getitem_140, [128, 232, 2, 7, 7]);  getitem_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_21: "bf16[128, 2, 232, 7, 7][22736, 1, 2, 3248, 464]cuda:0" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3, 4]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_16: "bf16[128, 2, 232, 7, 7][22736, 11368, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_34: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [128, 464, 7, 7]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_1: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_34, 1, 0, 232)
        slice_2: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_34, 1, 232, 464);  view_34 = None
        sub_54: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convolution_54, getitem_137)
        mul_378: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        unsqueeze_216: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_330, -1)
        unsqueeze_217: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        mul_384: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
        unsqueeze_218: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_331, -1);  primals_331 = None
        unsqueeze_219: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        add_274: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
        convert_element_type_165: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(add_274, torch.bfloat16);  add_274 = None
        relu_35: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.relu.default(convert_element_type_165);  convert_element_type_165 = None
        le_1: "b8[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_1: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.where.self(le_1, full_default, slice_2);  le_1 = slice_2 = None
        convert_element_type_185: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        squeeze_162: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        unsqueeze_238: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
        unsqueeze_239: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        sum_4: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_185, [0, 2, 3])
        convert_element_type_164: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32);  convolution_54 = None
        sub_60: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_164, unsqueeze_240);  convert_element_type_164 = unsqueeze_240 = None
        mul_401: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_185, sub_60)
        sum_5: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 2, 3]);  mul_401 = None
        mul_402: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.00015943877551020407)
        unsqueeze_241: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_402, 0);  mul_402 = None
        unsqueeze_242: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_403: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.00015943877551020407)
        squeeze_163: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_404: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_405: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
        unsqueeze_244: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
        unsqueeze_245: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        mul_406: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, primals_330);  primals_330 = None
        unsqueeze_247: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
        unsqueeze_248: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_407: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_246);  sub_60 = unsqueeze_246 = None
        sub_62: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_185, mul_407);  convert_element_type_185 = mul_407 = None
        sub_63: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_243);  sub_62 = unsqueeze_243 = None
        mul_408: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_249);  sub_63 = unsqueeze_249 = None
        mul_409: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_163);  sum_5 = squeeze_163 = None
        convert_element_type_187: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_408, torch.bfloat16);  mul_408 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_187, convert_element_type_162, convert_element_type_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_187 = convert_element_type_162 = convert_element_type_163 = None
        getitem_143: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_1[0]
        getitem_144: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_188: "f32[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None
        convert_element_type_189: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_143, torch.float32);  getitem_143 = None
        sum_6: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_189, [0, 2, 3])
        convert_element_type_161: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_53, torch.float32);  convolution_53 = None
        sub_64: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_161, unsqueeze_252);  convert_element_type_161 = unsqueeze_252 = None
        mul_410: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_189, sub_64)
        sum_7: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_410, [0, 2, 3]);  mul_410 = None
        mul_411: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.00015943877551020407)
        unsqueeze_253: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
        unsqueeze_254: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_412: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.00015943877551020407)
        mul_413: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_414: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_412, mul_413);  mul_412 = mul_413 = None
        unsqueeze_256: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
        unsqueeze_257: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
        unsqueeze_258: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
        mul_415: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, primals_324);  primals_324 = None
        unsqueeze_259: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_415, 0);  mul_415 = None
        unsqueeze_260: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_416: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_258);  sub_64 = unsqueeze_258 = None
        sub_66: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_189, mul_416);  convert_element_type_189 = mul_416 = None
        sub_67: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_255);  sub_66 = unsqueeze_255 = None
        mul_417: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_261);  sub_67 = unsqueeze_261 = None
        mul_418: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_160);  sum_7 = squeeze_160 = None
        convert_element_type_191: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_417, torch.bfloat16);  mul_417 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_191, relu_34, convert_element_type_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  convert_element_type_191 = convert_element_type_160 = None
        getitem_146: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_2[0]
        getitem_147: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_192: "f32[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
        le_2: "b8[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_2: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.where.self(le_2, full_default, getitem_146);  le_2 = getitem_146 = None
        convert_element_type_193: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sum_8: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_193, [0, 2, 3])
        convert_element_type_158: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32);  convolution_52 = None
        sub_68: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_158, unsqueeze_264);  convert_element_type_158 = unsqueeze_264 = None
        mul_419: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_193, sub_68)
        sum_9: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 2, 3]);  mul_419 = None
        mul_420: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 0.00015943877551020407)
        unsqueeze_265: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
        unsqueeze_266: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_421: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.00015943877551020407)
        mul_422: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_423: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
        unsqueeze_268: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
        unsqueeze_269: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
        unsqueeze_270: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
        mul_424: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, primals_318);  primals_318 = None
        unsqueeze_271: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_424, 0);  mul_424 = None
        unsqueeze_272: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_425: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_270);  sub_68 = unsqueeze_270 = None
        sub_70: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_193, mul_425);  convert_element_type_193 = mul_425 = None
        sub_71: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_267);  sub_70 = unsqueeze_267 = None
        mul_426: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_273);  sub_71 = unsqueeze_273 = None
        mul_427: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_157);  sum_9 = squeeze_157 = None
        convert_element_type_195: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_426, torch.bfloat16);  mul_426 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_195, getitem_131, convert_element_type_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_195 = getitem_131 = convert_element_type_157 = None
        getitem_149: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_3[0]
        getitem_150: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_196: "f32[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_150, torch.float32);  getitem_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_16: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([slice_1, getitem_149], 1);  slice_1 = getitem_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_35: "bf16[128, 232, 2, 7, 7][22736, 98, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(cat_16, [128, 232, 2, 7, 7]);  cat_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_22: "bf16[128, 2, 232, 7, 7][22736, 49, 98, 7, 1]cuda:0" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3, 4]);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_17: "bf16[128, 2, 232, 7, 7][22736, 11368, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_36: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [128, 464, 7, 7]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_3: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_36, 1, 0, 232)
        slice_4: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_36, 1, 232, 464);  view_36 = None
        sub_51: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_129)
        mul_357: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        unsqueeze_204: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_205: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_207: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_259: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        convert_element_type_156: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None
        relu_33: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.relu.default(convert_element_type_156);  convert_element_type_156 = None
        le_3: "b8[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_3: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.where.self(le_3, full_default, slice_4);  le_3 = slice_4 = None
        convert_element_type_197: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        squeeze_153: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        unsqueeze_274: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_275: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        sum_10: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_197, [0, 2, 3])
        convert_element_type_155: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32);  convolution_51 = None
        sub_72: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_155, unsqueeze_276);  convert_element_type_155 = unsqueeze_276 = None
        mul_428: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_197, sub_72)
        sum_11: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 2, 3]);  mul_428 = None
        mul_429: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.00015943877551020407)
        unsqueeze_277: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_429, 0);  mul_429 = None
        unsqueeze_278: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_430: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.00015943877551020407)
        squeeze_154: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_431: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_432: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, mul_431);  mul_430 = mul_431 = None
        unsqueeze_280: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
        unsqueeze_281: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
        unsqueeze_282: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
        mul_433: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_312);  primals_312 = None
        unsqueeze_283: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_433, 0);  mul_433 = None
        unsqueeze_284: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_434: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_282);  sub_72 = unsqueeze_282 = None
        sub_74: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_197, mul_434);  convert_element_type_197 = mul_434 = None
        sub_75: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_279);  sub_74 = unsqueeze_279 = None
        mul_435: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_285);  sub_75 = unsqueeze_285 = None
        mul_436: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_154);  sum_11 = squeeze_154 = None
        convert_element_type_199: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_435, torch.bfloat16);  mul_435 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_199, convert_element_type_153, convert_element_type_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_199 = convert_element_type_153 = convert_element_type_154 = None
        getitem_152: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_4[0]
        getitem_153: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_200: "f32[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None
        convert_element_type_201: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_152, torch.float32);  getitem_152 = None
        sum_12: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_201, [0, 2, 3])
        convert_element_type_152: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32);  convolution_50 = None
        sub_76: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_152, unsqueeze_288);  convert_element_type_152 = unsqueeze_288 = None
        mul_437: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_201, sub_76)
        sum_13: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
        mul_438: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.00015943877551020407)
        unsqueeze_289: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_438, 0);  mul_438 = None
        unsqueeze_290: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_439: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.00015943877551020407)
        mul_440: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_441: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_439, mul_440);  mul_439 = mul_440 = None
        unsqueeze_292: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
        unsqueeze_293: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
        unsqueeze_294: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
        mul_442: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_306);  primals_306 = None
        unsqueeze_295: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
        unsqueeze_296: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_443: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_294);  sub_76 = unsqueeze_294 = None
        sub_78: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_201, mul_443);  convert_element_type_201 = mul_443 = None
        sub_79: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_291);  sub_78 = unsqueeze_291 = None
        mul_444: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_297);  sub_79 = unsqueeze_297 = None
        mul_445: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_151);  sum_13 = squeeze_151 = None
        convert_element_type_203: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_444, torch.bfloat16);  mul_444 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_203, relu_32, convert_element_type_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  convert_element_type_203 = convert_element_type_151 = None
        getitem_155: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_5[0]
        getitem_156: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_204: "f32[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_156, torch.float32);  getitem_156 = None
        le_4: "b8[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_4: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.where.self(le_4, full_default, getitem_155);  le_4 = getitem_155 = None
        convert_element_type_205: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sum_14: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_205, [0, 2, 3])
        convert_element_type_149: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32);  convolution_49 = None
        sub_80: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_149, unsqueeze_300);  convert_element_type_149 = unsqueeze_300 = None
        mul_446: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_205, sub_80)
        sum_15: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3]);  mul_446 = None
        mul_447: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.00015943877551020407)
        unsqueeze_301: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
        unsqueeze_302: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_448: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.00015943877551020407)
        mul_449: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_450: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, mul_449);  mul_448 = mul_449 = None
        unsqueeze_304: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
        unsqueeze_305: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
        unsqueeze_306: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
        mul_451: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_300);  primals_300 = None
        unsqueeze_307: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
        unsqueeze_308: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_452: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_306);  sub_80 = unsqueeze_306 = None
        sub_82: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_205, mul_452);  convert_element_type_205 = mul_452 = None
        sub_83: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_303);  sub_82 = unsqueeze_303 = None
        mul_453: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_309);  sub_83 = unsqueeze_309 = None
        mul_454: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_148);  sum_15 = squeeze_148 = None
        convert_element_type_207: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_453, torch.bfloat16);  mul_453 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_207, getitem_123, convert_element_type_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_207 = getitem_123 = convert_element_type_148 = None
        getitem_158: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_6[0]
        getitem_159: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_208: "f32[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_17: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([slice_3, getitem_158], 1);  slice_3 = getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_37: "bf16[128, 232, 2, 7, 7][22736, 98, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(cat_17, [128, 232, 2, 7, 7]);  cat_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_23: "bf16[128, 2, 232, 7, 7][22736, 49, 98, 7, 1]cuda:0" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3, 4]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_18: "bf16[128, 2, 232, 7, 7][22736, 11368, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_38: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [128, 464, 7, 7]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_5: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_38, 1, 0, 232)
        slice_6: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_38, 1, 232, 464);  view_38 = None
        sub_48: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convolution_48, getitem_121)
        mul_336: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        unsqueeze_192: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_193: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_295, -1);  primals_295 = None
        unsqueeze_195: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_244: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        convert_element_type_147: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16);  add_244 = None
        relu_31: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.relu.default(convert_element_type_147);  convert_element_type_147 = None
        le_5: "b8[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_5: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.where.self(le_5, full_default, slice_6);  le_5 = slice_6 = None
        convert_element_type_209: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        squeeze_144: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        unsqueeze_310: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_311: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        sum_16: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_209, [0, 2, 3])
        convert_element_type_146: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_48, torch.float32);  convolution_48 = None
        sub_84: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_146, unsqueeze_312);  convert_element_type_146 = unsqueeze_312 = None
        mul_455: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_209, sub_84)
        sum_17: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 2, 3]);  mul_455 = None
        mul_456: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.00015943877551020407)
        unsqueeze_313: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
        unsqueeze_314: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_457: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 0.00015943877551020407)
        squeeze_145: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_458: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_459: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
        unsqueeze_316: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
        unsqueeze_317: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
        unsqueeze_318: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
        mul_460: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_294);  primals_294 = None
        unsqueeze_319: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_460, 0);  mul_460 = None
        unsqueeze_320: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_461: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_318);  sub_84 = unsqueeze_318 = None
        sub_86: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_209, mul_461);  convert_element_type_209 = mul_461 = None
        sub_87: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_315);  sub_86 = unsqueeze_315 = None
        mul_462: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_321);  sub_87 = unsqueeze_321 = None
        mul_463: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_145);  sum_17 = squeeze_145 = None
        convert_element_type_211: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_462, torch.bfloat16);  mul_462 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_211, convert_element_type_144, convert_element_type_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_211 = convert_element_type_144 = convert_element_type_145 = None
        getitem_161: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_7[0]
        getitem_162: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_212: "f32[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_162, torch.float32);  getitem_162 = None
        convert_element_type_213: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_161, torch.float32);  getitem_161 = None
        sum_18: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_213, [0, 2, 3])
        convert_element_type_143: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32);  convolution_47 = None
        sub_88: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_143, unsqueeze_324);  convert_element_type_143 = unsqueeze_324 = None
        mul_464: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_213, sub_88)
        sum_19: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None
        mul_465: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.00015943877551020407)
        unsqueeze_325: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
        unsqueeze_326: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_466: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.00015943877551020407)
        mul_467: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_468: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        unsqueeze_328: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
        unsqueeze_329: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
        unsqueeze_330: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
        mul_469: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_288);  primals_288 = None
        unsqueeze_331: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
        unsqueeze_332: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_470: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_330);  sub_88 = unsqueeze_330 = None
        sub_90: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_213, mul_470);  convert_element_type_213 = mul_470 = None
        sub_91: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_327);  sub_90 = unsqueeze_327 = None
        mul_471: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_333);  sub_91 = unsqueeze_333 = None
        mul_472: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_142);  sum_19 = squeeze_142 = None
        convert_element_type_215: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_471, torch.bfloat16);  mul_471 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_215, relu_30, convert_element_type_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  convert_element_type_215 = convert_element_type_142 = None
        getitem_164: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_8[0]
        getitem_165: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_216: "f32[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None
        le_6: "b8[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_6: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.where.self(le_6, full_default, getitem_164);  le_6 = getitem_164 = None
        convert_element_type_217: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sum_20: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_217, [0, 2, 3])
        convert_element_type_140: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32);  convolution_46 = None
        sub_92: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_140, unsqueeze_336);  convert_element_type_140 = unsqueeze_336 = None
        mul_473: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_217, sub_92)
        sum_21: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3]);  mul_473 = None
        mul_474: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.00015943877551020407)
        unsqueeze_337: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
        unsqueeze_338: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_475: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.00015943877551020407)
        mul_476: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_477: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_475, mul_476);  mul_475 = mul_476 = None
        unsqueeze_340: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
        unsqueeze_341: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
        unsqueeze_342: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
        mul_478: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_282);  primals_282 = None
        unsqueeze_343: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
        unsqueeze_344: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_479: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_342);  sub_92 = unsqueeze_342 = None
        sub_94: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_217, mul_479);  convert_element_type_217 = mul_479 = None
        sub_95: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_339);  sub_94 = unsqueeze_339 = None
        mul_480: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_345);  sub_95 = unsqueeze_345 = None
        mul_481: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_139);  sum_21 = squeeze_139 = None
        convert_element_type_219: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_480, torch.bfloat16);  mul_480 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_219, getitem_115, convert_element_type_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_219 = getitem_115 = convert_element_type_139 = None
        getitem_167: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_9[0]
        getitem_168: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        convert_element_type_220: "f32[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_168, torch.float32);  getitem_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_18: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([slice_5, getitem_167], 1);  slice_5 = getitem_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_39: "bf16[128, 232, 2, 7, 7][22736, 98, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(cat_18, [128, 232, 2, 7, 7]);  cat_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_24: "bf16[128, 2, 232, 7, 7][22736, 49, 98, 7, 1]cuda:0" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3, 4]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_19: "bf16[128, 2, 232, 7, 7][22736, 11368, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_40: "bf16[128, 464, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [128, 464, 7, 7]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        slice_7: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_40, 1, 0, 232)
        slice_8: "bf16[128, 232, 7, 7][22736, 49, 7, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_40, 1, 232, 464);  view_40 = None
        sub_45: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_113)
        mul_315: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        unsqueeze_180: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_181: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_321: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        unsqueeze_182: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_183: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_229: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        convert_element_type_138: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(add_229, torch.bfloat16);  add_229 = None
        relu_29: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.relu.default(convert_element_type_138);  convert_element_type_138 = None
        le_7: "b8[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_7: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.where.self(le_7, full_default, slice_8);  le_7 = slice_8 = None
        convert_element_type_221: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        squeeze_135: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        unsqueeze_346: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_347: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        sum_22: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_221, [0, 2, 3])
        convert_element_type_137: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32);  convolution_45 = None
        sub_96: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_137, unsqueeze_348);  convert_element_type_137 = unsqueeze_348 = None
        mul_482: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_221, sub_96)
        sum_23: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3]);  mul_482 = None
        mul_483: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 0.00015943877551020407)
        unsqueeze_349: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
        unsqueeze_350: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_484: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 0.00015943877551020407)
        squeeze_136: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_485: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_486: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
        unsqueeze_352: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
        unsqueeze_353: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
        unsqueeze_354: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
        mul_487: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_276);  primals_276 = None
        unsqueeze_355: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
        unsqueeze_356: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_488: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_354);  sub_96 = unsqueeze_354 = None
        sub_98: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_221, mul_488);  convert_element_type_221 = mul_488 = None
        sub_99: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_351);  sub_98 = unsqueeze_351 = None
        mul_489: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_357);  sub_99 = unsqueeze_357 = None
        mul_490: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_136);  sum_23 = squeeze_136 = None
        convert_element_type_223: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_489, torch.bfloat16);  mul_489 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_223, convert_element_type_135, convert_element_type_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_223 = convert_element_type_135 = convert_element_type_136 = None
        getitem_170: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_10[0]
        getitem_171: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_224: "f32[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None
        convert_element_type_225: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_170, torch.float32);  getitem_170 = None
        sum_24: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_225, [0, 2, 3])
        convert_element_type_134: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        sub_100: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_134, unsqueeze_360);  convert_element_type_134 = unsqueeze_360 = None
        mul_491: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_225, sub_100)
        sum_25: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
        mul_492: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 0.00015943877551020407)
        unsqueeze_361: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
        unsqueeze_362: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_493: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 0.00015943877551020407)
        mul_494: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_495: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
        unsqueeze_364: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
        unsqueeze_365: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
        mul_496: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_270);  primals_270 = None
        unsqueeze_367: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
        unsqueeze_368: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_497: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_366);  sub_100 = unsqueeze_366 = None
        sub_102: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_225, mul_497);  convert_element_type_225 = mul_497 = None
        sub_103: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_363);  sub_102 = unsqueeze_363 = None
        mul_498: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_369);  sub_103 = unsqueeze_369 = None
        mul_499: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_133);  sum_25 = squeeze_133 = None
        convert_element_type_227: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_498, torch.bfloat16);  mul_498 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_227, relu_28, convert_element_type_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  convert_element_type_227 = convert_element_type_133 = None
        getitem_173: "bf16[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = convolution_backward_11[0]
        getitem_174: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_228: "f32[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_174, torch.float32);  getitem_174 = None
        le_8: "b8[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_8: "bf16[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.aten.where.self(le_8, full_default, getitem_173);  le_8 = getitem_173 = None
        convert_element_type_229: "f32[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        sum_26: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_229, [0, 2, 3])
        convert_element_type_131: "f32[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        sub_104: "f32[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_131, unsqueeze_372);  convert_element_type_131 = unsqueeze_372 = None
        mul_500: "f32[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_229, sub_104)
        sum_27: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
        mul_501: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 3.985969387755102e-05)
        unsqueeze_373: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
        unsqueeze_374: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_502: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 3.985969387755102e-05)
        mul_503: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_504: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_502, mul_503);  mul_502 = mul_503 = None
        unsqueeze_376: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
        unsqueeze_377: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        mul_505: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_264);  primals_264 = None
        unsqueeze_379: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
        unsqueeze_380: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_506: "f32[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_378);  sub_104 = unsqueeze_378 = None
        sub_106: "f32[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_229, mul_506);  convert_element_type_229 = mul_506 = None
        sub_107: "f32[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_375);  sub_106 = unsqueeze_375 = None
        mul_507: "f32[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_381);  sub_107 = unsqueeze_381 = None
        mul_508: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_130);  sum_27 = squeeze_130 = None
        convert_element_type_231: "bf16[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_507, torch.bfloat16);  mul_507 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_231, view_23, convert_element_type_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_231 = convert_element_type_130 = None
        getitem_176: "bf16[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = convolution_backward_12[0]
        getitem_177: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_232: "f32[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None
        sub_42: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, getitem_107)
        mul_294: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        unsqueeze_168: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_258, -1)
        unsqueeze_169: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_300: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        unsqueeze_170: "f32[232, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
        unsqueeze_171: "f32[232, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_214: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        convert_element_type_129: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None
        relu_27: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.relu.default(convert_element_type_129);  convert_element_type_129 = None
        le_9: "b8[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_9: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.where.self(le_9, full_default, slice_7);  le_9 = slice_7 = None
        convert_element_type_233: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        squeeze_126: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        unsqueeze_382: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_383: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        sum_28: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_233, [0, 2, 3])
        convert_element_type_128: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        sub_108: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_128, unsqueeze_384);  convert_element_type_128 = unsqueeze_384 = None
        mul_509: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_233, sub_108)
        sum_29: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 2, 3]);  mul_509 = None
        mul_510: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
        unsqueeze_385: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
        unsqueeze_386: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_511: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 0.00015943877551020407)
        squeeze_127: "f32[232][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_512: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_513: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
        unsqueeze_388: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
        unsqueeze_389: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        mul_514: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_258);  primals_258 = None
        unsqueeze_391: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
        unsqueeze_392: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_515: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_390);  sub_108 = unsqueeze_390 = None
        sub_110: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_233, mul_515);  convert_element_type_233 = mul_515 = None
        sub_111: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_387);  sub_110 = unsqueeze_387 = None
        mul_516: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_393);  sub_111 = unsqueeze_393 = None
        mul_517: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_127);  sum_29 = squeeze_127 = None
        convert_element_type_235: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_516, torch.bfloat16);  mul_516 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_235, convert_element_type_126, convert_element_type_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_235 = convert_element_type_126 = convert_element_type_127 = None
        getitem_179: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = convolution_backward_13[0]
        getitem_180: "bf16[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_236: "f32[232, 232, 1, 1][232, 1, 232, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_180, torch.float32);  getitem_180 = None
        convert_element_type_237: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_179, torch.float32);  getitem_179 = None
        sum_30: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_237, [0, 2, 3])
        convert_element_type_125: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_112: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_125, unsqueeze_396);  convert_element_type_125 = unsqueeze_396 = None
        mul_518: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_237, sub_112)
        sum_31: "f32[232][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3]);  mul_518 = None
        mul_519: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
        unsqueeze_397: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
        unsqueeze_398: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_520: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
        mul_521: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_522: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_520, mul_521);  mul_520 = mul_521 = None
        unsqueeze_400: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
        unsqueeze_401: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
        unsqueeze_402: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
        mul_523: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_252);  primals_252 = None
        unsqueeze_403: "f32[1, 232][232, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
        unsqueeze_404: "f32[1, 232, 1][232, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 232, 1, 1][232, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_524: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_402);  sub_112 = unsqueeze_402 = None
        sub_114: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_237, mul_524);  convert_element_type_237 = mul_524 = None
        sub_115: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_399);  sub_114 = unsqueeze_399 = None
        mul_525: "f32[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_405);  sub_115 = unsqueeze_405 = None
        mul_526: "f32[232][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_124);  sum_31 = squeeze_124 = None
        convert_element_type_239: "bf16[128, 232, 7, 7][11368, 1, 1624, 232]cuda:0" = torch.ops.prims.convert_element_type.default(mul_525, torch.bfloat16);  mul_525 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_239, view_23, convert_element_type_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  convert_element_type_239 = view_23 = convert_element_type_124 = None
        getitem_182: "bf16[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = convolution_backward_14[0]
        getitem_183: "bf16[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        add_280: "bf16[128, 232, 14, 14][45472, 1, 3248, 232]cuda:0" = torch.ops.aten.add.Tensor(getitem_176, getitem_182);  getitem_176 = getitem_182 = None
        convert_element_type_240: "f32[232, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_183, torch.float32);  getitem_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_41: "bf16[128, 116, 2, 14, 14][45472, 2, 1, 3248, 232]cuda:0" = torch.ops.aten.reshape.default(add_280, [128, 116, 2, 14, 14]);  add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_25: "bf16[128, 2, 116, 14, 14][45472, 1, 2, 3248, 232]cuda:0" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3, 4]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_20: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_42: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [128, 232, 14, 14]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_9: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_42, 1, 0, 116)
        slice_10: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_42, 1, 116, 232);  view_42 = None
        sub_40: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_103)
        mul_280: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        unsqueeze_160: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_161: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_163: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_204: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        convert_element_type_123: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None
        relu_26: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.relu.default(convert_element_type_123);  convert_element_type_123 = None
        le_10: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_10: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_10, full_default, slice_10);  le_10 = slice_10 = None
        convert_element_type_241: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        squeeze_120: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        unsqueeze_406: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_407: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        sum_32: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_241, [0, 2, 3])
        convert_element_type_122: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_116: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_122, unsqueeze_408);  convert_element_type_122 = unsqueeze_408 = None
        mul_527: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_241, sub_116)
        sum_33: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3]);  mul_527 = None
        mul_528: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 3.985969387755102e-05)
        unsqueeze_409: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
        unsqueeze_410: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_529: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 3.985969387755102e-05)
        squeeze_121: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_530: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_531: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
        unsqueeze_412: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
        unsqueeze_413: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
        mul_532: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_246);  primals_246 = None
        unsqueeze_415: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
        unsqueeze_416: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_533: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_414);  sub_116 = unsqueeze_414 = None
        sub_118: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_241, mul_533);  convert_element_type_241 = mul_533 = None
        sub_119: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_411);  sub_118 = unsqueeze_411 = None
        mul_534: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_417);  sub_119 = unsqueeze_417 = None
        mul_535: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_121);  sum_33 = squeeze_121 = None
        convert_element_type_243: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_534, torch.bfloat16);  mul_534 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_243, convert_element_type_120, convert_element_type_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_243 = convert_element_type_120 = convert_element_type_121 = None
        getitem_185: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_15[0]
        getitem_186: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_244: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_186, torch.float32);  getitem_186 = None
        convert_element_type_245: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_185, torch.float32);  getitem_185 = None
        sum_34: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_245, [0, 2, 3])
        convert_element_type_119: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sub_120: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_119, unsqueeze_420);  convert_element_type_119 = unsqueeze_420 = None
        mul_536: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_245, sub_120)
        sum_35: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2, 3]);  mul_536 = None
        mul_537: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 3.985969387755102e-05)
        unsqueeze_421: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
        unsqueeze_422: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_538: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 3.985969387755102e-05)
        mul_539: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_540: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
        unsqueeze_424: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
        unsqueeze_425: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
        unsqueeze_426: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
        mul_541: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_240);  primals_240 = None
        unsqueeze_427: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
        unsqueeze_428: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_542: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_426);  sub_120 = unsqueeze_426 = None
        sub_122: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_245, mul_542);  convert_element_type_245 = mul_542 = None
        sub_123: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_423);  sub_122 = unsqueeze_423 = None
        mul_543: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_429);  sub_123 = unsqueeze_429 = None
        mul_544: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_118);  sum_35 = squeeze_118 = None
        convert_element_type_247: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_543, torch.bfloat16);  mul_543 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_247, relu_25, convert_element_type_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  convert_element_type_247 = convert_element_type_118 = None
        getitem_188: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_16[0]
        getitem_189: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_248: "f32[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_189, torch.float32);  getitem_189 = None
        le_11: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_11: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_11, full_default, getitem_188);  le_11 = getitem_188 = None
        convert_element_type_249: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sum_36: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_249, [0, 2, 3])
        convert_element_type_116: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32);  convolution_38 = None
        sub_124: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_116, unsqueeze_432);  convert_element_type_116 = unsqueeze_432 = None
        mul_545: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_249, sub_124)
        sum_37: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 2, 3]);  mul_545 = None
        mul_546: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 3.985969387755102e-05)
        unsqueeze_433: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
        unsqueeze_434: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_547: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 3.985969387755102e-05)
        mul_548: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_549: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
        unsqueeze_436: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
        unsqueeze_437: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
        unsqueeze_438: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
        mul_550: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_234);  primals_234 = None
        unsqueeze_439: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
        unsqueeze_440: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_551: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_438);  sub_124 = unsqueeze_438 = None
        sub_126: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_249, mul_551);  convert_element_type_249 = mul_551 = None
        sub_127: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_435);  sub_126 = unsqueeze_435 = None
        mul_552: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_441);  sub_127 = unsqueeze_441 = None
        mul_553: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_115);  sum_37 = squeeze_115 = None
        convert_element_type_251: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_552, torch.bfloat16);  mul_552 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_251, getitem_97, convert_element_type_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_251 = getitem_97 = convert_element_type_115 = None
        getitem_191: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_17[0]
        getitem_192: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_252: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_192, torch.float32);  getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_19: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([slice_9, getitem_191], 1);  slice_9 = getitem_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_43: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_19, [128, 116, 2, 14, 14]);  cat_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_26: "bf16[128, 2, 116, 14, 14][45472, 196, 392, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3, 4]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_21: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_44: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [128, 232, 14, 14]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_11: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_44, 1, 0, 116)
        slice_12: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_44, 1, 116, 232);  view_44 = None
        sub_37: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, getitem_95)
        mul_259: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        unsqueeze_148: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_151: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_189: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        convert_element_type_114: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None
        relu_24: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.relu.default(convert_element_type_114);  convert_element_type_114 = None
        le_12: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_12: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_12, full_default, slice_12);  le_12 = slice_12 = None
        convert_element_type_253: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32);  where_12 = None
        squeeze_111: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        unsqueeze_442: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_443: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        sum_38: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_253, [0, 2, 3])
        convert_element_type_113: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sub_128: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_113, unsqueeze_444);  convert_element_type_113 = unsqueeze_444 = None
        mul_554: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_253, sub_128)
        sum_39: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 2, 3]);  mul_554 = None
        mul_555: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 3.985969387755102e-05)
        unsqueeze_445: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
        unsqueeze_446: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_556: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 3.985969387755102e-05)
        squeeze_112: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_557: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_558: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_556, mul_557);  mul_556 = mul_557 = None
        unsqueeze_448: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
        unsqueeze_449: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
        unsqueeze_450: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
        mul_559: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  primals_228 = None
        unsqueeze_451: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
        unsqueeze_452: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_560: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_450);  sub_128 = unsqueeze_450 = None
        sub_130: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_253, mul_560);  convert_element_type_253 = mul_560 = None
        sub_131: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_447);  sub_130 = unsqueeze_447 = None
        mul_561: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_453);  sub_131 = unsqueeze_453 = None
        mul_562: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_112);  sum_39 = squeeze_112 = None
        convert_element_type_255: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_561, torch.bfloat16);  mul_561 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_255, convert_element_type_111, convert_element_type_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_255 = convert_element_type_111 = convert_element_type_112 = None
        getitem_194: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_18[0]
        getitem_195: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_256: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_195, torch.float32);  getitem_195 = None
        convert_element_type_257: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_194, torch.float32);  getitem_194 = None
        sum_40: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_257, [0, 2, 3])
        convert_element_type_110: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_132: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_110, unsqueeze_456);  convert_element_type_110 = unsqueeze_456 = None
        mul_563: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_257, sub_132)
        sum_41: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None
        mul_564: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 3.985969387755102e-05)
        unsqueeze_457: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
        unsqueeze_458: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_565: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 3.985969387755102e-05)
        mul_566: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_567: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_565, mul_566);  mul_565 = mul_566 = None
        unsqueeze_460: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
        unsqueeze_461: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
        unsqueeze_462: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
        mul_568: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_222);  primals_222 = None
        unsqueeze_463: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
        unsqueeze_464: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_569: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_462);  sub_132 = unsqueeze_462 = None
        sub_134: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_257, mul_569);  convert_element_type_257 = mul_569 = None
        sub_135: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_459);  sub_134 = unsqueeze_459 = None
        mul_570: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_465);  sub_135 = unsqueeze_465 = None
        mul_571: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_109);  sum_41 = squeeze_109 = None
        convert_element_type_259: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_570, torch.bfloat16);  mul_570 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_259, relu_23, convert_element_type_109, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  convert_element_type_259 = convert_element_type_109 = None
        getitem_197: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_19[0]
        getitem_198: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_260: "f32[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_198, torch.float32);  getitem_198 = None
        le_13: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_13: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_13, full_default, getitem_197);  le_13 = getitem_197 = None
        convert_element_type_261: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sum_42: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_261, [0, 2, 3])
        convert_element_type_107: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32);  convolution_35 = None
        sub_136: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_107, unsqueeze_468);  convert_element_type_107 = unsqueeze_468 = None
        mul_572: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_261, sub_136)
        sum_43: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 2, 3]);  mul_572 = None
        mul_573: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 3.985969387755102e-05)
        unsqueeze_469: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
        unsqueeze_470: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_574: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 3.985969387755102e-05)
        mul_575: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_576: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_574, mul_575);  mul_574 = mul_575 = None
        unsqueeze_472: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
        unsqueeze_473: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
        unsqueeze_474: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
        mul_577: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  primals_216 = None
        unsqueeze_475: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
        unsqueeze_476: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_578: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_474);  sub_136 = unsqueeze_474 = None
        sub_138: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_261, mul_578);  convert_element_type_261 = mul_578 = None
        sub_139: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_471);  sub_138 = unsqueeze_471 = None
        mul_579: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_477);  sub_139 = unsqueeze_477 = None
        mul_580: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_106);  sum_43 = squeeze_106 = None
        convert_element_type_263: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_579, torch.bfloat16);  mul_579 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_263, getitem_89, convert_element_type_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_263 = getitem_89 = convert_element_type_106 = None
        getitem_200: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_20[0]
        getitem_201: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_264: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_20: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([slice_11, getitem_200], 1);  slice_11 = getitem_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_45: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_20, [128, 116, 2, 14, 14]);  cat_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_27: "bf16[128, 2, 116, 14, 14][45472, 196, 392, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3, 4]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_22: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_46: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [128, 232, 14, 14]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_13: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_46, 1, 0, 116)
        slice_14: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_46, 1, 116, 232);  view_46 = None
        sub_34: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_87)
        mul_238: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        unsqueeze_136: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_137: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_139: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_174: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        convert_element_type_105: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.bfloat16);  add_174 = None
        relu_22: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.relu.default(convert_element_type_105);  convert_element_type_105 = None
        le_14: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_14: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_14, full_default, slice_14);  le_14 = slice_14 = None
        convert_element_type_265: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        squeeze_102: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        unsqueeze_478: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_479: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        sum_44: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_265, [0, 2, 3])
        convert_element_type_104: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_140: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_104, unsqueeze_480);  convert_element_type_104 = unsqueeze_480 = None
        mul_581: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, sub_140)
        sum_45: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3]);  mul_581 = None
        mul_582: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
        unsqueeze_481: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
        unsqueeze_482: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_583: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
        squeeze_103: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_584: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_585: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_583, mul_584);  mul_583 = mul_584 = None
        unsqueeze_484: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
        unsqueeze_485: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        mul_586: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_210);  primals_210 = None
        unsqueeze_487: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
        unsqueeze_488: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_587: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_486);  sub_140 = unsqueeze_486 = None
        sub_142: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_265, mul_587);  convert_element_type_265 = mul_587 = None
        sub_143: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_483);  sub_142 = unsqueeze_483 = None
        mul_588: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_489);  sub_143 = unsqueeze_489 = None
        mul_589: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_103);  sum_45 = squeeze_103 = None
        convert_element_type_267: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_588, torch.bfloat16);  mul_588 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_267, convert_element_type_102, convert_element_type_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_267 = convert_element_type_102 = convert_element_type_103 = None
        getitem_203: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_21[0]
        getitem_204: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        convert_element_type_268: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_204, torch.float32);  getitem_204 = None
        convert_element_type_269: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_203, torch.float32);  getitem_203 = None
        sum_46: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_269, [0, 2, 3])
        convert_element_type_101: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sub_144: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, unsqueeze_492);  convert_element_type_101 = unsqueeze_492 = None
        mul_590: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_269, sub_144)
        sum_47: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3]);  mul_590 = None
        mul_591: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 3.985969387755102e-05)
        unsqueeze_493: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
        unsqueeze_494: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_592: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
        mul_593: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_594: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        unsqueeze_496: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
        unsqueeze_497: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
        mul_595: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_204);  primals_204 = None
        unsqueeze_499: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
        unsqueeze_500: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_596: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_498);  sub_144 = unsqueeze_498 = None
        sub_146: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_269, mul_596);  convert_element_type_269 = mul_596 = None
        sub_147: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_495);  sub_146 = unsqueeze_495 = None
        mul_597: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_501);  sub_147 = unsqueeze_501 = None
        mul_598: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_100);  sum_47 = squeeze_100 = None
        convert_element_type_271: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_597, torch.bfloat16);  mul_597 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_271, relu_21, convert_element_type_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  convert_element_type_271 = convert_element_type_100 = None
        getitem_206: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_22[0]
        getitem_207: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_272: "f32[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None
        le_15: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_15: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_15, full_default, getitem_206);  le_15 = getitem_206 = None
        convert_element_type_273: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        sum_48: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_273, [0, 2, 3])
        convert_element_type_98: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_148: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_98, unsqueeze_504);  convert_element_type_98 = unsqueeze_504 = None
        mul_599: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_273, sub_148)
        sum_49: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None
        mul_600: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 3.985969387755102e-05)
        unsqueeze_505: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
        unsqueeze_506: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_601: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
        mul_602: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_603: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_601, mul_602);  mul_601 = mul_602 = None
        unsqueeze_508: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
        unsqueeze_509: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
        mul_604: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  primals_198 = None
        unsqueeze_511: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
        unsqueeze_512: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_605: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_510);  sub_148 = unsqueeze_510 = None
        sub_150: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_273, mul_605);  convert_element_type_273 = mul_605 = None
        sub_151: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_507);  sub_150 = unsqueeze_507 = None
        mul_606: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_513);  sub_151 = unsqueeze_513 = None
        mul_607: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_97);  sum_49 = squeeze_97 = None
        convert_element_type_275: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_606, torch.bfloat16);  mul_606 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_275, getitem_81, convert_element_type_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_275 = getitem_81 = convert_element_type_97 = None
        getitem_209: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_23[0]
        getitem_210: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_276: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_210, torch.float32);  getitem_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_21: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([slice_13, getitem_209], 1);  slice_13 = getitem_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_47: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_21, [128, 116, 2, 14, 14]);  cat_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_28: "bf16[128, 2, 116, 14, 14][45472, 196, 392, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3, 4]);  view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_23: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_48: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [128, 232, 14, 14]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_15: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_48, 1, 0, 116)
        slice_16: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_48, 1, 116, 232);  view_48 = None
        sub_31: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, getitem_79)
        mul_217: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        unsqueeze_124: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
        unsqueeze_127: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_159: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        convert_element_type_96: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None
        relu_20: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.relu.default(convert_element_type_96);  convert_element_type_96 = None
        le_16: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_16: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_16, full_default, slice_16);  le_16 = slice_16 = None
        convert_element_type_277: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        squeeze_93: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        unsqueeze_514: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_515: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        sum_50: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_277, [0, 2, 3])
        convert_element_type_95: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sub_152: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_95, unsqueeze_516);  convert_element_type_95 = unsqueeze_516 = None
        mul_608: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_277, sub_152)
        sum_51: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
        mul_609: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
        unsqueeze_517: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
        unsqueeze_518: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_610: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 3.985969387755102e-05)
        squeeze_94: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_611: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_612: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
        unsqueeze_520: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_521: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
        mul_613: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_192);  primals_192 = None
        unsqueeze_523: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
        unsqueeze_524: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_614: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_522);  sub_152 = unsqueeze_522 = None
        sub_154: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_277, mul_614);  convert_element_type_277 = mul_614 = None
        sub_155: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_519);  sub_154 = unsqueeze_519 = None
        mul_615: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_525);  sub_155 = unsqueeze_525 = None
        mul_616: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_94);  sum_51 = squeeze_94 = None
        convert_element_type_279: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_615, torch.bfloat16);  mul_615 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_279, convert_element_type_93, convert_element_type_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_279 = convert_element_type_93 = convert_element_type_94 = None
        getitem_212: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_24[0]
        getitem_213: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_280: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_213, torch.float32);  getitem_213 = None
        convert_element_type_281: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_212, torch.float32);  getitem_212 = None
        sum_52: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_281, [0, 2, 3])
        convert_element_type_92: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_156: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_92, unsqueeze_528);  convert_element_type_92 = unsqueeze_528 = None
        mul_617: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_281, sub_156)
        sum_53: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3]);  mul_617 = None
        mul_618: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 3.985969387755102e-05)
        unsqueeze_529: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
        unsqueeze_530: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_619: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 3.985969387755102e-05)
        mul_620: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_621: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
        unsqueeze_532: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
        unsqueeze_533: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
        unsqueeze_534: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
        mul_622: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_186);  primals_186 = None
        unsqueeze_535: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
        unsqueeze_536: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_623: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_534);  sub_156 = unsqueeze_534 = None
        sub_158: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_281, mul_623);  convert_element_type_281 = mul_623 = None
        sub_159: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_531);  sub_158 = unsqueeze_531 = None
        mul_624: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_537);  sub_159 = unsqueeze_537 = None
        mul_625: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_91);  sum_53 = squeeze_91 = None
        convert_element_type_283: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_624, torch.bfloat16);  mul_624 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_283, relu_19, convert_element_type_91, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  convert_element_type_283 = convert_element_type_91 = None
        getitem_215: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_25[0]
        getitem_216: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_284: "f32[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_216, torch.float32);  getitem_216 = None
        le_17: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_17: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_17, full_default, getitem_215);  le_17 = getitem_215 = None
        convert_element_type_285: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        sum_54: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_285, [0, 2, 3])
        convert_element_type_89: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_160: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_89, unsqueeze_540);  convert_element_type_89 = unsqueeze_540 = None
        mul_626: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_285, sub_160)
        sum_55: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3]);  mul_626 = None
        mul_627: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 3.985969387755102e-05)
        unsqueeze_541: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
        unsqueeze_542: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_628: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 3.985969387755102e-05)
        mul_629: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_630: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_628, mul_629);  mul_628 = mul_629 = None
        unsqueeze_544: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
        unsqueeze_545: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
        unsqueeze_546: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
        mul_631: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_180);  primals_180 = None
        unsqueeze_547: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
        unsqueeze_548: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_632: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_546);  sub_160 = unsqueeze_546 = None
        sub_162: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_285, mul_632);  convert_element_type_285 = mul_632 = None
        sub_163: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_543);  sub_162 = unsqueeze_543 = None
        mul_633: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_549);  sub_163 = unsqueeze_549 = None
        mul_634: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_88);  sum_55 = squeeze_88 = None
        convert_element_type_287: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_633, torch.bfloat16);  mul_633 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_287, getitem_73, convert_element_type_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_287 = getitem_73 = convert_element_type_88 = None
        getitem_218: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_26[0]
        getitem_219: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_288: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_219, torch.float32);  getitem_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_22: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([slice_15, getitem_218], 1);  slice_15 = getitem_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_49: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_22, [128, 116, 2, 14, 14]);  cat_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_29: "bf16[128, 2, 116, 14, 14][45472, 196, 392, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3, 4]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_24: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_50: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [128, 232, 14, 14]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_17: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_50, 1, 0, 116)
        slice_18: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_50, 1, 116, 232);  view_50 = None
        sub_28: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_71)
        mul_196: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        unsqueeze_112: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_113: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_115: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_144: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        convert_element_type_87: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None
        relu_18: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.relu.default(convert_element_type_87);  convert_element_type_87 = None
        le_18: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_18: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_18, full_default, slice_18);  le_18 = slice_18 = None
        convert_element_type_289: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32);  where_18 = None
        squeeze_84: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        unsqueeze_550: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_551: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        sum_56: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_289, [0, 2, 3])
        convert_element_type_86: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_164: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_86, unsqueeze_552);  convert_element_type_86 = unsqueeze_552 = None
        mul_635: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_289, sub_164)
        sum_57: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 2, 3]);  mul_635 = None
        mul_636: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 3.985969387755102e-05)
        unsqueeze_553: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
        unsqueeze_554: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_637: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 3.985969387755102e-05)
        squeeze_85: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_638: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_639: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_637, mul_638);  mul_637 = mul_638 = None
        unsqueeze_556: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
        unsqueeze_557: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
        unsqueeze_558: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
        mul_640: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_174);  primals_174 = None
        unsqueeze_559: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
        unsqueeze_560: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_641: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_558);  sub_164 = unsqueeze_558 = None
        sub_166: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_289, mul_641);  convert_element_type_289 = mul_641 = None
        sub_167: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_555);  sub_166 = unsqueeze_555 = None
        mul_642: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_561);  sub_167 = unsqueeze_561 = None
        mul_643: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_85);  sum_57 = squeeze_85 = None
        convert_element_type_291: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_642, torch.bfloat16);  mul_642 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_291, convert_element_type_84, convert_element_type_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_291 = convert_element_type_84 = convert_element_type_85 = None
        getitem_221: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_27[0]
        getitem_222: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        convert_element_type_292: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_222, torch.float32);  getitem_222 = None
        convert_element_type_293: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_221, torch.float32);  getitem_221 = None
        sum_58: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_293, [0, 2, 3])
        convert_element_type_83: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sub_168: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_83, unsqueeze_564);  convert_element_type_83 = unsqueeze_564 = None
        mul_644: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_293, sub_168)
        sum_59: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3]);  mul_644 = None
        mul_645: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 3.985969387755102e-05)
        unsqueeze_565: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
        unsqueeze_566: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_646: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 3.985969387755102e-05)
        mul_647: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_648: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
        unsqueeze_568: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
        unsqueeze_569: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
        unsqueeze_570: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
        mul_649: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  primals_168 = None
        unsqueeze_571: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
        unsqueeze_572: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_650: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_570);  sub_168 = unsqueeze_570 = None
        sub_170: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_293, mul_650);  convert_element_type_293 = mul_650 = None
        sub_171: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_567);  sub_170 = unsqueeze_567 = None
        mul_651: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_573);  sub_171 = unsqueeze_573 = None
        mul_652: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_82);  sum_59 = squeeze_82 = None
        convert_element_type_295: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_651, torch.bfloat16);  mul_651 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(convert_element_type_295, relu_17, convert_element_type_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  convert_element_type_295 = convert_element_type_82 = None
        getitem_224: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_28[0]
        getitem_225: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_296: "f32[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_225, torch.float32);  getitem_225 = None
        le_19: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_19: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_19, full_default, getitem_224);  le_19 = getitem_224 = None
        convert_element_type_297: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        sum_60: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_297, [0, 2, 3])
        convert_element_type_80: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_172: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, unsqueeze_576);  convert_element_type_80 = unsqueeze_576 = None
        mul_653: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_297, sub_172)
        sum_61: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 2, 3]);  mul_653 = None
        mul_654: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 3.985969387755102e-05)
        unsqueeze_577: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
        unsqueeze_578: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_655: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 3.985969387755102e-05)
        mul_656: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_657: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
        unsqueeze_580: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
        unsqueeze_581: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
        unsqueeze_582: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
        mul_658: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_162);  primals_162 = None
        unsqueeze_583: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
        unsqueeze_584: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_659: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_582);  sub_172 = unsqueeze_582 = None
        sub_174: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_297, mul_659);  convert_element_type_297 = mul_659 = None
        sub_175: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_579);  sub_174 = unsqueeze_579 = None
        mul_660: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_585);  sub_175 = unsqueeze_585 = None
        mul_661: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_79);  sum_61 = squeeze_79 = None
        convert_element_type_299: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_660, torch.bfloat16);  mul_660 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_299, getitem_65, convert_element_type_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_299 = getitem_65 = convert_element_type_79 = None
        getitem_227: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_29[0]
        getitem_228: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_300: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_228, torch.float32);  getitem_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_23: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([slice_17, getitem_227], 1);  slice_17 = getitem_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_51: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_23, [128, 116, 2, 14, 14]);  cat_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_30: "bf16[128, 2, 116, 14, 14][45472, 196, 392, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3, 4]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_25: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_52: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [128, 232, 14, 14]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_19: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_52, 1, 0, 116)
        slice_20: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_52, 1, 116, 232);  view_52 = None
        sub_25: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_63)
        mul_175: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_100: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_103: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_129: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        convert_element_type_78: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None
        relu_16: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.relu.default(convert_element_type_78);  convert_element_type_78 = None
        le_20: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_20: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_20, full_default, slice_20);  le_20 = slice_20 = None
        convert_element_type_301: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        squeeze_75: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        unsqueeze_586: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_587: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        sum_62: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_301, [0, 2, 3])
        convert_element_type_77: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sub_176: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_77, unsqueeze_588);  convert_element_type_77 = unsqueeze_588 = None
        mul_662: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_301, sub_176)
        sum_63: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None
        mul_663: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 3.985969387755102e-05)
        unsqueeze_589: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
        unsqueeze_590: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_664: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 3.985969387755102e-05)
        squeeze_76: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_665: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_666: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
        unsqueeze_592: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_593: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
        unsqueeze_594: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
        mul_667: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_156);  primals_156 = None
        unsqueeze_595: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
        unsqueeze_596: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_668: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_594);  sub_176 = unsqueeze_594 = None
        sub_178: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_301, mul_668);  convert_element_type_301 = mul_668 = None
        sub_179: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_591);  sub_178 = unsqueeze_591 = None
        mul_669: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_597);  sub_179 = unsqueeze_597 = None
        mul_670: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_76);  sum_63 = squeeze_76 = None
        convert_element_type_303: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_669, torch.bfloat16);  mul_669 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_303, convert_element_type_75, convert_element_type_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_303 = convert_element_type_75 = convert_element_type_76 = None
        getitem_230: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_30[0]
        getitem_231: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_304: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None
        convert_element_type_305: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_230, torch.float32);  getitem_230 = None
        sum_64: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_305, [0, 2, 3])
        convert_element_type_74: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_180: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_74, unsqueeze_600);  convert_element_type_74 = unsqueeze_600 = None
        mul_671: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_305, sub_180)
        sum_65: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3]);  mul_671 = None
        mul_672: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 3.985969387755102e-05)
        unsqueeze_601: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
        unsqueeze_602: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_673: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 3.985969387755102e-05)
        mul_674: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_675: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
        unsqueeze_604: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_605: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
        unsqueeze_606: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
        mul_676: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  primals_150 = None
        unsqueeze_607: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
        unsqueeze_608: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_677: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_606);  sub_180 = unsqueeze_606 = None
        sub_182: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_305, mul_677);  convert_element_type_305 = mul_677 = None
        sub_183: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_603);  sub_182 = unsqueeze_603 = None
        mul_678: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_609);  sub_183 = unsqueeze_609 = None
        mul_679: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_73);  sum_65 = squeeze_73 = None
        convert_element_type_307: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_678, torch.bfloat16);  mul_678 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_307, relu_15, convert_element_type_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  convert_element_type_307 = convert_element_type_73 = None
        getitem_233: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_31[0]
        getitem_234: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_308: "f32[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_234, torch.float32);  getitem_234 = None
        le_21: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_21: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_21, full_default, getitem_233);  le_21 = getitem_233 = None
        convert_element_type_309: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        sum_66: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_309, [0, 2, 3])
        convert_element_type_71: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32);  convolution_23 = None
        sub_184: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_71, unsqueeze_612);  convert_element_type_71 = unsqueeze_612 = None
        mul_680: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_309, sub_184)
        sum_67: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None
        mul_681: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 3.985969387755102e-05)
        unsqueeze_613: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
        unsqueeze_614: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_682: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 3.985969387755102e-05)
        mul_683: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_684: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
        unsqueeze_616: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_617: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
        unsqueeze_618: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
        mul_685: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_144);  primals_144 = None
        unsqueeze_619: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
        unsqueeze_620: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_686: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_618);  sub_184 = unsqueeze_618 = None
        sub_186: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_309, mul_686);  convert_element_type_309 = mul_686 = None
        sub_187: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_615);  sub_186 = unsqueeze_615 = None
        mul_687: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_621);  sub_187 = unsqueeze_621 = None
        mul_688: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_70);  sum_67 = squeeze_70 = None
        convert_element_type_311: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_687, torch.bfloat16);  mul_687 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_311, getitem_57, convert_element_type_70, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_311 = getitem_57 = convert_element_type_70 = None
        getitem_236: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_32[0]
        getitem_237: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_312: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_24: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([slice_19, getitem_236], 1);  slice_19 = getitem_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_53: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_24, [128, 116, 2, 14, 14]);  cat_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_31: "bf16[128, 2, 116, 14, 14][45472, 196, 392, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3, 4]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_26: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        view_54: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [128, 232, 14, 14]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_21: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_54, 1, 0, 116)
        slice_22: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_54, 1, 116, 232);  view_54 = None
        sub_22: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, getitem_55)
        mul_154: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        unsqueeze_88: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_89: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_91: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_114: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        convert_element_type_69: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None
        relu_14: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.relu.default(convert_element_type_69);  convert_element_type_69 = None
        le_22: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_22: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_22, full_default, slice_22);  le_22 = slice_22 = None
        convert_element_type_313: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_22, torch.float32);  where_22 = None
        squeeze_66: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        unsqueeze_622: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_623: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        sum_68: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_313, [0, 2, 3])
        convert_element_type_68: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_188: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_68, unsqueeze_624);  convert_element_type_68 = unsqueeze_624 = None
        mul_689: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_313, sub_188)
        sum_69: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
        mul_690: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 3.985969387755102e-05)
        unsqueeze_625: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
        unsqueeze_626: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_691: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 3.985969387755102e-05)
        squeeze_67: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_692: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_693: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_691, mul_692);  mul_691 = mul_692 = None
        unsqueeze_628: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
        unsqueeze_629: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
        unsqueeze_630: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
        mul_694: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_138);  primals_138 = None
        unsqueeze_631: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
        unsqueeze_632: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_695: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_630);  sub_188 = unsqueeze_630 = None
        sub_190: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_313, mul_695);  convert_element_type_313 = mul_695 = None
        sub_191: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_627);  sub_190 = unsqueeze_627 = None
        mul_696: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_633);  sub_191 = unsqueeze_633 = None
        mul_697: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_67);  sum_69 = squeeze_67 = None
        convert_element_type_315: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_696, torch.bfloat16);  mul_696 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_315, convert_element_type_66, convert_element_type_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_315 = convert_element_type_66 = convert_element_type_67 = None
        getitem_239: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_33[0]
        getitem_240: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_316: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_240, torch.float32);  getitem_240 = None
        convert_element_type_317: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_239, torch.float32);  getitem_239 = None
        sum_70: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_317, [0, 2, 3])
        convert_element_type_65: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32);  convolution_21 = None
        sub_192: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_65, unsqueeze_636);  convert_element_type_65 = unsqueeze_636 = None
        mul_698: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_317, sub_192)
        sum_71: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 2, 3]);  mul_698 = None
        mul_699: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 3.985969387755102e-05)
        unsqueeze_637: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
        unsqueeze_638: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_700: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 3.985969387755102e-05)
        mul_701: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_702: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_700, mul_701);  mul_700 = mul_701 = None
        unsqueeze_640: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
        unsqueeze_641: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
        unsqueeze_642: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
        mul_703: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_132);  primals_132 = None
        unsqueeze_643: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
        unsqueeze_644: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_704: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_642);  sub_192 = unsqueeze_642 = None
        sub_194: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_317, mul_704);  convert_element_type_317 = mul_704 = None
        sub_195: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_639);  sub_194 = unsqueeze_639 = None
        mul_705: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_645);  sub_195 = unsqueeze_645 = None
        mul_706: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_64);  sum_71 = squeeze_64 = None
        convert_element_type_319: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_705, torch.bfloat16);  mul_705 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_319, relu_13, convert_element_type_64, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  convert_element_type_319 = convert_element_type_64 = None
        getitem_242: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_34[0]
        getitem_243: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_320: "f32[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_243, torch.float32);  getitem_243 = None
        le_23: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_23: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_23, full_default, getitem_242);  le_23 = getitem_242 = None
        convert_element_type_321: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_23, torch.float32);  where_23 = None
        sum_72: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_321, [0, 2, 3])
        convert_element_type_62: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_196: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_62, unsqueeze_648);  convert_element_type_62 = unsqueeze_648 = None
        mul_707: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, sub_196)
        sum_73: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3]);  mul_707 = None
        mul_708: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 3.985969387755102e-05)
        unsqueeze_649: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
        unsqueeze_650: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_709: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 3.985969387755102e-05)
        mul_710: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_711: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
        unsqueeze_652: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_653: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
        unsqueeze_654: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
        mul_712: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_655: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
        unsqueeze_656: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_713: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_654);  sub_196 = unsqueeze_654 = None
        sub_198: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_321, mul_713);  convert_element_type_321 = mul_713 = None
        sub_199: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_651);  sub_198 = unsqueeze_651 = None
        mul_714: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_657);  sub_199 = unsqueeze_657 = None
        mul_715: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_61);  sum_73 = squeeze_61 = None
        convert_element_type_323: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_714, torch.bfloat16);  mul_714 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_323, getitem_49, convert_element_type_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_323 = getitem_49 = convert_element_type_61 = None
        getitem_245: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_35[0]
        getitem_246: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_324: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_246, torch.float32);  getitem_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_25: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([slice_21, getitem_245], 1);  slice_21 = getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_55: "bf16[128, 116, 2, 14, 14][45472, 392, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(cat_25, [128, 116, 2, 14, 14]);  cat_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_32: "bf16[128, 2, 116, 14, 14][45472, 196, 392, 14, 1]cuda:0" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3, 4]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_27: "bf16[128, 2, 116, 14, 14][45472, 22736, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_56: "bf16[128, 232, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [128, 232, 14, 14]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        slice_23: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_56, 1, 0, 116)
        slice_24: "bf16[128, 116, 14, 14][45472, 196, 14, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_56, 1, 116, 232);  view_56 = None
        sub_19: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_47)
        mul_133: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        unsqueeze_76: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_99: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        convert_element_type_60: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        relu_12: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.relu.default(convert_element_type_60);  convert_element_type_60 = None
        le_24: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_24: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_24, full_default, slice_24);  le_24 = slice_24 = None
        convert_element_type_325: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_24, torch.float32);  where_24 = None
        squeeze_57: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        unsqueeze_658: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_659: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        sum_74: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_325, [0, 2, 3])
        convert_element_type_59: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_200: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_59, unsqueeze_660);  convert_element_type_59 = unsqueeze_660 = None
        mul_716: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_325, sub_200)
        sum_75: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3]);  mul_716 = None
        mul_717: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 3.985969387755102e-05)
        unsqueeze_661: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
        unsqueeze_662: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_718: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 3.985969387755102e-05)
        squeeze_58: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_719: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_720: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
        unsqueeze_664: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_665: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
        unsqueeze_666: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
        mul_721: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_667: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
        unsqueeze_668: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_722: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_666);  sub_200 = unsqueeze_666 = None
        sub_202: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_325, mul_722);  convert_element_type_325 = mul_722 = None
        sub_203: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_663);  sub_202 = unsqueeze_663 = None
        mul_723: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_669);  sub_203 = unsqueeze_669 = None
        mul_724: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_58);  sum_75 = squeeze_58 = None
        convert_element_type_327: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_723, torch.bfloat16);  mul_723 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_327, convert_element_type_57, convert_element_type_58, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_327 = convert_element_type_57 = convert_element_type_58 = None
        getitem_248: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_36[0]
        getitem_249: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_328: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_249, torch.float32);  getitem_249 = None
        convert_element_type_329: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_248, torch.float32);  getitem_248 = None
        sum_76: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_329, [0, 2, 3])
        convert_element_type_56: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_204: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_56, unsqueeze_672);  convert_element_type_56 = unsqueeze_672 = None
        mul_725: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_329, sub_204)
        sum_77: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3]);  mul_725 = None
        mul_726: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 3.985969387755102e-05)
        unsqueeze_673: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
        unsqueeze_674: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_727: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 3.985969387755102e-05)
        mul_728: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_729: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
        unsqueeze_676: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
        unsqueeze_677: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
        unsqueeze_678: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
        mul_730: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_679: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
        unsqueeze_680: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_731: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_678);  sub_204 = unsqueeze_678 = None
        sub_206: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_329, mul_731);  convert_element_type_329 = mul_731 = None
        sub_207: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_675);  sub_206 = unsqueeze_675 = None
        mul_732: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_681);  sub_207 = unsqueeze_681 = None
        mul_733: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_55);  sum_77 = squeeze_55 = None
        convert_element_type_331: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_732, torch.bfloat16);  mul_732 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_331, relu_11, convert_element_type_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  convert_element_type_331 = convert_element_type_55 = None
        getitem_251: "bf16[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = convolution_backward_37[0]
        getitem_252: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        convert_element_type_332: "f32[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_252, torch.float32);  getitem_252 = None
        le_25: "b8[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_25: "bf16[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.aten.where.self(le_25, full_default, getitem_251);  le_25 = getitem_251 = None
        convert_element_type_333: "f32[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_25, torch.float32);  where_25 = None
        sum_78: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_333, [0, 2, 3])
        convert_element_type_53: "f32[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        sub_208: "f32[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_53, unsqueeze_684);  convert_element_type_53 = unsqueeze_684 = None
        mul_734: "f32[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_333, sub_208)
        sum_79: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_734, [0, 2, 3]);  mul_734 = None
        mul_735: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 9.964923469387754e-06)
        unsqueeze_685: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
        unsqueeze_686: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_736: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 9.964923469387754e-06)
        mul_737: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_738: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
        unsqueeze_688: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_689: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
        unsqueeze_690: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
        mul_739: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_691: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
        unsqueeze_692: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_740: "f32[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_690);  sub_208 = unsqueeze_690 = None
        sub_210: "f32[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_333, mul_740);  convert_element_type_333 = mul_740 = None
        sub_211: "f32[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_687);  sub_210 = unsqueeze_687 = None
        mul_741: "f32[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_693);  sub_211 = unsqueeze_693 = None
        mul_742: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_52);  sum_79 = squeeze_52 = None
        convert_element_type_335: "bf16[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_741, torch.bfloat16);  mul_741 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_335, view_7, convert_element_type_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_335 = convert_element_type_52 = None
        getitem_254: "bf16[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = convolution_backward_38[0]
        getitem_255: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_336: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_255, torch.float32);  getitem_255 = None
        sub_16: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_41)
        mul_112: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        unsqueeze_64: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[116, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[116, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_84: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_51: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None
        relu_10: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.relu.default(convert_element_type_51);  convert_element_type_51 = None
        le_26: "b8[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_26: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.where.self(le_26, full_default, slice_23);  le_26 = slice_23 = None
        convert_element_type_337: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(where_26, torch.float32);  where_26 = None
        squeeze_48: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        unsqueeze_694: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_695: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        sum_80: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_337, [0, 2, 3])
        convert_element_type_50: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_212: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, unsqueeze_696);  convert_element_type_50 = unsqueeze_696 = None
        mul_743: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_337, sub_212)
        sum_81: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_743, [0, 2, 3]);  mul_743 = None
        mul_744: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, 3.985969387755102e-05)
        unsqueeze_697: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
        unsqueeze_698: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_745: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
        squeeze_49: "f32[116][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_746: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_747: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_745, mul_746);  mul_745 = mul_746 = None
        unsqueeze_700: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
        unsqueeze_701: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
        unsqueeze_702: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
        mul_748: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_703: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
        unsqueeze_704: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_749: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_702);  sub_212 = unsqueeze_702 = None
        sub_214: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_337, mul_749);  convert_element_type_337 = mul_749 = None
        sub_215: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_699);  sub_214 = unsqueeze_699 = None
        mul_750: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_705);  sub_215 = unsqueeze_705 = None
        mul_751: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, squeeze_49);  sum_81 = squeeze_49 = None
        convert_element_type_339: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_750, torch.bfloat16);  mul_750 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_339, convert_element_type_48, convert_element_type_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_339 = convert_element_type_48 = convert_element_type_49 = None
        getitem_257: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = convolution_backward_39[0]
        getitem_258: "bf16[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_340: "f32[116, 116, 1, 1][116, 1, 116, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_258, torch.float32);  getitem_258 = None
        convert_element_type_341: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_257, torch.float32);  getitem_257 = None
        sum_82: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_341, [0, 2, 3])
        convert_element_type_47: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_216: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_47, unsqueeze_708);  convert_element_type_47 = unsqueeze_708 = None
        mul_752: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_341, sub_216)
        sum_83: "f32[116][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 2, 3]);  mul_752 = None
        mul_753: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 3.985969387755102e-05)
        unsqueeze_709: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
        unsqueeze_710: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_754: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 3.985969387755102e-05)
        mul_755: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_756: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_754, mul_755);  mul_754 = mul_755 = None
        unsqueeze_712: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
        unsqueeze_713: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
        unsqueeze_714: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
        mul_757: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_715: "f32[1, 116][116, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
        unsqueeze_716: "f32[1, 116, 1][116, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 116, 1, 1][116, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_758: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_714);  sub_216 = unsqueeze_714 = None
        sub_218: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_341, mul_758);  convert_element_type_341 = mul_758 = None
        sub_219: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_711);  sub_218 = unsqueeze_711 = None
        mul_759: "f32[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_717);  sub_219 = unsqueeze_717 = None
        mul_760: "f32[116][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_46);  sum_83 = squeeze_46 = None
        convert_element_type_343: "bf16[128, 116, 14, 14][22736, 1, 1624, 116]cuda:0" = torch.ops.prims.convert_element_type.default(mul_759, torch.bfloat16);  mul_759 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_343, view_7, convert_element_type_46, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  convert_element_type_343 = view_7 = convert_element_type_46 = None
        getitem_260: "bf16[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = convolution_backward_40[0]
        getitem_261: "bf16[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        add_281: "bf16[128, 116, 28, 28][90944, 1, 3248, 116]cuda:0" = torch.ops.aten.add.Tensor(getitem_254, getitem_260);  getitem_254 = getitem_260 = None
        convert_element_type_344: "f32[116, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_261, torch.float32);  getitem_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_57: "bf16[128, 58, 2, 28, 28][90944, 2, 1, 3248, 116]cuda:0" = torch.ops.aten.reshape.default(add_281, [128, 58, 2, 28, 28]);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_33: "bf16[128, 2, 58, 28, 28][90944, 1, 2, 3248, 116]cuda:0" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3, 4]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_28: "bf16[128, 2, 58, 28, 28][90944, 45472, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_58: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [128, 116, 28, 28]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_25: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_58, 1, 0, 58)
        slice_26: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_58, 1, 58, 116);  view_58 = None
        sub_14: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_37)
        mul_98: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        unsqueeze_56: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_74: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_45: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None
        relu_9: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.relu.default(convert_element_type_45);  convert_element_type_45 = None
        le_27: "b8[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_27: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.where.self(le_27, full_default, slice_26);  le_27 = slice_26 = None
        convert_element_type_345: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(where_27, torch.float32);  where_27 = None
        squeeze_42: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        unsqueeze_718: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_719: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        sum_84: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_345, [0, 2, 3])
        convert_element_type_44: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_220: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_44, unsqueeze_720);  convert_element_type_44 = unsqueeze_720 = None
        mul_761: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_345, sub_220)
        sum_85: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 2, 3]);  mul_761 = None
        mul_762: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 9.964923469387754e-06)
        unsqueeze_721: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
        unsqueeze_722: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_763: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 9.964923469387754e-06)
        squeeze_43: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_764: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_765: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_763, mul_764);  mul_763 = mul_764 = None
        unsqueeze_724: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
        unsqueeze_725: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
        unsqueeze_726: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
        mul_766: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_727: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
        unsqueeze_728: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_767: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_726);  sub_220 = unsqueeze_726 = None
        sub_222: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_345, mul_767);  convert_element_type_345 = mul_767 = None
        sub_223: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_723);  sub_222 = unsqueeze_723 = None
        mul_768: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_729);  sub_223 = unsqueeze_729 = None
        mul_769: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_43);  sum_85 = squeeze_43 = None
        convert_element_type_347: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_768, torch.bfloat16);  mul_768 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_347, convert_element_type_42, convert_element_type_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_347 = convert_element_type_42 = convert_element_type_43 = None
        getitem_263: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_41[0]
        getitem_264: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_348: "f32[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_264, torch.float32);  getitem_264 = None
        convert_element_type_349: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_263, torch.float32);  getitem_263 = None
        sum_86: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_349, [0, 2, 3])
        convert_element_type_41: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sub_224: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_41, unsqueeze_732);  convert_element_type_41 = unsqueeze_732 = None
        mul_770: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_349, sub_224)
        sum_87: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3]);  mul_770 = None
        mul_771: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 9.964923469387754e-06)
        unsqueeze_733: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
        unsqueeze_734: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_772: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 9.964923469387754e-06)
        mul_773: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_774: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_772, mul_773);  mul_772 = mul_773 = None
        unsqueeze_736: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_737: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
        unsqueeze_738: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
        mul_775: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_739: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
        unsqueeze_740: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_776: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_738);  sub_224 = unsqueeze_738 = None
        sub_226: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_349, mul_776);  convert_element_type_349 = mul_776 = None
        sub_227: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_735);  sub_226 = unsqueeze_735 = None
        mul_777: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_741);  sub_227 = unsqueeze_741 = None
        mul_778: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_40);  sum_87 = squeeze_40 = None
        convert_element_type_351: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_777, torch.bfloat16);  mul_777 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_351, relu_8, convert_element_type_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  convert_element_type_351 = convert_element_type_40 = None
        getitem_266: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_42[0]
        getitem_267: "bf16[58, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_352: "f32[58, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_267, torch.float32);  getitem_267 = None
        le_28: "b8[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_28: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.where.self(le_28, full_default, getitem_266);  le_28 = getitem_266 = None
        convert_element_type_353: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(where_28, torch.float32);  where_28 = None
        sum_88: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_353, [0, 2, 3])
        convert_element_type_38: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sub_228: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, unsqueeze_744);  convert_element_type_38 = unsqueeze_744 = None
        mul_779: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_353, sub_228)
        sum_89: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 2, 3]);  mul_779 = None
        mul_780: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 9.964923469387754e-06)
        unsqueeze_745: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
        unsqueeze_746: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_781: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 9.964923469387754e-06)
        mul_782: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_783: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
        unsqueeze_748: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
        unsqueeze_749: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
        unsqueeze_750: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
        mul_784: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_751: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
        unsqueeze_752: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_785: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_750);  sub_228 = unsqueeze_750 = None
        sub_230: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_353, mul_785);  convert_element_type_353 = mul_785 = None
        sub_231: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_747);  sub_230 = unsqueeze_747 = None
        mul_786: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_753);  sub_231 = unsqueeze_753 = None
        mul_787: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_37);  sum_89 = squeeze_37 = None
        convert_element_type_355: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_786, torch.bfloat16);  mul_786 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_355, getitem_31, convert_element_type_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_355 = getitem_31 = convert_element_type_37 = None
        getitem_269: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_43[0]
        getitem_270: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_356: "f32[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_270, torch.float32);  getitem_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_26: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([slice_25, getitem_269], 1);  slice_25 = getitem_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_59: "bf16[128, 58, 2, 28, 28][90944, 1568, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(cat_26, [128, 58, 2, 28, 28]);  cat_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_34: "bf16[128, 2, 58, 28, 28][90944, 784, 1568, 28, 1]cuda:0" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3, 4]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_29: "bf16[128, 2, 58, 28, 28][90944, 45472, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_60: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [128, 116, 28, 28]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_27: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_60, 1, 0, 58)
        slice_28: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_60, 1, 58, 116);  view_60 = None
        sub_11: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_29)
        mul_77: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        unsqueeze_44: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_59: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_36: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        relu_7: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.relu.default(convert_element_type_36);  convert_element_type_36 = None
        le_29: "b8[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_29: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.where.self(le_29, full_default, slice_28);  le_29 = slice_28 = None
        convert_element_type_357: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(where_29, torch.float32);  where_29 = None
        squeeze_33: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        unsqueeze_754: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_755: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        sum_90: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_357, [0, 2, 3])
        convert_element_type_35: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_232: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_35, unsqueeze_756);  convert_element_type_35 = unsqueeze_756 = None
        mul_788: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_357, sub_232)
        sum_91: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_788, [0, 2, 3]);  mul_788 = None
        mul_789: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 9.964923469387754e-06)
        unsqueeze_757: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
        unsqueeze_758: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_790: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 9.964923469387754e-06)
        squeeze_34: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_791: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_792: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_790, mul_791);  mul_790 = mul_791 = None
        unsqueeze_760: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
        unsqueeze_761: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
        unsqueeze_762: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
        mul_793: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_763: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
        unsqueeze_764: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_794: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_762);  sub_232 = unsqueeze_762 = None
        sub_234: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_357, mul_794);  convert_element_type_357 = mul_794 = None
        sub_235: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_759);  sub_234 = unsqueeze_759 = None
        mul_795: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_765);  sub_235 = unsqueeze_765 = None
        mul_796: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_34);  sum_91 = squeeze_34 = None
        convert_element_type_359: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_795, torch.bfloat16);  mul_795 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(convert_element_type_359, convert_element_type_33, convert_element_type_34, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_359 = convert_element_type_33 = convert_element_type_34 = None
        getitem_272: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_44[0]
        getitem_273: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_360: "f32[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_273, torch.float32);  getitem_273 = None
        convert_element_type_361: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_272, torch.float32);  getitem_272 = None
        sum_92: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_361, [0, 2, 3])
        convert_element_type_32: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_236: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_32, unsqueeze_768);  convert_element_type_32 = unsqueeze_768 = None
        mul_797: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_361, sub_236)
        sum_93: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 2, 3]);  mul_797 = None
        mul_798: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 9.964923469387754e-06)
        unsqueeze_769: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
        unsqueeze_770: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_799: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 9.964923469387754e-06)
        mul_800: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_801: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
        unsqueeze_772: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
        unsqueeze_773: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
        unsqueeze_774: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
        mul_802: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_775: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
        unsqueeze_776: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_803: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_774);  sub_236 = unsqueeze_774 = None
        sub_238: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_361, mul_803);  convert_element_type_361 = mul_803 = None
        sub_239: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_771);  sub_238 = unsqueeze_771 = None
        mul_804: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_777);  sub_239 = unsqueeze_777 = None
        mul_805: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_31);  sum_93 = squeeze_31 = None
        convert_element_type_363: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_804, torch.bfloat16);  mul_804 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(convert_element_type_363, relu_6, convert_element_type_31, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  convert_element_type_363 = convert_element_type_31 = None
        getitem_275: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_45[0]
        getitem_276: "bf16[58, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_364: "f32[58, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_276, torch.float32);  getitem_276 = None
        le_30: "b8[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_30: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.where.self(le_30, full_default, getitem_275);  le_30 = getitem_275 = None
        convert_element_type_365: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(where_30, torch.float32);  where_30 = None
        sum_94: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_365, [0, 2, 3])
        convert_element_type_29: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_240: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_29, unsqueeze_780);  convert_element_type_29 = unsqueeze_780 = None
        mul_806: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_365, sub_240)
        sum_95: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 2, 3]);  mul_806 = None
        mul_807: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 9.964923469387754e-06)
        unsqueeze_781: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_782: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_808: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 9.964923469387754e-06)
        mul_809: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_810: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_808, mul_809);  mul_808 = mul_809 = None
        unsqueeze_784: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
        unsqueeze_785: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
        unsqueeze_786: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
        mul_811: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_787: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
        unsqueeze_788: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
        unsqueeze_789: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
        mul_812: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_786);  sub_240 = unsqueeze_786 = None
        sub_242: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_365, mul_812);  convert_element_type_365 = mul_812 = None
        sub_243: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_783);  sub_242 = unsqueeze_783 = None
        mul_813: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_789);  sub_243 = unsqueeze_789 = None
        mul_814: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_28);  sum_95 = squeeze_28 = None
        convert_element_type_367: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_813, torch.bfloat16);  mul_813 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(convert_element_type_367, getitem_23, convert_element_type_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_367 = getitem_23 = convert_element_type_28 = None
        getitem_278: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_46[0]
        getitem_279: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = convolution_backward_46[1];  convolution_backward_46 = None
        convert_element_type_368: "f32[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_279, torch.float32);  getitem_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_27: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([slice_27, getitem_278], 1);  slice_27 = getitem_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_61: "bf16[128, 58, 2, 28, 28][90944, 1568, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(cat_27, [128, 58, 2, 28, 28]);  cat_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_35: "bf16[128, 2, 58, 28, 28][90944, 784, 1568, 28, 1]cuda:0" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3, 4]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_30: "bf16[128, 2, 58, 28, 28][90944, 45472, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_62: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [128, 116, 28, 28]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_29: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_62, 1, 0, 58)
        slice_30: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_62, 1, 58, 116);  view_62 = None
        sub_8: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_21)
        mul_56: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        unsqueeze_32: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_44: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        convert_element_type_27: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None
        relu_5: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.relu.default(convert_element_type_27);  convert_element_type_27 = None
        le_31: "b8[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_31: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.where.self(le_31, full_default, slice_30);  le_31 = slice_30 = None
        convert_element_type_369: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(where_31, torch.float32);  where_31 = None
        squeeze_24: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_790: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_791: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
        unsqueeze_792: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
        sum_96: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_369, [0, 2, 3])
        convert_element_type_26: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        sub_244: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, unsqueeze_792);  convert_element_type_26 = unsqueeze_792 = None
        mul_815: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_369, sub_244)
        sum_97: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2, 3]);  mul_815 = None
        mul_816: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 9.964923469387754e-06)
        unsqueeze_793: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
        unsqueeze_794: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
        unsqueeze_795: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
        mul_817: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 9.964923469387754e-06)
        squeeze_25: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_818: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_819: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_817, mul_818);  mul_817 = mul_818 = None
        unsqueeze_796: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
        unsqueeze_797: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
        unsqueeze_798: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
        mul_820: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_799: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
        unsqueeze_800: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
        unsqueeze_801: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
        mul_821: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_798);  sub_244 = unsqueeze_798 = None
        sub_246: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_369, mul_821);  convert_element_type_369 = mul_821 = None
        sub_247: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_795);  sub_246 = unsqueeze_795 = None
        mul_822: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_801);  sub_247 = unsqueeze_801 = None
        mul_823: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_25);  sum_97 = squeeze_25 = None
        convert_element_type_371: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_822, torch.bfloat16);  mul_822 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(convert_element_type_371, convert_element_type_24, convert_element_type_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_371 = convert_element_type_24 = convert_element_type_25 = None
        getitem_281: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_47[0]
        getitem_282: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = convolution_backward_47[1];  convolution_backward_47 = None
        convert_element_type_372: "f32[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_282, torch.float32);  getitem_282 = None
        convert_element_type_373: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_281, torch.float32);  getitem_281 = None
        sum_98: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_373, [0, 2, 3])
        convert_element_type_23: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_248: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, unsqueeze_804);  convert_element_type_23 = unsqueeze_804 = None
        mul_824: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_373, sub_248)
        sum_99: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_824, [0, 2, 3]);  mul_824 = None
        mul_825: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 9.964923469387754e-06)
        unsqueeze_805: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
        unsqueeze_806: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
        unsqueeze_807: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
        mul_826: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 9.964923469387754e-06)
        mul_827: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_828: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_826, mul_827);  mul_826 = mul_827 = None
        unsqueeze_808: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
        unsqueeze_809: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
        unsqueeze_810: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
        mul_829: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_811: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
        unsqueeze_812: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
        unsqueeze_813: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
        mul_830: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_810);  sub_248 = unsqueeze_810 = None
        sub_250: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_373, mul_830);  convert_element_type_373 = mul_830 = None
        sub_251: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_807);  sub_250 = unsqueeze_807 = None
        mul_831: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_813);  sub_251 = unsqueeze_813 = None
        mul_832: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_22);  sum_99 = squeeze_22 = None
        convert_element_type_375: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_831, torch.bfloat16);  mul_831 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(convert_element_type_375, relu_4, convert_element_type_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  convert_element_type_375 = convert_element_type_22 = None
        getitem_284: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_48[0]
        getitem_285: "bf16[58, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_48[1];  convolution_backward_48 = None
        convert_element_type_376: "f32[58, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_285, torch.float32);  getitem_285 = None
        le_32: "b8[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_32: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.where.self(le_32, full_default, getitem_284);  le_32 = getitem_284 = None
        convert_element_type_377: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(where_32, torch.float32);  where_32 = None
        sum_100: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_377, [0, 2, 3])
        convert_element_type_20: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_252: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, unsqueeze_816);  convert_element_type_20 = unsqueeze_816 = None
        mul_833: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_377, sub_252)
        sum_101: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 2, 3]);  mul_833 = None
        mul_834: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 9.964923469387754e-06)
        unsqueeze_817: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
        unsqueeze_818: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
        unsqueeze_819: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
        mul_835: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 9.964923469387754e-06)
        mul_836: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_837: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_835, mul_836);  mul_835 = mul_836 = None
        unsqueeze_820: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
        unsqueeze_821: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
        unsqueeze_822: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
        mul_838: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_823: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
        unsqueeze_824: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
        unsqueeze_825: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
        mul_839: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_822);  sub_252 = unsqueeze_822 = None
        sub_254: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_377, mul_839);  convert_element_type_377 = mul_839 = None
        sub_255: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_819);  sub_254 = unsqueeze_819 = None
        mul_840: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_825);  sub_255 = unsqueeze_825 = None
        mul_841: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_19);  sum_101 = squeeze_19 = None
        convert_element_type_379: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_840, torch.bfloat16);  mul_840 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(convert_element_type_379, getitem_15, convert_element_type_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_379 = getitem_15 = convert_element_type_19 = None
        getitem_287: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_49[0]
        getitem_288: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = convolution_backward_49[1];  convolution_backward_49 = None
        convert_element_type_380: "f32[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_288, torch.float32);  getitem_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_28: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([slice_29, getitem_287], 1);  slice_29 = getitem_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_63: "bf16[128, 58, 2, 28, 28][90944, 1568, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(cat_28, [128, 58, 2, 28, 28]);  cat_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_36: "bf16[128, 2, 58, 28, 28][90944, 784, 1568, 28, 1]cuda:0" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_31: "bf16[128, 2, 58, 28, 28][90944, 45472, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_64: "bf16[128, 116, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [128, 116, 28, 28]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        slice_31: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_64, 1, 0, 58)
        slice_32: "bf16[128, 58, 28, 28][90944, 784, 28, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_64, 1, 58, 116);  view_64 = None
        sub_5: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
        mul_35: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        unsqueeze_20: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_18: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        relu_3: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.relu.default(convert_element_type_18);  convert_element_type_18 = None
        le_33: "b8[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_33: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.where.self(le_33, full_default, slice_32);  le_33 = slice_32 = None
        convert_element_type_381: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(where_33, torch.float32);  where_33 = None
        squeeze_15: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        unsqueeze_826: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_827: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
        unsqueeze_828: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
        sum_102: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_381, [0, 2, 3])
        convert_element_type_17: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_256: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_17, unsqueeze_828);  convert_element_type_17 = unsqueeze_828 = None
        mul_842: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_381, sub_256)
        sum_103: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_842, [0, 2, 3]);  mul_842 = None
        mul_843: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 9.964923469387754e-06)
        unsqueeze_829: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
        unsqueeze_830: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
        unsqueeze_831: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
        mul_844: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 9.964923469387754e-06)
        squeeze_16: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_845: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_846: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
        unsqueeze_832: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
        unsqueeze_833: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
        unsqueeze_834: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
        mul_847: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_835: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
        unsqueeze_836: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
        unsqueeze_837: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
        mul_848: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_834);  sub_256 = unsqueeze_834 = None
        sub_258: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_381, mul_848);  convert_element_type_381 = mul_848 = None
        sub_259: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_831);  sub_258 = unsqueeze_831 = None
        mul_849: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_837);  sub_259 = unsqueeze_837 = None
        mul_850: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_16);  sum_103 = squeeze_16 = None
        convert_element_type_383: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_849, torch.bfloat16);  mul_849 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(convert_element_type_383, convert_element_type_15, convert_element_type_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_383 = convert_element_type_15 = convert_element_type_16 = None
        getitem_290: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = convolution_backward_50[0]
        getitem_291: "bf16[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = convolution_backward_50[1];  convolution_backward_50 = None
        convert_element_type_384: "f32[58, 58, 1, 1][58, 1, 58, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_291, torch.float32);  getitem_291 = None
        convert_element_type_385: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_290, torch.float32);  getitem_290 = None
        sum_104: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_385, [0, 2, 3])
        convert_element_type_14: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_260: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, unsqueeze_840);  convert_element_type_14 = unsqueeze_840 = None
        mul_851: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_385, sub_260)
        sum_105: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 2, 3]);  mul_851 = None
        mul_852: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 9.964923469387754e-06)
        unsqueeze_841: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
        unsqueeze_842: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
        unsqueeze_843: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
        mul_853: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 9.964923469387754e-06)
        mul_854: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_855: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_853, mul_854);  mul_853 = mul_854 = None
        unsqueeze_844: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
        unsqueeze_845: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
        unsqueeze_846: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
        mul_856: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_847: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
        unsqueeze_848: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
        unsqueeze_849: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
        mul_857: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_846);  sub_260 = unsqueeze_846 = None
        sub_262: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_385, mul_857);  convert_element_type_385 = mul_857 = None
        sub_263: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_843);  sub_262 = unsqueeze_843 = None
        mul_858: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_849);  sub_263 = unsqueeze_849 = None
        mul_859: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_13);  sum_105 = squeeze_13 = None
        convert_element_type_387: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_858, torch.bfloat16);  mul_858 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(convert_element_type_387, relu_2, convert_element_type_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  convert_element_type_387 = convert_element_type_13 = None
        getitem_293: "bf16[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = convolution_backward_51[0]
        getitem_294: "bf16[58, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_51[1];  convolution_backward_51 = None
        convert_element_type_388: "f32[58, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_294, torch.float32);  getitem_294 = None
        le_34: "b8[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_34: "bf16[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.aten.where.self(le_34, full_default, getitem_293);  le_34 = getitem_293 = None
        convert_element_type_389: "f32[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.prims.convert_element_type.default(where_34, torch.float32);  where_34 = None
        sum_106: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_389, [0, 2, 3])
        convert_element_type_11: "f32[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_264: "f32[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_11, unsqueeze_852);  convert_element_type_11 = unsqueeze_852 = None
        mul_860: "f32[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_389, sub_264)
        sum_107: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_860, [0, 2, 3]);  mul_860 = None
        mul_861: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 2.4912308673469386e-06)
        unsqueeze_853: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
        unsqueeze_854: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
        unsqueeze_855: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
        mul_862: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, 2.4912308673469386e-06)
        mul_863: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_864: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
        unsqueeze_856: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
        unsqueeze_857: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
        unsqueeze_858: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
        mul_865: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_859: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
        unsqueeze_860: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
        unsqueeze_861: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
        mul_866: "f32[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_858);  sub_264 = unsqueeze_858 = None
        sub_266: "f32[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_389, mul_866);  convert_element_type_389 = mul_866 = None
        sub_267: "f32[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_855);  sub_266 = unsqueeze_855 = None
        mul_867: "f32[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_861);  sub_267 = unsqueeze_861 = None
        mul_868: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, squeeze_10);  sum_107 = squeeze_10 = None
        convert_element_type_391: "bf16[128, 58, 56, 56][181888, 1, 3248, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_867, torch.bfloat16);  mul_867 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(convert_element_type_391, getitem_2, convert_element_type_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_391 = convert_element_type_10 = None
        getitem_296: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_52[0]
        getitem_297: "bf16[58, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_52[1];  convolution_backward_52 = None
        convert_element_type_392: "f32[58, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_297, torch.float32);  getitem_297 = None
        sub_2: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_14: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        unsqueeze_8: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[58, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[58, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_9: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        relu_1: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.relu.default(convert_element_type_9);  convert_element_type_9 = None
        le_35: "b8[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_35: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.where.self(le_35, full_default, slice_31);  le_35 = slice_31 = None
        convert_element_type_393: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(where_35, torch.float32);  where_35 = None
        squeeze_6: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        unsqueeze_862: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_863: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
        unsqueeze_864: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
        sum_108: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_393, [0, 2, 3])
        convert_element_type_8: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_268: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_8, unsqueeze_864);  convert_element_type_8 = unsqueeze_864 = None
        mul_869: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_393, sub_268)
        sum_109: "f32[58][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 2, 3]);  mul_869 = None
        mul_870: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 9.964923469387754e-06)
        unsqueeze_865: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
        unsqueeze_866: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
        unsqueeze_867: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
        mul_871: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 9.964923469387754e-06)
        squeeze_7: "f32[58][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_872: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_873: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_871, mul_872);  mul_871 = mul_872 = None
        unsqueeze_868: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
        unsqueeze_869: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
        unsqueeze_870: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
        mul_874: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_871: "f32[1, 58][58, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
        unsqueeze_872: "f32[1, 58, 1][58, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
        unsqueeze_873: "f32[1, 58, 1, 1][58, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
        mul_875: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_870);  sub_268 = unsqueeze_870 = None
        sub_270: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_393, mul_875);  convert_element_type_393 = mul_875 = None
        sub_271: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_867);  sub_270 = unsqueeze_867 = None
        mul_876: "f32[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_873);  sub_271 = unsqueeze_873 = None
        mul_877: "f32[58][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_7);  sum_109 = squeeze_7 = None
        convert_element_type_395: "bf16[128, 58, 28, 28][45472, 1, 1624, 58]cuda:0" = torch.ops.prims.convert_element_type.default(mul_876, torch.bfloat16);  mul_876 = None
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(convert_element_type_395, convert_element_type_6, convert_element_type_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_395 = convert_element_type_6 = convert_element_type_7 = None
        getitem_299: "bf16[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = convolution_backward_53[0]
        getitem_300: "bf16[58, 24, 1, 1][24, 1, 24, 24]cuda:0" = convolution_backward_53[1];  convolution_backward_53 = None
        convert_element_type_396: "f32[58, 24, 1, 1][24, 1, 24, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_300, torch.float32);  getitem_300 = None
        convert_element_type_397: "f32[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_299, torch.float32);  getitem_299 = None
        sum_110: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_397, [0, 2, 3])
        convert_element_type_5: "f32[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_272: "f32[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_876);  convert_element_type_5 = unsqueeze_876 = None
        mul_878: "f32[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_397, sub_272)
        sum_111: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 2, 3]);  mul_878 = None
        mul_879: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, 9.964923469387754e-06)
        unsqueeze_877: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
        unsqueeze_878: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
        unsqueeze_879: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
        mul_880: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 9.964923469387754e-06)
        mul_881: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_882: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_880, mul_881);  mul_880 = mul_881 = None
        unsqueeze_880: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
        unsqueeze_881: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
        unsqueeze_882: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
        mul_883: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_883: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
        unsqueeze_884: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
        unsqueeze_885: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
        mul_884: "f32[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_882);  sub_272 = unsqueeze_882 = None
        sub_274: "f32[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_397, mul_884);  convert_element_type_397 = mul_884 = None
        sub_275: "f32[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_879);  sub_274 = unsqueeze_879 = None
        mul_885: "f32[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_885);  sub_275 = unsqueeze_885 = None
        mul_886: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, squeeze_4);  sum_111 = squeeze_4 = None
        convert_element_type_399: "bf16[128, 24, 28, 28][18816, 1, 672, 24]cuda:0" = torch.ops.prims.convert_element_type.default(mul_885, torch.bfloat16);  mul_885 = None
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(convert_element_type_399, getitem_2, convert_element_type_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False]);  convert_element_type_399 = getitem_2 = convert_element_type_4 = None
        getitem_302: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = convolution_backward_54[0]
        getitem_303: "bf16[24, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_54[1];  convolution_backward_54 = None
        add_282: "bf16[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.aten.add.Tensor(getitem_296, getitem_302);  getitem_296 = getitem_302 = None
        convert_element_type_400: "f32[24, 1, 3, 3][9, 1, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_303, torch.float32);  getitem_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:156 in _forward_impl, code: x = self.maxpool(x)
        full_default_36: "f32[3072, 12544][12544, 1]cuda:0" = torch.ops.aten.full.default([3072, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_65: "bf16[3072, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(add_282, [3072, 3136]);  add_282 = None
        _low_memory_max_pool_offsets_to_indices: "i64[128, 24, 56, 56][75264, 1, 1344, 24]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        view_66: "i64[3072, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [3072, 3136]);  _low_memory_max_pool_offsets_to_indices = None
        convert_element_type_401: "f32[3072, 3136][3136, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_65, torch.float32);  view_65 = None
        scatter_add: "f32[3072, 12544][12544, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_36, 1, view_66, convert_element_type_401);  full_default_36 = view_66 = convert_element_type_401 = None
        view_67: "f32[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [128, 24, 112, 112]);  scatter_add = None
        convert_element_type_402: "bf16[128, 24, 112, 112][301056, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_67, torch.bfloat16);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:155 in _forward_impl, code: x = self.conv1(x)
        sub: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[24, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[24, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu: "bf16[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        le_36: "b8[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_36: "bf16[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.where.self(le_36, full_default, convert_element_type_402);  le_36 = full_default = convert_element_type_402 = None
        convert_element_type_403: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.prims.convert_element_type.default(where_36, torch.float32);  where_36 = None
        squeeze: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_886: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_887: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
        unsqueeze_888: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
        sum_112: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_403, [0, 2, 3])
        convert_element_type_2: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_276: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_888);  convert_element_type_2 = unsqueeze_888 = None
        mul_887: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, sub_276)
        sum_113: "f32[24][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 2, 3]);  mul_887 = None
        mul_888: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 6.228077168367346e-07)
        unsqueeze_889: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
        unsqueeze_890: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
        unsqueeze_891: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
        mul_889: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 6.228077168367346e-07)
        squeeze_1: "f32[24][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_890: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_891: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
        unsqueeze_892: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
        unsqueeze_893: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
        unsqueeze_894: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
        mul_892: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_895: "f32[1, 24][24, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
        unsqueeze_896: "f32[1, 24, 1][24, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
        unsqueeze_897: "f32[1, 24, 1, 1][24, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
        mul_893: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_894);  sub_276 = unsqueeze_894 = None
        sub_278: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_403, mul_893);  convert_element_type_403 = mul_893 = None
        sub_279: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.sub.Tensor(sub_278, unsqueeze_891);  sub_278 = unsqueeze_891 = None
        mul_894: "f32[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_897);  sub_279 = unsqueeze_897 = None
        mul_895: "f32[24][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_1);  sum_113 = squeeze_1 = None
        convert_element_type_405: "bf16[128, 24, 112, 112][301056, 1, 2688, 24]cuda:0" = torch.ops.prims.convert_element_type.default(mul_894, torch.bfloat16);  mul_894 = None
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(convert_element_type_405, convert_element_type_1, convert_element_type, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_405 = convert_element_type_1 = convert_element_type = None
        getitem_306: "bf16[24, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_55[1];  convolution_backward_55 = None
        convert_element_type_406: "f32[24, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_306, torch.float32);  getitem_306 = None
        return (convert_element_type_406, None, None, None, None, mul_895, sum_112, convert_element_type_400, None, None, None, mul_886, sum_110, convert_element_type_396, None, None, None, mul_877, sum_108, convert_element_type_392, None, None, None, mul_868, sum_106, convert_element_type_388, None, None, None, mul_859, sum_104, convert_element_type_384, None, None, None, mul_850, sum_102, convert_element_type_380, None, None, None, mul_841, sum_100, convert_element_type_376, None, None, None, mul_832, sum_98, convert_element_type_372, None, None, None, mul_823, sum_96, convert_element_type_368, None, None, None, mul_814, sum_94, convert_element_type_364, None, None, None, mul_805, sum_92, convert_element_type_360, None, None, None, mul_796, sum_90, convert_element_type_356, None, None, None, mul_787, sum_88, convert_element_type_352, None, None, None, mul_778, sum_86, convert_element_type_348, None, None, None, mul_769, sum_84, convert_element_type_344, None, None, None, mul_760, sum_82, convert_element_type_340, None, None, None, mul_751, sum_80, convert_element_type_336, None, None, None, mul_742, sum_78, convert_element_type_332, None, None, None, mul_733, sum_76, convert_element_type_328, None, None, None, mul_724, sum_74, convert_element_type_324, None, None, None, mul_715, sum_72, convert_element_type_320, None, None, None, mul_706, sum_70, convert_element_type_316, None, None, None, mul_697, sum_68, convert_element_type_312, None, None, None, mul_688, sum_66, convert_element_type_308, None, None, None, mul_679, sum_64, convert_element_type_304, None, None, None, mul_670, sum_62, convert_element_type_300, None, None, None, mul_661, sum_60, convert_element_type_296, None, None, None, mul_652, sum_58, convert_element_type_292, None, None, None, mul_643, sum_56, convert_element_type_288, None, None, None, mul_634, sum_54, convert_element_type_284, None, None, None, mul_625, sum_52, convert_element_type_280, None, None, None, mul_616, sum_50, convert_element_type_276, None, None, None, mul_607, sum_48, convert_element_type_272, None, None, None, mul_598, sum_46, convert_element_type_268, None, None, None, mul_589, sum_44, convert_element_type_264, None, None, None, mul_580, sum_42, convert_element_type_260, None, None, None, mul_571, sum_40, convert_element_type_256, None, None, None, mul_562, sum_38, convert_element_type_252, None, None, None, mul_553, sum_36, convert_element_type_248, None, None, None, mul_544, sum_34, convert_element_type_244, None, None, None, mul_535, sum_32, convert_element_type_240, None, None, None, mul_526, sum_30, convert_element_type_236, None, None, None, mul_517, sum_28, convert_element_type_232, None, None, None, mul_508, sum_26, convert_element_type_228, None, None, None, mul_499, sum_24, convert_element_type_224, None, None, None, mul_490, sum_22, convert_element_type_220, None, None, None, mul_481, sum_20, convert_element_type_216, None, None, None, mul_472, sum_18, convert_element_type_212, None, None, None, mul_463, sum_16, convert_element_type_208, None, None, None, mul_454, sum_14, convert_element_type_204, None, None, None, mul_445, sum_12, convert_element_type_200, None, None, None, mul_436, sum_10, convert_element_type_196, None, None, None, mul_427, sum_8, convert_element_type_192, None, None, None, mul_418, sum_6, convert_element_type_188, None, None, None, mul_409, sum_4, convert_element_type_184, None, None, None, mul_400, sum_2, convert_element_type_179, convert_element_type_180)
