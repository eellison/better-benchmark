class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[64][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_18: "f32[96][1]cuda:0", primals_24: "f32[96][1]cuda:0", primals_29: "f32[96][1]cuda:0", primals_35: "f32[96][1]cuda:0", primals_41: "f32[96][1]cuda:0", primals_47: "f32[192][1]cuda:0", primals_53: "f32[192][1]cuda:0", primals_58: "f32[192][1]cuda:0", primals_64: "f32[192][1]cuda:0", primals_70: "f32[192][1]cuda:0", primals_75: "f32[192][1]cuda:0", primals_81: "f32[192][1]cuda:0", primals_87: "f32[192][1]cuda:0", primals_92: "f32[192][1]cuda:0", primals_98: "f32[192][1]cuda:0", primals_104: "f32[192][1]cuda:0", primals_110: "f32[384][1]cuda:0", primals_116: "f32[384][1]cuda:0", primals_121: "f32[384][1]cuda:0", primals_127: "f32[384][1]cuda:0", primals_133: "f32[384][1]cuda:0", primals_138: "f32[384][1]cuda:0", primals_144: "f32[384][1]cuda:0", primals_150: "f32[384][1]cuda:0", primals_155: "f32[384][1]cuda:0", primals_161: "f32[384][1]cuda:0", primals_167: "f32[384][1]cuda:0", primals_172: "f32[384][1]cuda:0", primals_178: "f32[384][1]cuda:0", primals_184: "f32[384][1]cuda:0", primals_189: "f32[384][1]cuda:0", primals_195: "f32[384][1]cuda:0", primals_201: "f32[384][1]cuda:0", primals_206: "f32[384][1]cuda:0", primals_212: "f32[384][1]cuda:0", primals_218: "f32[384][1]cuda:0", primals_223: "f32[384][1]cuda:0", primals_229: "f32[384][1]cuda:0", primals_235: "f32[384][1]cuda:0", primals_240: "f32[384][1]cuda:0", primals_246: "f32[384][1]cuda:0", primals_252: "f32[384][1]cuda:0", primals_257: "f32[384][1]cuda:0", primals_263: "f32[384][1]cuda:0", primals_269: "f32[384][1]cuda:0", primals_274: "f32[384][1]cuda:0", primals_280: "f32[384][1]cuda:0", primals_286: "f32[384][1]cuda:0", primals_291: "f32[384][1]cuda:0", primals_297: "f32[384][1]cuda:0", primals_303: "f32[384][1]cuda:0", primals_308: "f32[384][1]cuda:0", primals_314: "f32[384][1]cuda:0", primals_320: "f32[384][1]cuda:0", primals_325: "f32[384][1]cuda:0", primals_331: "f32[384][1]cuda:0", primals_337: "f32[384][1]cuda:0", primals_343: "f32[1408][1]cuda:0", primals_344: "f32[1408][1]cuda:0", primals_349: "f32[1408][1]cuda:0", primals_350: "f32[1408][1]cuda:0", convert_element_type: "bf16[64, 3, 1, 1][3, 1, 3, 3]cuda:0", convert_element_type_1: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0", squeeze_1: "f32[64][1]cuda:0", convert_element_type_4: "bf16[64, 3, 3, 3][27, 1, 9, 3]cuda:0", convolution_1: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0", squeeze_4: "f32[64][1]cuda:0", relu: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0", convert_element_type_8: "bf16[96, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_2: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0", squeeze_7: "f32[96][1]cuda:0", convert_element_type_11: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_3: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0", squeeze_10: "f32[96][1]cuda:0", relu_1: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0", squeeze_13: "f32[96][1]cuda:0", convert_element_type_16: "bf16[96, 96, 1, 1][96, 1, 96, 96]cuda:0", convolution_4: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0", squeeze_16: "f32[96][1]cuda:0", convert_element_type_19: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0", convolution_5: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0", squeeze_19: "f32[96][1]cuda:0", relu_2: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0", convert_element_type_22: "bf16[192, 96, 1, 1][96, 1, 96, 96]cuda:0", convolution_6: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_22: "f32[192][1]cuda:0", convert_element_type_25: "bf16[192, 96, 3, 3][864, 1, 288, 96]cuda:0", convolution_7: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_25: "f32[192][1]cuda:0", relu_3: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_28: "f32[192][1]cuda:0", convert_element_type_30: "bf16[192, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_8: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_31: "f32[192][1]cuda:0", convert_element_type_33: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0", convolution_9: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_34: "f32[192][1]cuda:0", relu_4: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_37: "f32[192][1]cuda:0", convert_element_type_38: "bf16[192, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_10: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_40: "f32[192][1]cuda:0", convert_element_type_41: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0", convolution_11: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_43: "f32[192][1]cuda:0", relu_5: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_46: "f32[192][1]cuda:0", convert_element_type_46: "bf16[192, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_12: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_49: "f32[192][1]cuda:0", convert_element_type_49: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0", convolution_13: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", squeeze_52: "f32[192][1]cuda:0", relu_6: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0", convert_element_type_52: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0", convolution_14: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_55: "f32[384][1]cuda:0", convert_element_type_55: "bf16[384, 192, 3, 3][1728, 1, 576, 192]cuda:0", convolution_15: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_58: "f32[384][1]cuda:0", relu_7: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_61: "f32[384][1]cuda:0", convert_element_type_60: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_16: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_64: "f32[384][1]cuda:0", convert_element_type_63: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_17: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_67: "f32[384][1]cuda:0", relu_8: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_70: "f32[384][1]cuda:0", convert_element_type_68: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_18: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_73: "f32[384][1]cuda:0", convert_element_type_71: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_19: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_76: "f32[384][1]cuda:0", relu_9: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_79: "f32[384][1]cuda:0", convert_element_type_76: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_20: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_82: "f32[384][1]cuda:0", convert_element_type_79: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_21: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_85: "f32[384][1]cuda:0", relu_10: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_88: "f32[384][1]cuda:0", convert_element_type_84: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_22: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_91: "f32[384][1]cuda:0", convert_element_type_87: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_23: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_94: "f32[384][1]cuda:0", relu_11: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_97: "f32[384][1]cuda:0", convert_element_type_92: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_24: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_100: "f32[384][1]cuda:0", convert_element_type_95: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_25: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_103: "f32[384][1]cuda:0", relu_12: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_106: "f32[384][1]cuda:0", convert_element_type_100: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_26: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_109: "f32[384][1]cuda:0", convert_element_type_103: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_27: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_112: "f32[384][1]cuda:0", relu_13: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_115: "f32[384][1]cuda:0", convert_element_type_108: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_28: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_118: "f32[384][1]cuda:0", convert_element_type_111: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_29: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_121: "f32[384][1]cuda:0", relu_14: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_124: "f32[384][1]cuda:0", convert_element_type_116: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_30: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_127: "f32[384][1]cuda:0", convert_element_type_119: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_31: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_130: "f32[384][1]cuda:0", relu_15: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_133: "f32[384][1]cuda:0", convert_element_type_124: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_32: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_136: "f32[384][1]cuda:0", convert_element_type_127: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_33: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_139: "f32[384][1]cuda:0", relu_16: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_142: "f32[384][1]cuda:0", convert_element_type_132: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_34: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_145: "f32[384][1]cuda:0", convert_element_type_135: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_35: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_148: "f32[384][1]cuda:0", relu_17: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_151: "f32[384][1]cuda:0", convert_element_type_140: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_36: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_154: "f32[384][1]cuda:0", convert_element_type_143: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_37: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_157: "f32[384][1]cuda:0", relu_18: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_160: "f32[384][1]cuda:0", convert_element_type_148: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_38: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_163: "f32[384][1]cuda:0", convert_element_type_151: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_39: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_166: "f32[384][1]cuda:0", relu_19: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_169: "f32[384][1]cuda:0", convert_element_type_156: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_40: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_172: "f32[384][1]cuda:0", convert_element_type_159: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_41: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", squeeze_175: "f32[384][1]cuda:0", relu_20: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0", convert_element_type_162: "bf16[1408, 384, 1, 1][384, 1, 384, 384]cuda:0", convolution_42: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0", getitem_119: "f32[1, 1408, 1, 1][1408, 1, 1408, 1408]cuda:0", rsqrt_59: "f32[1, 1408, 1, 1][1408, 1, 1408, 1408]cuda:0", convert_element_type_165: "bf16[1408, 384, 3, 3][3456, 1, 1152, 384]cuda:0", convolution_43: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0", getitem_121: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0", rsqrt_60: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0", view: "bf16[128, 1408][1408, 1]cuda:0", permute_1: "bf16[1000, 1408][1408, 1]cuda:0", unsqueeze_270: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_282: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_294: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_306: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_318: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_330: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_342: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_354: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_366: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_378: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_390: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_402: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_414: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_426: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_438: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_450: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_462: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_474: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_486: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_498: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_510: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_522: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_534: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_546: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_558: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_570: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_582: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_594: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_606: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_618: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_630: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_642: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_654: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_666: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_678: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_690: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_702: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_714: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_726: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_738: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_750: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", unsqueeze_762: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_774: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_786: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_798: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_810: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_822: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_834: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_846: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_858: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_870: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_882: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0", unsqueeze_894: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_906: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_918: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_930: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_942: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_954: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_966: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        mm: "bf16[128, 1408][1408, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 1408][1408, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_177: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_178: "f32[1000, 1408][1408, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_179: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_177, torch.float32);  convert_element_type_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_2: "bf16[128, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 1408, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_183: "bf16[128, 1408, 1][1408, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2, 3);  view_2 = None
        squeeze_184: "bf16[128, 1408][1408, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze_183, 2);  squeeze_183 = None
        full: "bf16[180224][1]cuda:0" = torch.ops.aten.full.default([180224], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "bf16[180224][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full, squeeze_184, [128, 1408], [1408, 1], 0);  full = squeeze_184 = None
        as_strided_5: "bf16[128, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 1408, 1, 1], [1408, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "bf16[128, 1408, 7, 7][1408, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 1408, 7, 7]);  as_strided_5 = None
        div: "bf16[128, 1408, 7, 7][68992, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_59: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, getitem_119)
        mul_413: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
        unsqueeze_236: "f32[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_343, -1)
        unsqueeze_237: "f32[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_419: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
        unsqueeze_238: "f32[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_344, -1);  primals_344 = None
        unsqueeze_239: "f32[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_337: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
        convert_element_type_164: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.prims.convert_element_type.default(add_337, torch.bfloat16);  add_337 = None
        sub_60: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, getitem_121)
        mul_420: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
        unsqueeze_240: "f32[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_349, -1)
        unsqueeze_241: "f32[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        mul_426: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
        unsqueeze_242: "f32[1408, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_350, -1);  primals_350 = None
        unsqueeze_243: "f32[1408, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        add_342: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
        convert_element_type_167: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.prims.convert_element_type.default(add_342, torch.bfloat16);  add_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:748 in forward, code: x = self.conv_1x1(x) + self.conv_kxk(x)
        add_343: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_164, convert_element_type_167);  convert_element_type_164 = convert_element_type_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        relu_21: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.relu.default(add_343);  add_343 = None
        le: "b8[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.where.self(le, full_default, div);  le = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_180: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32);  where = None
        squeeze_180: "f32[1408][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        unsqueeze_244: "f32[1, 1408][1408, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
        unsqueeze_245: "f32[1, 1408, 1][1408, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        sum_2: "f32[1408][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_180, [0, 2, 3])
        convert_element_type_166: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32);  convolution_43 = None
        sub_61: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_166, unsqueeze_246);  convert_element_type_166 = unsqueeze_246 = None
        mul_427: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_180, sub_61)
        sum_3: "f32[1408][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
        mul_428: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.00015943877551020407)
        unsqueeze_247: "f32[1, 1408][1408, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
        unsqueeze_248: "f32[1, 1408, 1][1408, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_429: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.00015943877551020407)
        squeeze_181: "f32[1408][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_430: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
        mul_431: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
        unsqueeze_250: "f32[1, 1408][1408, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
        unsqueeze_251: "f32[1, 1408, 1][1408, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_432: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_181, primals_349);  primals_349 = None
        unsqueeze_253: "f32[1, 1408][1408, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
        unsqueeze_254: "f32[1, 1408, 1][1408, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_433: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_252);  sub_61 = unsqueeze_252 = None
        sub_63: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_180, mul_433);  mul_433 = None
        sub_64: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_249);  sub_63 = unsqueeze_249 = None
        mul_434: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_255);  sub_64 = unsqueeze_255 = None
        mul_435: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_181);  sum_3 = squeeze_181 = None
        convert_element_type_182: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16);  mul_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_182, relu_20, convert_element_type_165, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_182 = convert_element_type_165 = None
        getitem_122: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward[0]
        getitem_123: "bf16[1408, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_183: "f32[1408, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_177: "f32[1408][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        unsqueeze_256: "f32[1, 1408][1408, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
        unsqueeze_257: "f32[1, 1408, 1][1408, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
        unsqueeze_258: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
        sum_4: "f32[1408][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_180, [0, 2, 3])
        convert_element_type_163: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32);  convolution_42 = None
        sub_65: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, unsqueeze_258);  convert_element_type_163 = unsqueeze_258 = None
        mul_436: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_180, sub_65)
        sum_5: "f32[1408][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
        mul_437: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.00015943877551020407)
        unsqueeze_259: "f32[1, 1408][1408, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
        unsqueeze_260: "f32[1, 1408, 1][1408, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_438: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.00015943877551020407)
        squeeze_178: "f32[1408][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_439: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
        mul_440: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
        unsqueeze_262: "f32[1, 1408][1408, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
        unsqueeze_263: "f32[1, 1408, 1][1408, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_441: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_178, primals_343);  primals_343 = None
        unsqueeze_265: "f32[1, 1408][1408, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
        unsqueeze_266: "f32[1, 1408, 1][1408, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 1408, 1, 1][1408, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_442: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_264);  sub_65 = unsqueeze_264 = None
        sub_67: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_180, mul_442);  convert_element_type_180 = mul_442 = None
        sub_68: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_261);  sub_67 = unsqueeze_261 = None
        mul_443: "f32[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_267);  sub_68 = unsqueeze_267 = None
        mul_444: "f32[1408][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_178);  sum_5 = squeeze_178 = None
        convert_element_type_186: "bf16[128, 1408, 7, 7][68992, 1, 9856, 1408]cuda:0" = torch.ops.prims.convert_element_type.default(mul_443, torch.bfloat16);  mul_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_186, relu_20, convert_element_type_162, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_186 = convert_element_type_162 = None
        getitem_125: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_1[0]
        getitem_126: "bf16[1408, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        add_344: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_122, getitem_125);  getitem_122 = getitem_125 = None
        convert_element_type_187: "f32[1408, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_126, torch.float32);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_1: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_1: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_1, full_default, add_344);  le_1 = add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_188: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        sum_6: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_188, [0, 2, 3])
        convert_element_type_160: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        sub_69: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_160, unsqueeze_270);  convert_element_type_160 = unsqueeze_270 = None
        mul_445: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, sub_69)
        sum_7: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
        mul_446: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 3.985969387755102e-05)
        unsqueeze_271: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
        unsqueeze_272: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_447: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 3.985969387755102e-05)
        mul_448: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
        mul_449: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
        unsqueeze_274: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
        unsqueeze_275: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        mul_450: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_175, primals_337);  primals_337 = None
        unsqueeze_277: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
        unsqueeze_278: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_451: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_276);  sub_69 = unsqueeze_276 = None
        sub_71: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, mul_451);  mul_451 = None
        sub_72: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_273);  sub_71 = unsqueeze_273 = None
        mul_452: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_279);  sub_72 = unsqueeze_279 = None
        mul_453: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_175);  sum_7 = squeeze_175 = None
        convert_element_type_190: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_452, torch.bfloat16);  mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_190, relu_19, convert_element_type_159, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_190 = convert_element_type_159 = None
        getitem_128: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_2[0]
        getitem_129: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_191: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_8: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_188, [0, 2, 3])
        convert_element_type_157: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32);  convolution_40 = None
        sub_73: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_157, unsqueeze_282);  convert_element_type_157 = unsqueeze_282 = None
        mul_454: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, sub_73)
        sum_9: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
        mul_455: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 3.985969387755102e-05)
        unsqueeze_283: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
        unsqueeze_284: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_456: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 3.985969387755102e-05)
        mul_457: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
        mul_458: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
        unsqueeze_286: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
        unsqueeze_287: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        mul_459: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_172, primals_331);  primals_331 = None
        unsqueeze_289: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
        unsqueeze_290: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_460: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_288);  sub_73 = unsqueeze_288 = None
        sub_75: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, mul_460);  mul_460 = None
        sub_76: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_285);  sub_75 = unsqueeze_285 = None
        mul_461: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_291);  sub_76 = unsqueeze_291 = None
        mul_462: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_172);  sum_9 = squeeze_172 = None
        convert_element_type_194: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_461, torch.bfloat16);  mul_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_194, relu_19, convert_element_type_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_194 = convert_element_type_156 = None
        getitem_131: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_3[0]
        getitem_132: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        add_345: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, getitem_131);  getitem_128 = getitem_131 = None
        convert_element_type_195: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_132, torch.float32);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_10: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_188, [0, 2, 3])
        convert_element_type_154: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_19, torch.float32)
        sub_77: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_154, unsqueeze_294);  convert_element_type_154 = unsqueeze_294 = None
        mul_463: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_188, sub_77)
        sum_11: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
        mul_464: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 3.985969387755102e-05)
        unsqueeze_295: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
        unsqueeze_296: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_465: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 3.985969387755102e-05)
        mul_466: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
        mul_467: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
        unsqueeze_298: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
        unsqueeze_299: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        mul_468: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_169, primals_325);  primals_325 = None
        unsqueeze_301: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
        unsqueeze_302: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_469: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_300);  sub_77 = unsqueeze_300 = None
        sub_79: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_188, mul_469);  convert_element_type_188 = mul_469 = None
        sub_80: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_297);  sub_79 = unsqueeze_297 = None
        mul_470: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_303);  sub_80 = unsqueeze_303 = None
        mul_471: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_169);  sum_11 = squeeze_169 = None
        convert_element_type_198: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_470, torch.bfloat16);  mul_470 = None
        add_346: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_345, convert_element_type_198);  add_345 = convert_element_type_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_2: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_2: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_2, full_default, add_346);  le_2 = add_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_199: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sum_12: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_199, [0, 2, 3])
        convert_element_type_152: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32);  convolution_39 = None
        sub_81: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_152, unsqueeze_306);  convert_element_type_152 = unsqueeze_306 = None
        mul_472: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_199, sub_81)
        sum_13: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
        mul_473: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 3.985969387755102e-05)
        unsqueeze_307: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
        unsqueeze_308: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_474: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 3.985969387755102e-05)
        mul_475: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_476: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
        unsqueeze_310: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
        unsqueeze_311: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        mul_477: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_166, primals_320);  primals_320 = None
        unsqueeze_313: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
        unsqueeze_314: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_478: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_312);  sub_81 = unsqueeze_312 = None
        sub_83: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_199, mul_478);  mul_478 = None
        sub_84: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_309);  sub_83 = unsqueeze_309 = None
        mul_479: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_315);  sub_84 = unsqueeze_315 = None
        mul_480: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_166);  sum_13 = squeeze_166 = None
        convert_element_type_201: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_479, torch.bfloat16);  mul_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_201, relu_18, convert_element_type_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_201 = convert_element_type_151 = None
        getitem_134: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_4[0]
        getitem_135: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_202: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_14: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_199, [0, 2, 3])
        convert_element_type_149: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32);  convolution_38 = None
        sub_85: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_149, unsqueeze_318);  convert_element_type_149 = unsqueeze_318 = None
        mul_481: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_199, sub_85)
        sum_15: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 2, 3]);  mul_481 = None
        mul_482: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 3.985969387755102e-05)
        unsqueeze_319: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
        unsqueeze_320: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_483: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 3.985969387755102e-05)
        mul_484: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_485: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
        unsqueeze_322: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
        unsqueeze_323: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        mul_486: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_163, primals_314);  primals_314 = None
        unsqueeze_325: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
        unsqueeze_326: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_487: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_324);  sub_85 = unsqueeze_324 = None
        sub_87: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_199, mul_487);  mul_487 = None
        sub_88: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_321);  sub_87 = unsqueeze_321 = None
        mul_488: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_327);  sub_88 = unsqueeze_327 = None
        mul_489: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_163);  sum_15 = squeeze_163 = None
        convert_element_type_205: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_488, torch.bfloat16);  mul_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_205, relu_18, convert_element_type_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_205 = convert_element_type_148 = None
        getitem_137: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_5[0]
        getitem_138: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        add_347: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_134, getitem_137);  getitem_134 = getitem_137 = None
        convert_element_type_206: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_138, torch.float32);  getitem_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_16: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_199, [0, 2, 3])
        convert_element_type_146: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_18, torch.float32)
        sub_89: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_146, unsqueeze_330);  convert_element_type_146 = unsqueeze_330 = None
        mul_490: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_199, sub_89)
        sum_17: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3]);  mul_490 = None
        mul_491: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 3.985969387755102e-05)
        unsqueeze_331: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
        unsqueeze_332: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_492: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 3.985969387755102e-05)
        mul_493: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_494: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
        unsqueeze_334: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
        unsqueeze_335: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
        mul_495: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_160, primals_308);  primals_308 = None
        unsqueeze_337: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
        unsqueeze_338: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_496: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_336);  sub_89 = unsqueeze_336 = None
        sub_91: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_199, mul_496);  convert_element_type_199 = mul_496 = None
        sub_92: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_333);  sub_91 = unsqueeze_333 = None
        mul_497: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_339);  sub_92 = unsqueeze_339 = None
        mul_498: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_160);  sum_17 = squeeze_160 = None
        convert_element_type_209: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_497, torch.bfloat16);  mul_497 = None
        add_348: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_347, convert_element_type_209);  add_347 = convert_element_type_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_3: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_3: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_3, full_default, add_348);  le_3 = add_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_210: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        sum_18: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_210, [0, 2, 3])
        convert_element_type_144: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        sub_93: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_144, unsqueeze_342);  convert_element_type_144 = unsqueeze_342 = None
        mul_499: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_210, sub_93)
        sum_19: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
        mul_500: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 3.985969387755102e-05)
        unsqueeze_343: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
        unsqueeze_344: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_501: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 3.985969387755102e-05)
        mul_502: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_503: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
        unsqueeze_346: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
        unsqueeze_347: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        mul_504: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_157, primals_303);  primals_303 = None
        unsqueeze_349: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
        unsqueeze_350: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_505: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_348);  sub_93 = unsqueeze_348 = None
        sub_95: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_210, mul_505);  mul_505 = None
        sub_96: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_345);  sub_95 = unsqueeze_345 = None
        mul_506: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_351);  sub_96 = unsqueeze_351 = None
        mul_507: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_157);  sum_19 = squeeze_157 = None
        convert_element_type_212: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_506, torch.bfloat16);  mul_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_212, relu_17, convert_element_type_143, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_212 = convert_element_type_143 = None
        getitem_140: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_6[0]
        getitem_141: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_213: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_20: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_210, [0, 2, 3])
        convert_element_type_141: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32);  convolution_36 = None
        sub_97: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_141, unsqueeze_354);  convert_element_type_141 = unsqueeze_354 = None
        mul_508: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_210, sub_97)
        sum_21: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 2, 3]);  mul_508 = None
        mul_509: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 3.985969387755102e-05)
        unsqueeze_355: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_509, 0);  mul_509 = None
        unsqueeze_356: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_510: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 3.985969387755102e-05)
        mul_511: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_512: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_510, mul_511);  mul_510 = mul_511 = None
        unsqueeze_358: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
        unsqueeze_359: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        mul_513: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_154, primals_297);  primals_297 = None
        unsqueeze_361: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
        unsqueeze_362: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_514: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_360);  sub_97 = unsqueeze_360 = None
        sub_99: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_210, mul_514);  mul_514 = None
        sub_100: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_357);  sub_99 = unsqueeze_357 = None
        mul_515: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_363);  sub_100 = unsqueeze_363 = None
        mul_516: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_154);  sum_21 = squeeze_154 = None
        convert_element_type_216: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_515, torch.bfloat16);  mul_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_216, relu_17, convert_element_type_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_216 = convert_element_type_140 = None
        getitem_143: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_7[0]
        getitem_144: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        add_349: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_140, getitem_143);  getitem_140 = getitem_143 = None
        convert_element_type_217: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_22: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_210, [0, 2, 3])
        convert_element_type_138: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_17, torch.float32)
        sub_101: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_138, unsqueeze_366);  convert_element_type_138 = unsqueeze_366 = None
        mul_517: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_210, sub_101)
        sum_23: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 2, 3]);  mul_517 = None
        mul_518: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 3.985969387755102e-05)
        unsqueeze_367: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_518, 0);  mul_518 = None
        unsqueeze_368: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_519: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 3.985969387755102e-05)
        mul_520: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_521: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
        unsqueeze_370: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
        unsqueeze_371: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        mul_522: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_151, primals_291);  primals_291 = None
        unsqueeze_373: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
        unsqueeze_374: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_523: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_372);  sub_101 = unsqueeze_372 = None
        sub_103: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_210, mul_523);  convert_element_type_210 = mul_523 = None
        sub_104: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_369);  sub_103 = unsqueeze_369 = None
        mul_524: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_375);  sub_104 = unsqueeze_375 = None
        mul_525: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_151);  sum_23 = squeeze_151 = None
        convert_element_type_220: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_524, torch.bfloat16);  mul_524 = None
        add_350: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_349, convert_element_type_220);  add_349 = convert_element_type_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_4: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_4: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_4, full_default, add_350);  le_4 = add_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_221: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32);  where_4 = None
        sum_24: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_221, [0, 2, 3])
        convert_element_type_136: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32);  convolution_35 = None
        sub_105: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_136, unsqueeze_378);  convert_element_type_136 = unsqueeze_378 = None
        mul_526: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_221, sub_105)
        sum_25: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_526, [0, 2, 3]);  mul_526 = None
        mul_527: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 3.985969387755102e-05)
        unsqueeze_379: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
        unsqueeze_380: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_528: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 3.985969387755102e-05)
        mul_529: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_530: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_528, mul_529);  mul_528 = mul_529 = None
        unsqueeze_382: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
        unsqueeze_383: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        mul_531: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_148, primals_286);  primals_286 = None
        unsqueeze_385: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
        unsqueeze_386: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_532: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_384);  sub_105 = unsqueeze_384 = None
        sub_107: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_221, mul_532);  mul_532 = None
        sub_108: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_381);  sub_107 = unsqueeze_381 = None
        mul_533: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_387);  sub_108 = unsqueeze_387 = None
        mul_534: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_148);  sum_25 = squeeze_148 = None
        convert_element_type_223: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_533, torch.bfloat16);  mul_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_223, relu_16, convert_element_type_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_223 = convert_element_type_135 = None
        getitem_146: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_8[0]
        getitem_147: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_224: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_26: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_221, [0, 2, 3])
        convert_element_type_133: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        sub_109: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, unsqueeze_390);  convert_element_type_133 = unsqueeze_390 = None
        mul_535: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_221, sub_109)
        sum_27: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
        mul_536: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 3.985969387755102e-05)
        unsqueeze_391: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
        unsqueeze_392: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_537: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 3.985969387755102e-05)
        mul_538: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_539: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
        unsqueeze_394: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
        unsqueeze_395: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_540: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_145, primals_280);  primals_280 = None
        unsqueeze_397: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
        unsqueeze_398: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_541: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_396);  sub_109 = unsqueeze_396 = None
        sub_111: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_221, mul_541);  mul_541 = None
        sub_112: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_393);  sub_111 = unsqueeze_393 = None
        mul_542: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_399);  sub_112 = unsqueeze_399 = None
        mul_543: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_145);  sum_27 = squeeze_145 = None
        convert_element_type_227: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_542, torch.bfloat16);  mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_227, relu_16, convert_element_type_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_227 = convert_element_type_132 = None
        getitem_149: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_9[0]
        getitem_150: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        add_351: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_146, getitem_149);  getitem_146 = getitem_149 = None
        convert_element_type_228: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_150, torch.float32);  getitem_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_28: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_221, [0, 2, 3])
        convert_element_type_130: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_16, torch.float32)
        sub_113: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_130, unsqueeze_402);  convert_element_type_130 = unsqueeze_402 = None
        mul_544: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_221, sub_113)
        sum_29: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3]);  mul_544 = None
        mul_545: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 3.985969387755102e-05)
        unsqueeze_403: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
        unsqueeze_404: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_546: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 3.985969387755102e-05)
        mul_547: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_548: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
        unsqueeze_406: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_407: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        mul_549: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_142, primals_274);  primals_274 = None
        unsqueeze_409: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
        unsqueeze_410: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_550: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_408);  sub_113 = unsqueeze_408 = None
        sub_115: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_221, mul_550);  convert_element_type_221 = mul_550 = None
        sub_116: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_405);  sub_115 = unsqueeze_405 = None
        mul_551: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_411);  sub_116 = unsqueeze_411 = None
        mul_552: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_142);  sum_29 = squeeze_142 = None
        convert_element_type_231: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_551, torch.bfloat16);  mul_551 = None
        add_352: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_351, convert_element_type_231);  add_351 = convert_element_type_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_5: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_5: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_5, full_default, add_352);  le_5 = add_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_232: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        sum_30: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_232, [0, 2, 3])
        convert_element_type_128: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32);  convolution_33 = None
        sub_117: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_128, unsqueeze_414);  convert_element_type_128 = unsqueeze_414 = None
        mul_553: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_232, sub_117)
        sum_31: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 2, 3]);  mul_553 = None
        mul_554: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 3.985969387755102e-05)
        unsqueeze_415: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
        unsqueeze_416: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_555: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 3.985969387755102e-05)
        mul_556: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_557: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_555, mul_556);  mul_555 = mul_556 = None
        unsqueeze_418: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
        unsqueeze_419: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        mul_558: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_139, primals_269);  primals_269 = None
        unsqueeze_421: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
        unsqueeze_422: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_559: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_420);  sub_117 = unsqueeze_420 = None
        sub_119: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_232, mul_559);  mul_559 = None
        sub_120: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_417);  sub_119 = unsqueeze_417 = None
        mul_560: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_423);  sub_120 = unsqueeze_423 = None
        mul_561: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_139);  sum_31 = squeeze_139 = None
        convert_element_type_234: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_560, torch.bfloat16);  mul_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_234, relu_15, convert_element_type_127, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_234 = convert_element_type_127 = None
        getitem_152: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_10[0]
        getitem_153: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_235: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_32: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_232, [0, 2, 3])
        convert_element_type_125: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32);  convolution_32 = None
        sub_121: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_125, unsqueeze_426);  convert_element_type_125 = unsqueeze_426 = None
        mul_562: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_232, sub_121)
        sum_33: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2, 3]);  mul_562 = None
        mul_563: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 3.985969387755102e-05)
        unsqueeze_427: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
        unsqueeze_428: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_564: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 3.985969387755102e-05)
        mul_565: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_566: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
        unsqueeze_430: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
        unsqueeze_431: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        mul_567: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_136, primals_263);  primals_263 = None
        unsqueeze_433: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
        unsqueeze_434: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_568: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_432);  sub_121 = unsqueeze_432 = None
        sub_123: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_232, mul_568);  mul_568 = None
        sub_124: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_429);  sub_123 = unsqueeze_429 = None
        mul_569: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_435);  sub_124 = unsqueeze_435 = None
        mul_570: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_136);  sum_33 = squeeze_136 = None
        convert_element_type_238: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_569, torch.bfloat16);  mul_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_238, relu_15, convert_element_type_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_238 = convert_element_type_124 = None
        getitem_155: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_11[0]
        getitem_156: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        add_353: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_152, getitem_155);  getitem_152 = getitem_155 = None
        convert_element_type_239: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_156, torch.float32);  getitem_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_34: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_232, [0, 2, 3])
        convert_element_type_122: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_15, torch.float32)
        sub_125: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_122, unsqueeze_438);  convert_element_type_122 = unsqueeze_438 = None
        mul_571: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_232, sub_125)
        sum_35: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
        mul_572: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 3.985969387755102e-05)
        unsqueeze_439: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
        unsqueeze_440: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_573: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 3.985969387755102e-05)
        mul_574: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_575: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
        unsqueeze_442: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
        unsqueeze_443: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        mul_576: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_133, primals_257);  primals_257 = None
        unsqueeze_445: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
        unsqueeze_446: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_577: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_444);  sub_125 = unsqueeze_444 = None
        sub_127: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_232, mul_577);  convert_element_type_232 = mul_577 = None
        sub_128: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_441);  sub_127 = unsqueeze_441 = None
        mul_578: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_447);  sub_128 = unsqueeze_447 = None
        mul_579: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_133);  sum_35 = squeeze_133 = None
        convert_element_type_242: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_578, torch.bfloat16);  mul_578 = None
        add_354: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_353, convert_element_type_242);  add_353 = convert_element_type_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_6: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_6: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_6, full_default, add_354);  le_6 = add_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_243: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sum_36: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_243, [0, 2, 3])
        convert_element_type_120: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        sub_129: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_120, unsqueeze_450);  convert_element_type_120 = unsqueeze_450 = None
        mul_580: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_243, sub_129)
        sum_37: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_580, [0, 2, 3]);  mul_580 = None
        mul_581: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 3.985969387755102e-05)
        unsqueeze_451: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
        unsqueeze_452: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_582: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 3.985969387755102e-05)
        mul_583: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_584: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
        unsqueeze_454: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
        unsqueeze_455: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
        unsqueeze_456: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
        mul_585: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_130, primals_252);  primals_252 = None
        unsqueeze_457: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
        unsqueeze_458: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_586: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_456);  sub_129 = unsqueeze_456 = None
        sub_131: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_243, mul_586);  mul_586 = None
        sub_132: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_453);  sub_131 = unsqueeze_453 = None
        mul_587: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_459);  sub_132 = unsqueeze_459 = None
        mul_588: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_130);  sum_37 = squeeze_130 = None
        convert_element_type_245: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_587, torch.bfloat16);  mul_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_245, relu_14, convert_element_type_119, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_245 = convert_element_type_119 = None
        getitem_158: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_12[0]
        getitem_159: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_246: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_38: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_243, [0, 2, 3])
        convert_element_type_117: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32);  convolution_30 = None
        sub_133: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_117, unsqueeze_462);  convert_element_type_117 = unsqueeze_462 = None
        mul_589: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_243, sub_133)
        sum_39: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 2, 3]);  mul_589 = None
        mul_590: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 3.985969387755102e-05)
        unsqueeze_463: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
        unsqueeze_464: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_591: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 3.985969387755102e-05)
        mul_592: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_593: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, mul_592);  mul_591 = mul_592 = None
        unsqueeze_466: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
        unsqueeze_467: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        mul_594: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_127, primals_246);  primals_246 = None
        unsqueeze_469: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
        unsqueeze_470: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_595: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_468);  sub_133 = unsqueeze_468 = None
        sub_135: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_243, mul_595);  mul_595 = None
        sub_136: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_465);  sub_135 = unsqueeze_465 = None
        mul_596: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_471);  sub_136 = unsqueeze_471 = None
        mul_597: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_127);  sum_39 = squeeze_127 = None
        convert_element_type_249: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_596, torch.bfloat16);  mul_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_249, relu_14, convert_element_type_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_249 = convert_element_type_116 = None
        getitem_161: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_13[0]
        getitem_162: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        add_355: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, getitem_161);  getitem_158 = getitem_161 = None
        convert_element_type_250: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_162, torch.float32);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_40: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_243, [0, 2, 3])
        convert_element_type_114: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_14, torch.float32)
        sub_137: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_114, unsqueeze_474);  convert_element_type_114 = unsqueeze_474 = None
        mul_598: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_243, sub_137)
        sum_41: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 2, 3]);  mul_598 = None
        mul_599: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 3.985969387755102e-05)
        unsqueeze_475: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
        unsqueeze_476: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_600: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 3.985969387755102e-05)
        mul_601: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_602: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_600, mul_601);  mul_600 = mul_601 = None
        unsqueeze_478: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
        unsqueeze_479: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        mul_603: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_124, primals_240);  primals_240 = None
        unsqueeze_481: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
        unsqueeze_482: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_604: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_480);  sub_137 = unsqueeze_480 = None
        sub_139: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_243, mul_604);  convert_element_type_243 = mul_604 = None
        sub_140: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_477);  sub_139 = unsqueeze_477 = None
        mul_605: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_483);  sub_140 = unsqueeze_483 = None
        mul_606: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_124);  sum_41 = squeeze_124 = None
        convert_element_type_253: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_605, torch.bfloat16);  mul_605 = None
        add_356: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_355, convert_element_type_253);  add_355 = convert_element_type_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_7: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_7: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_7, full_default, add_356);  le_7 = add_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_254: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sum_42: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_254, [0, 2, 3])
        convert_element_type_112: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32);  convolution_29 = None
        sub_141: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_112, unsqueeze_486);  convert_element_type_112 = unsqueeze_486 = None
        mul_607: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_254, sub_141)
        sum_43: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3]);  mul_607 = None
        mul_608: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_42, 3.985969387755102e-05)
        unsqueeze_487: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
        unsqueeze_488: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_609: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, 3.985969387755102e-05)
        mul_610: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_611: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
        unsqueeze_490: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
        unsqueeze_491: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_612: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_121, primals_235);  primals_235 = None
        unsqueeze_493: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_494: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_613: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_492);  sub_141 = unsqueeze_492 = None
        sub_143: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_254, mul_613);  mul_613 = None
        sub_144: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_489);  sub_143 = unsqueeze_489 = None
        mul_614: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_495);  sub_144 = unsqueeze_495 = None
        mul_615: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_43, squeeze_121);  sum_43 = squeeze_121 = None
        convert_element_type_256: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_614, torch.bfloat16);  mul_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_256, relu_13, convert_element_type_111, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_256 = convert_element_type_111 = None
        getitem_164: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_14[0]
        getitem_165: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_257: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_44: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_254, [0, 2, 3])
        convert_element_type_109: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        sub_145: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_109, unsqueeze_498);  convert_element_type_109 = unsqueeze_498 = None
        mul_616: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_254, sub_145)
        sum_45: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 2, 3]);  mul_616 = None
        mul_617: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_44, 3.985969387755102e-05)
        unsqueeze_499: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_617, 0);  mul_617 = None
        unsqueeze_500: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_618: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, 3.985969387755102e-05)
        mul_619: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_620: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_618, mul_619);  mul_618 = mul_619 = None
        unsqueeze_502: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
        unsqueeze_503: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_621: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_118, primals_229);  primals_229 = None
        unsqueeze_505: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
        unsqueeze_506: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_622: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_504);  sub_145 = unsqueeze_504 = None
        sub_147: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_254, mul_622);  mul_622 = None
        sub_148: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_501);  sub_147 = unsqueeze_501 = None
        mul_623: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_507);  sub_148 = unsqueeze_507 = None
        mul_624: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_45, squeeze_118);  sum_45 = squeeze_118 = None
        convert_element_type_260: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_623, torch.bfloat16);  mul_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_260, relu_13, convert_element_type_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_260 = convert_element_type_108 = None
        getitem_167: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_15[0]
        getitem_168: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        add_357: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_164, getitem_167);  getitem_164 = getitem_167 = None
        convert_element_type_261: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_168, torch.float32);  getitem_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_46: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_254, [0, 2, 3])
        convert_element_type_106: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_13, torch.float32)
        sub_149: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_106, unsqueeze_510);  convert_element_type_106 = unsqueeze_510 = None
        mul_625: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_254, sub_149)
        sum_47: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 2, 3]);  mul_625 = None
        mul_626: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_46, 3.985969387755102e-05)
        unsqueeze_511: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
        unsqueeze_512: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_627: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
        mul_628: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_629: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
        unsqueeze_514: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
        unsqueeze_515: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_630: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_115, primals_223);  primals_223 = None
        unsqueeze_517: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
        unsqueeze_518: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_631: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_516);  sub_149 = unsqueeze_516 = None
        sub_151: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_254, mul_631);  convert_element_type_254 = mul_631 = None
        sub_152: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_513);  sub_151 = unsqueeze_513 = None
        mul_632: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_519);  sub_152 = unsqueeze_519 = None
        mul_633: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_47, squeeze_115);  sum_47 = squeeze_115 = None
        convert_element_type_264: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_632, torch.bfloat16);  mul_632 = None
        add_358: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_357, convert_element_type_264);  add_357 = convert_element_type_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_8: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_8: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_8, full_default, add_358);  le_8 = add_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_265: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32);  where_8 = None
        sum_48: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_265, [0, 2, 3])
        convert_element_type_104: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32);  convolution_27 = None
        sub_153: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_104, unsqueeze_522);  convert_element_type_104 = unsqueeze_522 = None
        mul_634: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, sub_153)
        sum_49: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 2, 3]);  mul_634 = None
        mul_635: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 3.985969387755102e-05)
        unsqueeze_523: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
        unsqueeze_524: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_636: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
        mul_637: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_638: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
        unsqueeze_526: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
        unsqueeze_527: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_639: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_112, primals_218);  primals_218 = None
        unsqueeze_529: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
        unsqueeze_530: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_640: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_528);  sub_153 = unsqueeze_528 = None
        sub_155: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_265, mul_640);  mul_640 = None
        sub_156: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_525);  sub_155 = unsqueeze_525 = None
        mul_641: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_531);  sub_156 = unsqueeze_531 = None
        mul_642: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_112);  sum_49 = squeeze_112 = None
        convert_element_type_267: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_641, torch.bfloat16);  mul_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_267, relu_12, convert_element_type_103, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_267 = convert_element_type_103 = None
        getitem_170: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_16[0]
        getitem_171: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_268: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_50: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_265, [0, 2, 3])
        convert_element_type_101: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32);  convolution_26 = None
        sub_157: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, unsqueeze_534);  convert_element_type_101 = unsqueeze_534 = None
        mul_643: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, sub_157)
        sum_51: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 2, 3]);  mul_643 = None
        mul_644: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
        unsqueeze_535: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
        unsqueeze_536: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_645: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 3.985969387755102e-05)
        mul_646: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_647: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
        unsqueeze_538: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
        unsqueeze_539: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_648: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_109, primals_212);  primals_212 = None
        unsqueeze_541: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
        unsqueeze_542: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_649: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_540);  sub_157 = unsqueeze_540 = None
        sub_159: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_265, mul_649);  mul_649 = None
        sub_160: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_537);  sub_159 = unsqueeze_537 = None
        mul_650: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_543);  sub_160 = unsqueeze_543 = None
        mul_651: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_109);  sum_51 = squeeze_109 = None
        convert_element_type_271: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_650, torch.bfloat16);  mul_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_271, relu_12, convert_element_type_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_271 = convert_element_type_100 = None
        getitem_173: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_17[0]
        getitem_174: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        add_359: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_170, getitem_173);  getitem_170 = getitem_173 = None
        convert_element_type_272: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_174, torch.float32);  getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_52: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_265, [0, 2, 3])
        convert_element_type_98: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_12, torch.float32)
        sub_161: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_98, unsqueeze_546);  convert_element_type_98 = unsqueeze_546 = None
        mul_652: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, sub_161)
        sum_53: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 2, 3]);  mul_652 = None
        mul_653: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 3.985969387755102e-05)
        unsqueeze_547: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
        unsqueeze_548: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_654: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 3.985969387755102e-05)
        mul_655: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_656: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
        unsqueeze_550: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
        unsqueeze_551: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        mul_657: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_106, primals_206);  primals_206 = None
        unsqueeze_553: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
        unsqueeze_554: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_658: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_552);  sub_161 = unsqueeze_552 = None
        sub_163: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_265, mul_658);  convert_element_type_265 = mul_658 = None
        sub_164: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_549);  sub_163 = unsqueeze_549 = None
        mul_659: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_555);  sub_164 = unsqueeze_555 = None
        mul_660: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_106);  sum_53 = squeeze_106 = None
        convert_element_type_275: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_659, torch.bfloat16);  mul_659 = None
        add_360: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_359, convert_element_type_275);  add_359 = convert_element_type_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_9: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_9: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_9, full_default, add_360);  le_9 = add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_276: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        sum_54: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_276, [0, 2, 3])
        convert_element_type_96: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        sub_165: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_96, unsqueeze_558);  convert_element_type_96 = unsqueeze_558 = None
        mul_661: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_276, sub_165)
        sum_55: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
        mul_662: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 3.985969387755102e-05)
        unsqueeze_559: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
        unsqueeze_560: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_663: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 3.985969387755102e-05)
        mul_664: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_665: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
        unsqueeze_562: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
        unsqueeze_563: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        mul_666: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_103, primals_201);  primals_201 = None
        unsqueeze_565: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_566: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_667: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_564);  sub_165 = unsqueeze_564 = None
        sub_167: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_276, mul_667);  mul_667 = None
        sub_168: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_561);  sub_167 = unsqueeze_561 = None
        mul_668: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_567);  sub_168 = unsqueeze_567 = None
        mul_669: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_103);  sum_55 = squeeze_103 = None
        convert_element_type_278: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_668, torch.bfloat16);  mul_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_278, relu_11, convert_element_type_95, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_278 = convert_element_type_95 = None
        getitem_176: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_18[0]
        getitem_177: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        convert_element_type_279: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_177, torch.float32);  getitem_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_56: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_276, [0, 2, 3])
        convert_element_type_93: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32);  convolution_24 = None
        sub_169: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_93, unsqueeze_570);  convert_element_type_93 = unsqueeze_570 = None
        mul_670: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_276, sub_169)
        sum_57: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
        mul_671: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 3.985969387755102e-05)
        unsqueeze_571: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
        unsqueeze_572: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_672: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 3.985969387755102e-05)
        mul_673: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_674: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
        unsqueeze_574: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
        unsqueeze_575: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        mul_675: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_100, primals_195);  primals_195 = None
        unsqueeze_577: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_578: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_676: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_576);  sub_169 = unsqueeze_576 = None
        sub_171: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_276, mul_676);  mul_676 = None
        sub_172: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_573);  sub_171 = unsqueeze_573 = None
        mul_677: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_579);  sub_172 = unsqueeze_579 = None
        mul_678: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_100);  sum_57 = squeeze_100 = None
        convert_element_type_282: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_677, torch.bfloat16);  mul_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_282, relu_11, convert_element_type_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_282 = convert_element_type_92 = None
        getitem_179: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_19[0]
        getitem_180: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        add_361: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_176, getitem_179);  getitem_176 = getitem_179 = None
        convert_element_type_283: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_180, torch.float32);  getitem_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_58: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_276, [0, 2, 3])
        convert_element_type_90: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_11, torch.float32)
        sub_173: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_90, unsqueeze_582);  convert_element_type_90 = unsqueeze_582 = None
        mul_679: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_276, sub_173)
        sum_59: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
        mul_680: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 3.985969387755102e-05)
        unsqueeze_583: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
        unsqueeze_584: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_681: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 3.985969387755102e-05)
        mul_682: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_683: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
        unsqueeze_586: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
        unsqueeze_587: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        mul_684: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_97, primals_189);  primals_189 = None
        unsqueeze_589: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_590: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_685: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_588);  sub_173 = unsqueeze_588 = None
        sub_175: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_276, mul_685);  convert_element_type_276 = mul_685 = None
        sub_176: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_585);  sub_175 = unsqueeze_585 = None
        mul_686: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_591);  sub_176 = unsqueeze_591 = None
        mul_687: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_97);  sum_59 = squeeze_97 = None
        convert_element_type_286: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_686, torch.bfloat16);  mul_686 = None
        add_362: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_361, convert_element_type_286);  add_361 = convert_element_type_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_10: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_10: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_10, full_default, add_362);  le_10 = add_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_287: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        sum_60: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_287, [0, 2, 3])
        convert_element_type_88: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32);  convolution_23 = None
        sub_177: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_88, unsqueeze_594);  convert_element_type_88 = unsqueeze_594 = None
        mul_688: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, sub_177)
        sum_61: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3]);  mul_688 = None
        mul_689: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_60, 3.985969387755102e-05)
        unsqueeze_595: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
        unsqueeze_596: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_690: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, 3.985969387755102e-05)
        mul_691: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_692: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
        unsqueeze_598: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
        unsqueeze_599: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        mul_693: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_184);  primals_184 = None
        unsqueeze_601: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
        unsqueeze_602: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_694: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_600);  sub_177 = unsqueeze_600 = None
        sub_179: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_287, mul_694);  mul_694 = None
        sub_180: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_597);  sub_179 = unsqueeze_597 = None
        mul_695: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_603);  sub_180 = unsqueeze_603 = None
        mul_696: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_61, squeeze_94);  sum_61 = squeeze_94 = None
        convert_element_type_289: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_695, torch.bfloat16);  mul_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_289, relu_10, convert_element_type_87, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_289 = convert_element_type_87 = None
        getitem_182: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_20[0]
        getitem_183: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_290: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_183, torch.float32);  getitem_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_62: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_287, [0, 2, 3])
        convert_element_type_85: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        sub_181: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_85, unsqueeze_606);  convert_element_type_85 = unsqueeze_606 = None
        mul_697: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, sub_181)
        sum_63: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
        mul_698: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_62, 3.985969387755102e-05)
        unsqueeze_607: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
        unsqueeze_608: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_699: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, 3.985969387755102e-05)
        mul_700: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_701: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
        unsqueeze_610: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
        unsqueeze_611: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        mul_702: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_178);  primals_178 = None
        unsqueeze_613: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
        unsqueeze_614: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_703: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_612);  sub_181 = unsqueeze_612 = None
        sub_183: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_287, mul_703);  mul_703 = None
        sub_184: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_609);  sub_183 = unsqueeze_609 = None
        mul_704: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_615);  sub_184 = unsqueeze_615 = None
        mul_705: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_63, squeeze_91);  sum_63 = squeeze_91 = None
        convert_element_type_293: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_704, torch.bfloat16);  mul_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(convert_element_type_293, relu_10, convert_element_type_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_293 = convert_element_type_84 = None
        getitem_185: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_21[0]
        getitem_186: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        add_363: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_182, getitem_185);  getitem_182 = getitem_185 = None
        convert_element_type_294: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_186, torch.float32);  getitem_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_64: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_287, [0, 2, 3])
        convert_element_type_82: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_10, torch.float32)
        sub_185: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_82, unsqueeze_618);  convert_element_type_82 = unsqueeze_618 = None
        mul_706: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_287, sub_185)
        sum_65: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3]);  mul_706 = None
        mul_707: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_64, 3.985969387755102e-05)
        unsqueeze_619: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
        unsqueeze_620: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_708: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, 3.985969387755102e-05)
        mul_709: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_710: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
        unsqueeze_622: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
        unsqueeze_623: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        mul_711: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_172);  primals_172 = None
        unsqueeze_625: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_626: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_712: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_624);  sub_185 = unsqueeze_624 = None
        sub_187: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_287, mul_712);  convert_element_type_287 = mul_712 = None
        sub_188: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_621);  sub_187 = unsqueeze_621 = None
        mul_713: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_627);  sub_188 = unsqueeze_627 = None
        mul_714: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_65, squeeze_88);  sum_65 = squeeze_88 = None
        convert_element_type_297: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_713, torch.bfloat16);  mul_713 = None
        add_364: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_363, convert_element_type_297);  add_363 = convert_element_type_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_11: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_11: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_11, full_default, add_364);  le_11 = add_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_298: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sum_66: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_298, [0, 2, 3])
        convert_element_type_80: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32);  convolution_21 = None
        sub_189: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, unsqueeze_630);  convert_element_type_80 = unsqueeze_630 = None
        mul_715: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_298, sub_189)
        sum_67: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
        mul_716: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_66, 3.985969387755102e-05)
        unsqueeze_631: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
        unsqueeze_632: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_717: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, 3.985969387755102e-05)
        mul_718: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_719: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
        unsqueeze_634: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
        unsqueeze_635: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        mul_720: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_167);  primals_167 = None
        unsqueeze_637: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_638: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_721: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_636);  sub_189 = unsqueeze_636 = None
        sub_191: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_298, mul_721);  mul_721 = None
        sub_192: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_633);  sub_191 = unsqueeze_633 = None
        mul_722: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_639);  sub_192 = unsqueeze_639 = None
        mul_723: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_67, squeeze_85);  sum_67 = squeeze_85 = None
        convert_element_type_300: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_722, torch.bfloat16);  mul_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(convert_element_type_300, relu_9, convert_element_type_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_300 = convert_element_type_79 = None
        getitem_188: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_22[0]
        getitem_189: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_301: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_189, torch.float32);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_68: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_298, [0, 2, 3])
        convert_element_type_77: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32);  convolution_20 = None
        sub_193: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_77, unsqueeze_642);  convert_element_type_77 = unsqueeze_642 = None
        mul_724: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_298, sub_193)
        sum_69: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 2, 3]);  mul_724 = None
        mul_725: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_68, 3.985969387755102e-05)
        unsqueeze_643: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
        unsqueeze_644: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_726: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, 3.985969387755102e-05)
        mul_727: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_728: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_726, mul_727);  mul_726 = mul_727 = None
        unsqueeze_646: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
        unsqueeze_647: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
        mul_729: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_161);  primals_161 = None
        unsqueeze_649: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
        unsqueeze_650: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_730: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_648);  sub_193 = unsqueeze_648 = None
        sub_195: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_298, mul_730);  mul_730 = None
        sub_196: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_645);  sub_195 = unsqueeze_645 = None
        mul_731: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_651);  sub_196 = unsqueeze_651 = None
        mul_732: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_69, squeeze_82);  sum_69 = squeeze_82 = None
        convert_element_type_304: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_731, torch.bfloat16);  mul_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_304, relu_9, convert_element_type_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_304 = convert_element_type_76 = None
        getitem_191: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_23[0]
        getitem_192: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        add_365: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_188, getitem_191);  getitem_188 = getitem_191 = None
        convert_element_type_305: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_192, torch.float32);  getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_70: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_298, [0, 2, 3])
        convert_element_type_74: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_9, torch.float32)
        sub_197: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_74, unsqueeze_654);  convert_element_type_74 = unsqueeze_654 = None
        mul_733: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_298, sub_197)
        sum_71: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
        mul_734: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_70, 3.985969387755102e-05)
        unsqueeze_655: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
        unsqueeze_656: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_735: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, 3.985969387755102e-05)
        mul_736: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_737: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
        unsqueeze_658: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
        unsqueeze_659: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        mul_738: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_155);  primals_155 = None
        unsqueeze_661: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_662: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_739: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_660);  sub_197 = unsqueeze_660 = None
        sub_199: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_298, mul_739);  convert_element_type_298 = mul_739 = None
        sub_200: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_657);  sub_199 = unsqueeze_657 = None
        mul_740: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_663);  sub_200 = unsqueeze_663 = None
        mul_741: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_71, squeeze_79);  sum_71 = squeeze_79 = None
        convert_element_type_308: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_740, torch.bfloat16);  mul_740 = None
        add_366: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_365, convert_element_type_308);  add_365 = convert_element_type_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_12: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_12: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_12, full_default, add_366);  le_12 = add_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_309: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32);  where_12 = None
        sum_72: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_309, [0, 2, 3])
        convert_element_type_72: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_201: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_72, unsqueeze_666);  convert_element_type_72 = unsqueeze_666 = None
        mul_742: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_309, sub_201)
        sum_73: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 2, 3]);  mul_742 = None
        mul_743: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_72, 3.985969387755102e-05)
        unsqueeze_667: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
        unsqueeze_668: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_744: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, 3.985969387755102e-05)
        mul_745: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_746: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
        unsqueeze_670: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
        unsqueeze_671: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        mul_747: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_150);  primals_150 = None
        unsqueeze_673: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
        unsqueeze_674: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_748: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_672);  sub_201 = unsqueeze_672 = None
        sub_203: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_309, mul_748);  mul_748 = None
        sub_204: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_669);  sub_203 = unsqueeze_669 = None
        mul_749: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_675);  sub_204 = unsqueeze_675 = None
        mul_750: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_73, squeeze_76);  sum_73 = squeeze_76 = None
        convert_element_type_311: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_749, torch.bfloat16);  mul_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(convert_element_type_311, relu_8, convert_element_type_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_311 = convert_element_type_71 = None
        getitem_194: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_24[0]
        getitem_195: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        convert_element_type_312: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_195, torch.float32);  getitem_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_74: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_309, [0, 2, 3])
        convert_element_type_69: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_205: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_69, unsqueeze_678);  convert_element_type_69 = unsqueeze_678 = None
        mul_751: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_309, sub_205)
        sum_75: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3]);  mul_751 = None
        mul_752: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_74, 3.985969387755102e-05)
        unsqueeze_679: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
        unsqueeze_680: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_753: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, 3.985969387755102e-05)
        mul_754: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_755: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
        unsqueeze_682: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
        unsqueeze_683: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        mul_756: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_144);  primals_144 = None
        unsqueeze_685: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
        unsqueeze_686: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_757: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_684);  sub_205 = unsqueeze_684 = None
        sub_207: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_309, mul_757);  mul_757 = None
        sub_208: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_681);  sub_207 = unsqueeze_681 = None
        mul_758: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_687);  sub_208 = unsqueeze_687 = None
        mul_759: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_75, squeeze_73);  sum_75 = squeeze_73 = None
        convert_element_type_315: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_758, torch.bfloat16);  mul_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(convert_element_type_315, relu_8, convert_element_type_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_315 = convert_element_type_68 = None
        getitem_197: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_25[0]
        getitem_198: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        add_367: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_194, getitem_197);  getitem_194 = getitem_197 = None
        convert_element_type_316: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_198, torch.float32);  getitem_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_76: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_309, [0, 2, 3])
        convert_element_type_66: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_8, torch.float32)
        sub_209: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_66, unsqueeze_690);  convert_element_type_66 = unsqueeze_690 = None
        mul_760: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_309, sub_209)
        sum_77: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 2, 3]);  mul_760 = None
        mul_761: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_76, 3.985969387755102e-05)
        unsqueeze_691: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
        unsqueeze_692: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_762: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, 3.985969387755102e-05)
        mul_763: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_764: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
        unsqueeze_694: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
        unsqueeze_695: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        mul_765: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_138);  primals_138 = None
        unsqueeze_697: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
        unsqueeze_698: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_766: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_696);  sub_209 = unsqueeze_696 = None
        sub_211: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_309, mul_766);  convert_element_type_309 = mul_766 = None
        sub_212: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_693);  sub_211 = unsqueeze_693 = None
        mul_767: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_699);  sub_212 = unsqueeze_699 = None
        mul_768: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_77, squeeze_70);  sum_77 = squeeze_70 = None
        convert_element_type_319: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_767, torch.bfloat16);  mul_767 = None
        add_368: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_367, convert_element_type_319);  add_367 = convert_element_type_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_13: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_13: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_13, full_default, add_368);  le_13 = add_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_320: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sum_78: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_320, [0, 2, 3])
        convert_element_type_64: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        sub_213: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_64, unsqueeze_702);  convert_element_type_64 = unsqueeze_702 = None
        mul_769: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_320, sub_213)
        sum_79: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
        mul_770: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_78, 3.985969387755102e-05)
        unsqueeze_703: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
        unsqueeze_704: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_771: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, 3.985969387755102e-05)
        mul_772: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_773: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
        unsqueeze_706: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
        unsqueeze_707: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
        mul_774: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_133);  primals_133 = None
        unsqueeze_709: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_710: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_775: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_708);  sub_213 = unsqueeze_708 = None
        sub_215: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_320, mul_775);  mul_775 = None
        sub_216: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_705);  sub_215 = unsqueeze_705 = None
        mul_776: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_711);  sub_216 = unsqueeze_711 = None
        mul_777: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_79, squeeze_67);  sum_79 = squeeze_67 = None
        convert_element_type_322: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_776, torch.bfloat16);  mul_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_322, relu_7, convert_element_type_63, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_322 = convert_element_type_63 = None
        getitem_200: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_26[0]
        getitem_201: "bf16[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_323: "f32[384, 384, 3, 3][3456, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_201, torch.float32);  getitem_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_80: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_320, [0, 2, 3])
        convert_element_type_61: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_217: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_61, unsqueeze_714);  convert_element_type_61 = unsqueeze_714 = None
        mul_778: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_320, sub_217)
        sum_81: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
        mul_779: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_80, 3.985969387755102e-05)
        unsqueeze_715: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
        unsqueeze_716: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_780: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
        mul_781: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_782: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
        unsqueeze_718: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
        unsqueeze_719: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        mul_783: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_127);  primals_127 = None
        unsqueeze_721: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
        unsqueeze_722: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_784: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_720);  sub_217 = unsqueeze_720 = None
        sub_219: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_320, mul_784);  mul_784 = None
        sub_220: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_717);  sub_219 = unsqueeze_717 = None
        mul_785: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_723);  sub_220 = unsqueeze_723 = None
        mul_786: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_81, squeeze_64);  sum_81 = squeeze_64 = None
        convert_element_type_326: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_785, torch.bfloat16);  mul_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(convert_element_type_326, relu_7, convert_element_type_60, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_326 = convert_element_type_60 = None
        getitem_203: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = convolution_backward_27[0]
        getitem_204: "bf16[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        add_369: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(getitem_200, getitem_203);  getitem_200 = getitem_203 = None
        convert_element_type_327: "f32[384, 384, 1, 1][384, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_204, torch.float32);  getitem_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_82: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_320, [0, 2, 3])
        convert_element_type_58: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(relu_7, torch.float32)
        sub_221: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_58, unsqueeze_726);  convert_element_type_58 = unsqueeze_726 = None
        mul_787: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_320, sub_221)
        sum_83: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
        mul_788: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_82, 3.985969387755102e-05)
        unsqueeze_727: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
        unsqueeze_728: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_789: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, 3.985969387755102e-05)
        mul_790: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_791: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
        unsqueeze_730: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
        unsqueeze_731: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        mul_792: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_121);  primals_121 = None
        unsqueeze_733: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
        unsqueeze_734: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_793: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_732);  sub_221 = unsqueeze_732 = None
        sub_223: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_320, mul_793);  convert_element_type_320 = mul_793 = None
        sub_224: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_729);  sub_223 = unsqueeze_729 = None
        mul_794: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_735);  sub_224 = unsqueeze_735 = None
        mul_795: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_83, squeeze_61);  sum_83 = squeeze_61 = None
        convert_element_type_330: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_794, torch.bfloat16);  mul_794 = None
        add_370: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.add.Tensor(add_369, convert_element_type_330);  add_369 = convert_element_type_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_14: "b8[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_14: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.where.self(le_14, full_default, add_370);  le_14 = add_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_331: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32);  where_14 = None
        sum_84: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_331, [0, 2, 3])
        convert_element_type_56: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_225: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_56, unsqueeze_738);  convert_element_type_56 = unsqueeze_738 = None
        mul_796: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_331, sub_225)
        sum_85: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
        mul_797: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_84, 3.985969387755102e-05)
        unsqueeze_739: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
        unsqueeze_740: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_798: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, 3.985969387755102e-05)
        mul_799: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_800: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
        unsqueeze_742: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
        unsqueeze_743: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
        unsqueeze_744: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
        mul_801: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_116);  primals_116 = None
        unsqueeze_745: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
        unsqueeze_746: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_802: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_744);  sub_225 = unsqueeze_744 = None
        sub_227: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, mul_802);  mul_802 = None
        sub_228: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_741);  sub_227 = unsqueeze_741 = None
        mul_803: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_747);  sub_228 = unsqueeze_747 = None
        mul_804: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_85, squeeze_58);  sum_85 = squeeze_58 = None
        convert_element_type_333: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_803, torch.bfloat16);  mul_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(convert_element_type_333, relu_6, convert_element_type_55, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_333 = convert_element_type_55 = None
        getitem_206: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_28[0]
        getitem_207: "bf16[384, 192, 3, 3][1728, 1, 576, 192]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_334: "f32[384, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_207, torch.float32);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_86: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_331, [0, 2, 3])
        convert_element_type_53: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_229: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_53, unsqueeze_750);  convert_element_type_53 = unsqueeze_750 = None
        mul_805: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_331, sub_229)
        sum_87: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
        mul_806: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_86, 3.985969387755102e-05)
        unsqueeze_751: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_752: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_807: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, 3.985969387755102e-05)
        mul_808: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_809: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
        unsqueeze_754: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
        unsqueeze_755: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        mul_810: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_110);  primals_110 = None
        unsqueeze_757: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
        unsqueeze_758: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_811: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_756);  sub_229 = unsqueeze_756 = None
        sub_231: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_331, mul_811);  convert_element_type_331 = mul_811 = None
        sub_232: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_753);  sub_231 = unsqueeze_753 = None
        mul_812: "f32[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_759);  sub_232 = unsqueeze_759 = None
        mul_813: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_87, squeeze_55);  sum_87 = squeeze_55 = None
        convert_element_type_337: "bf16[128, 384, 14, 14][75264, 1, 5376, 384]cuda:0" = torch.ops.prims.convert_element_type.default(mul_812, torch.bfloat16);  mul_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_337, relu_6, convert_element_type_52, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_337 = convert_element_type_52 = None
        getitem_209: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_29[0]
        getitem_210: "bf16[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        add_371: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(getitem_206, getitem_209);  getitem_206 = getitem_209 = None
        convert_element_type_338: "f32[384, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_210, torch.float32);  getitem_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_15: "b8[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_15: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.where.self(le_15, full_default, add_371);  le_15 = add_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_339: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        sum_88: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_339, [0, 2, 3])
        convert_element_type_50: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sub_233: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, unsqueeze_762);  convert_element_type_50 = unsqueeze_762 = None
        mul_814: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_339, sub_233)
        sum_89: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None
        mul_815: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_88, 9.964923469387754e-06)
        unsqueeze_763: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
        unsqueeze_764: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_816: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, 9.964923469387754e-06)
        mul_817: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_818: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
        unsqueeze_766: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
        unsqueeze_767: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
        unsqueeze_768: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
        mul_819: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_104);  primals_104 = None
        unsqueeze_769: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
        unsqueeze_770: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_820: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_768);  sub_233 = unsqueeze_768 = None
        sub_235: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_339, mul_820);  mul_820 = None
        sub_236: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_765);  sub_235 = unsqueeze_765 = None
        mul_821: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_771);  sub_236 = unsqueeze_771 = None
        mul_822: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_89, squeeze_52);  sum_89 = squeeze_52 = None
        convert_element_type_341: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_821, torch.bfloat16);  mul_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(convert_element_type_341, relu_5, convert_element_type_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_341 = convert_element_type_49 = None
        getitem_212: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_30[0]
        getitem_213: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        convert_element_type_342: "f32[192, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_213, torch.float32);  getitem_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_90: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_339, [0, 2, 3])
        convert_element_type_47: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sub_237: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_47, unsqueeze_774);  convert_element_type_47 = unsqueeze_774 = None
        mul_823: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_339, sub_237)
        sum_91: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 2, 3]);  mul_823 = None
        mul_824: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_90, 9.964923469387754e-06)
        unsqueeze_775: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
        unsqueeze_776: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_825: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, 9.964923469387754e-06)
        mul_826: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_827: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_825, mul_826);  mul_825 = mul_826 = None
        unsqueeze_778: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
        unsqueeze_779: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
        unsqueeze_780: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
        mul_828: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_98);  primals_98 = None
        unsqueeze_781: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
        unsqueeze_782: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_829: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_780);  sub_237 = unsqueeze_780 = None
        sub_239: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_339, mul_829);  mul_829 = None
        sub_240: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_777);  sub_239 = unsqueeze_777 = None
        mul_830: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_783);  sub_240 = unsqueeze_783 = None
        mul_831: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_91, squeeze_49);  sum_91 = squeeze_49 = None
        convert_element_type_345: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_830, torch.bfloat16);  mul_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(convert_element_type_345, relu_5, convert_element_type_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_345 = convert_element_type_46 = None
        getitem_215: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_31[0]
        getitem_216: "bf16[192, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        add_372: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(getitem_212, getitem_215);  getitem_212 = getitem_215 = None
        convert_element_type_346: "f32[192, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_216, torch.float32);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_92: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_339, [0, 2, 3])
        convert_element_type_44: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(relu_5, torch.float32)
        sub_241: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_44, unsqueeze_786);  convert_element_type_44 = unsqueeze_786 = None
        mul_832: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_339, sub_241)
        sum_93: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_832, [0, 2, 3]);  mul_832 = None
        mul_833: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_92, 9.964923469387754e-06)
        unsqueeze_787: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
        unsqueeze_788: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
        unsqueeze_789: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
        mul_834: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, 9.964923469387754e-06)
        mul_835: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_836: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_834, mul_835);  mul_834 = mul_835 = None
        unsqueeze_790: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
        unsqueeze_791: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
        unsqueeze_792: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
        mul_837: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_92);  primals_92 = None
        unsqueeze_793: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
        unsqueeze_794: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
        unsqueeze_795: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
        mul_838: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_792);  sub_241 = unsqueeze_792 = None
        sub_243: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_339, mul_838);  convert_element_type_339 = mul_838 = None
        sub_244: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_789);  sub_243 = unsqueeze_789 = None
        mul_839: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_795);  sub_244 = unsqueeze_795 = None
        mul_840: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_93, squeeze_46);  sum_93 = squeeze_46 = None
        convert_element_type_349: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_839, torch.bfloat16);  mul_839 = None
        add_373: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_372, convert_element_type_349);  add_372 = convert_element_type_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_16: "b8[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_16: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.where.self(le_16, full_default, add_373);  le_16 = add_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_350: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        sum_94: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_350, [0, 2, 3])
        convert_element_type_42: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_245: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_42, unsqueeze_798);  convert_element_type_42 = unsqueeze_798 = None
        mul_841: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_350, sub_245)
        sum_95: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_841, [0, 2, 3]);  mul_841 = None
        mul_842: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_94, 9.964923469387754e-06)
        unsqueeze_799: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
        unsqueeze_800: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
        unsqueeze_801: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
        mul_843: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, 9.964923469387754e-06)
        mul_844: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_845: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_843, mul_844);  mul_843 = mul_844 = None
        unsqueeze_802: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
        unsqueeze_803: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
        unsqueeze_804: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
        mul_846: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_87);  primals_87 = None
        unsqueeze_805: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
        unsqueeze_806: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
        unsqueeze_807: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
        mul_847: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_804);  sub_245 = unsqueeze_804 = None
        sub_247: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_350, mul_847);  mul_847 = None
        sub_248: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_801);  sub_247 = unsqueeze_801 = None
        mul_848: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_807);  sub_248 = unsqueeze_807 = None
        mul_849: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_95, squeeze_43);  sum_95 = squeeze_43 = None
        convert_element_type_352: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_848, torch.bfloat16);  mul_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(convert_element_type_352, relu_4, convert_element_type_41, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_352 = convert_element_type_41 = None
        getitem_218: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_32[0]
        getitem_219: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_353: "f32[192, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_219, torch.float32);  getitem_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_96: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_350, [0, 2, 3])
        convert_element_type_39: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_249: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_39, unsqueeze_810);  convert_element_type_39 = unsqueeze_810 = None
        mul_850: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_350, sub_249)
        sum_97: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_850, [0, 2, 3]);  mul_850 = None
        mul_851: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_96, 9.964923469387754e-06)
        unsqueeze_811: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
        unsqueeze_812: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
        unsqueeze_813: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
        mul_852: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, 9.964923469387754e-06)
        mul_853: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_854: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_852, mul_853);  mul_852 = mul_853 = None
        unsqueeze_814: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
        unsqueeze_815: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
        unsqueeze_816: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
        mul_855: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_81);  primals_81 = None
        unsqueeze_817: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
        unsqueeze_818: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
        unsqueeze_819: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
        mul_856: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_816);  sub_249 = unsqueeze_816 = None
        sub_251: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_350, mul_856);  mul_856 = None
        sub_252: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_813);  sub_251 = unsqueeze_813 = None
        mul_857: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_819);  sub_252 = unsqueeze_819 = None
        mul_858: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_97, squeeze_40);  sum_97 = squeeze_40 = None
        convert_element_type_356: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_857, torch.bfloat16);  mul_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_356, relu_4, convert_element_type_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_356 = convert_element_type_38 = None
        getitem_221: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_33[0]
        getitem_222: "bf16[192, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        add_374: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(getitem_218, getitem_221);  getitem_218 = getitem_221 = None
        convert_element_type_357: "f32[192, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_222, torch.float32);  getitem_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_98: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_350, [0, 2, 3])
        convert_element_type_36: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(relu_4, torch.float32)
        sub_253: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_36, unsqueeze_822);  convert_element_type_36 = unsqueeze_822 = None
        mul_859: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_350, sub_253)
        sum_99: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_859, [0, 2, 3]);  mul_859 = None
        mul_860: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_98, 9.964923469387754e-06)
        unsqueeze_823: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_860, 0);  mul_860 = None
        unsqueeze_824: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
        unsqueeze_825: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
        mul_861: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, 9.964923469387754e-06)
        mul_862: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_863: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_861, mul_862);  mul_861 = mul_862 = None
        unsqueeze_826: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
        unsqueeze_827: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
        unsqueeze_828: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
        mul_864: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_75);  primals_75 = None
        unsqueeze_829: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
        unsqueeze_830: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
        unsqueeze_831: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
        mul_865: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_828);  sub_253 = unsqueeze_828 = None
        sub_255: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_350, mul_865);  convert_element_type_350 = mul_865 = None
        sub_256: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_825);  sub_255 = unsqueeze_825 = None
        mul_866: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_831);  sub_256 = unsqueeze_831 = None
        mul_867: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_99, squeeze_37);  sum_99 = squeeze_37 = None
        convert_element_type_360: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_866, torch.bfloat16);  mul_866 = None
        add_375: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_374, convert_element_type_360);  add_374 = convert_element_type_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_17: "b8[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_17: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.where.self(le_17, full_default, add_375);  le_17 = add_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_361: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_17, torch.float32);  where_17 = None
        sum_100: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_361, [0, 2, 3])
        convert_element_type_34: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_257: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_34, unsqueeze_834);  convert_element_type_34 = unsqueeze_834 = None
        mul_868: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_361, sub_257)
        sum_101: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_868, [0, 2, 3]);  mul_868 = None
        mul_869: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_100, 9.964923469387754e-06)
        unsqueeze_835: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
        unsqueeze_836: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
        unsqueeze_837: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
        mul_870: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, 9.964923469387754e-06)
        mul_871: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_872: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_870, mul_871);  mul_870 = mul_871 = None
        unsqueeze_838: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
        unsqueeze_839: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
        unsqueeze_840: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
        mul_873: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_70);  primals_70 = None
        unsqueeze_841: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
        unsqueeze_842: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
        unsqueeze_843: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
        mul_874: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_840);  sub_257 = unsqueeze_840 = None
        sub_259: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_361, mul_874);  mul_874 = None
        sub_260: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_837);  sub_259 = unsqueeze_837 = None
        mul_875: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_843);  sub_260 = unsqueeze_843 = None
        mul_876: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_101, squeeze_34);  sum_101 = squeeze_34 = None
        convert_element_type_363: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_875, torch.bfloat16);  mul_875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(convert_element_type_363, relu_3, convert_element_type_33, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_363 = convert_element_type_33 = None
        getitem_224: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_34[0]
        getitem_225: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        convert_element_type_364: "f32[192, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_225, torch.float32);  getitem_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_102: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_361, [0, 2, 3])
        convert_element_type_31: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        sub_261: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_31, unsqueeze_846);  convert_element_type_31 = unsqueeze_846 = None
        mul_877: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_361, sub_261)
        sum_103: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_877, [0, 2, 3]);  mul_877 = None
        mul_878: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_102, 9.964923469387754e-06)
        unsqueeze_847: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
        unsqueeze_848: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
        unsqueeze_849: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
        mul_879: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, 9.964923469387754e-06)
        mul_880: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_881: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
        unsqueeze_850: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
        unsqueeze_851: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
        unsqueeze_852: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
        mul_882: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_64);  primals_64 = None
        unsqueeze_853: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
        unsqueeze_854: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
        unsqueeze_855: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
        mul_883: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_852);  sub_261 = unsqueeze_852 = None
        sub_263: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_361, mul_883);  mul_883 = None
        sub_264: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_849);  sub_263 = unsqueeze_849 = None
        mul_884: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_855);  sub_264 = unsqueeze_855 = None
        mul_885: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_103, squeeze_31);  sum_103 = squeeze_31 = None
        convert_element_type_367: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_884, torch.bfloat16);  mul_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(convert_element_type_367, relu_3, convert_element_type_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_367 = convert_element_type_30 = None
        getitem_227: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = convolution_backward_35[0]
        getitem_228: "bf16[192, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        add_376: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(getitem_224, getitem_227);  getitem_224 = getitem_227 = None
        convert_element_type_368: "f32[192, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_228, torch.float32);  getitem_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_104: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_361, [0, 2, 3])
        convert_element_type_28: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(relu_3, torch.float32)
        sub_265: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_28, unsqueeze_858);  convert_element_type_28 = unsqueeze_858 = None
        mul_886: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_361, sub_265)
        sum_105: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_886, [0, 2, 3]);  mul_886 = None
        mul_887: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_104, 9.964923469387754e-06)
        unsqueeze_859: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_887, 0);  mul_887 = None
        unsqueeze_860: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
        unsqueeze_861: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
        mul_888: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, 9.964923469387754e-06)
        mul_889: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_890: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_888, mul_889);  mul_888 = mul_889 = None
        unsqueeze_862: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
        unsqueeze_863: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
        unsqueeze_864: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
        mul_891: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_58);  primals_58 = None
        unsqueeze_865: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
        unsqueeze_866: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
        unsqueeze_867: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
        mul_892: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_864);  sub_265 = unsqueeze_864 = None
        sub_267: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_361, mul_892);  convert_element_type_361 = mul_892 = None
        sub_268: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_861);  sub_267 = unsqueeze_861 = None
        mul_893: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_867);  sub_268 = unsqueeze_867 = None
        mul_894: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_105, squeeze_28);  sum_105 = squeeze_28 = None
        convert_element_type_371: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_893, torch.bfloat16);  mul_893 = None
        add_377: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.add.Tensor(add_376, convert_element_type_371);  add_376 = convert_element_type_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_18: "b8[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_18: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.where.self(le_18, full_default, add_377);  le_18 = add_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_372: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(where_18, torch.float32);  where_18 = None
        sum_106: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_372, [0, 2, 3])
        convert_element_type_26: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_269: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, unsqueeze_870);  convert_element_type_26 = unsqueeze_870 = None
        mul_895: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_372, sub_269)
        sum_107: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_895, [0, 2, 3]);  mul_895 = None
        mul_896: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_106, 9.964923469387754e-06)
        unsqueeze_871: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
        unsqueeze_872: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
        unsqueeze_873: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
        mul_897: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, 9.964923469387754e-06)
        mul_898: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_899: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_897, mul_898);  mul_897 = mul_898 = None
        unsqueeze_874: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
        unsqueeze_875: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
        unsqueeze_876: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
        mul_900: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_53);  primals_53 = None
        unsqueeze_877: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
        unsqueeze_878: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
        unsqueeze_879: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
        mul_901: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_876);  sub_269 = unsqueeze_876 = None
        sub_271: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_372, mul_901);  mul_901 = None
        sub_272: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_873);  sub_271 = unsqueeze_873 = None
        mul_902: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_879);  sub_272 = unsqueeze_879 = None
        mul_903: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_107, squeeze_25);  sum_107 = squeeze_25 = None
        convert_element_type_374: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_902, torch.bfloat16);  mul_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_374, relu_2, convert_element_type_25, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_374 = convert_element_type_25 = None
        getitem_230: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = convolution_backward_36[0]
        getitem_231: "bf16[192, 96, 3, 3][864, 1, 288, 96]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_375: "f32[192, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_231, torch.float32);  getitem_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_108: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_372, [0, 2, 3])
        convert_element_type_23: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_273: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, unsqueeze_882);  convert_element_type_23 = unsqueeze_882 = None
        mul_904: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_372, sub_273)
        sum_109: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 2, 3]);  mul_904 = None
        mul_905: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_108, 9.964923469387754e-06)
        unsqueeze_883: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_905, 0);  mul_905 = None
        unsqueeze_884: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
        unsqueeze_885: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
        mul_906: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, 9.964923469387754e-06)
        mul_907: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_908: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_906, mul_907);  mul_906 = mul_907 = None
        unsqueeze_886: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
        unsqueeze_887: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
        unsqueeze_888: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
        mul_909: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_47);  primals_47 = None
        unsqueeze_889: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
        unsqueeze_890: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
        unsqueeze_891: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
        mul_910: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_888);  sub_273 = unsqueeze_888 = None
        sub_275: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_372, mul_910);  convert_element_type_372 = mul_910 = None
        sub_276: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_885);  sub_275 = unsqueeze_885 = None
        mul_911: "f32[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_891);  sub_276 = unsqueeze_891 = None
        mul_912: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_109, squeeze_22);  sum_109 = squeeze_22 = None
        convert_element_type_378: "bf16[128, 192, 28, 28][150528, 1, 5376, 192]cuda:0" = torch.ops.prims.convert_element_type.default(mul_911, torch.bfloat16);  mul_911 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(convert_element_type_378, relu_2, convert_element_type_22, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_378 = convert_element_type_22 = None
        getitem_233: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = convolution_backward_37[0]
        getitem_234: "bf16[192, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        add_378: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(getitem_230, getitem_233);  getitem_230 = getitem_233 = None
        convert_element_type_379: "f32[192, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_234, torch.float32);  getitem_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_19: "b8[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_19: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.where.self(le_19, full_default, add_378);  le_19 = add_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_380: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_19, torch.float32);  where_19 = None
        sum_110: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_380, [0, 2, 3])
        convert_element_type_20: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_277: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, unsqueeze_894);  convert_element_type_20 = unsqueeze_894 = None
        mul_913: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, sub_277)
        sum_111: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 2, 3]);  mul_913 = None
        mul_914: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_110, 2.4912308673469386e-06)
        unsqueeze_895: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
        unsqueeze_896: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
        unsqueeze_897: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
        mul_915: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, 2.4912308673469386e-06)
        mul_916: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_917: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_915, mul_916);  mul_915 = mul_916 = None
        unsqueeze_898: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
        unsqueeze_899: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
        unsqueeze_900: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
        mul_918: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_41);  primals_41 = None
        unsqueeze_901: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
        unsqueeze_902: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
        unsqueeze_903: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
        mul_919: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_900);  sub_277 = unsqueeze_900 = None
        sub_279: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_380, mul_919);  mul_919 = None
        sub_280: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_897);  sub_279 = unsqueeze_897 = None
        mul_920: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_903);  sub_280 = unsqueeze_903 = None
        mul_921: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_111, squeeze_19);  sum_111 = squeeze_19 = None
        convert_element_type_382: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_920, torch.bfloat16);  mul_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(convert_element_type_382, relu_1, convert_element_type_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_382 = convert_element_type_19 = None
        getitem_236: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = convolution_backward_38[0]
        getitem_237: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_383: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_237, torch.float32);  getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_112: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_380, [0, 2, 3])
        convert_element_type_17: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_281: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_17, unsqueeze_906);  convert_element_type_17 = unsqueeze_906 = None
        mul_922: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, sub_281)
        sum_113: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_922, [0, 2, 3]);  mul_922 = None
        mul_923: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 2.4912308673469386e-06)
        unsqueeze_907: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
        unsqueeze_908: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
        unsqueeze_909: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
        mul_924: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 2.4912308673469386e-06)
        mul_925: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_926: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_924, mul_925);  mul_924 = mul_925 = None
        unsqueeze_910: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
        unsqueeze_911: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
        unsqueeze_912: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
        mul_927: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_35);  primals_35 = None
        unsqueeze_913: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
        unsqueeze_914: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
        unsqueeze_915: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
        mul_928: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_912);  sub_281 = unsqueeze_912 = None
        sub_283: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_380, mul_928);  mul_928 = None
        sub_284: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_909);  sub_283 = unsqueeze_909 = None
        mul_929: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_915);  sub_284 = unsqueeze_915 = None
        mul_930: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_16);  sum_113 = squeeze_16 = None
        convert_element_type_386: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_929, torch.bfloat16);  mul_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_386, relu_1, convert_element_type_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_386 = convert_element_type_16 = None
        getitem_239: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = convolution_backward_39[0]
        getitem_240: "bf16[96, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        add_379: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(getitem_236, getitem_239);  getitem_236 = getitem_239 = None
        convert_element_type_387: "f32[96, 96, 1, 1][96, 1, 96, 96]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_240, torch.float32);  getitem_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_114: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_380, [0, 2, 3])
        convert_element_type_14: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(relu_1, torch.float32)
        sub_285: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, unsqueeze_918);  convert_element_type_14 = unsqueeze_918 = None
        mul_931: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, sub_285)
        sum_115: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_931, [0, 2, 3]);  mul_931 = None
        mul_932: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, 2.4912308673469386e-06)
        unsqueeze_919: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_920: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
        unsqueeze_921: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
        mul_933: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, 2.4912308673469386e-06)
        mul_934: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_935: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_933, mul_934);  mul_933 = mul_934 = None
        unsqueeze_922: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
        unsqueeze_923: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
        unsqueeze_924: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
        mul_936: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_29);  primals_29 = None
        unsqueeze_925: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
        unsqueeze_926: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
        unsqueeze_927: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
        mul_937: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_924);  sub_285 = unsqueeze_924 = None
        sub_287: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_380, mul_937);  convert_element_type_380 = mul_937 = None
        sub_288: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_921);  sub_287 = unsqueeze_921 = None
        mul_938: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_927);  sub_288 = unsqueeze_927 = None
        mul_939: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, squeeze_13);  sum_115 = squeeze_13 = None
        convert_element_type_390: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_938, torch.bfloat16);  mul_938 = None
        add_380: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.add.Tensor(add_379, convert_element_type_390);  add_379 = convert_element_type_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_20: "b8[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_20: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.where.self(le_20, full_default, add_380);  le_20 = add_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_391: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(where_20, torch.float32);  where_20 = None
        sum_116: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_391, [0, 2, 3])
        convert_element_type_12: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_289: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_12, unsqueeze_930);  convert_element_type_12 = unsqueeze_930 = None
        mul_940: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_391, sub_289)
        sum_117: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_940, [0, 2, 3]);  mul_940 = None
        mul_941: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, 2.4912308673469386e-06)
        unsqueeze_931: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
        unsqueeze_932: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
        unsqueeze_933: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
        mul_942: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 2.4912308673469386e-06)
        mul_943: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_944: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
        unsqueeze_934: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
        unsqueeze_935: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
        unsqueeze_936: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
        mul_945: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_937: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
        unsqueeze_938: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
        unsqueeze_939: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
        mul_946: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_936);  sub_289 = unsqueeze_936 = None
        sub_291: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_391, mul_946);  mul_946 = None
        sub_292: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_933);  sub_291 = unsqueeze_933 = None
        mul_947: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_939);  sub_292 = unsqueeze_939 = None
        mul_948: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, squeeze_10);  sum_117 = squeeze_10 = None
        convert_element_type_393: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_947, torch.bfloat16);  mul_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_393, relu, convert_element_type_11, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_393 = convert_element_type_11 = None
        getitem_242: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = convolution_backward_40[0]
        getitem_243: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_394: "f32[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_243, torch.float32);  getitem_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_118: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_391, [0, 2, 3])
        convert_element_type_9: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_293: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_9, unsqueeze_942);  convert_element_type_9 = unsqueeze_942 = None
        mul_949: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_391, sub_293)
        sum_119: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2, 3]);  mul_949 = None
        mul_950: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 2.4912308673469386e-06)
        unsqueeze_943: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
        unsqueeze_944: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
        unsqueeze_945: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
        mul_951: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, 2.4912308673469386e-06)
        mul_952: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_953: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
        unsqueeze_946: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
        unsqueeze_947: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
        unsqueeze_948: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
        mul_954: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_949: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
        unsqueeze_950: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
        unsqueeze_951: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
        mul_955: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_948);  sub_293 = unsqueeze_948 = None
        sub_295: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_391, mul_955);  convert_element_type_391 = mul_955 = None
        sub_296: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_945);  sub_295 = unsqueeze_945 = None
        mul_956: "f32[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_951);  sub_296 = unsqueeze_951 = None
        mul_957: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, squeeze_7);  sum_119 = squeeze_7 = None
        convert_element_type_397: "bf16[128, 96, 56, 56][301056, 1, 5376, 96]cuda:0" = torch.ops.prims.convert_element_type.default(mul_956, torch.bfloat16);  mul_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(convert_element_type_397, relu, convert_element_type_8, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_397 = convert_element_type_8 = None
        getitem_245: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = convolution_backward_41[0]
        getitem_246: "bf16[96, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        add_381: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(getitem_242, getitem_245);  getitem_242 = getitem_245 = None
        convert_element_type_398: "f32[96, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_246, torch.float32);  getitem_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/byobnet.py:755 in forward, code: return self.act(x)
        le_21: "b8[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_21: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.where.self(le_21, full_default, add_381);  le_21 = full_default = add_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_399: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_21, torch.float32);  where_21 = None
        sum_120: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_399, [0, 2, 3])
        convert_element_type_6: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_297: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_6, unsqueeze_954);  convert_element_type_6 = unsqueeze_954 = None
        mul_958: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_399, sub_297)
        sum_121: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_958, [0, 2, 3]);  mul_958 = None
        mul_959: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 6.228077168367346e-07)
        unsqueeze_955: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
        unsqueeze_956: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
        unsqueeze_957: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
        mul_960: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 6.228077168367346e-07)
        mul_961: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_962: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
        unsqueeze_958: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
        unsqueeze_959: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
        unsqueeze_960: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
        mul_963: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_961: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
        unsqueeze_962: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
        unsqueeze_963: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
        mul_964: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_960);  sub_297 = unsqueeze_960 = None
        sub_299: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_399, mul_964);  mul_964 = None
        sub_300: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_957);  sub_299 = unsqueeze_957 = None
        mul_965: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_963);  sub_300 = unsqueeze_963 = None
        mul_966: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_4);  sum_121 = squeeze_4 = None
        convert_element_type_401: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_965, torch.bfloat16);  mul_965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_401, convert_element_type_1, convert_element_type_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_401 = convert_element_type_4 = None
        getitem_249: "bf16[64, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_402: "f32[64, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_249, torch.float32);  getitem_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_122: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_399, [0, 2, 3])
        convert_element_type_2: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_301: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_966);  convert_element_type_2 = unsqueeze_966 = None
        mul_967: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_399, sub_301)
        sum_123: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_967, [0, 2, 3]);  mul_967 = None
        mul_968: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, 6.228077168367346e-07)
        unsqueeze_967: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
        unsqueeze_968: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
        unsqueeze_969: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
        mul_969: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, 6.228077168367346e-07)
        mul_970: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_971: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_969, mul_970);  mul_969 = mul_970 = None
        unsqueeze_970: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
        unsqueeze_971: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
        unsqueeze_972: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
        mul_972: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_973: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
        unsqueeze_974: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
        unsqueeze_975: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
        mul_973: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_972);  sub_301 = unsqueeze_972 = None
        sub_303: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_399, mul_973);  convert_element_type_399 = mul_973 = None
        sub_304: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_969);  sub_303 = unsqueeze_969 = None
        mul_974: "f32[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_975);  sub_304 = unsqueeze_975 = None
        mul_975: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, squeeze_1);  sum_123 = squeeze_1 = None
        convert_element_type_405: "bf16[128, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_974, torch.bfloat16);  mul_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_405, convert_element_type_1, convert_element_type, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_405 = convert_element_type_1 = convert_element_type = None
        getitem_252: "bf16[64, 3, 1, 1][3, 1, 3, 3]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_406: "f32[64, 3, 1, 1][3, 1, 3, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_252, torch.float32);  getitem_252 = None
        return (convert_element_type_406, None, None, None, None, mul_975, sum_122, convert_element_type_402, None, None, None, mul_966, sum_120, convert_element_type_398, None, None, None, mul_957, sum_118, convert_element_type_394, None, None, None, mul_948, sum_116, None, None, None, mul_939, sum_114, convert_element_type_387, None, None, None, mul_930, sum_112, convert_element_type_383, None, None, None, mul_921, sum_110, convert_element_type_379, None, None, None, mul_912, sum_108, convert_element_type_375, None, None, None, mul_903, sum_106, None, None, None, mul_894, sum_104, convert_element_type_368, None, None, None, mul_885, sum_102, convert_element_type_364, None, None, None, mul_876, sum_100, None, None, None, mul_867, sum_98, convert_element_type_357, None, None, None, mul_858, sum_96, convert_element_type_353, None, None, None, mul_849, sum_94, None, None, None, mul_840, sum_92, convert_element_type_346, None, None, None, mul_831, sum_90, convert_element_type_342, None, None, None, mul_822, sum_88, convert_element_type_338, None, None, None, mul_813, sum_86, convert_element_type_334, None, None, None, mul_804, sum_84, None, None, None, mul_795, sum_82, convert_element_type_327, None, None, None, mul_786, sum_80, convert_element_type_323, None, None, None, mul_777, sum_78, None, None, None, mul_768, sum_76, convert_element_type_316, None, None, None, mul_759, sum_74, convert_element_type_312, None, None, None, mul_750, sum_72, None, None, None, mul_741, sum_70, convert_element_type_305, None, None, None, mul_732, sum_68, convert_element_type_301, None, None, None, mul_723, sum_66, None, None, None, mul_714, sum_64, convert_element_type_294, None, None, None, mul_705, sum_62, convert_element_type_290, None, None, None, mul_696, sum_60, None, None, None, mul_687, sum_58, convert_element_type_283, None, None, None, mul_678, sum_56, convert_element_type_279, None, None, None, mul_669, sum_54, None, None, None, mul_660, sum_52, convert_element_type_272, None, None, None, mul_651, sum_50, convert_element_type_268, None, None, None, mul_642, sum_48, None, None, None, mul_633, sum_46, convert_element_type_261, None, None, None, mul_624, sum_44, convert_element_type_257, None, None, None, mul_615, sum_42, None, None, None, mul_606, sum_40, convert_element_type_250, None, None, None, mul_597, sum_38, convert_element_type_246, None, None, None, mul_588, sum_36, None, None, None, mul_579, sum_34, convert_element_type_239, None, None, None, mul_570, sum_32, convert_element_type_235, None, None, None, mul_561, sum_30, None, None, None, mul_552, sum_28, convert_element_type_228, None, None, None, mul_543, sum_26, convert_element_type_224, None, None, None, mul_534, sum_24, None, None, None, mul_525, sum_22, convert_element_type_217, None, None, None, mul_516, sum_20, convert_element_type_213, None, None, None, mul_507, sum_18, None, None, None, mul_498, sum_16, convert_element_type_206, None, None, None, mul_489, sum_14, convert_element_type_202, None, None, None, mul_480, sum_12, None, None, None, mul_471, sum_10, convert_element_type_195, None, None, None, mul_462, sum_8, convert_element_type_191, None, None, None, mul_453, sum_6, convert_element_type_187, None, None, None, mul_444, sum_4, convert_element_type_183, None, None, None, mul_435, sum_2, convert_element_type_178, convert_element_type_179)
