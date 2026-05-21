class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", _shape_param_0, mul_446: "f32[128, 3072, 6, 6]", getitem_115: "f32[3072, 1536, 1, 1]", _shape_param_1, primals_230: "f32[3072, 1536, 1, 1]", _shape_param_2, unsqueeze_58: "f32[1, 3072, 1]", squeeze_113: "f32[3072]", primals_231: "f32[3072, 1, 1, 1]", _shape_param_3, _shape_param_4, convolution_77: "f32[128, 1536, 6, 6]", sigmoid_11: "f32[128, 1536, 1, 1]", mul_457: "f32[128, 1536, 6, 6]", mul_464: "f32[128, 1536, 1, 1]", where: "f32[128, 768, 1, 1]", add_123: "f32[128, 1536, 6, 6]", getitem_124: "f32[1536, 768, 1, 1]", _shape_param_5, primals_222: "f32[1536, 768, 1, 1]", _shape_param_6, unsqueeze_66: "f32[1, 1536, 1]", squeeze_111: "f32[1536]", primals_223: "f32[1536, 1, 1, 1]", _shape_param_7, _shape_param_8, mul_482: "f32[128, 768, 6, 6]", getitem_127: "f32[768, 128, 3, 3]", _shape_param_9, primals_219: "f32[768, 128, 3, 3]", _shape_param_10, unsqueeze_74: "f32[1, 768, 1]", squeeze_109: "f32[768]", primals_220: "f32[768, 1, 1, 1]", _shape_param_11, _shape_param_12, mul_500: "f32[128, 768, 6, 6]", getitem_130: "f32[768, 128, 3, 3]", _shape_param_13, primals_216: "f32[768, 128, 3, 3]", _shape_param_14, unsqueeze_82: "f32[1, 768, 1]", squeeze_107: "f32[768]", primals_217: "f32[768, 1, 1, 1]", _shape_param_15, _shape_param_16, mul_518: "f32[128, 768, 6, 6]", getitem_133: "f32[768, 1536, 1, 1]", _shape_param_17, primals_213: "f32[768, 1536, 1, 1]", _shape_param_18, unsqueeze_90: "f32[1, 768, 1]", squeeze_105: "f32[768]", primals_214: "f32[768, 1, 1, 1]", _shape_param_19, _shape_param_20, convolution_71: "f32[128, 1536, 6, 6]", sigmoid_10: "f32[128, 1536, 1, 1]", mul_538: "f32[128, 1536, 6, 6]", mul_545: "f32[128, 1536, 1, 1]", where_1: "f32[128, 768, 1, 1]", add_133: "f32[128, 1536, 6, 6]", getitem_142: "f32[1536, 768, 1, 1]", _shape_param_21, primals_205: "f32[1536, 768, 1, 1]", _shape_param_22, unsqueeze_98: "f32[1, 1536, 1]", squeeze_103: "f32[1536]", primals_206: "f32[1536, 1, 1, 1]", _shape_param_23, _shape_param_24, mul_563: "f32[128, 768, 6, 6]", getitem_145: "f32[768, 128, 3, 3]", _shape_param_25, primals_202: "f32[768, 128, 3, 3]", _shape_param_26, unsqueeze_106: "f32[1, 768, 1]", squeeze_101: "f32[768]", primals_203: "f32[768, 1, 1, 1]", _shape_param_27, _shape_param_28, mul_581: "f32[128, 768, 6, 6]", getitem_148: "f32[768, 128, 3, 3]", _shape_param_29, primals_199: "f32[768, 128, 3, 3]", _shape_param_30, unsqueeze_114: "f32[1, 768, 1]", squeeze_99: "f32[768]", primals_200: "f32[768, 1, 1, 1]", _shape_param_31, _shape_param_32, mul_599: "f32[128, 768, 6, 6]", getitem_151: "f32[768, 1536, 1, 1]", _shape_param_33, primals_196: "f32[768, 1536, 1, 1]", _shape_param_34, unsqueeze_122: "f32[1, 768, 1]", squeeze_97: "f32[768]", primals_197: "f32[768, 1, 1, 1]", _shape_param_35, _shape_param_36, mul_619: "f32[128, 1536, 6, 6]", mul_363: "f32[128, 1536, 6, 6]", mul_626: "f32[128, 1536, 1, 1]", where_2: "f32[128, 768, 1, 1]", add_143: "f32[128, 1536, 6, 6]", getitem_160: "f32[1536, 768, 1, 1]", _shape_param_37, primals_188: "f32[1536, 768, 1, 1]", _shape_param_38, unsqueeze_130: "f32[1, 1536, 1]", squeeze_95: "f32[1536]", primals_189: "f32[1536, 1, 1, 1]", _shape_param_39, _shape_param_40, mul_644: "f32[128, 768, 6, 6]", getitem_163: "f32[768, 128, 3, 3]", _shape_param_41, primals_185: "f32[768, 128, 3, 3]", _shape_param_42, unsqueeze_138: "f32[1, 768, 1]", squeeze_93: "f32[768]", primals_186: "f32[768, 1, 1, 1]", _shape_param_43, _shape_param_44, mul_662: "f32[128, 768, 6, 6]", getitem_166: "f32[768, 128, 3, 3]", _shape_param_45, primals_182: "f32[768, 128, 3, 3]", _shape_param_46, unsqueeze_146: "f32[1, 768, 1]", squeeze_91: "f32[768]", primals_183: "f32[768, 1, 1, 1]", _shape_param_47, _shape_param_48, mul_680: "f32[128, 768, 12, 12]", getitem_169: "f32[768, 1536, 1, 1]", _shape_param_49, primals_179: "f32[768, 1536, 1, 1]", _shape_param_50, unsqueeze_154: "f32[1, 768, 1]", squeeze_89: "f32[768]", primals_180: "f32[768, 1, 1, 1]", _shape_param_51, _shape_param_52, add_142: "f32[128, 1536, 6, 6]", getitem_172: "f32[1536, 1536, 1, 1]", _shape_param_53, primals_176: "f32[1536, 1536, 1, 1]", _shape_param_54, unsqueeze_162: "f32[1, 1536, 1]", squeeze_87: "f32[1536]", primals_177: "f32[1536, 1, 1, 1]", _shape_param_55, _shape_param_56, convolution_58: "f32[128, 1536, 12, 12]", sigmoid_8: "f32[128, 1536, 1, 1]", mul_710: "f32[128, 1536, 12, 12]", mul_717: "f32[128, 1536, 1, 1]", where_3: "f32[128, 768, 1, 1]", add_153: "f32[128, 1536, 12, 12]", getitem_181: "f32[1536, 768, 1, 1]", _shape_param_57, primals_168: "f32[1536, 768, 1, 1]", _shape_param_58, unsqueeze_170: "f32[1, 1536, 1]", squeeze_85: "f32[1536]", primals_169: "f32[1536, 1, 1, 1]", _shape_param_59, _shape_param_60, mul_735: "f32[128, 768, 12, 12]", getitem_184: "f32[768, 128, 3, 3]", _shape_param_61, primals_165: "f32[768, 128, 3, 3]", _shape_param_62, unsqueeze_178: "f32[1, 768, 1]", squeeze_83: "f32[768]", primals_166: "f32[768, 1, 1, 1]", _shape_param_63, _shape_param_64, mul_753: "f32[128, 768, 12, 12]", getitem_187: "f32[768, 128, 3, 3]", _shape_param_65, primals_162: "f32[768, 128, 3, 3]", _shape_param_66, unsqueeze_186: "f32[1, 768, 1]", squeeze_81: "f32[768]", primals_163: "f32[768, 1, 1, 1]", _shape_param_67, _shape_param_68, mul_771: "f32[128, 768, 12, 12]", getitem_190: "f32[768, 1536, 1, 1]", _shape_param_69, primals_159: "f32[768, 1536, 1, 1]", _shape_param_70, unsqueeze_194: "f32[1, 768, 1]", squeeze_79: "f32[768]", primals_160: "f32[768, 1, 1, 1]", _shape_param_71, _shape_param_72, convolution_52: "f32[128, 1536, 12, 12]", sigmoid_7: "f32[128, 1536, 1, 1]", mul_791: "f32[128, 1536, 12, 12]", mul_798: "f32[128, 1536, 1, 1]", where_4: "f32[128, 768, 1, 1]", add_163: "f32[128, 1536, 12, 12]", getitem_199: "f32[1536, 768, 1, 1]", _shape_param_73, primals_151: "f32[1536, 768, 1, 1]", _shape_param_74, unsqueeze_202: "f32[1, 1536, 1]", squeeze_77: "f32[1536]", primals_152: "f32[1536, 1, 1, 1]", _shape_param_75, _shape_param_76, mul_816: "f32[128, 768, 12, 12]", getitem_202: "f32[768, 128, 3, 3]", _shape_param_77, primals_148: "f32[768, 128, 3, 3]", _shape_param_78, unsqueeze_210: "f32[1, 768, 1]", squeeze_75: "f32[768]", primals_149: "f32[768, 1, 1, 1]", _shape_param_79, _shape_param_80, mul_834: "f32[128, 768, 12, 12]", getitem_205: "f32[768, 128, 3, 3]", _shape_param_81, primals_145: "f32[768, 128, 3, 3]", _shape_param_82, unsqueeze_218: "f32[1, 768, 1]", squeeze_73: "f32[768]", primals_146: "f32[768, 1, 1, 1]", _shape_param_83, _shape_param_84, mul_852: "f32[128, 768, 12, 12]", getitem_208: "f32[768, 1536, 1, 1]", _shape_param_85, primals_142: "f32[768, 1536, 1, 1]", _shape_param_86, unsqueeze_226: "f32[1, 768, 1]", squeeze_71: "f32[768]", primals_143: "f32[768, 1, 1, 1]", _shape_param_87, _shape_param_88, convolution_46: "f32[128, 1536, 12, 12]", sigmoid_6: "f32[128, 1536, 1, 1]", mul_872: "f32[128, 1536, 12, 12]", mul_879: "f32[128, 1536, 1, 1]", where_5: "f32[128, 768, 1, 1]", add_173: "f32[128, 1536, 12, 12]", getitem_217: "f32[1536, 768, 1, 1]", _shape_param_89, primals_134: "f32[1536, 768, 1, 1]", _shape_param_90, unsqueeze_234: "f32[1, 1536, 1]", squeeze_69: "f32[1536]", primals_135: "f32[1536, 1, 1, 1]", _shape_param_91, _shape_param_92, mul_897: "f32[128, 768, 12, 12]", getitem_220: "f32[768, 128, 3, 3]", _shape_param_93, primals_131: "f32[768, 128, 3, 3]", _shape_param_94, unsqueeze_242: "f32[1, 768, 1]", squeeze_67: "f32[768]", primals_132: "f32[768, 1, 1, 1]", _shape_param_95, _shape_param_96, mul_915: "f32[128, 768, 12, 12]", getitem_223: "f32[768, 128, 3, 3]", _shape_param_97, primals_128: "f32[768, 128, 3, 3]", _shape_param_98, unsqueeze_250: "f32[1, 768, 1]", squeeze_65: "f32[768]", primals_129: "f32[768, 1, 1, 1]", _shape_param_99, _shape_param_100, mul_933: "f32[128, 768, 12, 12]", getitem_226: "f32[768, 1536, 1, 1]", _shape_param_101, primals_125: "f32[768, 1536, 1, 1]", _shape_param_102, unsqueeze_258: "f32[1, 768, 1]", squeeze_63: "f32[768]", primals_126: "f32[768, 1, 1, 1]", _shape_param_103, _shape_param_104, convolution_40: "f32[128, 1536, 12, 12]", sigmoid_5: "f32[128, 1536, 1, 1]", mul_953: "f32[128, 1536, 12, 12]", mul_960: "f32[128, 1536, 1, 1]", where_6: "f32[128, 768, 1, 1]", add_183: "f32[128, 1536, 12, 12]", getitem_235: "f32[1536, 768, 1, 1]", _shape_param_105, primals_117: "f32[1536, 768, 1, 1]", _shape_param_106, unsqueeze_266: "f32[1, 1536, 1]", squeeze_61: "f32[1536]", primals_118: "f32[1536, 1, 1, 1]", _shape_param_107, _shape_param_108, mul_978: "f32[128, 768, 12, 12]", getitem_238: "f32[768, 128, 3, 3]", _shape_param_109, primals_114: "f32[768, 128, 3, 3]", _shape_param_110, unsqueeze_274: "f32[1, 768, 1]", squeeze_59: "f32[768]", primals_115: "f32[768, 1, 1, 1]", _shape_param_111, _shape_param_112, mul_996: "f32[128, 768, 12, 12]", getitem_241: "f32[768, 128, 3, 3]", _shape_param_113, primals_111: "f32[768, 128, 3, 3]", _shape_param_114, unsqueeze_282: "f32[1, 768, 1]", squeeze_57: "f32[768]", primals_112: "f32[768, 1, 1, 1]", _shape_param_115, _shape_param_116, mul_1014: "f32[128, 768, 12, 12]", getitem_244: "f32[768, 1536, 1, 1]", _shape_param_117, primals_108: "f32[768, 1536, 1, 1]", _shape_param_118, unsqueeze_290: "f32[1, 768, 1]", squeeze_55: "f32[768]", primals_109: "f32[768, 1, 1, 1]", _shape_param_119, _shape_param_120, convolution_34: "f32[128, 1536, 12, 12]", sigmoid_4: "f32[128, 1536, 1, 1]", mul_1034: "f32[128, 1536, 12, 12]", mul_1041: "f32[128, 1536, 1, 1]", where_7: "f32[128, 768, 1, 1]", add_193: "f32[128, 1536, 12, 12]", getitem_253: "f32[1536, 768, 1, 1]", _shape_param_121, primals_100: "f32[1536, 768, 1, 1]", _shape_param_122, unsqueeze_298: "f32[1, 1536, 1]", squeeze_53: "f32[1536]", primals_101: "f32[1536, 1, 1, 1]", _shape_param_123, _shape_param_124, mul_1059: "f32[128, 768, 12, 12]", getitem_256: "f32[768, 128, 3, 3]", _shape_param_125, primals_97: "f32[768, 128, 3, 3]", _shape_param_126, unsqueeze_306: "f32[1, 768, 1]", squeeze_51: "f32[768]", primals_98: "f32[768, 1, 1, 1]", _shape_param_127, _shape_param_128, mul_1077: "f32[128, 768, 12, 12]", getitem_259: "f32[768, 128, 3, 3]", _shape_param_129, primals_94: "f32[768, 128, 3, 3]", _shape_param_130, unsqueeze_314: "f32[1, 768, 1]", squeeze_49: "f32[768]", primals_95: "f32[768, 1, 1, 1]", _shape_param_131, _shape_param_132, mul_1095: "f32[128, 768, 12, 12]", getitem_262: "f32[768, 1536, 1, 1]", _shape_param_133, primals_91: "f32[768, 1536, 1, 1]", _shape_param_134, unsqueeze_322: "f32[1, 768, 1]", squeeze_47: "f32[768]", primals_92: "f32[768, 1, 1, 1]", _shape_param_135, _shape_param_136, mul_1115: "f32[128, 1536, 12, 12]", mul_162: "f32[128, 1536, 12, 12]", mul_1122: "f32[128, 1536, 1, 1]", where_8: "f32[128, 768, 1, 1]", add_203: "f32[128, 1536, 12, 12]", getitem_271: "f32[1536, 768, 1, 1]", _shape_param_137, primals_83: "f32[1536, 768, 1, 1]", _shape_param_138, unsqueeze_330: "f32[1, 1536, 1]", squeeze_45: "f32[1536]", primals_84: "f32[1536, 1, 1, 1]", _shape_param_139, _shape_param_140, mul_1140: "f32[128, 768, 12, 12]", getitem_274: "f32[768, 128, 3, 3]", _shape_param_141, primals_80: "f32[768, 128, 3, 3]", _shape_param_142, unsqueeze_338: "f32[1, 768, 1]", squeeze_43: "f32[768]", primals_81: "f32[768, 1, 1, 1]", _shape_param_143, _shape_param_144, mul_1158: "f32[128, 768, 12, 12]", getitem_277: "f32[768, 128, 3, 3]", _shape_param_145, primals_77: "f32[768, 128, 3, 3]", _shape_param_146, unsqueeze_346: "f32[1, 768, 1]", squeeze_41: "f32[768]", primals_78: "f32[768, 1, 1, 1]", _shape_param_147, _shape_param_148, mul_1176: "f32[128, 768, 24, 24]", getitem_280: "f32[768, 512, 1, 1]", _shape_param_149, primals_74: "f32[768, 512, 1, 1]", _shape_param_150, unsqueeze_354: "f32[1, 768, 1]", squeeze_39: "f32[768]", primals_75: "f32[768, 1, 1, 1]", _shape_param_151, _shape_param_152, add_202: "f32[128, 1536, 12, 12]", getitem_283: "f32[1536, 512, 1, 1]", _shape_param_153, primals_71: "f32[1536, 512, 1, 1]", _shape_param_154, unsqueeze_362: "f32[1, 1536, 1]", squeeze_37: "f32[1536]", primals_72: "f32[1536, 1, 1, 1]", _shape_param_155, _shape_param_156, convolution_21: "f32[128, 512, 24, 24]", sigmoid_2: "f32[128, 512, 1, 1]", mul_1206: "f32[128, 512, 24, 24]", mul_1213: "f32[128, 512, 1, 1]", where_9: "f32[128, 256, 1, 1]", add_213: "f32[128, 512, 24, 24]", getitem_292: "f32[512, 256, 1, 1]", _shape_param_157, primals_63: "f32[512, 256, 1, 1]", _shape_param_158, unsqueeze_370: "f32[1, 512, 1]", squeeze_35: "f32[512]", primals_64: "f32[512, 1, 1, 1]", _shape_param_159, _shape_param_160, mul_1231: "f32[128, 256, 24, 24]", getitem_295: "f32[256, 128, 3, 3]", _shape_param_161, primals_60: "f32[256, 128, 3, 3]", _shape_param_162, unsqueeze_378: "f32[1, 256, 1]", squeeze_33: "f32[256]", primals_61: "f32[256, 1, 1, 1]", _shape_param_163, _shape_param_164, mul_1249: "f32[128, 256, 24, 24]", getitem_298: "f32[256, 128, 3, 3]", _shape_param_165, primals_57: "f32[256, 128, 3, 3]", _shape_param_166, unsqueeze_386: "f32[1, 256, 1]", squeeze_31: "f32[256]", primals_58: "f32[256, 1, 1, 1]", _shape_param_167, _shape_param_168, mul_1267: "f32[128, 256, 24, 24]", getitem_301: "f32[256, 512, 1, 1]", _shape_param_169, primals_54: "f32[256, 512, 1, 1]", _shape_param_170, unsqueeze_394: "f32[1, 256, 1]", squeeze_29: "f32[256]", primals_55: "f32[256, 1, 1, 1]", _shape_param_171, _shape_param_172, mul_1287: "f32[128, 512, 24, 24]", mul_93: "f32[128, 512, 24, 24]", mul_1294: "f32[128, 512, 1, 1]", where_10: "f32[128, 256, 1, 1]", add_223: "f32[128, 512, 24, 24]", getitem_310: "f32[512, 256, 1, 1]", _shape_param_173, primals_46: "f32[512, 256, 1, 1]", _shape_param_174, unsqueeze_402: "f32[1, 512, 1]", squeeze_27: "f32[512]", primals_47: "f32[512, 1, 1, 1]", _shape_param_175, _shape_param_176, mul_1312: "f32[128, 256, 24, 24]", getitem_313: "f32[256, 128, 3, 3]", _shape_param_177, primals_43: "f32[256, 128, 3, 3]", _shape_param_178, unsqueeze_410: "f32[1, 256, 1]", squeeze_25: "f32[256]", primals_44: "f32[256, 1, 1, 1]", _shape_param_179, _shape_param_180, mul_1330: "f32[128, 256, 24, 24]", getitem_316: "f32[256, 128, 3, 3]", _shape_param_181, primals_40: "f32[256, 128, 3, 3]", _shape_param_182, unsqueeze_418: "f32[1, 256, 1]", squeeze_23: "f32[256]", primals_41: "f32[256, 1, 1, 1]", _shape_param_183, _shape_param_184, mul_1348: "f32[128, 256, 48, 48]", getitem_319: "f32[256, 256, 1, 1]", _shape_param_185, primals_37: "f32[256, 256, 1, 1]", _shape_param_186, unsqueeze_426: "f32[1, 256, 1]", squeeze_21: "f32[256]", primals_38: "f32[256, 1, 1, 1]", _shape_param_187, _shape_param_188, add_222: "f32[128, 512, 24, 24]", getitem_322: "f32[512, 256, 1, 1]", _shape_param_189, primals_34: "f32[512, 256, 1, 1]", _shape_param_190, unsqueeze_434: "f32[1, 512, 1]", squeeze_19: "f32[512]", primals_35: "f32[512, 1, 1, 1]", _shape_param_191, _shape_param_192, mul_1378: "f32[128, 256, 48, 48]", mul_57: "f32[128, 256, 48, 48]", mul_1385: "f32[128, 256, 1, 1]", where_11: "f32[128, 128, 1, 1]", add_233: "f32[128, 256, 48, 48]", getitem_331: "f32[256, 128, 1, 1]", _shape_param_193, primals_26: "f32[256, 128, 1, 1]", _shape_param_194, unsqueeze_442: "f32[1, 256, 1]", squeeze_17: "f32[256]", primals_27: "f32[256, 1, 1, 1]", _shape_param_195, _shape_param_196, mul_1403: "f32[128, 128, 48, 48]", getitem_334: "f32[128, 128, 3, 3]", _shape_param_197, primals_23: "f32[128, 128, 3, 3]", _shape_param_198, unsqueeze_450: "f32[1, 128, 1]", squeeze_15: "f32[128]", primals_24: "f32[128, 1, 1, 1]", _shape_param_199, _shape_param_200, mul_1421: "f32[128, 128, 48, 48]", getitem_337: "f32[128, 128, 3, 3]", _shape_param_201, primals_20: "f32[128, 128, 3, 3]", _shape_param_202, unsqueeze_458: "f32[1, 128, 1]", squeeze_13: "f32[128]", primals_21: "f32[128, 1, 1, 1]", _shape_param_203, _shape_param_204, mul_1439: "f32[128, 128, 48, 48]", getitem_340: "f32[128, 128, 1, 1]", _shape_param_205, primals_17: "f32[128, 128, 1, 1]", _shape_param_206, unsqueeze_466: "f32[1, 128, 1]", squeeze_11: "f32[128]", primals_18: "f32[128, 1, 1, 1]", _shape_param_207, _shape_param_208, mul_1377: "f32[128, 256, 48, 48]", getitem_343: "f32[256, 128, 1, 1]", _shape_param_209, primals_14: "f32[256, 128, 1, 1]", _shape_param_210, unsqueeze_474: "f32[1, 256, 1]", squeeze_9: "f32[256]", primals_15: "f32[256, 1, 1, 1]", _shape_param_211, _shape_param_212, mul_1468: "f32[128, 128, 48, 48]", getitem_346: "f32[128, 64, 3, 3]", _shape_param_213, primals_11: "f32[128, 64, 3, 3]", _shape_param_214, unsqueeze_482: "f32[1, 128, 1]", squeeze_7: "f32[128]", primals_12: "f32[128, 1, 1, 1]", _shape_param_215, _shape_param_216, mul_1486: "f32[128, 64, 96, 96]", getitem_349: "f32[64, 32, 3, 3]", _shape_param_217, primals_8: "f32[64, 32, 3, 3]", _shape_param_218, unsqueeze_490: "f32[1, 64, 1]", squeeze_5: "f32[64]", primals_9: "f32[64, 1, 1, 1]", _shape_param_219, _shape_param_220, mul_1504: "f32[128, 32, 96, 96]", getitem_352: "f32[32, 16, 3, 3]", _shape_param_221, primals_5: "f32[32, 16, 3, 3]", _shape_param_222, unsqueeze_498: "f32[1, 32, 1]", squeeze_3: "f32[32]", primals_6: "f32[32, 1, 1, 1]", _shape_param_223, _shape_param_224, mul_1522: "f32[128, 16, 96, 96]", getitem_355: "f32[16, 3, 3, 3]", _shape_param_225, primals_2: "f32[16, 3, 3, 3]", _shape_param_226, unsqueeze_506: "f32[1, 16, 1]", squeeze_1: "f32[16]", primals_3: "f32[16, 1, 1, 1]", _shape_param_227, _shape_param_228):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_1: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3]);  mul_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_1: "f32[1, 3072, 1536]" = torch.ops.aten.reshape.default(getitem_115, _shape_param_1);  getitem_115 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_2: "f32[3072]" = torch.ops.aten.sum.dim_IntList(reshape_default_1, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_2: "f32[1, 3072, 1536]" = torch.ops.aten.reshape.default(primals_230, _shape_param_2);  primals_230 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_2, unsqueeze_58);  reshape_default_2 = unsqueeze_58 = None
        mul_tensor: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_1, sub_tensor)
        sum_dim_int_list_3: "f32[3072]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2]);  mul_tensor = None
        mul_tensor_1: "f32[3072]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.0006510416666666666);  sum_dim_int_list_2 = None
        unsqueeze_default: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        mul_tensor_2: "f32[3072]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, 0.0006510416666666666)
        mul_tensor_3: "f32[3072]" = torch.ops.aten.mul.Tensor(squeeze_113, squeeze_113)
        mul_tensor_4: "f32[3072]" = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
        unsqueeze_default_2: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(mul_tensor_4, 0);  mul_tensor_4 = None
        unsqueeze_default_3: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 2);  unsqueeze_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_5: "f32[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_231, 0.02551551815399144);  primals_231 = None
        reshape_default_3: "f32[3072]" = torch.ops.aten.reshape.default(mul_tensor_5, [-1]);  mul_tensor_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_6: "f32[3072]" = torch.ops.aten.mul.Tensor(squeeze_113, reshape_default_3);  reshape_default_3 = None
        unsqueeze_default_4: "f32[1, 3072]" = torch.ops.aten.unsqueeze.default(mul_tensor_6, 0);  mul_tensor_6 = None
        unsqueeze_default_5: "f32[1, 3072, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 2);  unsqueeze_default_4 = None
        mul_tensor_7: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        sub_tensor_1: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_1, mul_tensor_7);  reshape_default_1 = mul_tensor_7 = None
        sub_tensor_2: "f32[1, 3072, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_1);  sub_tensor_1 = unsqueeze_default_1 = None
        mul_tensor_8: "f32[1, 3072, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_2, unsqueeze_default_5);  sub_tensor_2 = unsqueeze_default_5 = None
        mul_tensor_9: "f32[3072]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_3, squeeze_113);  sum_dim_int_list_3 = squeeze_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_4: "f32[3072, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_9, _shape_param_3);  mul_tensor_9 = _shape_param_3 = None
        mul_tensor_10: "f32[3072, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_4, 0.02551551815399144);  reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_5: "f32[3072, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_8, _shape_param_4);  mul_tensor_8 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_11: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_77, sigmoid_11);  convolution_77 = sigmoid_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_12: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_11, 2.0);  mul_tensor_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_13: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_457, mul_tensor_12);  mul_457 = mul_tensor_12 = None
        sum_default: "f32[]" = torch.ops.aten.sum.default(mul_tensor_13);  mul_tensor_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_4: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3]);  where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_6: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 2, 3]);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_6: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(getitem_124, _shape_param_5);  getitem_124 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_7: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_6, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_7: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(primals_222, _shape_param_6);  primals_222 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_3: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_7, unsqueeze_66);  reshape_default_7 = unsqueeze_66 = None
        mul_tensor_14: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(reshape_default_6, sub_tensor_3)
        sum_dim_int_list_8: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 2]);  mul_tensor_14 = None
        mul_tensor_15: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_7, 0.0013020833333333333);  sum_dim_int_list_7 = None
        unsqueeze_default_6: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_15, 0);  mul_tensor_15 = None
        unsqueeze_default_7: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        mul_tensor_16: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_8, 0.0013020833333333333)
        mul_tensor_17: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, squeeze_111)
        mul_tensor_18: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_16, mul_tensor_17);  mul_tensor_16 = mul_tensor_17 = None
        unsqueeze_default_8: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_18, 0);  mul_tensor_18 = None
        unsqueeze_default_9: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 2);  unsqueeze_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_19: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_223, 0.03608439182435161);  primals_223 = None
        reshape_default_8: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_19, [-1]);  mul_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_20: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_111, reshape_default_8);  reshape_default_8 = None
        unsqueeze_default_10: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_20, 0);  mul_tensor_20 = None
        unsqueeze_default_11: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 2);  unsqueeze_default_10 = None
        mul_tensor_21: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_9);  sub_tensor_3 = unsqueeze_default_9 = None
        sub_tensor_4: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_6, mul_tensor_21);  reshape_default_6 = mul_tensor_21 = None
        sub_tensor_5: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_4, unsqueeze_default_7);  sub_tensor_4 = unsqueeze_default_7 = None
        mul_tensor_22: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_5, unsqueeze_default_11);  sub_tensor_5 = unsqueeze_default_11 = None
        mul_tensor_23: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_8, squeeze_111);  sum_dim_int_list_8 = squeeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_9: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_23, _shape_param_7);  mul_tensor_23 = _shape_param_7 = None
        mul_tensor_24: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_9, 0.03608439182435161);  reshape_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_10: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_22, _shape_param_8);  mul_tensor_22 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_9: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3]);  mul_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_127, memory_format = torch.contiguous_format);  getitem_127 = None
        reshape_default_11: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default, _shape_param_9);  clone_default = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_11, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_1: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_219, memory_format = torch.contiguous_format);  primals_219 = None
        reshape_default_12: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_1, _shape_param_10);  clone_default_1 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_6: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_12, unsqueeze_74);  reshape_default_12 = unsqueeze_74 = None
        mul_tensor_25: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_11, sub_tensor_6)
        sum_dim_int_list_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 2]);  mul_tensor_25 = None
        mul_tensor_26: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_10, 0.0008680555555555555);  sum_dim_int_list_10 = None
        unsqueeze_default_12: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_26, 0);  mul_tensor_26 = None
        unsqueeze_default_13: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        mul_tensor_27: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_11, 0.0008680555555555555)
        mul_tensor_28: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_tensor_29: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_27, mul_tensor_28);  mul_tensor_27 = mul_tensor_28 = None
        unsqueeze_default_14: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_29, 0);  mul_tensor_29 = None
        unsqueeze_default_15: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_14, 2);  unsqueeze_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_30: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_220, 0.02946278254943948);  primals_220 = None
        reshape_default_13: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_30, [-1]);  mul_tensor_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_31: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_109, reshape_default_13);  reshape_default_13 = None
        unsqueeze_default_16: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_31, 0);  mul_tensor_31 = None
        unsqueeze_default_17: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 2);  unsqueeze_default_16 = None
        mul_tensor_32: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_6, unsqueeze_default_15);  sub_tensor_6 = unsqueeze_default_15 = None
        sub_tensor_7: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_11, mul_tensor_32);  reshape_default_11 = mul_tensor_32 = None
        sub_tensor_8: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_7, unsqueeze_default_13);  sub_tensor_7 = unsqueeze_default_13 = None
        mul_tensor_33: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_8, unsqueeze_default_17);  sub_tensor_8 = unsqueeze_default_17 = None
        mul_tensor_34: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_11, squeeze_109);  sum_dim_int_list_11 = squeeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_14: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_34, _shape_param_11);  mul_tensor_34 = _shape_param_11 = None
        mul_tensor_35: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_14, 0.02946278254943948);  reshape_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_15: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_33, _shape_param_12);  mul_tensor_33 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_2: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_130, memory_format = torch.contiguous_format);  getitem_130 = None
        reshape_default_16: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_2, _shape_param_13);  clone_default_2 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_16, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_3: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_216, memory_format = torch.contiguous_format);  primals_216 = None
        reshape_default_17: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_3, _shape_param_14);  clone_default_3 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_9: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_17, unsqueeze_82);  reshape_default_17 = unsqueeze_82 = None
        mul_tensor_36: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_16, sub_tensor_9)
        sum_dim_int_list_14: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 2]);  mul_tensor_36 = None
        mul_tensor_37: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_13, 0.0008680555555555555);  sum_dim_int_list_13 = None
        unsqueeze_default_18: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_37, 0);  mul_tensor_37 = None
        unsqueeze_default_19: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 2);  unsqueeze_default_18 = None
        mul_tensor_38: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_14, 0.0008680555555555555)
        mul_tensor_39: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_107, squeeze_107)
        mul_tensor_40: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
        unsqueeze_default_20: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_40, 0);  mul_tensor_40 = None
        unsqueeze_default_21: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_20, 2);  unsqueeze_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_41: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_217, 0.02946278254943948);  primals_217 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_41, [-1]);  mul_tensor_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_42: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_107, reshape_default_18);  reshape_default_18 = None
        unsqueeze_default_22: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_42, 0);  mul_tensor_42 = None
        unsqueeze_default_23: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 2);  unsqueeze_default_22 = None
        mul_tensor_43: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_9, unsqueeze_default_21);  sub_tensor_9 = unsqueeze_default_21 = None
        sub_tensor_10: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_16, mul_tensor_43);  reshape_default_16 = mul_tensor_43 = None
        sub_tensor_11: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_10, unsqueeze_default_19);  sub_tensor_10 = unsqueeze_default_19 = None
        mul_tensor_44: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_11, unsqueeze_default_23);  sub_tensor_11 = unsqueeze_default_23 = None
        mul_tensor_45: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_14, squeeze_107);  sum_dim_int_list_14 = squeeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_19: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_45, _shape_param_15);  mul_tensor_45 = _shape_param_15 = None
        mul_tensor_46: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_19, 0.02946278254943948);  reshape_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_20: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_44, _shape_param_16);  mul_tensor_44 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3]);  mul_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_21: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(getitem_133, _shape_param_17);  getitem_133 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_21, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_22: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(primals_213, _shape_param_18);  primals_213 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_12: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_22, unsqueeze_90);  reshape_default_22 = unsqueeze_90 = None
        mul_tensor_47: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_21, sub_tensor_12)
        sum_dim_int_list_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 2]);  mul_tensor_47 = None
        mul_tensor_48: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_16, 0.0006510416666666666);  sum_dim_int_list_16 = None
        unsqueeze_default_24: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_48, 0);  mul_tensor_48 = None
        unsqueeze_default_25: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 2);  unsqueeze_default_24 = None
        mul_tensor_49: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_17, 0.0006510416666666666)
        mul_tensor_50: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_105, squeeze_105)
        mul_tensor_51: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_49, mul_tensor_50);  mul_tensor_49 = mul_tensor_50 = None
        unsqueeze_default_26: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_51, 0);  mul_tensor_51 = None
        unsqueeze_default_27: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_26, 2);  unsqueeze_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_52: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_214, 0.02551551815399144);  primals_214 = None
        reshape_default_23: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_52, [-1]);  mul_tensor_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_53: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_105, reshape_default_23);  reshape_default_23 = None
        unsqueeze_default_28: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_53, 0);  mul_tensor_53 = None
        unsqueeze_default_29: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 2);  unsqueeze_default_28 = None
        mul_tensor_54: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_12, unsqueeze_default_27);  sub_tensor_12 = unsqueeze_default_27 = None
        sub_tensor_13: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_21, mul_tensor_54);  reshape_default_21 = mul_tensor_54 = None
        sub_tensor_14: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_13, unsqueeze_default_25);  sub_tensor_13 = unsqueeze_default_25 = None
        mul_tensor_55: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_14, unsqueeze_default_29);  sub_tensor_14 = unsqueeze_default_29 = None
        mul_tensor_56: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_17, squeeze_105);  sum_dim_int_list_17 = squeeze_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_24: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_56, _shape_param_19);  mul_tensor_56 = _shape_param_19 = None
        mul_tensor_57: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_24, 0.02551551815399144);  reshape_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_25: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_55, _shape_param_20);  mul_tensor_55 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_58: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(convolution_71, sigmoid_10);  convolution_71 = sigmoid_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_59: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_tensor_58, 2.0);  mul_tensor_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_60: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_538, mul_tensor_59);  mul_538 = mul_tensor_59 = None
        sum_default_1: "f32[]" = torch.ops.aten.sum.default(mul_tensor_60);  mul_tensor_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_18: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 2, 3]);  mul_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3]);  where_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_20: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 2, 3]);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_26: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(getitem_142, _shape_param_21);  getitem_142 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_21: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_26, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_27: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(primals_205, _shape_param_22);  primals_205 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_15: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_27, unsqueeze_98);  reshape_default_27 = unsqueeze_98 = None
        mul_tensor_61: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(reshape_default_26, sub_tensor_15)
        sum_dim_int_list_22: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_61, [0, 2]);  mul_tensor_61 = None
        mul_tensor_62: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_21, 0.0013020833333333333);  sum_dim_int_list_21 = None
        unsqueeze_default_30: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_62, 0);  mul_tensor_62 = None
        unsqueeze_default_31: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 2);  unsqueeze_default_30 = None
        mul_tensor_63: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_22, 0.0013020833333333333)
        mul_tensor_64: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_tensor_65: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None
        unsqueeze_default_32: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_65, 0);  mul_tensor_65 = None
        unsqueeze_default_33: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_32, 2);  unsqueeze_default_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_66: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_206, 0.03608439182435161);  primals_206 = None
        reshape_default_28: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_66, [-1]);  mul_tensor_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_67: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_103, reshape_default_28);  reshape_default_28 = None
        unsqueeze_default_34: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_67, 0);  mul_tensor_67 = None
        unsqueeze_default_35: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 2);  unsqueeze_default_34 = None
        mul_tensor_68: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_15, unsqueeze_default_33);  sub_tensor_15 = unsqueeze_default_33 = None
        sub_tensor_16: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_26, mul_tensor_68);  reshape_default_26 = mul_tensor_68 = None
        sub_tensor_17: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_16, unsqueeze_default_31);  sub_tensor_16 = unsqueeze_default_31 = None
        mul_tensor_69: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_17, unsqueeze_default_35);  sub_tensor_17 = unsqueeze_default_35 = None
        mul_tensor_70: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_22, squeeze_103);  sum_dim_int_list_22 = squeeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_29: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_70, _shape_param_23);  mul_tensor_70 = _shape_param_23 = None
        mul_tensor_71: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_29, 0.03608439182435161);  reshape_default_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_30: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_69, _shape_param_24);  mul_tensor_69 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_4: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_145, memory_format = torch.contiguous_format);  getitem_145 = None
        reshape_default_31: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_4, _shape_param_25);  clone_default_4 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_31, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_5: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_202, memory_format = torch.contiguous_format);  primals_202 = None
        reshape_default_32: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_5, _shape_param_26);  clone_default_5 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_18: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_32, unsqueeze_106);  reshape_default_32 = unsqueeze_106 = None
        mul_tensor_72: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_31, sub_tensor_18)
        sum_dim_int_list_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_72, [0, 2]);  mul_tensor_72 = None
        mul_tensor_73: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_24, 0.0008680555555555555);  sum_dim_int_list_24 = None
        unsqueeze_default_36: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_73, 0);  mul_tensor_73 = None
        unsqueeze_default_37: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 2);  unsqueeze_default_36 = None
        mul_tensor_74: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_25, 0.0008680555555555555)
        mul_tensor_75: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_101, squeeze_101)
        mul_tensor_76: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_74, mul_tensor_75);  mul_tensor_74 = mul_tensor_75 = None
        unsqueeze_default_38: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_76, 0);  mul_tensor_76 = None
        unsqueeze_default_39: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_38, 2);  unsqueeze_default_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_77: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_203, 0.02946278254943948);  primals_203 = None
        reshape_default_33: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_77, [-1]);  mul_tensor_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_78: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_101, reshape_default_33);  reshape_default_33 = None
        unsqueeze_default_40: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_78, 0);  mul_tensor_78 = None
        unsqueeze_default_41: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 2);  unsqueeze_default_40 = None
        mul_tensor_79: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_18, unsqueeze_default_39);  sub_tensor_18 = unsqueeze_default_39 = None
        sub_tensor_19: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_31, mul_tensor_79);  reshape_default_31 = mul_tensor_79 = None
        sub_tensor_20: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_19, unsqueeze_default_37);  sub_tensor_19 = unsqueeze_default_37 = None
        mul_tensor_80: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_20, unsqueeze_default_41);  sub_tensor_20 = unsqueeze_default_41 = None
        mul_tensor_81: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_25, squeeze_101);  sum_dim_int_list_25 = squeeze_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_34: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_81, _shape_param_27);  mul_tensor_81 = _shape_param_27 = None
        mul_tensor_82: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_34, 0.02946278254943948);  reshape_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_35: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_80, _shape_param_28);  mul_tensor_80 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3]);  mul_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_6: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_148, memory_format = torch.contiguous_format);  getitem_148 = None
        reshape_default_36: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_6, _shape_param_29);  clone_default_6 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_36, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_7: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_199, memory_format = torch.contiguous_format);  primals_199 = None
        reshape_default_37: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_7, _shape_param_30);  clone_default_7 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_21: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_37, unsqueeze_114);  reshape_default_37 = unsqueeze_114 = None
        mul_tensor_83: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_36, sub_tensor_21)
        sum_dim_int_list_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_83, [0, 2]);  mul_tensor_83 = None
        mul_tensor_84: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_27, 0.0008680555555555555);  sum_dim_int_list_27 = None
        unsqueeze_default_42: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_84, 0);  mul_tensor_84 = None
        unsqueeze_default_43: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 2);  unsqueeze_default_42 = None
        mul_tensor_85: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_28, 0.0008680555555555555)
        mul_tensor_86: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_99, squeeze_99)
        mul_tensor_87: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        unsqueeze_default_44: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_87, 0);  mul_tensor_87 = None
        unsqueeze_default_45: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_44, 2);  unsqueeze_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_88: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_200, 0.02946278254943948);  primals_200 = None
        reshape_default_38: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_88, [-1]);  mul_tensor_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_89: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_99, reshape_default_38);  reshape_default_38 = None
        unsqueeze_default_46: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_89, 0);  mul_tensor_89 = None
        unsqueeze_default_47: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 2);  unsqueeze_default_46 = None
        mul_tensor_90: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_21, unsqueeze_default_45);  sub_tensor_21 = unsqueeze_default_45 = None
        sub_tensor_22: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_36, mul_tensor_90);  reshape_default_36 = mul_tensor_90 = None
        sub_tensor_23: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_22, unsqueeze_default_43);  sub_tensor_22 = unsqueeze_default_43 = None
        mul_tensor_91: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_23, unsqueeze_default_47);  sub_tensor_23 = unsqueeze_default_47 = None
        mul_tensor_92: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_28, squeeze_99);  sum_dim_int_list_28 = squeeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_39: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_92, _shape_param_31);  mul_tensor_92 = _shape_param_31 = None
        mul_tensor_93: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_39, 0.02946278254943948);  reshape_default_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_40: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_91, _shape_param_32);  mul_tensor_91 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_41: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(getitem_151, _shape_param_33);  getitem_151 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_41, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_42: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(primals_196, _shape_param_34);  primals_196 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_24: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_42, unsqueeze_122);  reshape_default_42 = unsqueeze_122 = None
        mul_tensor_94: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_41, sub_tensor_24)
        sum_dim_int_list_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_94, [0, 2]);  mul_tensor_94 = None
        mul_tensor_95: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_30, 0.0006510416666666666);  sum_dim_int_list_30 = None
        unsqueeze_default_48: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_95, 0);  mul_tensor_95 = None
        unsqueeze_default_49: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 2);  unsqueeze_default_48 = None
        mul_tensor_96: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_31, 0.0006510416666666666)
        mul_tensor_97: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_tensor_98: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_96, mul_tensor_97);  mul_tensor_96 = mul_tensor_97 = None
        unsqueeze_default_50: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_98, 0);  mul_tensor_98 = None
        unsqueeze_default_51: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_50, 2);  unsqueeze_default_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_99: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_197, 0.02551551815399144);  primals_197 = None
        reshape_default_43: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_99, [-1]);  mul_tensor_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_100: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_97, reshape_default_43);  reshape_default_43 = None
        unsqueeze_default_52: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_100, 0);  mul_tensor_100 = None
        unsqueeze_default_53: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 2);  unsqueeze_default_52 = None
        mul_tensor_101: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_24, unsqueeze_default_51);  sub_tensor_24 = unsqueeze_default_51 = None
        sub_tensor_25: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_41, mul_tensor_101);  reshape_default_41 = mul_tensor_101 = None
        sub_tensor_26: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_25, unsqueeze_default_49);  sub_tensor_25 = unsqueeze_default_49 = None
        mul_tensor_102: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_26, unsqueeze_default_53);  sub_tensor_26 = unsqueeze_default_53 = None
        mul_tensor_103: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_31, squeeze_97);  sum_dim_int_list_31 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_44: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_103, _shape_param_35);  mul_tensor_103 = _shape_param_35 = None
        mul_tensor_104: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_44, 0.02551551815399144);  reshape_default_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_45: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_102, _shape_param_36);  mul_tensor_102 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_105: "f32[128, 1536, 6, 6]" = torch.ops.aten.mul.Tensor(mul_619, mul_363);  mul_619 = mul_363 = None
        sum_default_2: "f32[]" = torch.ops.aten.sum.default(mul_tensor_105);  mul_tensor_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_32: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3]);  mul_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3]);  where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_34: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 2, 3]);  add_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_46: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(getitem_160, _shape_param_37);  getitem_160 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_35: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_46, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_47: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(primals_188, _shape_param_38);  primals_188 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_27: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_47, unsqueeze_130);  reshape_default_47 = unsqueeze_130 = None
        mul_tensor_106: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(reshape_default_46, sub_tensor_27)
        sum_dim_int_list_36: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_106, [0, 2]);  mul_tensor_106 = None
        mul_tensor_107: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_35, 0.0013020833333333333);  sum_dim_int_list_35 = None
        unsqueeze_default_54: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_107, 0);  mul_tensor_107 = None
        unsqueeze_default_55: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_54, 2);  unsqueeze_default_54 = None
        mul_tensor_108: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_36, 0.0013020833333333333)
        mul_tensor_109: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, squeeze_95)
        mul_tensor_110: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None
        unsqueeze_default_56: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_110, 0);  mul_tensor_110 = None
        unsqueeze_default_57: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_56, 2);  unsqueeze_default_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_111: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_189, 0.03608439182435161);  primals_189 = None
        reshape_default_48: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_111, [-1]);  mul_tensor_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_112: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_95, reshape_default_48);  reshape_default_48 = None
        unsqueeze_default_58: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_112, 0);  mul_tensor_112 = None
        unsqueeze_default_59: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 2);  unsqueeze_default_58 = None
        mul_tensor_113: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_27, unsqueeze_default_57);  sub_tensor_27 = unsqueeze_default_57 = None
        sub_tensor_28: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_46, mul_tensor_113);  reshape_default_46 = mul_tensor_113 = None
        sub_tensor_29: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_28, unsqueeze_default_55);  sub_tensor_28 = unsqueeze_default_55 = None
        mul_tensor_114: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_29, unsqueeze_default_59);  sub_tensor_29 = unsqueeze_default_59 = None
        mul_tensor_115: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_36, squeeze_95);  sum_dim_int_list_36 = squeeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_49: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_115, _shape_param_39);  mul_tensor_115 = _shape_param_39 = None
        mul_tensor_116: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_49, 0.03608439182435161);  reshape_default_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_50: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_114, _shape_param_40);  mul_tensor_114 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3]);  mul_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_8: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_163, memory_format = torch.contiguous_format);  getitem_163 = None
        reshape_default_51: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_8, _shape_param_41);  clone_default_8 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_51, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_9: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_185, memory_format = torch.contiguous_format);  primals_185 = None
        reshape_default_52: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_9, _shape_param_42);  clone_default_9 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_30: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_52, unsqueeze_138);  reshape_default_52 = unsqueeze_138 = None
        mul_tensor_117: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_51, sub_tensor_30)
        sum_dim_int_list_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_117, [0, 2]);  mul_tensor_117 = None
        mul_tensor_118: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_38, 0.0008680555555555555);  sum_dim_int_list_38 = None
        unsqueeze_default_60: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_118, 0);  mul_tensor_118 = None
        unsqueeze_default_61: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_60, 2);  unsqueeze_default_60 = None
        mul_tensor_119: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_39, 0.0008680555555555555)
        mul_tensor_120: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_93, squeeze_93)
        mul_tensor_121: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_119, mul_tensor_120);  mul_tensor_119 = mul_tensor_120 = None
        unsqueeze_default_62: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_121, 0);  mul_tensor_121 = None
        unsqueeze_default_63: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_62, 2);  unsqueeze_default_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_122: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_186, 0.02946278254943948);  primals_186 = None
        reshape_default_53: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_122, [-1]);  mul_tensor_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_123: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_93, reshape_default_53);  reshape_default_53 = None
        unsqueeze_default_64: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_123, 0);  mul_tensor_123 = None
        unsqueeze_default_65: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_64, 2);  unsqueeze_default_64 = None
        mul_tensor_124: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_30, unsqueeze_default_63);  sub_tensor_30 = unsqueeze_default_63 = None
        sub_tensor_31: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_51, mul_tensor_124);  reshape_default_51 = mul_tensor_124 = None
        sub_tensor_32: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_31, unsqueeze_default_61);  sub_tensor_31 = unsqueeze_default_61 = None
        mul_tensor_125: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_32, unsqueeze_default_65);  sub_tensor_32 = unsqueeze_default_65 = None
        mul_tensor_126: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_39, squeeze_93);  sum_dim_int_list_39 = squeeze_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_54: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_126, _shape_param_43);  mul_tensor_126 = _shape_param_43 = None
        mul_tensor_127: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_54, 0.02946278254943948);  reshape_default_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_55: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_125, _shape_param_44);  mul_tensor_125 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_10: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_166, memory_format = torch.contiguous_format);  getitem_166 = None
        reshape_default_56: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_10, _shape_param_45);  clone_default_10 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_56, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_11: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_182, memory_format = torch.contiguous_format);  primals_182 = None
        reshape_default_57: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_11, _shape_param_46);  clone_default_11 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_33: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_57, unsqueeze_146);  reshape_default_57 = unsqueeze_146 = None
        mul_tensor_128: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_56, sub_tensor_33)
        sum_dim_int_list_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_128, [0, 2]);  mul_tensor_128 = None
        mul_tensor_129: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_41, 0.0008680555555555555);  sum_dim_int_list_41 = None
        unsqueeze_default_66: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_129, 0);  mul_tensor_129 = None
        unsqueeze_default_67: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_66, 2);  unsqueeze_default_66 = None
        mul_tensor_130: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_42, 0.0008680555555555555)
        mul_tensor_131: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_tensor_132: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        unsqueeze_default_68: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_132, 0);  mul_tensor_132 = None
        unsqueeze_default_69: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_68, 2);  unsqueeze_default_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_133: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_183, 0.02946278254943948);  primals_183 = None
        reshape_default_58: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_133, [-1]);  mul_tensor_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_134: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_91, reshape_default_58);  reshape_default_58 = None
        unsqueeze_default_70: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_134, 0);  mul_tensor_134 = None
        unsqueeze_default_71: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_70, 2);  unsqueeze_default_70 = None
        mul_tensor_135: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_33, unsqueeze_default_69);  sub_tensor_33 = unsqueeze_default_69 = None
        sub_tensor_34: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_56, mul_tensor_135);  reshape_default_56 = mul_tensor_135 = None
        sub_tensor_35: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_34, unsqueeze_default_67);  sub_tensor_34 = unsqueeze_default_67 = None
        mul_tensor_136: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_35, unsqueeze_default_71);  sub_tensor_35 = unsqueeze_default_71 = None
        mul_tensor_137: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_42, squeeze_91);  sum_dim_int_list_42 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_59: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_137, _shape_param_47);  mul_tensor_137 = _shape_param_47 = None
        mul_tensor_138: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_59, 0.02946278254943948);  reshape_default_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_60: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_136, _shape_param_48);  mul_tensor_136 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_61: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(getitem_169, _shape_param_49);  getitem_169 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_61, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_62: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(primals_179, _shape_param_50);  primals_179 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_36: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_62, unsqueeze_154);  reshape_default_62 = unsqueeze_154 = None
        mul_tensor_139: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_61, sub_tensor_36)
        sum_dim_int_list_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_139, [0, 2]);  mul_tensor_139 = None
        mul_tensor_140: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_44, 0.0006510416666666666);  sum_dim_int_list_44 = None
        unsqueeze_default_72: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_140, 0);  mul_tensor_140 = None
        unsqueeze_default_73: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_72, 2);  unsqueeze_default_72 = None
        mul_tensor_141: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_45, 0.0006510416666666666)
        mul_tensor_142: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_89, squeeze_89)
        mul_tensor_143: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_141, mul_tensor_142);  mul_tensor_141 = mul_tensor_142 = None
        unsqueeze_default_74: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_143, 0);  mul_tensor_143 = None
        unsqueeze_default_75: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_74, 2);  unsqueeze_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_144: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_180, 0.02551551815399144);  primals_180 = None
        reshape_default_63: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_144, [-1]);  mul_tensor_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_145: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_89, reshape_default_63);  reshape_default_63 = None
        unsqueeze_default_76: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_145, 0);  mul_tensor_145 = None
        unsqueeze_default_77: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_76, 2);  unsqueeze_default_76 = None
        mul_tensor_146: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_36, unsqueeze_default_75);  sub_tensor_36 = unsqueeze_default_75 = None
        sub_tensor_37: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_61, mul_tensor_146);  reshape_default_61 = mul_tensor_146 = None
        sub_tensor_38: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_37, unsqueeze_default_73);  sub_tensor_37 = unsqueeze_default_73 = None
        mul_tensor_147: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_38, unsqueeze_default_77);  sub_tensor_38 = unsqueeze_default_77 = None
        mul_tensor_148: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_45, squeeze_89);  sum_dim_int_list_45 = squeeze_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_64: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_148, _shape_param_51);  mul_tensor_148 = _shape_param_51 = None
        mul_tensor_149: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_64, 0.02551551815399144);  reshape_default_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_65: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_147, _shape_param_52);  mul_tensor_147 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_46: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 2, 3]);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_66: "f32[1, 1536, 1536]" = torch.ops.aten.reshape.default(getitem_172, _shape_param_53);  getitem_172 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_47: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_66, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_67: "f32[1, 1536, 1536]" = torch.ops.aten.reshape.default(primals_176, _shape_param_54);  primals_176 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_39: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_67, unsqueeze_162);  reshape_default_67 = unsqueeze_162 = None
        mul_tensor_150: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_66, sub_tensor_39)
        sum_dim_int_list_48: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_150, [0, 2]);  mul_tensor_150 = None
        mul_tensor_151: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_47, 0.0006510416666666666);  sum_dim_int_list_47 = None
        unsqueeze_default_78: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_151, 0);  mul_tensor_151 = None
        unsqueeze_default_79: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_78, 2);  unsqueeze_default_78 = None
        mul_tensor_152: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_48, 0.0006510416666666666)
        mul_tensor_153: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, squeeze_87)
        mul_tensor_154: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_152, mul_tensor_153);  mul_tensor_152 = mul_tensor_153 = None
        unsqueeze_default_80: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_154, 0);  mul_tensor_154 = None
        unsqueeze_default_81: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_80, 2);  unsqueeze_default_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_155: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_177, 0.02551551815399144);  primals_177 = None
        reshape_default_68: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_155, [-1]);  mul_tensor_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_156: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_87, reshape_default_68);  reshape_default_68 = None
        unsqueeze_default_82: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_156, 0);  mul_tensor_156 = None
        unsqueeze_default_83: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_82, 2);  unsqueeze_default_82 = None
        mul_tensor_157: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_39, unsqueeze_default_81);  sub_tensor_39 = unsqueeze_default_81 = None
        sub_tensor_40: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_66, mul_tensor_157);  reshape_default_66 = mul_tensor_157 = None
        sub_tensor_41: "f32[1, 1536, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_40, unsqueeze_default_79);  sub_tensor_40 = unsqueeze_default_79 = None
        mul_tensor_158: "f32[1, 1536, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_41, unsqueeze_default_83);  sub_tensor_41 = unsqueeze_default_83 = None
        mul_tensor_159: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_48, squeeze_87);  sum_dim_int_list_48 = squeeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_69: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_159, _shape_param_55);  mul_tensor_159 = _shape_param_55 = None
        mul_tensor_160: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_69, 0.02551551815399144);  reshape_default_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_70: "f32[1536, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_158, _shape_param_56);  mul_tensor_158 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_161: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_58, sigmoid_8);  convolution_58 = sigmoid_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_162: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_161, 2.0);  mul_tensor_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_163: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_710, mul_tensor_162);  mul_710 = mul_tensor_162 = None
        sum_default_3: "f32[]" = torch.ops.aten.sum.default(mul_tensor_163);  mul_tensor_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_49: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3]);  where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_51: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 2, 3]);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_71: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(getitem_181, _shape_param_57);  getitem_181 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_52: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_71, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_72: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(primals_168, _shape_param_58);  primals_168 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_42: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_72, unsqueeze_170);  reshape_default_72 = unsqueeze_170 = None
        mul_tensor_164: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(reshape_default_71, sub_tensor_42)
        sum_dim_int_list_53: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_164, [0, 2]);  mul_tensor_164 = None
        mul_tensor_165: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_52, 0.0013020833333333333);  sum_dim_int_list_52 = None
        unsqueeze_default_84: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_165, 0);  mul_tensor_165 = None
        unsqueeze_default_85: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_84, 2);  unsqueeze_default_84 = None
        mul_tensor_166: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_53, 0.0013020833333333333)
        mul_tensor_167: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_tensor_168: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_166, mul_tensor_167);  mul_tensor_166 = mul_tensor_167 = None
        unsqueeze_default_86: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_168, 0);  mul_tensor_168 = None
        unsqueeze_default_87: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_86, 2);  unsqueeze_default_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_169: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_169, 0.03608439182435161);  primals_169 = None
        reshape_default_73: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_169, [-1]);  mul_tensor_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_170: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_85, reshape_default_73);  reshape_default_73 = None
        unsqueeze_default_88: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_170, 0);  mul_tensor_170 = None
        unsqueeze_default_89: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_88, 2);  unsqueeze_default_88 = None
        mul_tensor_171: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_42, unsqueeze_default_87);  sub_tensor_42 = unsqueeze_default_87 = None
        sub_tensor_43: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_71, mul_tensor_171);  reshape_default_71 = mul_tensor_171 = None
        sub_tensor_44: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_43, unsqueeze_default_85);  sub_tensor_43 = unsqueeze_default_85 = None
        mul_tensor_172: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_44, unsqueeze_default_89);  sub_tensor_44 = unsqueeze_default_89 = None
        mul_tensor_173: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_53, squeeze_85);  sum_dim_int_list_53 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_74: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_173, _shape_param_59);  mul_tensor_173 = _shape_param_59 = None
        mul_tensor_174: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_74, 0.03608439182435161);  reshape_default_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_75: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_172, _shape_param_60);  mul_tensor_172 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_735, [0, 2, 3]);  mul_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_12: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_184, memory_format = torch.contiguous_format);  getitem_184 = None
        reshape_default_76: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_12, _shape_param_61);  clone_default_12 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_76, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_13: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_165, memory_format = torch.contiguous_format);  primals_165 = None
        reshape_default_77: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_13, _shape_param_62);  clone_default_13 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_45: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_77, unsqueeze_178);  reshape_default_77 = unsqueeze_178 = None
        mul_tensor_175: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_76, sub_tensor_45)
        sum_dim_int_list_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_175, [0, 2]);  mul_tensor_175 = None
        mul_tensor_176: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_55, 0.0008680555555555555);  sum_dim_int_list_55 = None
        unsqueeze_default_90: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_176, 0);  mul_tensor_176 = None
        unsqueeze_default_91: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_90, 2);  unsqueeze_default_90 = None
        mul_tensor_177: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_56, 0.0008680555555555555)
        mul_tensor_178: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_83, squeeze_83)
        mul_tensor_179: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_177, mul_tensor_178);  mul_tensor_177 = mul_tensor_178 = None
        unsqueeze_default_92: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_179, 0);  mul_tensor_179 = None
        unsqueeze_default_93: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_92, 2);  unsqueeze_default_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_180: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_166, 0.02946278254943948);  primals_166 = None
        reshape_default_78: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_180, [-1]);  mul_tensor_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_181: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_83, reshape_default_78);  reshape_default_78 = None
        unsqueeze_default_94: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_181, 0);  mul_tensor_181 = None
        unsqueeze_default_95: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_94, 2);  unsqueeze_default_94 = None
        mul_tensor_182: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_45, unsqueeze_default_93);  sub_tensor_45 = unsqueeze_default_93 = None
        sub_tensor_46: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_76, mul_tensor_182);  reshape_default_76 = mul_tensor_182 = None
        sub_tensor_47: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_46, unsqueeze_default_91);  sub_tensor_46 = unsqueeze_default_91 = None
        mul_tensor_183: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_47, unsqueeze_default_95);  sub_tensor_47 = unsqueeze_default_95 = None
        mul_tensor_184: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_56, squeeze_83);  sum_dim_int_list_56 = squeeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_79: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_184, _shape_param_63);  mul_tensor_184 = _shape_param_63 = None
        mul_tensor_185: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_79, 0.02946278254943948);  reshape_default_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_80: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_183, _shape_param_64);  mul_tensor_183 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3]);  mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_14: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_187, memory_format = torch.contiguous_format);  getitem_187 = None
        reshape_default_81: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_14, _shape_param_65);  clone_default_14 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_81, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_15: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_162, memory_format = torch.contiguous_format);  primals_162 = None
        reshape_default_82: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_15, _shape_param_66);  clone_default_15 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_48: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_82, unsqueeze_186);  reshape_default_82 = unsqueeze_186 = None
        mul_tensor_186: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_81, sub_tensor_48)
        sum_dim_int_list_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_186, [0, 2]);  mul_tensor_186 = None
        mul_tensor_187: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_58, 0.0008680555555555555);  sum_dim_int_list_58 = None
        unsqueeze_default_96: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_187, 0);  mul_tensor_187 = None
        unsqueeze_default_97: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_96, 2);  unsqueeze_default_96 = None
        mul_tensor_188: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_59, 0.0008680555555555555)
        mul_tensor_189: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_81, squeeze_81)
        mul_tensor_190: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_188, mul_tensor_189);  mul_tensor_188 = mul_tensor_189 = None
        unsqueeze_default_98: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_190, 0);  mul_tensor_190 = None
        unsqueeze_default_99: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_98, 2);  unsqueeze_default_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_191: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_163, 0.02946278254943948);  primals_163 = None
        reshape_default_83: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_191, [-1]);  mul_tensor_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_192: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_81, reshape_default_83);  reshape_default_83 = None
        unsqueeze_default_100: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_192, 0);  mul_tensor_192 = None
        unsqueeze_default_101: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_100, 2);  unsqueeze_default_100 = None
        mul_tensor_193: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_48, unsqueeze_default_99);  sub_tensor_48 = unsqueeze_default_99 = None
        sub_tensor_49: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_81, mul_tensor_193);  reshape_default_81 = mul_tensor_193 = None
        sub_tensor_50: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_49, unsqueeze_default_97);  sub_tensor_49 = unsqueeze_default_97 = None
        mul_tensor_194: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_50, unsqueeze_default_101);  sub_tensor_50 = unsqueeze_default_101 = None
        mul_tensor_195: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_59, squeeze_81);  sum_dim_int_list_59 = squeeze_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_84: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_195, _shape_param_67);  mul_tensor_195 = _shape_param_67 = None
        mul_tensor_196: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_84, 0.02946278254943948);  reshape_default_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_85: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_194, _shape_param_68);  mul_tensor_194 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_86: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(getitem_190, _shape_param_69);  getitem_190 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_86, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_87: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(primals_159, _shape_param_70);  primals_159 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_51: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_87, unsqueeze_194);  reshape_default_87 = unsqueeze_194 = None
        mul_tensor_197: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_86, sub_tensor_51)
        sum_dim_int_list_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_197, [0, 2]);  mul_tensor_197 = None
        mul_tensor_198: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_61, 0.0006510416666666666);  sum_dim_int_list_61 = None
        unsqueeze_default_102: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_198, 0);  mul_tensor_198 = None
        unsqueeze_default_103: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_102, 2);  unsqueeze_default_102 = None
        mul_tensor_199: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_62, 0.0006510416666666666)
        mul_tensor_200: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_tensor_201: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_199, mul_tensor_200);  mul_tensor_199 = mul_tensor_200 = None
        unsqueeze_default_104: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_201, 0);  mul_tensor_201 = None
        unsqueeze_default_105: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_104, 2);  unsqueeze_default_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_202: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_160, 0.02551551815399144);  primals_160 = None
        reshape_default_88: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_202, [-1]);  mul_tensor_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_203: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, reshape_default_88);  reshape_default_88 = None
        unsqueeze_default_106: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_203, 0);  mul_tensor_203 = None
        unsqueeze_default_107: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_106, 2);  unsqueeze_default_106 = None
        mul_tensor_204: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_51, unsqueeze_default_105);  sub_tensor_51 = unsqueeze_default_105 = None
        sub_tensor_52: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_86, mul_tensor_204);  reshape_default_86 = mul_tensor_204 = None
        sub_tensor_53: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_52, unsqueeze_default_103);  sub_tensor_52 = unsqueeze_default_103 = None
        mul_tensor_205: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_53, unsqueeze_default_107);  sub_tensor_53 = unsqueeze_default_107 = None
        mul_tensor_206: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_62, squeeze_79);  sum_dim_int_list_62 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_89: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_206, _shape_param_71);  mul_tensor_206 = _shape_param_71 = None
        mul_tensor_207: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_89, 0.02551551815399144);  reshape_default_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_90: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_205, _shape_param_72);  mul_tensor_205 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_208: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_52, sigmoid_7);  convolution_52 = sigmoid_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_209: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_208, 2.0);  mul_tensor_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_210: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_791, mul_tensor_209);  mul_791 = mul_tensor_209 = None
        sum_default_4: "f32[]" = torch.ops.aten.sum.default(mul_tensor_210);  mul_tensor_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_63: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2, 3]);  mul_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3]);  where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_65: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 2, 3]);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_91: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(getitem_199, _shape_param_73);  getitem_199 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_66: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_91, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_92: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(primals_151, _shape_param_74);  primals_151 = _shape_param_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_54: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_92, unsqueeze_202);  reshape_default_92 = unsqueeze_202 = None
        mul_tensor_211: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(reshape_default_91, sub_tensor_54)
        sum_dim_int_list_67: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_211, [0, 2]);  mul_tensor_211 = None
        mul_tensor_212: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_66, 0.0013020833333333333);  sum_dim_int_list_66 = None
        unsqueeze_default_108: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_212, 0);  mul_tensor_212 = None
        unsqueeze_default_109: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_108, 2);  unsqueeze_default_108 = None
        mul_tensor_213: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_67, 0.0013020833333333333)
        mul_tensor_214: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, squeeze_77)
        mul_tensor_215: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_213, mul_tensor_214);  mul_tensor_213 = mul_tensor_214 = None
        unsqueeze_default_110: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_215, 0);  mul_tensor_215 = None
        unsqueeze_default_111: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_110, 2);  unsqueeze_default_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_216: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_152, 0.03608439182435161);  primals_152 = None
        reshape_default_93: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_216, [-1]);  mul_tensor_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_217: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_77, reshape_default_93);  reshape_default_93 = None
        unsqueeze_default_112: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_217, 0);  mul_tensor_217 = None
        unsqueeze_default_113: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_112, 2);  unsqueeze_default_112 = None
        mul_tensor_218: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_54, unsqueeze_default_111);  sub_tensor_54 = unsqueeze_default_111 = None
        sub_tensor_55: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_91, mul_tensor_218);  reshape_default_91 = mul_tensor_218 = None
        sub_tensor_56: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_55, unsqueeze_default_109);  sub_tensor_55 = unsqueeze_default_109 = None
        mul_tensor_219: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_56, unsqueeze_default_113);  sub_tensor_56 = unsqueeze_default_113 = None
        mul_tensor_220: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_67, squeeze_77);  sum_dim_int_list_67 = squeeze_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_94: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_220, _shape_param_75);  mul_tensor_220 = _shape_param_75 = None
        mul_tensor_221: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_94, 0.03608439182435161);  reshape_default_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_95: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_219, _shape_param_76);  mul_tensor_219 = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 2, 3]);  mul_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_16: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_202, memory_format = torch.contiguous_format);  getitem_202 = None
        reshape_default_96: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_16, _shape_param_77);  clone_default_16 = _shape_param_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_96, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_17: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_148, memory_format = torch.contiguous_format);  primals_148 = None
        reshape_default_97: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_17, _shape_param_78);  clone_default_17 = _shape_param_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_57: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_97, unsqueeze_210);  reshape_default_97 = unsqueeze_210 = None
        mul_tensor_222: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_96, sub_tensor_57)
        sum_dim_int_list_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_222, [0, 2]);  mul_tensor_222 = None
        mul_tensor_223: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_69, 0.0008680555555555555);  sum_dim_int_list_69 = None
        unsqueeze_default_114: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_223, 0);  mul_tensor_223 = None
        unsqueeze_default_115: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_114, 2);  unsqueeze_default_114 = None
        mul_tensor_224: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_70, 0.0008680555555555555)
        mul_tensor_225: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_75, squeeze_75)
        mul_tensor_226: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_224, mul_tensor_225);  mul_tensor_224 = mul_tensor_225 = None
        unsqueeze_default_116: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_226, 0);  mul_tensor_226 = None
        unsqueeze_default_117: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_116, 2);  unsqueeze_default_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_227: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_149, 0.02946278254943948);  primals_149 = None
        reshape_default_98: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_227, [-1]);  mul_tensor_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_228: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_75, reshape_default_98);  reshape_default_98 = None
        unsqueeze_default_118: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_228, 0);  mul_tensor_228 = None
        unsqueeze_default_119: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_118, 2);  unsqueeze_default_118 = None
        mul_tensor_229: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_57, unsqueeze_default_117);  sub_tensor_57 = unsqueeze_default_117 = None
        sub_tensor_58: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_96, mul_tensor_229);  reshape_default_96 = mul_tensor_229 = None
        sub_tensor_59: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_58, unsqueeze_default_115);  sub_tensor_58 = unsqueeze_default_115 = None
        mul_tensor_230: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_59, unsqueeze_default_119);  sub_tensor_59 = unsqueeze_default_119 = None
        mul_tensor_231: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_70, squeeze_75);  sum_dim_int_list_70 = squeeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_99: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_231, _shape_param_79);  mul_tensor_231 = _shape_param_79 = None
        mul_tensor_232: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_99, 0.02946278254943948);  reshape_default_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_100: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_230, _shape_param_80);  mul_tensor_230 = _shape_param_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_834, [0, 2, 3]);  mul_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_18: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_205, memory_format = torch.contiguous_format);  getitem_205 = None
        reshape_default_101: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_18, _shape_param_81);  clone_default_18 = _shape_param_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_101, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_19: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_145, memory_format = torch.contiguous_format);  primals_145 = None
        reshape_default_102: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_19, _shape_param_82);  clone_default_19 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_60: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_102, unsqueeze_218);  reshape_default_102 = unsqueeze_218 = None
        mul_tensor_233: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_101, sub_tensor_60)
        sum_dim_int_list_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_233, [0, 2]);  mul_tensor_233 = None
        mul_tensor_234: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_72, 0.0008680555555555555);  sum_dim_int_list_72 = None
        unsqueeze_default_120: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_234, 0);  mul_tensor_234 = None
        unsqueeze_default_121: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_120, 2);  unsqueeze_default_120 = None
        mul_tensor_235: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_73, 0.0008680555555555555)
        mul_tensor_236: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_tensor_237: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_235, mul_tensor_236);  mul_tensor_235 = mul_tensor_236 = None
        unsqueeze_default_122: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_237, 0);  mul_tensor_237 = None
        unsqueeze_default_123: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_122, 2);  unsqueeze_default_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_238: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_146, 0.02946278254943948);  primals_146 = None
        reshape_default_103: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_238, [-1]);  mul_tensor_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_239: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, reshape_default_103);  reshape_default_103 = None
        unsqueeze_default_124: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_239, 0);  mul_tensor_239 = None
        unsqueeze_default_125: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_124, 2);  unsqueeze_default_124 = None
        mul_tensor_240: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_60, unsqueeze_default_123);  sub_tensor_60 = unsqueeze_default_123 = None
        sub_tensor_61: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_101, mul_tensor_240);  reshape_default_101 = mul_tensor_240 = None
        sub_tensor_62: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_61, unsqueeze_default_121);  sub_tensor_61 = unsqueeze_default_121 = None
        mul_tensor_241: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_62, unsqueeze_default_125);  sub_tensor_62 = unsqueeze_default_125 = None
        mul_tensor_242: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_73, squeeze_73);  sum_dim_int_list_73 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_104: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_242, _shape_param_83);  mul_tensor_242 = _shape_param_83 = None
        mul_tensor_243: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_104, 0.02946278254943948);  reshape_default_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_105: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_241, _shape_param_84);  mul_tensor_241 = _shape_param_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_74: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_852, [0, 2, 3]);  mul_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_106: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(getitem_208, _shape_param_85);  getitem_208 = _shape_param_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_106, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_107: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(primals_142, _shape_param_86);  primals_142 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_63: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_107, unsqueeze_226);  reshape_default_107 = unsqueeze_226 = None
        mul_tensor_244: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_106, sub_tensor_63)
        sum_dim_int_list_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_244, [0, 2]);  mul_tensor_244 = None
        mul_tensor_245: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_75, 0.0006510416666666666);  sum_dim_int_list_75 = None
        unsqueeze_default_126: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_245, 0);  mul_tensor_245 = None
        unsqueeze_default_127: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_126, 2);  unsqueeze_default_126 = None
        mul_tensor_246: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_76, 0.0006510416666666666)
        mul_tensor_247: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_71, squeeze_71)
        mul_tensor_248: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_246, mul_tensor_247);  mul_tensor_246 = mul_tensor_247 = None
        unsqueeze_default_128: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_248, 0);  mul_tensor_248 = None
        unsqueeze_default_129: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_128, 2);  unsqueeze_default_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_249: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_143, 0.02551551815399144);  primals_143 = None
        reshape_default_108: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_249, [-1]);  mul_tensor_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_250: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_71, reshape_default_108);  reshape_default_108 = None
        unsqueeze_default_130: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_250, 0);  mul_tensor_250 = None
        unsqueeze_default_131: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_130, 2);  unsqueeze_default_130 = None
        mul_tensor_251: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_63, unsqueeze_default_129);  sub_tensor_63 = unsqueeze_default_129 = None
        sub_tensor_64: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_106, mul_tensor_251);  reshape_default_106 = mul_tensor_251 = None
        sub_tensor_65: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_64, unsqueeze_default_127);  sub_tensor_64 = unsqueeze_default_127 = None
        mul_tensor_252: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_65, unsqueeze_default_131);  sub_tensor_65 = unsqueeze_default_131 = None
        mul_tensor_253: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_76, squeeze_71);  sum_dim_int_list_76 = squeeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_109: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_253, _shape_param_87);  mul_tensor_253 = _shape_param_87 = None
        mul_tensor_254: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_109, 0.02551551815399144);  reshape_default_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_110: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_252, _shape_param_88);  mul_tensor_252 = _shape_param_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_255: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_46, sigmoid_6);  convolution_46 = sigmoid_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_256: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_255, 2.0);  mul_tensor_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_257: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_872, mul_tensor_256);  mul_872 = mul_tensor_256 = None
        sum_default_5: "f32[]" = torch.ops.aten.sum.default(mul_tensor_257);  mul_tensor_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_77: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3]);  mul_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3]);  where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_79: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 2, 3]);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_111: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(getitem_217, _shape_param_89);  getitem_217 = _shape_param_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_80: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_111, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_112: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(primals_134, _shape_param_90);  primals_134 = _shape_param_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_66: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_112, unsqueeze_234);  reshape_default_112 = unsqueeze_234 = None
        mul_tensor_258: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(reshape_default_111, sub_tensor_66)
        sum_dim_int_list_81: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_258, [0, 2]);  mul_tensor_258 = None
        mul_tensor_259: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_80, 0.0013020833333333333);  sum_dim_int_list_80 = None
        unsqueeze_default_132: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_259, 0);  mul_tensor_259 = None
        unsqueeze_default_133: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_132, 2);  unsqueeze_default_132 = None
        mul_tensor_260: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_81, 0.0013020833333333333)
        mul_tensor_261: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, squeeze_69)
        mul_tensor_262: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_260, mul_tensor_261);  mul_tensor_260 = mul_tensor_261 = None
        unsqueeze_default_134: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_262, 0);  mul_tensor_262 = None
        unsqueeze_default_135: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_134, 2);  unsqueeze_default_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_263: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_135, 0.03608439182435161);  primals_135 = None
        reshape_default_113: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_263, [-1]);  mul_tensor_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_264: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_69, reshape_default_113);  reshape_default_113 = None
        unsqueeze_default_136: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_264, 0);  mul_tensor_264 = None
        unsqueeze_default_137: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_136, 2);  unsqueeze_default_136 = None
        mul_tensor_265: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_66, unsqueeze_default_135);  sub_tensor_66 = unsqueeze_default_135 = None
        sub_tensor_67: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_111, mul_tensor_265);  reshape_default_111 = mul_tensor_265 = None
        sub_tensor_68: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_67, unsqueeze_default_133);  sub_tensor_67 = unsqueeze_default_133 = None
        mul_tensor_266: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_68, unsqueeze_default_137);  sub_tensor_68 = unsqueeze_default_137 = None
        mul_tensor_267: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_81, squeeze_69);  sum_dim_int_list_81 = squeeze_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_114: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_267, _shape_param_91);  mul_tensor_267 = _shape_param_91 = None
        mul_tensor_268: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_114, 0.03608439182435161);  reshape_default_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_115: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_266, _shape_param_92);  mul_tensor_266 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3]);  mul_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_20: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_220, memory_format = torch.contiguous_format);  getitem_220 = None
        reshape_default_116: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_20, _shape_param_93);  clone_default_20 = _shape_param_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_116, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_21: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_131, memory_format = torch.contiguous_format);  primals_131 = None
        reshape_default_117: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_21, _shape_param_94);  clone_default_21 = _shape_param_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_69: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_117, unsqueeze_242);  reshape_default_117 = unsqueeze_242 = None
        mul_tensor_269: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_116, sub_tensor_69)
        sum_dim_int_list_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_269, [0, 2]);  mul_tensor_269 = None
        mul_tensor_270: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_83, 0.0008680555555555555);  sum_dim_int_list_83 = None
        unsqueeze_default_138: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_270, 0);  mul_tensor_270 = None
        unsqueeze_default_139: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_138, 2);  unsqueeze_default_138 = None
        mul_tensor_271: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_84, 0.0008680555555555555)
        mul_tensor_272: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_tensor_273: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_271, mul_tensor_272);  mul_tensor_271 = mul_tensor_272 = None
        unsqueeze_default_140: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_273, 0);  mul_tensor_273 = None
        unsqueeze_default_141: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_140, 2);  unsqueeze_default_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_274: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_132, 0.02946278254943948);  primals_132 = None
        reshape_default_118: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_274, [-1]);  mul_tensor_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_275: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, reshape_default_118);  reshape_default_118 = None
        unsqueeze_default_142: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_275, 0);  mul_tensor_275 = None
        unsqueeze_default_143: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_142, 2);  unsqueeze_default_142 = None
        mul_tensor_276: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_69, unsqueeze_default_141);  sub_tensor_69 = unsqueeze_default_141 = None
        sub_tensor_70: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_116, mul_tensor_276);  reshape_default_116 = mul_tensor_276 = None
        sub_tensor_71: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_70, unsqueeze_default_139);  sub_tensor_70 = unsqueeze_default_139 = None
        mul_tensor_277: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_71, unsqueeze_default_143);  sub_tensor_71 = unsqueeze_default_143 = None
        mul_tensor_278: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_84, squeeze_67);  sum_dim_int_list_84 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_119: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_278, _shape_param_95);  mul_tensor_278 = _shape_param_95 = None
        mul_tensor_279: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_119, 0.02946278254943948);  reshape_default_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_120: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_277, _shape_param_96);  mul_tensor_277 = _shape_param_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_915, [0, 2, 3]);  mul_915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_22: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_223, memory_format = torch.contiguous_format);  getitem_223 = None
        reshape_default_121: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_22, _shape_param_97);  clone_default_22 = _shape_param_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_121, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_23: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_128, memory_format = torch.contiguous_format);  primals_128 = None
        reshape_default_122: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_23, _shape_param_98);  clone_default_23 = _shape_param_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_72: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_122, unsqueeze_250);  reshape_default_122 = unsqueeze_250 = None
        mul_tensor_280: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_121, sub_tensor_72)
        sum_dim_int_list_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_280, [0, 2]);  mul_tensor_280 = None
        mul_tensor_281: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_86, 0.0008680555555555555);  sum_dim_int_list_86 = None
        unsqueeze_default_144: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_281, 0);  mul_tensor_281 = None
        unsqueeze_default_145: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_144, 2);  unsqueeze_default_144 = None
        mul_tensor_282: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_87, 0.0008680555555555555)
        mul_tensor_283: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_65, squeeze_65)
        mul_tensor_284: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_282, mul_tensor_283);  mul_tensor_282 = mul_tensor_283 = None
        unsqueeze_default_146: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_284, 0);  mul_tensor_284 = None
        unsqueeze_default_147: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_146, 2);  unsqueeze_default_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_285: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_129, 0.02946278254943948);  primals_129 = None
        reshape_default_123: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_285, [-1]);  mul_tensor_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_286: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_65, reshape_default_123);  reshape_default_123 = None
        unsqueeze_default_148: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_286, 0);  mul_tensor_286 = None
        unsqueeze_default_149: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_148, 2);  unsqueeze_default_148 = None
        mul_tensor_287: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_72, unsqueeze_default_147);  sub_tensor_72 = unsqueeze_default_147 = None
        sub_tensor_73: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_121, mul_tensor_287);  reshape_default_121 = mul_tensor_287 = None
        sub_tensor_74: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_73, unsqueeze_default_145);  sub_tensor_73 = unsqueeze_default_145 = None
        mul_tensor_288: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_74, unsqueeze_default_149);  sub_tensor_74 = unsqueeze_default_149 = None
        mul_tensor_289: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_87, squeeze_65);  sum_dim_int_list_87 = squeeze_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_124: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_289, _shape_param_99);  mul_tensor_289 = _shape_param_99 = None
        mul_tensor_290: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_124, 0.02946278254943948);  reshape_default_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_125: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_288, _shape_param_100);  mul_tensor_288 = _shape_param_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_933, [0, 2, 3]);  mul_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_126: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(getitem_226, _shape_param_101);  getitem_226 = _shape_param_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_126, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_127: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(primals_125, _shape_param_102);  primals_125 = _shape_param_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_75: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_127, unsqueeze_258);  reshape_default_127 = unsqueeze_258 = None
        mul_tensor_291: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_126, sub_tensor_75)
        sum_dim_int_list_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_291, [0, 2]);  mul_tensor_291 = None
        mul_tensor_292: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_89, 0.0006510416666666666);  sum_dim_int_list_89 = None
        unsqueeze_default_150: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_292, 0);  mul_tensor_292 = None
        unsqueeze_default_151: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_150, 2);  unsqueeze_default_150 = None
        mul_tensor_293: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_90, 0.0006510416666666666)
        mul_tensor_294: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_63, squeeze_63)
        mul_tensor_295: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_293, mul_tensor_294);  mul_tensor_293 = mul_tensor_294 = None
        unsqueeze_default_152: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_295, 0);  mul_tensor_295 = None
        unsqueeze_default_153: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_152, 2);  unsqueeze_default_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_296: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_126, 0.02551551815399144);  primals_126 = None
        reshape_default_128: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_296, [-1]);  mul_tensor_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_297: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_63, reshape_default_128);  reshape_default_128 = None
        unsqueeze_default_154: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_297, 0);  mul_tensor_297 = None
        unsqueeze_default_155: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_154, 2);  unsqueeze_default_154 = None
        mul_tensor_298: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_75, unsqueeze_default_153);  sub_tensor_75 = unsqueeze_default_153 = None
        sub_tensor_76: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_126, mul_tensor_298);  reshape_default_126 = mul_tensor_298 = None
        sub_tensor_77: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_76, unsqueeze_default_151);  sub_tensor_76 = unsqueeze_default_151 = None
        mul_tensor_299: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_77, unsqueeze_default_155);  sub_tensor_77 = unsqueeze_default_155 = None
        mul_tensor_300: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_90, squeeze_63);  sum_dim_int_list_90 = squeeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_129: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_300, _shape_param_103);  mul_tensor_300 = _shape_param_103 = None
        mul_tensor_301: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_129, 0.02551551815399144);  reshape_default_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_130: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_299, _shape_param_104);  mul_tensor_299 = _shape_param_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_302: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_40, sigmoid_5);  convolution_40 = sigmoid_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_303: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_302, 2.0);  mul_tensor_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_304: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_953, mul_tensor_303);  mul_953 = mul_tensor_303 = None
        sum_default_6: "f32[]" = torch.ops.aten.sum.default(mul_tensor_304);  mul_tensor_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_91: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 2, 3]);  mul_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3]);  where_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_93: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_183, [0, 2, 3]);  add_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_131: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(getitem_235, _shape_param_105);  getitem_235 = _shape_param_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_94: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_131, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_132: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(primals_117, _shape_param_106);  primals_117 = _shape_param_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_78: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_132, unsqueeze_266);  reshape_default_132 = unsqueeze_266 = None
        mul_tensor_305: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(reshape_default_131, sub_tensor_78)
        sum_dim_int_list_95: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_305, [0, 2]);  mul_tensor_305 = None
        mul_tensor_306: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_94, 0.0013020833333333333);  sum_dim_int_list_94 = None
        unsqueeze_default_156: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_306, 0);  mul_tensor_306 = None
        unsqueeze_default_157: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_156, 2);  unsqueeze_default_156 = None
        mul_tensor_307: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_95, 0.0013020833333333333)
        mul_tensor_308: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_tensor_309: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_307, mul_tensor_308);  mul_tensor_307 = mul_tensor_308 = None
        unsqueeze_default_158: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_309, 0);  mul_tensor_309 = None
        unsqueeze_default_159: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_158, 2);  unsqueeze_default_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_310: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_118, 0.03608439182435161);  primals_118 = None
        reshape_default_133: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_310, [-1]);  mul_tensor_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_311: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_61, reshape_default_133);  reshape_default_133 = None
        unsqueeze_default_160: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_311, 0);  mul_tensor_311 = None
        unsqueeze_default_161: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_160, 2);  unsqueeze_default_160 = None
        mul_tensor_312: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_78, unsqueeze_default_159);  sub_tensor_78 = unsqueeze_default_159 = None
        sub_tensor_79: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_131, mul_tensor_312);  reshape_default_131 = mul_tensor_312 = None
        sub_tensor_80: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_79, unsqueeze_default_157);  sub_tensor_79 = unsqueeze_default_157 = None
        mul_tensor_313: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_80, unsqueeze_default_161);  sub_tensor_80 = unsqueeze_default_161 = None
        mul_tensor_314: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_95, squeeze_61);  sum_dim_int_list_95 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_134: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_314, _shape_param_107);  mul_tensor_314 = _shape_param_107 = None
        mul_tensor_315: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_134, 0.03608439182435161);  reshape_default_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_135: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_313, _shape_param_108);  mul_tensor_313 = _shape_param_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_978, [0, 2, 3]);  mul_978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_24: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_238, memory_format = torch.contiguous_format);  getitem_238 = None
        reshape_default_136: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_24, _shape_param_109);  clone_default_24 = _shape_param_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_136, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_25: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_114, memory_format = torch.contiguous_format);  primals_114 = None
        reshape_default_137: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_25, _shape_param_110);  clone_default_25 = _shape_param_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_81: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_137, unsqueeze_274);  reshape_default_137 = unsqueeze_274 = None
        mul_tensor_316: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_136, sub_tensor_81)
        sum_dim_int_list_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_316, [0, 2]);  mul_tensor_316 = None
        mul_tensor_317: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_97, 0.0008680555555555555);  sum_dim_int_list_97 = None
        unsqueeze_default_162: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_317, 0);  mul_tensor_317 = None
        unsqueeze_default_163: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_162, 2);  unsqueeze_default_162 = None
        mul_tensor_318: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_98, 0.0008680555555555555)
        mul_tensor_319: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_59, squeeze_59)
        mul_tensor_320: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_318, mul_tensor_319);  mul_tensor_318 = mul_tensor_319 = None
        unsqueeze_default_164: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_320, 0);  mul_tensor_320 = None
        unsqueeze_default_165: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_164, 2);  unsqueeze_default_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_321: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_115, 0.02946278254943948);  primals_115 = None
        reshape_default_138: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_321, [-1]);  mul_tensor_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_322: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_59, reshape_default_138);  reshape_default_138 = None
        unsqueeze_default_166: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_322, 0);  mul_tensor_322 = None
        unsqueeze_default_167: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_166, 2);  unsqueeze_default_166 = None
        mul_tensor_323: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_81, unsqueeze_default_165);  sub_tensor_81 = unsqueeze_default_165 = None
        sub_tensor_82: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_136, mul_tensor_323);  reshape_default_136 = mul_tensor_323 = None
        sub_tensor_83: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_82, unsqueeze_default_163);  sub_tensor_82 = unsqueeze_default_163 = None
        mul_tensor_324: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_83, unsqueeze_default_167);  sub_tensor_83 = unsqueeze_default_167 = None
        mul_tensor_325: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_98, squeeze_59);  sum_dim_int_list_98 = squeeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_139: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_325, _shape_param_111);  mul_tensor_325 = _shape_param_111 = None
        mul_tensor_326: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_139, 0.02946278254943948);  reshape_default_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_140: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_324, _shape_param_112);  mul_tensor_324 = _shape_param_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_996, [0, 2, 3]);  mul_996 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_26: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_241, memory_format = torch.contiguous_format);  getitem_241 = None
        reshape_default_141: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_26, _shape_param_113);  clone_default_26 = _shape_param_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_141, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_27: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_111, memory_format = torch.contiguous_format);  primals_111 = None
        reshape_default_142: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_27, _shape_param_114);  clone_default_27 = _shape_param_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_84: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_142, unsqueeze_282);  reshape_default_142 = unsqueeze_282 = None
        mul_tensor_327: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_141, sub_tensor_84)
        sum_dim_int_list_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_327, [0, 2]);  mul_tensor_327 = None
        mul_tensor_328: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_100, 0.0008680555555555555);  sum_dim_int_list_100 = None
        unsqueeze_default_168: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_328, 0);  mul_tensor_328 = None
        unsqueeze_default_169: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_168, 2);  unsqueeze_default_168 = None
        mul_tensor_329: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_101, 0.0008680555555555555)
        mul_tensor_330: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_57, squeeze_57)
        mul_tensor_331: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_329, mul_tensor_330);  mul_tensor_329 = mul_tensor_330 = None
        unsqueeze_default_170: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_331, 0);  mul_tensor_331 = None
        unsqueeze_default_171: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_170, 2);  unsqueeze_default_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_332: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_112, 0.02946278254943948);  primals_112 = None
        reshape_default_143: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_332, [-1]);  mul_tensor_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_333: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_57, reshape_default_143);  reshape_default_143 = None
        unsqueeze_default_172: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_333, 0);  mul_tensor_333 = None
        unsqueeze_default_173: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_172, 2);  unsqueeze_default_172 = None
        mul_tensor_334: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_84, unsqueeze_default_171);  sub_tensor_84 = unsqueeze_default_171 = None
        sub_tensor_85: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_141, mul_tensor_334);  reshape_default_141 = mul_tensor_334 = None
        sub_tensor_86: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_85, unsqueeze_default_169);  sub_tensor_85 = unsqueeze_default_169 = None
        mul_tensor_335: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_86, unsqueeze_default_173);  sub_tensor_86 = unsqueeze_default_173 = None
        mul_tensor_336: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_101, squeeze_57);  sum_dim_int_list_101 = squeeze_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_144: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_336, _shape_param_115);  mul_tensor_336 = _shape_param_115 = None
        mul_tensor_337: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_144, 0.02946278254943948);  reshape_default_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_145: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_335, _shape_param_116);  mul_tensor_335 = _shape_param_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1014, [0, 2, 3]);  mul_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_146: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(getitem_244, _shape_param_117);  getitem_244 = _shape_param_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_146, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_147: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(primals_108, _shape_param_118);  primals_108 = _shape_param_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_87: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_147, unsqueeze_290);  reshape_default_147 = unsqueeze_290 = None
        mul_tensor_338: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_146, sub_tensor_87)
        sum_dim_int_list_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_338, [0, 2]);  mul_tensor_338 = None
        mul_tensor_339: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_103, 0.0006510416666666666);  sum_dim_int_list_103 = None
        unsqueeze_default_174: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_339, 0);  mul_tensor_339 = None
        unsqueeze_default_175: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_174, 2);  unsqueeze_default_174 = None
        mul_tensor_340: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_104, 0.0006510416666666666)
        mul_tensor_341: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_tensor_342: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_340, mul_tensor_341);  mul_tensor_340 = mul_tensor_341 = None
        unsqueeze_default_176: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_342, 0);  mul_tensor_342 = None
        unsqueeze_default_177: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_176, 2);  unsqueeze_default_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_343: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_109, 0.02551551815399144);  primals_109 = None
        reshape_default_148: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_343, [-1]);  mul_tensor_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_344: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, reshape_default_148);  reshape_default_148 = None
        unsqueeze_default_178: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_344, 0);  mul_tensor_344 = None
        unsqueeze_default_179: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_178, 2);  unsqueeze_default_178 = None
        mul_tensor_345: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_87, unsqueeze_default_177);  sub_tensor_87 = unsqueeze_default_177 = None
        sub_tensor_88: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_146, mul_tensor_345);  reshape_default_146 = mul_tensor_345 = None
        sub_tensor_89: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_88, unsqueeze_default_175);  sub_tensor_88 = unsqueeze_default_175 = None
        mul_tensor_346: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_89, unsqueeze_default_179);  sub_tensor_89 = unsqueeze_default_179 = None
        mul_tensor_347: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_104, squeeze_55);  sum_dim_int_list_104 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_149: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_347, _shape_param_119);  mul_tensor_347 = _shape_param_119 = None
        mul_tensor_348: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_149, 0.02551551815399144);  reshape_default_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_150: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_346, _shape_param_120);  mul_tensor_346 = _shape_param_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_349: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(convolution_34, sigmoid_4);  convolution_34 = sigmoid_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_350: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_tensor_349, 2.0);  mul_tensor_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_351: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1034, mul_tensor_350);  mul_1034 = mul_tensor_350 = None
        sum_default_7: "f32[]" = torch.ops.aten.sum.default(mul_tensor_351);  mul_tensor_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_105: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1041, [0, 2, 3]);  mul_1041 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3]);  where_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_107: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_193, [0, 2, 3]);  add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_151: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(getitem_253, _shape_param_121);  getitem_253 = _shape_param_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_108: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_151, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_152: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(primals_100, _shape_param_122);  primals_100 = _shape_param_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_90: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_152, unsqueeze_298);  reshape_default_152 = unsqueeze_298 = None
        mul_tensor_352: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(reshape_default_151, sub_tensor_90)
        sum_dim_int_list_109: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_352, [0, 2]);  mul_tensor_352 = None
        mul_tensor_353: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_108, 0.0013020833333333333);  sum_dim_int_list_108 = None
        unsqueeze_default_180: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_353, 0);  mul_tensor_353 = None
        unsqueeze_default_181: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_180, 2);  unsqueeze_default_180 = None
        mul_tensor_354: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_109, 0.0013020833333333333)
        mul_tensor_355: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, squeeze_53)
        mul_tensor_356: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_354, mul_tensor_355);  mul_tensor_354 = mul_tensor_355 = None
        unsqueeze_default_182: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_356, 0);  mul_tensor_356 = None
        unsqueeze_default_183: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_182, 2);  unsqueeze_default_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_357: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_101, 0.03608439182435161);  primals_101 = None
        reshape_default_153: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_357, [-1]);  mul_tensor_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_358: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_53, reshape_default_153);  reshape_default_153 = None
        unsqueeze_default_184: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_358, 0);  mul_tensor_358 = None
        unsqueeze_default_185: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_184, 2);  unsqueeze_default_184 = None
        mul_tensor_359: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_90, unsqueeze_default_183);  sub_tensor_90 = unsqueeze_default_183 = None
        sub_tensor_91: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_151, mul_tensor_359);  reshape_default_151 = mul_tensor_359 = None
        sub_tensor_92: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_91, unsqueeze_default_181);  sub_tensor_91 = unsqueeze_default_181 = None
        mul_tensor_360: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_92, unsqueeze_default_185);  sub_tensor_92 = unsqueeze_default_185 = None
        mul_tensor_361: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_109, squeeze_53);  sum_dim_int_list_109 = squeeze_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_154: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_361, _shape_param_123);  mul_tensor_361 = _shape_param_123 = None
        mul_tensor_362: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_154, 0.03608439182435161);  reshape_default_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_155: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_360, _shape_param_124);  mul_tensor_360 = _shape_param_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1059, [0, 2, 3]);  mul_1059 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_28: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_256, memory_format = torch.contiguous_format);  getitem_256 = None
        reshape_default_156: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_28, _shape_param_125);  clone_default_28 = _shape_param_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_156, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_29: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_97, memory_format = torch.contiguous_format);  primals_97 = None
        reshape_default_157: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_29, _shape_param_126);  clone_default_29 = _shape_param_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_93: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_157, unsqueeze_306);  reshape_default_157 = unsqueeze_306 = None
        mul_tensor_363: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_156, sub_tensor_93)
        sum_dim_int_list_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_363, [0, 2]);  mul_tensor_363 = None
        mul_tensor_364: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_111, 0.0008680555555555555);  sum_dim_int_list_111 = None
        unsqueeze_default_186: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_364, 0);  mul_tensor_364 = None
        unsqueeze_default_187: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_186, 2);  unsqueeze_default_186 = None
        mul_tensor_365: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_112, 0.0008680555555555555)
        mul_tensor_366: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_51, squeeze_51)
        mul_tensor_367: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_365, mul_tensor_366);  mul_tensor_365 = mul_tensor_366 = None
        unsqueeze_default_188: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_367, 0);  mul_tensor_367 = None
        unsqueeze_default_189: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_188, 2);  unsqueeze_default_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_368: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_98, 0.02946278254943948);  primals_98 = None
        reshape_default_158: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_368, [-1]);  mul_tensor_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_369: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_51, reshape_default_158);  reshape_default_158 = None
        unsqueeze_default_190: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_369, 0);  mul_tensor_369 = None
        unsqueeze_default_191: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_190, 2);  unsqueeze_default_190 = None
        mul_tensor_370: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_93, unsqueeze_default_189);  sub_tensor_93 = unsqueeze_default_189 = None
        sub_tensor_94: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_156, mul_tensor_370);  reshape_default_156 = mul_tensor_370 = None
        sub_tensor_95: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_94, unsqueeze_default_187);  sub_tensor_94 = unsqueeze_default_187 = None
        mul_tensor_371: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_95, unsqueeze_default_191);  sub_tensor_95 = unsqueeze_default_191 = None
        mul_tensor_372: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_112, squeeze_51);  sum_dim_int_list_112 = squeeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_159: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_372, _shape_param_127);  mul_tensor_372 = _shape_param_127 = None
        mul_tensor_373: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_159, 0.02946278254943948);  reshape_default_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_160: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_371, _shape_param_128);  mul_tensor_371 = _shape_param_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1077, [0, 2, 3]);  mul_1077 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_30: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_259, memory_format = torch.contiguous_format);  getitem_259 = None
        reshape_default_161: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_30, _shape_param_129);  clone_default_30 = _shape_param_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_161, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_31: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_94, memory_format = torch.contiguous_format);  primals_94 = None
        reshape_default_162: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_31, _shape_param_130);  clone_default_31 = _shape_param_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_96: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_162, unsqueeze_314);  reshape_default_162 = unsqueeze_314 = None
        mul_tensor_374: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_161, sub_tensor_96)
        sum_dim_int_list_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_374, [0, 2]);  mul_tensor_374 = None
        mul_tensor_375: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_114, 0.0008680555555555555);  sum_dim_int_list_114 = None
        unsqueeze_default_192: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_375, 0);  mul_tensor_375 = None
        unsqueeze_default_193: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_192, 2);  unsqueeze_default_192 = None
        mul_tensor_376: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_115, 0.0008680555555555555)
        mul_tensor_377: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_tensor_378: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_376, mul_tensor_377);  mul_tensor_376 = mul_tensor_377 = None
        unsqueeze_default_194: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_378, 0);  mul_tensor_378 = None
        unsqueeze_default_195: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_194, 2);  unsqueeze_default_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_379: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_95, 0.02946278254943948);  primals_95 = None
        reshape_default_163: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_379, [-1]);  mul_tensor_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_380: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_49, reshape_default_163);  reshape_default_163 = None
        unsqueeze_default_196: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_380, 0);  mul_tensor_380 = None
        unsqueeze_default_197: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_196, 2);  unsqueeze_default_196 = None
        mul_tensor_381: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_96, unsqueeze_default_195);  sub_tensor_96 = unsqueeze_default_195 = None
        sub_tensor_97: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_161, mul_tensor_381);  reshape_default_161 = mul_tensor_381 = None
        sub_tensor_98: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_97, unsqueeze_default_193);  sub_tensor_97 = unsqueeze_default_193 = None
        mul_tensor_382: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_98, unsqueeze_default_197);  sub_tensor_98 = unsqueeze_default_197 = None
        mul_tensor_383: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_115, squeeze_49);  sum_dim_int_list_115 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_164: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_383, _shape_param_131);  mul_tensor_383 = _shape_param_131 = None
        mul_tensor_384: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_164, 0.02946278254943948);  reshape_default_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_165: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_382, _shape_param_132);  mul_tensor_382 = _shape_param_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1095, [0, 2, 3]);  mul_1095 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_166: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(getitem_262, _shape_param_133);  getitem_262 = _shape_param_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_166, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_167: "f32[1, 768, 1536]" = torch.ops.aten.reshape.default(primals_91, _shape_param_134);  primals_91 = _shape_param_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_99: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_167, unsqueeze_322);  reshape_default_167 = unsqueeze_322 = None
        mul_tensor_385: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(reshape_default_166, sub_tensor_99)
        sum_dim_int_list_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_385, [0, 2]);  mul_tensor_385 = None
        mul_tensor_386: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_117, 0.0006510416666666666);  sum_dim_int_list_117 = None
        unsqueeze_default_198: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_386, 0);  mul_tensor_386 = None
        unsqueeze_default_199: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_198, 2);  unsqueeze_default_198 = None
        mul_tensor_387: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_118, 0.0006510416666666666)
        mul_tensor_388: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_47, squeeze_47)
        mul_tensor_389: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_387, mul_tensor_388);  mul_tensor_387 = mul_tensor_388 = None
        unsqueeze_default_200: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_389, 0);  mul_tensor_389 = None
        unsqueeze_default_201: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_200, 2);  unsqueeze_default_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_390: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_92, 0.02551551815399144);  primals_92 = None
        reshape_default_168: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_390, [-1]);  mul_tensor_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_391: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_47, reshape_default_168);  reshape_default_168 = None
        unsqueeze_default_202: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_391, 0);  mul_tensor_391 = None
        unsqueeze_default_203: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_202, 2);  unsqueeze_default_202 = None
        mul_tensor_392: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_99, unsqueeze_default_201);  sub_tensor_99 = unsqueeze_default_201 = None
        sub_tensor_100: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(reshape_default_166, mul_tensor_392);  reshape_default_166 = mul_tensor_392 = None
        sub_tensor_101: "f32[1, 768, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_100, unsqueeze_default_199);  sub_tensor_100 = unsqueeze_default_199 = None
        mul_tensor_393: "f32[1, 768, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor_101, unsqueeze_default_203);  sub_tensor_101 = unsqueeze_default_203 = None
        mul_tensor_394: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_118, squeeze_47);  sum_dim_int_list_118 = squeeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_169: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_394, _shape_param_135);  mul_tensor_394 = _shape_param_135 = None
        mul_tensor_395: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_169, 0.02551551815399144);  reshape_default_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_170: "f32[768, 1536, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_393, _shape_param_136);  mul_tensor_393 = _shape_param_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_396: "f32[128, 1536, 12, 12]" = torch.ops.aten.mul.Tensor(mul_1115, mul_162);  mul_1115 = mul_162 = None
        sum_default_8: "f32[]" = torch.ops.aten.sum.default(mul_tensor_396);  mul_tensor_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_119: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_1122, [0, 2, 3]);  mul_1122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3]);  where_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_121: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 2, 3]);  add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_171: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(getitem_271, _shape_param_137);  getitem_271 = _shape_param_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_122: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_171, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_172: "f32[1, 1536, 768]" = torch.ops.aten.reshape.default(primals_83, _shape_param_138);  primals_83 = _shape_param_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_102: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_172, unsqueeze_330);  reshape_default_172 = unsqueeze_330 = None
        mul_tensor_397: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(reshape_default_171, sub_tensor_102)
        sum_dim_int_list_123: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_397, [0, 2]);  mul_tensor_397 = None
        mul_tensor_398: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_122, 0.0013020833333333333);  sum_dim_int_list_122 = None
        unsqueeze_default_204: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_398, 0);  mul_tensor_398 = None
        unsqueeze_default_205: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_204, 2);  unsqueeze_default_204 = None
        mul_tensor_399: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_123, 0.0013020833333333333)
        mul_tensor_400: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, squeeze_45)
        mul_tensor_401: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_399, mul_tensor_400);  mul_tensor_399 = mul_tensor_400 = None
        unsqueeze_default_206: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_401, 0);  mul_tensor_401 = None
        unsqueeze_default_207: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_206, 2);  unsqueeze_default_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_402: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_84, 0.03608439182435161);  primals_84 = None
        reshape_default_173: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_402, [-1]);  mul_tensor_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_403: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_45, reshape_default_173);  reshape_default_173 = None
        unsqueeze_default_208: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_403, 0);  mul_tensor_403 = None
        unsqueeze_default_209: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_208, 2);  unsqueeze_default_208 = None
        mul_tensor_404: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_102, unsqueeze_default_207);  sub_tensor_102 = unsqueeze_default_207 = None
        sub_tensor_103: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(reshape_default_171, mul_tensor_404);  reshape_default_171 = mul_tensor_404 = None
        sub_tensor_104: "f32[1, 1536, 768]" = torch.ops.aten.sub.Tensor(sub_tensor_103, unsqueeze_default_205);  sub_tensor_103 = unsqueeze_default_205 = None
        mul_tensor_405: "f32[1, 1536, 768]" = torch.ops.aten.mul.Tensor(sub_tensor_104, unsqueeze_default_209);  sub_tensor_104 = unsqueeze_default_209 = None
        mul_tensor_406: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_123, squeeze_45);  sum_dim_int_list_123 = squeeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_174: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_406, _shape_param_139);  mul_tensor_406 = _shape_param_139 = None
        mul_tensor_407: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_174, 0.03608439182435161);  reshape_default_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_175: "f32[1536, 768, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_405, _shape_param_140);  mul_tensor_405 = _shape_param_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1140, [0, 2, 3]);  mul_1140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_32: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_274, memory_format = torch.contiguous_format);  getitem_274 = None
        reshape_default_176: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_32, _shape_param_141);  clone_default_32 = _shape_param_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_176, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_33: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_80, memory_format = torch.contiguous_format);  primals_80 = None
        reshape_default_177: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_33, _shape_param_142);  clone_default_33 = _shape_param_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_105: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_177, unsqueeze_338);  reshape_default_177 = unsqueeze_338 = None
        mul_tensor_408: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_176, sub_tensor_105)
        sum_dim_int_list_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_408, [0, 2]);  mul_tensor_408 = None
        mul_tensor_409: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_125, 0.0008680555555555555);  sum_dim_int_list_125 = None
        unsqueeze_default_210: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_409, 0);  mul_tensor_409 = None
        unsqueeze_default_211: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_210, 2);  unsqueeze_default_210 = None
        mul_tensor_410: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_126, 0.0008680555555555555)
        mul_tensor_411: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_tensor_412: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_410, mul_tensor_411);  mul_tensor_410 = mul_tensor_411 = None
        unsqueeze_default_212: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_412, 0);  mul_tensor_412 = None
        unsqueeze_default_213: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_212, 2);  unsqueeze_default_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_413: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_81, 0.02946278254943948);  primals_81 = None
        reshape_default_178: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_413, [-1]);  mul_tensor_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_414: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_43, reshape_default_178);  reshape_default_178 = None
        unsqueeze_default_214: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_414, 0);  mul_tensor_414 = None
        unsqueeze_default_215: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_214, 2);  unsqueeze_default_214 = None
        mul_tensor_415: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_105, unsqueeze_default_213);  sub_tensor_105 = unsqueeze_default_213 = None
        sub_tensor_106: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_176, mul_tensor_415);  reshape_default_176 = mul_tensor_415 = None
        sub_tensor_107: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_106, unsqueeze_default_211);  sub_tensor_106 = unsqueeze_default_211 = None
        mul_tensor_416: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_107, unsqueeze_default_215);  sub_tensor_107 = unsqueeze_default_215 = None
        mul_tensor_417: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_126, squeeze_43);  sum_dim_int_list_126 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_179: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_417, _shape_param_143);  mul_tensor_417 = _shape_param_143 = None
        mul_tensor_418: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_179, 0.02946278254943948);  reshape_default_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_180: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_416, _shape_param_144);  mul_tensor_416 = _shape_param_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1158, [0, 2, 3]);  mul_1158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_34: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_277, memory_format = torch.contiguous_format);  getitem_277 = None
        reshape_default_181: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_34, _shape_param_145);  clone_default_34 = _shape_param_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_181, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_35: "f32[768, 128, 3, 3]" = torch.ops.aten.clone.default(primals_77, memory_format = torch.contiguous_format);  primals_77 = None
        reshape_default_182: "f32[1, 768, 1152]" = torch.ops.aten.reshape.default(clone_default_35, _shape_param_146);  clone_default_35 = _shape_param_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_108: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_182, unsqueeze_346);  reshape_default_182 = unsqueeze_346 = None
        mul_tensor_419: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_181, sub_tensor_108)
        sum_dim_int_list_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_419, [0, 2]);  mul_tensor_419 = None
        mul_tensor_420: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_128, 0.0008680555555555555);  sum_dim_int_list_128 = None
        unsqueeze_default_216: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_420, 0);  mul_tensor_420 = None
        unsqueeze_default_217: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_216, 2);  unsqueeze_default_216 = None
        mul_tensor_421: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_129, 0.0008680555555555555)
        mul_tensor_422: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_41, squeeze_41)
        mul_tensor_423: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_421, mul_tensor_422);  mul_tensor_421 = mul_tensor_422 = None
        unsqueeze_default_218: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_423, 0);  mul_tensor_423 = None
        unsqueeze_default_219: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_218, 2);  unsqueeze_default_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_424: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_78, 0.02946278254943948);  primals_78 = None
        reshape_default_183: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_424, [-1]);  mul_tensor_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_425: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_41, reshape_default_183);  reshape_default_183 = None
        unsqueeze_default_220: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_425, 0);  mul_tensor_425 = None
        unsqueeze_default_221: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_220, 2);  unsqueeze_default_220 = None
        mul_tensor_426: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_108, unsqueeze_default_219);  sub_tensor_108 = unsqueeze_default_219 = None
        sub_tensor_109: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_181, mul_tensor_426);  reshape_default_181 = mul_tensor_426 = None
        sub_tensor_110: "f32[1, 768, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_109, unsqueeze_default_217);  sub_tensor_109 = unsqueeze_default_217 = None
        mul_tensor_427: "f32[1, 768, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_110, unsqueeze_default_221);  sub_tensor_110 = unsqueeze_default_221 = None
        mul_tensor_428: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_129, squeeze_41);  sum_dim_int_list_129 = squeeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_184: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_428, _shape_param_147);  mul_tensor_428 = _shape_param_147 = None
        mul_tensor_429: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_184, 0.02946278254943948);  reshape_default_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_185: "f32[768, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_427, _shape_param_148);  mul_tensor_427 = _shape_param_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_1176, [0, 2, 3]);  mul_1176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_186: "f32[1, 768, 512]" = torch.ops.aten.reshape.default(getitem_280, _shape_param_149);  getitem_280 = _shape_param_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(reshape_default_186, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_187: "f32[1, 768, 512]" = torch.ops.aten.reshape.default(primals_74, _shape_param_150);  primals_74 = _shape_param_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_111: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(reshape_default_187, unsqueeze_354);  reshape_default_187 = unsqueeze_354 = None
        mul_tensor_430: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(reshape_default_186, sub_tensor_111)
        sum_dim_int_list_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_430, [0, 2]);  mul_tensor_430 = None
        mul_tensor_431: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_131, 0.001953125);  sum_dim_int_list_131 = None
        unsqueeze_default_222: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_431, 0);  mul_tensor_431 = None
        unsqueeze_default_223: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_222, 2);  unsqueeze_default_222 = None
        mul_tensor_432: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_132, 0.001953125)
        mul_tensor_433: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_39, squeeze_39)
        mul_tensor_434: "f32[768]" = torch.ops.aten.mul.Tensor(mul_tensor_432, mul_tensor_433);  mul_tensor_432 = mul_tensor_433 = None
        unsqueeze_default_224: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_434, 0);  mul_tensor_434 = None
        unsqueeze_default_225: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_224, 2);  unsqueeze_default_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_435: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_75, 0.04419417382415922);  primals_75 = None
        reshape_default_188: "f32[768]" = torch.ops.aten.reshape.default(mul_tensor_435, [-1]);  mul_tensor_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_436: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_39, reshape_default_188);  reshape_default_188 = None
        unsqueeze_default_226: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_tensor_436, 0);  mul_tensor_436 = None
        unsqueeze_default_227: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_226, 2);  unsqueeze_default_226 = None
        mul_tensor_437: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_111, unsqueeze_default_225);  sub_tensor_111 = unsqueeze_default_225 = None
        sub_tensor_112: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(reshape_default_186, mul_tensor_437);  reshape_default_186 = mul_tensor_437 = None
        sub_tensor_113: "f32[1, 768, 512]" = torch.ops.aten.sub.Tensor(sub_tensor_112, unsqueeze_default_223);  sub_tensor_112 = unsqueeze_default_223 = None
        mul_tensor_438: "f32[1, 768, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_113, unsqueeze_default_227);  sub_tensor_113 = unsqueeze_default_227 = None
        mul_tensor_439: "f32[768]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_132, squeeze_39);  sum_dim_int_list_132 = squeeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_189: "f32[768, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_439, _shape_param_151);  mul_tensor_439 = _shape_param_151 = None
        mul_tensor_440: "f32[768, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_189, 0.04419417382415922);  reshape_default_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_190: "f32[768, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_438, _shape_param_152);  mul_tensor_438 = _shape_param_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_133: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_202, [0, 2, 3]);  add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_191: "f32[1, 1536, 512]" = torch.ops.aten.reshape.default(getitem_283, _shape_param_153);  getitem_283 = _shape_param_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_134: "f32[1536]" = torch.ops.aten.sum.dim_IntList(reshape_default_191, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_192: "f32[1, 1536, 512]" = torch.ops.aten.reshape.default(primals_71, _shape_param_154);  primals_71 = _shape_param_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_114: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(reshape_default_192, unsqueeze_362);  reshape_default_192 = unsqueeze_362 = None
        mul_tensor_441: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(reshape_default_191, sub_tensor_114)
        sum_dim_int_list_135: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_441, [0, 2]);  mul_tensor_441 = None
        mul_tensor_442: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_134, 0.001953125);  sum_dim_int_list_134 = None
        unsqueeze_default_228: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_442, 0);  mul_tensor_442 = None
        unsqueeze_default_229: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_228, 2);  unsqueeze_default_228 = None
        mul_tensor_443: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_135, 0.001953125)
        mul_tensor_444: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_tensor_445: "f32[1536]" = torch.ops.aten.mul.Tensor(mul_tensor_443, mul_tensor_444);  mul_tensor_443 = mul_tensor_444 = None
        unsqueeze_default_230: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_445, 0);  mul_tensor_445 = None
        unsqueeze_default_231: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_230, 2);  unsqueeze_default_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_446: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_72, 0.04419417382415922);  primals_72 = None
        reshape_default_193: "f32[1536]" = torch.ops.aten.reshape.default(mul_tensor_446, [-1]);  mul_tensor_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_447: "f32[1536]" = torch.ops.aten.mul.Tensor(squeeze_37, reshape_default_193);  reshape_default_193 = None
        unsqueeze_default_232: "f32[1, 1536]" = torch.ops.aten.unsqueeze.default(mul_tensor_447, 0);  mul_tensor_447 = None
        unsqueeze_default_233: "f32[1, 1536, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_232, 2);  unsqueeze_default_232 = None
        mul_tensor_448: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_114, unsqueeze_default_231);  sub_tensor_114 = unsqueeze_default_231 = None
        sub_tensor_115: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(reshape_default_191, mul_tensor_448);  reshape_default_191 = mul_tensor_448 = None
        sub_tensor_116: "f32[1, 1536, 512]" = torch.ops.aten.sub.Tensor(sub_tensor_115, unsqueeze_default_229);  sub_tensor_115 = unsqueeze_default_229 = None
        mul_tensor_449: "f32[1, 1536, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_116, unsqueeze_default_233);  sub_tensor_116 = unsqueeze_default_233 = None
        mul_tensor_450: "f32[1536]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_135, squeeze_37);  sum_dim_int_list_135 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_194: "f32[1536, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_450, _shape_param_155);  mul_tensor_450 = _shape_param_155 = None
        mul_tensor_451: "f32[1536, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_194, 0.04419417382415922);  reshape_default_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_195: "f32[1536, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_449, _shape_param_156);  mul_tensor_449 = _shape_param_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_tensor_452: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(convolution_21, sigmoid_2);  convolution_21 = sigmoid_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:276 in forward, code: out = self.attn_gain * self.attn_last(out)
        mul_tensor_453: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_tensor_452, 2.0);  mul_tensor_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_454: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1206, mul_tensor_453);  mul_1206 = mul_tensor_453 = None
        sum_default_9: "f32[]" = torch.ops.aten.sum.default(mul_tensor_454);  mul_tensor_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_136: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1213, [0, 2, 3]);  mul_1213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_137: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3]);  where_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_138: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_213, [0, 2, 3]);  add_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_196: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(getitem_292, _shape_param_157);  getitem_292 = _shape_param_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_139: "f32[512]" = torch.ops.aten.sum.dim_IntList(reshape_default_196, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_197: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(primals_63, _shape_param_158);  primals_63 = _shape_param_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_117: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(reshape_default_197, unsqueeze_370);  reshape_default_197 = unsqueeze_370 = None
        mul_tensor_455: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(reshape_default_196, sub_tensor_117)
        sum_dim_int_list_140: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_455, [0, 2]);  mul_tensor_455 = None
        mul_tensor_456: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_139, 0.00390625);  sum_dim_int_list_139 = None
        unsqueeze_default_234: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_456, 0);  mul_tensor_456 = None
        unsqueeze_default_235: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_234, 2);  unsqueeze_default_234 = None
        mul_tensor_457: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_140, 0.00390625)
        mul_tensor_458: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, squeeze_35)
        mul_tensor_459: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_457, mul_tensor_458);  mul_tensor_457 = mul_tensor_458 = None
        unsqueeze_default_236: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_459, 0);  mul_tensor_459 = None
        unsqueeze_default_237: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_236, 2);  unsqueeze_default_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_460: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_64, 0.0625);  primals_64 = None
        reshape_default_198: "f32[512]" = torch.ops.aten.reshape.default(mul_tensor_460, [-1]);  mul_tensor_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_461: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_35, reshape_default_198);  reshape_default_198 = None
        unsqueeze_default_238: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_461, 0);  mul_tensor_461 = None
        unsqueeze_default_239: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_238, 2);  unsqueeze_default_238 = None
        mul_tensor_462: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_117, unsqueeze_default_237);  sub_tensor_117 = unsqueeze_default_237 = None
        sub_tensor_118: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(reshape_default_196, mul_tensor_462);  reshape_default_196 = mul_tensor_462 = None
        sub_tensor_119: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_tensor_118, unsqueeze_default_235);  sub_tensor_118 = unsqueeze_default_235 = None
        mul_tensor_463: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_119, unsqueeze_default_239);  sub_tensor_119 = unsqueeze_default_239 = None
        mul_tensor_464: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_140, squeeze_35);  sum_dim_int_list_140 = squeeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_199: "f32[512, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_464, _shape_param_159);  mul_tensor_464 = _shape_param_159 = None
        mul_tensor_465: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_199, 0.0625);  reshape_default_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_200: "f32[512, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_463, _shape_param_160);  mul_tensor_463 = _shape_param_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_141: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1231, [0, 2, 3]);  mul_1231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_36: "f32[256, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_295, memory_format = torch.contiguous_format);  getitem_295 = None
        reshape_default_201: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(clone_default_36, _shape_param_161);  clone_default_36 = _shape_param_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_142: "f32[256]" = torch.ops.aten.sum.dim_IntList(reshape_default_201, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_37: "f32[256, 128, 3, 3]" = torch.ops.aten.clone.default(primals_60, memory_format = torch.contiguous_format);  primals_60 = None
        reshape_default_202: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(clone_default_37, _shape_param_162);  clone_default_37 = _shape_param_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_120: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_202, unsqueeze_378);  reshape_default_202 = unsqueeze_378 = None
        mul_tensor_466: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_201, sub_tensor_120)
        sum_dim_int_list_143: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_466, [0, 2]);  mul_tensor_466 = None
        mul_tensor_467: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_142, 0.0008680555555555555);  sum_dim_int_list_142 = None
        unsqueeze_default_240: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_467, 0);  mul_tensor_467 = None
        unsqueeze_default_241: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_240, 2);  unsqueeze_default_240 = None
        mul_tensor_468: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_143, 0.0008680555555555555)
        mul_tensor_469: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, squeeze_33)
        mul_tensor_470: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_468, mul_tensor_469);  mul_tensor_468 = mul_tensor_469 = None
        unsqueeze_default_242: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_470, 0);  mul_tensor_470 = None
        unsqueeze_default_243: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_242, 2);  unsqueeze_default_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_471: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_61, 0.02946278254943948);  primals_61 = None
        reshape_default_203: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_471, [-1]);  mul_tensor_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_472: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_33, reshape_default_203);  reshape_default_203 = None
        unsqueeze_default_244: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_472, 0);  mul_tensor_472 = None
        unsqueeze_default_245: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_244, 2);  unsqueeze_default_244 = None
        mul_tensor_473: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_120, unsqueeze_default_243);  sub_tensor_120 = unsqueeze_default_243 = None
        sub_tensor_121: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_201, mul_tensor_473);  reshape_default_201 = mul_tensor_473 = None
        sub_tensor_122: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_121, unsqueeze_default_241);  sub_tensor_121 = unsqueeze_default_241 = None
        mul_tensor_474: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_122, unsqueeze_default_245);  sub_tensor_122 = unsqueeze_default_245 = None
        mul_tensor_475: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_143, squeeze_33);  sum_dim_int_list_143 = squeeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_204: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_475, _shape_param_163);  mul_tensor_475 = _shape_param_163 = None
        mul_tensor_476: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_204, 0.02946278254943948);  reshape_default_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_205: "f32[256, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_474, _shape_param_164);  mul_tensor_474 = _shape_param_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_144: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1249, [0, 2, 3]);  mul_1249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_38: "f32[256, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_298, memory_format = torch.contiguous_format);  getitem_298 = None
        reshape_default_206: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(clone_default_38, _shape_param_165);  clone_default_38 = _shape_param_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(reshape_default_206, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_39: "f32[256, 128, 3, 3]" = torch.ops.aten.clone.default(primals_57, memory_format = torch.contiguous_format);  primals_57 = None
        reshape_default_207: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(clone_default_39, _shape_param_166);  clone_default_39 = _shape_param_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_123: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_207, unsqueeze_386);  reshape_default_207 = unsqueeze_386 = None
        mul_tensor_477: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_206, sub_tensor_123)
        sum_dim_int_list_146: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_477, [0, 2]);  mul_tensor_477 = None
        mul_tensor_478: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_145, 0.0008680555555555555);  sum_dim_int_list_145 = None
        unsqueeze_default_246: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_478, 0);  mul_tensor_478 = None
        unsqueeze_default_247: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_246, 2);  unsqueeze_default_246 = None
        mul_tensor_479: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_146, 0.0008680555555555555)
        mul_tensor_480: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_tensor_481: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_479, mul_tensor_480);  mul_tensor_479 = mul_tensor_480 = None
        unsqueeze_default_248: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_481, 0);  mul_tensor_481 = None
        unsqueeze_default_249: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_248, 2);  unsqueeze_default_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_482: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_58, 0.02946278254943948);  primals_58 = None
        reshape_default_208: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_482, [-1]);  mul_tensor_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_483: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, reshape_default_208);  reshape_default_208 = None
        unsqueeze_default_250: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_483, 0);  mul_tensor_483 = None
        unsqueeze_default_251: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_250, 2);  unsqueeze_default_250 = None
        mul_tensor_484: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_123, unsqueeze_default_249);  sub_tensor_123 = unsqueeze_default_249 = None
        sub_tensor_124: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_206, mul_tensor_484);  reshape_default_206 = mul_tensor_484 = None
        sub_tensor_125: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_124, unsqueeze_default_247);  sub_tensor_124 = unsqueeze_default_247 = None
        mul_tensor_485: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_125, unsqueeze_default_251);  sub_tensor_125 = unsqueeze_default_251 = None
        mul_tensor_486: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_146, squeeze_31);  sum_dim_int_list_146 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_209: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_486, _shape_param_167);  mul_tensor_486 = _shape_param_167 = None
        mul_tensor_487: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_209, 0.02946278254943948);  reshape_default_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_210: "f32[256, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_485, _shape_param_168);  mul_tensor_485 = _shape_param_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_147: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1267, [0, 2, 3]);  mul_1267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_211: "f32[1, 256, 512]" = torch.ops.aten.reshape.default(getitem_301, _shape_param_169);  getitem_301 = _shape_param_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_148: "f32[256]" = torch.ops.aten.sum.dim_IntList(reshape_default_211, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_212: "f32[1, 256, 512]" = torch.ops.aten.reshape.default(primals_54, _shape_param_170);  primals_54 = _shape_param_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_126: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(reshape_default_212, unsqueeze_394);  reshape_default_212 = unsqueeze_394 = None
        mul_tensor_488: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(reshape_default_211, sub_tensor_126)
        sum_dim_int_list_149: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_488, [0, 2]);  mul_tensor_488 = None
        mul_tensor_489: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_148, 0.001953125);  sum_dim_int_list_148 = None
        unsqueeze_default_252: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_489, 0);  mul_tensor_489 = None
        unsqueeze_default_253: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_252, 2);  unsqueeze_default_252 = None
        mul_tensor_490: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_149, 0.001953125)
        mul_tensor_491: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, squeeze_29)
        mul_tensor_492: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_490, mul_tensor_491);  mul_tensor_490 = mul_tensor_491 = None
        unsqueeze_default_254: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_492, 0);  mul_tensor_492 = None
        unsqueeze_default_255: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_254, 2);  unsqueeze_default_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_493: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_55, 0.04419417382415922);  primals_55 = None
        reshape_default_213: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_493, [-1]);  mul_tensor_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_494: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_29, reshape_default_213);  reshape_default_213 = None
        unsqueeze_default_256: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_494, 0);  mul_tensor_494 = None
        unsqueeze_default_257: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_256, 2);  unsqueeze_default_256 = None
        mul_tensor_495: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_126, unsqueeze_default_255);  sub_tensor_126 = unsqueeze_default_255 = None
        sub_tensor_127: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(reshape_default_211, mul_tensor_495);  reshape_default_211 = mul_tensor_495 = None
        sub_tensor_128: "f32[1, 256, 512]" = torch.ops.aten.sub.Tensor(sub_tensor_127, unsqueeze_default_253);  sub_tensor_127 = unsqueeze_default_253 = None
        mul_tensor_496: "f32[1, 256, 512]" = torch.ops.aten.mul.Tensor(sub_tensor_128, unsqueeze_default_257);  sub_tensor_128 = unsqueeze_default_257 = None
        mul_tensor_497: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_149, squeeze_29);  sum_dim_int_list_149 = squeeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_214: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_497, _shape_param_171);  mul_tensor_497 = _shape_param_171 = None
        mul_tensor_498: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_214, 0.04419417382415922);  reshape_default_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_215: "f32[256, 512, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_496, _shape_param_172);  mul_tensor_496 = _shape_param_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_499: "f32[128, 512, 24, 24]" = torch.ops.aten.mul.Tensor(mul_1287, mul_93);  mul_1287 = mul_93 = None
        sum_default_10: "f32[]" = torch.ops.aten.sum.default(mul_tensor_499);  mul_tensor_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_150: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_1294, [0, 2, 3]);  mul_1294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3]);  where_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_152: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 2, 3]);  add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_216: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(getitem_310, _shape_param_173);  getitem_310 = _shape_param_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_153: "f32[512]" = torch.ops.aten.sum.dim_IntList(reshape_default_216, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_217: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(primals_46, _shape_param_174);  primals_46 = _shape_param_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_129: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(reshape_default_217, unsqueeze_402);  reshape_default_217 = unsqueeze_402 = None
        mul_tensor_500: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(reshape_default_216, sub_tensor_129)
        sum_dim_int_list_154: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_500, [0, 2]);  mul_tensor_500 = None
        mul_tensor_501: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_153, 0.00390625);  sum_dim_int_list_153 = None
        unsqueeze_default_258: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_501, 0);  mul_tensor_501 = None
        unsqueeze_default_259: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_258, 2);  unsqueeze_default_258 = None
        mul_tensor_502: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_154, 0.00390625)
        mul_tensor_503: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, squeeze_27)
        mul_tensor_504: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_502, mul_tensor_503);  mul_tensor_502 = mul_tensor_503 = None
        unsqueeze_default_260: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_504, 0);  mul_tensor_504 = None
        unsqueeze_default_261: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_260, 2);  unsqueeze_default_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_505: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_47, 0.0625);  primals_47 = None
        reshape_default_218: "f32[512]" = torch.ops.aten.reshape.default(mul_tensor_505, [-1]);  mul_tensor_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_506: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_27, reshape_default_218);  reshape_default_218 = None
        unsqueeze_default_262: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_506, 0);  mul_tensor_506 = None
        unsqueeze_default_263: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_262, 2);  unsqueeze_default_262 = None
        mul_tensor_507: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_129, unsqueeze_default_261);  sub_tensor_129 = unsqueeze_default_261 = None
        sub_tensor_130: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(reshape_default_216, mul_tensor_507);  reshape_default_216 = mul_tensor_507 = None
        sub_tensor_131: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_tensor_130, unsqueeze_default_259);  sub_tensor_130 = unsqueeze_default_259 = None
        mul_tensor_508: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_131, unsqueeze_default_263);  sub_tensor_131 = unsqueeze_default_263 = None
        mul_tensor_509: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_154, squeeze_27);  sum_dim_int_list_154 = squeeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_219: "f32[512, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_509, _shape_param_175);  mul_tensor_509 = _shape_param_175 = None
        mul_tensor_510: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_219, 0.0625);  reshape_default_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_220: "f32[512, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_508, _shape_param_176);  mul_tensor_508 = _shape_param_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_155: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1312, [0, 2, 3]);  mul_1312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_40: "f32[256, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_313, memory_format = torch.contiguous_format);  getitem_313 = None
        reshape_default_221: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(clone_default_40, _shape_param_177);  clone_default_40 = _shape_param_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_156: "f32[256]" = torch.ops.aten.sum.dim_IntList(reshape_default_221, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_41: "f32[256, 128, 3, 3]" = torch.ops.aten.clone.default(primals_43, memory_format = torch.contiguous_format);  primals_43 = None
        reshape_default_222: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(clone_default_41, _shape_param_178);  clone_default_41 = _shape_param_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_132: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_222, unsqueeze_410);  reshape_default_222 = unsqueeze_410 = None
        mul_tensor_511: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_221, sub_tensor_132)
        sum_dim_int_list_157: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_511, [0, 2]);  mul_tensor_511 = None
        mul_tensor_512: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_156, 0.0008680555555555555);  sum_dim_int_list_156 = None
        unsqueeze_default_264: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_512, 0);  mul_tensor_512 = None
        unsqueeze_default_265: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_264, 2);  unsqueeze_default_264 = None
        mul_tensor_513: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_157, 0.0008680555555555555)
        mul_tensor_514: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_tensor_515: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_513, mul_tensor_514);  mul_tensor_513 = mul_tensor_514 = None
        unsqueeze_default_266: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_515, 0);  mul_tensor_515 = None
        unsqueeze_default_267: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_266, 2);  unsqueeze_default_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_516: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_44, 0.02946278254943948);  primals_44 = None
        reshape_default_223: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_516, [-1]);  mul_tensor_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_517: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, reshape_default_223);  reshape_default_223 = None
        unsqueeze_default_268: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_517, 0);  mul_tensor_517 = None
        unsqueeze_default_269: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_268, 2);  unsqueeze_default_268 = None
        mul_tensor_518: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_132, unsqueeze_default_267);  sub_tensor_132 = unsqueeze_default_267 = None
        sub_tensor_133: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_221, mul_tensor_518);  reshape_default_221 = mul_tensor_518 = None
        sub_tensor_134: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_133, unsqueeze_default_265);  sub_tensor_133 = unsqueeze_default_265 = None
        mul_tensor_519: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_134, unsqueeze_default_269);  sub_tensor_134 = unsqueeze_default_269 = None
        mul_tensor_520: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_157, squeeze_25);  sum_dim_int_list_157 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_224: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_520, _shape_param_179);  mul_tensor_520 = _shape_param_179 = None
        mul_tensor_521: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_224, 0.02946278254943948);  reshape_default_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_225: "f32[256, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_519, _shape_param_180);  mul_tensor_519 = _shape_param_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1330, [0, 2, 3]);  mul_1330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_42: "f32[256, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_316, memory_format = torch.contiguous_format);  getitem_316 = None
        reshape_default_226: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(clone_default_42, _shape_param_181);  clone_default_42 = _shape_param_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(reshape_default_226, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_43: "f32[256, 128, 3, 3]" = torch.ops.aten.clone.default(primals_40, memory_format = torch.contiguous_format);  primals_40 = None
        reshape_default_227: "f32[1, 256, 1152]" = torch.ops.aten.reshape.default(clone_default_43, _shape_param_182);  clone_default_43 = _shape_param_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_135: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_227, unsqueeze_418);  reshape_default_227 = unsqueeze_418 = None
        mul_tensor_522: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_226, sub_tensor_135)
        sum_dim_int_list_160: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_522, [0, 2]);  mul_tensor_522 = None
        mul_tensor_523: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_159, 0.0008680555555555555);  sum_dim_int_list_159 = None
        unsqueeze_default_270: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_523, 0);  mul_tensor_523 = None
        unsqueeze_default_271: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_270, 2);  unsqueeze_default_270 = None
        mul_tensor_524: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_160, 0.0008680555555555555)
        mul_tensor_525: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, squeeze_23)
        mul_tensor_526: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_524, mul_tensor_525);  mul_tensor_524 = mul_tensor_525 = None
        unsqueeze_default_272: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_526, 0);  mul_tensor_526 = None
        unsqueeze_default_273: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_272, 2);  unsqueeze_default_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_527: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_41, 0.02946278254943948);  primals_41 = None
        reshape_default_228: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_527, [-1]);  mul_tensor_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_528: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_23, reshape_default_228);  reshape_default_228 = None
        unsqueeze_default_274: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_528, 0);  mul_tensor_528 = None
        unsqueeze_default_275: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_274, 2);  unsqueeze_default_274 = None
        mul_tensor_529: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_135, unsqueeze_default_273);  sub_tensor_135 = unsqueeze_default_273 = None
        sub_tensor_136: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_226, mul_tensor_529);  reshape_default_226 = mul_tensor_529 = None
        sub_tensor_137: "f32[1, 256, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_136, unsqueeze_default_271);  sub_tensor_136 = unsqueeze_default_271 = None
        mul_tensor_530: "f32[1, 256, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_137, unsqueeze_default_275);  sub_tensor_137 = unsqueeze_default_275 = None
        mul_tensor_531: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_160, squeeze_23);  sum_dim_int_list_160 = squeeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_229: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_531, _shape_param_183);  mul_tensor_531 = _shape_param_183 = None
        mul_tensor_532: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_229, 0.02946278254943948);  reshape_default_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_230: "f32[256, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_530, _shape_param_184);  mul_tensor_530 = _shape_param_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_161: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1348, [0, 2, 3]);  mul_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_231: "f32[1, 256, 256]" = torch.ops.aten.reshape.default(getitem_319, _shape_param_185);  getitem_319 = _shape_param_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_162: "f32[256]" = torch.ops.aten.sum.dim_IntList(reshape_default_231, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_232: "f32[1, 256, 256]" = torch.ops.aten.reshape.default(primals_37, _shape_param_186);  primals_37 = _shape_param_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_138: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(reshape_default_232, unsqueeze_426);  reshape_default_232 = unsqueeze_426 = None
        mul_tensor_533: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(reshape_default_231, sub_tensor_138)
        sum_dim_int_list_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_533, [0, 2]);  mul_tensor_533 = None
        mul_tensor_534: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_162, 0.00390625);  sum_dim_int_list_162 = None
        unsqueeze_default_276: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_534, 0);  mul_tensor_534 = None
        unsqueeze_default_277: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_276, 2);  unsqueeze_default_276 = None
        mul_tensor_535: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_163, 0.00390625)
        mul_tensor_536: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, squeeze_21)
        mul_tensor_537: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_535, mul_tensor_536);  mul_tensor_535 = mul_tensor_536 = None
        unsqueeze_default_278: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_537, 0);  mul_tensor_537 = None
        unsqueeze_default_279: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_278, 2);  unsqueeze_default_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_538: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_38, 0.0625);  primals_38 = None
        reshape_default_233: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_538, [-1]);  mul_tensor_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_539: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_21, reshape_default_233);  reshape_default_233 = None
        unsqueeze_default_280: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_539, 0);  mul_tensor_539 = None
        unsqueeze_default_281: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_280, 2);  unsqueeze_default_280 = None
        mul_tensor_540: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_138, unsqueeze_default_279);  sub_tensor_138 = unsqueeze_default_279 = None
        sub_tensor_139: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(reshape_default_231, mul_tensor_540);  reshape_default_231 = mul_tensor_540 = None
        sub_tensor_140: "f32[1, 256, 256]" = torch.ops.aten.sub.Tensor(sub_tensor_139, unsqueeze_default_277);  sub_tensor_139 = unsqueeze_default_277 = None
        mul_tensor_541: "f32[1, 256, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_140, unsqueeze_default_281);  sub_tensor_140 = unsqueeze_default_281 = None
        mul_tensor_542: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_163, squeeze_21);  sum_dim_int_list_163 = squeeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_234: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_542, _shape_param_187);  mul_tensor_542 = _shape_param_187 = None
        mul_tensor_543: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_234, 0.0625);  reshape_default_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_235: "f32[256, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_541, _shape_param_188);  mul_tensor_541 = _shape_param_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_164: "f32[512]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 2, 3]);  add_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_236: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(getitem_322, _shape_param_189);  getitem_322 = _shape_param_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_165: "f32[512]" = torch.ops.aten.sum.dim_IntList(reshape_default_236, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_237: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(primals_34, _shape_param_190);  primals_34 = _shape_param_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_141: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(reshape_default_237, unsqueeze_434);  reshape_default_237 = unsqueeze_434 = None
        mul_tensor_544: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(reshape_default_236, sub_tensor_141)
        sum_dim_int_list_166: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_544, [0, 2]);  mul_tensor_544 = None
        mul_tensor_545: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_165, 0.00390625);  sum_dim_int_list_165 = None
        unsqueeze_default_282: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_545, 0);  mul_tensor_545 = None
        unsqueeze_default_283: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_282, 2);  unsqueeze_default_282 = None
        mul_tensor_546: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_166, 0.00390625)
        mul_tensor_547: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_tensor_548: "f32[512]" = torch.ops.aten.mul.Tensor(mul_tensor_546, mul_tensor_547);  mul_tensor_546 = mul_tensor_547 = None
        unsqueeze_default_284: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_548, 0);  mul_tensor_548 = None
        unsqueeze_default_285: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_284, 2);  unsqueeze_default_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_549: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_35, 0.0625);  primals_35 = None
        reshape_default_238: "f32[512]" = torch.ops.aten.reshape.default(mul_tensor_549, [-1]);  mul_tensor_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_550: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_19, reshape_default_238);  reshape_default_238 = None
        unsqueeze_default_286: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_tensor_550, 0);  mul_tensor_550 = None
        unsqueeze_default_287: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_286, 2);  unsqueeze_default_286 = None
        mul_tensor_551: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_141, unsqueeze_default_285);  sub_tensor_141 = unsqueeze_default_285 = None
        sub_tensor_142: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(reshape_default_236, mul_tensor_551);  reshape_default_236 = mul_tensor_551 = None
        sub_tensor_143: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_tensor_142, unsqueeze_default_283);  sub_tensor_142 = unsqueeze_default_283 = None
        mul_tensor_552: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(sub_tensor_143, unsqueeze_default_287);  sub_tensor_143 = unsqueeze_default_287 = None
        mul_tensor_553: "f32[512]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_166, squeeze_19);  sum_dim_int_list_166 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_239: "f32[512, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_553, _shape_param_191);  mul_tensor_553 = _shape_param_191 = None
        mul_tensor_554: "f32[512, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_239, 0.0625);  reshape_default_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_240: "f32[512, 256, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_552, _shape_param_192);  mul_tensor_552 = _shape_param_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/nfnet.py:280 in forward, code: out.mul_(self.skipinit_gain)
        mul_tensor_555: "f32[128, 256, 48, 48]" = torch.ops.aten.mul.Tensor(mul_1378, mul_57);  mul_1378 = mul_57 = None
        sum_default_11: "f32[]" = torch.ops.aten.sum.default(mul_tensor_555);  mul_tensor_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        sum_dim_int_list_167: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1385, [0, 2, 3]);  mul_1385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        sum_dim_int_list_168: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3]);  where_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_169: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 2, 3]);  add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_241: "f32[1, 256, 128]" = torch.ops.aten.reshape.default(getitem_331, _shape_param_193);  getitem_331 = _shape_param_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_170: "f32[256]" = torch.ops.aten.sum.dim_IntList(reshape_default_241, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_242: "f32[1, 256, 128]" = torch.ops.aten.reshape.default(primals_26, _shape_param_194);  primals_26 = _shape_param_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_144: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(reshape_default_242, unsqueeze_442);  reshape_default_242 = unsqueeze_442 = None
        mul_tensor_556: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(reshape_default_241, sub_tensor_144)
        sum_dim_int_list_171: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_556, [0, 2]);  mul_tensor_556 = None
        mul_tensor_557: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_170, 0.0078125);  sum_dim_int_list_170 = None
        unsqueeze_default_288: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_557, 0);  mul_tensor_557 = None
        unsqueeze_default_289: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_288, 2);  unsqueeze_default_288 = None
        mul_tensor_558: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_171, 0.0078125)
        mul_tensor_559: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, squeeze_17)
        mul_tensor_560: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_558, mul_tensor_559);  mul_tensor_558 = mul_tensor_559 = None
        unsqueeze_default_290: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_560, 0);  mul_tensor_560 = None
        unsqueeze_default_291: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_290, 2);  unsqueeze_default_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_561: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_27, 0.08838834764831845);  primals_27 = None
        reshape_default_243: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_561, [-1]);  mul_tensor_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_562: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_17, reshape_default_243);  reshape_default_243 = None
        unsqueeze_default_292: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_562, 0);  mul_tensor_562 = None
        unsqueeze_default_293: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_292, 2);  unsqueeze_default_292 = None
        mul_tensor_563: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_144, unsqueeze_default_291);  sub_tensor_144 = unsqueeze_default_291 = None
        sub_tensor_145: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(reshape_default_241, mul_tensor_563);  reshape_default_241 = mul_tensor_563 = None
        sub_tensor_146: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_145, unsqueeze_default_289);  sub_tensor_145 = unsqueeze_default_289 = None
        mul_tensor_564: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_146, unsqueeze_default_293);  sub_tensor_146 = unsqueeze_default_293 = None
        mul_tensor_565: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_171, squeeze_17);  sum_dim_int_list_171 = squeeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_244: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_565, _shape_param_195);  mul_tensor_565 = _shape_param_195 = None
        mul_tensor_566: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_244, 0.08838834764831845);  reshape_default_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_245: "f32[256, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_564, _shape_param_196);  mul_tensor_564 = _shape_param_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_172: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1403, [0, 2, 3]);  mul_1403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_44: "f32[128, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_334, memory_format = torch.contiguous_format);  getitem_334 = None
        reshape_default_246: "f32[1, 128, 1152]" = torch.ops.aten.reshape.default(clone_default_44, _shape_param_197);  clone_default_44 = _shape_param_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_173: "f32[128]" = torch.ops.aten.sum.dim_IntList(reshape_default_246, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_45: "f32[128, 128, 3, 3]" = torch.ops.aten.clone.default(primals_23, memory_format = torch.contiguous_format);  primals_23 = None
        reshape_default_247: "f32[1, 128, 1152]" = torch.ops.aten.reshape.default(clone_default_45, _shape_param_198);  clone_default_45 = _shape_param_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_147: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_247, unsqueeze_450);  reshape_default_247 = unsqueeze_450 = None
        mul_tensor_567: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_246, sub_tensor_147)
        sum_dim_int_list_174: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_567, [0, 2]);  mul_tensor_567 = None
        mul_tensor_568: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_173, 0.0008680555555555555);  sum_dim_int_list_173 = None
        unsqueeze_default_294: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_568, 0);  mul_tensor_568 = None
        unsqueeze_default_295: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_294, 2);  unsqueeze_default_294 = None
        mul_tensor_569: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_174, 0.0008680555555555555)
        mul_tensor_570: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, squeeze_15)
        mul_tensor_571: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_569, mul_tensor_570);  mul_tensor_569 = mul_tensor_570 = None
        unsqueeze_default_296: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_571, 0);  mul_tensor_571 = None
        unsqueeze_default_297: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_296, 2);  unsqueeze_default_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_572: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_24, 0.02946278254943948);  primals_24 = None
        reshape_default_248: "f32[128]" = torch.ops.aten.reshape.default(mul_tensor_572, [-1]);  mul_tensor_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_573: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_15, reshape_default_248);  reshape_default_248 = None
        unsqueeze_default_298: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_573, 0);  mul_tensor_573 = None
        unsqueeze_default_299: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_298, 2);  unsqueeze_default_298 = None
        mul_tensor_574: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_147, unsqueeze_default_297);  sub_tensor_147 = unsqueeze_default_297 = None
        sub_tensor_148: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_246, mul_tensor_574);  reshape_default_246 = mul_tensor_574 = None
        sub_tensor_149: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_148, unsqueeze_default_295);  sub_tensor_148 = unsqueeze_default_295 = None
        mul_tensor_575: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_149, unsqueeze_default_299);  sub_tensor_149 = unsqueeze_default_299 = None
        mul_tensor_576: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_174, squeeze_15);  sum_dim_int_list_174 = squeeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_249: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_576, _shape_param_199);  mul_tensor_576 = _shape_param_199 = None
        mul_tensor_577: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_249, 0.02946278254943948);  reshape_default_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_250: "f32[128, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_575, _shape_param_200);  mul_tensor_575 = _shape_param_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_175: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1421, [0, 2, 3]);  mul_1421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_46: "f32[128, 128, 3, 3]" = torch.ops.aten.clone.default(getitem_337, memory_format = torch.contiguous_format);  getitem_337 = None
        reshape_default_251: "f32[1, 128, 1152]" = torch.ops.aten.reshape.default(clone_default_46, _shape_param_201);  clone_default_46 = _shape_param_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_176: "f32[128]" = torch.ops.aten.sum.dim_IntList(reshape_default_251, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_47: "f32[128, 128, 3, 3]" = torch.ops.aten.clone.default(primals_20, memory_format = torch.contiguous_format);  primals_20 = None
        reshape_default_252: "f32[1, 128, 1152]" = torch.ops.aten.reshape.default(clone_default_47, _shape_param_202);  clone_default_47 = _shape_param_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_150: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_252, unsqueeze_458);  reshape_default_252 = unsqueeze_458 = None
        mul_tensor_578: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(reshape_default_251, sub_tensor_150)
        sum_dim_int_list_177: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_578, [0, 2]);  mul_tensor_578 = None
        mul_tensor_579: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_176, 0.0008680555555555555);  sum_dim_int_list_176 = None
        unsqueeze_default_300: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_579, 0);  mul_tensor_579 = None
        unsqueeze_default_301: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_300, 2);  unsqueeze_default_300 = None
        mul_tensor_580: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_177, 0.0008680555555555555)
        mul_tensor_581: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_tensor_582: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_580, mul_tensor_581);  mul_tensor_580 = mul_tensor_581 = None
        unsqueeze_default_302: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_582, 0);  mul_tensor_582 = None
        unsqueeze_default_303: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_302, 2);  unsqueeze_default_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_583: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_21, 0.02946278254943948);  primals_21 = None
        reshape_default_253: "f32[128]" = torch.ops.aten.reshape.default(mul_tensor_583, [-1]);  mul_tensor_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_584: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, reshape_default_253);  reshape_default_253 = None
        unsqueeze_default_304: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_584, 0);  mul_tensor_584 = None
        unsqueeze_default_305: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_304, 2);  unsqueeze_default_304 = None
        mul_tensor_585: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_150, unsqueeze_default_303);  sub_tensor_150 = unsqueeze_default_303 = None
        sub_tensor_151: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(reshape_default_251, mul_tensor_585);  reshape_default_251 = mul_tensor_585 = None
        sub_tensor_152: "f32[1, 128, 1152]" = torch.ops.aten.sub.Tensor(sub_tensor_151, unsqueeze_default_301);  sub_tensor_151 = unsqueeze_default_301 = None
        mul_tensor_586: "f32[1, 128, 1152]" = torch.ops.aten.mul.Tensor(sub_tensor_152, unsqueeze_default_305);  sub_tensor_152 = unsqueeze_default_305 = None
        mul_tensor_587: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_177, squeeze_13);  sum_dim_int_list_177 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_254: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_587, _shape_param_203);  mul_tensor_587 = _shape_param_203 = None
        mul_tensor_588: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_254, 0.02946278254943948);  reshape_default_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_255: "f32[128, 128, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_586, _shape_param_204);  mul_tensor_586 = _shape_param_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_178: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1439, [0, 2, 3]);  mul_1439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_256: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(getitem_340, _shape_param_205);  getitem_340 = _shape_param_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_179: "f32[128]" = torch.ops.aten.sum.dim_IntList(reshape_default_256, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_257: "f32[1, 128, 128]" = torch.ops.aten.reshape.default(primals_17, _shape_param_206);  primals_17 = _shape_param_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_153: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(reshape_default_257, unsqueeze_466);  reshape_default_257 = unsqueeze_466 = None
        mul_tensor_589: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(reshape_default_256, sub_tensor_153)
        sum_dim_int_list_180: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_589, [0, 2]);  mul_tensor_589 = None
        mul_tensor_590: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_179, 0.0078125);  sum_dim_int_list_179 = None
        unsqueeze_default_306: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_590, 0);  mul_tensor_590 = None
        unsqueeze_default_307: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_306, 2);  unsqueeze_default_306 = None
        mul_tensor_591: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_180, 0.0078125)
        mul_tensor_592: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, squeeze_11)
        mul_tensor_593: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_591, mul_tensor_592);  mul_tensor_591 = mul_tensor_592 = None
        unsqueeze_default_308: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_593, 0);  mul_tensor_593 = None
        unsqueeze_default_309: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_308, 2);  unsqueeze_default_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_594: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_18, 0.08838834764831845);  primals_18 = None
        reshape_default_258: "f32[128]" = torch.ops.aten.reshape.default(mul_tensor_594, [-1]);  mul_tensor_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_595: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_11, reshape_default_258);  reshape_default_258 = None
        unsqueeze_default_310: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_595, 0);  mul_tensor_595 = None
        unsqueeze_default_311: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_310, 2);  unsqueeze_default_310 = None
        mul_tensor_596: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_153, unsqueeze_default_309);  sub_tensor_153 = unsqueeze_default_309 = None
        sub_tensor_154: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(reshape_default_256, mul_tensor_596);  reshape_default_256 = mul_tensor_596 = None
        sub_tensor_155: "f32[1, 128, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_154, unsqueeze_default_307);  sub_tensor_154 = unsqueeze_default_307 = None
        mul_tensor_597: "f32[1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_155, unsqueeze_default_311);  sub_tensor_155 = unsqueeze_default_311 = None
        mul_tensor_598: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_180, squeeze_11);  sum_dim_int_list_180 = squeeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_259: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_598, _shape_param_207);  mul_tensor_598 = _shape_param_207 = None
        mul_tensor_599: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_259, 0.08838834764831845);  reshape_default_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_260: "f32[128, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_597, _shape_param_208);  mul_tensor_597 = _shape_param_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1377, [0, 2, 3]);  mul_1377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        reshape_default_261: "f32[1, 256, 128]" = torch.ops.aten.reshape.default(getitem_343, _shape_param_209);  getitem_343 = _shape_param_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(reshape_default_261, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_262: "f32[1, 256, 128]" = torch.ops.aten.reshape.default(primals_14, _shape_param_210);  primals_14 = _shape_param_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_156: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(reshape_default_262, unsqueeze_474);  reshape_default_262 = unsqueeze_474 = None
        mul_tensor_600: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(reshape_default_261, sub_tensor_156)
        sum_dim_int_list_183: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_600, [0, 2]);  mul_tensor_600 = None
        mul_tensor_601: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_182, 0.0078125);  sum_dim_int_list_182 = None
        unsqueeze_default_312: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_601, 0);  mul_tensor_601 = None
        unsqueeze_default_313: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_312, 2);  unsqueeze_default_312 = None
        mul_tensor_602: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_183, 0.0078125)
        mul_tensor_603: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, squeeze_9)
        mul_tensor_604: "f32[256]" = torch.ops.aten.mul.Tensor(mul_tensor_602, mul_tensor_603);  mul_tensor_602 = mul_tensor_603 = None
        unsqueeze_default_314: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_604, 0);  mul_tensor_604 = None
        unsqueeze_default_315: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_314, 2);  unsqueeze_default_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_605: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_15, 0.08838834764831845);  primals_15 = None
        reshape_default_263: "f32[256]" = torch.ops.aten.reshape.default(mul_tensor_605, [-1]);  mul_tensor_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_606: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_9, reshape_default_263);  reshape_default_263 = None
        unsqueeze_default_316: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_tensor_606, 0);  mul_tensor_606 = None
        unsqueeze_default_317: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_316, 2);  unsqueeze_default_316 = None
        mul_tensor_607: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_156, unsqueeze_default_315);  sub_tensor_156 = unsqueeze_default_315 = None
        sub_tensor_157: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(reshape_default_261, mul_tensor_607);  reshape_default_261 = mul_tensor_607 = None
        sub_tensor_158: "f32[1, 256, 128]" = torch.ops.aten.sub.Tensor(sub_tensor_157, unsqueeze_default_313);  sub_tensor_157 = unsqueeze_default_313 = None
        mul_tensor_608: "f32[1, 256, 128]" = torch.ops.aten.mul.Tensor(sub_tensor_158, unsqueeze_default_317);  sub_tensor_158 = unsqueeze_default_317 = None
        mul_tensor_609: "f32[256]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_183, squeeze_9);  sum_dim_int_list_183 = squeeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_264: "f32[256, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_609, _shape_param_211);  mul_tensor_609 = _shape_param_211 = None
        mul_tensor_610: "f32[256, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_264, 0.08838834764831845);  reshape_default_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_265: "f32[256, 128, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_608, _shape_param_212);  mul_tensor_608 = _shape_param_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_184: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1468, [0, 2, 3]);  mul_1468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_48: "f32[128, 64, 3, 3]" = torch.ops.aten.clone.default(getitem_346, memory_format = torch.contiguous_format);  getitem_346 = None
        reshape_default_266: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(clone_default_48, _shape_param_213);  clone_default_48 = _shape_param_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_185: "f32[128]" = torch.ops.aten.sum.dim_IntList(reshape_default_266, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_49: "f32[128, 64, 3, 3]" = torch.ops.aten.clone.default(primals_11, memory_format = torch.contiguous_format);  primals_11 = None
        reshape_default_267: "f32[1, 128, 576]" = torch.ops.aten.reshape.default(clone_default_49, _shape_param_214);  clone_default_49 = _shape_param_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_159: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(reshape_default_267, unsqueeze_482);  reshape_default_267 = unsqueeze_482 = None
        mul_tensor_611: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(reshape_default_266, sub_tensor_159)
        sum_dim_int_list_186: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_611, [0, 2]);  mul_tensor_611 = None
        mul_tensor_612: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_185, 0.001736111111111111);  sum_dim_int_list_185 = None
        unsqueeze_default_318: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_612, 0);  mul_tensor_612 = None
        unsqueeze_default_319: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_318, 2);  unsqueeze_default_318 = None
        mul_tensor_613: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_186, 0.001736111111111111)
        mul_tensor_614: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_tensor_615: "f32[128]" = torch.ops.aten.mul.Tensor(mul_tensor_613, mul_tensor_614);  mul_tensor_613 = mul_tensor_614 = None
        unsqueeze_default_320: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_615, 0);  mul_tensor_615 = None
        unsqueeze_default_321: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_320, 2);  unsqueeze_default_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_616: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_12, 0.041666666666666664);  primals_12 = None
        reshape_default_268: "f32[128]" = torch.ops.aten.reshape.default(mul_tensor_616, [-1]);  mul_tensor_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_617: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, reshape_default_268);  reshape_default_268 = None
        unsqueeze_default_322: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_tensor_617, 0);  mul_tensor_617 = None
        unsqueeze_default_323: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_322, 2);  unsqueeze_default_322 = None
        mul_tensor_618: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_tensor_159, unsqueeze_default_321);  sub_tensor_159 = unsqueeze_default_321 = None
        sub_tensor_160: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(reshape_default_266, mul_tensor_618);  reshape_default_266 = mul_tensor_618 = None
        sub_tensor_161: "f32[1, 128, 576]" = torch.ops.aten.sub.Tensor(sub_tensor_160, unsqueeze_default_319);  sub_tensor_160 = unsqueeze_default_319 = None
        mul_tensor_619: "f32[1, 128, 576]" = torch.ops.aten.mul.Tensor(sub_tensor_161, unsqueeze_default_323);  sub_tensor_161 = unsqueeze_default_323 = None
        mul_tensor_620: "f32[128]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_186, squeeze_7);  sum_dim_int_list_186 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_269: "f32[128, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_620, _shape_param_215);  mul_tensor_620 = _shape_param_215 = None
        mul_tensor_621: "f32[128, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_269, 0.041666666666666664);  reshape_default_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_270: "f32[128, 64, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_619, _shape_param_216);  mul_tensor_619 = _shape_param_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_187: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1486, [0, 2, 3]);  mul_1486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_50: "f32[64, 32, 3, 3]" = torch.ops.aten.clone.default(getitem_349, memory_format = torch.contiguous_format);  getitem_349 = None
        reshape_default_271: "f32[1, 64, 288]" = torch.ops.aten.reshape.default(clone_default_50, _shape_param_217);  clone_default_50 = _shape_param_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_188: "f32[64]" = torch.ops.aten.sum.dim_IntList(reshape_default_271, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_51: "f32[64, 32, 3, 3]" = torch.ops.aten.clone.default(primals_8, memory_format = torch.contiguous_format);  primals_8 = None
        reshape_default_272: "f32[1, 64, 288]" = torch.ops.aten.reshape.default(clone_default_51, _shape_param_218);  clone_default_51 = _shape_param_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_162: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(reshape_default_272, unsqueeze_490);  reshape_default_272 = unsqueeze_490 = None
        mul_tensor_622: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(reshape_default_271, sub_tensor_162)
        sum_dim_int_list_189: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_tensor_622, [0, 2]);  mul_tensor_622 = None
        mul_tensor_623: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_188, 0.003472222222222222);  sum_dim_int_list_188 = None
        unsqueeze_default_324: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_623, 0);  mul_tensor_623 = None
        unsqueeze_default_325: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_324, 2);  unsqueeze_default_324 = None
        mul_tensor_624: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_189, 0.003472222222222222)
        mul_tensor_625: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, squeeze_5)
        mul_tensor_626: "f32[64]" = torch.ops.aten.mul.Tensor(mul_tensor_624, mul_tensor_625);  mul_tensor_624 = mul_tensor_625 = None
        unsqueeze_default_326: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_626, 0);  mul_tensor_626 = None
        unsqueeze_default_327: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_326, 2);  unsqueeze_default_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_627: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_9, 0.05892556509887896);  primals_9 = None
        reshape_default_273: "f32[64]" = torch.ops.aten.reshape.default(mul_tensor_627, [-1]);  mul_tensor_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_628: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_5, reshape_default_273);  reshape_default_273 = None
        unsqueeze_default_328: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_tensor_628, 0);  mul_tensor_628 = None
        unsqueeze_default_329: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_328, 2);  unsqueeze_default_328 = None
        mul_tensor_629: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_tensor_162, unsqueeze_default_327);  sub_tensor_162 = unsqueeze_default_327 = None
        sub_tensor_163: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(reshape_default_271, mul_tensor_629);  reshape_default_271 = mul_tensor_629 = None
        sub_tensor_164: "f32[1, 64, 288]" = torch.ops.aten.sub.Tensor(sub_tensor_163, unsqueeze_default_325);  sub_tensor_163 = unsqueeze_default_325 = None
        mul_tensor_630: "f32[1, 64, 288]" = torch.ops.aten.mul.Tensor(sub_tensor_164, unsqueeze_default_329);  sub_tensor_164 = unsqueeze_default_329 = None
        mul_tensor_631: "f32[64]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_189, squeeze_5);  sum_dim_int_list_189 = squeeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_274: "f32[64, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_631, _shape_param_219);  mul_tensor_631 = _shape_param_219 = None
        mul_tensor_632: "f32[64, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_274, 0.05892556509887896);  reshape_default_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_275: "f32[64, 32, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_630, _shape_param_220);  mul_tensor_630 = _shape_param_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_190: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1504, [0, 2, 3]);  mul_1504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_52: "f32[32, 16, 3, 3]" = torch.ops.aten.clone.default(getitem_352, memory_format = torch.contiguous_format);  getitem_352 = None
        reshape_default_276: "f32[1, 32, 144]" = torch.ops.aten.reshape.default(clone_default_52, _shape_param_221);  clone_default_52 = _shape_param_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_191: "f32[32]" = torch.ops.aten.sum.dim_IntList(reshape_default_276, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_53: "f32[32, 16, 3, 3]" = torch.ops.aten.clone.default(primals_5, memory_format = torch.contiguous_format);  primals_5 = None
        reshape_default_277: "f32[1, 32, 144]" = torch.ops.aten.reshape.default(clone_default_53, _shape_param_222);  clone_default_53 = _shape_param_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_165: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(reshape_default_277, unsqueeze_498);  reshape_default_277 = unsqueeze_498 = None
        mul_tensor_633: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(reshape_default_276, sub_tensor_165)
        sum_dim_int_list_192: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_tensor_633, [0, 2]);  mul_tensor_633 = None
        mul_tensor_634: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_191, 0.006944444444444444);  sum_dim_int_list_191 = None
        unsqueeze_default_330: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_634, 0);  mul_tensor_634 = None
        unsqueeze_default_331: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_330, 2);  unsqueeze_default_330 = None
        mul_tensor_635: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_192, 0.006944444444444444)
        mul_tensor_636: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, squeeze_3)
        mul_tensor_637: "f32[32]" = torch.ops.aten.mul.Tensor(mul_tensor_635, mul_tensor_636);  mul_tensor_635 = mul_tensor_636 = None
        unsqueeze_default_332: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_637, 0);  mul_tensor_637 = None
        unsqueeze_default_333: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_332, 2);  unsqueeze_default_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_638: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_6, 0.08333333333333333);  primals_6 = None
        reshape_default_278: "f32[32]" = torch.ops.aten.reshape.default(mul_tensor_638, [-1]);  mul_tensor_638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_639: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_3, reshape_default_278);  reshape_default_278 = None
        unsqueeze_default_334: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_tensor_639, 0);  mul_tensor_639 = None
        unsqueeze_default_335: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_334, 2);  unsqueeze_default_334 = None
        mul_tensor_640: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_tensor_165, unsqueeze_default_333);  sub_tensor_165 = unsqueeze_default_333 = None
        sub_tensor_166: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(reshape_default_276, mul_tensor_640);  reshape_default_276 = mul_tensor_640 = None
        sub_tensor_167: "f32[1, 32, 144]" = torch.ops.aten.sub.Tensor(sub_tensor_166, unsqueeze_default_331);  sub_tensor_166 = unsqueeze_default_331 = None
        mul_tensor_641: "f32[1, 32, 144]" = torch.ops.aten.mul.Tensor(sub_tensor_167, unsqueeze_default_335);  sub_tensor_167 = unsqueeze_default_335 = None
        mul_tensor_642: "f32[32]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_192, squeeze_3);  sum_dim_int_list_192 = squeeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_279: "f32[32, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_642, _shape_param_223);  mul_tensor_642 = _shape_param_223 = None
        mul_tensor_643: "f32[32, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_279, 0.08333333333333333);  reshape_default_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_280: "f32[32, 16, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_641, _shape_param_224);  mul_tensor_641 = _shape_param_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:232 in forward, code: return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        sum_dim_int_list_193: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1522, [0, 2, 3]);  mul_1522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:231 in forward, code: ).reshape_as(self.weight)
        clone_default_54: "f32[16, 3, 3, 3]" = torch.ops.aten.clone.default(getitem_355, memory_format = torch.contiguous_format);  getitem_355 = None
        reshape_default_281: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(clone_default_54, _shape_param_225);  clone_default_54 = _shape_param_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sum_dim_int_list_194: "f32[16]" = torch.ops.aten.sum.dim_IntList(reshape_default_281, [0, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        clone_default_55: "f32[16, 3, 3, 3]" = torch.ops.aten.clone.default(primals_2, memory_format = torch.contiguous_format);  primals_2 = None
        reshape_default_282: "f32[1, 16, 27]" = torch.ops.aten.reshape.default(clone_default_55, _shape_param_226);  clone_default_55 = _shape_param_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        sub_tensor_168: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(reshape_default_282, unsqueeze_506);  reshape_default_282 = unsqueeze_506 = None
        mul_tensor_644: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(reshape_default_281, sub_tensor_168)
        sum_dim_int_list_195: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_tensor_644, [0, 2]);  mul_tensor_644 = None
        mul_tensor_645: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_194, 0.037037037037037035);  sum_dim_int_list_194 = None
        unsqueeze_default_336: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_645, 0);  mul_tensor_645 = None
        unsqueeze_default_337: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_336, 2);  unsqueeze_default_336 = None
        mul_tensor_646: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_195, 0.037037037037037035)
        mul_tensor_647: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_tensor_648: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_646, mul_tensor_647);  mul_tensor_646 = mul_tensor_647 = None
        unsqueeze_default_338: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_648, 0);  mul_tensor_648 = None
        unsqueeze_default_339: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_338, 2);  unsqueeze_default_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        mul_tensor_649: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(primals_3, 0.19245008972987526);  primals_3 = None
        reshape_default_283: "f32[16]" = torch.ops.aten.reshape.default(mul_tensor_649, [-1]);  mul_tensor_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:223 in forward, code: weight = F.batch_norm(
        mul_tensor_650: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, reshape_default_283);  reshape_default_283 = None
        unsqueeze_default_340: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_tensor_650, 0);  mul_tensor_650 = None
        unsqueeze_default_341: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_340, 2);  unsqueeze_default_340 = None
        mul_tensor_651: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor_168, unsqueeze_default_339);  sub_tensor_168 = unsqueeze_default_339 = None
        sub_tensor_169: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(reshape_default_281, mul_tensor_651);  reshape_default_281 = mul_tensor_651 = None
        sub_tensor_170: "f32[1, 16, 27]" = torch.ops.aten.sub.Tensor(sub_tensor_169, unsqueeze_default_337);  sub_tensor_169 = unsqueeze_default_337 = None
        mul_tensor_652: "f32[1, 16, 27]" = torch.ops.aten.mul.Tensor(sub_tensor_170, unsqueeze_default_341);  sub_tensor_170 = unsqueeze_default_341 = None
        mul_tensor_653: "f32[16]" = torch.ops.aten.mul.Tensor(sum_dim_int_list_195, squeeze_1);  sum_dim_int_list_195 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:227 in forward, code: weight=(self.gain * self.scale).view(-1),
        reshape_default_284: "f32[16, 1, 1, 1]" = torch.ops.aten.reshape.default(mul_tensor_653, _shape_param_227);  mul_tensor_653 = _shape_param_227 = None
        mul_tensor_654: "f32[16, 1, 1, 1]" = torch.ops.aten.mul.Tensor(reshape_default_284, 0.19245008972987526);  reshape_default_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/std_conv.py:224 in forward, code: self.weight.reshape(1, self.out_channels, -1),
        reshape_default_285: "f32[16, 3, 3, 3]" = torch.ops.aten.reshape.default(mul_tensor_652, _shape_param_228);  mul_tensor_652 = _shape_param_228 = None
        return (permute_default, reshape_default, sum_dim_int_list_1, mul_tensor_10, reshape_default_5, sum_default, sum_dim_int_list_4, sum_dim_int_list_5, sum_dim_int_list_6, mul_tensor_24, reshape_default_10, sum_dim_int_list_9, mul_tensor_35, reshape_default_15, sum_dim_int_list_12, mul_tensor_46, reshape_default_20, sum_dim_int_list_15, mul_tensor_57, reshape_default_25, sum_default_1, sum_dim_int_list_18, sum_dim_int_list_19, sum_dim_int_list_20, mul_tensor_71, reshape_default_30, sum_dim_int_list_23, mul_tensor_82, reshape_default_35, sum_dim_int_list_26, mul_tensor_93, reshape_default_40, sum_dim_int_list_29, mul_tensor_104, reshape_default_45, sum_default_2, sum_dim_int_list_32, sum_dim_int_list_33, sum_dim_int_list_34, mul_tensor_116, reshape_default_50, sum_dim_int_list_37, mul_tensor_127, reshape_default_55, sum_dim_int_list_40, mul_tensor_138, reshape_default_60, sum_dim_int_list_43, mul_tensor_149, reshape_default_65, sum_dim_int_list_46, mul_tensor_160, reshape_default_70, sum_default_3, sum_dim_int_list_49, sum_dim_int_list_50, sum_dim_int_list_51, mul_tensor_174, reshape_default_75, sum_dim_int_list_54, mul_tensor_185, reshape_default_80, sum_dim_int_list_57, mul_tensor_196, reshape_default_85, sum_dim_int_list_60, mul_tensor_207, reshape_default_90, sum_default_4, sum_dim_int_list_63, sum_dim_int_list_64, sum_dim_int_list_65, mul_tensor_221, reshape_default_95, sum_dim_int_list_68, mul_tensor_232, reshape_default_100, sum_dim_int_list_71, mul_tensor_243, reshape_default_105, sum_dim_int_list_74, mul_tensor_254, reshape_default_110, sum_default_5, sum_dim_int_list_77, sum_dim_int_list_78, sum_dim_int_list_79, mul_tensor_268, reshape_default_115, sum_dim_int_list_82, mul_tensor_279, reshape_default_120, sum_dim_int_list_85, mul_tensor_290, reshape_default_125, sum_dim_int_list_88, mul_tensor_301, reshape_default_130, sum_default_6, sum_dim_int_list_91, sum_dim_int_list_92, sum_dim_int_list_93, mul_tensor_315, reshape_default_135, sum_dim_int_list_96, mul_tensor_326, reshape_default_140, sum_dim_int_list_99, mul_tensor_337, reshape_default_145, sum_dim_int_list_102, mul_tensor_348, reshape_default_150, sum_default_7, sum_dim_int_list_105, sum_dim_int_list_106, sum_dim_int_list_107, mul_tensor_362, reshape_default_155, sum_dim_int_list_110, mul_tensor_373, reshape_default_160, sum_dim_int_list_113, mul_tensor_384, reshape_default_165, sum_dim_int_list_116, mul_tensor_395, reshape_default_170, sum_default_8, sum_dim_int_list_119, sum_dim_int_list_120, sum_dim_int_list_121, mul_tensor_407, reshape_default_175, sum_dim_int_list_124, mul_tensor_418, reshape_default_180, sum_dim_int_list_127, mul_tensor_429, reshape_default_185, sum_dim_int_list_130, mul_tensor_440, reshape_default_190, sum_dim_int_list_133, mul_tensor_451, reshape_default_195, sum_default_9, sum_dim_int_list_136, sum_dim_int_list_137, sum_dim_int_list_138, mul_tensor_465, reshape_default_200, sum_dim_int_list_141, mul_tensor_476, reshape_default_205, sum_dim_int_list_144, mul_tensor_487, reshape_default_210, sum_dim_int_list_147, mul_tensor_498, reshape_default_215, sum_default_10, sum_dim_int_list_150, sum_dim_int_list_151, sum_dim_int_list_152, mul_tensor_510, reshape_default_220, sum_dim_int_list_155, mul_tensor_521, reshape_default_225, sum_dim_int_list_158, mul_tensor_532, reshape_default_230, sum_dim_int_list_161, mul_tensor_543, reshape_default_235, sum_dim_int_list_164, mul_tensor_554, reshape_default_240, sum_default_11, sum_dim_int_list_167, sum_dim_int_list_168, sum_dim_int_list_169, mul_tensor_566, reshape_default_245, sum_dim_int_list_172, mul_tensor_577, reshape_default_250, sum_dim_int_list_175, mul_tensor_588, reshape_default_255, sum_dim_int_list_178, mul_tensor_599, reshape_default_260, sum_dim_int_list_181, mul_tensor_610, reshape_default_265, sum_dim_int_list_184, mul_tensor_621, reshape_default_270, sum_dim_int_list_187, mul_tensor_632, reshape_default_275, sum_dim_int_list_190, mul_tensor_643, reshape_default_280, sum_dim_int_list_193, mul_tensor_654, reshape_default_285)
