class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024]", primals_2: "f32[32128, 768]", primals_3: "f32[768]", primals_4: "f32[768, 768]", primals_5: "f32[768, 768]", primals_6: "f32[768, 768]", primals_8: "f32[768, 768]", primals_9: "f32[768]", primals_10: "f32[3072, 768]", primals_11: "f32[768, 3072]", primals_12: "f32[768]", primals_13: "f32[768, 768]", primals_14: "f32[768, 768]", primals_15: "f32[768, 768]", primals_16: "f32[768, 768]", primals_17: "f32[768]", primals_18: "f32[3072, 768]", primals_19: "f32[768, 3072]", primals_20: "f32[768]", primals_21: "f32[768, 768]", primals_22: "f32[768, 768]", primals_23: "f32[768, 768]", primals_24: "f32[768, 768]", primals_25: "f32[768]", primals_26: "f32[3072, 768]", primals_27: "f32[768, 3072]", primals_28: "f32[768]", primals_29: "f32[768, 768]", primals_30: "f32[768, 768]", primals_31: "f32[768, 768]", primals_32: "f32[768, 768]", primals_33: "f32[768]", primals_34: "f32[3072, 768]", primals_35: "f32[768, 3072]", primals_36: "f32[768]", primals_37: "f32[768, 768]", primals_38: "f32[768, 768]", primals_39: "f32[768, 768]", primals_40: "f32[768, 768]", primals_41: "f32[768]", primals_42: "f32[3072, 768]", primals_43: "f32[768, 3072]", primals_44: "f32[768]", primals_45: "f32[768, 768]", primals_46: "f32[768, 768]", primals_47: "f32[768, 768]", primals_48: "f32[768, 768]", primals_49: "f32[768]", primals_50: "f32[3072, 768]", primals_51: "f32[768, 3072]", primals_52: "f32[768]", primals_53: "f32[768, 768]", primals_54: "f32[768, 768]", primals_55: "f32[768, 768]", primals_56: "f32[768, 768]", primals_57: "f32[768]", primals_58: "f32[3072, 768]", primals_59: "f32[768, 3072]", primals_60: "f32[768]", primals_61: "f32[768, 768]", primals_62: "f32[768, 768]", primals_63: "f32[768, 768]", primals_64: "f32[768, 768]", primals_65: "f32[768]", primals_66: "f32[3072, 768]", primals_67: "f32[768, 3072]", primals_68: "f32[768]", primals_69: "f32[768, 768]", primals_70: "f32[768, 768]", primals_71: "f32[768, 768]", primals_72: "f32[768, 768]", primals_73: "f32[768]", primals_74: "f32[3072, 768]", primals_75: "f32[768, 3072]", primals_76: "f32[768]", primals_77: "f32[768, 768]", primals_78: "f32[768, 768]", primals_79: "f32[768, 768]", primals_80: "f32[768, 768]", primals_81: "f32[768]", primals_82: "f32[3072, 768]", primals_83: "f32[768, 3072]", primals_84: "f32[768]", primals_85: "f32[768, 768]", primals_86: "f32[768, 768]", primals_87: "f32[768, 768]", primals_88: "f32[768, 768]", primals_89: "f32[768]", primals_90: "f32[3072, 768]", primals_91: "f32[768, 3072]", primals_92: "f32[768]", primals_93: "f32[768, 768]", primals_94: "f32[768, 768]", primals_95: "f32[768, 768]", primals_96: "f32[768, 768]", primals_97: "f32[768]", primals_98: "f32[3072, 768]", primals_99: "f32[768, 3072]", primals_100: "f32[768]", primals_101: "i64[8, 1024]", primals_102: "f32[768]", primals_103: "f32[768, 768]", primals_104: "f32[768, 768]", primals_105: "f32[768, 768]", primals_107: "f32[768, 768]", primals_108: "f32[768]", primals_109: "f32[768, 768]", primals_110: "f32[768, 768]", primals_111: "f32[768, 768]", primals_112: "f32[768, 768]", primals_113: "f32[768]", primals_114: "f32[3072, 768]", primals_115: "f32[768, 3072]", primals_116: "f32[768]", primals_117: "f32[768, 768]", primals_118: "f32[768, 768]", primals_119: "f32[768, 768]", primals_120: "f32[768, 768]", primals_121: "f32[768]", primals_122: "f32[768, 768]", primals_123: "f32[768, 768]", primals_124: "f32[768, 768]", primals_125: "f32[768, 768]", primals_126: "f32[768]", primals_127: "f32[3072, 768]", primals_128: "f32[768, 3072]", primals_129: "f32[768]", primals_130: "f32[768, 768]", primals_131: "f32[768, 768]", primals_132: "f32[768, 768]", primals_133: "f32[768, 768]", primals_134: "f32[768]", primals_135: "f32[768, 768]", primals_136: "f32[768, 768]", primals_137: "f32[768, 768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[3072, 768]", primals_141: "f32[768, 3072]", primals_142: "f32[768]", primals_143: "f32[768, 768]", primals_144: "f32[768, 768]", primals_145: "f32[768, 768]", primals_146: "f32[768, 768]", primals_147: "f32[768]", primals_148: "f32[768, 768]", primals_149: "f32[768, 768]", primals_150: "f32[768, 768]", primals_151: "f32[768, 768]", primals_152: "f32[768]", primals_153: "f32[3072, 768]", primals_154: "f32[768, 3072]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768, 768]", primals_158: "f32[768, 768]", primals_159: "f32[768, 768]", primals_160: "f32[768]", primals_161: "f32[768, 768]", primals_162: "f32[768, 768]", primals_163: "f32[768, 768]", primals_164: "f32[768, 768]", primals_165: "f32[768]", primals_166: "f32[3072, 768]", primals_167: "f32[768, 3072]", primals_168: "f32[768]", primals_169: "f32[768, 768]", primals_170: "f32[768, 768]", primals_171: "f32[768, 768]", primals_172: "f32[768, 768]", primals_173: "f32[768]", primals_174: "f32[768, 768]", primals_175: "f32[768, 768]", primals_176: "f32[768, 768]", primals_177: "f32[768, 768]", primals_178: "f32[768]", primals_179: "f32[3072, 768]", primals_180: "f32[768, 3072]", primals_181: "f32[768]", primals_182: "f32[768, 768]", primals_183: "f32[768, 768]", primals_184: "f32[768, 768]", primals_185: "f32[768, 768]", primals_186: "f32[768]", primals_187: "f32[768, 768]", primals_188: "f32[768, 768]", primals_189: "f32[768, 768]", primals_190: "f32[768, 768]", primals_191: "f32[768]", primals_192: "f32[3072, 768]", primals_193: "f32[768, 3072]", primals_194: "f32[768]", primals_195: "f32[768, 768]", primals_196: "f32[768, 768]", primals_197: "f32[768, 768]", primals_198: "f32[768, 768]", primals_199: "f32[768]", primals_200: "f32[768, 768]", primals_201: "f32[768, 768]", primals_202: "f32[768, 768]", primals_203: "f32[768, 768]", primals_204: "f32[768]", primals_205: "f32[3072, 768]", primals_206: "f32[768, 3072]", primals_207: "f32[768]", primals_208: "f32[768, 768]", primals_209: "f32[768, 768]", primals_210: "f32[768, 768]", primals_211: "f32[768, 768]", primals_212: "f32[768]", primals_213: "f32[768, 768]", primals_214: "f32[768, 768]", primals_215: "f32[768, 768]", primals_216: "f32[768, 768]", primals_217: "f32[768]", primals_218: "f32[3072, 768]", primals_219: "f32[768, 3072]", primals_220: "f32[768]", primals_221: "f32[768, 768]", primals_222: "f32[768, 768]", primals_223: "f32[768, 768]", primals_224: "f32[768, 768]", primals_225: "f32[768]", primals_226: "f32[768, 768]", primals_227: "f32[768, 768]", primals_228: "f32[768, 768]", primals_229: "f32[768, 768]", primals_230: "f32[768]", primals_231: "f32[3072, 768]", primals_232: "f32[768, 3072]", primals_233: "f32[768]", primals_234: "f32[768, 768]", primals_235: "f32[768, 768]", primals_236: "f32[768, 768]", primals_237: "f32[768, 768]", primals_238: "f32[768]", primals_239: "f32[768, 768]", primals_240: "f32[768, 768]", primals_241: "f32[768, 768]", primals_242: "f32[768, 768]", primals_243: "f32[768]", primals_244: "f32[3072, 768]", primals_245: "f32[768, 3072]", primals_246: "f32[768]", primals_247: "f32[768, 768]", primals_248: "f32[768, 768]", primals_249: "f32[768, 768]", primals_250: "f32[768, 768]", primals_251: "f32[768]", primals_252: "f32[768, 768]", primals_253: "f32[768, 768]", primals_254: "f32[768, 768]", primals_255: "f32[768, 768]", primals_256: "f32[768]", primals_257: "f32[3072, 768]", primals_258: "f32[768, 3072]", primals_259: "f32[768]", embedding: "f32[8, 1024, 768]", ge: "b8[1, 1, 1024, 1]", gt: "b8[8, 1024, 768]", rsqrt: "f32[8, 1024, 1]", view_1: "f32[8192, 768]", bmm: "f32[96, 1024, 1024]", add_6: "i64[1024, 1024]", embedding_1: "f32[1024, 1024, 12]", amax: "f32[8, 12, 1024, 1]", sum_1: "f32[8, 12, 1024, 1]", gt_2: "b8[8, 12, 1024, 1024]", view_20: "f32[8192, 768]", gt_3: "b8[8, 1024, 768]", add_9: "f32[8, 1024, 768]", rsqrt_1: "f32[8, 1024, 1]", view_22: "f32[8192, 768]", gt_4: "b8[8, 1024, 3072]", view_24: "f32[8192, 3072]", gt_5: "b8[8, 1024, 768]", add_11: "f32[8, 1024, 768]", rsqrt_2: "f32[8, 1024, 1]", view_26: "f32[8192, 768]", div_3: "f32[8, 12, 1024, 1024]", gt_6: "b8[8, 12, 1024, 1024]", view_45: "f32[8192, 768]", gt_7: "b8[8, 1024, 768]", add_14: "f32[8, 1024, 768]", rsqrt_3: "f32[8, 1024, 1]", view_47: "f32[8192, 768]", gt_8: "b8[8, 1024, 3072]", view_49: "f32[8192, 3072]", gt_9: "b8[8, 1024, 768]", add_16: "f32[8, 1024, 768]", rsqrt_4: "f32[8, 1024, 1]", view_51: "f32[8192, 768]", div_4: "f32[8, 12, 1024, 1024]", gt_10: "b8[8, 12, 1024, 1024]", view_70: "f32[8192, 768]", gt_11: "b8[8, 1024, 768]", add_19: "f32[8, 1024, 768]", rsqrt_5: "f32[8, 1024, 1]", view_72: "f32[8192, 768]", gt_12: "b8[8, 1024, 3072]", view_74: "f32[8192, 3072]", gt_13: "b8[8, 1024, 768]", add_21: "f32[8, 1024, 768]", rsqrt_6: "f32[8, 1024, 1]", view_76: "f32[8192, 768]", div_5: "f32[8, 12, 1024, 1024]", gt_14: "b8[8, 12, 1024, 1024]", view_95: "f32[8192, 768]", gt_15: "b8[8, 1024, 768]", add_24: "f32[8, 1024, 768]", rsqrt_7: "f32[8, 1024, 1]", view_97: "f32[8192, 768]", gt_16: "b8[8, 1024, 3072]", view_99: "f32[8192, 3072]", gt_17: "b8[8, 1024, 768]", add_26: "f32[8, 1024, 768]", rsqrt_8: "f32[8, 1024, 1]", view_101: "f32[8192, 768]", div_6: "f32[8, 12, 1024, 1024]", gt_18: "b8[8, 12, 1024, 1024]", view_120: "f32[8192, 768]", gt_19: "b8[8, 1024, 768]", add_29: "f32[8, 1024, 768]", rsqrt_9: "f32[8, 1024, 1]", view_122: "f32[8192, 768]", gt_20: "b8[8, 1024, 3072]", view_124: "f32[8192, 3072]", gt_21: "b8[8, 1024, 768]", add_31: "f32[8, 1024, 768]", rsqrt_10: "f32[8, 1024, 1]", view_126: "f32[8192, 768]", div_7: "f32[8, 12, 1024, 1024]", gt_22: "b8[8, 12, 1024, 1024]", view_145: "f32[8192, 768]", gt_23: "b8[8, 1024, 768]", add_34: "f32[8, 1024, 768]", rsqrt_11: "f32[8, 1024, 1]", view_147: "f32[8192, 768]", gt_24: "b8[8, 1024, 3072]", view_149: "f32[8192, 3072]", gt_25: "b8[8, 1024, 768]", add_36: "f32[8, 1024, 768]", rsqrt_12: "f32[8, 1024, 1]", view_151: "f32[8192, 768]", div_8: "f32[8, 12, 1024, 1024]", gt_26: "b8[8, 12, 1024, 1024]", view_170: "f32[8192, 768]", gt_27: "b8[8, 1024, 768]", add_39: "f32[8, 1024, 768]", rsqrt_13: "f32[8, 1024, 1]", view_172: "f32[8192, 768]", gt_28: "b8[8, 1024, 3072]", view_174: "f32[8192, 3072]", gt_29: "b8[8, 1024, 768]", add_41: "f32[8, 1024, 768]", rsqrt_14: "f32[8, 1024, 1]", view_176: "f32[8192, 768]", div_9: "f32[8, 12, 1024, 1024]", gt_30: "b8[8, 12, 1024, 1024]", view_195: "f32[8192, 768]", gt_31: "b8[8, 1024, 768]", add_44: "f32[8, 1024, 768]", rsqrt_15: "f32[8, 1024, 1]", view_197: "f32[8192, 768]", gt_32: "b8[8, 1024, 3072]", view_199: "f32[8192, 3072]", gt_33: "b8[8, 1024, 768]", add_46: "f32[8, 1024, 768]", rsqrt_16: "f32[8, 1024, 1]", view_201: "f32[8192, 768]", div_10: "f32[8, 12, 1024, 1024]", gt_34: "b8[8, 12, 1024, 1024]", view_220: "f32[8192, 768]", gt_35: "b8[8, 1024, 768]", add_49: "f32[8, 1024, 768]", rsqrt_17: "f32[8, 1024, 1]", view_222: "f32[8192, 768]", gt_36: "b8[8, 1024, 3072]", view_224: "f32[8192, 3072]", gt_37: "b8[8, 1024, 768]", add_51: "f32[8, 1024, 768]", rsqrt_18: "f32[8, 1024, 1]", view_226: "f32[8192, 768]", div_11: "f32[8, 12, 1024, 1024]", gt_38: "b8[8, 12, 1024, 1024]", view_245: "f32[8192, 768]", gt_39: "b8[8, 1024, 768]", add_54: "f32[8, 1024, 768]", rsqrt_19: "f32[8, 1024, 1]", view_247: "f32[8192, 768]", gt_40: "b8[8, 1024, 3072]", view_249: "f32[8192, 3072]", gt_41: "b8[8, 1024, 768]", add_56: "f32[8, 1024, 768]", rsqrt_20: "f32[8, 1024, 1]", view_251: "f32[8192, 768]", div_12: "f32[8, 12, 1024, 1024]", gt_42: "b8[8, 12, 1024, 1024]", view_270: "f32[8192, 768]", gt_43: "b8[8, 1024, 768]", add_59: "f32[8, 1024, 768]", rsqrt_21: "f32[8, 1024, 1]", view_272: "f32[8192, 768]", gt_44: "b8[8, 1024, 3072]", view_274: "f32[8192, 3072]", gt_45: "b8[8, 1024, 768]", add_61: "f32[8, 1024, 768]", rsqrt_22: "f32[8, 1024, 1]", view_276: "f32[8192, 768]", div_13: "f32[8, 12, 1024, 1024]", gt_46: "b8[8, 12, 1024, 1024]", view_295: "f32[8192, 768]", gt_47: "b8[8, 1024, 768]", add_64: "f32[8, 1024, 768]", rsqrt_23: "f32[8, 1024, 1]", view_297: "f32[8192, 768]", gt_48: "b8[8, 1024, 3072]", view_299: "f32[8192, 3072]", gt_49: "b8[8, 1024, 768]", add_66: "f32[8, 1024, 768]", rsqrt_24: "f32[8, 1024, 1]", gt_50: "b8[8, 1024, 768]", mul_151: "f32[8, 1024, 768]", where_2: "i64[8, 1024]", embedding_2: "f32[8, 1024, 768]", gt_51: "b8[8, 1024, 768]", rsqrt_25: "f32[8, 1024, 1]", view_303: "f32[8192, 768]", add_75: "i64[1024, 1024]", div_16: "f32[8, 12, 1024, 1024]", gt_52: "b8[8, 12, 1024, 1024]", view_322: "f32[8192, 768]", gt_53: "b8[8, 1024, 768]", add_78: "f32[8, 1024, 768]", rsqrt_26: "f32[8, 1024, 1]", view_324: "f32[8192, 768]", div_17: "f32[8, 12, 1024, 1024]", gt_54: "b8[8, 12, 1024, 1024]", view_343: "f32[8192, 768]", gt_55: "b8[8, 1024, 768]", add_82: "f32[8, 1024, 768]", rsqrt_27: "f32[8, 1024, 1]", view_345: "f32[8192, 768]", gt_56: "b8[8, 1024, 3072]", view_347: "f32[8192, 3072]", gt_57: "b8[8, 1024, 768]", add_84: "f32[8, 1024, 768]", rsqrt_28: "f32[8, 1024, 1]", view_349: "f32[8192, 768]", div_18: "f32[8, 12, 1024, 1024]", gt_58: "b8[8, 12, 1024, 1024]", view_368: "f32[8192, 768]", gt_59: "b8[8, 1024, 768]", add_87: "f32[8, 1024, 768]", rsqrt_29: "f32[8, 1024, 1]", view_370: "f32[8192, 768]", div_19: "f32[8, 12, 1024, 1024]", gt_60: "b8[8, 12, 1024, 1024]", view_389: "f32[8192, 768]", gt_61: "b8[8, 1024, 768]", add_90: "f32[8, 1024, 768]", rsqrt_30: "f32[8, 1024, 1]", view_391: "f32[8192, 768]", gt_62: "b8[8, 1024, 3072]", view_393: "f32[8192, 3072]", gt_63: "b8[8, 1024, 768]", add_92: "f32[8, 1024, 768]", rsqrt_31: "f32[8, 1024, 1]", view_395: "f32[8192, 768]", div_20: "f32[8, 12, 1024, 1024]", gt_64: "b8[8, 12, 1024, 1024]", view_414: "f32[8192, 768]", gt_65: "b8[8, 1024, 768]", add_95: "f32[8, 1024, 768]", rsqrt_32: "f32[8, 1024, 1]", view_416: "f32[8192, 768]", div_21: "f32[8, 12, 1024, 1024]", gt_66: "b8[8, 12, 1024, 1024]", view_435: "f32[8192, 768]", gt_67: "b8[8, 1024, 768]", add_98: "f32[8, 1024, 768]", rsqrt_33: "f32[8, 1024, 1]", view_437: "f32[8192, 768]", gt_68: "b8[8, 1024, 3072]", view_439: "f32[8192, 3072]", gt_69: "b8[8, 1024, 768]", add_100: "f32[8, 1024, 768]", rsqrt_34: "f32[8, 1024, 1]", view_441: "f32[8192, 768]", div_22: "f32[8, 12, 1024, 1024]", gt_70: "b8[8, 12, 1024, 1024]", view_460: "f32[8192, 768]", gt_71: "b8[8, 1024, 768]", add_103: "f32[8, 1024, 768]", rsqrt_35: "f32[8, 1024, 1]", view_462: "f32[8192, 768]", div_23: "f32[8, 12, 1024, 1024]", gt_72: "b8[8, 12, 1024, 1024]", view_481: "f32[8192, 768]", gt_73: "b8[8, 1024, 768]", add_106: "f32[8, 1024, 768]", rsqrt_36: "f32[8, 1024, 1]", view_483: "f32[8192, 768]", gt_74: "b8[8, 1024, 3072]", view_485: "f32[8192, 3072]", gt_75: "b8[8, 1024, 768]", add_108: "f32[8, 1024, 768]", rsqrt_37: "f32[8, 1024, 1]", view_487: "f32[8192, 768]", div_24: "f32[8, 12, 1024, 1024]", gt_76: "b8[8, 12, 1024, 1024]", view_506: "f32[8192, 768]", gt_77: "b8[8, 1024, 768]", add_111: "f32[8, 1024, 768]", rsqrt_38: "f32[8, 1024, 1]", view_508: "f32[8192, 768]", div_25: "f32[8, 12, 1024, 1024]", gt_78: "b8[8, 12, 1024, 1024]", view_527: "f32[8192, 768]", gt_79: "b8[8, 1024, 768]", add_114: "f32[8, 1024, 768]", rsqrt_39: "f32[8, 1024, 1]", view_529: "f32[8192, 768]", gt_80: "b8[8, 1024, 3072]", view_531: "f32[8192, 3072]", gt_81: "b8[8, 1024, 768]", add_116: "f32[8, 1024, 768]", rsqrt_40: "f32[8, 1024, 1]", view_533: "f32[8192, 768]", div_26: "f32[8, 12, 1024, 1024]", gt_82: "b8[8, 12, 1024, 1024]", view_552: "f32[8192, 768]", gt_83: "b8[8, 1024, 768]", add_119: "f32[8, 1024, 768]", rsqrt_41: "f32[8, 1024, 1]", view_554: "f32[8192, 768]", div_27: "f32[8, 12, 1024, 1024]", gt_84: "b8[8, 12, 1024, 1024]", view_573: "f32[8192, 768]", gt_85: "b8[8, 1024, 768]", add_122: "f32[8, 1024, 768]", rsqrt_42: "f32[8, 1024, 1]", view_575: "f32[8192, 768]", gt_86: "b8[8, 1024, 3072]", view_577: "f32[8192, 3072]", gt_87: "b8[8, 1024, 768]", add_124: "f32[8, 1024, 768]", rsqrt_43: "f32[8, 1024, 1]", view_579: "f32[8192, 768]", div_28: "f32[8, 12, 1024, 1024]", gt_88: "b8[8, 12, 1024, 1024]", view_598: "f32[8192, 768]", gt_89: "b8[8, 1024, 768]", add_127: "f32[8, 1024, 768]", rsqrt_44: "f32[8, 1024, 1]", view_600: "f32[8192, 768]", div_29: "f32[8, 12, 1024, 1024]", gt_90: "b8[8, 12, 1024, 1024]", view_619: "f32[8192, 768]", gt_91: "b8[8, 1024, 768]", add_130: "f32[8, 1024, 768]", rsqrt_45: "f32[8, 1024, 1]", view_621: "f32[8192, 768]", gt_92: "b8[8, 1024, 3072]", view_623: "f32[8192, 3072]", gt_93: "b8[8, 1024, 768]", add_132: "f32[8, 1024, 768]", rsqrt_46: "f32[8, 1024, 1]", view_625: "f32[8192, 768]", div_30: "f32[8, 12, 1024, 1024]", gt_94: "b8[8, 12, 1024, 1024]", view_644: "f32[8192, 768]", gt_95: "b8[8, 1024, 768]", add_135: "f32[8, 1024, 768]", rsqrt_47: "f32[8, 1024, 1]", view_646: "f32[8192, 768]", div_31: "f32[8, 12, 1024, 1024]", gt_96: "b8[8, 12, 1024, 1024]", view_665: "f32[8192, 768]", gt_97: "b8[8, 1024, 768]", add_138: "f32[8, 1024, 768]", rsqrt_48: "f32[8, 1024, 1]", view_667: "f32[8192, 768]", gt_98: "b8[8, 1024, 3072]", view_669: "f32[8192, 3072]", gt_99: "b8[8, 1024, 768]", add_140: "f32[8, 1024, 768]", rsqrt_49: "f32[8, 1024, 1]", view_671: "f32[8192, 768]", div_32: "f32[8, 12, 1024, 1024]", gt_100: "b8[8, 12, 1024, 1024]", view_690: "f32[8192, 768]", gt_101: "b8[8, 1024, 768]", add_143: "f32[8, 1024, 768]", rsqrt_50: "f32[8, 1024, 1]", view_692: "f32[8192, 768]", div_33: "f32[8, 12, 1024, 1024]", gt_102: "b8[8, 12, 1024, 1024]", view_711: "f32[8192, 768]", gt_103: "b8[8, 1024, 768]", add_146: "f32[8, 1024, 768]", rsqrt_51: "f32[8, 1024, 1]", view_713: "f32[8192, 768]", gt_104: "b8[8, 1024, 3072]", view_715: "f32[8192, 3072]", gt_105: "b8[8, 1024, 768]", add_148: "f32[8, 1024, 768]", rsqrt_52: "f32[8, 1024, 1]", view_717: "f32[8192, 768]", div_34: "f32[8, 12, 1024, 1024]", gt_106: "b8[8, 12, 1024, 1024]", view_736: "f32[8192, 768]", gt_107: "b8[8, 1024, 768]", add_151: "f32[8, 1024, 768]", rsqrt_53: "f32[8, 1024, 1]", view_738: "f32[8192, 768]", div_35: "f32[8, 12, 1024, 1024]", gt_108: "b8[8, 12, 1024, 1024]", view_757: "f32[8192, 768]", gt_109: "b8[8, 1024, 768]", add_154: "f32[8, 1024, 768]", rsqrt_54: "f32[8, 1024, 1]", view_759: "f32[8192, 768]", gt_110: "b8[8, 1024, 3072]", view_761: "f32[8192, 3072]", gt_111: "b8[8, 1024, 768]", add_156: "f32[8, 1024, 768]", rsqrt_55: "f32[8, 1024, 1]", view_763: "f32[8192, 768]", div_36: "f32[8, 12, 1024, 1024]", gt_112: "b8[8, 12, 1024, 1024]", view_782: "f32[8192, 768]", gt_113: "b8[8, 1024, 768]", add_159: "f32[8, 1024, 768]", rsqrt_56: "f32[8, 1024, 1]", view_784: "f32[8192, 768]", div_37: "f32[8, 12, 1024, 1024]", gt_114: "b8[8, 12, 1024, 1024]", view_803: "f32[8192, 768]", gt_115: "b8[8, 1024, 768]", add_162: "f32[8, 1024, 768]", rsqrt_57: "f32[8, 1024, 1]", view_805: "f32[8192, 768]", gt_116: "b8[8, 1024, 3072]", view_807: "f32[8192, 3072]", gt_117: "b8[8, 1024, 768]", add_164: "f32[8, 1024, 768]", rsqrt_58: "f32[8, 1024, 1]", view_809: "f32[8192, 768]", div_38: "f32[8, 12, 1024, 1024]", gt_118: "b8[8, 12, 1024, 1024]", view_828: "f32[8192, 768]", gt_119: "b8[8, 1024, 768]", add_167: "f32[8, 1024, 768]", rsqrt_59: "f32[8, 1024, 1]", view_830: "f32[8192, 768]", div_39: "f32[8, 12, 1024, 1024]", gt_120: "b8[8, 12, 1024, 1024]", view_849: "f32[8192, 768]", gt_121: "b8[8, 1024, 768]", add_170: "f32[8, 1024, 768]", rsqrt_60: "f32[8, 1024, 1]", view_851: "f32[8192, 768]", gt_122: "b8[8, 1024, 3072]", view_853: "f32[8192, 3072]", gt_123: "b8[8, 1024, 768]", add_172: "f32[8, 1024, 768]", rsqrt_61: "f32[8, 1024, 1]", gt_124: "b8[8, 1024, 768]", view_855: "f32[8192, 768]", view_856: "f32[8, 1024, 32128]", amax_36: "f32[8192, 1]", log_2: "f32[8192, 1]", convert_element_type_5: "f32[]", le_1: "b8[8, 1024, 3072]", permute_392: "f32[96, 1024, 1024]", permute_393: "f32[96, 64, 1024]", permute_394: "f32[96, 64, 1024]", permute_395: "f32[96, 1024, 64]", permute_417: "f32[96, 1024, 1024]", permute_418: "f32[96, 64, 1024]", permute_419: "f32[96, 64, 1024]", permute_420: "f32[96, 1024, 64]", le_2: "b8[8, 1024, 3072]", permute_450: "f32[96, 1024, 1024]", permute_451: "f32[96, 64, 1024]", permute_452: "f32[96, 64, 1024]", permute_453: "f32[96, 1024, 64]", permute_475: "f32[96, 1024, 1024]", permute_476: "f32[96, 64, 1024]", permute_477: "f32[96, 64, 1024]", permute_478: "f32[96, 1024, 64]", le_3: "b8[8, 1024, 3072]", permute_508: "f32[96, 1024, 1024]", permute_509: "f32[96, 64, 1024]", permute_510: "f32[96, 64, 1024]", permute_511: "f32[96, 1024, 64]", permute_533: "f32[96, 1024, 1024]", permute_534: "f32[96, 64, 1024]", permute_535: "f32[96, 64, 1024]", permute_536: "f32[96, 1024, 64]", le_4: "b8[8, 1024, 3072]", permute_566: "f32[96, 1024, 1024]", permute_567: "f32[96, 64, 1024]", permute_568: "f32[96, 64, 1024]", permute_569: "f32[96, 1024, 64]", permute_591: "f32[96, 1024, 1024]", permute_592: "f32[96, 64, 1024]", permute_593: "f32[96, 64, 1024]", permute_594: "f32[96, 1024, 64]", le_5: "b8[8, 1024, 3072]", permute_624: "f32[96, 1024, 1024]", permute_625: "f32[96, 64, 1024]", permute_626: "f32[96, 64, 1024]", permute_627: "f32[96, 1024, 64]", permute_649: "f32[96, 1024, 1024]", permute_650: "f32[96, 64, 1024]", permute_651: "f32[96, 64, 1024]", permute_652: "f32[96, 1024, 64]", le_6: "b8[8, 1024, 3072]", permute_682: "f32[96, 1024, 1024]", permute_683: "f32[96, 64, 1024]", permute_684: "f32[96, 64, 1024]", permute_685: "f32[96, 1024, 64]", permute_707: "f32[96, 1024, 1024]", permute_708: "f32[96, 64, 1024]", permute_709: "f32[96, 64, 1024]", permute_710: "f32[96, 1024, 64]", le_7: "b8[8, 1024, 3072]", permute_740: "f32[96, 1024, 1024]", permute_741: "f32[96, 64, 1024]", permute_742: "f32[96, 64, 1024]", permute_743: "f32[96, 1024, 64]", permute_765: "f32[96, 1024, 1024]", permute_766: "f32[96, 64, 1024]", permute_767: "f32[96, 64, 1024]", permute_768: "f32[96, 1024, 64]", le_8: "b8[8, 1024, 3072]", permute_798: "f32[96, 1024, 1024]", permute_799: "f32[96, 64, 1024]", permute_800: "f32[96, 64, 1024]", permute_801: "f32[96, 1024, 64]", permute_823: "f32[96, 1024, 1024]", permute_824: "f32[96, 64, 1024]", permute_825: "f32[96, 64, 1024]", permute_826: "f32[96, 1024, 64]", le_9: "b8[8, 1024, 3072]", permute_856: "f32[96, 1024, 1024]", permute_857: "f32[96, 64, 1024]", permute_858: "f32[96, 64, 1024]", permute_859: "f32[96, 1024, 64]", permute_881: "f32[96, 1024, 1024]", permute_882: "f32[96, 64, 1024]", permute_883: "f32[96, 64, 1024]", permute_884: "f32[96, 1024, 64]", le_10: "b8[8, 1024, 3072]", permute_914: "f32[96, 1024, 1024]", permute_915: "f32[96, 64, 1024]", permute_916: "f32[96, 64, 1024]", permute_917: "f32[96, 1024, 64]", permute_939: "f32[96, 1024, 1024]", permute_940: "f32[96, 64, 1024]", permute_941: "f32[96, 64, 1024]", permute_942: "f32[96, 1024, 64]", le_11: "b8[8, 1024, 3072]", permute_972: "f32[96, 1024, 1024]", permute_973: "f32[96, 64, 1024]", permute_974: "f32[96, 64, 1024]", permute_975: "f32[96, 1024, 64]", permute_997: "f32[96, 1024, 1024]", permute_998: "f32[96, 64, 1024]", permute_999: "f32[96, 64, 1024]", permute_1000: "f32[96, 1024, 64]", le_12: "b8[8, 1024, 3072]", permute_1030: "f32[96, 1024, 1024]", permute_1031: "f32[96, 64, 1024]", permute_1032: "f32[96, 64, 1024]", permute_1033: "f32[96, 1024, 64]", permute_1055: "f32[96, 1024, 1024]", permute_1056: "f32[96, 64, 1024]", permute_1058: "f32[96, 64, 1024]", permute_1059: "f32[96, 1024, 64]", le_13: "b8[8, 1024, 3072]", permute_1089: "f32[96, 1024, 1024]", permute_1090: "f32[96, 64, 1024]", permute_1091: "f32[96, 64, 1024]", permute_1092: "f32[96, 1024, 64]", le_14: "b8[8, 1024, 3072]", permute_1122: "f32[96, 1024, 1024]", permute_1123: "f32[96, 64, 1024]", permute_1124: "f32[96, 64, 1024]", permute_1125: "f32[96, 1024, 64]", le_15: "b8[8, 1024, 3072]", permute_1155: "f32[96, 1024, 1024]", permute_1156: "f32[96, 64, 1024]", permute_1157: "f32[96, 64, 1024]", permute_1158: "f32[96, 1024, 64]", le_16: "b8[8, 1024, 3072]", permute_1188: "f32[96, 1024, 1024]", permute_1189: "f32[96, 64, 1024]", permute_1190: "f32[96, 64, 1024]", permute_1191: "f32[96, 1024, 64]", le_17: "b8[8, 1024, 3072]", permute_1221: "f32[96, 1024, 1024]", permute_1222: "f32[96, 64, 1024]", permute_1223: "f32[96, 64, 1024]", permute_1224: "f32[96, 1024, 64]", le_18: "b8[8, 1024, 3072]", permute_1254: "f32[96, 1024, 1024]", permute_1255: "f32[96, 64, 1024]", permute_1256: "f32[96, 64, 1024]", permute_1257: "f32[96, 1024, 64]", le_19: "b8[8, 1024, 3072]", permute_1287: "f32[96, 1024, 1024]", permute_1288: "f32[96, 64, 1024]", permute_1289: "f32[96, 64, 1024]", permute_1290: "f32[96, 1024, 64]", le_20: "b8[8, 1024, 3072]", permute_1320: "f32[96, 1024, 1024]", permute_1321: "f32[96, 64, 1024]", permute_1322: "f32[96, 64, 1024]", permute_1323: "f32[96, 1024, 64]", le_21: "b8[8, 1024, 3072]", permute_1353: "f32[96, 1024, 1024]", permute_1354: "f32[96, 64, 1024]", permute_1355: "f32[96, 64, 1024]", permute_1356: "f32[96, 1024, 64]", le_22: "b8[8, 1024, 3072]", permute_1386: "f32[96, 1024, 1024]", permute_1387: "f32[96, 64, 1024]", permute_1388: "f32[96, 64, 1024]", permute_1389: "f32[96, 1024, 64]", le_23: "b8[8, 1024, 3072]", permute_1419: "f32[96, 1024, 1024]", permute_1420: "f32[96, 64, 1024]", permute_1421: "f32[96, 64, 1024]", permute_1422: "f32[96, 1024, 64]", le_24: "b8[8, 1024, 3072]", permute_1452: "f32[96, 1024, 1024]", permute_1453: "f32[96, 64, 1024]", permute_1455: "f32[96, 64, 1024]", permute_1456: "f32[96, 1024, 64]", tangents_1: "f32[]", tangents_2: "f32[8, 1024, 32128]", tangents_3: "f32[8, 1024, 768]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        div_41: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_5);  tangents_1 = convert_element_type_5 = None
        view_858: "i64[8192]" = torch.ops.aten.reshape.default(primals_101, [-1]);  primals_101 = None
        unsqueeze_19: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(view_858, 1);  view_858 = None
        ne_3: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_19, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:599 in _shift_right, code: shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        full_default_4: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_8: "i64[8192, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_19, full_default_4);  unsqueeze_19 = full_default_4 = None

        # No stacktrace found for following nodes
        iota_default: "i64[32128]" = torch.ops.prims.iota.default(32128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 32128]" = torch.ops.aten.reshape.default(iota_default, [1, 32128]);  iota_default = None
        expand_default: "i64[8192, 32128]" = torch.ops.aten.expand.default(where_8, [8192, 32128]);  where_8 = None
        eq_tensor: "b8[8192, 32128]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_self: "f32[8192, 32128]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_9: "f32[8192, 1]" = torch.ops.aten.where.self(ne_3, div_41, full_default);  ne_3 = div_41 = None
        mul_376: "f32[8192, 32128]" = torch.ops.aten.mul.Tensor(where_self, where_9);  where_self = where_9 = None
        view_857: "f32[8192, 32128]" = torch.ops.aten.reshape.default(view_856, [-1, 32128]);  view_856 = None
        sub_38: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(view_857, amax_36);  view_857 = amax_36 = None
        sub_39: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(sub_38, log_2);  sub_38 = log_2 = None
        exp_37: "f32[8192, 32128]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
        sum_40: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [1], True)
        mul_377: "f32[8192, 32128]" = torch.ops.aten.mul.Tensor(exp_37, sum_40);  exp_37 = sum_40 = None
        sub_40: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
        view_859: "f32[8, 1024, 32128]" = torch.ops.aten.reshape.default(sub_40, [8, 1024, 32128]);  sub_40 = None
        add_174: "f32[8, 1024, 32128]" = torch.ops.aten.add.Tensor(tangents_2, view_859);  tangents_2 = view_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        view_860: "f32[8192, 32128]" = torch.ops.aten.reshape.default(add_174, [8192, 32128]);  add_174 = None
        permute_375: "f32[32128, 8192]" = torch.ops.aten.permute.default(view_860, [1, 0])
        mm_193: "f32[32128, 768]" = torch.ops.aten.mm.default(permute_375, view_855);  permute_375 = view_855 = None
        permute_374: "f32[768, 32128]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_377: "f32[32128, 768]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
        mm_194: "f32[8192, 768]" = torch.ops.aten.mm.default(view_860, permute_377);  view_860 = permute_377 = None
        view_861: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_194, [8, 1024, 768]);  mm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_378: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_861, 0.03608439182435161);  view_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_6: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_124, torch.float32);  gt_124 = None
        mul_379: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_380: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_378, mul_379);  mul_378 = mul_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_381: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_380, primals_259);  primals_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_371: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_172, rsqrt_61)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_382: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_380, mul_371);  mul_380 = mul_371 = None
        sum_41: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1], True);  mul_382 = None
        view_862: "f32[768]" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_383: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_381, add_172)
        mul_384: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_381, rsqrt_61);  mul_381 = None
        sum_42: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
        pow_63: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_61, 3);  rsqrt_61 = None
        mul_385: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_42, -0.5);  sum_42 = None
        mul_386: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_385, pow_63);  mul_385 = pow_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_147: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_386, [8, 1024, 768]);  mul_386 = None
        div_42: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_147, 768);  expand_147 = None
        pow_64: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_172, 1.0);  add_172 = None
        mul_387: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
        mul_388: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_42, mul_387);  div_42 = mul_387 = None
        add_175: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_384, mul_388);  mul_384 = mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_7: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_123, torch.float32);  gt_123 = None
        mul_389: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_390: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_175, mul_389);  mul_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_863: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_390, [8192, 768]);  mul_390 = None
        permute_379: "f32[768, 8192]" = torch.ops.aten.permute.default(view_863, [1, 0])
        mm_195: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_379, view_853);  permute_379 = view_853 = None
        permute_373: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_258, [1, 0]);  primals_258 = None
        permute_381: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_373, [1, 0]);  permute_373 = None
        mm_196: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_863, permute_381);  view_863 = permute_381 = None
        view_864: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_196, [8, 1024, 3072]);  mm_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_122, torch.float32);  gt_122 = None
        mul_391: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_392: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_864, mul_391);  view_864 = mul_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_10: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_1, full_default, mul_392);  le_1 = mul_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_865: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_10, [8192, 3072]);  where_10 = None
        permute_383: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_865, [1, 0])
        mm_197: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_383, view_851);  permute_383 = view_851 = None
        permute_372: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
        permute_385: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
        mm_198: "f32[8192, 768]" = torch.ops.aten.mm.default(view_865, permute_385);  view_865 = permute_385 = None
        view_866: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_198, [8, 1024, 768]);  mm_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_393: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_866, primals_256);  primals_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_365: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_170, rsqrt_60)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_394: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_866, mul_365);  view_866 = mul_365 = None
        sum_43: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_394, [0, 1], True);  mul_394 = None
        view_867: "f32[768]" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_395: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_393, add_170)
        mul_396: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_393, rsqrt_60);  mul_393 = None
        sum_44: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True);  mul_395 = None
        add_176: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_175, mul_396);  add_175 = mul_396 = None
        pow_65: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_60, 3);  rsqrt_60 = None
        mul_397: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_44, -0.5);  sum_44 = None
        mul_398: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_397, pow_65);  mul_397 = pow_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_148: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_398, [8, 1024, 768]);  mul_398 = None
        div_43: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_148, 768);  expand_148 = None
        pow_66: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_170, 1.0);  add_170 = None
        mul_399: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
        mul_400: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_43, mul_399);  div_43 = mul_399 = None
        add_177: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_176, mul_400);  add_176 = mul_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_9: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_121, torch.float32);  gt_121 = None
        mul_401: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_402: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_177, mul_401);  mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_868: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_402, [8192, 768]);  mul_402 = None
        permute_387: "f32[768, 8192]" = torch.ops.aten.permute.default(view_868, [1, 0])
        mm_199: "f32[768, 768]" = torch.ops.aten.mm.default(permute_387, view_849);  permute_387 = view_849 = None
        permute_371: "f32[768, 768]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
        permute_389: "f32[768, 768]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
        mm_200: "f32[8192, 768]" = torch.ops.aten.mm.default(view_868, permute_389);  view_868 = permute_389 = None
        view_869: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_200, [8, 1024, 768]);  mm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_870: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_869, [8, 1024, 12, 64]);  view_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_391: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_870, [0, 2, 1, 3]);  view_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_149: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
        view_871: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_149, [96, 1024, 64]);  clone_149 = None
        bmm_72: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_392, view_871);  permute_392 = None
        bmm_73: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_871, permute_393);  view_871 = permute_393 = None
        view_872: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_72, [8, 12, 1024, 64]);  bmm_72 = None
        view_873: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_73, [8, 12, 1024, 1024]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_10: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_120, torch.float32);  gt_120 = None
        mul_403: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_404: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_873, mul_403);  view_873 = mul_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_405: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_404, div_39);  mul_404 = None
        sum_45: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [-1], True)
        neg_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_39);  div_39 = None
        fma: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_2, sum_45, mul_405);  neg_2 = sum_45 = mul_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_874: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma, [96, 1024, 1024]);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_74: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_394, view_874);  permute_394 = None
        bmm_75: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_874, permute_395);  view_874 = permute_395 = None
        view_879: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_74, [8, 12, 64, 1024]);  bmm_74 = None
        view_880: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_75, [8, 12, 1024, 64]);  bmm_75 = None
        permute_396: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_879, [0, 1, 3, 2]);  view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_397: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_872, [0, 2, 1, 3]);  view_872 = None
        clone_152: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_397, memory_format = torch.contiguous_format);  permute_397 = None
        view_881: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_152, [8, 1024, 768]);  clone_152 = None
        view_882: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_881, [8192, 768]);  view_881 = None
        permute_398: "f32[768, 8192]" = torch.ops.aten.permute.default(view_882, [1, 0])
        view_836: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_201: "f32[768, 768]" = torch.ops.aten.mm.default(permute_398, view_836);  permute_398 = view_836 = None
        permute_367: "f32[768, 768]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
        permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
        mm_202: "f32[8192, 768]" = torch.ops.aten.mm.default(view_882, permute_400);  view_882 = permute_400 = None
        view_883: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_202, [8, 1024, 768]);  mm_202 = None
        add_178: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(tangents_3, view_883);  tangents_3 = view_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_402: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_396, [0, 2, 1, 3]);  permute_396 = None
        view_884: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_402, [8, 1024, 768]);  permute_402 = None
        clone_153: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_884, memory_format = torch.contiguous_format);  view_884 = None
        view_885: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_153, [8192, 768]);  clone_153 = None
        permute_403: "f32[768, 8192]" = torch.ops.aten.permute.default(view_885, [1, 0])
        view_833: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_203: "f32[768, 768]" = torch.ops.aten.mm.default(permute_403, view_833);  permute_403 = view_833 = None
        permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
        permute_405: "f32[768, 768]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
        mm_204: "f32[8192, 768]" = torch.ops.aten.mm.default(view_885, permute_405);  view_885 = permute_405 = None
        view_886: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_204, [8, 1024, 768]);  mm_204 = None
        add_179: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_178, view_886);  add_178 = view_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_407: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_880, [0, 2, 1, 3]);  view_880 = None
        clone_154: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_407, memory_format = torch.contiguous_format);  permute_407 = None
        view_887: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_154, [8, 1024, 768]);  clone_154 = None
        view_888: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_887, [8192, 768]);  view_887 = None
        permute_408: "f32[768, 8192]" = torch.ops.aten.permute.default(view_888, [1, 0])
        mm_205: "f32[768, 768]" = torch.ops.aten.mm.default(permute_408, view_830);  permute_408 = view_830 = None
        permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
        permute_410: "f32[768, 768]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
        mm_206: "f32[8192, 768]" = torch.ops.aten.mm.default(view_888, permute_410);  view_888 = permute_410 = None
        view_889: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_206, [8, 1024, 768]);  mm_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_406: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_889, primals_251);  primals_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_359: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_167, rsqrt_59)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_407: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_889, mul_359);  view_889 = mul_359 = None
        sum_46: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1], True);  mul_407 = None
        view_890: "f32[768]" = torch.ops.aten.reshape.default(sum_46, [768]);  sum_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_408: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_406, add_167)
        mul_409: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_406, rsqrt_59);  mul_406 = None
        sum_47: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [2], True);  mul_408 = None
        add_180: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_177, mul_409);  add_177 = mul_409 = None
        pow_67: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_59, 3);  rsqrt_59 = None
        mul_410: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_47, -0.5);  sum_47 = None
        mul_411: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_410, pow_67);  mul_410 = pow_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_149: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_411, [8, 1024, 768]);  mul_411 = None
        div_44: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_149, 768);  expand_149 = None
        pow_68: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_167, 1.0);  add_167 = None
        mul_412: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
        mul_413: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_44, mul_412);  div_44 = mul_412 = None
        add_181: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_180, mul_413);  add_180 = mul_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_11: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_119, torch.float32);  gt_119 = None
        mul_414: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_415: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_181, mul_414);  mul_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_891: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_415, [8192, 768]);  mul_415 = None
        permute_412: "f32[768, 8192]" = torch.ops.aten.permute.default(view_891, [1, 0])
        mm_207: "f32[768, 768]" = torch.ops.aten.mm.default(permute_412, view_828);  permute_412 = view_828 = None
        permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(primals_250, [1, 0]);  primals_250 = None
        permute_414: "f32[768, 768]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
        mm_208: "f32[8192, 768]" = torch.ops.aten.mm.default(view_891, permute_414);  view_891 = permute_414 = None
        view_892: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_208, [8, 1024, 768]);  mm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_893: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_892, [8, 1024, 12, 64]);  view_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_416: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_156: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_416, memory_format = torch.contiguous_format);  permute_416 = None
        view_894: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_156, [96, 1024, 64]);  clone_156 = None
        bmm_76: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_417, view_894);  permute_417 = None
        bmm_77: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_894, permute_418);  view_894 = permute_418 = None
        view_895: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_76, [8, 12, 1024, 64]);  bmm_76 = None
        view_896: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_77, [8, 12, 1024, 1024]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_12: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_118, torch.float32);  gt_118 = None
        mul_416: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_417: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_896, mul_416);  view_896 = mul_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_418: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_417, div_38);  mul_417 = None
        sum_48: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [-1], True)
        neg_3: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_38);  div_38 = None
        fma_1: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_3, sum_48, mul_418);  neg_3 = sum_48 = mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_897: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_1, [96, 1024, 1024]);  fma_1 = None
        view_899: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_897, [8, 12, 1024, 1024]);  view_897 = None
        view_900: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_899, [96, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_78: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_419, view_900);  permute_419 = None
        bmm_79: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_900, permute_420);  view_900 = permute_420 = None
        view_902: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_78, [8, 12, 64, 1024]);  bmm_78 = None
        view_903: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_79, [8, 12, 1024, 64]);  bmm_79 = None
        permute_421: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_902, [0, 1, 3, 2]);  view_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_422: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_895, [0, 2, 1, 3]);  view_895 = None
        clone_159: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_422, memory_format = torch.contiguous_format);  permute_422 = None
        view_904: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_159, [8, 1024, 768]);  clone_159 = None
        view_905: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_904, [8192, 768]);  view_904 = None
        permute_423: "f32[768, 8192]" = torch.ops.aten.permute.default(view_905, [1, 0])
        mm_209: "f32[768, 768]" = torch.ops.aten.mm.default(permute_423, view_809);  permute_423 = None
        permute_358: "f32[768, 768]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
        permute_425: "f32[768, 768]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
        mm_210: "f32[8192, 768]" = torch.ops.aten.mm.default(view_905, permute_425);  view_905 = permute_425 = None
        view_906: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_210, [8, 1024, 768]);  mm_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_427: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_421, [0, 2, 1, 3]);  permute_421 = None
        view_907: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_427, [8, 1024, 768]);  permute_427 = None
        clone_160: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_907, memory_format = torch.contiguous_format);  view_907 = None
        view_908: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_160, [8192, 768]);  clone_160 = None
        permute_428: "f32[768, 8192]" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_211: "f32[768, 768]" = torch.ops.aten.mm.default(permute_428, view_809);  permute_428 = None
        permute_356: "f32[768, 768]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
        permute_430: "f32[768, 768]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
        mm_212: "f32[8192, 768]" = torch.ops.aten.mm.default(view_908, permute_430);  view_908 = permute_430 = None
        view_909: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_212, [8, 1024, 768]);  mm_212 = None
        add_182: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_906, view_909);  view_906 = view_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_432: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_903, [0, 2, 1, 3]);  view_903 = None
        clone_161: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_432, memory_format = torch.contiguous_format);  permute_432 = None
        view_910: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_161, [8, 1024, 768]);  clone_161 = None
        view_911: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_910, [8192, 768]);  view_910 = None
        permute_433: "f32[768, 8192]" = torch.ops.aten.permute.default(view_911, [1, 0])
        mm_213: "f32[768, 768]" = torch.ops.aten.mm.default(permute_433, view_809);  permute_433 = view_809 = None
        permute_354: "f32[768, 768]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
        permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
        mm_214: "f32[8192, 768]" = torch.ops.aten.mm.default(view_911, permute_435);  view_911 = permute_435 = None
        view_912: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_214, [8, 1024, 768]);  mm_214 = None
        add_183: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_182, view_912);  add_182 = view_912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_419: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_183, primals_246);  primals_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_353: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_164, rsqrt_58)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_420: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_183, mul_353);  add_183 = mul_353 = None
        sum_49: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 1], True);  mul_420 = None
        view_913: "f32[768]" = torch.ops.aten.reshape.default(sum_49, [768]);  sum_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_421: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_419, add_164)
        mul_422: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_419, rsqrt_58);  mul_419 = None
        sum_50: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True);  mul_421 = None
        add_184: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_181, mul_422);  add_181 = mul_422 = None
        pow_69: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_58, 3);  rsqrt_58 = None
        mul_423: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_50, -0.5);  sum_50 = None
        mul_424: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_423, pow_69);  mul_423 = pow_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_150: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_424, [8, 1024, 768]);  mul_424 = None
        div_45: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_150, 768);  expand_150 = None
        pow_70: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_164, 1.0);  add_164 = None
        mul_425: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
        mul_426: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_45, mul_425);  div_45 = mul_425 = None
        add_185: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_184, mul_426);  add_184 = mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_13: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_117, torch.float32);  gt_117 = None
        mul_427: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_428: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_185, mul_427);  mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_914: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_428, [8192, 768]);  mul_428 = None
        permute_437: "f32[768, 8192]" = torch.ops.aten.permute.default(view_914, [1, 0])
        mm_215: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_437, view_807);  permute_437 = view_807 = None
        permute_353: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
        permute_439: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_353, [1, 0]);  permute_353 = None
        mm_216: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_914, permute_439);  view_914 = permute_439 = None
        view_915: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_216, [8, 1024, 3072]);  mm_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_116, torch.float32);  gt_116 = None
        mul_429: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_430: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_915, mul_429);  view_915 = mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_11: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_2, full_default, mul_430);  le_2 = mul_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_916: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_11, [8192, 3072]);  where_11 = None
        permute_441: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_916, [1, 0])
        mm_217: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_441, view_805);  permute_441 = view_805 = None
        permute_352: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
        permute_443: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
        mm_218: "f32[8192, 768]" = torch.ops.aten.mm.default(view_916, permute_443);  view_916 = permute_443 = None
        view_917: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_218, [8, 1024, 768]);  mm_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_431: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_917, primals_243);  primals_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_347: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_162, rsqrt_57)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_432: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_917, mul_347);  view_917 = mul_347 = None
        sum_51: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1], True);  mul_432 = None
        view_918: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_433: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_431, add_162)
        mul_434: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_431, rsqrt_57);  mul_431 = None
        sum_52: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [2], True);  mul_433 = None
        add_186: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_185, mul_434);  add_185 = mul_434 = None
        pow_71: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_57, 3);  rsqrt_57 = None
        mul_435: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_52, -0.5);  sum_52 = None
        mul_436: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_435, pow_71);  mul_435 = pow_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_151: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_436, [8, 1024, 768]);  mul_436 = None
        div_46: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_151, 768);  expand_151 = None
        pow_72: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_162, 1.0);  add_162 = None
        mul_437: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
        mul_438: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_46, mul_437);  div_46 = mul_437 = None
        add_187: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_186, mul_438);  add_186 = mul_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_15: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_115, torch.float32);  gt_115 = None
        mul_439: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_440: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_187, mul_439);  mul_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_919: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_440, [8192, 768]);  mul_440 = None
        permute_445: "f32[768, 8192]" = torch.ops.aten.permute.default(view_919, [1, 0])
        mm_219: "f32[768, 768]" = torch.ops.aten.mm.default(permute_445, view_803);  permute_445 = view_803 = None
        permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(primals_242, [1, 0]);  primals_242 = None
        permute_447: "f32[768, 768]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
        mm_220: "f32[8192, 768]" = torch.ops.aten.mm.default(view_919, permute_447);  view_919 = permute_447 = None
        view_920: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_220, [8, 1024, 768]);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_921: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_920, [8, 1024, 12, 64]);  view_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_449: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_921, [0, 2, 1, 3]);  view_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_165: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_449, memory_format = torch.contiguous_format);  permute_449 = None
        view_922: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_165, [96, 1024, 64]);  clone_165 = None
        bmm_80: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_450, view_922);  permute_450 = None
        bmm_81: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_922, permute_451);  view_922 = permute_451 = None
        view_923: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_80, [8, 12, 1024, 64]);  bmm_80 = None
        view_924: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_81, [8, 12, 1024, 1024]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_16: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_114, torch.float32);  gt_114 = None
        mul_441: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_442: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_924, mul_441);  view_924 = mul_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_443: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_442, div_37);  mul_442 = None
        sum_53: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [-1], True)
        neg_4: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_2: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_4, sum_53, mul_443);  neg_4 = sum_53 = mul_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_925: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_2, [96, 1024, 1024]);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_82: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_452, view_925);  permute_452 = None
        bmm_83: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_925, permute_453);  view_925 = permute_453 = None
        view_930: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_82, [8, 12, 64, 1024]);  bmm_82 = None
        view_931: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_83, [8, 12, 1024, 64]);  bmm_83 = None
        permute_454: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_930, [0, 1, 3, 2]);  view_930 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_455: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None
        clone_168: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
        view_932: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_168, [8, 1024, 768]);  clone_168 = None
        view_933: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_932, [8192, 768]);  view_932 = None
        permute_456: "f32[768, 8192]" = torch.ops.aten.permute.default(view_933, [1, 0])
        view_790: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_221: "f32[768, 768]" = torch.ops.aten.mm.default(permute_456, view_790);  permute_456 = view_790 = None
        permute_347: "f32[768, 768]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
        permute_458: "f32[768, 768]" = torch.ops.aten.permute.default(permute_347, [1, 0]);  permute_347 = None
        mm_222: "f32[8192, 768]" = torch.ops.aten.mm.default(view_933, permute_458);  view_933 = permute_458 = None
        view_934: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_222, [8, 1024, 768]);  mm_222 = None
        add_188: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_179, view_934);  add_179 = view_934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_460: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_454, [0, 2, 1, 3]);  permute_454 = None
        view_935: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_460, [8, 1024, 768]);  permute_460 = None
        clone_169: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_935, memory_format = torch.contiguous_format);  view_935 = None
        view_936: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_169, [8192, 768]);  clone_169 = None
        permute_461: "f32[768, 8192]" = torch.ops.aten.permute.default(view_936, [1, 0])
        view_787: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_223: "f32[768, 768]" = torch.ops.aten.mm.default(permute_461, view_787);  permute_461 = view_787 = None
        permute_345: "f32[768, 768]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
        permute_463: "f32[768, 768]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
        mm_224: "f32[8192, 768]" = torch.ops.aten.mm.default(view_936, permute_463);  view_936 = permute_463 = None
        view_937: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_224, [8, 1024, 768]);  mm_224 = None
        add_189: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_188, view_937);  add_188 = view_937 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_465: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_931, [0, 2, 1, 3]);  view_931 = None
        clone_170: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
        view_938: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_170, [8, 1024, 768]);  clone_170 = None
        view_939: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_938, [8192, 768]);  view_938 = None
        permute_466: "f32[768, 8192]" = torch.ops.aten.permute.default(view_939, [1, 0])
        mm_225: "f32[768, 768]" = torch.ops.aten.mm.default(permute_466, view_784);  permute_466 = view_784 = None
        permute_343: "f32[768, 768]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
        permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
        mm_226: "f32[8192, 768]" = torch.ops.aten.mm.default(view_939, permute_468);  view_939 = permute_468 = None
        view_940: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_226, [8, 1024, 768]);  mm_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_444: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_940, primals_238);  primals_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_341: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_159, rsqrt_56)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_445: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_940, mul_341);  view_940 = mul_341 = None
        sum_54: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1], True);  mul_445 = None
        view_941: "f32[768]" = torch.ops.aten.reshape.default(sum_54, [768]);  sum_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_446: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_444, add_159)
        mul_447: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_444, rsqrt_56);  mul_444 = None
        sum_55: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True);  mul_446 = None
        add_190: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_187, mul_447);  add_187 = mul_447 = None
        pow_73: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_56, 3);  rsqrt_56 = None
        mul_448: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_55, -0.5);  sum_55 = None
        mul_449: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_448, pow_73);  mul_448 = pow_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_152: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_449, [8, 1024, 768]);  mul_449 = None
        div_47: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_152, 768);  expand_152 = None
        pow_74: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_159, 1.0);  add_159 = None
        mul_450: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
        mul_451: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_47, mul_450);  div_47 = mul_450 = None
        add_191: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_190, mul_451);  add_190 = mul_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_17: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_113, torch.float32);  gt_113 = None
        mul_452: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_453: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_191, mul_452);  mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_942: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_453, [8192, 768]);  mul_453 = None
        permute_470: "f32[768, 8192]" = torch.ops.aten.permute.default(view_942, [1, 0])
        mm_227: "f32[768, 768]" = torch.ops.aten.mm.default(permute_470, view_782);  permute_470 = view_782 = None
        permute_342: "f32[768, 768]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
        permute_472: "f32[768, 768]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
        mm_228: "f32[8192, 768]" = torch.ops.aten.mm.default(view_942, permute_472);  view_942 = permute_472 = None
        view_943: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_228, [8, 1024, 768]);  mm_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_944: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_943, [8, 1024, 12, 64]);  view_943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_474: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_944, [0, 2, 1, 3]);  view_944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_172: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_474, memory_format = torch.contiguous_format);  permute_474 = None
        view_945: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_172, [96, 1024, 64]);  clone_172 = None
        bmm_84: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_475, view_945);  permute_475 = None
        bmm_85: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_945, permute_476);  view_945 = permute_476 = None
        view_946: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_84, [8, 12, 1024, 64]);  bmm_84 = None
        view_947: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_85, [8, 12, 1024, 1024]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_18: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_112, torch.float32);  gt_112 = None
        mul_454: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_455: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_947, mul_454);  view_947 = mul_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_456: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_455, div_36);  mul_455 = None
        sum_56: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [-1], True)
        neg_5: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_36);  div_36 = None
        fma_3: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_5, sum_56, mul_456);  neg_5 = sum_56 = mul_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_948: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_3, [96, 1024, 1024]);  fma_3 = None
        view_950: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_948, [8, 12, 1024, 1024]);  view_948 = None
        view_951: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_950, [96, 1024, 1024])
        add_192: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_899, view_950);  view_899 = view_950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_86: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_477, view_951);  permute_477 = None
        bmm_87: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_951, permute_478);  view_951 = permute_478 = None
        view_953: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_86, [8, 12, 64, 1024]);  bmm_86 = None
        view_954: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_87, [8, 12, 1024, 64]);  bmm_87 = None
        permute_479: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_953, [0, 1, 3, 2]);  view_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_480: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_946, [0, 2, 1, 3]);  view_946 = None
        clone_175: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_480, memory_format = torch.contiguous_format);  permute_480 = None
        view_955: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_175, [8, 1024, 768]);  clone_175 = None
        view_956: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_955, [8192, 768]);  view_955 = None
        permute_481: "f32[768, 8192]" = torch.ops.aten.permute.default(view_956, [1, 0])
        mm_229: "f32[768, 768]" = torch.ops.aten.mm.default(permute_481, view_763);  permute_481 = None
        permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
        permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
        mm_230: "f32[8192, 768]" = torch.ops.aten.mm.default(view_956, permute_483);  view_956 = permute_483 = None
        view_957: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_230, [8, 1024, 768]);  mm_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_485: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_479, [0, 2, 1, 3]);  permute_479 = None
        view_958: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_485, [8, 1024, 768]);  permute_485 = None
        clone_176: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_958, memory_format = torch.contiguous_format);  view_958 = None
        view_959: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_176, [8192, 768]);  clone_176 = None
        permute_486: "f32[768, 8192]" = torch.ops.aten.permute.default(view_959, [1, 0])
        mm_231: "f32[768, 768]" = torch.ops.aten.mm.default(permute_486, view_763);  permute_486 = None
        permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
        permute_488: "f32[768, 768]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
        mm_232: "f32[8192, 768]" = torch.ops.aten.mm.default(view_959, permute_488);  view_959 = permute_488 = None
        view_960: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_232, [8, 1024, 768]);  mm_232 = None
        add_193: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_957, view_960);  view_957 = view_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_490: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_954, [0, 2, 1, 3]);  view_954 = None
        clone_177: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
        view_961: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_177, [8, 1024, 768]);  clone_177 = None
        view_962: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_961, [8192, 768]);  view_961 = None
        permute_491: "f32[768, 8192]" = torch.ops.aten.permute.default(view_962, [1, 0])
        mm_233: "f32[768, 768]" = torch.ops.aten.mm.default(permute_491, view_763);  permute_491 = view_763 = None
        permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
        permute_493: "f32[768, 768]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
        mm_234: "f32[8192, 768]" = torch.ops.aten.mm.default(view_962, permute_493);  view_962 = permute_493 = None
        view_963: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_234, [8, 1024, 768]);  mm_234 = None
        add_194: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_193, view_963);  add_193 = view_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_457: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_194, primals_233);  primals_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_335: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_156, rsqrt_55)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_458: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_194, mul_335);  add_194 = mul_335 = None
        sum_57: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1], True);  mul_458 = None
        view_964: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_459: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_457, add_156)
        mul_460: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_457, rsqrt_55);  mul_457 = None
        sum_58: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True);  mul_459 = None
        add_195: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_191, mul_460);  add_191 = mul_460 = None
        pow_75: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_55, 3);  rsqrt_55 = None
        mul_461: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_58, -0.5);  sum_58 = None
        mul_462: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_461, pow_75);  mul_461 = pow_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_153: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_462, [8, 1024, 768]);  mul_462 = None
        div_48: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_153, 768);  expand_153 = None
        pow_76: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_156, 1.0);  add_156 = None
        mul_463: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_76, 2.0);  pow_76 = None
        mul_464: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_48, mul_463);  div_48 = mul_463 = None
        add_196: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_195, mul_464);  add_195 = mul_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_19: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_111, torch.float32);  gt_111 = None
        mul_465: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_466: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_196, mul_465);  mul_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_965: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_466, [8192, 768]);  mul_466 = None
        permute_495: "f32[768, 8192]" = torch.ops.aten.permute.default(view_965, [1, 0])
        mm_235: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_495, view_761);  permute_495 = view_761 = None
        permute_333: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
        permute_497: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
        mm_236: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_965, permute_497);  view_965 = permute_497 = None
        view_966: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_236, [8, 1024, 3072]);  mm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_110, torch.float32);  gt_110 = None
        mul_467: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_468: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_966, mul_467);  view_966 = mul_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_12: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_3, full_default, mul_468);  le_3 = mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_967: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_12, [8192, 3072]);  where_12 = None
        permute_499: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_967, [1, 0])
        mm_237: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_499, view_759);  permute_499 = view_759 = None
        permute_332: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
        permute_501: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
        mm_238: "f32[8192, 768]" = torch.ops.aten.mm.default(view_967, permute_501);  view_967 = permute_501 = None
        view_968: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_238, [8, 1024, 768]);  mm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_469: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_968, primals_230);  primals_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_329: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_154, rsqrt_54)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_470: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_968, mul_329);  view_968 = mul_329 = None
        sum_59: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 1], True);  mul_470 = None
        view_969: "f32[768]" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_471: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_469, add_154)
        mul_472: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_469, rsqrt_54);  mul_469 = None
        sum_60: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_471, [2], True);  mul_471 = None
        add_197: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_196, mul_472);  add_196 = mul_472 = None
        pow_77: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_54, 3);  rsqrt_54 = None
        mul_473: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_60, -0.5);  sum_60 = None
        mul_474: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_473, pow_77);  mul_473 = pow_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_154: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_474, [8, 1024, 768]);  mul_474 = None
        div_49: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_154, 768);  expand_154 = None
        pow_78: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_154, 1.0);  add_154 = None
        mul_475: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_78, 2.0);  pow_78 = None
        mul_476: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_49, mul_475);  div_49 = mul_475 = None
        add_198: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_197, mul_476);  add_197 = mul_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_21: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_109, torch.float32);  gt_109 = None
        mul_477: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_478: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_198, mul_477);  mul_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_970: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_478, [8192, 768]);  mul_478 = None
        permute_503: "f32[768, 8192]" = torch.ops.aten.permute.default(view_970, [1, 0])
        mm_239: "f32[768, 768]" = torch.ops.aten.mm.default(permute_503, view_757);  permute_503 = view_757 = None
        permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
        permute_505: "f32[768, 768]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
        mm_240: "f32[8192, 768]" = torch.ops.aten.mm.default(view_970, permute_505);  view_970 = permute_505 = None
        view_971: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_240, [8, 1024, 768]);  mm_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_972: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_971, [8, 1024, 12, 64]);  view_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_507: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_972, [0, 2, 1, 3]);  view_972 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_181: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_507, memory_format = torch.contiguous_format);  permute_507 = None
        view_973: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_181, [96, 1024, 64]);  clone_181 = None
        bmm_88: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_508, view_973);  permute_508 = None
        bmm_89: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_973, permute_509);  view_973 = permute_509 = None
        view_974: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_88, [8, 12, 1024, 64]);  bmm_88 = None
        view_975: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_89, [8, 12, 1024, 1024]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_22: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_108, torch.float32);  gt_108 = None
        mul_479: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_480: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_975, mul_479);  view_975 = mul_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_481: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_480, div_35);  mul_480 = None
        sum_61: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_481, [-1], True)
        neg_6: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_35);  div_35 = None
        fma_4: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_6, sum_61, mul_481);  neg_6 = sum_61 = mul_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_976: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_4, [96, 1024, 1024]);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_90: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_510, view_976);  permute_510 = None
        bmm_91: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_976, permute_511);  view_976 = permute_511 = None
        view_981: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_90, [8, 12, 64, 1024]);  bmm_90 = None
        view_982: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_91, [8, 12, 1024, 64]);  bmm_91 = None
        permute_512: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_981, [0, 1, 3, 2]);  view_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_513: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_974, [0, 2, 1, 3]);  view_974 = None
        clone_184: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_513, memory_format = torch.contiguous_format);  permute_513 = None
        view_983: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_184, [8, 1024, 768]);  clone_184 = None
        view_984: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_983, [8192, 768]);  view_983 = None
        permute_514: "f32[768, 8192]" = torch.ops.aten.permute.default(view_984, [1, 0])
        view_744: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_241: "f32[768, 768]" = torch.ops.aten.mm.default(permute_514, view_744);  permute_514 = view_744 = None
        permute_327: "f32[768, 768]" = torch.ops.aten.permute.default(primals_228, [1, 0]);  primals_228 = None
        permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
        mm_242: "f32[8192, 768]" = torch.ops.aten.mm.default(view_984, permute_516);  view_984 = permute_516 = None
        view_985: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_242, [8, 1024, 768]);  mm_242 = None
        add_199: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_189, view_985);  add_189 = view_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_518: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_512, [0, 2, 1, 3]);  permute_512 = None
        view_986: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_518, [8, 1024, 768]);  permute_518 = None
        clone_185: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_986, memory_format = torch.contiguous_format);  view_986 = None
        view_987: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_185, [8192, 768]);  clone_185 = None
        permute_519: "f32[768, 8192]" = torch.ops.aten.permute.default(view_987, [1, 0])
        view_741: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_243: "f32[768, 768]" = torch.ops.aten.mm.default(permute_519, view_741);  permute_519 = view_741 = None
        permute_325: "f32[768, 768]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
        permute_521: "f32[768, 768]" = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
        mm_244: "f32[8192, 768]" = torch.ops.aten.mm.default(view_987, permute_521);  view_987 = permute_521 = None
        view_988: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_244, [8, 1024, 768]);  mm_244 = None
        add_200: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_199, view_988);  add_199 = view_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_523: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_982, [0, 2, 1, 3]);  view_982 = None
        clone_186: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
        view_989: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_186, [8, 1024, 768]);  clone_186 = None
        view_990: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_989, [8192, 768]);  view_989 = None
        permute_524: "f32[768, 8192]" = torch.ops.aten.permute.default(view_990, [1, 0])
        mm_245: "f32[768, 768]" = torch.ops.aten.mm.default(permute_524, view_738);  permute_524 = view_738 = None
        permute_323: "f32[768, 768]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
        permute_526: "f32[768, 768]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
        mm_246: "f32[8192, 768]" = torch.ops.aten.mm.default(view_990, permute_526);  view_990 = permute_526 = None
        view_991: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_246, [8, 1024, 768]);  mm_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_482: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_991, primals_225);  primals_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_323: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_151, rsqrt_53)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_483: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_991, mul_323);  view_991 = mul_323 = None
        sum_62: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 1], True);  mul_483 = None
        view_992: "f32[768]" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_484: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_482, add_151)
        mul_485: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_482, rsqrt_53);  mul_482 = None
        sum_63: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True);  mul_484 = None
        add_201: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_198, mul_485);  add_198 = mul_485 = None
        pow_79: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_53, 3);  rsqrt_53 = None
        mul_486: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_63, -0.5);  sum_63 = None
        mul_487: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_486, pow_79);  mul_486 = pow_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_155: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_487, [8, 1024, 768]);  mul_487 = None
        div_50: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_155, 768);  expand_155 = None
        pow_80: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_151, 1.0);  add_151 = None
        mul_488: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_80, 2.0);  pow_80 = None
        mul_489: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_50, mul_488);  div_50 = mul_488 = None
        add_202: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_201, mul_489);  add_201 = mul_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_23: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_107, torch.float32);  gt_107 = None
        mul_490: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_491: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_202, mul_490);  mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_993: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_491, [8192, 768]);  mul_491 = None
        permute_528: "f32[768, 8192]" = torch.ops.aten.permute.default(view_993, [1, 0])
        mm_247: "f32[768, 768]" = torch.ops.aten.mm.default(permute_528, view_736);  permute_528 = view_736 = None
        permute_322: "f32[768, 768]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
        permute_530: "f32[768, 768]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
        mm_248: "f32[8192, 768]" = torch.ops.aten.mm.default(view_993, permute_530);  view_993 = permute_530 = None
        view_994: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_248, [8, 1024, 768]);  mm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_995: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_994, [8, 1024, 12, 64]);  view_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_532: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_995, [0, 2, 1, 3]);  view_995 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_188: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_532, memory_format = torch.contiguous_format);  permute_532 = None
        view_996: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_188, [96, 1024, 64]);  clone_188 = None
        bmm_92: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_533, view_996);  permute_533 = None
        bmm_93: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_996, permute_534);  view_996 = permute_534 = None
        view_997: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_92, [8, 12, 1024, 64]);  bmm_92 = None
        view_998: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_93, [8, 12, 1024, 1024]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_24: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_106, torch.float32);  gt_106 = None
        mul_492: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_493: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_998, mul_492);  view_998 = mul_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_494: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_493, div_34);  mul_493 = None
        sum_64: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [-1], True)
        neg_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_34);  div_34 = None
        fma_5: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_7, sum_64, mul_494);  neg_7 = sum_64 = mul_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_999: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_5, [96, 1024, 1024]);  fma_5 = None
        view_1001: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_999, [8, 12, 1024, 1024]);  view_999 = None
        view_1002: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1001, [96, 1024, 1024])
        add_203: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_192, view_1001);  add_192 = view_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_94: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_535, view_1002);  permute_535 = None
        bmm_95: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1002, permute_536);  view_1002 = permute_536 = None
        view_1004: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_94, [8, 12, 64, 1024]);  bmm_94 = None
        view_1005: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_95, [8, 12, 1024, 64]);  bmm_95 = None
        permute_537: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1004, [0, 1, 3, 2]);  view_1004 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_538: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_997, [0, 2, 1, 3]);  view_997 = None
        clone_191: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_538, memory_format = torch.contiguous_format);  permute_538 = None
        view_1006: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_191, [8, 1024, 768]);  clone_191 = None
        view_1007: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1006, [8192, 768]);  view_1006 = None
        permute_539: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1007, [1, 0])
        mm_249: "f32[768, 768]" = torch.ops.aten.mm.default(permute_539, view_717);  permute_539 = None
        permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
        permute_541: "f32[768, 768]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
        mm_250: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1007, permute_541);  view_1007 = permute_541 = None
        view_1008: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_250, [8, 1024, 768]);  mm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_543: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_537, [0, 2, 1, 3]);  permute_537 = None
        view_1009: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_543, [8, 1024, 768]);  permute_543 = None
        clone_192: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1009, memory_format = torch.contiguous_format);  view_1009 = None
        view_1010: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_192, [8192, 768]);  clone_192 = None
        permute_544: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1010, [1, 0])
        mm_251: "f32[768, 768]" = torch.ops.aten.mm.default(permute_544, view_717);  permute_544 = None
        permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(primals_222, [1, 0]);  primals_222 = None
        permute_546: "f32[768, 768]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
        mm_252: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1010, permute_546);  view_1010 = permute_546 = None
        view_1011: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_252, [8, 1024, 768]);  mm_252 = None
        add_204: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1008, view_1011);  view_1008 = view_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_548: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1005, [0, 2, 1, 3]);  view_1005 = None
        clone_193: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_548, memory_format = torch.contiguous_format);  permute_548 = None
        view_1012: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_193, [8, 1024, 768]);  clone_193 = None
        view_1013: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1012, [8192, 768]);  view_1012 = None
        permute_549: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1013, [1, 0])
        mm_253: "f32[768, 768]" = torch.ops.aten.mm.default(permute_549, view_717);  permute_549 = view_717 = None
        permute_314: "f32[768, 768]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
        permute_551: "f32[768, 768]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
        mm_254: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1013, permute_551);  view_1013 = permute_551 = None
        view_1014: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_254, [8, 1024, 768]);  mm_254 = None
        add_205: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_204, view_1014);  add_204 = view_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_495: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_205, primals_220);  primals_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_317: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_148, rsqrt_52)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_496: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_205, mul_317);  add_205 = mul_317 = None
        sum_65: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 1], True);  mul_496 = None
        view_1015: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_497: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_495, add_148)
        mul_498: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_495, rsqrt_52);  mul_495 = None
        sum_66: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
        add_206: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_202, mul_498);  add_202 = mul_498 = None
        pow_81: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_52, 3);  rsqrt_52 = None
        mul_499: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_66, -0.5);  sum_66 = None
        mul_500: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_499, pow_81);  mul_499 = pow_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_156: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_500, [8, 1024, 768]);  mul_500 = None
        div_51: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_156, 768);  expand_156 = None
        pow_82: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_148, 1.0);  add_148 = None
        mul_501: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_82, 2.0);  pow_82 = None
        mul_502: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_51, mul_501);  div_51 = mul_501 = None
        add_207: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_206, mul_502);  add_206 = mul_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_25: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_105, torch.float32);  gt_105 = None
        mul_503: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_504: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_207, mul_503);  mul_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1016: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_504, [8192, 768]);  mul_504 = None
        permute_553: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1016, [1, 0])
        mm_255: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_553, view_715);  permute_553 = view_715 = None
        permute_313: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
        permute_555: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
        mm_256: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1016, permute_555);  view_1016 = permute_555 = None
        view_1017: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_256, [8, 1024, 3072]);  mm_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_104, torch.float32);  gt_104 = None
        mul_505: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_506: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1017, mul_505);  view_1017 = mul_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_13: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_4, full_default, mul_506);  le_4 = mul_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1018: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_13, [8192, 3072]);  where_13 = None
        permute_557: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1018, [1, 0])
        mm_257: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_557, view_713);  permute_557 = view_713 = None
        permute_312: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
        permute_559: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
        mm_258: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1018, permute_559);  view_1018 = permute_559 = None
        view_1019: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_258, [8, 1024, 768]);  mm_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_507: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1019, primals_217);  primals_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_311: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_146, rsqrt_51)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_508: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1019, mul_311);  view_1019 = mul_311 = None
        sum_67: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 1], True);  mul_508 = None
        view_1020: "f32[768]" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_509: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_507, add_146)
        mul_510: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_507, rsqrt_51);  mul_507 = None
        sum_68: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True);  mul_509 = None
        add_208: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_207, mul_510);  add_207 = mul_510 = None
        pow_83: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_51, 3);  rsqrt_51 = None
        mul_511: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_68, -0.5);  sum_68 = None
        mul_512: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_511, pow_83);  mul_511 = pow_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_157: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_512, [8, 1024, 768]);  mul_512 = None
        div_52: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_157, 768);  expand_157 = None
        pow_84: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_146, 1.0);  add_146 = None
        mul_513: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
        mul_514: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_52, mul_513);  div_52 = mul_513 = None
        add_209: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_208, mul_514);  add_208 = mul_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_27: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_103, torch.float32);  gt_103 = None
        mul_515: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_516: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_209, mul_515);  mul_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1021: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_516, [8192, 768]);  mul_516 = None
        permute_561: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1021, [1, 0])
        mm_259: "f32[768, 768]" = torch.ops.aten.mm.default(permute_561, view_711);  permute_561 = view_711 = None
        permute_311: "f32[768, 768]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
        permute_563: "f32[768, 768]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
        mm_260: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1021, permute_563);  view_1021 = permute_563 = None
        view_1022: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_260, [8, 1024, 768]);  mm_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1023: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1022, [8, 1024, 12, 64]);  view_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_565: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1023, [0, 2, 1, 3]);  view_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_197: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_565, memory_format = torch.contiguous_format);  permute_565 = None
        view_1024: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_197, [96, 1024, 64]);  clone_197 = None
        bmm_96: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_566, view_1024);  permute_566 = None
        bmm_97: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1024, permute_567);  view_1024 = permute_567 = None
        view_1025: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_96, [8, 12, 1024, 64]);  bmm_96 = None
        view_1026: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_97, [8, 12, 1024, 1024]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_28: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_102, torch.float32);  gt_102 = None
        mul_517: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_518: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1026, mul_517);  view_1026 = mul_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_519: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_518, div_33);  mul_518 = None
        sum_69: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_519, [-1], True)
        neg_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_33);  div_33 = None
        fma_6: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_8, sum_69, mul_519);  neg_8 = sum_69 = mul_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1027: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_6, [96, 1024, 1024]);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_98: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_568, view_1027);  permute_568 = None
        bmm_99: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1027, permute_569);  view_1027 = permute_569 = None
        view_1032: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_98, [8, 12, 64, 1024]);  bmm_98 = None
        view_1033: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_99, [8, 12, 1024, 64]);  bmm_99 = None
        permute_570: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1032, [0, 1, 3, 2]);  view_1032 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_571: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1025, [0, 2, 1, 3]);  view_1025 = None
        clone_200: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_571, memory_format = torch.contiguous_format);  permute_571 = None
        view_1034: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_200, [8, 1024, 768]);  clone_200 = None
        view_1035: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1034, [8192, 768]);  view_1034 = None
        permute_572: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1035, [1, 0])
        view_698: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_261: "f32[768, 768]" = torch.ops.aten.mm.default(permute_572, view_698);  permute_572 = view_698 = None
        permute_307: "f32[768, 768]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
        permute_574: "f32[768, 768]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
        mm_262: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1035, permute_574);  view_1035 = permute_574 = None
        view_1036: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_262, [8, 1024, 768]);  mm_262 = None
        add_210: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_200, view_1036);  add_200 = view_1036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_576: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_570, [0, 2, 1, 3]);  permute_570 = None
        view_1037: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_576, [8, 1024, 768]);  permute_576 = None
        clone_201: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1037, memory_format = torch.contiguous_format);  view_1037 = None
        view_1038: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_201, [8192, 768]);  clone_201 = None
        permute_577: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1038, [1, 0])
        view_695: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_263: "f32[768, 768]" = torch.ops.aten.mm.default(permute_577, view_695);  permute_577 = view_695 = None
        permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
        permute_579: "f32[768, 768]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        mm_264: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1038, permute_579);  view_1038 = permute_579 = None
        view_1039: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_264, [8, 1024, 768]);  mm_264 = None
        add_211: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_210, view_1039);  add_210 = view_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_581: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1033, [0, 2, 1, 3]);  view_1033 = None
        clone_202: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_581, memory_format = torch.contiguous_format);  permute_581 = None
        view_1040: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_202, [8, 1024, 768]);  clone_202 = None
        view_1041: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1040, [8192, 768]);  view_1040 = None
        permute_582: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1041, [1, 0])
        mm_265: "f32[768, 768]" = torch.ops.aten.mm.default(permute_582, view_692);  permute_582 = view_692 = None
        permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
        permute_584: "f32[768, 768]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
        mm_266: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1041, permute_584);  view_1041 = permute_584 = None
        view_1042: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_266, [8, 1024, 768]);  mm_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_520: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1042, primals_212);  primals_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_305: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_143, rsqrt_50)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_521: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1042, mul_305);  view_1042 = mul_305 = None
        sum_70: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 1], True);  mul_521 = None
        view_1043: "f32[768]" = torch.ops.aten.reshape.default(sum_70, [768]);  sum_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_522: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_520, add_143)
        mul_523: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_520, rsqrt_50);  mul_520 = None
        sum_71: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True);  mul_522 = None
        add_212: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_209, mul_523);  add_209 = mul_523 = None
        pow_85: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_50, 3);  rsqrt_50 = None
        mul_524: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_71, -0.5);  sum_71 = None
        mul_525: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_524, pow_85);  mul_524 = pow_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_158: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_525, [8, 1024, 768]);  mul_525 = None
        div_53: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_158, 768);  expand_158 = None
        pow_86: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_143, 1.0);  add_143 = None
        mul_526: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
        mul_527: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_53, mul_526);  div_53 = mul_526 = None
        add_213: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_212, mul_527);  add_212 = mul_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_29: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_101, torch.float32);  gt_101 = None
        mul_528: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_529: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_213, mul_528);  mul_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1044: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_529, [8192, 768]);  mul_529 = None
        permute_586: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1044, [1, 0])
        mm_267: "f32[768, 768]" = torch.ops.aten.mm.default(permute_586, view_690);  permute_586 = view_690 = None
        permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
        permute_588: "f32[768, 768]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
        mm_268: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1044, permute_588);  view_1044 = permute_588 = None
        view_1045: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_268, [8, 1024, 768]);  mm_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1046: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1045, [8, 1024, 12, 64]);  view_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_590: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1046, [0, 2, 1, 3]);  view_1046 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_204: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_590, memory_format = torch.contiguous_format);  permute_590 = None
        view_1047: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_204, [96, 1024, 64]);  clone_204 = None
        bmm_100: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_591, view_1047);  permute_591 = None
        bmm_101: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1047, permute_592);  view_1047 = permute_592 = None
        view_1048: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_100, [8, 12, 1024, 64]);  bmm_100 = None
        view_1049: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_101, [8, 12, 1024, 1024]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_30: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_100, torch.float32);  gt_100 = None
        mul_530: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_531: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1049, mul_530);  view_1049 = mul_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_532: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_531, div_32);  mul_531 = None
        sum_72: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [-1], True)
        neg_9: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_32);  div_32 = None
        fma_7: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_9, sum_72, mul_532);  neg_9 = sum_72 = mul_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1050: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_7, [96, 1024, 1024]);  fma_7 = None
        view_1052: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1050, [8, 12, 1024, 1024]);  view_1050 = None
        view_1053: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1052, [96, 1024, 1024])
        add_214: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_203, view_1052);  add_203 = view_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_102: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_593, view_1053);  permute_593 = None
        bmm_103: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1053, permute_594);  view_1053 = permute_594 = None
        view_1055: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_102, [8, 12, 64, 1024]);  bmm_102 = None
        view_1056: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_103, [8, 12, 1024, 64]);  bmm_103 = None
        permute_595: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1055, [0, 1, 3, 2]);  view_1055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_596: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1048, [0, 2, 1, 3]);  view_1048 = None
        clone_207: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_596, memory_format = torch.contiguous_format);  permute_596 = None
        view_1057: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_207, [8, 1024, 768]);  clone_207 = None
        view_1058: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1057, [8192, 768]);  view_1057 = None
        permute_597: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1058, [1, 0])
        mm_269: "f32[768, 768]" = torch.ops.aten.mm.default(permute_597, view_671);  permute_597 = None
        permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
        permute_599: "f32[768, 768]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
        mm_270: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1058, permute_599);  view_1058 = permute_599 = None
        view_1059: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_270, [8, 1024, 768]);  mm_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_601: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_595, [0, 2, 1, 3]);  permute_595 = None
        view_1060: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_601, [8, 1024, 768]);  permute_601 = None
        clone_208: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1060, memory_format = torch.contiguous_format);  view_1060 = None
        view_1061: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_208, [8192, 768]);  clone_208 = None
        permute_602: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1061, [1, 0])
        mm_271: "f32[768, 768]" = torch.ops.aten.mm.default(permute_602, view_671);  permute_602 = None
        permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
        permute_604: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
        mm_272: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1061, permute_604);  view_1061 = permute_604 = None
        view_1062: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_272, [8, 1024, 768]);  mm_272 = None
        add_215: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1059, view_1062);  view_1059 = view_1062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_606: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1056, [0, 2, 1, 3]);  view_1056 = None
        clone_209: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_606, memory_format = torch.contiguous_format);  permute_606 = None
        view_1063: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_209, [8, 1024, 768]);  clone_209 = None
        view_1064: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1063, [8192, 768]);  view_1063 = None
        permute_607: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1064, [1, 0])
        mm_273: "f32[768, 768]" = torch.ops.aten.mm.default(permute_607, view_671);  permute_607 = view_671 = None
        permute_294: "f32[768, 768]" = torch.ops.aten.permute.default(primals_208, [1, 0]);  primals_208 = None
        permute_609: "f32[768, 768]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
        mm_274: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1064, permute_609);  view_1064 = permute_609 = None
        view_1065: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_274, [8, 1024, 768]);  mm_274 = None
        add_216: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_215, view_1065);  add_215 = view_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_533: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_216, primals_207);  primals_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_299: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_140, rsqrt_49)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_534: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_216, mul_299);  add_216 = mul_299 = None
        sum_73: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_534, [0, 1], True);  mul_534 = None
        view_1066: "f32[768]" = torch.ops.aten.reshape.default(sum_73, [768]);  sum_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_535: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_533, add_140)
        mul_536: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_533, rsqrt_49);  mul_533 = None
        sum_74: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_535, [2], True);  mul_535 = None
        add_217: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_213, mul_536);  add_213 = mul_536 = None
        pow_87: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_49, 3);  rsqrt_49 = None
        mul_537: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_74, -0.5);  sum_74 = None
        mul_538: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_537, pow_87);  mul_537 = pow_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_159: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_538, [8, 1024, 768]);  mul_538 = None
        div_54: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_159, 768);  expand_159 = None
        pow_88: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_140, 1.0);  add_140 = None
        mul_539: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_88, 2.0);  pow_88 = None
        mul_540: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_54, mul_539);  div_54 = mul_539 = None
        add_218: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_217, mul_540);  add_217 = mul_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_31: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_99, torch.float32);  gt_99 = None
        mul_541: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_542: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_218, mul_541);  mul_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1067: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_542, [8192, 768]);  mul_542 = None
        permute_611: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1067, [1, 0])
        mm_275: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_611, view_669);  permute_611 = view_669 = None
        permute_293: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
        permute_613: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
        mm_276: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1067, permute_613);  view_1067 = permute_613 = None
        view_1068: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_276, [8, 1024, 3072]);  mm_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_98, torch.float32);  gt_98 = None
        mul_543: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_544: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1068, mul_543);  view_1068 = mul_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_14: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_5, full_default, mul_544);  le_5 = mul_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1069: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_14, [8192, 3072]);  where_14 = None
        permute_615: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1069, [1, 0])
        mm_277: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_615, view_667);  permute_615 = view_667 = None
        permute_292: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
        permute_617: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
        mm_278: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1069, permute_617);  view_1069 = permute_617 = None
        view_1070: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_278, [8, 1024, 768]);  mm_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_545: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1070, primals_204);  primals_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_293: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_138, rsqrt_48)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_546: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1070, mul_293);  view_1070 = mul_293 = None
        sum_75: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 1], True);  mul_546 = None
        view_1071: "f32[768]" = torch.ops.aten.reshape.default(sum_75, [768]);  sum_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_547: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_545, add_138)
        mul_548: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_545, rsqrt_48);  mul_545 = None
        sum_76: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_547, [2], True);  mul_547 = None
        add_219: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_218, mul_548);  add_218 = mul_548 = None
        pow_89: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_48, 3);  rsqrt_48 = None
        mul_549: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_76, -0.5);  sum_76 = None
        mul_550: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_549, pow_89);  mul_549 = pow_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_160: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_550, [8, 1024, 768]);  mul_550 = None
        div_55: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_160, 768);  expand_160 = None
        pow_90: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_138, 1.0);  add_138 = None
        mul_551: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_90, 2.0);  pow_90 = None
        mul_552: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_55, mul_551);  div_55 = mul_551 = None
        add_220: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_219, mul_552);  add_219 = mul_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_33: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_97, torch.float32);  gt_97 = None
        mul_553: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_554: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_220, mul_553);  mul_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1072: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_554, [8192, 768]);  mul_554 = None
        permute_619: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1072, [1, 0])
        mm_279: "f32[768, 768]" = torch.ops.aten.mm.default(permute_619, view_665);  permute_619 = view_665 = None
        permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
        permute_621: "f32[768, 768]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
        mm_280: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1072, permute_621);  view_1072 = permute_621 = None
        view_1073: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_280, [8, 1024, 768]);  mm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1074: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1073, [8, 1024, 12, 64]);  view_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_623: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1074, [0, 2, 1, 3]);  view_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_213: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_623, memory_format = torch.contiguous_format);  permute_623 = None
        view_1075: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_213, [96, 1024, 64]);  clone_213 = None
        bmm_104: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_624, view_1075);  permute_624 = None
        bmm_105: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1075, permute_625);  view_1075 = permute_625 = None
        view_1076: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_104, [8, 12, 1024, 64]);  bmm_104 = None
        view_1077: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_105, [8, 12, 1024, 1024]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_34: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_96, torch.float32);  gt_96 = None
        mul_555: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_556: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1077, mul_555);  view_1077 = mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_557: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_556, div_31);  mul_556 = None
        sum_77: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [-1], True)
        neg_10: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_31);  div_31 = None
        fma_8: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_10, sum_77, mul_557);  neg_10 = sum_77 = mul_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1078: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_8, [96, 1024, 1024]);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_106: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_626, view_1078);  permute_626 = None
        bmm_107: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1078, permute_627);  view_1078 = permute_627 = None
        view_1083: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_106, [8, 12, 64, 1024]);  bmm_106 = None
        view_1084: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_107, [8, 12, 1024, 64]);  bmm_107 = None
        permute_628: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1083, [0, 1, 3, 2]);  view_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_629: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1076, [0, 2, 1, 3]);  view_1076 = None
        clone_216: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
        view_1085: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_216, [8, 1024, 768]);  clone_216 = None
        view_1086: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1085, [8192, 768]);  view_1085 = None
        permute_630: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1086, [1, 0])
        view_652: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_281: "f32[768, 768]" = torch.ops.aten.mm.default(permute_630, view_652);  permute_630 = view_652 = None
        permute_287: "f32[768, 768]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
        permute_632: "f32[768, 768]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
        mm_282: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1086, permute_632);  view_1086 = permute_632 = None
        view_1087: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_282, [8, 1024, 768]);  mm_282 = None
        add_221: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_211, view_1087);  add_211 = view_1087 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_634: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_628, [0, 2, 1, 3]);  permute_628 = None
        view_1088: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_634, [8, 1024, 768]);  permute_634 = None
        clone_217: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1088, memory_format = torch.contiguous_format);  view_1088 = None
        view_1089: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_217, [8192, 768]);  clone_217 = None
        permute_635: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1089, [1, 0])
        view_649: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_283: "f32[768, 768]" = torch.ops.aten.mm.default(permute_635, view_649);  permute_635 = view_649 = None
        permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
        permute_637: "f32[768, 768]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
        mm_284: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1089, permute_637);  view_1089 = permute_637 = None
        view_1090: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_284, [8, 1024, 768]);  mm_284 = None
        add_222: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_221, view_1090);  add_221 = view_1090 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_639: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1084, [0, 2, 1, 3]);  view_1084 = None
        clone_218: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_639, memory_format = torch.contiguous_format);  permute_639 = None
        view_1091: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_218, [8, 1024, 768]);  clone_218 = None
        view_1092: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1091, [8192, 768]);  view_1091 = None
        permute_640: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1092, [1, 0])
        mm_285: "f32[768, 768]" = torch.ops.aten.mm.default(permute_640, view_646);  permute_640 = view_646 = None
        permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
        permute_642: "f32[768, 768]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
        mm_286: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1092, permute_642);  view_1092 = permute_642 = None
        view_1093: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_286, [8, 1024, 768]);  mm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_558: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1093, primals_199);  primals_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_287: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_135, rsqrt_47)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_559: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1093, mul_287);  view_1093 = mul_287 = None
        sum_78: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 1], True);  mul_559 = None
        view_1094: "f32[768]" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_560: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_558, add_135)
        mul_561: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_558, rsqrt_47);  mul_558 = None
        sum_79: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True);  mul_560 = None
        add_223: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_220, mul_561);  add_220 = mul_561 = None
        pow_91: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_47, 3);  rsqrt_47 = None
        mul_562: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_79, -0.5);  sum_79 = None
        mul_563: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_562, pow_91);  mul_562 = pow_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_161: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_563, [8, 1024, 768]);  mul_563 = None
        div_56: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_161, 768);  expand_161 = None
        pow_92: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_135, 1.0);  add_135 = None
        mul_564: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_92, 2.0);  pow_92 = None
        mul_565: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_56, mul_564);  div_56 = mul_564 = None
        add_224: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_223, mul_565);  add_223 = mul_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_35: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_95, torch.float32);  gt_95 = None
        mul_566: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_567: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_224, mul_566);  mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1095: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_567, [8192, 768]);  mul_567 = None
        permute_644: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1095, [1, 0])
        mm_287: "f32[768, 768]" = torch.ops.aten.mm.default(permute_644, view_644);  permute_644 = view_644 = None
        permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        permute_646: "f32[768, 768]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
        mm_288: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1095, permute_646);  view_1095 = permute_646 = None
        view_1096: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_288, [8, 1024, 768]);  mm_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1097: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1096, [8, 1024, 12, 64]);  view_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_648: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1097, [0, 2, 1, 3]);  view_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_220: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_648, memory_format = torch.contiguous_format);  permute_648 = None
        view_1098: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_220, [96, 1024, 64]);  clone_220 = None
        bmm_108: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_649, view_1098);  permute_649 = None
        bmm_109: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1098, permute_650);  view_1098 = permute_650 = None
        view_1099: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_108, [8, 12, 1024, 64]);  bmm_108 = None
        view_1100: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_109, [8, 12, 1024, 1024]);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_36: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_94, torch.float32);  gt_94 = None
        mul_568: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_569: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1100, mul_568);  view_1100 = mul_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_570: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_569, div_30);  mul_569 = None
        sum_80: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [-1], True)
        neg_11: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_30);  div_30 = None
        fma_9: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_11, sum_80, mul_570);  neg_11 = sum_80 = mul_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1101: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_9, [96, 1024, 1024]);  fma_9 = None
        view_1103: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1101, [8, 12, 1024, 1024]);  view_1101 = None
        view_1104: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1103, [96, 1024, 1024])
        add_225: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_214, view_1103);  add_214 = view_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_110: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_651, view_1104);  permute_651 = None
        bmm_111: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1104, permute_652);  view_1104 = permute_652 = None
        view_1106: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_110, [8, 12, 64, 1024]);  bmm_110 = None
        view_1107: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_111, [8, 12, 1024, 64]);  bmm_111 = None
        permute_653: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1106, [0, 1, 3, 2]);  view_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_654: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1099, [0, 2, 1, 3]);  view_1099 = None
        clone_223: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_654, memory_format = torch.contiguous_format);  permute_654 = None
        view_1108: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_223, [8, 1024, 768]);  clone_223 = None
        view_1109: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1108, [8192, 768]);  view_1108 = None
        permute_655: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1109, [1, 0])
        mm_289: "f32[768, 768]" = torch.ops.aten.mm.default(permute_655, view_625);  permute_655 = None
        permute_278: "f32[768, 768]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
        permute_657: "f32[768, 768]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
        mm_290: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1109, permute_657);  view_1109 = permute_657 = None
        view_1110: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_290, [8, 1024, 768]);  mm_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_659: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_653, [0, 2, 1, 3]);  permute_653 = None
        view_1111: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_659, [8, 1024, 768]);  permute_659 = None
        clone_224: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1111, memory_format = torch.contiguous_format);  view_1111 = None
        view_1112: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_224, [8192, 768]);  clone_224 = None
        permute_660: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1112, [1, 0])
        mm_291: "f32[768, 768]" = torch.ops.aten.mm.default(permute_660, view_625);  permute_660 = None
        permute_276: "f32[768, 768]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
        permute_662: "f32[768, 768]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
        mm_292: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1112, permute_662);  view_1112 = permute_662 = None
        view_1113: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_292, [8, 1024, 768]);  mm_292 = None
        add_226: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1110, view_1113);  view_1110 = view_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_664: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1107, [0, 2, 1, 3]);  view_1107 = None
        clone_225: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_664, memory_format = torch.contiguous_format);  permute_664 = None
        view_1114: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_225, [8, 1024, 768]);  clone_225 = None
        view_1115: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1114, [8192, 768]);  view_1114 = None
        permute_665: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1115, [1, 0])
        mm_293: "f32[768, 768]" = torch.ops.aten.mm.default(permute_665, view_625);  permute_665 = view_625 = None
        permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
        permute_667: "f32[768, 768]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
        mm_294: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1115, permute_667);  view_1115 = permute_667 = None
        view_1116: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_294, [8, 1024, 768]);  mm_294 = None
        add_227: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_226, view_1116);  add_226 = view_1116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_571: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_227, primals_194);  primals_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_281: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_132, rsqrt_46)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_572: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_227, mul_281);  add_227 = mul_281 = None
        sum_81: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 1], True);  mul_572 = None
        view_1117: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_573: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_571, add_132)
        mul_574: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_571, rsqrt_46);  mul_571 = None
        sum_82: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True);  mul_573 = None
        add_228: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_224, mul_574);  add_224 = mul_574 = None
        pow_93: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_46, 3);  rsqrt_46 = None
        mul_575: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_82, -0.5);  sum_82 = None
        mul_576: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_575, pow_93);  mul_575 = pow_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_162: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_576, [8, 1024, 768]);  mul_576 = None
        div_57: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_162, 768);  expand_162 = None
        pow_94: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_132, 1.0);  add_132 = None
        mul_577: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_94, 2.0);  pow_94 = None
        mul_578: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_57, mul_577);  div_57 = mul_577 = None
        add_229: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_228, mul_578);  add_228 = mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_37: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_93, torch.float32);  gt_93 = None
        mul_579: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_580: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_229, mul_579);  mul_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1118: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_580, [8192, 768]);  mul_580 = None
        permute_669: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1118, [1, 0])
        mm_295: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_669, view_623);  permute_669 = view_623 = None
        permute_273: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
        permute_671: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
        mm_296: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1118, permute_671);  view_1118 = permute_671 = None
        view_1119: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_296, [8, 1024, 3072]);  mm_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_38: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_92, torch.float32);  gt_92 = None
        mul_581: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_582: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1119, mul_581);  view_1119 = mul_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_15: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_6, full_default, mul_582);  le_6 = mul_582 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1120: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_15, [8192, 3072]);  where_15 = None
        permute_673: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1120, [1, 0])
        mm_297: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_673, view_621);  permute_673 = view_621 = None
        permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        permute_675: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
        mm_298: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1120, permute_675);  view_1120 = permute_675 = None
        view_1121: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_298, [8, 1024, 768]);  mm_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_583: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1121, primals_191);  primals_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_275: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_130, rsqrt_45)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_584: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1121, mul_275);  view_1121 = mul_275 = None
        sum_83: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1], True);  mul_584 = None
        view_1122: "f32[768]" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_585: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_583, add_130)
        mul_586: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_583, rsqrt_45);  mul_583 = None
        sum_84: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_585, [2], True);  mul_585 = None
        add_230: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_229, mul_586);  add_229 = mul_586 = None
        pow_95: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_45, 3);  rsqrt_45 = None
        mul_587: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_84, -0.5);  sum_84 = None
        mul_588: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_587, pow_95);  mul_587 = pow_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_163: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_588, [8, 1024, 768]);  mul_588 = None
        div_58: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_163, 768);  expand_163 = None
        pow_96: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_130, 1.0);  add_130 = None
        mul_589: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_96, 2.0);  pow_96 = None
        mul_590: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_58, mul_589);  div_58 = mul_589 = None
        add_231: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_230, mul_590);  add_230 = mul_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_39: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_91, torch.float32);  gt_91 = None
        mul_591: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
        mul_592: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_231, mul_591);  mul_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1123: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_592, [8192, 768]);  mul_592 = None
        permute_677: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1123, [1, 0])
        mm_299: "f32[768, 768]" = torch.ops.aten.mm.default(permute_677, view_619);  permute_677 = view_619 = None
        permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_679: "f32[768, 768]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
        mm_300: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1123, permute_679);  view_1123 = permute_679 = None
        view_1124: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_300, [8, 1024, 768]);  mm_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1125: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1124, [8, 1024, 12, 64]);  view_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_681: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1125, [0, 2, 1, 3]);  view_1125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_229: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_681, memory_format = torch.contiguous_format);  permute_681 = None
        view_1126: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_229, [96, 1024, 64]);  clone_229 = None
        bmm_112: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_682, view_1126);  permute_682 = None
        bmm_113: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1126, permute_683);  view_1126 = permute_683 = None
        view_1127: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_112, [8, 12, 1024, 64]);  bmm_112 = None
        view_1128: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_113, [8, 12, 1024, 1024]);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_40: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_90, torch.float32);  gt_90 = None
        mul_593: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
        mul_594: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1128, mul_593);  view_1128 = mul_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_595: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_594, div_29);  mul_594 = None
        sum_85: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [-1], True)
        neg_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_29);  div_29 = None
        fma_10: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_12, sum_85, mul_595);  neg_12 = sum_85 = mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1129: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_10, [96, 1024, 1024]);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_114: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_684, view_1129);  permute_684 = None
        bmm_115: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1129, permute_685);  view_1129 = permute_685 = None
        view_1134: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_114, [8, 12, 64, 1024]);  bmm_114 = None
        view_1135: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_115, [8, 12, 1024, 64]);  bmm_115 = None
        permute_686: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1134, [0, 1, 3, 2]);  view_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_687: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1127, [0, 2, 1, 3]);  view_1127 = None
        clone_232: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_687, memory_format = torch.contiguous_format);  permute_687 = None
        view_1136: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_232, [8, 1024, 768]);  clone_232 = None
        view_1137: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1136, [8192, 768]);  view_1136 = None
        permute_688: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1137, [1, 0])
        view_606: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_301: "f32[768, 768]" = torch.ops.aten.mm.default(permute_688, view_606);  permute_688 = view_606 = None
        permute_267: "f32[768, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        permute_690: "f32[768, 768]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
        mm_302: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1137, permute_690);  view_1137 = permute_690 = None
        view_1138: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_302, [8, 1024, 768]);  mm_302 = None
        add_232: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_222, view_1138);  add_222 = view_1138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_692: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_686, [0, 2, 1, 3]);  permute_686 = None
        view_1139: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_692, [8, 1024, 768]);  permute_692 = None
        clone_233: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1139, memory_format = torch.contiguous_format);  view_1139 = None
        view_1140: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_233, [8192, 768]);  clone_233 = None
        permute_693: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1140, [1, 0])
        view_603: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_303: "f32[768, 768]" = torch.ops.aten.mm.default(permute_693, view_603);  permute_693 = view_603 = None
        permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        permute_695: "f32[768, 768]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        mm_304: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1140, permute_695);  view_1140 = permute_695 = None
        view_1141: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_304, [8, 1024, 768]);  mm_304 = None
        add_233: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_232, view_1141);  add_232 = view_1141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_697: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1135, [0, 2, 1, 3]);  view_1135 = None
        clone_234: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_697, memory_format = torch.contiguous_format);  permute_697 = None
        view_1142: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_234, [8, 1024, 768]);  clone_234 = None
        view_1143: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1142, [8192, 768]);  view_1142 = None
        permute_698: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1143, [1, 0])
        mm_305: "f32[768, 768]" = torch.ops.aten.mm.default(permute_698, view_600);  permute_698 = view_600 = None
        permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
        permute_700: "f32[768, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        mm_306: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1143, permute_700);  view_1143 = permute_700 = None
        view_1144: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_306, [8, 1024, 768]);  mm_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_596: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1144, primals_186);  primals_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_269: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_127, rsqrt_44)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_597: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1144, mul_269);  view_1144 = mul_269 = None
        sum_86: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 1], True);  mul_597 = None
        view_1145: "f32[768]" = torch.ops.aten.reshape.default(sum_86, [768]);  sum_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_598: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_596, add_127)
        mul_599: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_596, rsqrt_44);  mul_596 = None
        sum_87: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_598, [2], True);  mul_598 = None
        add_234: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_231, mul_599);  add_231 = mul_599 = None
        pow_97: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_44, 3);  rsqrt_44 = None
        mul_600: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_87, -0.5);  sum_87 = None
        mul_601: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_600, pow_97);  mul_600 = pow_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_164: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_601, [8, 1024, 768]);  mul_601 = None
        div_59: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_164, 768);  expand_164 = None
        pow_98: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_127, 1.0);  add_127 = None
        mul_602: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_98, 2.0);  pow_98 = None
        mul_603: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_59, mul_602);  div_59 = mul_602 = None
        add_235: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_234, mul_603);  add_234 = mul_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_41: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_89, torch.float32);  gt_89 = None
        mul_604: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
        mul_605: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_235, mul_604);  mul_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1146: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_605, [8192, 768]);  mul_605 = None
        permute_702: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1146, [1, 0])
        mm_307: "f32[768, 768]" = torch.ops.aten.mm.default(permute_702, view_598);  permute_702 = view_598 = None
        permute_262: "f32[768, 768]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        permute_704: "f32[768, 768]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        mm_308: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1146, permute_704);  view_1146 = permute_704 = None
        view_1147: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_308, [8, 1024, 768]);  mm_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1148: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1147, [8, 1024, 12, 64]);  view_1147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_706: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1148, [0, 2, 1, 3]);  view_1148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_236: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_706, memory_format = torch.contiguous_format);  permute_706 = None
        view_1149: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_236, [96, 1024, 64]);  clone_236 = None
        bmm_116: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_707, view_1149);  permute_707 = None
        bmm_117: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1149, permute_708);  view_1149 = permute_708 = None
        view_1150: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_116, [8, 12, 1024, 64]);  bmm_116 = None
        view_1151: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_117, [8, 12, 1024, 1024]);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_42: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_88, torch.float32);  gt_88 = None
        mul_606: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
        mul_607: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1151, mul_606);  view_1151 = mul_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_608: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_607, div_28);  mul_607 = None
        sum_88: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [-1], True)
        neg_13: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_28);  div_28 = None
        fma_11: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_13, sum_88, mul_608);  neg_13 = sum_88 = mul_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1152: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_11, [96, 1024, 1024]);  fma_11 = None
        view_1154: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1152, [8, 12, 1024, 1024]);  view_1152 = None
        view_1155: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1154, [96, 1024, 1024])
        add_236: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_225, view_1154);  add_225 = view_1154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_118: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_709, view_1155);  permute_709 = None
        bmm_119: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1155, permute_710);  view_1155 = permute_710 = None
        view_1157: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_118, [8, 12, 64, 1024]);  bmm_118 = None
        view_1158: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_119, [8, 12, 1024, 64]);  bmm_119 = None
        permute_711: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1157, [0, 1, 3, 2]);  view_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_712: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1150, [0, 2, 1, 3]);  view_1150 = None
        clone_239: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
        view_1159: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_239, [8, 1024, 768]);  clone_239 = None
        view_1160: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1159, [8192, 768]);  view_1159 = None
        permute_713: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1160, [1, 0])
        mm_309: "f32[768, 768]" = torch.ops.aten.mm.default(permute_713, view_579);  permute_713 = None
        permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_715: "f32[768, 768]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
        mm_310: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1160, permute_715);  view_1160 = permute_715 = None
        view_1161: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_310, [8, 1024, 768]);  mm_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_717: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_711, [0, 2, 1, 3]);  permute_711 = None
        view_1162: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_717, [8, 1024, 768]);  permute_717 = None
        clone_240: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1162, memory_format = torch.contiguous_format);  view_1162 = None
        view_1163: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_240, [8192, 768]);  clone_240 = None
        permute_718: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1163, [1, 0])
        mm_311: "f32[768, 768]" = torch.ops.aten.mm.default(permute_718, view_579);  permute_718 = None
        permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
        permute_720: "f32[768, 768]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
        mm_312: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1163, permute_720);  view_1163 = permute_720 = None
        view_1164: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_312, [8, 1024, 768]);  mm_312 = None
        add_237: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1161, view_1164);  view_1161 = view_1164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_722: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1158, [0, 2, 1, 3]);  view_1158 = None
        clone_241: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_722, memory_format = torch.contiguous_format);  permute_722 = None
        view_1165: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_241, [8, 1024, 768]);  clone_241 = None
        view_1166: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1165, [8192, 768]);  view_1165 = None
        permute_723: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1166, [1, 0])
        mm_313: "f32[768, 768]" = torch.ops.aten.mm.default(permute_723, view_579);  permute_723 = view_579 = None
        permute_254: "f32[768, 768]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        permute_725: "f32[768, 768]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        mm_314: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1166, permute_725);  view_1166 = permute_725 = None
        view_1167: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_314, [8, 1024, 768]);  mm_314 = None
        add_238: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_237, view_1167);  add_237 = view_1167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_609: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_238, primals_181);  primals_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_263: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_124, rsqrt_43)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_610: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_238, mul_263);  add_238 = mul_263 = None
        sum_89: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 1], True);  mul_610 = None
        view_1168: "f32[768]" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_611: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_609, add_124)
        mul_612: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_609, rsqrt_43);  mul_609 = None
        sum_90: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_611, [2], True);  mul_611 = None
        add_239: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_235, mul_612);  add_235 = mul_612 = None
        pow_99: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_43, 3);  rsqrt_43 = None
        mul_613: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_90, -0.5);  sum_90 = None
        mul_614: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_613, pow_99);  mul_613 = pow_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_165: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_614, [8, 1024, 768]);  mul_614 = None
        div_60: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_165, 768);  expand_165 = None
        pow_100: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_124, 1.0);  add_124 = None
        mul_615: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_100, 2.0);  pow_100 = None
        mul_616: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_60, mul_615);  div_60 = mul_615 = None
        add_240: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_239, mul_616);  add_239 = mul_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_43: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_87, torch.float32);  gt_87 = None
        mul_617: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
        mul_618: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_240, mul_617);  mul_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1169: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_618, [8192, 768]);  mul_618 = None
        permute_727: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1169, [1, 0])
        mm_315: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_727, view_577);  permute_727 = view_577 = None
        permute_253: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
        permute_729: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        mm_316: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1169, permute_729);  view_1169 = permute_729 = None
        view_1170: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_316, [8, 1024, 3072]);  mm_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_44: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_86, torch.float32);  gt_86 = None
        mul_619: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
        mul_620: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1170, mul_619);  view_1170 = mul_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_16: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_7, full_default, mul_620);  le_7 = mul_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1171: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_16, [8192, 3072]);  where_16 = None
        permute_731: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1171, [1, 0])
        mm_317: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_731, view_575);  permute_731 = view_575 = None
        permute_252: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
        permute_733: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        mm_318: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1171, permute_733);  view_1171 = permute_733 = None
        view_1172: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_318, [8, 1024, 768]);  mm_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_621: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1172, primals_178);  primals_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_257: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_122, rsqrt_42)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_622: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1172, mul_257);  view_1172 = mul_257 = None
        sum_91: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 1], True);  mul_622 = None
        view_1173: "f32[768]" = torch.ops.aten.reshape.default(sum_91, [768]);  sum_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_623: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_621, add_122)
        mul_624: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_621, rsqrt_42);  mul_621 = None
        sum_92: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True);  mul_623 = None
        add_241: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_240, mul_624);  add_240 = mul_624 = None
        pow_101: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_42, 3);  rsqrt_42 = None
        mul_625: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_92, -0.5);  sum_92 = None
        mul_626: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_625, pow_101);  mul_625 = pow_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_166: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_626, [8, 1024, 768]);  mul_626 = None
        div_61: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_166, 768);  expand_166 = None
        pow_102: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_122, 1.0);  add_122 = None
        mul_627: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_102, 2.0);  pow_102 = None
        mul_628: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_61, mul_627);  div_61 = mul_627 = None
        add_242: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_241, mul_628);  add_241 = mul_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_45: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_85, torch.float32);  gt_85 = None
        mul_629: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
        mul_630: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_242, mul_629);  mul_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1174: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_630, [8192, 768]);  mul_630 = None
        permute_735: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1174, [1, 0])
        mm_319: "f32[768, 768]" = torch.ops.aten.mm.default(permute_735, view_573);  permute_735 = view_573 = None
        permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
        permute_737: "f32[768, 768]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        mm_320: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1174, permute_737);  view_1174 = permute_737 = None
        view_1175: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_320, [8, 1024, 768]);  mm_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1176: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1175, [8, 1024, 12, 64]);  view_1175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_739: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1176, [0, 2, 1, 3]);  view_1176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_245: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_739, memory_format = torch.contiguous_format);  permute_739 = None
        view_1177: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_245, [96, 1024, 64]);  clone_245 = None
        bmm_120: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_740, view_1177);  permute_740 = None
        bmm_121: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1177, permute_741);  view_1177 = permute_741 = None
        view_1178: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_120, [8, 12, 1024, 64]);  bmm_120 = None
        view_1179: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_121, [8, 12, 1024, 1024]);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_46: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_84, torch.float32);  gt_84 = None
        mul_631: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
        mul_632: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1179, mul_631);  view_1179 = mul_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_633: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_632, div_27);  mul_632 = None
        sum_93: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_633, [-1], True)
        neg_14: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_27);  div_27 = None
        fma_12: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_14, sum_93, mul_633);  neg_14 = sum_93 = mul_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1180: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_12, [96, 1024, 1024]);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_122: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_742, view_1180);  permute_742 = None
        bmm_123: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1180, permute_743);  view_1180 = permute_743 = None
        view_1185: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_122, [8, 12, 64, 1024]);  bmm_122 = None
        view_1186: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_123, [8, 12, 1024, 64]);  bmm_123 = None
        permute_744: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1185, [0, 1, 3, 2]);  view_1185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_745: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1178, [0, 2, 1, 3]);  view_1178 = None
        clone_248: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_745, memory_format = torch.contiguous_format);  permute_745 = None
        view_1187: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_248, [8, 1024, 768]);  clone_248 = None
        view_1188: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1187, [8192, 768]);  view_1187 = None
        permute_746: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1188, [1, 0])
        view_560: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_321: "f32[768, 768]" = torch.ops.aten.mm.default(permute_746, view_560);  permute_746 = view_560 = None
        permute_247: "f32[768, 768]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        permute_748: "f32[768, 768]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        mm_322: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1188, permute_748);  view_1188 = permute_748 = None
        view_1189: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_322, [8, 1024, 768]);  mm_322 = None
        add_243: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_233, view_1189);  add_233 = view_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_750: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_744, [0, 2, 1, 3]);  permute_744 = None
        view_1190: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_750, [8, 1024, 768]);  permute_750 = None
        clone_249: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1190, memory_format = torch.contiguous_format);  view_1190 = None
        view_1191: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_249, [8192, 768]);  clone_249 = None
        permute_751: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1191, [1, 0])
        view_557: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_323: "f32[768, 768]" = torch.ops.aten.mm.default(permute_751, view_557);  permute_751 = view_557 = None
        permute_245: "f32[768, 768]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        permute_753: "f32[768, 768]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
        mm_324: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1191, permute_753);  view_1191 = permute_753 = None
        view_1192: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_324, [8, 1024, 768]);  mm_324 = None
        add_244: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_243, view_1192);  add_243 = view_1192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_755: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1186, [0, 2, 1, 3]);  view_1186 = None
        clone_250: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_755, memory_format = torch.contiguous_format);  permute_755 = None
        view_1193: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_250, [8, 1024, 768]);  clone_250 = None
        view_1194: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1193, [8192, 768]);  view_1193 = None
        permute_756: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1194, [1, 0])
        mm_325: "f32[768, 768]" = torch.ops.aten.mm.default(permute_756, view_554);  permute_756 = view_554 = None
        permute_243: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        permute_758: "f32[768, 768]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        mm_326: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1194, permute_758);  view_1194 = permute_758 = None
        view_1195: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_326, [8, 1024, 768]);  mm_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_634: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1195, primals_173);  primals_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_251: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_119, rsqrt_41)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_635: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1195, mul_251);  view_1195 = mul_251 = None
        sum_94: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 1], True);  mul_635 = None
        view_1196: "f32[768]" = torch.ops.aten.reshape.default(sum_94, [768]);  sum_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_636: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_634, add_119)
        mul_637: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_634, rsqrt_41);  mul_634 = None
        sum_95: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_636, [2], True);  mul_636 = None
        add_245: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_242, mul_637);  add_242 = mul_637 = None
        pow_103: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_41, 3);  rsqrt_41 = None
        mul_638: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_95, -0.5);  sum_95 = None
        mul_639: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_638, pow_103);  mul_638 = pow_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_167: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_639, [8, 1024, 768]);  mul_639 = None
        div_62: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_167, 768);  expand_167 = None
        pow_104: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_119, 1.0);  add_119 = None
        mul_640: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_104, 2.0);  pow_104 = None
        mul_641: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_62, mul_640);  div_62 = mul_640 = None
        add_246: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_245, mul_641);  add_245 = mul_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_47: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_83, torch.float32);  gt_83 = None
        mul_642: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
        mul_643: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_246, mul_642);  mul_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1197: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_643, [8192, 768]);  mul_643 = None
        permute_760: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1197, [1, 0])
        mm_327: "f32[768, 768]" = torch.ops.aten.mm.default(permute_760, view_552);  permute_760 = view_552 = None
        permute_242: "f32[768, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        permute_762: "f32[768, 768]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        mm_328: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1197, permute_762);  view_1197 = permute_762 = None
        view_1198: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_328, [8, 1024, 768]);  mm_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1199: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1198, [8, 1024, 12, 64]);  view_1198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_764: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1199, [0, 2, 1, 3]);  view_1199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_252: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_764, memory_format = torch.contiguous_format);  permute_764 = None
        view_1200: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_252, [96, 1024, 64]);  clone_252 = None
        bmm_124: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_765, view_1200);  permute_765 = None
        bmm_125: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1200, permute_766);  view_1200 = permute_766 = None
        view_1201: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_124, [8, 12, 1024, 64]);  bmm_124 = None
        view_1202: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_125, [8, 12, 1024, 1024]);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_48: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_82, torch.float32);  gt_82 = None
        mul_644: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
        mul_645: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1202, mul_644);  view_1202 = mul_644 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_646: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_645, div_26);  mul_645 = None
        sum_96: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_646, [-1], True)
        neg_15: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_26);  div_26 = None
        fma_13: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_15, sum_96, mul_646);  neg_15 = sum_96 = mul_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1203: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_13, [96, 1024, 1024]);  fma_13 = None
        view_1205: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1203, [8, 12, 1024, 1024]);  view_1203 = None
        view_1206: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1205, [96, 1024, 1024])
        add_247: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_236, view_1205);  add_236 = view_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_126: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_767, view_1206);  permute_767 = None
        bmm_127: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1206, permute_768);  view_1206 = permute_768 = None
        view_1208: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_126, [8, 12, 64, 1024]);  bmm_126 = None
        view_1209: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_127, [8, 12, 1024, 64]);  bmm_127 = None
        permute_769: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1208, [0, 1, 3, 2]);  view_1208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_770: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1201, [0, 2, 1, 3]);  view_1201 = None
        clone_255: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_770, memory_format = torch.contiguous_format);  permute_770 = None
        view_1210: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_255, [8, 1024, 768]);  clone_255 = None
        view_1211: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1210, [8192, 768]);  view_1210 = None
        permute_771: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1211, [1, 0])
        mm_329: "f32[768, 768]" = torch.ops.aten.mm.default(permute_771, view_533);  permute_771 = None
        permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_773: "f32[768, 768]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
        mm_330: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1211, permute_773);  view_1211 = permute_773 = None
        view_1212: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_330, [8, 1024, 768]);  mm_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_775: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_769, [0, 2, 1, 3]);  permute_769 = None
        view_1213: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_775, [8, 1024, 768]);  permute_775 = None
        clone_256: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1213, memory_format = torch.contiguous_format);  view_1213 = None
        view_1214: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_256, [8192, 768]);  clone_256 = None
        permute_776: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1214, [1, 0])
        mm_331: "f32[768, 768]" = torch.ops.aten.mm.default(permute_776, view_533);  permute_776 = None
        permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        permute_778: "f32[768, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
        mm_332: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1214, permute_778);  view_1214 = permute_778 = None
        view_1215: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_332, [8, 1024, 768]);  mm_332 = None
        add_248: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1212, view_1215);  view_1212 = view_1215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_780: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1209, [0, 2, 1, 3]);  view_1209 = None
        clone_257: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_780, memory_format = torch.contiguous_format);  permute_780 = None
        view_1216: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_257, [8, 1024, 768]);  clone_257 = None
        view_1217: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1216, [8192, 768]);  view_1216 = None
        permute_781: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1217, [1, 0])
        mm_333: "f32[768, 768]" = torch.ops.aten.mm.default(permute_781, view_533);  permute_781 = view_533 = None
        permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        permute_783: "f32[768, 768]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
        mm_334: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1217, permute_783);  view_1217 = permute_783 = None
        view_1218: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_334, [8, 1024, 768]);  mm_334 = None
        add_249: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_248, view_1218);  add_248 = view_1218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_647: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_249, primals_168);  primals_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_245: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_116, rsqrt_40)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_648: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_249, mul_245);  add_249 = mul_245 = None
        sum_97: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_648, [0, 1], True);  mul_648 = None
        view_1219: "f32[768]" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_649: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_647, add_116)
        mul_650: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_647, rsqrt_40);  mul_647 = None
        sum_98: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2], True);  mul_649 = None
        add_250: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_246, mul_650);  add_246 = mul_650 = None
        pow_105: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_40, 3);  rsqrt_40 = None
        mul_651: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_98, -0.5);  sum_98 = None
        mul_652: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_651, pow_105);  mul_651 = pow_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_168: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_652, [8, 1024, 768]);  mul_652 = None
        div_63: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_168, 768);  expand_168 = None
        pow_106: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_116, 1.0);  add_116 = None
        mul_653: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_106, 2.0);  pow_106 = None
        mul_654: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_63, mul_653);  div_63 = mul_653 = None
        add_251: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_250, mul_654);  add_250 = mul_654 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_49: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_81, torch.float32);  gt_81 = None
        mul_655: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
        mul_656: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_251, mul_655);  mul_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1220: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_656, [8192, 768]);  mul_656 = None
        permute_785: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1220, [1, 0])
        mm_335: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_785, view_531);  permute_785 = view_531 = None
        permute_233: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
        permute_787: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        mm_336: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1220, permute_787);  view_1220 = permute_787 = None
        view_1221: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_336, [8, 1024, 3072]);  mm_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_50: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_80, torch.float32);  gt_80 = None
        mul_657: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
        mul_658: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1221, mul_657);  view_1221 = mul_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_17: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_8, full_default, mul_658);  le_8 = mul_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1222: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_17, [8192, 3072]);  where_17 = None
        permute_789: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1222, [1, 0])
        mm_337: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_789, view_529);  permute_789 = view_529 = None
        permute_232: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        permute_791: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        mm_338: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1222, permute_791);  view_1222 = permute_791 = None
        view_1223: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_338, [8, 1024, 768]);  mm_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_659: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1223, primals_165);  primals_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_239: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_114, rsqrt_39)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_660: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1223, mul_239);  view_1223 = mul_239 = None
        sum_99: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_660, [0, 1], True);  mul_660 = None
        view_1224: "f32[768]" = torch.ops.aten.reshape.default(sum_99, [768]);  sum_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_661: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_659, add_114)
        mul_662: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_659, rsqrt_39);  mul_659 = None
        sum_100: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_661, [2], True);  mul_661 = None
        add_252: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_251, mul_662);  add_251 = mul_662 = None
        pow_107: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_39, 3);  rsqrt_39 = None
        mul_663: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_100, -0.5);  sum_100 = None
        mul_664: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_663, pow_107);  mul_663 = pow_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_169: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_664, [8, 1024, 768]);  mul_664 = None
        div_64: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_169, 768);  expand_169 = None
        pow_108: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_114, 1.0);  add_114 = None
        mul_665: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_108, 2.0);  pow_108 = None
        mul_666: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_64, mul_665);  div_64 = mul_665 = None
        add_253: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_252, mul_666);  add_252 = mul_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_51: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_79, torch.float32);  gt_79 = None
        mul_667: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
        mul_668: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_253, mul_667);  mul_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1225: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_668, [8192, 768]);  mul_668 = None
        permute_793: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1225, [1, 0])
        mm_339: "f32[768, 768]" = torch.ops.aten.mm.default(permute_793, view_527);  permute_793 = view_527 = None
        permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
        permute_795: "f32[768, 768]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        mm_340: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1225, permute_795);  view_1225 = permute_795 = None
        view_1226: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_340, [8, 1024, 768]);  mm_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1227: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1226, [8, 1024, 12, 64]);  view_1226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_797: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1227, [0, 2, 1, 3]);  view_1227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_261: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format);  permute_797 = None
        view_1228: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_261, [96, 1024, 64]);  clone_261 = None
        bmm_128: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_798, view_1228);  permute_798 = None
        bmm_129: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1228, permute_799);  view_1228 = permute_799 = None
        view_1229: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_128, [8, 12, 1024, 64]);  bmm_128 = None
        view_1230: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_129, [8, 12, 1024, 1024]);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_52: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_78, torch.float32);  gt_78 = None
        mul_669: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_670: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1230, mul_669);  view_1230 = mul_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_671: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_670, div_25);  mul_670 = None
        sum_101: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_671, [-1], True)
        neg_16: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_14: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_16, sum_101, mul_671);  neg_16 = sum_101 = mul_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1231: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_14, [96, 1024, 1024]);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_130: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_800, view_1231);  permute_800 = None
        bmm_131: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1231, permute_801);  view_1231 = permute_801 = None
        view_1236: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_130, [8, 12, 64, 1024]);  bmm_130 = None
        view_1237: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_131, [8, 12, 1024, 64]);  bmm_131 = None
        permute_802: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1236, [0, 1, 3, 2]);  view_1236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_803: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1229, [0, 2, 1, 3]);  view_1229 = None
        clone_264: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_803, memory_format = torch.contiguous_format);  permute_803 = None
        view_1238: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_264, [8, 1024, 768]);  clone_264 = None
        view_1239: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1238, [8192, 768]);  view_1238 = None
        permute_804: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1239, [1, 0])
        view_514: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_341: "f32[768, 768]" = torch.ops.aten.mm.default(permute_804, view_514);  permute_804 = view_514 = None
        permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        permute_806: "f32[768, 768]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
        mm_342: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1239, permute_806);  view_1239 = permute_806 = None
        view_1240: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_342, [8, 1024, 768]);  mm_342 = None
        add_254: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_244, view_1240);  add_244 = view_1240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_808: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_802, [0, 2, 1, 3]);  permute_802 = None
        view_1241: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_808, [8, 1024, 768]);  permute_808 = None
        clone_265: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1241, memory_format = torch.contiguous_format);  view_1241 = None
        view_1242: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_265, [8192, 768]);  clone_265 = None
        permute_809: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1242, [1, 0])
        view_511: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_343: "f32[768, 768]" = torch.ops.aten.mm.default(permute_809, view_511);  permute_809 = view_511 = None
        permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_811: "f32[768, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
        mm_344: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1242, permute_811);  view_1242 = permute_811 = None
        view_1243: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_344, [8, 1024, 768]);  mm_344 = None
        add_255: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_254, view_1243);  add_254 = view_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_813: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1237, [0, 2, 1, 3]);  view_1237 = None
        clone_266: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_813, memory_format = torch.contiguous_format);  permute_813 = None
        view_1244: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_266, [8, 1024, 768]);  clone_266 = None
        view_1245: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1244, [8192, 768]);  view_1244 = None
        permute_814: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1245, [1, 0])
        mm_345: "f32[768, 768]" = torch.ops.aten.mm.default(permute_814, view_508);  permute_814 = view_508 = None
        permute_223: "f32[768, 768]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
        permute_816: "f32[768, 768]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        mm_346: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1245, permute_816);  view_1245 = permute_816 = None
        view_1246: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_346, [8, 1024, 768]);  mm_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_672: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1246, primals_160);  primals_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_233: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_111, rsqrt_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_673: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1246, mul_233);  view_1246 = mul_233 = None
        sum_102: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_673, [0, 1], True);  mul_673 = None
        view_1247: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_674: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_672, add_111)
        mul_675: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_672, rsqrt_38);  mul_672 = None
        sum_103: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_674, [2], True);  mul_674 = None
        add_256: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_253, mul_675);  add_253 = mul_675 = None
        pow_109: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_38, 3);  rsqrt_38 = None
        mul_676: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_103, -0.5);  sum_103 = None
        mul_677: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_676, pow_109);  mul_676 = pow_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_170: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_677, [8, 1024, 768]);  mul_677 = None
        div_65: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_170, 768);  expand_170 = None
        pow_110: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_111, 1.0);  add_111 = None
        mul_678: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_110, 2.0);  pow_110 = None
        mul_679: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_65, mul_678);  div_65 = mul_678 = None
        add_257: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_256, mul_679);  add_256 = mul_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_53: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_77, torch.float32);  gt_77 = None
        mul_680: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
        mul_681: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_257, mul_680);  mul_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1248: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_681, [8192, 768]);  mul_681 = None
        permute_818: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1248, [1, 0])
        mm_347: "f32[768, 768]" = torch.ops.aten.mm.default(permute_818, view_506);  permute_818 = view_506 = None
        permute_222: "f32[768, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        permute_820: "f32[768, 768]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        mm_348: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1248, permute_820);  view_1248 = permute_820 = None
        view_1249: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_348, [8, 1024, 768]);  mm_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1250: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1249, [8, 1024, 12, 64]);  view_1249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_822: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1250, [0, 2, 1, 3]);  view_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_268: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_822, memory_format = torch.contiguous_format);  permute_822 = None
        view_1251: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_268, [96, 1024, 64]);  clone_268 = None
        bmm_132: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_823, view_1251);  permute_823 = None
        bmm_133: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1251, permute_824);  view_1251 = permute_824 = None
        view_1252: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_132, [8, 12, 1024, 64]);  bmm_132 = None
        view_1253: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_133, [8, 12, 1024, 1024]);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_54: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_76, torch.float32);  gt_76 = None
        mul_682: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_683: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1253, mul_682);  view_1253 = mul_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_684: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_683, div_24);  mul_683 = None
        sum_104: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_684, [-1], True)
        neg_17: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_24);  div_24 = None
        fma_15: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_17, sum_104, mul_684);  neg_17 = sum_104 = mul_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1254: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_15, [96, 1024, 1024]);  fma_15 = None
        view_1256: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1254, [8, 12, 1024, 1024]);  view_1254 = None
        view_1257: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1256, [96, 1024, 1024])
        add_258: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_247, view_1256);  add_247 = view_1256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_134: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_825, view_1257);  permute_825 = None
        bmm_135: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1257, permute_826);  view_1257 = permute_826 = None
        view_1259: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_134, [8, 12, 64, 1024]);  bmm_134 = None
        view_1260: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_135, [8, 12, 1024, 64]);  bmm_135 = None
        permute_827: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1259, [0, 1, 3, 2]);  view_1259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_828: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1252, [0, 2, 1, 3]);  view_1252 = None
        clone_271: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_828, memory_format = torch.contiguous_format);  permute_828 = None
        view_1261: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_271, [8, 1024, 768]);  clone_271 = None
        view_1262: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1261, [8192, 768]);  view_1261 = None
        permute_829: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1262, [1, 0])
        mm_349: "f32[768, 768]" = torch.ops.aten.mm.default(permute_829, view_487);  permute_829 = None
        permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
        permute_831: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        mm_350: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1262, permute_831);  view_1262 = permute_831 = None
        view_1263: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_350, [8, 1024, 768]);  mm_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_833: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_827, [0, 2, 1, 3]);  permute_827 = None
        view_1264: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_833, [8, 1024, 768]);  permute_833 = None
        clone_272: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1264, memory_format = torch.contiguous_format);  view_1264 = None
        view_1265: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_272, [8192, 768]);  clone_272 = None
        permute_834: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1265, [1, 0])
        mm_351: "f32[768, 768]" = torch.ops.aten.mm.default(permute_834, view_487);  permute_834 = None
        permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        permute_836: "f32[768, 768]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        mm_352: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1265, permute_836);  view_1265 = permute_836 = None
        view_1266: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_352, [8, 1024, 768]);  mm_352 = None
        add_259: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1263, view_1266);  view_1263 = view_1266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_838: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1260, [0, 2, 1, 3]);  view_1260 = None
        clone_273: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
        view_1267: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_273, [8, 1024, 768]);  clone_273 = None
        view_1268: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1267, [8192, 768]);  view_1267 = None
        permute_839: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1268, [1, 0])
        mm_353: "f32[768, 768]" = torch.ops.aten.mm.default(permute_839, view_487);  permute_839 = view_487 = None
        permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        permute_841: "f32[768, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        mm_354: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1268, permute_841);  view_1268 = permute_841 = None
        view_1269: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_354, [8, 1024, 768]);  mm_354 = None
        add_260: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_259, view_1269);  add_259 = view_1269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_685: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_260, primals_155);  primals_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_227: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_108, rsqrt_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_686: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_260, mul_227);  add_260 = mul_227 = None
        sum_105: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_686, [0, 1], True);  mul_686 = None
        view_1270: "f32[768]" = torch.ops.aten.reshape.default(sum_105, [768]);  sum_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_687: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_685, add_108)
        mul_688: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_685, rsqrt_37);  mul_685 = None
        sum_106: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_687, [2], True);  mul_687 = None
        add_261: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_257, mul_688);  add_257 = mul_688 = None
        pow_111: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_37, 3);  rsqrt_37 = None
        mul_689: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_106, -0.5);  sum_106 = None
        mul_690: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_689, pow_111);  mul_689 = pow_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_171: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_690, [8, 1024, 768]);  mul_690 = None
        div_66: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_171, 768);  expand_171 = None
        pow_112: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_108, 1.0);  add_108 = None
        mul_691: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_112, 2.0);  pow_112 = None
        mul_692: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_66, mul_691);  div_66 = mul_691 = None
        add_262: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_261, mul_692);  add_261 = mul_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_55: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_75, torch.float32);  gt_75 = None
        mul_693: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_694: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_262, mul_693);  mul_693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1271: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_694, [8192, 768]);  mul_694 = None
        permute_843: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1271, [1, 0])
        mm_355: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_843, view_485);  permute_843 = view_485 = None
        permute_213: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        permute_845: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        mm_356: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1271, permute_845);  view_1271 = permute_845 = None
        view_1272: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_356, [8, 1024, 3072]);  mm_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_56: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_74, torch.float32);  gt_74 = None
        mul_695: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
        mul_696: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1272, mul_695);  view_1272 = mul_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_18: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_9, full_default, mul_696);  le_9 = mul_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1273: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_18, [8192, 3072]);  where_18 = None
        permute_847: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1273, [1, 0])
        mm_357: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_847, view_483);  permute_847 = view_483 = None
        permute_212: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_849: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        mm_358: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1273, permute_849);  view_1273 = permute_849 = None
        view_1274: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_358, [8, 1024, 768]);  mm_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_697: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1274, primals_152);  primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_221: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_106, rsqrt_36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_698: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1274, mul_221);  view_1274 = mul_221 = None
        sum_107: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 1], True);  mul_698 = None
        view_1275: "f32[768]" = torch.ops.aten.reshape.default(sum_107, [768]);  sum_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_699: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_697, add_106)
        mul_700: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_697, rsqrt_36);  mul_697 = None
        sum_108: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_699, [2], True);  mul_699 = None
        add_263: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_262, mul_700);  add_262 = mul_700 = None
        pow_113: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_36, 3);  rsqrt_36 = None
        mul_701: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_108, -0.5);  sum_108 = None
        mul_702: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_701, pow_113);  mul_701 = pow_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_172: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_702, [8, 1024, 768]);  mul_702 = None
        div_67: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_172, 768);  expand_172 = None
        pow_114: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_106, 1.0);  add_106 = None
        mul_703: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_114, 2.0);  pow_114 = None
        mul_704: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_67, mul_703);  div_67 = mul_703 = None
        add_264: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_263, mul_704);  add_263 = mul_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_57: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_73, torch.float32);  gt_73 = None
        mul_705: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
        mul_706: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_264, mul_705);  mul_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1276: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_706, [8192, 768]);  mul_706 = None
        permute_851: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1276, [1, 0])
        mm_359: "f32[768, 768]" = torch.ops.aten.mm.default(permute_851, view_481);  permute_851 = view_481 = None
        permute_211: "f32[768, 768]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        permute_853: "f32[768, 768]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        mm_360: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1276, permute_853);  view_1276 = permute_853 = None
        view_1277: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_360, [8, 1024, 768]);  mm_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1278: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1277, [8, 1024, 12, 64]);  view_1277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_855: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1278, [0, 2, 1, 3]);  view_1278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_277: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_855, memory_format = torch.contiguous_format);  permute_855 = None
        view_1279: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_277, [96, 1024, 64]);  clone_277 = None
        bmm_136: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_856, view_1279);  permute_856 = None
        bmm_137: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1279, permute_857);  view_1279 = permute_857 = None
        view_1280: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_136, [8, 12, 1024, 64]);  bmm_136 = None
        view_1281: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_137, [8, 12, 1024, 1024]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_58: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_72, torch.float32);  gt_72 = None
        mul_707: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
        mul_708: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1281, mul_707);  view_1281 = mul_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_709: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_708, div_23);  mul_708 = None
        sum_109: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_709, [-1], True)
        neg_18: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma_16: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_18, sum_109, mul_709);  neg_18 = sum_109 = mul_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1282: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_16, [96, 1024, 1024]);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_138: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_858, view_1282);  permute_858 = None
        bmm_139: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1282, permute_859);  view_1282 = permute_859 = None
        view_1287: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_138, [8, 12, 64, 1024]);  bmm_138 = None
        view_1288: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_139, [8, 12, 1024, 64]);  bmm_139 = None
        permute_860: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1287, [0, 1, 3, 2]);  view_1287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_861: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1280, [0, 2, 1, 3]);  view_1280 = None
        clone_280: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_861, memory_format = torch.contiguous_format);  permute_861 = None
        view_1289: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_280, [8, 1024, 768]);  clone_280 = None
        view_1290: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1289, [8192, 768]);  view_1289 = None
        permute_862: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1290, [1, 0])
        view_468: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_361: "f32[768, 768]" = torch.ops.aten.mm.default(permute_862, view_468);  permute_862 = view_468 = None
        permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        permute_864: "f32[768, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        mm_362: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1290, permute_864);  view_1290 = permute_864 = None
        view_1291: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_362, [8, 1024, 768]);  mm_362 = None
        add_265: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_255, view_1291);  add_255 = view_1291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_866: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_860, [0, 2, 1, 3]);  permute_860 = None
        view_1292: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_866, [8, 1024, 768]);  permute_866 = None
        clone_281: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1292, memory_format = torch.contiguous_format);  view_1292 = None
        view_1293: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_281, [8192, 768]);  clone_281 = None
        permute_867: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1293, [1, 0])
        view_465: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_363: "f32[768, 768]" = torch.ops.aten.mm.default(permute_867, view_465);  permute_867 = view_465 = None
        permute_205: "f32[768, 768]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        permute_869: "f32[768, 768]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        mm_364: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1293, permute_869);  view_1293 = permute_869 = None
        view_1294: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_364, [8, 1024, 768]);  mm_364 = None
        add_266: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_265, view_1294);  add_265 = view_1294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_871: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1288, [0, 2, 1, 3]);  view_1288 = None
        clone_282: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_871, memory_format = torch.contiguous_format);  permute_871 = None
        view_1295: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_282, [8, 1024, 768]);  clone_282 = None
        view_1296: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1295, [8192, 768]);  view_1295 = None
        permute_872: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1296, [1, 0])
        mm_365: "f32[768, 768]" = torch.ops.aten.mm.default(permute_872, view_462);  permute_872 = view_462 = None
        permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        permute_874: "f32[768, 768]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        mm_366: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1296, permute_874);  view_1296 = permute_874 = None
        view_1297: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_366, [8, 1024, 768]);  mm_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_710: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1297, primals_147);  primals_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_215: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_103, rsqrt_35)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_711: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1297, mul_215);  view_1297 = mul_215 = None
        sum_110: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_711, [0, 1], True);  mul_711 = None
        view_1298: "f32[768]" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_712: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_710, add_103)
        mul_713: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_710, rsqrt_35);  mul_710 = None
        sum_111: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_712, [2], True);  mul_712 = None
        add_267: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_264, mul_713);  add_264 = mul_713 = None
        pow_115: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_35, 3);  rsqrt_35 = None
        mul_714: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_111, -0.5);  sum_111 = None
        mul_715: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_714, pow_115);  mul_714 = pow_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_173: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_715, [8, 1024, 768]);  mul_715 = None
        div_68: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_173, 768);  expand_173 = None
        pow_116: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_103, 1.0);  add_103 = None
        mul_716: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_116, 2.0);  pow_116 = None
        mul_717: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_68, mul_716);  div_68 = mul_716 = None
        add_268: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_267, mul_717);  add_267 = mul_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_59: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_71, torch.float32);  gt_71 = None
        mul_718: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
        mul_719: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_268, mul_718);  mul_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1299: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_719, [8192, 768]);  mul_719 = None
        permute_876: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1299, [1, 0])
        mm_367: "f32[768, 768]" = torch.ops.aten.mm.default(permute_876, view_460);  permute_876 = view_460 = None
        permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        permute_878: "f32[768, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        mm_368: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1299, permute_878);  view_1299 = permute_878 = None
        view_1300: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_368, [8, 1024, 768]);  mm_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1301: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1300, [8, 1024, 12, 64]);  view_1300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_880: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1301, [0, 2, 1, 3]);  view_1301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_284: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
        view_1302: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_284, [96, 1024, 64]);  clone_284 = None
        bmm_140: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_881, view_1302);  permute_881 = None
        bmm_141: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1302, permute_882);  view_1302 = permute_882 = None
        view_1303: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_140, [8, 12, 1024, 64]);  bmm_140 = None
        view_1304: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_141, [8, 12, 1024, 1024]);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_60: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_70, torch.float32);  gt_70 = None
        mul_720: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
        mul_721: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1304, mul_720);  view_1304 = mul_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_722: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_721, div_22);  mul_721 = None
        sum_112: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_722, [-1], True)
        neg_19: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_17: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_19, sum_112, mul_722);  neg_19 = sum_112 = mul_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1305: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_17, [96, 1024, 1024]);  fma_17 = None
        view_1307: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1305, [8, 12, 1024, 1024]);  view_1305 = None
        view_1308: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1307, [96, 1024, 1024])
        add_269: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_258, view_1307);  add_258 = view_1307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_142: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_883, view_1308);  permute_883 = None
        bmm_143: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1308, permute_884);  view_1308 = permute_884 = None
        view_1310: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_142, [8, 12, 64, 1024]);  bmm_142 = None
        view_1311: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_143, [8, 12, 1024, 64]);  bmm_143 = None
        permute_885: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1310, [0, 1, 3, 2]);  view_1310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_886: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1303, [0, 2, 1, 3]);  view_1303 = None
        clone_287: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_886, memory_format = torch.contiguous_format);  permute_886 = None
        view_1312: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_287, [8, 1024, 768]);  clone_287 = None
        view_1313: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1312, [8192, 768]);  view_1312 = None
        permute_887: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1313, [1, 0])
        mm_369: "f32[768, 768]" = torch.ops.aten.mm.default(permute_887, view_441);  permute_887 = None
        permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        permute_889: "f32[768, 768]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        mm_370: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1313, permute_889);  view_1313 = permute_889 = None
        view_1314: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_370, [8, 1024, 768]);  mm_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_891: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_885, [0, 2, 1, 3]);  permute_885 = None
        view_1315: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_891, [8, 1024, 768]);  permute_891 = None
        clone_288: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1315, memory_format = torch.contiguous_format);  view_1315 = None
        view_1316: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_288, [8192, 768]);  clone_288 = None
        permute_892: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1316, [1, 0])
        mm_371: "f32[768, 768]" = torch.ops.aten.mm.default(permute_892, view_441);  permute_892 = None
        permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        permute_894: "f32[768, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        mm_372: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1316, permute_894);  view_1316 = permute_894 = None
        view_1317: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_372, [8, 1024, 768]);  mm_372 = None
        add_270: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1314, view_1317);  view_1314 = view_1317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_896: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1311, [0, 2, 1, 3]);  view_1311 = None
        clone_289: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_896, memory_format = torch.contiguous_format);  permute_896 = None
        view_1318: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_289, [8, 1024, 768]);  clone_289 = None
        view_1319: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1318, [8192, 768]);  view_1318 = None
        permute_897: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1319, [1, 0])
        mm_373: "f32[768, 768]" = torch.ops.aten.mm.default(permute_897, view_441);  permute_897 = view_441 = None
        permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        permute_899: "f32[768, 768]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        mm_374: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1319, permute_899);  view_1319 = permute_899 = None
        view_1320: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_374, [8, 1024, 768]);  mm_374 = None
        add_271: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_270, view_1320);  add_270 = view_1320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_723: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_271, primals_142);  primals_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_209: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_100, rsqrt_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_724: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_271, mul_209);  add_271 = mul_209 = None
        sum_113: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 1], True);  mul_724 = None
        view_1321: "f32[768]" = torch.ops.aten.reshape.default(sum_113, [768]);  sum_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_725: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_723, add_100)
        mul_726: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_723, rsqrt_34);  mul_723 = None
        sum_114: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_725, [2], True);  mul_725 = None
        add_272: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_268, mul_726);  add_268 = mul_726 = None
        pow_117: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_34, 3);  rsqrt_34 = None
        mul_727: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_114, -0.5);  sum_114 = None
        mul_728: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_727, pow_117);  mul_727 = pow_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_174: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_728, [8, 1024, 768]);  mul_728 = None
        div_69: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_174, 768);  expand_174 = None
        pow_118: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_100, 1.0);  add_100 = None
        mul_729: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_118, 2.0);  pow_118 = None
        mul_730: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_69, mul_729);  div_69 = mul_729 = None
        add_273: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_272, mul_730);  add_272 = mul_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_61: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_69, torch.float32);  gt_69 = None
        mul_731: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
        mul_732: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_273, mul_731);  mul_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1322: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_732, [8192, 768]);  mul_732 = None
        permute_901: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1322, [1, 0])
        mm_375: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_901, view_439);  permute_901 = view_439 = None
        permute_193: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_903: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        mm_376: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1322, permute_903);  view_1322 = permute_903 = None
        view_1323: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_376, [8, 1024, 3072]);  mm_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_62: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_68, torch.float32);  gt_68 = None
        mul_733: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_734: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1323, mul_733);  view_1323 = mul_733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_19: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_10, full_default, mul_734);  le_10 = mul_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1324: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_19, [8192, 3072]);  where_19 = None
        permute_905: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1324, [1, 0])
        mm_377: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_905, view_437);  permute_905 = view_437 = None
        permute_192: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        permute_907: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
        mm_378: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1324, permute_907);  view_1324 = permute_907 = None
        view_1325: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_378, [8, 1024, 768]);  mm_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_735: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1325, primals_139);  primals_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_203: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_98, rsqrt_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_736: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1325, mul_203);  view_1325 = mul_203 = None
        sum_115: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 1], True);  mul_736 = None
        view_1326: "f32[768]" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_737: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_735, add_98)
        mul_738: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_735, rsqrt_33);  mul_735 = None
        sum_116: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_737, [2], True);  mul_737 = None
        add_274: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_273, mul_738);  add_273 = mul_738 = None
        pow_119: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_33, 3);  rsqrt_33 = None
        mul_739: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_116, -0.5);  sum_116 = None
        mul_740: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_739, pow_119);  mul_739 = pow_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_175: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_740, [8, 1024, 768]);  mul_740 = None
        div_70: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_175, 768);  expand_175 = None
        pow_120: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_98, 1.0);  add_98 = None
        mul_741: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_120, 2.0);  pow_120 = None
        mul_742: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_70, mul_741);  div_70 = mul_741 = None
        add_275: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_274, mul_742);  add_274 = mul_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_63: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_67, torch.float32);  gt_67 = None
        mul_743: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
        mul_744: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_275, mul_743);  mul_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1327: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_744, [8192, 768]);  mul_744 = None
        permute_909: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1327, [1, 0])
        mm_379: "f32[768, 768]" = torch.ops.aten.mm.default(permute_909, view_435);  permute_909 = view_435 = None
        permute_191: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        permute_911: "f32[768, 768]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        mm_380: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1327, permute_911);  view_1327 = permute_911 = None
        view_1328: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_380, [8, 1024, 768]);  mm_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1329: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1328, [8, 1024, 12, 64]);  view_1328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_913: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1329, [0, 2, 1, 3]);  view_1329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_293: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_913, memory_format = torch.contiguous_format);  permute_913 = None
        view_1330: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_293, [96, 1024, 64]);  clone_293 = None
        bmm_144: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_914, view_1330);  permute_914 = None
        bmm_145: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1330, permute_915);  view_1330 = permute_915 = None
        view_1331: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_144, [8, 12, 1024, 64]);  bmm_144 = None
        view_1332: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_145, [8, 12, 1024, 1024]);  bmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_64: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_66, torch.float32);  gt_66 = None
        mul_745: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
        mul_746: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1332, mul_745);  view_1332 = mul_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_747: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_746, div_21);  mul_746 = None
        sum_117: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [-1], True)
        neg_20: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_18: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_20, sum_117, mul_747);  neg_20 = sum_117 = mul_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1333: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_18, [96, 1024, 1024]);  fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_146: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_916, view_1333);  permute_916 = None
        bmm_147: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1333, permute_917);  view_1333 = permute_917 = None
        view_1338: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_146, [8, 12, 64, 1024]);  bmm_146 = None
        view_1339: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_147, [8, 12, 1024, 64]);  bmm_147 = None
        permute_918: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1338, [0, 1, 3, 2]);  view_1338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_919: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1331, [0, 2, 1, 3]);  view_1331 = None
        clone_296: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_919, memory_format = torch.contiguous_format);  permute_919 = None
        view_1340: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_296, [8, 1024, 768]);  clone_296 = None
        view_1341: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1340, [8192, 768]);  view_1340 = None
        permute_920: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1341, [1, 0])
        view_422: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_381: "f32[768, 768]" = torch.ops.aten.mm.default(permute_920, view_422);  permute_920 = view_422 = None
        permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        permute_922: "f32[768, 768]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        mm_382: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1341, permute_922);  view_1341 = permute_922 = None
        view_1342: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_382, [8, 1024, 768]);  mm_382 = None
        add_276: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_266, view_1342);  add_266 = view_1342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_924: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_918, [0, 2, 1, 3]);  permute_918 = None
        view_1343: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_924, [8, 1024, 768]);  permute_924 = None
        clone_297: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1343, memory_format = torch.contiguous_format);  view_1343 = None
        view_1344: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_297, [8192, 768]);  clone_297 = None
        permute_925: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1344, [1, 0])
        view_419: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_383: "f32[768, 768]" = torch.ops.aten.mm.default(permute_925, view_419);  permute_925 = view_419 = None
        permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        permute_927: "f32[768, 768]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        mm_384: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1344, permute_927);  view_1344 = permute_927 = None
        view_1345: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_384, [8, 1024, 768]);  mm_384 = None
        add_277: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_276, view_1345);  add_276 = view_1345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_929: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1339, [0, 2, 1, 3]);  view_1339 = None
        clone_298: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_929, memory_format = torch.contiguous_format);  permute_929 = None
        view_1346: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_298, [8, 1024, 768]);  clone_298 = None
        view_1347: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1346, [8192, 768]);  view_1346 = None
        permute_930: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1347, [1, 0])
        mm_385: "f32[768, 768]" = torch.ops.aten.mm.default(permute_930, view_416);  permute_930 = view_416 = None
        permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
        permute_932: "f32[768, 768]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
        mm_386: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1347, permute_932);  view_1347 = permute_932 = None
        view_1348: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_386, [8, 1024, 768]);  mm_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_748: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1348, primals_134);  primals_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_197: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_95, rsqrt_32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_749: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1348, mul_197);  view_1348 = mul_197 = None
        sum_118: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 1], True);  mul_749 = None
        view_1349: "f32[768]" = torch.ops.aten.reshape.default(sum_118, [768]);  sum_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_750: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_748, add_95)
        mul_751: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_748, rsqrt_32);  mul_748 = None
        sum_119: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_750, [2], True);  mul_750 = None
        add_278: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_275, mul_751);  add_275 = mul_751 = None
        pow_121: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_32, 3);  rsqrt_32 = None
        mul_752: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_119, -0.5);  sum_119 = None
        mul_753: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_752, pow_121);  mul_752 = pow_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_176: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_753, [8, 1024, 768]);  mul_753 = None
        div_71: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_176, 768);  expand_176 = None
        pow_122: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_95, 1.0);  add_95 = None
        mul_754: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_122, 2.0);  pow_122 = None
        mul_755: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_71, mul_754);  div_71 = mul_754 = None
        add_279: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_278, mul_755);  add_278 = mul_755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_65: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_65, torch.float32);  gt_65 = None
        mul_756: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
        mul_757: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_279, mul_756);  mul_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1350: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_757, [8192, 768]);  mul_757 = None
        permute_934: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1350, [1, 0])
        mm_387: "f32[768, 768]" = torch.ops.aten.mm.default(permute_934, view_414);  permute_934 = view_414 = None
        permute_182: "f32[768, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        permute_936: "f32[768, 768]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
        mm_388: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1350, permute_936);  view_1350 = permute_936 = None
        view_1351: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_388, [8, 1024, 768]);  mm_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1352: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1351, [8, 1024, 12, 64]);  view_1351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_938: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1352, [0, 2, 1, 3]);  view_1352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_300: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_938, memory_format = torch.contiguous_format);  permute_938 = None
        view_1353: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_300, [96, 1024, 64]);  clone_300 = None
        bmm_148: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_939, view_1353);  permute_939 = None
        bmm_149: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1353, permute_940);  view_1353 = permute_940 = None
        view_1354: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_148, [8, 12, 1024, 64]);  bmm_148 = None
        view_1355: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_149, [8, 12, 1024, 1024]);  bmm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_66: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_64, torch.float32);  gt_64 = None
        mul_758: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_759: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1355, mul_758);  view_1355 = mul_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_760: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_759, div_20);  mul_759 = None
        sum_120: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_760, [-1], True)
        neg_21: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_19: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_21, sum_120, mul_760);  neg_21 = sum_120 = mul_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1356: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_19, [96, 1024, 1024]);  fma_19 = None
        view_1358: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1356, [8, 12, 1024, 1024]);  view_1356 = None
        view_1359: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1358, [96, 1024, 1024])
        add_280: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_269, view_1358);  add_269 = view_1358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_150: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_941, view_1359);  permute_941 = None
        bmm_151: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1359, permute_942);  view_1359 = permute_942 = None
        view_1361: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_150, [8, 12, 64, 1024]);  bmm_150 = None
        view_1362: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_151, [8, 12, 1024, 64]);  bmm_151 = None
        permute_943: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1361, [0, 1, 3, 2]);  view_1361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_944: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1354, [0, 2, 1, 3]);  view_1354 = None
        clone_303: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_944, memory_format = torch.contiguous_format);  permute_944 = None
        view_1363: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_303, [8, 1024, 768]);  clone_303 = None
        view_1364: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1363, [8192, 768]);  view_1363 = None
        permute_945: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1364, [1, 0])
        mm_389: "f32[768, 768]" = torch.ops.aten.mm.default(permute_945, view_395);  permute_945 = None
        permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        permute_947: "f32[768, 768]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        mm_390: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1364, permute_947);  view_1364 = permute_947 = None
        view_1365: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_390, [8, 1024, 768]);  mm_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_949: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_943, [0, 2, 1, 3]);  permute_943 = None
        view_1366: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_949, [8, 1024, 768]);  permute_949 = None
        clone_304: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1366, memory_format = torch.contiguous_format);  view_1366 = None
        view_1367: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_304, [8192, 768]);  clone_304 = None
        permute_950: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1367, [1, 0])
        mm_391: "f32[768, 768]" = torch.ops.aten.mm.default(permute_950, view_395);  permute_950 = None
        permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        permute_952: "f32[768, 768]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_392: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1367, permute_952);  view_1367 = permute_952 = None
        view_1368: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_392, [8, 1024, 768]);  mm_392 = None
        add_281: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1365, view_1368);  view_1365 = view_1368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_954: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1362, [0, 2, 1, 3]);  view_1362 = None
        clone_305: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_954, memory_format = torch.contiguous_format);  permute_954 = None
        view_1369: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_305, [8, 1024, 768]);  clone_305 = None
        view_1370: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1369, [8192, 768]);  view_1369 = None
        permute_955: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1370, [1, 0])
        mm_393: "f32[768, 768]" = torch.ops.aten.mm.default(permute_955, view_395);  permute_955 = view_395 = None
        permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        permute_957: "f32[768, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_394: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1370, permute_957);  view_1370 = permute_957 = None
        view_1371: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_394, [8, 1024, 768]);  mm_394 = None
        add_282: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_281, view_1371);  add_281 = view_1371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_761: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_282, primals_129);  primals_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_191: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_92, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_762: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_282, mul_191);  add_282 = mul_191 = None
        sum_121: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 1], True);  mul_762 = None
        view_1372: "f32[768]" = torch.ops.aten.reshape.default(sum_121, [768]);  sum_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_763: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_761, add_92)
        mul_764: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_761, rsqrt_31);  mul_761 = None
        sum_122: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [2], True);  mul_763 = None
        add_283: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_279, mul_764);  add_279 = mul_764 = None
        pow_123: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_31, 3);  rsqrt_31 = None
        mul_765: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_122, -0.5);  sum_122 = None
        mul_766: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_765, pow_123);  mul_765 = pow_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_177: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_766, [8, 1024, 768]);  mul_766 = None
        div_72: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_177, 768);  expand_177 = None
        pow_124: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_92, 1.0);  add_92 = None
        mul_767: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_124, 2.0);  pow_124 = None
        mul_768: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_72, mul_767);  div_72 = mul_767 = None
        add_284: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_283, mul_768);  add_283 = mul_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_67: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_63, torch.float32);  gt_63 = None
        mul_769: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
        mul_770: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_284, mul_769);  mul_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1373: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_770, [8192, 768]);  mul_770 = None
        permute_959: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1373, [1, 0])
        mm_395: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_959, view_393);  permute_959 = view_393 = None
        permute_173: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_961: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_396: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1373, permute_961);  view_1373 = permute_961 = None
        view_1374: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_396, [8, 1024, 3072]);  mm_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_68: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_62, torch.float32);  gt_62 = None
        mul_771: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_772: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1374, mul_771);  view_1374 = mul_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_20: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_11, full_default, mul_772);  le_11 = mul_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1375: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_20, [8192, 3072]);  where_20 = None
        permute_963: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1375, [1, 0])
        mm_397: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_963, view_391);  permute_963 = view_391 = None
        permute_172: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_965: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        mm_398: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1375, permute_965);  view_1375 = permute_965 = None
        view_1376: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_398, [8, 1024, 768]);  mm_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_773: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1376, primals_126);  primals_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_185: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_90, rsqrt_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_774: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1376, mul_185);  view_1376 = mul_185 = None
        sum_123: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_774, [0, 1], True);  mul_774 = None
        view_1377: "f32[768]" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_775: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_773, add_90)
        mul_776: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_773, rsqrt_30);  mul_773 = None
        sum_124: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True);  mul_775 = None
        add_285: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_284, mul_776);  add_284 = mul_776 = None
        pow_125: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_30, 3);  rsqrt_30 = None
        mul_777: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_124, -0.5);  sum_124 = None
        mul_778: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_777, pow_125);  mul_777 = pow_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_178: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_778, [8, 1024, 768]);  mul_778 = None
        div_73: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_178, 768);  expand_178 = None
        pow_126: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_90, 1.0);  add_90 = None
        mul_779: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_126, 2.0);  pow_126 = None
        mul_780: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_73, mul_779);  div_73 = mul_779 = None
        add_286: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_285, mul_780);  add_285 = mul_780 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_69: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_61, torch.float32);  gt_61 = None
        mul_781: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
        mul_782: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_286, mul_781);  mul_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1378: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_782, [8192, 768]);  mul_782 = None
        permute_967: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1378, [1, 0])
        mm_399: "f32[768, 768]" = torch.ops.aten.mm.default(permute_967, view_389);  permute_967 = view_389 = None
        permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        permute_969: "f32[768, 768]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
        mm_400: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1378, permute_969);  view_1378 = permute_969 = None
        view_1379: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_400, [8, 1024, 768]);  mm_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1380: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1379, [8, 1024, 12, 64]);  view_1379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_971: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1380, [0, 2, 1, 3]);  view_1380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_309: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_971, memory_format = torch.contiguous_format);  permute_971 = None
        view_1381: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_309, [96, 1024, 64]);  clone_309 = None
        bmm_152: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_972, view_1381);  permute_972 = None
        bmm_153: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1381, permute_973);  view_1381 = permute_973 = None
        view_1382: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_152, [8, 12, 1024, 64]);  bmm_152 = None
        view_1383: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_153, [8, 12, 1024, 1024]);  bmm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_70: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_60, torch.float32);  gt_60 = None
        mul_783: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
        mul_784: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1383, mul_783);  view_1383 = mul_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_785: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_784, div_19);  mul_784 = None
        sum_125: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_785, [-1], True)
        neg_22: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_20: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_22, sum_125, mul_785);  neg_22 = sum_125 = mul_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1384: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_20, [96, 1024, 1024]);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_154: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_974, view_1384);  permute_974 = None
        bmm_155: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1384, permute_975);  view_1384 = permute_975 = None
        view_1389: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_154, [8, 12, 64, 1024]);  bmm_154 = None
        view_1390: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_155, [8, 12, 1024, 64]);  bmm_155 = None
        permute_976: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1389, [0, 1, 3, 2]);  view_1389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_977: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1382, [0, 2, 1, 3]);  view_1382 = None
        clone_312: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_977, memory_format = torch.contiguous_format);  permute_977 = None
        view_1391: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_312, [8, 1024, 768]);  clone_312 = None
        view_1392: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1391, [8192, 768]);  view_1391 = None
        permute_978: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1392, [1, 0])
        view_376: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_401: "f32[768, 768]" = torch.ops.aten.mm.default(permute_978, view_376);  permute_978 = view_376 = None
        permute_167: "f32[768, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        permute_980: "f32[768, 768]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_402: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1392, permute_980);  view_1392 = permute_980 = None
        view_1393: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_402, [8, 1024, 768]);  mm_402 = None
        add_287: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_277, view_1393);  add_277 = view_1393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_982: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_976, [0, 2, 1, 3]);  permute_976 = None
        view_1394: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_982, [8, 1024, 768]);  permute_982 = None
        clone_313: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1394, memory_format = torch.contiguous_format);  view_1394 = None
        view_1395: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_313, [8192, 768]);  clone_313 = None
        permute_983: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1395, [1, 0])
        view_373: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_403: "f32[768, 768]" = torch.ops.aten.mm.default(permute_983, view_373);  permute_983 = view_373 = None
        permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        permute_985: "f32[768, 768]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_404: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1395, permute_985);  view_1395 = permute_985 = None
        view_1396: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_404, [8, 1024, 768]);  mm_404 = None
        add_288: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_287, view_1396);  add_287 = view_1396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_987: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1390, [0, 2, 1, 3]);  view_1390 = None
        clone_314: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_987, memory_format = torch.contiguous_format);  permute_987 = None
        view_1397: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_314, [8, 1024, 768]);  clone_314 = None
        view_1398: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1397, [8192, 768]);  view_1397 = None
        permute_988: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1398, [1, 0])
        mm_405: "f32[768, 768]" = torch.ops.aten.mm.default(permute_988, view_370);  permute_988 = view_370 = None
        permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_990: "f32[768, 768]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_406: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1398, permute_990);  view_1398 = permute_990 = None
        view_1399: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_406, [8, 1024, 768]);  mm_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_786: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1399, primals_121);  primals_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_179: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_87, rsqrt_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_787: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1399, mul_179);  view_1399 = mul_179 = None
        sum_126: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 1], True);  mul_787 = None
        view_1400: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_788: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_786, add_87)
        mul_789: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_786, rsqrt_29);  mul_786 = None
        sum_127: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_788, [2], True);  mul_788 = None
        add_289: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_286, mul_789);  add_286 = mul_789 = None
        pow_127: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_29, 3);  rsqrt_29 = None
        mul_790: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_127, -0.5);  sum_127 = None
        mul_791: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_790, pow_127);  mul_790 = pow_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_179: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_791, [8, 1024, 768]);  mul_791 = None
        div_74: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_179, 768);  expand_179 = None
        pow_128: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_87, 1.0);  add_87 = None
        mul_792: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_128, 2.0);  pow_128 = None
        mul_793: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_74, mul_792);  div_74 = mul_792 = None
        add_290: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_289, mul_793);  add_289 = mul_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_71: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_59, torch.float32);  gt_59 = None
        mul_794: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
        mul_795: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_290, mul_794);  mul_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1401: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_795, [8192, 768]);  mul_795 = None
        permute_992: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1401, [1, 0])
        mm_407: "f32[768, 768]" = torch.ops.aten.mm.default(permute_992, view_368);  permute_992 = view_368 = None
        permute_162: "f32[768, 768]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_994: "f32[768, 768]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_408: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1401, permute_994);  view_1401 = permute_994 = None
        view_1402: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_408, [8, 1024, 768]);  mm_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1403: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1402, [8, 1024, 12, 64]);  view_1402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_996: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1403, [0, 2, 1, 3]);  view_1403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_316: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_996, memory_format = torch.contiguous_format);  permute_996 = None
        view_1404: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_316, [96, 1024, 64]);  clone_316 = None
        bmm_156: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_997, view_1404);  permute_997 = None
        bmm_157: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1404, permute_998);  view_1404 = permute_998 = None
        view_1405: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_156, [8, 12, 1024, 64]);  bmm_156 = None
        view_1406: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_157, [8, 12, 1024, 1024]);  bmm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_72: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_58, torch.float32);  gt_58 = None
        mul_796: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
        mul_797: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1406, mul_796);  view_1406 = mul_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_798: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_797, div_18);  mul_797 = None
        sum_128: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_798, [-1], True)
        neg_23: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_21: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_23, sum_128, mul_798);  neg_23 = sum_128 = mul_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1407: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_21, [96, 1024, 1024]);  fma_21 = None
        view_1409: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1407, [8, 12, 1024, 1024]);  view_1407 = None
        view_1410: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1409, [96, 1024, 1024])
        add_291: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_280, view_1409);  add_280 = view_1409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_158: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_999, view_1410);  permute_999 = None
        bmm_159: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1410, permute_1000);  view_1410 = permute_1000 = None
        view_1412: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_158, [8, 12, 64, 1024]);  bmm_158 = None
        view_1413: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_159, [8, 12, 1024, 64]);  bmm_159 = None
        permute_1001: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1412, [0, 1, 3, 2]);  view_1412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1002: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1405, [0, 2, 1, 3]);  view_1405 = None
        clone_319: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1002, memory_format = torch.contiguous_format);  permute_1002 = None
        view_1414: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_319, [8, 1024, 768]);  clone_319 = None
        view_1415: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1414, [8192, 768]);  view_1414 = None
        permute_1003: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1415, [1, 0])
        mm_409: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1003, view_349);  permute_1003 = None
        permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        permute_1005: "f32[768, 768]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        mm_410: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1415, permute_1005);  view_1415 = permute_1005 = None
        view_1416: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_410, [8, 1024, 768]);  mm_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1007: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1001, [0, 2, 1, 3]);  permute_1001 = None
        view_1417: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1007, [8, 1024, 768]);  permute_1007 = None
        clone_320: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1417, memory_format = torch.contiguous_format);  view_1417 = None
        view_1418: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_320, [8192, 768]);  clone_320 = None
        permute_1008: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1418, [1, 0])
        mm_411: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1008, view_349);  permute_1008 = None
        permute_156: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        permute_1010: "f32[768, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_412: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1418, permute_1010);  view_1418 = permute_1010 = None
        view_1419: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_412, [8, 1024, 768]);  mm_412 = None
        add_292: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1416, view_1419);  view_1416 = view_1419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1012: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1413, [0, 2, 1, 3]);  view_1413 = None
        clone_321: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None
        view_1420: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_321, [8, 1024, 768]);  clone_321 = None
        view_1421: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1420, [8192, 768]);  view_1420 = None
        permute_1013: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1421, [1, 0])
        mm_413: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1013, view_349);  permute_1013 = view_349 = None
        permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        permute_1015: "f32[768, 768]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_414: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1421, permute_1015);  view_1421 = permute_1015 = None
        view_1422: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_414, [8, 1024, 768]);  mm_414 = None
        add_293: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_292, view_1422);  add_292 = view_1422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_799: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_293, primals_116);  primals_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_173: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_800: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_293, mul_173);  add_293 = mul_173 = None
        sum_129: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_800, [0, 1], True);  mul_800 = None
        view_1423: "f32[768]" = torch.ops.aten.reshape.default(sum_129, [768]);  sum_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_801: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_799, add_84)
        mul_802: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_799, rsqrt_28);  mul_799 = None
        sum_130: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_801, [2], True);  mul_801 = None
        add_294: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_290, mul_802);  add_290 = mul_802 = None
        pow_129: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_28, 3);  rsqrt_28 = None
        mul_803: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_130, -0.5);  sum_130 = None
        mul_804: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_803, pow_129);  mul_803 = pow_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_180: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_804, [8, 1024, 768]);  mul_804 = None
        div_75: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_180, 768);  expand_180 = None
        pow_130: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 1.0);  add_84 = None
        mul_805: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_130, 2.0);  pow_130 = None
        mul_806: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_75, mul_805);  div_75 = mul_805 = None
        add_295: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_294, mul_806);  add_294 = mul_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_73: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_807: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
        mul_808: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_295, mul_807);  mul_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1424: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_808, [8192, 768]);  mul_808 = None
        permute_1017: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1424, [1, 0])
        mm_415: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1017, view_347);  permute_1017 = view_347 = None
        permute_153: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_1019: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_416: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1424, permute_1019);  view_1424 = permute_1019 = None
        view_1425: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_416, [8, 1024, 3072]);  mm_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_74: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_56, torch.float32);  gt_56 = None
        mul_809: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1.1111111111111112);  convert_element_type_74 = None
        mul_810: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1425, mul_809);  view_1425 = mul_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_21: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_12, full_default, mul_810);  le_12 = mul_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1426: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_21, [8192, 3072]);  where_21 = None
        permute_1021: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1426, [1, 0])
        mm_417: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1021, view_345);  permute_1021 = view_345 = None
        permute_152: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_1023: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_418: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1426, permute_1023);  view_1426 = permute_1023 = None
        view_1427: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_418, [8, 1024, 768]);  mm_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_811: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1427, primals_113);  primals_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_167: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_82, rsqrt_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_812: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1427, mul_167);  view_1427 = mul_167 = None
        sum_131: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_812, [0, 1], True);  mul_812 = None
        view_1428: "f32[768]" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_813: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_811, add_82)
        mul_814: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_811, rsqrt_27);  mul_811 = None
        sum_132: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_813, [2], True);  mul_813 = None
        add_296: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_295, mul_814);  add_295 = mul_814 = None
        pow_131: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_27, 3);  rsqrt_27 = None
        mul_815: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_132, -0.5);  sum_132 = None
        mul_816: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_815, pow_131);  mul_815 = pow_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_181: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_816, [8, 1024, 768]);  mul_816 = None
        div_76: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_181, 768);  expand_181 = None
        pow_132: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_82, 1.0);  add_82 = None
        mul_817: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_132, 2.0);  pow_132 = None
        mul_818: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_76, mul_817);  div_76 = mul_817 = None
        add_297: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_296, mul_818);  add_296 = mul_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_75: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_55, torch.float32);  gt_55 = None
        mul_819: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 1.1111111111111112);  convert_element_type_75 = None
        mul_820: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_297, mul_819);  mul_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1429: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_820, [8192, 768]);  mul_820 = None
        permute_1025: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1429, [1, 0])
        mm_419: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1025, view_343);  permute_1025 = view_343 = None
        permute_151: "f32[768, 768]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        permute_1027: "f32[768, 768]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_420: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1429, permute_1027);  view_1429 = permute_1027 = None
        view_1430: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_420, [8, 1024, 768]);  mm_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1431: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1430, [8, 1024, 12, 64]);  view_1430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1029: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1431, [0, 2, 1, 3]);  view_1431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_325: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1029, memory_format = torch.contiguous_format);  permute_1029 = None
        view_1432: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_325, [96, 1024, 64]);  clone_325 = None
        bmm_160: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1030, view_1432);  permute_1030 = None
        bmm_161: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1432, permute_1031);  view_1432 = permute_1031 = None
        view_1433: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_160, [8, 12, 1024, 64]);  bmm_160 = None
        view_1434: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_161, [8, 12, 1024, 1024]);  bmm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_76: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_54, torch.float32);  gt_54 = None
        mul_821: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 1.1111111111111112);  convert_element_type_76 = None
        mul_822: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1434, mul_821);  view_1434 = mul_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_823: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_822, div_17);  mul_822 = None
        sum_133: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_823, [-1], True)
        neg_24: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_22: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_24, sum_133, mul_823);  neg_24 = sum_133 = mul_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1435: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_22, [96, 1024, 1024]);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_162: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1032, view_1435);  permute_1032 = None
        bmm_163: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1435, permute_1033);  view_1435 = permute_1033 = None
        view_1440: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_162, [8, 12, 64, 1024]);  bmm_162 = None
        view_1441: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_163, [8, 12, 1024, 64]);  bmm_163 = None
        permute_1034: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1440, [0, 1, 3, 2]);  view_1440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1035: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1433, [0, 2, 1, 3]);  view_1433 = None
        clone_328: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1035, memory_format = torch.contiguous_format);  permute_1035 = None
        view_1442: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_328, [8, 1024, 768]);  clone_328 = None
        view_1443: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1442, [8192, 768]);  view_1442 = None
        permute_1036: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1443, [1, 0])
        view_330: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768])
        mm_421: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1036, view_330);  permute_1036 = view_330 = None
        permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_1038: "f32[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        mm_422: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1443, permute_1038);  view_1443 = permute_1038 = None
        view_1444: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_422, [8, 1024, 768]);  mm_422 = None
        add_298: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_288, view_1444);  add_288 = view_1444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1040: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1034, [0, 2, 1, 3]);  permute_1034 = None
        view_1445: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1040, [8, 1024, 768]);  permute_1040 = None
        clone_329: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1445, memory_format = torch.contiguous_format);  view_1445 = None
        view_1446: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_329, [8192, 768]);  clone_329 = None
        permute_1041: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1446, [1, 0])
        view_327: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_151, [8192, 768]);  mul_151 = None
        mm_423: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1041, view_327);  permute_1041 = view_327 = None
        permute_145: "f32[768, 768]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        permute_1043: "f32[768, 768]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_424: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1446, permute_1043);  view_1446 = permute_1043 = None
        view_1447: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_424, [8, 1024, 768]);  mm_424 = None
        add_299: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_298, view_1447);  add_298 = view_1447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1045: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1441, [0, 2, 1, 3]);  view_1441 = None
        clone_330: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1045, memory_format = torch.contiguous_format);  permute_1045 = None
        view_1448: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_330, [8, 1024, 768]);  clone_330 = None
        view_1449: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1448, [8192, 768]);  view_1448 = None
        permute_1046: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1449, [1, 0])
        mm_425: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1046, view_324);  permute_1046 = view_324 = None
        permute_143: "f32[768, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_1048: "f32[768, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_426: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1449, permute_1048);  view_1449 = permute_1048 = None
        view_1450: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_426, [8, 1024, 768]);  mm_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_824: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1450, primals_108);  primals_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_161: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_825: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1450, mul_161);  view_1450 = mul_161 = None
        sum_134: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_825, [0, 1], True);  mul_825 = None
        view_1451: "f32[768]" = torch.ops.aten.reshape.default(sum_134, [768]);  sum_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_826: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_824, add_78)
        mul_827: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_824, rsqrt_26);  mul_824 = None
        sum_135: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_826, [2], True);  mul_826 = None
        add_300: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_297, mul_827);  add_297 = mul_827 = None
        pow_133: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_26, 3);  rsqrt_26 = None
        mul_828: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_135, -0.5);  sum_135 = None
        mul_829: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_828, pow_133);  mul_828 = pow_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_182: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_829, [8, 1024, 768]);  mul_829 = None
        div_77: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_182, 768);  expand_182 = None
        pow_134: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 1.0);  add_78 = None
        mul_830: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_134, 2.0);  pow_134 = None
        mul_831: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_77, mul_830);  div_77 = mul_830 = None
        add_301: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_300, mul_831);  add_300 = mul_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_77: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_53, torch.float32);  gt_53 = None
        mul_832: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_77, 1.1111111111111112);  convert_element_type_77 = None
        mul_833: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_301, mul_832);  mul_832 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1452: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_833, [8192, 768]);  mul_833 = None
        permute_1050: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1452, [1, 0])
        mm_427: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1050, view_322);  permute_1050 = view_322 = None
        permute_142: "f32[768, 768]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        permute_1052: "f32[768, 768]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_428: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1452, permute_1052);  view_1452 = permute_1052 = None
        view_1453: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_428, [8, 1024, 768]);  mm_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1454: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1453, [8, 1024, 12, 64]);  view_1453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1054: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1454, [0, 2, 1, 3]);  view_1454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_332: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1054, memory_format = torch.contiguous_format);  permute_1054 = None
        view_1455: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_332, [96, 1024, 64]);  clone_332 = None
        bmm_164: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1055, view_1455);  permute_1055 = None
        bmm_165: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1455, permute_1056);  view_1455 = permute_1056 = None
        view_1456: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_164, [8, 12, 1024, 64]);  bmm_164 = None
        view_1457: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_165, [8, 12, 1024, 1024]);  bmm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_78: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_52, torch.float32);  gt_52 = None
        mul_834: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 1.1111111111111112);  convert_element_type_78 = None
        mul_835: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1457, mul_834);  view_1457 = mul_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_836: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_835, div_16);  mul_835 = None
        sum_136: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_836, [-1], True)
        neg_25: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_23: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_25, sum_136, mul_836);  neg_25 = sum_136 = mul_836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1458: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_23, [96, 1024, 1024]);  fma_23 = None
        view_1460: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1458, [8, 12, 1024, 1024]);  view_1458 = None
        view_1461: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1460, [96, 1024, 1024])
        add_302: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_291, view_1460);  add_291 = view_1460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_137: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_302, [0], True);  add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_1: "f32[12, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_137, 0);  sum_137 = None
        permute_1057: "f32[1024, 1024, 12]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_1: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_75, -1)
        unsqueeze_20: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_22: "f32[1024, 1024, 12]" = torch.ops.aten.where.self(unsqueeze_20, full_default, permute_1057);  unsqueeze_20 = permute_1057 = None
        clone_335: "f32[1024, 1024, 12]" = torch.ops.aten.clone.default(where_22, memory_format = torch.contiguous_format);  where_22 = None
        full_default_30: "f32[32, 12]" = torch.ops.aten.full.default([32, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[32, 12]" = torch.ops.aten.index_put.default(full_default_30, [add_75], clone_335, True);  add_75 = clone_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_166: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1058, view_1461);  permute_1058 = None
        bmm_167: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1461, permute_1059);  view_1461 = permute_1059 = None
        view_1463: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_166, [8, 12, 64, 1024]);  bmm_166 = None
        view_1464: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_167, [8, 12, 1024, 64]);  bmm_167 = None
        permute_1060: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1463, [0, 1, 3, 2]);  view_1463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1061: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1456, [0, 2, 1, 3]);  view_1456 = None
        clone_336: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1061, memory_format = torch.contiguous_format);  permute_1061 = None
        view_1465: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_336, [8, 1024, 768]);  clone_336 = None
        view_1466: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1465, [8192, 768]);  view_1465 = None
        permute_1062: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1466, [1, 0])
        mm_429: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1062, view_303);  permute_1062 = None
        permute_137: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_1064: "f32[768, 768]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        mm_430: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1466, permute_1064);  view_1466 = permute_1064 = None
        view_1467: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_430, [8, 1024, 768]);  mm_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1066: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1060, [0, 2, 1, 3]);  permute_1060 = None
        view_1468: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1066, [8, 1024, 768]);  permute_1066 = None
        clone_337: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1468, memory_format = torch.contiguous_format);  view_1468 = None
        view_1469: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_337, [8192, 768]);  clone_337 = None
        permute_1067: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1469, [1, 0])
        mm_431: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1067, view_303);  permute_1067 = None
        permute_135: "f32[768, 768]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_1069: "f32[768, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
        mm_432: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1469, permute_1069);  view_1469 = permute_1069 = None
        view_1470: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_432, [8, 1024, 768]);  mm_432 = None
        add_303: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1467, view_1470);  view_1467 = view_1470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1071: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1464, [0, 2, 1, 3]);  view_1464 = None
        clone_338: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1071, memory_format = torch.contiguous_format);  permute_1071 = None
        view_1471: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_338, [8, 1024, 768]);  clone_338 = None
        view_1472: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1471, [8192, 768]);  view_1471 = None
        permute_1072: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1472, [1, 0])
        mm_433: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1072, view_303);  permute_1072 = view_303 = None
        permute_133: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        permute_1074: "f32[768, 768]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_434: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1472, permute_1074);  view_1472 = permute_1074 = None
        view_1473: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_434, [8, 1024, 768]);  mm_434 = None
        add_304: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_303, view_1473);  add_303 = view_1473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_837: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_304, primals_102);  primals_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_152: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt_51, embedding_2);  embedding_2 = None
        mul_153: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_152, 1.1111111111111112);  mul_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_154: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_153, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_838: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_304, mul_154);  add_304 = mul_154 = None
        sum_138: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_838, [0, 1], True);  mul_838 = None
        view_1474: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_839: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_837, mul_153)
        mul_840: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_837, rsqrt_25);  mul_837 = None
        sum_139: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_839, [2], True);  mul_839 = None
        add_305: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_301, mul_840);  add_301 = mul_840 = None
        pow_135: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_25, 3);  rsqrt_25 = None
        mul_841: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_139, -0.5);  sum_139 = None
        mul_842: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_841, pow_135);  mul_841 = pow_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_183: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_842, [8, 1024, 768]);  mul_842 = None
        div_78: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_183, 768);  expand_183 = None
        pow_136: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_153, 1.0);  mul_153 = None
        mul_843: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_136, 2.0);  pow_136 = None
        mul_844: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_78, mul_843);  div_78 = mul_843 = None
        add_306: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_305, mul_844);  add_305 = mul_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_79: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_845: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_79, 1.1111111111111112);  convert_element_type_79 = None
        mul_846: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_306, mul_845);  add_306 = mul_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_2: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(where_2, -1)
        unsqueeze_21: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_23: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_21, full_default, mul_846);  unsqueeze_21 = mul_846 = None
        full_default_32: "f32[32128, 768]" = torch.ops.aten.full.default([32128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[32128, 768]" = torch.ops.aten.index_put.default(full_default_32, [where_2], where_23, True);  where_2 = where_23 = None
        add_307: "f32[32128, 768]" = torch.ops.aten.add.Tensor(mm_193, index_put_1);  mm_193 = index_put_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_80: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_50, torch.float32);  gt_50 = None
        mul_847: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
        mul_848: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_299, mul_847);  add_299 = mul_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_849: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_848, primals_100);  primals_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_148: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_66, rsqrt_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_850: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_848, mul_148);  mul_848 = mul_148 = None
        sum_140: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_850, [0, 1], True);  mul_850 = None
        view_1475: "f32[768]" = torch.ops.aten.reshape.default(sum_140, [768]);  sum_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_851: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_849, add_66)
        mul_852: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_849, rsqrt_24);  mul_849 = None
        sum_141: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_851, [2], True);  mul_851 = None
        pow_137: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_24, 3);  rsqrt_24 = None
        mul_853: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_141, -0.5);  sum_141 = None
        mul_854: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_853, pow_137);  mul_853 = pow_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_184: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_854, [8, 1024, 768]);  mul_854 = None
        div_79: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_184, 768);  expand_184 = None
        pow_138: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_66, 1.0);  add_66 = None
        mul_855: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_138, 2.0);  pow_138 = None
        mul_856: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_79, mul_855);  div_79 = mul_855 = None
        add_308: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(mul_852, mul_856);  mul_852 = mul_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_81: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_49, torch.float32);  gt_49 = None
        mul_857: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 1.1111111111111112);  convert_element_type_81 = None
        mul_858: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_308, mul_857);  mul_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1476: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_858, [8192, 768]);  mul_858 = None
        permute_1076: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1476, [1, 0])
        mm_435: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1076, view_299);  permute_1076 = view_299 = None
        permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_1078: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_436: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1476, permute_1078);  view_1476 = permute_1078 = None
        view_1477: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_436, [8, 1024, 3072]);  mm_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_82: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_48, torch.float32);  gt_48 = None
        mul_859: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 1.1111111111111112);  convert_element_type_82 = None
        mul_860: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1477, mul_859);  view_1477 = mul_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_24: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_13, full_default, mul_860);  le_13 = mul_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1478: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_24, [8192, 3072]);  where_24 = None
        permute_1080: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1478, [1, 0])
        mm_437: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1080, view_297);  permute_1080 = view_297 = None
        permute_131: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        permute_1082: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_438: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1478, permute_1082);  view_1478 = permute_1082 = None
        view_1479: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_438, [8, 1024, 768]);  mm_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_861: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1479, primals_97);  primals_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_142: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_64, rsqrt_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_862: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1479, mul_142);  view_1479 = mul_142 = None
        sum_142: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_862, [0, 1], True);  mul_862 = None
        view_1480: "f32[768]" = torch.ops.aten.reshape.default(sum_142, [768]);  sum_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_863: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_861, add_64)
        mul_864: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_861, rsqrt_23);  mul_861 = None
        sum_143: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_863, [2], True);  mul_863 = None
        add_309: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_308, mul_864);  add_308 = mul_864 = None
        pow_139: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_23, 3);  rsqrt_23 = None
        mul_865: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_143, -0.5);  sum_143 = None
        mul_866: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_865, pow_139);  mul_865 = pow_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_185: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_866, [8, 1024, 768]);  mul_866 = None
        div_80: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_185, 768);  expand_185 = None
        pow_140: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_64, 1.0);  add_64 = None
        mul_867: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_140, 2.0);  pow_140 = None
        mul_868: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_80, mul_867);  div_80 = mul_867 = None
        add_310: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_309, mul_868);  add_309 = mul_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_83: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_47, torch.float32);  gt_47 = None
        mul_869: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_83, 1.1111111111111112);  convert_element_type_83 = None
        mul_870: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_310, mul_869);  mul_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1481: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_870, [8192, 768]);  mul_870 = None
        permute_1084: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1481, [1, 0])
        mm_439: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1084, view_295);  permute_1084 = view_295 = None
        permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_1086: "f32[768, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_440: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1481, permute_1086);  view_1481 = permute_1086 = None
        view_1482: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_440, [8, 1024, 768]);  mm_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1483: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1482, [8, 1024, 12, 64]);  view_1482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1088: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1483, [0, 2, 1, 3]);  view_1483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_344: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1088, memory_format = torch.contiguous_format);  permute_1088 = None
        view_1484: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_344, [96, 1024, 64]);  clone_344 = None
        bmm_168: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1089, view_1484);  permute_1089 = None
        bmm_169: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1484, permute_1090);  view_1484 = permute_1090 = None
        view_1485: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_168, [8, 12, 1024, 64]);  bmm_168 = None
        view_1486: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_169, [8, 12, 1024, 1024]);  bmm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_84: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_46, torch.float32);  gt_46 = None
        mul_871: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1.1111111111111112);  convert_element_type_84 = None
        mul_872: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1486, mul_871);  view_1486 = mul_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_873: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_872, div_13);  mul_872 = None
        sum_144: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_873, [-1], True)
        neg_26: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_24: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_26, sum_144, mul_873);  neg_26 = sum_144 = mul_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1487: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_24, [96, 1024, 1024]);  fma_24 = None
        view_1489: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1487, [8, 12, 1024, 1024]);  view_1487 = None
        view_1490: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1489, [96, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_170: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1091, view_1490);  permute_1091 = None
        bmm_171: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1490, permute_1092);  view_1490 = permute_1092 = None
        view_1492: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_170, [8, 12, 64, 1024]);  bmm_170 = None
        view_1493: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_171, [8, 12, 1024, 64]);  bmm_171 = None
        permute_1093: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1492, [0, 1, 3, 2]);  view_1492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1094: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1485, [0, 2, 1, 3]);  view_1485 = None
        clone_347: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1094, memory_format = torch.contiguous_format);  permute_1094 = None
        view_1494: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_347, [8, 1024, 768]);  clone_347 = None
        view_1495: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1494, [8192, 768]);  view_1494 = None
        permute_1095: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1495, [1, 0])
        mm_441: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1095, view_276);  permute_1095 = None
        permute_126: "f32[768, 768]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_1097: "f32[768, 768]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
        mm_442: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1495, permute_1097);  view_1495 = permute_1097 = None
        view_1496: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_442, [8, 1024, 768]);  mm_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1099: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1093, [0, 2, 1, 3]);  permute_1093 = None
        view_1497: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1099, [8, 1024, 768]);  permute_1099 = None
        clone_348: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1497, memory_format = torch.contiguous_format);  view_1497 = None
        view_1498: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_348, [8192, 768]);  clone_348 = None
        permute_1100: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1498, [1, 0])
        mm_443: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1100, view_276);  permute_1100 = None
        permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_1102: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
        mm_444: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1498, permute_1102);  view_1498 = permute_1102 = None
        view_1499: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_444, [8, 1024, 768]);  mm_444 = None
        add_311: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1496, view_1499);  view_1496 = view_1499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1104: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1493, [0, 2, 1, 3]);  view_1493 = None
        clone_349: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1104, memory_format = torch.contiguous_format);  permute_1104 = None
        view_1500: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_349, [8, 1024, 768]);  clone_349 = None
        view_1501: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1500, [8192, 768]);  view_1500 = None
        permute_1105: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1501, [1, 0])
        mm_445: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1105, view_276);  permute_1105 = view_276 = None
        permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_1107: "f32[768, 768]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_446: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1501, permute_1107);  view_1501 = permute_1107 = None
        view_1502: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_446, [8, 1024, 768]);  mm_446 = None
        add_312: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_311, view_1502);  add_311 = view_1502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_874: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_312, primals_92);  primals_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_136: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_61, rsqrt_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_875: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_312, mul_136);  add_312 = mul_136 = None
        sum_145: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_875, [0, 1], True);  mul_875 = None
        view_1503: "f32[768]" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_876: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_874, add_61)
        mul_877: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_874, rsqrt_22);  mul_874 = None
        sum_146: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_876, [2], True);  mul_876 = None
        add_313: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_310, mul_877);  add_310 = mul_877 = None
        pow_141: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_22, 3);  rsqrt_22 = None
        mul_878: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_146, -0.5);  sum_146 = None
        mul_879: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_878, pow_141);  mul_878 = pow_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_186: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_879, [8, 1024, 768]);  mul_879 = None
        div_81: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_186, 768);  expand_186 = None
        pow_142: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_61, 1.0);  add_61 = None
        mul_880: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_142, 2.0);  pow_142 = None
        mul_881: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_81, mul_880);  div_81 = mul_880 = None
        add_314: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_313, mul_881);  add_313 = mul_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_85: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_45, torch.float32);  gt_45 = None
        mul_882: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 1.1111111111111112);  convert_element_type_85 = None
        mul_883: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_314, mul_882);  mul_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1504: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_883, [8192, 768]);  mul_883 = None
        permute_1109: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1504, [1, 0])
        mm_447: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1109, view_274);  permute_1109 = view_274 = None
        permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_1111: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_448: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1504, permute_1111);  view_1504 = permute_1111 = None
        view_1505: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_448, [8, 1024, 3072]);  mm_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_86: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_44, torch.float32);  gt_44 = None
        mul_884: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 1.1111111111111112);  convert_element_type_86 = None
        mul_885: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1505, mul_884);  view_1505 = mul_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_25: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_14, full_default, mul_885);  le_14 = mul_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1506: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_25, [8192, 3072]);  where_25 = None
        permute_1113: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1506, [1, 0])
        mm_449: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1113, view_272);  permute_1113 = view_272 = None
        permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_1115: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_450: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1506, permute_1115);  view_1506 = permute_1115 = None
        view_1507: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_450, [8, 1024, 768]);  mm_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_886: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1507, primals_89);  primals_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_130: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_59, rsqrt_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_887: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1507, mul_130);  view_1507 = mul_130 = None
        sum_147: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 1], True);  mul_887 = None
        view_1508: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_888: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_886, add_59)
        mul_889: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_886, rsqrt_21);  mul_886 = None
        sum_148: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_888, [2], True);  mul_888 = None
        add_315: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_314, mul_889);  add_314 = mul_889 = None
        pow_143: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_21, 3);  rsqrt_21 = None
        mul_890: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_148, -0.5);  sum_148 = None
        mul_891: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_890, pow_143);  mul_890 = pow_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_187: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_891, [8, 1024, 768]);  mul_891 = None
        div_82: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_187, 768);  expand_187 = None
        pow_144: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_59, 1.0);  add_59 = None
        mul_892: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_144, 2.0);  pow_144 = None
        mul_893: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_82, mul_892);  div_82 = mul_892 = None
        add_316: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_315, mul_893);  add_315 = mul_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_87: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_43, torch.float32);  gt_43 = None
        mul_894: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
        mul_895: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_316, mul_894);  mul_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1509: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_895, [8192, 768]);  mul_895 = None
        permute_1117: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1509, [1, 0])
        mm_451: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1117, view_270);  permute_1117 = view_270 = None
        permute_119: "f32[768, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_1119: "f32[768, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_452: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1509, permute_1119);  view_1509 = permute_1119 = None
        view_1510: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_452, [8, 1024, 768]);  mm_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1511: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1510, [8, 1024, 12, 64]);  view_1510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1121: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1511, [0, 2, 1, 3]);  view_1511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_353: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1121, memory_format = torch.contiguous_format);  permute_1121 = None
        view_1512: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_353, [96, 1024, 64]);  clone_353 = None
        bmm_172: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1122, view_1512);  permute_1122 = None
        bmm_173: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1512, permute_1123);  view_1512 = permute_1123 = None
        view_1513: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_172, [8, 12, 1024, 64]);  bmm_172 = None
        view_1514: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_173, [8, 12, 1024, 1024]);  bmm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_88: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_42, torch.float32);  gt_42 = None
        mul_896: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 1.1111111111111112);  convert_element_type_88 = None
        mul_897: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1514, mul_896);  view_1514 = mul_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_898: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_897, div_12);  mul_897 = None
        sum_149: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_898, [-1], True)
        neg_27: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_25: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_27, sum_149, mul_898);  neg_27 = sum_149 = mul_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1515: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_25, [96, 1024, 1024]);  fma_25 = None
        view_1517: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1515, [8, 12, 1024, 1024]);  view_1515 = None
        view_1518: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1517, [96, 1024, 1024])
        add_317: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_1489, view_1517);  view_1489 = view_1517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_174: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1124, view_1518);  permute_1124 = None
        bmm_175: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1518, permute_1125);  view_1518 = permute_1125 = None
        view_1520: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_174, [8, 12, 64, 1024]);  bmm_174 = None
        view_1521: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_175, [8, 12, 1024, 64]);  bmm_175 = None
        permute_1126: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1520, [0, 1, 3, 2]);  view_1520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1127: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1513, [0, 2, 1, 3]);  view_1513 = None
        clone_356: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1127, memory_format = torch.contiguous_format);  permute_1127 = None
        view_1522: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_356, [8, 1024, 768]);  clone_356 = None
        view_1523: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1522, [8192, 768]);  view_1522 = None
        permute_1128: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1523, [1, 0])
        mm_453: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1128, view_251);  permute_1128 = None
        permute_115: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_1130: "f32[768, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
        mm_454: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1523, permute_1130);  view_1523 = permute_1130 = None
        view_1524: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_454, [8, 1024, 768]);  mm_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1132: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1126, [0, 2, 1, 3]);  permute_1126 = None
        view_1525: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1132, [8, 1024, 768]);  permute_1132 = None
        clone_357: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1525, memory_format = torch.contiguous_format);  view_1525 = None
        view_1526: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_357, [8192, 768]);  clone_357 = None
        permute_1133: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1526, [1, 0])
        mm_455: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1133, view_251);  permute_1133 = None
        permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_1135: "f32[768, 768]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        mm_456: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1526, permute_1135);  view_1526 = permute_1135 = None
        view_1527: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_456, [8, 1024, 768]);  mm_456 = None
        add_318: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1524, view_1527);  view_1524 = view_1527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1137: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1521, [0, 2, 1, 3]);  view_1521 = None
        clone_358: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1137, memory_format = torch.contiguous_format);  permute_1137 = None
        view_1528: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_358, [8, 1024, 768]);  clone_358 = None
        view_1529: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1528, [8192, 768]);  view_1528 = None
        permute_1138: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1529, [1, 0])
        mm_457: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1138, view_251);  permute_1138 = view_251 = None
        permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_1140: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_458: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1529, permute_1140);  view_1529 = permute_1140 = None
        view_1530: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_458, [8, 1024, 768]);  mm_458 = None
        add_319: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_318, view_1530);  add_318 = view_1530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_899: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_319, primals_84);  primals_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_124: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_56, rsqrt_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_900: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_319, mul_124);  add_319 = mul_124 = None
        sum_150: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_900, [0, 1], True);  mul_900 = None
        view_1531: "f32[768]" = torch.ops.aten.reshape.default(sum_150, [768]);  sum_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_901: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_899, add_56)
        mul_902: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_899, rsqrt_20);  mul_899 = None
        sum_151: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_901, [2], True);  mul_901 = None
        add_320: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_316, mul_902);  add_316 = mul_902 = None
        pow_145: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_20, 3);  rsqrt_20 = None
        mul_903: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_151, -0.5);  sum_151 = None
        mul_904: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_903, pow_145);  mul_903 = pow_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_188: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_904, [8, 1024, 768]);  mul_904 = None
        div_83: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_188, 768);  expand_188 = None
        pow_146: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_56, 1.0);  add_56 = None
        mul_905: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_146, 2.0);  pow_146 = None
        mul_906: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_83, mul_905);  div_83 = mul_905 = None
        add_321: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_320, mul_906);  add_320 = mul_906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_89: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_41, torch.float32);  gt_41 = None
        mul_907: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.1111111111111112);  convert_element_type_89 = None
        mul_908: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_321, mul_907);  mul_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1532: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_908, [8192, 768]);  mul_908 = None
        permute_1142: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1532, [1, 0])
        mm_459: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1142, view_249);  permute_1142 = view_249 = None
        permute_110: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_1144: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_460: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1532, permute_1144);  view_1532 = permute_1144 = None
        view_1533: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_460, [8, 1024, 3072]);  mm_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_90: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_40, torch.float32);  gt_40 = None
        mul_909: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 1.1111111111111112);  convert_element_type_90 = None
        mul_910: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1533, mul_909);  view_1533 = mul_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_26: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_15, full_default, mul_910);  le_15 = mul_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1534: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_26, [8192, 3072]);  where_26 = None
        permute_1146: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1534, [1, 0])
        mm_461: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1146, view_247);  permute_1146 = view_247 = None
        permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        permute_1148: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_462: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1534, permute_1148);  view_1534 = permute_1148 = None
        view_1535: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_462, [8, 1024, 768]);  mm_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_911: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1535, primals_81);  primals_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_118: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_912: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1535, mul_118);  view_1535 = mul_118 = None
        sum_152: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_912, [0, 1], True);  mul_912 = None
        view_1536: "f32[768]" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_913: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_911, add_54)
        mul_914: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_911, rsqrt_19);  mul_911 = None
        sum_153: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_913, [2], True);  mul_913 = None
        add_322: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_321, mul_914);  add_321 = mul_914 = None
        pow_147: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_19, 3);  rsqrt_19 = None
        mul_915: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_153, -0.5);  sum_153 = None
        mul_916: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_915, pow_147);  mul_915 = pow_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_189: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_916, [8, 1024, 768]);  mul_916 = None
        div_84: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_189, 768);  expand_189 = None
        pow_148: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 1.0);  add_54 = None
        mul_917: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_148, 2.0);  pow_148 = None
        mul_918: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_84, mul_917);  div_84 = mul_917 = None
        add_323: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_322, mul_918);  add_322 = mul_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_91: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_39, torch.float32);  gt_39 = None
        mul_919: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 1.1111111111111112);  convert_element_type_91 = None
        mul_920: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_323, mul_919);  mul_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1537: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_920, [8192, 768]);  mul_920 = None
        permute_1150: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1537, [1, 0])
        mm_463: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1150, view_245);  permute_1150 = view_245 = None
        permute_108: "f32[768, 768]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_1152: "f32[768, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_464: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1537, permute_1152);  view_1537 = permute_1152 = None
        view_1538: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_464, [8, 1024, 768]);  mm_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1539: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1538, [8, 1024, 12, 64]);  view_1538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1154: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1539, [0, 2, 1, 3]);  view_1539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_362: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1154, memory_format = torch.contiguous_format);  permute_1154 = None
        view_1540: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_362, [96, 1024, 64]);  clone_362 = None
        bmm_176: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1155, view_1540);  permute_1155 = None
        bmm_177: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1540, permute_1156);  view_1540 = permute_1156 = None
        view_1541: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_176, [8, 12, 1024, 64]);  bmm_176 = None
        view_1542: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_177, [8, 12, 1024, 1024]);  bmm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_92: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_38, torch.float32);  gt_38 = None
        mul_921: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 1.1111111111111112);  convert_element_type_92 = None
        mul_922: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1542, mul_921);  view_1542 = mul_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_923: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_922, div_11);  mul_922 = None
        sum_154: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_923, [-1], True)
        neg_28: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_26: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_28, sum_154, mul_923);  neg_28 = sum_154 = mul_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1543: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_26, [96, 1024, 1024]);  fma_26 = None
        view_1545: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1543, [8, 12, 1024, 1024]);  view_1543 = None
        view_1546: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1545, [96, 1024, 1024])
        add_324: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_317, view_1545);  add_317 = view_1545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_178: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1157, view_1546);  permute_1157 = None
        bmm_179: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1546, permute_1158);  view_1546 = permute_1158 = None
        view_1548: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_178, [8, 12, 64, 1024]);  bmm_178 = None
        view_1549: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_179, [8, 12, 1024, 64]);  bmm_179 = None
        permute_1159: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1548, [0, 1, 3, 2]);  view_1548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1160: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1541, [0, 2, 1, 3]);  view_1541 = None
        clone_365: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1160, memory_format = torch.contiguous_format);  permute_1160 = None
        view_1550: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_365, [8, 1024, 768]);  clone_365 = None
        view_1551: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1550, [8192, 768]);  view_1550 = None
        permute_1161: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1551, [1, 0])
        mm_465: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1161, view_226);  permute_1161 = None
        permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_1163: "f32[768, 768]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
        mm_466: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1551, permute_1163);  view_1551 = permute_1163 = None
        view_1552: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_466, [8, 1024, 768]);  mm_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1165: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1159, [0, 2, 1, 3]);  permute_1159 = None
        view_1553: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1165, [8, 1024, 768]);  permute_1165 = None
        clone_366: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1553, memory_format = torch.contiguous_format);  view_1553 = None
        view_1554: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_366, [8192, 768]);  clone_366 = None
        permute_1166: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1554, [1, 0])
        mm_467: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1166, view_226);  permute_1166 = None
        permute_102: "f32[768, 768]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_1168: "f32[768, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        mm_468: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1554, permute_1168);  view_1554 = permute_1168 = None
        view_1555: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_468, [8, 1024, 768]);  mm_468 = None
        add_325: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1552, view_1555);  view_1552 = view_1555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1170: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1549, [0, 2, 1, 3]);  view_1549 = None
        clone_367: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1170, memory_format = torch.contiguous_format);  permute_1170 = None
        view_1556: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_367, [8, 1024, 768]);  clone_367 = None
        view_1557: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1556, [8192, 768]);  view_1556 = None
        permute_1171: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1557, [1, 0])
        mm_469: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1171, view_226);  permute_1171 = view_226 = None
        permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_1173: "f32[768, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_470: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1557, permute_1173);  view_1557 = permute_1173 = None
        view_1558: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_470, [8, 1024, 768]);  mm_470 = None
        add_326: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_325, view_1558);  add_325 = view_1558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_924: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_326, primals_76);  primals_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_112: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_51, rsqrt_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_925: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_326, mul_112);  add_326 = mul_112 = None
        sum_155: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_925, [0, 1], True);  mul_925 = None
        view_1559: "f32[768]" = torch.ops.aten.reshape.default(sum_155, [768]);  sum_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_926: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_924, add_51)
        mul_927: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_924, rsqrt_18);  mul_924 = None
        sum_156: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_926, [2], True);  mul_926 = None
        add_327: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_323, mul_927);  add_323 = mul_927 = None
        pow_149: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_18, 3);  rsqrt_18 = None
        mul_928: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_156, -0.5);  sum_156 = None
        mul_929: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_928, pow_149);  mul_928 = pow_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_190: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_929, [8, 1024, 768]);  mul_929 = None
        div_85: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_190, 768);  expand_190 = None
        pow_150: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_51, 1.0);  add_51 = None
        mul_930: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_150, 2.0);  pow_150 = None
        mul_931: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_85, mul_930);  div_85 = mul_930 = None
        add_328: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_327, mul_931);  add_327 = mul_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_93: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_932: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_93, 1.1111111111111112);  convert_element_type_93 = None
        mul_933: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_328, mul_932);  mul_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1560: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_933, [8192, 768]);  mul_933 = None
        permute_1175: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1560, [1, 0])
        mm_471: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1175, view_224);  permute_1175 = view_224 = None
        permute_99: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_1177: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_472: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1560, permute_1177);  view_1560 = permute_1177 = None
        view_1561: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_472, [8, 1024, 3072]);  mm_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_94: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_934: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 1.1111111111111112);  convert_element_type_94 = None
        mul_935: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1561, mul_934);  view_1561 = mul_934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_27: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_16, full_default, mul_935);  le_16 = mul_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1562: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_27, [8192, 3072]);  where_27 = None
        permute_1179: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1562, [1, 0])
        mm_473: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1179, view_222);  permute_1179 = view_222 = None
        permute_98: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_1181: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_474: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1562, permute_1181);  view_1562 = permute_1181 = None
        view_1563: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_474, [8, 1024, 768]);  mm_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_936: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1563, primals_73);  primals_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_106: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_49, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_937: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1563, mul_106);  view_1563 = mul_106 = None
        sum_157: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 1], True);  mul_937 = None
        view_1564: "f32[768]" = torch.ops.aten.reshape.default(sum_157, [768]);  sum_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_938: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_936, add_49)
        mul_939: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_936, rsqrt_17);  mul_936 = None
        sum_158: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_938, [2], True);  mul_938 = None
        add_329: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_328, mul_939);  add_328 = mul_939 = None
        pow_151: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_17, 3);  rsqrt_17 = None
        mul_940: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_158, -0.5);  sum_158 = None
        mul_941: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_940, pow_151);  mul_940 = pow_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_191: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_941, [8, 1024, 768]);  mul_941 = None
        div_86: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_191, 768);  expand_191 = None
        pow_152: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_49, 1.0);  add_49 = None
        mul_942: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_152, 2.0);  pow_152 = None
        mul_943: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_86, mul_942);  div_86 = mul_942 = None
        add_330: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_329, mul_943);  add_329 = mul_943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_95: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_944: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_95, 1.1111111111111112);  convert_element_type_95 = None
        mul_945: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_330, mul_944);  mul_944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1565: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_945, [8192, 768]);  mul_945 = None
        permute_1183: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1565, [1, 0])
        mm_475: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1183, view_220);  permute_1183 = view_220 = None
        permute_97: "f32[768, 768]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_1185: "f32[768, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_476: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1565, permute_1185);  view_1565 = permute_1185 = None
        view_1566: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_476, [8, 1024, 768]);  mm_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1567: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1566, [8, 1024, 12, 64]);  view_1566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1187: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1567, [0, 2, 1, 3]);  view_1567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_371: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1187, memory_format = torch.contiguous_format);  permute_1187 = None
        view_1568: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_371, [96, 1024, 64]);  clone_371 = None
        bmm_180: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1188, view_1568);  permute_1188 = None
        bmm_181: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1568, permute_1189);  view_1568 = permute_1189 = None
        view_1569: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_180, [8, 12, 1024, 64]);  bmm_180 = None
        view_1570: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_181, [8, 12, 1024, 1024]);  bmm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_96: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_946: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_96, 1.1111111111111112);  convert_element_type_96 = None
        mul_947: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1570, mul_946);  view_1570 = mul_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_948: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_947, div_10);  mul_947 = None
        sum_159: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_948, [-1], True)
        neg_29: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_27: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_29, sum_159, mul_948);  neg_29 = sum_159 = mul_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1571: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_27, [96, 1024, 1024]);  fma_27 = None
        view_1573: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1571, [8, 12, 1024, 1024]);  view_1571 = None
        view_1574: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1573, [96, 1024, 1024])
        add_331: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_324, view_1573);  add_324 = view_1573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_182: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1190, view_1574);  permute_1190 = None
        bmm_183: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1574, permute_1191);  view_1574 = permute_1191 = None
        view_1576: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_182, [8, 12, 64, 1024]);  bmm_182 = None
        view_1577: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_183, [8, 12, 1024, 64]);  bmm_183 = None
        permute_1192: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1576, [0, 1, 3, 2]);  view_1576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1193: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1569, [0, 2, 1, 3]);  view_1569 = None
        clone_374: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1193, memory_format = torch.contiguous_format);  permute_1193 = None
        view_1578: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_374, [8, 1024, 768]);  clone_374 = None
        view_1579: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1578, [8192, 768]);  view_1578 = None
        permute_1194: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1579, [1, 0])
        mm_477: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1194, view_201);  permute_1194 = None
        permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_1196: "f32[768, 768]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        mm_478: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1579, permute_1196);  view_1579 = permute_1196 = None
        view_1580: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_478, [8, 1024, 768]);  mm_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1198: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1192, [0, 2, 1, 3]);  permute_1192 = None
        view_1581: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1198, [8, 1024, 768]);  permute_1198 = None
        clone_375: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1581, memory_format = torch.contiguous_format);  view_1581 = None
        view_1582: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_375, [8192, 768]);  clone_375 = None
        permute_1199: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1582, [1, 0])
        mm_479: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1199, view_201);  permute_1199 = None
        permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_1201: "f32[768, 768]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        mm_480: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1582, permute_1201);  view_1582 = permute_1201 = None
        view_1583: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_480, [8, 1024, 768]);  mm_480 = None
        add_332: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1580, view_1583);  view_1580 = view_1583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1203: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1577, [0, 2, 1, 3]);  view_1577 = None
        clone_376: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1203, memory_format = torch.contiguous_format);  permute_1203 = None
        view_1584: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_376, [8, 1024, 768]);  clone_376 = None
        view_1585: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1584, [8192, 768]);  view_1584 = None
        permute_1204: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1585, [1, 0])
        mm_481: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1204, view_201);  permute_1204 = view_201 = None
        permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_1206: "f32[768, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_482: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1585, permute_1206);  view_1585 = permute_1206 = None
        view_1586: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_482, [8, 1024, 768]);  mm_482 = None
        add_333: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_332, view_1586);  add_332 = view_1586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_949: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_333, primals_68);  primals_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_100: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_46, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_950: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_333, mul_100);  add_333 = mul_100 = None
        sum_160: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_950, [0, 1], True);  mul_950 = None
        view_1587: "f32[768]" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_951: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_949, add_46)
        mul_952: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_949, rsqrt_16);  mul_949 = None
        sum_161: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_951, [2], True);  mul_951 = None
        add_334: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_330, mul_952);  add_330 = mul_952 = None
        pow_153: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
        mul_953: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_161, -0.5);  sum_161 = None
        mul_954: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_953, pow_153);  mul_953 = pow_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_192: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_954, [8, 1024, 768]);  mul_954 = None
        div_87: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_192, 768);  expand_192 = None
        pow_154: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_46, 1.0);  add_46 = None
        mul_955: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_154, 2.0);  pow_154 = None
        mul_956: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_87, mul_955);  div_87 = mul_955 = None
        add_335: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_334, mul_956);  add_334 = mul_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_97: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_957: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 1.1111111111111112);  convert_element_type_97 = None
        mul_958: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_335, mul_957);  mul_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1588: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_958, [8192, 768]);  mul_958 = None
        permute_1208: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1588, [1, 0])
        mm_483: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1208, view_199);  permute_1208 = view_199 = None
        permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_1210: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_484: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1588, permute_1210);  view_1588 = permute_1210 = None
        view_1589: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_484, [8, 1024, 3072]);  mm_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_98: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_959: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 1.1111111111111112);  convert_element_type_98 = None
        mul_960: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1589, mul_959);  view_1589 = mul_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_28: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_17, full_default, mul_960);  le_17 = mul_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1590: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_28, [8192, 3072]);  where_28 = None
        permute_1212: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1590, [1, 0])
        mm_485: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1212, view_197);  permute_1212 = view_197 = None
        permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_1214: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_486: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1590, permute_1214);  view_1590 = permute_1214 = None
        view_1591: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_486, [8, 1024, 768]);  mm_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_961: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1591, primals_65);  primals_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_94: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_962: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1591, mul_94);  view_1591 = mul_94 = None
        sum_162: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_962, [0, 1], True);  mul_962 = None
        view_1592: "f32[768]" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_963: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_961, add_44)
        mul_964: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_961, rsqrt_15);  mul_961 = None
        sum_163: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_963, [2], True);  mul_963 = None
        add_336: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_335, mul_964);  add_335 = mul_964 = None
        pow_155: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
        mul_965: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_163, -0.5);  sum_163 = None
        mul_966: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_965, pow_155);  mul_965 = pow_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_193: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_966, [8, 1024, 768]);  mul_966 = None
        div_88: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_193, 768);  expand_193 = None
        pow_156: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 1.0);  add_44 = None
        mul_967: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_156, 2.0);  pow_156 = None
        mul_968: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_88, mul_967);  div_88 = mul_967 = None
        add_337: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_336, mul_968);  add_336 = mul_968 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_99: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_969: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_99, 1.1111111111111112);  convert_element_type_99 = None
        mul_970: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_337, mul_969);  mul_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1593: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_970, [8192, 768]);  mul_970 = None
        permute_1216: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1593, [1, 0])
        mm_487: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1216, view_195);  permute_1216 = view_195 = None
        permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_1218: "f32[768, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_488: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1593, permute_1218);  view_1593 = permute_1218 = None
        view_1594: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_488, [8, 1024, 768]);  mm_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1595: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1594, [8, 1024, 12, 64]);  view_1594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1220: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1595, [0, 2, 1, 3]);  view_1595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_380: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1220, memory_format = torch.contiguous_format);  permute_1220 = None
        view_1596: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_380, [96, 1024, 64]);  clone_380 = None
        bmm_184: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1221, view_1596);  permute_1221 = None
        bmm_185: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1596, permute_1222);  view_1596 = permute_1222 = None
        view_1597: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_184, [8, 12, 1024, 64]);  bmm_184 = None
        view_1598: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_185, [8, 12, 1024, 1024]);  bmm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_100: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_971: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_100, 1.1111111111111112);  convert_element_type_100 = None
        mul_972: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1598, mul_971);  view_1598 = mul_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_973: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_972, div_9);  mul_972 = None
        sum_164: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_973, [-1], True)
        neg_30: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_28: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_30, sum_164, mul_973);  neg_30 = sum_164 = mul_973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1599: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_28, [96, 1024, 1024]);  fma_28 = None
        view_1601: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1599, [8, 12, 1024, 1024]);  view_1599 = None
        view_1602: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1601, [96, 1024, 1024])
        add_338: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_331, view_1601);  add_331 = view_1601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_186: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1223, view_1602);  permute_1223 = None
        bmm_187: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1602, permute_1224);  view_1602 = permute_1224 = None
        view_1604: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_186, [8, 12, 64, 1024]);  bmm_186 = None
        view_1605: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_187, [8, 12, 1024, 64]);  bmm_187 = None
        permute_1225: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1604, [0, 1, 3, 2]);  view_1604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1226: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1597, [0, 2, 1, 3]);  view_1597 = None
        clone_383: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1226, memory_format = torch.contiguous_format);  permute_1226 = None
        view_1606: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_383, [8, 1024, 768]);  clone_383 = None
        view_1607: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1606, [8192, 768]);  view_1606 = None
        permute_1227: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1607, [1, 0])
        mm_489: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1227, view_176);  permute_1227 = None
        permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_1229: "f32[768, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        mm_490: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1607, permute_1229);  view_1607 = permute_1229 = None
        view_1608: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_490, [8, 1024, 768]);  mm_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1231: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1225, [0, 2, 1, 3]);  permute_1225 = None
        view_1609: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1231, [8, 1024, 768]);  permute_1231 = None
        clone_384: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1609, memory_format = torch.contiguous_format);  view_1609 = None
        view_1610: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_384, [8192, 768]);  clone_384 = None
        permute_1232: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1610, [1, 0])
        mm_491: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1232, view_176);  permute_1232 = None
        permute_80: "f32[768, 768]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_1234: "f32[768, 768]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        mm_492: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1610, permute_1234);  view_1610 = permute_1234 = None
        view_1611: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_492, [8, 1024, 768]);  mm_492 = None
        add_339: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1608, view_1611);  view_1608 = view_1611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1236: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1605, [0, 2, 1, 3]);  view_1605 = None
        clone_385: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1236, memory_format = torch.contiguous_format);  permute_1236 = None
        view_1612: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_385, [8, 1024, 768]);  clone_385 = None
        view_1613: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1612, [8192, 768]);  view_1612 = None
        permute_1237: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1613, [1, 0])
        mm_493: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1237, view_176);  permute_1237 = view_176 = None
        permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_1239: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_494: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1613, permute_1239);  view_1613 = permute_1239 = None
        view_1614: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_494, [8, 1024, 768]);  mm_494 = None
        add_340: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_339, view_1614);  add_339 = view_1614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_974: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_340, primals_60);  primals_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_88: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_41, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_975: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_340, mul_88);  add_340 = mul_88 = None
        sum_165: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_975, [0, 1], True);  mul_975 = None
        view_1615: "f32[768]" = torch.ops.aten.reshape.default(sum_165, [768]);  sum_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_976: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_974, add_41)
        mul_977: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_974, rsqrt_14);  mul_974 = None
        sum_166: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_976, [2], True);  mul_976 = None
        add_341: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_337, mul_977);  add_337 = mul_977 = None
        pow_157: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
        mul_978: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_166, -0.5);  sum_166 = None
        mul_979: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_978, pow_157);  mul_978 = pow_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_194: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_979, [8, 1024, 768]);  mul_979 = None
        div_89: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_194, 768);  expand_194 = None
        pow_158: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_41, 1.0);  add_41 = None
        mul_980: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_158, 2.0);  pow_158 = None
        mul_981: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_89, mul_980);  div_89 = mul_980 = None
        add_342: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_341, mul_981);  add_341 = mul_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_101: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_982: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_101, 1.1111111111111112);  convert_element_type_101 = None
        mul_983: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_342, mul_982);  mul_982 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1616: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_983, [8192, 768]);  mul_983 = None
        permute_1241: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1616, [1, 0])
        mm_495: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1241, view_174);  permute_1241 = view_174 = None
        permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_1243: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_496: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1616, permute_1243);  view_1616 = permute_1243 = None
        view_1617: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_496, [8, 1024, 3072]);  mm_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_102: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_984: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_102, 1.1111111111111112);  convert_element_type_102 = None
        mul_985: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1617, mul_984);  view_1617 = mul_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_29: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_18, full_default, mul_985);  le_18 = mul_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1618: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_29, [8192, 3072]);  where_29 = None
        permute_1245: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1618, [1, 0])
        mm_497: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1245, view_172);  permute_1245 = view_172 = None
        permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        permute_1247: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_498: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1618, permute_1247);  view_1618 = permute_1247 = None
        view_1619: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_498, [8, 1024, 768]);  mm_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_986: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1619, primals_57);  primals_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_82: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_39, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_987: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1619, mul_82);  view_1619 = mul_82 = None
        sum_167: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_987, [0, 1], True);  mul_987 = None
        view_1620: "f32[768]" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_988: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_986, add_39)
        mul_989: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_986, rsqrt_13);  mul_986 = None
        sum_168: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_988, [2], True);  mul_988 = None
        add_343: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_342, mul_989);  add_342 = mul_989 = None
        pow_159: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
        mul_990: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_168, -0.5);  sum_168 = None
        mul_991: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_990, pow_159);  mul_990 = pow_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_195: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_991, [8, 1024, 768]);  mul_991 = None
        div_90: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_195, 768);  expand_195 = None
        pow_160: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_39, 1.0);  add_39 = None
        mul_992: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_160, 2.0);  pow_160 = None
        mul_993: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_90, mul_992);  div_90 = mul_992 = None
        add_344: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_343, mul_993);  add_343 = mul_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_103: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_994: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_103, 1.1111111111111112);  convert_element_type_103 = None
        mul_995: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_344, mul_994);  mul_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1621: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_995, [8192, 768]);  mul_995 = None
        permute_1249: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1621, [1, 0])
        mm_499: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1249, view_170);  permute_1249 = view_170 = None
        permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_1251: "f32[768, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_500: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1621, permute_1251);  view_1621 = permute_1251 = None
        view_1622: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_500, [8, 1024, 768]);  mm_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1623: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1622, [8, 1024, 12, 64]);  view_1622 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1253: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1623, [0, 2, 1, 3]);  view_1623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_389: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1253, memory_format = torch.contiguous_format);  permute_1253 = None
        view_1624: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_389, [96, 1024, 64]);  clone_389 = None
        bmm_188: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1254, view_1624);  permute_1254 = None
        bmm_189: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1624, permute_1255);  view_1624 = permute_1255 = None
        view_1625: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_188, [8, 12, 1024, 64]);  bmm_188 = None
        view_1626: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_189, [8, 12, 1024, 1024]);  bmm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_104: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_996: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_104, 1.1111111111111112);  convert_element_type_104 = None
        mul_997: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1626, mul_996);  view_1626 = mul_996 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_998: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_997, div_8);  mul_997 = None
        sum_169: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_998, [-1], True)
        neg_31: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_29: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_31, sum_169, mul_998);  neg_31 = sum_169 = mul_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1627: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_29, [96, 1024, 1024]);  fma_29 = None
        view_1629: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1627, [8, 12, 1024, 1024]);  view_1627 = None
        view_1630: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1629, [96, 1024, 1024])
        add_345: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_338, view_1629);  add_338 = view_1629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_190: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1256, view_1630);  permute_1256 = None
        bmm_191: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1630, permute_1257);  view_1630 = permute_1257 = None
        view_1632: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_190, [8, 12, 64, 1024]);  bmm_190 = None
        view_1633: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_191, [8, 12, 1024, 64]);  bmm_191 = None
        permute_1258: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1632, [0, 1, 3, 2]);  view_1632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1259: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1625, [0, 2, 1, 3]);  view_1625 = None
        clone_392: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1259, memory_format = torch.contiguous_format);  permute_1259 = None
        view_1634: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_392, [8, 1024, 768]);  clone_392 = None
        view_1635: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1634, [8192, 768]);  view_1634 = None
        permute_1260: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1635, [1, 0])
        mm_501: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1260, view_151);  permute_1260 = None
        permute_71: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_1262: "f32[768, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_502: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1635, permute_1262);  view_1635 = permute_1262 = None
        view_1636: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_502, [8, 1024, 768]);  mm_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1264: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1258, [0, 2, 1, 3]);  permute_1258 = None
        view_1637: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1264, [8, 1024, 768]);  permute_1264 = None
        clone_393: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1637, memory_format = torch.contiguous_format);  view_1637 = None
        view_1638: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_393, [8192, 768]);  clone_393 = None
        permute_1265: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1638, [1, 0])
        mm_503: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1265, view_151);  permute_1265 = None
        permute_69: "f32[768, 768]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        permute_1267: "f32[768, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        mm_504: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1638, permute_1267);  view_1638 = permute_1267 = None
        view_1639: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_504, [8, 1024, 768]);  mm_504 = None
        add_346: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1636, view_1639);  view_1636 = view_1639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1269: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1633, [0, 2, 1, 3]);  view_1633 = None
        clone_394: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1269, memory_format = torch.contiguous_format);  permute_1269 = None
        view_1640: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_394, [8, 1024, 768]);  clone_394 = None
        view_1641: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1640, [8192, 768]);  view_1640 = None
        permute_1270: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1641, [1, 0])
        mm_505: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1270, view_151);  permute_1270 = view_151 = None
        permute_67: "f32[768, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        permute_1272: "f32[768, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_506: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1641, permute_1272);  view_1641 = permute_1272 = None
        view_1642: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_506, [8, 1024, 768]);  mm_506 = None
        add_347: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_346, view_1642);  add_346 = view_1642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_999: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_347, primals_52);  primals_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_76: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_36, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1000: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_347, mul_76);  add_347 = mul_76 = None
        sum_170: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1000, [0, 1], True);  mul_1000 = None
        view_1643: "f32[768]" = torch.ops.aten.reshape.default(sum_170, [768]);  sum_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1001: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_999, add_36)
        mul_1002: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_999, rsqrt_12);  mul_999 = None
        sum_171: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1001, [2], True);  mul_1001 = None
        add_348: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_344, mul_1002);  add_344 = mul_1002 = None
        pow_161: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
        mul_1003: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_171, -0.5);  sum_171 = None
        mul_1004: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1003, pow_161);  mul_1003 = pow_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_196: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1004, [8, 1024, 768]);  mul_1004 = None
        div_91: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_196, 768);  expand_196 = None
        pow_162: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_36, 1.0);  add_36 = None
        mul_1005: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_162, 2.0);  pow_162 = None
        mul_1006: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_91, mul_1005);  div_91 = mul_1005 = None
        add_349: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_348, mul_1006);  add_348 = mul_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_105: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_1007: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_105, 1.1111111111111112);  convert_element_type_105 = None
        mul_1008: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_349, mul_1007);  mul_1007 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1644: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1008, [8192, 768]);  mul_1008 = None
        permute_1274: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1644, [1, 0])
        mm_507: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1274, view_149);  permute_1274 = view_149 = None
        permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_1276: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_508: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1644, permute_1276);  view_1644 = permute_1276 = None
        view_1645: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_508, [8, 1024, 3072]);  mm_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_106: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_1009: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_106, 1.1111111111111112);  convert_element_type_106 = None
        mul_1010: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1645, mul_1009);  view_1645 = mul_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_30: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_19, full_default, mul_1010);  le_19 = mul_1010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1646: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_30, [8192, 3072]);  where_30 = None
        permute_1278: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1646, [1, 0])
        mm_509: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1278, view_147);  permute_1278 = view_147 = None
        permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_1280: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_510: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1646, permute_1280);  view_1646 = permute_1280 = None
        view_1647: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_510, [8, 1024, 768]);  mm_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1011: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1647, primals_49);  primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_70: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1012: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1647, mul_70);  view_1647 = mul_70 = None
        sum_172: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1012, [0, 1], True);  mul_1012 = None
        view_1648: "f32[768]" = torch.ops.aten.reshape.default(sum_172, [768]);  sum_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1013: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1011, add_34)
        mul_1014: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1011, rsqrt_11);  mul_1011 = None
        sum_173: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1013, [2], True);  mul_1013 = None
        add_350: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_349, mul_1014);  add_349 = mul_1014 = None
        pow_163: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_11, 3);  rsqrt_11 = None
        mul_1015: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_173, -0.5);  sum_173 = None
        mul_1016: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1015, pow_163);  mul_1015 = pow_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_197: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1016, [8, 1024, 768]);  mul_1016 = None
        div_92: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_197, 768);  expand_197 = None
        pow_164: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 1.0);  add_34 = None
        mul_1017: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_164, 2.0);  pow_164 = None
        mul_1018: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_92, mul_1017);  div_92 = mul_1017 = None
        add_351: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_350, mul_1018);  add_350 = mul_1018 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_107: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_1019: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_107, 1.1111111111111112);  convert_element_type_107 = None
        mul_1020: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_351, mul_1019);  mul_1019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1649: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1020, [8192, 768]);  mul_1020 = None
        permute_1282: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1649, [1, 0])
        mm_511: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1282, view_145);  permute_1282 = view_145 = None
        permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_1284: "f32[768, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_512: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1649, permute_1284);  view_1649 = permute_1284 = None
        view_1650: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_512, [8, 1024, 768]);  mm_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1651: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1650, [8, 1024, 12, 64]);  view_1650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1286: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1651, [0, 2, 1, 3]);  view_1651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_398: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1286, memory_format = torch.contiguous_format);  permute_1286 = None
        view_1652: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_398, [96, 1024, 64]);  clone_398 = None
        bmm_192: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1287, view_1652);  permute_1287 = None
        bmm_193: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1652, permute_1288);  view_1652 = permute_1288 = None
        view_1653: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_192, [8, 12, 1024, 64]);  bmm_192 = None
        view_1654: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_193, [8, 12, 1024, 1024]);  bmm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_108: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_1021: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_108, 1.1111111111111112);  convert_element_type_108 = None
        mul_1022: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1654, mul_1021);  view_1654 = mul_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_1023: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1022, div_7);  mul_1022 = None
        sum_174: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1023, [-1], True)
        neg_32: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_30: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_32, sum_174, mul_1023);  neg_32 = sum_174 = mul_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1655: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_30, [96, 1024, 1024]);  fma_30 = None
        view_1657: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1655, [8, 12, 1024, 1024]);  view_1655 = None
        view_1658: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1657, [96, 1024, 1024])
        add_352: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_345, view_1657);  add_345 = view_1657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_194: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1289, view_1658);  permute_1289 = None
        bmm_195: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1658, permute_1290);  view_1658 = permute_1290 = None
        view_1660: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_194, [8, 12, 64, 1024]);  bmm_194 = None
        view_1661: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_195, [8, 12, 1024, 64]);  bmm_195 = None
        permute_1291: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1660, [0, 1, 3, 2]);  view_1660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1292: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1653, [0, 2, 1, 3]);  view_1653 = None
        clone_401: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1292, memory_format = torch.contiguous_format);  permute_1292 = None
        view_1662: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_401, [8, 1024, 768]);  clone_401 = None
        view_1663: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1662, [8192, 768]);  view_1662 = None
        permute_1293: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1663, [1, 0])
        mm_513: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1293, view_126);  permute_1293 = None
        permute_60: "f32[768, 768]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
        permute_1295: "f32[768, 768]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_514: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1663, permute_1295);  view_1663 = permute_1295 = None
        view_1664: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_514, [8, 1024, 768]);  mm_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1297: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1291, [0, 2, 1, 3]);  permute_1291 = None
        view_1665: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1297, [8, 1024, 768]);  permute_1297 = None
        clone_402: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1665, memory_format = torch.contiguous_format);  view_1665 = None
        view_1666: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_402, [8192, 768]);  clone_402 = None
        permute_1298: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1666, [1, 0])
        mm_515: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1298, view_126);  permute_1298 = None
        permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_1300: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        mm_516: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1666, permute_1300);  view_1666 = permute_1300 = None
        view_1667: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_516, [8, 1024, 768]);  mm_516 = None
        add_353: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1664, view_1667);  view_1664 = view_1667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1302: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1661, [0, 2, 1, 3]);  view_1661 = None
        clone_403: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1302, memory_format = torch.contiguous_format);  permute_1302 = None
        view_1668: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_403, [8, 1024, 768]);  clone_403 = None
        view_1669: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1668, [8192, 768]);  view_1668 = None
        permute_1303: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1669, [1, 0])
        mm_517: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1303, view_126);  permute_1303 = view_126 = None
        permute_56: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_1305: "f32[768, 768]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_518: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1669, permute_1305);  view_1669 = permute_1305 = None
        view_1670: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_518, [8, 1024, 768]);  mm_518 = None
        add_354: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_353, view_1670);  add_353 = view_1670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1024: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_354, primals_44);  primals_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_64: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1025: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_354, mul_64);  add_354 = mul_64 = None
        sum_175: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1025, [0, 1], True);  mul_1025 = None
        view_1671: "f32[768]" = torch.ops.aten.reshape.default(sum_175, [768]);  sum_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1026: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1024, add_31)
        mul_1027: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1024, rsqrt_10);  mul_1024 = None
        sum_176: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1026, [2], True);  mul_1026 = None
        add_355: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_351, mul_1027);  add_351 = mul_1027 = None
        pow_165: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_10, 3);  rsqrt_10 = None
        mul_1028: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_176, -0.5);  sum_176 = None
        mul_1029: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1028, pow_165);  mul_1028 = pow_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_198: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1029, [8, 1024, 768]);  mul_1029 = None
        div_93: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_198, 768);  expand_198 = None
        pow_166: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 1.0);  add_31 = None
        mul_1030: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_166, 2.0);  pow_166 = None
        mul_1031: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_93, mul_1030);  div_93 = mul_1030 = None
        add_356: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_355, mul_1031);  add_355 = mul_1031 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_109: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_1032: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_109, 1.1111111111111112);  convert_element_type_109 = None
        mul_1033: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_356, mul_1032);  mul_1032 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1672: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1033, [8192, 768]);  mul_1033 = None
        permute_1307: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1672, [1, 0])
        mm_519: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1307, view_124);  permute_1307 = view_124 = None
        permute_55: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_1309: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_520: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1672, permute_1309);  view_1672 = permute_1309 = None
        view_1673: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_520, [8, 1024, 3072]);  mm_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_110: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_1034: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_110, 1.1111111111111112);  convert_element_type_110 = None
        mul_1035: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1673, mul_1034);  view_1673 = mul_1034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_31: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_20, full_default, mul_1035);  le_20 = mul_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1674: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_31, [8192, 3072]);  where_31 = None
        permute_1311: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1674, [1, 0])
        mm_521: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1311, view_122);  permute_1311 = view_122 = None
        permute_54: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_1313: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_522: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1674, permute_1313);  view_1674 = permute_1313 = None
        view_1675: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_522, [8, 1024, 768]);  mm_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1036: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1675, primals_41);  primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_58: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_29, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1037: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1675, mul_58);  view_1675 = mul_58 = None
        sum_177: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1037, [0, 1], True);  mul_1037 = None
        view_1676: "f32[768]" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1038: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1036, add_29)
        mul_1039: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1036, rsqrt_9);  mul_1036 = None
        sum_178: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1038, [2], True);  mul_1038 = None
        add_357: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_356, mul_1039);  add_356 = mul_1039 = None
        pow_167: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_9, 3);  rsqrt_9 = None
        mul_1040: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_178, -0.5);  sum_178 = None
        mul_1041: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1040, pow_167);  mul_1040 = pow_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_199: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1041, [8, 1024, 768]);  mul_1041 = None
        div_94: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_199, 768);  expand_199 = None
        pow_168: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_29, 1.0);  add_29 = None
        mul_1042: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_168, 2.0);  pow_168 = None
        mul_1043: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_94, mul_1042);  div_94 = mul_1042 = None
        add_358: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_357, mul_1043);  add_357 = mul_1043 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_111: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_1044: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_111, 1.1111111111111112);  convert_element_type_111 = None
        mul_1045: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_358, mul_1044);  mul_1044 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1677: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1045, [8192, 768]);  mul_1045 = None
        permute_1315: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1677, [1, 0])
        mm_523: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1315, view_120);  permute_1315 = view_120 = None
        permute_53: "f32[768, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_1317: "f32[768, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_524: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1677, permute_1317);  view_1677 = permute_1317 = None
        view_1678: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_524, [8, 1024, 768]);  mm_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1679: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1678, [8, 1024, 12, 64]);  view_1678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1319: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1679, [0, 2, 1, 3]);  view_1679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_407: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1319, memory_format = torch.contiguous_format);  permute_1319 = None
        view_1680: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_407, [96, 1024, 64]);  clone_407 = None
        bmm_196: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1320, view_1680);  permute_1320 = None
        bmm_197: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1680, permute_1321);  view_1680 = permute_1321 = None
        view_1681: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_196, [8, 12, 1024, 64]);  bmm_196 = None
        view_1682: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_197, [8, 12, 1024, 1024]);  bmm_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_112: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_1046: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_112, 1.1111111111111112);  convert_element_type_112 = None
        mul_1047: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1682, mul_1046);  view_1682 = mul_1046 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_1048: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1047, div_6);  mul_1047 = None
        sum_179: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1048, [-1], True)
        neg_33: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_31: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_33, sum_179, mul_1048);  neg_33 = sum_179 = mul_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1683: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_31, [96, 1024, 1024]);  fma_31 = None
        view_1685: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1683, [8, 12, 1024, 1024]);  view_1683 = None
        view_1686: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1685, [96, 1024, 1024])
        add_359: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_352, view_1685);  add_352 = view_1685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_198: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1322, view_1686);  permute_1322 = None
        bmm_199: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1686, permute_1323);  view_1686 = permute_1323 = None
        view_1688: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_198, [8, 12, 64, 1024]);  bmm_198 = None
        view_1689: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_199, [8, 12, 1024, 64]);  bmm_199 = None
        permute_1324: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1688, [0, 1, 3, 2]);  view_1688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1325: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1681, [0, 2, 1, 3]);  view_1681 = None
        clone_410: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1325, memory_format = torch.contiguous_format);  permute_1325 = None
        view_1690: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_410, [8, 1024, 768]);  clone_410 = None
        view_1691: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1690, [8192, 768]);  view_1690 = None
        permute_1326: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1691, [1, 0])
        mm_525: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1326, view_101);  permute_1326 = None
        permute_49: "f32[768, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_1328: "f32[768, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        mm_526: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1691, permute_1328);  view_1691 = permute_1328 = None
        view_1692: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_526, [8, 1024, 768]);  mm_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1330: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1324, [0, 2, 1, 3]);  permute_1324 = None
        view_1693: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1330, [8, 1024, 768]);  permute_1330 = None
        clone_411: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1693, memory_format = torch.contiguous_format);  view_1693 = None
        view_1694: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_411, [8192, 768]);  clone_411 = None
        permute_1331: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1694, [1, 0])
        mm_527: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1331, view_101);  permute_1331 = None
        permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_1333: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_528: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1694, permute_1333);  view_1694 = permute_1333 = None
        view_1695: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_528, [8, 1024, 768]);  mm_528 = None
        add_360: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1692, view_1695);  view_1692 = view_1695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1335: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1689, [0, 2, 1, 3]);  view_1689 = None
        clone_412: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1335, memory_format = torch.contiguous_format);  permute_1335 = None
        view_1696: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_412, [8, 1024, 768]);  clone_412 = None
        view_1697: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1696, [8192, 768]);  view_1696 = None
        permute_1336: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1697, [1, 0])
        mm_529: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1336, view_101);  permute_1336 = view_101 = None
        permute_45: "f32[768, 768]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_1338: "f32[768, 768]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_530: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1697, permute_1338);  view_1697 = permute_1338 = None
        view_1698: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_530, [8, 1024, 768]);  mm_530 = None
        add_361: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_360, view_1698);  add_360 = view_1698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1049: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_361, primals_36);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_52: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1050: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_361, mul_52);  add_361 = mul_52 = None
        sum_180: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1050, [0, 1], True);  mul_1050 = None
        view_1699: "f32[768]" = torch.ops.aten.reshape.default(sum_180, [768]);  sum_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1051: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1049, add_26)
        mul_1052: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1049, rsqrt_8);  mul_1049 = None
        sum_181: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1051, [2], True);  mul_1051 = None
        add_362: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_358, mul_1052);  add_358 = mul_1052 = None
        pow_169: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_8, 3);  rsqrt_8 = None
        mul_1053: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_181, -0.5);  sum_181 = None
        mul_1054: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1053, pow_169);  mul_1053 = pow_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_200: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1054, [8, 1024, 768]);  mul_1054 = None
        div_95: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_200, 768);  expand_200 = None
        pow_170: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 1.0);  add_26 = None
        mul_1055: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_170, 2.0);  pow_170 = None
        mul_1056: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_95, mul_1055);  div_95 = mul_1055 = None
        add_363: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_362, mul_1056);  add_362 = mul_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_113: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_1057: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_113, 1.1111111111111112);  convert_element_type_113 = None
        mul_1058: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_363, mul_1057);  mul_1057 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1700: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1058, [8192, 768]);  mul_1058 = None
        permute_1340: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1700, [1, 0])
        mm_531: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1340, view_99);  permute_1340 = view_99 = None
        permute_44: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_1342: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_532: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1700, permute_1342);  view_1700 = permute_1342 = None
        view_1701: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_532, [8, 1024, 3072]);  mm_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_114: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_1059: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_114, 1.1111111111111112);  convert_element_type_114 = None
        mul_1060: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1701, mul_1059);  view_1701 = mul_1059 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_32: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_21, full_default, mul_1060);  le_21 = mul_1060 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1702: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_32, [8192, 3072]);  where_32 = None
        permute_1344: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1702, [1, 0])
        mm_533: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1344, view_97);  permute_1344 = view_97 = None
        permute_43: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_1346: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_534: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1702, permute_1346);  view_1702 = permute_1346 = None
        view_1703: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_534, [8, 1024, 768]);  mm_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1061: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1703, primals_33);  primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_46: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1062: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1703, mul_46);  view_1703 = mul_46 = None
        sum_182: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1062, [0, 1], True);  mul_1062 = None
        view_1704: "f32[768]" = torch.ops.aten.reshape.default(sum_182, [768]);  sum_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1063: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1061, add_24)
        mul_1064: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1061, rsqrt_7);  mul_1061 = None
        sum_183: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1063, [2], True);  mul_1063 = None
        add_364: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_363, mul_1064);  add_363 = mul_1064 = None
        pow_171: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_7, 3);  rsqrt_7 = None
        mul_1065: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_183, -0.5);  sum_183 = None
        mul_1066: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1065, pow_171);  mul_1065 = pow_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_201: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1066, [8, 1024, 768]);  mul_1066 = None
        div_96: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_201, 768);  expand_201 = None
        pow_172: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 1.0);  add_24 = None
        mul_1067: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_172, 2.0);  pow_172 = None
        mul_1068: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_96, mul_1067);  div_96 = mul_1067 = None
        add_365: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_364, mul_1068);  add_364 = mul_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_115: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_1069: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_115, 1.1111111111111112);  convert_element_type_115 = None
        mul_1070: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_365, mul_1069);  mul_1069 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1705: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1070, [8192, 768]);  mul_1070 = None
        permute_1348: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1705, [1, 0])
        mm_535: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1348, view_95);  permute_1348 = view_95 = None
        permute_42: "f32[768, 768]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_1350: "f32[768, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_536: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1705, permute_1350);  view_1705 = permute_1350 = None
        view_1706: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_536, [8, 1024, 768]);  mm_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1707: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1706, [8, 1024, 12, 64]);  view_1706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1352: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1707, [0, 2, 1, 3]);  view_1707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_416: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1352, memory_format = torch.contiguous_format);  permute_1352 = None
        view_1708: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_416, [96, 1024, 64]);  clone_416 = None
        bmm_200: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1353, view_1708);  permute_1353 = None
        bmm_201: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1708, permute_1354);  view_1708 = permute_1354 = None
        view_1709: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_200, [8, 12, 1024, 64]);  bmm_200 = None
        view_1710: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_201, [8, 12, 1024, 1024]);  bmm_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_116: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_1071: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_116, 1.1111111111111112);  convert_element_type_116 = None
        mul_1072: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1710, mul_1071);  view_1710 = mul_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_1073: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1072, div_5);  mul_1072 = None
        sum_184: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1073, [-1], True)
        neg_34: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_32: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_34, sum_184, mul_1073);  neg_34 = sum_184 = mul_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1711: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_32, [96, 1024, 1024]);  fma_32 = None
        view_1713: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1711, [8, 12, 1024, 1024]);  view_1711 = None
        view_1714: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1713, [96, 1024, 1024])
        add_366: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_359, view_1713);  add_359 = view_1713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_202: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1355, view_1714);  permute_1355 = None
        bmm_203: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1714, permute_1356);  view_1714 = permute_1356 = None
        view_1716: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_202, [8, 12, 64, 1024]);  bmm_202 = None
        view_1717: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_203, [8, 12, 1024, 64]);  bmm_203 = None
        permute_1357: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1716, [0, 1, 3, 2]);  view_1716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1358: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1709, [0, 2, 1, 3]);  view_1709 = None
        clone_419: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1358, memory_format = torch.contiguous_format);  permute_1358 = None
        view_1718: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_419, [8, 1024, 768]);  clone_419 = None
        view_1719: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1718, [8192, 768]);  view_1718 = None
        permute_1359: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1719, [1, 0])
        mm_537: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1359, view_76);  permute_1359 = None
        permute_38: "f32[768, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_1361: "f32[768, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm_538: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1719, permute_1361);  view_1719 = permute_1361 = None
        view_1720: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_538, [8, 1024, 768]);  mm_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1363: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1357, [0, 2, 1, 3]);  permute_1357 = None
        view_1721: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1363, [8, 1024, 768]);  permute_1363 = None
        clone_420: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1721, memory_format = torch.contiguous_format);  view_1721 = None
        view_1722: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_420, [8192, 768]);  clone_420 = None
        permute_1364: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1722, [1, 0])
        mm_539: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1364, view_76);  permute_1364 = None
        permute_36: "f32[768, 768]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_1366: "f32[768, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_540: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1722, permute_1366);  view_1722 = permute_1366 = None
        view_1723: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_540, [8, 1024, 768]);  mm_540 = None
        add_367: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1720, view_1723);  view_1720 = view_1723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1368: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1717, [0, 2, 1, 3]);  view_1717 = None
        clone_421: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1368, memory_format = torch.contiguous_format);  permute_1368 = None
        view_1724: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_421, [8, 1024, 768]);  clone_421 = None
        view_1725: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1724, [8192, 768]);  view_1724 = None
        permute_1369: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1725, [1, 0])
        mm_541: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1369, view_76);  permute_1369 = view_76 = None
        permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_1371: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_542: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1725, permute_1371);  view_1725 = permute_1371 = None
        view_1726: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_542, [8, 1024, 768]);  mm_542 = None
        add_368: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_367, view_1726);  add_367 = view_1726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1074: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_368, primals_28);  primals_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_40: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1075: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_368, mul_40);  add_368 = mul_40 = None
        sum_185: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1075, [0, 1], True);  mul_1075 = None
        view_1727: "f32[768]" = torch.ops.aten.reshape.default(sum_185, [768]);  sum_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1076: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1074, add_21)
        mul_1077: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1074, rsqrt_6);  mul_1074 = None
        sum_186: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1076, [2], True);  mul_1076 = None
        add_369: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_365, mul_1077);  add_365 = mul_1077 = None
        pow_173: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_6, 3);  rsqrt_6 = None
        mul_1078: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_186, -0.5);  sum_186 = None
        mul_1079: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1078, pow_173);  mul_1078 = pow_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_202: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1079, [8, 1024, 768]);  mul_1079 = None
        div_97: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_202, 768);  expand_202 = None
        pow_174: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 1.0);  add_21 = None
        mul_1080: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_174, 2.0);  pow_174 = None
        mul_1081: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_97, mul_1080);  div_97 = mul_1080 = None
        add_370: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_369, mul_1081);  add_369 = mul_1081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_117: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_1082: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_117, 1.1111111111111112);  convert_element_type_117 = None
        mul_1083: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_370, mul_1082);  mul_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1728: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1083, [8192, 768]);  mul_1083 = None
        permute_1373: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1728, [1, 0])
        mm_543: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1373, view_74);  permute_1373 = view_74 = None
        permute_33: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_1375: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_544: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1728, permute_1375);  view_1728 = permute_1375 = None
        view_1729: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_544, [8, 1024, 3072]);  mm_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_118: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_1084: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_118, 1.1111111111111112);  convert_element_type_118 = None
        mul_1085: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1729, mul_1084);  view_1729 = mul_1084 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_33: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_22, full_default, mul_1085);  le_22 = mul_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1730: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_33, [8192, 3072]);  where_33 = None
        permute_1377: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1730, [1, 0])
        mm_545: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1377, view_72);  permute_1377 = view_72 = None
        permute_32: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_1379: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_546: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1730, permute_1379);  view_1730 = permute_1379 = None
        view_1731: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_546, [8, 1024, 768]);  mm_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1086: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1731, primals_25);  primals_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_34: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1087: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1731, mul_34);  view_1731 = mul_34 = None
        sum_187: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1087, [0, 1], True);  mul_1087 = None
        view_1732: "f32[768]" = torch.ops.aten.reshape.default(sum_187, [768]);  sum_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1088: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1086, add_19)
        mul_1089: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1086, rsqrt_5);  mul_1086 = None
        sum_188: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1088, [2], True);  mul_1088 = None
        add_371: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_370, mul_1089);  add_370 = mul_1089 = None
        pow_175: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_5, 3);  rsqrt_5 = None
        mul_1090: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_188, -0.5);  sum_188 = None
        mul_1091: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1090, pow_175);  mul_1090 = pow_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_203: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1091, [8, 1024, 768]);  mul_1091 = None
        div_98: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_203, 768);  expand_203 = None
        pow_176: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_19, 1.0);  add_19 = None
        mul_1092: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_176, 2.0);  pow_176 = None
        mul_1093: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_98, mul_1092);  div_98 = mul_1092 = None
        add_372: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_371, mul_1093);  add_371 = mul_1093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_119: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_1094: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_119, 1.1111111111111112);  convert_element_type_119 = None
        mul_1095: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_372, mul_1094);  mul_1094 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1733: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1095, [8192, 768]);  mul_1095 = None
        permute_1381: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1733, [1, 0])
        mm_547: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1381, view_70);  permute_1381 = view_70 = None
        permute_31: "f32[768, 768]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_1383: "f32[768, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_548: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1733, permute_1383);  view_1733 = permute_1383 = None
        view_1734: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_548, [8, 1024, 768]);  mm_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1735: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1734, [8, 1024, 12, 64]);  view_1734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1385: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1735, [0, 2, 1, 3]);  view_1735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_425: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1385, memory_format = torch.contiguous_format);  permute_1385 = None
        view_1736: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_425, [96, 1024, 64]);  clone_425 = None
        bmm_204: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1386, view_1736);  permute_1386 = None
        bmm_205: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1736, permute_1387);  view_1736 = permute_1387 = None
        view_1737: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_204, [8, 12, 1024, 64]);  bmm_204 = None
        view_1738: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_205, [8, 12, 1024, 1024]);  bmm_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_120: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_1096: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_120, 1.1111111111111112);  convert_element_type_120 = None
        mul_1097: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1738, mul_1096);  view_1738 = mul_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_1098: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1097, div_4);  mul_1097 = None
        sum_189: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1098, [-1], True)
        neg_35: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_33: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_35, sum_189, mul_1098);  neg_35 = sum_189 = mul_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1739: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_33, [96, 1024, 1024]);  fma_33 = None
        view_1741: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1739, [8, 12, 1024, 1024]);  view_1739 = None
        view_1742: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1741, [96, 1024, 1024])
        add_373: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_366, view_1741);  add_366 = view_1741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_206: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1388, view_1742);  permute_1388 = None
        bmm_207: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1742, permute_1389);  view_1742 = permute_1389 = None
        view_1744: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_206, [8, 12, 64, 1024]);  bmm_206 = None
        view_1745: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_207, [8, 12, 1024, 64]);  bmm_207 = None
        permute_1390: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1744, [0, 1, 3, 2]);  view_1744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1391: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1737, [0, 2, 1, 3]);  view_1737 = None
        clone_428: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1391, memory_format = torch.contiguous_format);  permute_1391 = None
        view_1746: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_428, [8, 1024, 768]);  clone_428 = None
        view_1747: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1746, [8192, 768]);  view_1746 = None
        permute_1392: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1747, [1, 0])
        mm_549: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1392, view_51);  permute_1392 = None
        permute_27: "f32[768, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_1394: "f32[768, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        mm_550: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1747, permute_1394);  view_1747 = permute_1394 = None
        view_1748: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_550, [8, 1024, 768]);  mm_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1396: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1390, [0, 2, 1, 3]);  permute_1390 = None
        view_1749: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1396, [8, 1024, 768]);  permute_1396 = None
        clone_429: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1749, memory_format = torch.contiguous_format);  view_1749 = None
        view_1750: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_429, [8192, 768]);  clone_429 = None
        permute_1397: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1750, [1, 0])
        mm_551: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1397, view_51);  permute_1397 = None
        permute_25: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_1399: "f32[768, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        mm_552: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1750, permute_1399);  view_1750 = permute_1399 = None
        view_1751: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_552, [8, 1024, 768]);  mm_552 = None
        add_374: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1748, view_1751);  view_1748 = view_1751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1401: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1745, [0, 2, 1, 3]);  view_1745 = None
        clone_430: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1401, memory_format = torch.contiguous_format);  permute_1401 = None
        view_1752: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_430, [8, 1024, 768]);  clone_430 = None
        view_1753: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1752, [8192, 768]);  view_1752 = None
        permute_1402: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1753, [1, 0])
        mm_553: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1402, view_51);  permute_1402 = view_51 = None
        permute_23: "f32[768, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_1404: "f32[768, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_554: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1753, permute_1404);  view_1753 = permute_1404 = None
        view_1754: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_554, [8, 1024, 768]);  mm_554 = None
        add_375: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_374, view_1754);  add_374 = view_1754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1099: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_375, primals_20);  primals_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_28: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1100: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_375, mul_28);  add_375 = mul_28 = None
        sum_190: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1100, [0, 1], True);  mul_1100 = None
        view_1755: "f32[768]" = torch.ops.aten.reshape.default(sum_190, [768]);  sum_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1101: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1099, add_16)
        mul_1102: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1099, rsqrt_4);  mul_1099 = None
        sum_191: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1101, [2], True);  mul_1101 = None
        add_376: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_372, mul_1102);  add_372 = mul_1102 = None
        pow_177: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
        mul_1103: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_191, -0.5);  sum_191 = None
        mul_1104: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1103, pow_177);  mul_1103 = pow_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_204: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1104, [8, 1024, 768]);  mul_1104 = None
        div_99: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_204, 768);  expand_204 = None
        pow_178: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 1.0);  add_16 = None
        mul_1105: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_178, 2.0);  pow_178 = None
        mul_1106: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_99, mul_1105);  div_99 = mul_1105 = None
        add_377: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_376, mul_1106);  add_376 = mul_1106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_121: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_1107: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_121, 1.1111111111111112);  convert_element_type_121 = None
        mul_1108: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_377, mul_1107);  mul_1107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1756: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1108, [8192, 768]);  mul_1108 = None
        permute_1406: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1756, [1, 0])
        mm_555: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1406, view_49);  permute_1406 = view_49 = None
        permute_22: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_1408: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_556: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1756, permute_1408);  view_1756 = permute_1408 = None
        view_1757: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_556, [8, 1024, 3072]);  mm_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_122: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_1109: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_122, 1.1111111111111112);  convert_element_type_122 = None
        mul_1110: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1757, mul_1109);  view_1757 = mul_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_34: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_23, full_default, mul_1110);  le_23 = mul_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1758: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_34, [8192, 3072]);  where_34 = None
        permute_1410: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1758, [1, 0])
        mm_557: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1410, view_47);  permute_1410 = view_47 = None
        permute_21: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_1412: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_558: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1758, permute_1412);  view_1758 = permute_1412 = None
        view_1759: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_558, [8, 1024, 768]);  mm_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1111: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1759, primals_17);  primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_22: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1112: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1759, mul_22);  view_1759 = mul_22 = None
        sum_192: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1112, [0, 1], True);  mul_1112 = None
        view_1760: "f32[768]" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1113: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1111, add_14)
        mul_1114: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1111, rsqrt_3);  mul_1111 = None
        sum_193: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1113, [2], True);  mul_1113 = None
        add_378: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_377, mul_1114);  add_377 = mul_1114 = None
        pow_179: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_1115: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_193, -0.5);  sum_193 = None
        mul_1116: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1115, pow_179);  mul_1115 = pow_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_205: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1116, [8, 1024, 768]);  mul_1116 = None
        div_100: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_205, 768);  expand_205 = None
        pow_180: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_14, 1.0);  add_14 = None
        mul_1117: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_180, 2.0);  pow_180 = None
        mul_1118: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_100, mul_1117);  div_100 = mul_1117 = None
        add_379: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_378, mul_1118);  add_378 = mul_1118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_123: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_1119: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_123, 1.1111111111111112);  convert_element_type_123 = None
        mul_1120: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_379, mul_1119);  mul_1119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1761: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1120, [8192, 768]);  mul_1120 = None
        permute_1414: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1761, [1, 0])
        mm_559: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1414, view_45);  permute_1414 = view_45 = None
        permute_20: "f32[768, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_1416: "f32[768, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_560: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1761, permute_1416);  view_1761 = permute_1416 = None
        view_1762: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_560, [8, 1024, 768]);  mm_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1763: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1762, [8, 1024, 12, 64]);  view_1762 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1418: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1763, [0, 2, 1, 3]);  view_1763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_434: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1418, memory_format = torch.contiguous_format);  permute_1418 = None
        view_1764: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_434, [96, 1024, 64]);  clone_434 = None
        bmm_208: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1419, view_1764);  permute_1419 = None
        bmm_209: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1764, permute_1420);  view_1764 = permute_1420 = None
        view_1765: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_208, [8, 12, 1024, 64]);  bmm_208 = None
        view_1766: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_209, [8, 12, 1024, 1024]);  bmm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_124: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_1121: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 1.1111111111111112);  convert_element_type_124 = None
        mul_1122: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1766, mul_1121);  view_1766 = mul_1121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_1123: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1122, div_3);  mul_1122 = None
        sum_194: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1123, [-1], True)
        neg_36: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_34: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_36, sum_194, mul_1123);  neg_36 = sum_194 = mul_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1767: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_34, [96, 1024, 1024]);  fma_34 = None
        view_1769: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1767, [8, 12, 1024, 1024]);  view_1767 = None
        view_1770: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1769, [96, 1024, 1024])
        add_380: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_373, view_1769);  add_373 = view_1769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_210: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1421, view_1770);  permute_1421 = None
        bmm_211: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1770, permute_1422);  view_1770 = permute_1422 = None
        view_1772: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_210, [8, 12, 64, 1024]);  bmm_210 = None
        view_1773: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_211, [8, 12, 1024, 64]);  bmm_211 = None
        permute_1423: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1772, [0, 1, 3, 2]);  view_1772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1424: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1765, [0, 2, 1, 3]);  view_1765 = None
        clone_437: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1424, memory_format = torch.contiguous_format);  permute_1424 = None
        view_1774: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_437, [8, 1024, 768]);  clone_437 = None
        view_1775: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1774, [8192, 768]);  view_1774 = None
        permute_1425: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1775, [1, 0])
        mm_561: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1425, view_26);  permute_1425 = None
        permute_16: "f32[768, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_1427: "f32[768, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_562: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1775, permute_1427);  view_1775 = permute_1427 = None
        view_1776: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_562, [8, 1024, 768]);  mm_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1429: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1423, [0, 2, 1, 3]);  permute_1423 = None
        view_1777: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1429, [8, 1024, 768]);  permute_1429 = None
        clone_438: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1777, memory_format = torch.contiguous_format);  view_1777 = None
        view_1778: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_438, [8192, 768]);  clone_438 = None
        permute_1430: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1778, [1, 0])
        mm_563: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1430, view_26);  permute_1430 = None
        permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_1432: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_564: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1778, permute_1432);  view_1778 = permute_1432 = None
        view_1779: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_564, [8, 1024, 768]);  mm_564 = None
        add_381: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1776, view_1779);  view_1776 = view_1779 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1434: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1773, [0, 2, 1, 3]);  view_1773 = None
        clone_439: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1434, memory_format = torch.contiguous_format);  permute_1434 = None
        view_1780: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_439, [8, 1024, 768]);  clone_439 = None
        view_1781: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1780, [8192, 768]);  view_1780 = None
        permute_1435: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1781, [1, 0])
        mm_565: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1435, view_26);  permute_1435 = view_26 = None
        permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_1437: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_566: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1781, permute_1437);  view_1781 = permute_1437 = None
        view_1782: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_566, [8, 1024, 768]);  mm_566 = None
        add_382: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_381, view_1782);  add_381 = view_1782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1124: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_382, primals_12);  primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_16: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1125: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_382, mul_16);  add_382 = mul_16 = None
        sum_195: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1125, [0, 1], True);  mul_1125 = None
        view_1783: "f32[768]" = torch.ops.aten.reshape.default(sum_195, [768]);  sum_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1126: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1124, add_11)
        mul_1127: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1124, rsqrt_2);  mul_1124 = None
        sum_196: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1126, [2], True);  mul_1126 = None
        add_383: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_379, mul_1127);  add_379 = mul_1127 = None
        pow_181: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_1128: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_196, -0.5);  sum_196 = None
        mul_1129: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1128, pow_181);  mul_1128 = pow_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_206: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1129, [8, 1024, 768]);  mul_1129 = None
        div_101: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_206, 768);  expand_206 = None
        pow_182: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 1.0);  add_11 = None
        mul_1130: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_182, 2.0);  pow_182 = None
        mul_1131: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_101, mul_1130);  div_101 = mul_1130 = None
        add_384: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_383, mul_1131);  add_383 = mul_1131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_125: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_1132: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_125, 1.1111111111111112);  convert_element_type_125 = None
        mul_1133: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_384, mul_1132);  mul_1132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_1784: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1133, [8192, 768]);  mul_1133 = None
        permute_1439: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1784, [1, 0])
        mm_567: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1439, view_24);  permute_1439 = view_24 = None
        permute_11: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_1441: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_568: "f32[8192, 3072]" = torch.ops.aten.mm.default(view_1784, permute_1441);  view_1784 = permute_1441 = None
        view_1785: "f32[8, 1024, 3072]" = torch.ops.aten.reshape.default(mm_568, [8, 1024, 3072]);  mm_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_126: "f32[8, 1024, 3072]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_1134: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 1.1111111111111112);  convert_element_type_126 = None
        mul_1135: "f32[8, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1785, mul_1134);  view_1785 = mul_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_35: "f32[8, 1024, 3072]" = torch.ops.aten.where.self(le_24, full_default, mul_1135);  le_24 = mul_1135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_1786: "f32[8192, 3072]" = torch.ops.aten.reshape.default(where_35, [8192, 3072]);  where_35 = None
        permute_1443: "f32[3072, 8192]" = torch.ops.aten.permute.default(view_1786, [1, 0])
        mm_569: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1443, view_22);  permute_1443 = view_22 = None
        permute_10: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_1445: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_570: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1786, permute_1445);  view_1786 = permute_1445 = None
        view_1787: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_570, [8, 1024, 768]);  mm_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1136: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1787, primals_9);  primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_10: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1137: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(view_1787, mul_10);  view_1787 = mul_10 = None
        sum_197: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1137, [0, 1], True);  mul_1137 = None
        view_1788: "f32[768]" = torch.ops.aten.reshape.default(sum_197, [768]);  sum_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1138: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1136, add_9)
        mul_1139: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1136, rsqrt_1);  mul_1136 = None
        sum_198: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1138, [2], True);  mul_1138 = None
        add_385: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_384, mul_1139);  add_384 = mul_1139 = None
        pow_183: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_1140: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_198, -0.5);  sum_198 = None
        mul_1141: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1140, pow_183);  mul_1140 = pow_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_207: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1141, [8, 1024, 768]);  mul_1141 = None
        div_102: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_207, 768);  expand_207 = None
        pow_184: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 1.0);  add_9 = None
        mul_1142: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_184, 2.0);  pow_184 = None
        mul_1143: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_102, mul_1142);  div_102 = mul_1142 = None
        add_386: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_385, mul_1143);  add_385 = mul_1143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_127: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_1144: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_127, 1.1111111111111112);  convert_element_type_127 = None
        mul_1145: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_386, mul_1144);  mul_1144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_1789: "f32[8192, 768]" = torch.ops.aten.reshape.default(mul_1145, [8192, 768]);  mul_1145 = None
        permute_1447: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1789, [1, 0])
        mm_571: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1447, view_20);  permute_1447 = view_20 = None
        permute_9: "f32[768, 768]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_1449: "f32[768, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_572: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1789, permute_1449);  view_1789 = permute_1449 = None
        view_1790: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_572, [8, 1024, 768]);  mm_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1791: "f32[8, 1024, 12, 64]" = torch.ops.aten.reshape.default(view_1790, [8, 1024, 12, 64]);  view_1790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1451: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1791, [0, 2, 1, 3]);  view_1791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_443: "f32[8, 12, 1024, 64]" = torch.ops.aten.clone.default(permute_1451, memory_format = torch.contiguous_format);  permute_1451 = None
        view_1792: "f32[96, 1024, 64]" = torch.ops.aten.reshape.default(clone_443, [96, 1024, 64]);  clone_443 = None
        bmm_212: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(permute_1452, view_1792);  permute_1452 = None
        bmm_213: "f32[96, 1024, 1024]" = torch.ops.aten.bmm.default(view_1792, permute_1453);  view_1792 = permute_1453 = None
        view_1793: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_212, [8, 12, 1024, 64]);  bmm_212 = None
        view_1794: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_213, [8, 12, 1024, 1024]);  bmm_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_128: "f32[8, 12, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_1146: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_128, 1.1111111111111112);  convert_element_type_128 = None
        mul_1147: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_1794, mul_1146);  view_1794 = mul_1146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(ge, [8, -1, 1024, 1024]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_12: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(bmm, [8, 12, 1024, 1024]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f32[12, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_5: "f32[1, 12, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_5, where);  unsqueeze_5 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_8: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        sub_1: "f32[8, 12, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_8, amax);  add_8 = amax = None
        exp: "f32[8, 12, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div_2: "f32[8, 12, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_1148: "f32[8, 12, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_1147, div_2);  mul_1147 = None
        sum_199: "f32[8, 12, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1148, [-1], True)
        neg_37: "f32[8, 12, 1024, 1024]" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_35: "f32[8, 12, 1024, 1024]" = torch.ops.prims.fma.default(neg_37, sum_199, mul_1148);  neg_37 = sum_199 = mul_1148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_1795: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(fma_35, [96, 1024, 1024]);  fma_35 = None
        view_1797: "f32[8, 12, 1024, 1024]" = torch.ops.aten.reshape.default(view_1795, [8, 12, 1024, 1024]);  view_1795 = None
        view_1798: "f32[96, 1024, 1024]" = torch.ops.aten.reshape.default(view_1797, [96, 1024, 1024])
        add_387: "f32[8, 12, 1024, 1024]" = torch.ops.aten.add.Tensor(add_380, view_1797);  add_380 = view_1797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_200: "f32[1, 12, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_387, [0], True);  add_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_2: "f32[12, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_200, 0);  sum_200 = None
        permute_1454: "f32[1024, 1024, 12]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_3: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_6, -1)
        unsqueeze_22: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
        where_36: "f32[1024, 1024, 12]" = torch.ops.aten.where.self(unsqueeze_22, full_default, permute_1454);  unsqueeze_22 = permute_1454 = None
        clone_446: "f32[1024, 1024, 12]" = torch.ops.aten.clone.default(where_36, memory_format = torch.contiguous_format);  where_36 = None
        index_put_2: "f32[32, 12]" = torch.ops.aten.index_put.default(full_default_30, [add_6], clone_446, True);  full_default_30 = add_6 = clone_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_214: "f32[96, 64, 1024]" = torch.ops.aten.bmm.default(permute_1455, view_1798);  permute_1455 = None
        bmm_215: "f32[96, 1024, 64]" = torch.ops.aten.bmm.default(view_1798, permute_1456);  view_1798 = permute_1456 = None
        view_1800: "f32[8, 12, 64, 1024]" = torch.ops.aten.reshape.default(bmm_214, [8, 12, 64, 1024]);  bmm_214 = None
        view_1801: "f32[8, 12, 1024, 64]" = torch.ops.aten.reshape.default(bmm_215, [8, 12, 1024, 64]);  bmm_215 = None
        permute_1457: "f32[8, 12, 1024, 64]" = torch.ops.aten.permute.default(view_1800, [0, 1, 3, 2]);  view_1800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1458: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1793, [0, 2, 1, 3]);  view_1793 = None
        clone_447: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1458, memory_format = torch.contiguous_format);  permute_1458 = None
        view_1802: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_447, [8, 1024, 768]);  clone_447 = None
        view_1803: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1802, [8192, 768]);  view_1802 = None
        permute_1459: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1803, [1, 0])
        mm_573: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1459, view_1);  permute_1459 = None
        permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_1461: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_574: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1803, permute_1461);  view_1803 = permute_1461 = None
        view_1804: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_574, [8, 1024, 768]);  mm_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1463: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(permute_1457, [0, 2, 1, 3]);  permute_1457 = None
        view_1805: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(permute_1463, [8, 1024, 768]);  permute_1463 = None
        clone_448: "f32[8, 1024, 768]" = torch.ops.aten.clone.default(view_1805, memory_format = torch.contiguous_format);  view_1805 = None
        view_1806: "f32[8192, 768]" = torch.ops.aten.reshape.default(clone_448, [8192, 768]);  clone_448 = None
        permute_1464: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1806, [1, 0])
        mm_575: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1464, view_1);  permute_1464 = None
        permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_1466: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_576: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1806, permute_1466);  view_1806 = permute_1466 = None
        view_1807: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_576, [8, 1024, 768]);  mm_576 = None
        add_388: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(view_1804, view_1807);  view_1804 = view_1807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1468: "f32[8, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1801, [0, 2, 1, 3]);  view_1801 = None
        clone_449: "f32[8, 1024, 12, 64]" = torch.ops.aten.clone.default(permute_1468, memory_format = torch.contiguous_format);  permute_1468 = None
        view_1808: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(clone_449, [8, 1024, 768]);  clone_449 = None
        view_1809: "f32[8192, 768]" = torch.ops.aten.reshape.default(view_1808, [8192, 768]);  view_1808 = None
        permute_1469: "f32[768, 8192]" = torch.ops.aten.permute.default(view_1809, [1, 0])
        mm_577: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1469, view_1);  permute_1469 = view_1 = None
        permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_1471: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_578: "f32[8192, 768]" = torch.ops.aten.mm.default(view_1809, permute_1471);  view_1809 = permute_1471 = None
        view_1810: "f32[8, 1024, 768]" = torch.ops.aten.reshape.default(mm_578, [8, 1024, 768]);  mm_578 = None
        add_389: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_388, view_1810);  add_388 = view_1810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1149: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_389, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_1: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_2: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_1150: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_389, mul_2);  add_389 = mul_2 = None
        sum_201: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_1150, [0, 1], True);  mul_1150 = None
        view_1811: "f32[768]" = torch.ops.aten.reshape.default(sum_201, [768]);  sum_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1151: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1149, mul_1)
        mul_1152: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1149, rsqrt);  mul_1149 = None
        sum_202: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_1151, [2], True);  mul_1151 = None
        add_390: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_386, mul_1152);  add_386 = mul_1152 = None
        pow_185: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_1153: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_202, -0.5);  sum_202 = None
        mul_1154: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_1153, pow_185);  mul_1153 = pow_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_208: "f32[8, 1024, 768]" = torch.ops.aten.expand.default(mul_1154, [8, 1024, 768]);  mul_1154 = None
        div_103: "f32[8, 1024, 768]" = torch.ops.aten.div.Scalar(expand_208, 768);  expand_208 = None
        pow_186: "f32[8, 1024, 768]" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 1.0);  mul_1 = None
        mul_1155: "f32[8, 1024, 768]" = torch.ops.aten.mul.Scalar(pow_186, 2.0);  pow_186 = None
        mul_1156: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(div_103, mul_1155);  div_103 = mul_1155 = None
        add_391: "f32[8, 1024, 768]" = torch.ops.aten.add.Tensor(add_390, mul_1156);  add_390 = mul_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_129: "f32[8, 1024, 768]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_1157: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_129, 1.1111111111111112);  convert_element_type_129 = None
        mul_1158: "f32[8, 1024, 768]" = torch.ops.aten.mul.Tensor(add_391, mul_1157);  add_391 = mul_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_4: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_23: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_4, -1);  eq_4 = None
        where_37: "f32[8, 1024, 768]" = torch.ops.aten.where.self(unsqueeze_23, full_default, mul_1158);  unsqueeze_23 = full_default = mul_1158 = None
        index_put_3: "f32[32128, 768]" = torch.ops.aten.index_put.default(full_default_32, [primals_1], where_37, True);  full_default_32 = primals_1 = where_37 = None
        add_392: "f32[32128, 768]" = torch.ops.aten.add.Tensor(add_307, index_put_3);  add_307 = index_put_3 = None
        return (None, add_392, view_1811, mm_577, mm_575, mm_573, index_put_2, mm_571, view_1788, mm_569, mm_567, view_1783, mm_565, mm_563, mm_561, mm_559, view_1760, mm_557, mm_555, view_1755, mm_553, mm_551, mm_549, mm_547, view_1732, mm_545, mm_543, view_1727, mm_541, mm_539, mm_537, mm_535, view_1704, mm_533, mm_531, view_1699, mm_529, mm_527, mm_525, mm_523, view_1676, mm_521, mm_519, view_1671, mm_517, mm_515, mm_513, mm_511, view_1648, mm_509, mm_507, view_1643, mm_505, mm_503, mm_501, mm_499, view_1620, mm_497, mm_495, view_1615, mm_493, mm_491, mm_489, mm_487, view_1592, mm_485, mm_483, view_1587, mm_481, mm_479, mm_477, mm_475, view_1564, mm_473, mm_471, view_1559, mm_469, mm_467, mm_465, mm_463, view_1536, mm_461, mm_459, view_1531, mm_457, mm_455, mm_453, mm_451, view_1508, mm_449, mm_447, view_1503, mm_445, mm_443, mm_441, mm_439, view_1480, mm_437, mm_435, view_1475, None, view_1474, mm_433, mm_431, mm_429, index_put, mm_427, view_1451, mm_425, mm_423, mm_421, mm_419, view_1428, mm_417, mm_415, view_1423, mm_413, mm_411, mm_409, mm_407, view_1400, mm_405, mm_403, mm_401, mm_399, view_1377, mm_397, mm_395, view_1372, mm_393, mm_391, mm_389, mm_387, view_1349, mm_385, mm_383, mm_381, mm_379, view_1326, mm_377, mm_375, view_1321, mm_373, mm_371, mm_369, mm_367, view_1298, mm_365, mm_363, mm_361, mm_359, view_1275, mm_357, mm_355, view_1270, mm_353, mm_351, mm_349, mm_347, view_1247, mm_345, mm_343, mm_341, mm_339, view_1224, mm_337, mm_335, view_1219, mm_333, mm_331, mm_329, mm_327, view_1196, mm_325, mm_323, mm_321, mm_319, view_1173, mm_317, mm_315, view_1168, mm_313, mm_311, mm_309, mm_307, view_1145, mm_305, mm_303, mm_301, mm_299, view_1122, mm_297, mm_295, view_1117, mm_293, mm_291, mm_289, mm_287, view_1094, mm_285, mm_283, mm_281, mm_279, view_1071, mm_277, mm_275, view_1066, mm_273, mm_271, mm_269, mm_267, view_1043, mm_265, mm_263, mm_261, mm_259, view_1020, mm_257, mm_255, view_1015, mm_253, mm_251, mm_249, mm_247, view_992, mm_245, mm_243, mm_241, mm_239, view_969, mm_237, mm_235, view_964, mm_233, mm_231, mm_229, mm_227, view_941, mm_225, mm_223, mm_221, mm_219, view_918, mm_217, mm_215, view_913, mm_213, mm_211, mm_209, mm_207, view_890, mm_205, mm_203, mm_201, mm_199, view_867, mm_197, mm_195, view_862)
