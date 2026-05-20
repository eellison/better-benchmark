class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 128]", primals_2: "f32[50257, 2048]", primals_4: "f32[2048]", primals_6: "f32[2048, 2048]", primals_7: "f32[2048, 2048]", primals_8: "f32[2048, 2048]", primals_9: "b8[1, 1, 2048, 2048]", primals_10: "f32[2048, 2048]", primals_12: "f32[2048]", primals_14: "f32[8192, 2048]", primals_16: "f32[2048, 8192]", primals_18: "f32[2048]", primals_20: "f32[2048, 2048]", primals_21: "f32[2048, 2048]", primals_22: "f32[2048, 2048]", primals_23: "b8[1, 1, 2048, 2048]", primals_24: "f32[2048, 2048]", primals_26: "f32[2048]", primals_28: "f32[8192, 2048]", primals_30: "f32[2048, 8192]", primals_32: "f32[2048]", primals_34: "f32[2048, 2048]", primals_35: "f32[2048, 2048]", primals_36: "f32[2048, 2048]", primals_37: "b8[1, 1, 2048, 2048]", primals_38: "f32[2048, 2048]", primals_40: "f32[2048]", primals_42: "f32[8192, 2048]", primals_44: "f32[2048, 8192]", primals_46: "f32[2048]", primals_48: "f32[2048, 2048]", primals_49: "f32[2048, 2048]", primals_50: "f32[2048, 2048]", primals_51: "b8[1, 1, 2048, 2048]", primals_52: "f32[2048, 2048]", primals_54: "f32[2048]", primals_56: "f32[8192, 2048]", primals_58: "f32[2048, 8192]", primals_60: "f32[2048]", primals_62: "f32[2048, 2048]", primals_63: "f32[2048, 2048]", primals_64: "f32[2048, 2048]", primals_65: "b8[1, 1, 2048, 2048]", primals_66: "f32[2048, 2048]", primals_68: "f32[2048]", primals_70: "f32[8192, 2048]", primals_72: "f32[2048, 8192]", primals_74: "f32[2048]", primals_76: "f32[2048, 2048]", primals_77: "f32[2048, 2048]", primals_78: "f32[2048, 2048]", primals_79: "b8[1, 1, 2048, 2048]", primals_80: "f32[2048, 2048]", primals_82: "f32[2048]", primals_84: "f32[8192, 2048]", primals_86: "f32[2048, 8192]", primals_88: "f32[2048]", primals_90: "f32[2048, 2048]", primals_91: "f32[2048, 2048]", primals_92: "f32[2048, 2048]", primals_93: "b8[1, 1, 2048, 2048]", primals_94: "f32[2048, 2048]", primals_96: "f32[2048]", primals_98: "f32[8192, 2048]", primals_100: "f32[2048, 8192]", primals_102: "f32[2048]", primals_104: "f32[2048, 2048]", primals_105: "f32[2048, 2048]", primals_106: "f32[2048, 2048]", primals_107: "b8[1, 1, 2048, 2048]", primals_108: "f32[2048, 2048]", primals_110: "f32[2048]", primals_112: "f32[8192, 2048]", primals_114: "f32[2048, 8192]", primals_116: "f32[2048]", primals_118: "f32[2048, 2048]", primals_119: "f32[2048, 2048]", primals_120: "f32[2048, 2048]", primals_121: "b8[1, 1, 2048, 2048]", primals_122: "f32[2048, 2048]", primals_124: "f32[2048]", primals_126: "f32[8192, 2048]", primals_128: "f32[2048, 8192]", primals_130: "f32[2048]", primals_132: "f32[2048, 2048]", primals_133: "f32[2048, 2048]", primals_134: "f32[2048, 2048]", primals_135: "b8[1, 1, 2048, 2048]", primals_136: "f32[2048, 2048]", primals_138: "f32[2048]", primals_140: "f32[8192, 2048]", primals_142: "f32[2048, 8192]", primals_144: "f32[2048]", primals_146: "f32[2048, 2048]", primals_147: "f32[2048, 2048]", primals_148: "f32[2048, 2048]", primals_149: "b8[1, 1, 2048, 2048]", primals_150: "f32[2048, 2048]", primals_152: "f32[2048]", primals_154: "f32[8192, 2048]", primals_156: "f32[2048, 8192]", primals_158: "f32[2048]", primals_160: "f32[2048, 2048]", primals_161: "f32[2048, 2048]", primals_162: "f32[2048, 2048]", primals_163: "b8[1, 1, 2048, 2048]", primals_164: "f32[2048, 2048]", primals_166: "f32[2048]", primals_168: "f32[8192, 2048]", primals_170: "f32[2048, 8192]", primals_172: "f32[2048]", primals_174: "f32[2048, 2048]", primals_175: "f32[2048, 2048]", primals_176: "f32[2048, 2048]", primals_177: "b8[1, 1, 2048, 2048]", primals_178: "f32[2048, 2048]", primals_180: "f32[2048]", primals_182: "f32[8192, 2048]", primals_184: "f32[2048, 8192]", primals_186: "f32[2048]", primals_188: "f32[2048, 2048]", primals_189: "f32[2048, 2048]", primals_190: "f32[2048, 2048]", primals_191: "b8[1, 1, 2048, 2048]", primals_192: "f32[2048, 2048]", primals_194: "f32[2048]", primals_196: "f32[8192, 2048]", primals_198: "f32[2048, 8192]", primals_200: "f32[2048]", primals_202: "f32[2048, 2048]", primals_203: "f32[2048, 2048]", primals_204: "f32[2048, 2048]", primals_205: "b8[1, 1, 2048, 2048]", primals_206: "f32[2048, 2048]", primals_208: "f32[2048]", primals_210: "f32[8192, 2048]", primals_212: "f32[2048, 8192]", primals_214: "f32[2048]", primals_216: "f32[2048, 2048]", primals_217: "f32[2048, 2048]", primals_218: "f32[2048, 2048]", primals_219: "b8[1, 1, 2048, 2048]", primals_220: "f32[2048, 2048]", primals_222: "f32[2048]", primals_224: "f32[8192, 2048]", primals_226: "f32[2048, 8192]", primals_228: "f32[2048]", primals_230: "f32[2048, 2048]", primals_231: "f32[2048, 2048]", primals_232: "f32[2048, 2048]", primals_233: "b8[1, 1, 2048, 2048]", primals_234: "f32[2048, 2048]", primals_236: "f32[2048]", primals_238: "f32[8192, 2048]", primals_240: "f32[2048, 8192]", primals_242: "f32[2048]", primals_244: "f32[2048, 2048]", primals_245: "f32[2048, 2048]", primals_246: "f32[2048, 2048]", primals_247: "b8[1, 1, 2048, 2048]", primals_248: "f32[2048, 2048]", primals_250: "f32[2048]", primals_252: "f32[8192, 2048]", primals_254: "f32[2048, 8192]", primals_256: "f32[2048]", primals_258: "f32[2048, 2048]", primals_259: "f32[2048, 2048]", primals_260: "f32[2048, 2048]", primals_261: "b8[1, 1, 2048, 2048]", primals_262: "f32[2048, 2048]", primals_264: "f32[2048]", primals_266: "f32[8192, 2048]", primals_268: "f32[2048, 8192]", primals_270: "f32[2048]", primals_272: "f32[2048, 2048]", primals_273: "f32[2048, 2048]", primals_274: "f32[2048, 2048]", primals_275: "b8[1, 1, 2048, 2048]", primals_276: "f32[2048, 2048]", primals_278: "f32[2048]", primals_280: "f32[8192, 2048]", primals_282: "f32[2048, 8192]", primals_284: "f32[2048]", primals_286: "f32[2048, 2048]", primals_287: "f32[2048, 2048]", primals_288: "f32[2048, 2048]", primals_289: "b8[1, 1, 2048, 2048]", primals_290: "f32[2048, 2048]", primals_292: "f32[2048]", primals_294: "f32[8192, 2048]", primals_296: "f32[2048, 8192]", primals_298: "f32[2048]", primals_300: "f32[2048, 2048]", primals_301: "f32[2048, 2048]", primals_302: "f32[2048, 2048]", primals_303: "b8[1, 1, 2048, 2048]", primals_304: "f32[2048, 2048]", primals_306: "f32[2048]", primals_308: "f32[8192, 2048]", primals_310: "f32[2048, 8192]", primals_312: "f32[2048]", primals_314: "f32[2048, 2048]", primals_315: "f32[2048, 2048]", primals_316: "f32[2048, 2048]", primals_317: "b8[1, 1, 2048, 2048]", primals_318: "f32[2048, 2048]", primals_320: "f32[2048]", primals_322: "f32[8192, 2048]", primals_324: "f32[2048, 8192]", primals_326: "f32[2048]", primals_328: "f32[2048, 2048]", primals_329: "f32[2048, 2048]", primals_330: "f32[2048, 2048]", primals_331: "b8[1, 1, 2048, 2048]", primals_332: "f32[2048, 2048]", primals_334: "f32[2048]", primals_336: "f32[8192, 2048]", primals_338: "f32[2048, 8192]", primals_340: "f32[2048]", embedding: "f32[32, 128, 2048]", unsqueeze: "i64[1, 128]", embedding_1: "f32[1, 128, 2048]", getitem_1: "f32[32, 128, 1]", rsqrt: "f32[32, 128, 1]", view: "f32[4096, 2048]", div: "f32[32, 16, 128, 128]", view_16: "f32[4096, 2048]", mul_2: "f32[32, 128, 2048]", view_18: "f32[4096, 2048]", addmm_1: "f32[4096, 8192]", view_20: "f32[4096, 8192]", mul_8: "f32[32, 128, 2048]", view_22: "f32[4096, 2048]", div_1: "f32[32, 16, 128, 128]", view_38: "f32[4096, 2048]", mul_10: "f32[32, 128, 2048]", view_40: "f32[4096, 2048]", addmm_4: "f32[4096, 8192]", view_42: "f32[4096, 8192]", mul_16: "f32[32, 128, 2048]", view_44: "f32[4096, 2048]", div_2: "f32[32, 16, 128, 128]", view_60: "f32[4096, 2048]", mul_18: "f32[32, 128, 2048]", view_62: "f32[4096, 2048]", addmm_7: "f32[4096, 8192]", view_64: "f32[4096, 8192]", mul_24: "f32[32, 128, 2048]", view_66: "f32[4096, 2048]", div_3: "f32[32, 16, 128, 128]", view_82: "f32[4096, 2048]", mul_26: "f32[32, 128, 2048]", view_84: "f32[4096, 2048]", addmm_10: "f32[4096, 8192]", view_86: "f32[4096, 8192]", mul_32: "f32[32, 128, 2048]", view_88: "f32[4096, 2048]", div_4: "f32[32, 16, 128, 128]", view_104: "f32[4096, 2048]", mul_34: "f32[32, 128, 2048]", view_106: "f32[4096, 2048]", addmm_13: "f32[4096, 8192]", view_108: "f32[4096, 8192]", mul_40: "f32[32, 128, 2048]", view_110: "f32[4096, 2048]", div_5: "f32[32, 16, 128, 128]", view_126: "f32[4096, 2048]", mul_42: "f32[32, 128, 2048]", view_128: "f32[4096, 2048]", addmm_16: "f32[4096, 8192]", view_130: "f32[4096, 8192]", mul_48: "f32[32, 128, 2048]", view_132: "f32[4096, 2048]", div_6: "f32[32, 16, 128, 128]", view_148: "f32[4096, 2048]", mul_50: "f32[32, 128, 2048]", view_150: "f32[4096, 2048]", addmm_19: "f32[4096, 8192]", view_152: "f32[4096, 8192]", mul_56: "f32[32, 128, 2048]", view_154: "f32[4096, 2048]", div_7: "f32[32, 16, 128, 128]", view_170: "f32[4096, 2048]", mul_58: "f32[32, 128, 2048]", view_172: "f32[4096, 2048]", addmm_22: "f32[4096, 8192]", view_174: "f32[4096, 8192]", mul_64: "f32[32, 128, 2048]", view_176: "f32[4096, 2048]", div_8: "f32[32, 16, 128, 128]", view_192: "f32[4096, 2048]", mul_66: "f32[32, 128, 2048]", view_194: "f32[4096, 2048]", addmm_25: "f32[4096, 8192]", view_196: "f32[4096, 8192]", mul_72: "f32[32, 128, 2048]", view_198: "f32[4096, 2048]", div_9: "f32[32, 16, 128, 128]", view_214: "f32[4096, 2048]", mul_74: "f32[32, 128, 2048]", view_216: "f32[4096, 2048]", addmm_28: "f32[4096, 8192]", view_218: "f32[4096, 8192]", mul_80: "f32[32, 128, 2048]", view_220: "f32[4096, 2048]", div_10: "f32[32, 16, 128, 128]", view_236: "f32[4096, 2048]", mul_82: "f32[32, 128, 2048]", view_238: "f32[4096, 2048]", addmm_31: "f32[4096, 8192]", view_240: "f32[4096, 8192]", mul_88: "f32[32, 128, 2048]", view_242: "f32[4096, 2048]", div_11: "f32[32, 16, 128, 128]", view_258: "f32[4096, 2048]", mul_90: "f32[32, 128, 2048]", view_260: "f32[4096, 2048]", addmm_34: "f32[4096, 8192]", view_262: "f32[4096, 8192]", mul_96: "f32[32, 128, 2048]", view_264: "f32[4096, 2048]", div_12: "f32[32, 16, 128, 128]", view_280: "f32[4096, 2048]", mul_98: "f32[32, 128, 2048]", view_282: "f32[4096, 2048]", addmm_37: "f32[4096, 8192]", view_284: "f32[4096, 8192]", mul_104: "f32[32, 128, 2048]", view_286: "f32[4096, 2048]", div_13: "f32[32, 16, 128, 128]", view_302: "f32[4096, 2048]", mul_106: "f32[32, 128, 2048]", view_304: "f32[4096, 2048]", addmm_40: "f32[4096, 8192]", view_306: "f32[4096, 8192]", mul_112: "f32[32, 128, 2048]", view_308: "f32[4096, 2048]", div_14: "f32[32, 16, 128, 128]", view_324: "f32[4096, 2048]", mul_114: "f32[32, 128, 2048]", view_326: "f32[4096, 2048]", addmm_43: "f32[4096, 8192]", view_328: "f32[4096, 8192]", mul_120: "f32[32, 128, 2048]", view_330: "f32[4096, 2048]", div_15: "f32[32, 16, 128, 128]", view_346: "f32[4096, 2048]", mul_122: "f32[32, 128, 2048]", view_348: "f32[4096, 2048]", addmm_46: "f32[4096, 8192]", view_350: "f32[4096, 8192]", mul_128: "f32[32, 128, 2048]", view_352: "f32[4096, 2048]", div_16: "f32[32, 16, 128, 128]", view_368: "f32[4096, 2048]", mul_130: "f32[32, 128, 2048]", view_370: "f32[4096, 2048]", addmm_49: "f32[4096, 8192]", view_372: "f32[4096, 8192]", mul_136: "f32[32, 128, 2048]", view_374: "f32[4096, 2048]", div_17: "f32[32, 16, 128, 128]", view_390: "f32[4096, 2048]", mul_138: "f32[32, 128, 2048]", view_392: "f32[4096, 2048]", addmm_52: "f32[4096, 8192]", view_394: "f32[4096, 8192]", mul_144: "f32[32, 128, 2048]", view_396: "f32[4096, 2048]", div_18: "f32[32, 16, 128, 128]", view_412: "f32[4096, 2048]", mul_146: "f32[32, 128, 2048]", view_414: "f32[4096, 2048]", addmm_55: "f32[4096, 8192]", view_416: "f32[4096, 8192]", mul_152: "f32[32, 128, 2048]", view_418: "f32[4096, 2048]", div_19: "f32[32, 16, 128, 128]", view_434: "f32[4096, 2048]", mul_154: "f32[32, 128, 2048]", view_436: "f32[4096, 2048]", addmm_58: "f32[4096, 8192]", view_438: "f32[4096, 8192]", mul_160: "f32[32, 128, 2048]", view_440: "f32[4096, 2048]", div_20: "f32[32, 16, 128, 128]", view_456: "f32[4096, 2048]", mul_162: "f32[32, 128, 2048]", view_458: "f32[4096, 2048]", addmm_61: "f32[4096, 8192]", view_460: "f32[4096, 8192]", mul_168: "f32[32, 128, 2048]", view_462: "f32[4096, 2048]", div_21: "f32[32, 16, 128, 128]", view_478: "f32[4096, 2048]", mul_170: "f32[32, 128, 2048]", view_480: "f32[4096, 2048]", addmm_64: "f32[4096, 8192]", view_482: "f32[4096, 8192]", mul_176: "f32[32, 128, 2048]", view_484: "f32[4096, 2048]", div_22: "f32[32, 16, 128, 128]", view_500: "f32[4096, 2048]", mul_178: "f32[32, 128, 2048]", view_502: "f32[4096, 2048]", addmm_67: "f32[4096, 8192]", view_504: "f32[4096, 8192]", mul_184: "f32[32, 128, 2048]", view_506: "f32[4096, 2048]", div_23: "f32[32, 16, 128, 128]", view_522: "f32[4096, 2048]", mul_186: "f32[32, 128, 2048]", view_524: "f32[4096, 2048]", addmm_70: "f32[4096, 8192]", view_526: "f32[4096, 8192]", mul_192: "f32[32, 128, 2048]", view_529: "f32[4096, 2048]", view_530: "f32[32, 128, 50257]", constant_pad_nd: "i64[32, 129]", amax_24: "f32[4096, 1]", log: "f32[4096, 1]", convert_element_type: "f32[]", div_26: "f32[32, 128, 1]", div_27: "f32[32, 128, 1]", permute_283: "f32[512, 128, 128]", permute_284: "f32[512, 128, 128]", permute_285: "f32[512, 128, 128]", div_28: "f32[32, 128, 1]", div_29: "f32[32, 128, 1]", permute_316: "f32[512, 128, 128]", permute_317: "f32[512, 128, 128]", permute_318: "f32[512, 128, 128]", div_30: "f32[32, 128, 1]", div_31: "f32[32, 128, 1]", permute_349: "f32[512, 128, 128]", permute_350: "f32[512, 128, 128]", permute_351: "f32[512, 128, 128]", div_32: "f32[32, 128, 1]", div_33: "f32[32, 128, 1]", permute_382: "f32[512, 128, 128]", permute_383: "f32[512, 128, 128]", permute_384: "f32[512, 128, 128]", div_34: "f32[32, 128, 1]", div_35: "f32[32, 128, 1]", permute_415: "f32[512, 128, 128]", permute_416: "f32[512, 128, 128]", permute_417: "f32[512, 128, 128]", div_36: "f32[32, 128, 1]", div_37: "f32[32, 128, 1]", permute_448: "f32[512, 128, 128]", permute_449: "f32[512, 128, 128]", permute_450: "f32[512, 128, 128]", div_38: "f32[32, 128, 1]", div_39: "f32[32, 128, 1]", permute_481: "f32[512, 128, 128]", permute_482: "f32[512, 128, 128]", permute_483: "f32[512, 128, 128]", div_40: "f32[32, 128, 1]", div_41: "f32[32, 128, 1]", permute_514: "f32[512, 128, 128]", permute_515: "f32[512, 128, 128]", permute_516: "f32[512, 128, 128]", div_42: "f32[32, 128, 1]", div_43: "f32[32, 128, 1]", permute_547: "f32[512, 128, 128]", permute_548: "f32[512, 128, 128]", permute_549: "f32[512, 128, 128]", div_44: "f32[32, 128, 1]", div_45: "f32[32, 128, 1]", permute_580: "f32[512, 128, 128]", permute_581: "f32[512, 128, 128]", permute_582: "f32[512, 128, 128]", div_46: "f32[32, 128, 1]", div_47: "f32[32, 128, 1]", permute_613: "f32[512, 128, 128]", permute_614: "f32[512, 128, 128]", permute_615: "f32[512, 128, 128]", div_48: "f32[32, 128, 1]", div_49: "f32[32, 128, 1]", permute_646: "f32[512, 128, 128]", permute_647: "f32[512, 128, 128]", permute_648: "f32[512, 128, 128]", div_50: "f32[32, 128, 1]", div_51: "f32[32, 128, 1]", permute_679: "f32[512, 128, 128]", permute_680: "f32[512, 128, 128]", permute_681: "f32[512, 128, 128]", div_52: "f32[32, 128, 1]", div_53: "f32[32, 128, 1]", permute_712: "f32[512, 128, 128]", permute_713: "f32[512, 128, 128]", permute_714: "f32[512, 128, 128]", div_54: "f32[32, 128, 1]", div_55: "f32[32, 128, 1]", permute_745: "f32[512, 128, 128]", permute_746: "f32[512, 128, 128]", permute_747: "f32[512, 128, 128]", div_56: "f32[32, 128, 1]", div_57: "f32[32, 128, 1]", permute_778: "f32[512, 128, 128]", permute_779: "f32[512, 128, 128]", permute_780: "f32[512, 128, 128]", div_58: "f32[32, 128, 1]", div_59: "f32[32, 128, 1]", permute_811: "f32[512, 128, 128]", permute_812: "f32[512, 128, 128]", permute_813: "f32[512, 128, 128]", div_60: "f32[32, 128, 1]", div_61: "f32[32, 128, 1]", permute_844: "f32[512, 128, 128]", permute_845: "f32[512, 128, 128]", permute_846: "f32[512, 128, 128]", div_62: "f32[32, 128, 1]", div_63: "f32[32, 128, 1]", permute_877: "f32[512, 128, 128]", permute_878: "f32[512, 128, 128]", permute_879: "f32[512, 128, 128]", div_64: "f32[32, 128, 1]", div_65: "f32[32, 128, 1]", permute_910: "f32[512, 128, 128]", permute_911: "f32[512, 128, 128]", permute_912: "f32[512, 128, 128]", div_66: "f32[32, 128, 1]", div_67: "f32[32, 128, 1]", permute_943: "f32[512, 128, 128]", permute_944: "f32[512, 128, 128]", permute_945: "f32[512, 128, 128]", div_68: "f32[32, 128, 1]", div_69: "f32[32, 128, 1]", permute_976: "f32[512, 128, 128]", permute_977: "f32[512, 128, 128]", permute_978: "f32[512, 128, 128]", div_70: "f32[32, 128, 1]", div_71: "f32[32, 128, 1]", permute_1009: "f32[512, 128, 128]", permute_1010: "f32[512, 128, 128]", permute_1011: "f32[512, 128, 128]", div_72: "f32[32, 128, 1]", div_73: "f32[32, 128, 1]", permute_1042: "f32[512, 128, 128]", permute_1043: "f32[512, 128, 128]", permute_1044: "f32[512, 128, 128]", tangents_1: "f32[]", tangents_2: "f32[32, 128, 50257]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_25: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_52: "i64[32, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_169: "i64[32, 128]" = torch.ops.aten.clone.default(slice_52, memory_format = torch.contiguous_format);  slice_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_532: "i64[4096]" = torch.ops.aten.reshape.default(clone_169, [-1]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_11: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(view_532, 1);  view_532 = None
        ne_4: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_11, -100)
        full_default_27: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "i64[4096, 1]" = torch.ops.aten.where.self(ne_4, unsqueeze_11, full_default_27);  unsqueeze_11 = full_default_27 = None

        # No stacktrace found for following nodes
        iota_default: "i64[50257]" = torch.ops.prims.iota.default(50257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50257]" = torch.ops.aten.reshape.default(iota_default, [1, 50257]);  iota_default = None
        expand_default: "i64[4096, 50257]" = torch.ops.aten.expand.default(where_27, [4096, 50257]);  where_27 = None
        eq_tensor: "b8[4096, 50257]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[4096, 50257]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_28: "f32[4096, 1]" = torch.ops.aten.where.self(ne_4, div_25, full_default_1);  ne_4 = div_25 = None
        mul_194: "f32[4096, 50257]" = torch.ops.aten.mul.Tensor(where_self, where_28);  where_self = where_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_531: "f32[4096, 50257]" = torch.ops.aten.reshape.default(view_530, [-1, 50257]);  view_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_75: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(view_531, amax_24);  view_531 = amax_24 = None
        sub_76: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = log = None
        exp_25: "f32[4096, 50257]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_28: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [1], True)
        mul_195: "f32[4096, 50257]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_77: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_533: "f32[32, 128, 50257]" = torch.ops.aten.reshape.default(sub_77, [32, 128, 50257]);  sub_77 = None
        add_222: "f32[32, 128, 50257]" = torch.ops.aten.add.Tensor(tangents_2, view_533);  tangents_2 = view_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_534: "f32[4096, 50257]" = torch.ops.aten.reshape.default(add_222, [4096, 50257]);  add_222 = None
        permute_265: "f32[50257, 4096]" = torch.ops.aten.permute.default(view_534, [1, 0])
        constant_pad_nd_default_2: "f32[50260, 4096]" = torch.ops.aten.constant_pad_nd.default(permute_265, [0, 0, 0, 3]);  permute_265 = None
        mm_default_1: "f32[50260, 2048]" = torch.ops.aten.mm.default(constant_pad_nd_default_2, view_529);  constant_pad_nd_default_2 = view_529 = None
        slice_tensor: "f32[50257, 2048]" = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -3);  mm_default_1 = None
        permute_264: "f32[2048, 50257]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_267: "f32[50257, 2048]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        constant_pad_nd_default: "f32[4096, 50260]" = torch.ops.aten.constant_pad_nd.default(view_534, [0, 3, 0, 0]);  view_534 = None
        constant_pad_nd_default_1: "f32[50260, 2048]" = torch.ops.aten.constant_pad_nd.default(permute_267, [0, 0, 0, 3]);  permute_267 = None
        mm_default: "f32[4096, 2048]" = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        view_535: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_default, [32, 128, 2048]);  mm_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:492 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_197: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_535, primals_340);  primals_340 = None
        mul_198: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_197, 2048)
        sum_29: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
        mul_199: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_197, mul_192);  mul_197 = None
        sum_30: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
        mul_200: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_192, sum_30);  sum_30 = None
        sub_79: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_198, sum_29);  mul_198 = sum_29 = None
        sub_80: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_79, mul_200);  sub_79 = mul_200 = None
        mul_201: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_26, sub_80);  div_26 = sub_80 = None
        mul_202: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_535, mul_192);  mul_192 = None
        sum_31: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
        sum_32: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_535, [0, 1]);  view_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_537: "f32[4096, 2048]" = torch.ops.aten.reshape.default(mul_201, [4096, 2048])
        permute_263: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_338, [1, 0]);  primals_338 = None
        permute_269: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        mm_75: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_537, permute_269);  permute_269 = None
        permute_270: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_537, [1, 0])
        mm_76: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_270, view_526);  permute_270 = view_526 = None
        sum_33: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
        view_538: "f32[2048]" = torch.ops.aten.reshape.default(sum_33, [2048]);  sum_33 = None
        view_539: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_75, [32, 128, 8192]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_525: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_70, [32, 128, 8192]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_188: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        mul_203: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_539, mul_188);  mul_188 = None
        pow_24: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_525, 3.0)
        mul_189: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_217: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_525, mul_189);  mul_189 = None
        mul_190: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_217, 0.7978845608028654);  add_217 = None
        tanh_23: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_218: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_204: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_539, add_218);  view_539 = add_218 = None
        mul_205: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_81: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_205);  mul_205 = None
        mul_206: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_203, sub_81);  mul_203 = sub_81 = None
        mul_207: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_206, 0.7978845608028654);  mul_206 = None
        mul_208: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_207, 0.044715)
        pow_25: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_525, 2.0);  view_525 = None
        mul_209: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_25, 3.0);  pow_25 = None
        mul_210: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
        add_223: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_207, mul_210);  mul_207 = mul_210 = None
        mul_211: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_204, 0.5);  mul_204 = None
        add_224: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_223, mul_211);  add_223 = mul_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_540: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_224, [4096, 8192]);  add_224 = None
        permute_262: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_336, [1, 0]);  primals_336 = None
        permute_273: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        mm_77: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_540, permute_273);  permute_273 = None
        permute_274: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_540, [1, 0])
        mm_78: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_274, view_524);  permute_274 = view_524 = None
        sum_34: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True);  view_540 = None
        view_541: "f32[8192]" = torch.ops.aten.reshape.default(sum_34, [8192]);  sum_34 = None
        view_542: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_77, [32, 128, 2048]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_213: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_542, primals_334);  primals_334 = None
        mul_214: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_213, 2048)
        sum_35: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
        mul_215: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_213, mul_186);  mul_213 = None
        sum_36: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
        mul_216: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_186, sum_36);  sum_36 = None
        sub_83: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_214, sum_35);  mul_214 = sum_35 = None
        sub_84: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_83, mul_216);  sub_83 = mul_216 = None
        mul_217: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_27, sub_84);  div_27 = sub_84 = None
        mul_218: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_542, mul_186);  mul_186 = None
        sum_37: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
        sum_38: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_542, [0, 1]);  view_542 = None
        add_225: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_201, mul_217);  mul_201 = mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_543: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_225, [4096, 2048])
        permute_261: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_332, [1, 0]);  primals_332 = None
        permute_277: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        mm_79: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_543, permute_277);  permute_277 = None
        permute_278: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_543, [1, 0])
        mm_80: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_278, view_522);  permute_278 = view_522 = None
        sum_39: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True);  view_543 = None
        view_544: "f32[2048]" = torch.ops.aten.reshape.default(sum_39, [2048]);  sum_39 = None
        view_545: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_79, [32, 128, 2048]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_546: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_545, [32, 128, 16, 128]);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_281: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_170: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_547: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_170, [512, 128, 128]);  clone_170 = None
        expand_96: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_23, [32, 16, 128, 128])
        view_518: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_96, [512, 128, 128]);  expand_96 = None
        permute_282: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_518, [0, 2, 1]);  view_518 = None
        bmm_48: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_282, view_547);  permute_282 = None
        bmm_49: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_547, permute_283);  view_547 = permute_283 = None
        view_548: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_48, [32, 16, 128, 128]);  bmm_48 = None
        view_549: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_49, [32, 16, 128, 128]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_219: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_549, div_23);  view_549 = None
        sum_40: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
        neg_1: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_1, sum_40, mul_219);  neg_1 = sum_40 = mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_50: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_331, 2, 0, 128);  primals_331 = None
        slice_51: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_50, 3, 0, 128);  slice_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_29: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_51, fma, full_default_1);  slice_51 = fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_550: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_29, [512, 128, 128]);  where_29 = None
        bmm_50: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_284, view_550);  permute_284 = None
        bmm_51: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_550, permute_285);  view_550 = permute_285 = None
        view_551: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_50, [32, 16, 128, 128]);  bmm_50 = None
        view_552: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_51, [32, 16, 128, 128]);  bmm_51 = None
        permute_286: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_551, [0, 1, 3, 2]);  view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_287: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_171: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_553: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_171, [32, 128, 2048]);  clone_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_288: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_286, [0, 2, 1, 3]);  permute_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_554: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_288, [32, 128, 2048]);  permute_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_289: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_552, [0, 2, 1, 3]);  view_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_172: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
        view_555: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_172, [32, 128, 2048]);  clone_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_556: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_553, [4096, 2048]);  view_553 = None
        permute_290: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_81: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_290, view_506);  permute_290 = None
        permute_255: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_330, [1, 0]);  primals_330 = None
        permute_292: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        mm_82: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_556, permute_292);  view_556 = permute_292 = None
        view_557: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_82, [32, 128, 2048]);  mm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_173: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_554, memory_format = torch.contiguous_format);  view_554 = None
        view_558: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_173, [4096, 2048]);  clone_173 = None
        permute_294: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_558, [1, 0])
        mm_83: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_294, view_506);  permute_294 = None
        permute_254: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
        permute_296: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        mm_84: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_558, permute_296);  view_558 = permute_296 = None
        view_559: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_84, [32, 128, 2048]);  mm_84 = None
        add_226: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_557, view_559);  view_557 = view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_560: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_555, [4096, 2048]);  view_555 = None
        permute_298: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_85: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_298, view_506);  permute_298 = view_506 = None
        permute_253: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_328, [1, 0]);  primals_328 = None
        permute_300: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        mm_86: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_560, permute_300);  view_560 = permute_300 = None
        view_561: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_86, [32, 128, 2048]);  mm_86 = None
        add_227: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_226, view_561);  add_226 = view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_221: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_227, primals_326);  primals_326 = None
        mul_222: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_221, 2048)
        sum_41: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True)
        mul_223: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_221, mul_184);  mul_221 = None
        sum_42: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True);  mul_223 = None
        mul_224: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_184, sum_42);  sum_42 = None
        sub_86: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_222, sum_41);  mul_222 = sum_41 = None
        sub_87: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_86, mul_224);  sub_86 = mul_224 = None
        mul_225: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_28, sub_87);  div_28 = sub_87 = None
        mul_226: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_227, mul_184);  mul_184 = None
        sum_43: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_226, [0, 1]);  mul_226 = None
        sum_44: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None
        add_228: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_225, mul_225);  add_225 = mul_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_562: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_228, [4096, 2048])
        permute_252: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_324, [1, 0]);  primals_324 = None
        permute_302: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        mm_87: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_562, permute_302);  permute_302 = None
        permute_303: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_88: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_303, view_504);  permute_303 = view_504 = None
        sum_45: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        view_563: "f32[2048]" = torch.ops.aten.reshape.default(sum_45, [2048]);  sum_45 = None
        view_564: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_87, [32, 128, 8192]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_503: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_67, [32, 128, 8192]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_180: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        mul_227: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_564, mul_180);  mul_180 = None
        pow_23: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_503, 3.0)
        mul_181: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_208: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_503, mul_181);  mul_181 = None
        mul_182: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_208, 0.7978845608028654);  add_208 = None
        tanh_22: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
        add_209: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_228: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_564, add_209);  view_564 = add_209 = None
        mul_229: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_88: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_229);  mul_229 = None
        mul_230: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_227, sub_88);  mul_227 = sub_88 = None
        mul_231: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_230, 0.7978845608028654);  mul_230 = None
        mul_232: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_231, 0.044715)
        pow_26: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_503, 2.0);  view_503 = None
        mul_233: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_234: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        add_229: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_231, mul_234);  mul_231 = mul_234 = None
        mul_235: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_228, 0.5);  mul_228 = None
        add_230: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_229, mul_235);  add_229 = mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_565: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_230, [4096, 8192]);  add_230 = None
        permute_251: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_322, [1, 0]);  primals_322 = None
        permute_306: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        mm_89: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_565, permute_306);  permute_306 = None
        permute_307: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_90: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_307, view_502);  permute_307 = view_502 = None
        sum_46: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
        view_566: "f32[8192]" = torch.ops.aten.reshape.default(sum_46, [8192]);  sum_46 = None
        view_567: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_89, [32, 128, 2048]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_237: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_567, primals_320);  primals_320 = None
        mul_238: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_237, 2048)
        sum_47: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
        mul_239: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_237, mul_178);  mul_237 = None
        sum_48: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
        mul_240: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_178, sum_48);  sum_48 = None
        sub_90: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_238, sum_47);  mul_238 = sum_47 = None
        sub_91: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_90, mul_240);  sub_90 = mul_240 = None
        mul_241: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_29, sub_91);  div_29 = sub_91 = None
        mul_242: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_567, mul_178);  mul_178 = None
        sum_49: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
        sum_50: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_567, [0, 1]);  view_567 = None
        add_231: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_228, mul_241);  add_228 = mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_568: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_231, [4096, 2048])
        permute_250: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_318, [1, 0]);  primals_318 = None
        permute_310: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        mm_91: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_568, permute_310);  permute_310 = None
        permute_311: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_92: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_311, view_500);  permute_311 = view_500 = None
        sum_51: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_568, [0], True);  view_568 = None
        view_569: "f32[2048]" = torch.ops.aten.reshape.default(sum_51, [2048]);  sum_51 = None
        view_570: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_91, [32, 128, 2048]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_571: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_570, [32, 128, 16, 128]);  view_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_314: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_174: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
        view_572: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_174, [512, 128, 128]);  clone_174 = None
        expand_92: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_22, [32, 16, 128, 128])
        view_496: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_92, [512, 128, 128]);  expand_92 = None
        permute_315: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
        bmm_52: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_315, view_572);  permute_315 = None
        bmm_53: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_572, permute_316);  view_572 = permute_316 = None
        view_573: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_52, [32, 16, 128, 128]);  bmm_52 = None
        view_574: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_53, [32, 16, 128, 128]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_243: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_574, div_22);  view_574 = None
        sum_52: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [-1], True)
        neg_2: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_1: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_52, mul_243);  neg_2 = sum_52 = mul_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_48: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_317, 2, 0, 128);  primals_317 = None
        slice_49: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_48, 3, 0, 128);  slice_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_30: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_49, fma_1, full_default_1);  slice_49 = fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_575: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_30, [512, 128, 128]);  where_30 = None
        bmm_54: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_317, view_575);  permute_317 = None
        bmm_55: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_575, permute_318);  view_575 = permute_318 = None
        view_576: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_54, [32, 16, 128, 128]);  bmm_54 = None
        view_577: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_55, [32, 16, 128, 128]);  bmm_55 = None
        permute_319: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_576, [0, 1, 3, 2]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_320: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_175: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_578: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_175, [32, 128, 2048]);  clone_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_321: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_319, [0, 2, 1, 3]);  permute_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_579: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_321, [32, 128, 2048]);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_322: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_577, [0, 2, 1, 3]);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_176: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
        view_580: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_176, [32, 128, 2048]);  clone_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_581: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_578, [4096, 2048]);  view_578 = None
        permute_323: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_581, [1, 0])
        mm_93: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_323, view_484);  permute_323 = None
        permute_244: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_316, [1, 0]);  primals_316 = None
        permute_325: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        mm_94: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_581, permute_325);  view_581 = permute_325 = None
        view_582: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_94, [32, 128, 2048]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_177: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_579, memory_format = torch.contiguous_format);  view_579 = None
        view_583: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_177, [4096, 2048]);  clone_177 = None
        permute_327: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_583, [1, 0])
        mm_95: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_327, view_484);  permute_327 = None
        permute_243: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
        permute_329: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        mm_96: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_583, permute_329);  view_583 = permute_329 = None
        view_584: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_96, [32, 128, 2048]);  mm_96 = None
        add_232: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_582, view_584);  view_582 = view_584 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_585: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_580, [4096, 2048]);  view_580 = None
        permute_331: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_97: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_331, view_484);  permute_331 = view_484 = None
        permute_242: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_314, [1, 0]);  primals_314 = None
        permute_333: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        mm_98: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_585, permute_333);  view_585 = permute_333 = None
        view_586: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_98, [32, 128, 2048]);  mm_98 = None
        add_233: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_232, view_586);  add_232 = view_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_245: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_233, primals_312);  primals_312 = None
        mul_246: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_245, 2048)
        sum_53: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
        mul_247: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_245, mul_176);  mul_245 = None
        sum_54: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
        mul_248: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_176, sum_54);  sum_54 = None
        sub_93: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_246, sum_53);  mul_246 = sum_53 = None
        sub_94: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_93, mul_248);  sub_93 = mul_248 = None
        mul_249: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_30, sub_94);  div_30 = sub_94 = None
        mul_250: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_233, mul_176);  mul_176 = None
        sum_55: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
        sum_56: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
        add_234: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_231, mul_249);  add_231 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_587: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_234, [4096, 2048])
        permute_241: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_310, [1, 0]);  primals_310 = None
        permute_335: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        mm_99: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_587, permute_335);  permute_335 = None
        permute_336: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_587, [1, 0])
        mm_100: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_336, view_482);  permute_336 = view_482 = None
        sum_57: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
        view_588: "f32[2048]" = torch.ops.aten.reshape.default(sum_57, [2048]);  sum_57 = None
        view_589: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_99, [32, 128, 8192]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_481: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_64, [32, 128, 8192]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_172: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
        mul_251: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_589, mul_172);  mul_172 = None
        pow_22: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_481, 3.0)
        mul_173: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_199: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_481, mul_173);  mul_173 = None
        mul_174: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_199, 0.7978845608028654);  add_199 = None
        tanh_21: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
        add_200: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_252: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_589, add_200);  view_589 = add_200 = None
        mul_253: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_95: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_253);  mul_253 = None
        mul_254: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_251, sub_95);  mul_251 = sub_95 = None
        mul_255: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_254, 0.7978845608028654);  mul_254 = None
        mul_256: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_255, 0.044715)
        pow_27: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_481, 2.0);  view_481 = None
        mul_257: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_27, 3.0);  pow_27 = None
        mul_258: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        add_235: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_255, mul_258);  mul_255 = mul_258 = None
        mul_259: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_252, 0.5);  mul_252 = None
        add_236: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_235, mul_259);  add_235 = mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_590: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_236, [4096, 8192]);  add_236 = None
        permute_240: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_308, [1, 0]);  primals_308 = None
        permute_339: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        mm_101: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_590, permute_339);  permute_339 = None
        permute_340: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_102: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_340, view_480);  permute_340 = view_480 = None
        sum_58: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        view_591: "f32[8192]" = torch.ops.aten.reshape.default(sum_58, [8192]);  sum_58 = None
        view_592: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_101, [32, 128, 2048]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_261: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_592, primals_306);  primals_306 = None
        mul_262: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_261, 2048)
        sum_59: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True)
        mul_263: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_261, mul_170);  mul_261 = None
        sum_60: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True);  mul_263 = None
        mul_264: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, sum_60);  sum_60 = None
        sub_97: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_262, sum_59);  mul_262 = sum_59 = None
        sub_98: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_97, mul_264);  sub_97 = mul_264 = None
        mul_265: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_31, sub_98);  div_31 = sub_98 = None
        mul_266: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_592, mul_170);  mul_170 = None
        sum_61: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 1]);  mul_266 = None
        sum_62: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_592, [0, 1]);  view_592 = None
        add_237: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_234, mul_265);  add_234 = mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_593: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_237, [4096, 2048])
        permute_239: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
        permute_343: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        mm_103: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_593, permute_343);  permute_343 = None
        permute_344: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_104: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_344, view_478);  permute_344 = view_478 = None
        sum_63: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
        view_594: "f32[2048]" = torch.ops.aten.reshape.default(sum_63, [2048]);  sum_63 = None
        view_595: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_103, [32, 128, 2048]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_596: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_595, [32, 128, 16, 128]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_347: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_178: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        view_597: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_178, [512, 128, 128]);  clone_178 = None
        expand_88: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_21, [32, 16, 128, 128])
        view_474: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_88, [512, 128, 128]);  expand_88 = None
        permute_348: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None
        bmm_56: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_348, view_597);  permute_348 = None
        bmm_57: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_597, permute_349);  view_597 = permute_349 = None
        view_598: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_56, [32, 16, 128, 128]);  bmm_56 = None
        view_599: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_57, [32, 16, 128, 128]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_267: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_599, div_21);  view_599 = None
        sum_64: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [-1], True)
        neg_3: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_2: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_64, mul_267);  neg_3 = sum_64 = mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_46: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_303, 2, 0, 128);  primals_303 = None
        slice_47: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_46, 3, 0, 128);  slice_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_31: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_47, fma_2, full_default_1);  slice_47 = fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_600: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_31, [512, 128, 128]);  where_31 = None
        bmm_58: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_350, view_600);  permute_350 = None
        bmm_59: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_600, permute_351);  view_600 = permute_351 = None
        view_601: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_58, [32, 16, 128, 128]);  bmm_58 = None
        view_602: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_59, [32, 16, 128, 128]);  bmm_59 = None
        permute_352: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_601, [0, 1, 3, 2]);  view_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_353: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_179: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_603: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_179, [32, 128, 2048]);  clone_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_354: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_352, [0, 2, 1, 3]);  permute_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_604: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_354, [32, 128, 2048]);  permute_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_355: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_602, [0, 2, 1, 3]);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_180: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
        view_605: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_180, [32, 128, 2048]);  clone_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_606: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_603, [4096, 2048]);  view_603 = None
        permute_356: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_606, [1, 0])
        mm_105: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_356, view_462);  permute_356 = None
        permute_233: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_302, [1, 0]);  primals_302 = None
        permute_358: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        mm_106: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_606, permute_358);  view_606 = permute_358 = None
        view_607: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_106, [32, 128, 2048]);  mm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_181: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_604, memory_format = torch.contiguous_format);  view_604 = None
        view_608: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_181, [4096, 2048]);  clone_181 = None
        permute_360: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_608, [1, 0])
        mm_107: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_360, view_462);  permute_360 = None
        permute_232: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
        permute_362: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        mm_108: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_608, permute_362);  view_608 = permute_362 = None
        view_609: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_108, [32, 128, 2048]);  mm_108 = None
        add_238: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_607, view_609);  view_607 = view_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_610: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_605, [4096, 2048]);  view_605 = None
        permute_364: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_109: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_364, view_462);  permute_364 = view_462 = None
        permute_231: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_300, [1, 0]);  primals_300 = None
        permute_366: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        mm_110: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_610, permute_366);  view_610 = permute_366 = None
        view_611: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_110, [32, 128, 2048]);  mm_110 = None
        add_239: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_238, view_611);  add_238 = view_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_269: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_239, primals_298);  primals_298 = None
        mul_270: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_269, 2048)
        sum_65: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
        mul_271: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_269, mul_168);  mul_269 = None
        sum_66: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        mul_272: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_168, sum_66);  sum_66 = None
        sub_100: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_270, sum_65);  mul_270 = sum_65 = None
        sub_101: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_100, mul_272);  sub_100 = mul_272 = None
        mul_273: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_32, sub_101);  div_32 = sub_101 = None
        mul_274: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_239, mul_168);  mul_168 = None
        sum_67: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
        sum_68: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
        add_240: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_237, mul_273);  add_237 = mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_612: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_240, [4096, 2048])
        permute_230: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_296, [1, 0]);  primals_296 = None
        permute_368: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        mm_111: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_612, permute_368);  permute_368 = None
        permute_369: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_112: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_369, view_460);  permute_369 = view_460 = None
        sum_69: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        view_613: "f32[2048]" = torch.ops.aten.reshape.default(sum_69, [2048]);  sum_69 = None
        view_614: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_111, [32, 128, 8192]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_459: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_61, [32, 128, 8192]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_164: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_459, 0.5)
        mul_275: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_614, mul_164);  mul_164 = None
        pow_21: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_459, 3.0)
        mul_165: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_190: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_459, mul_165);  mul_165 = None
        mul_166: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_190, 0.7978845608028654);  add_190 = None
        tanh_20: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_191: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_276: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_614, add_191);  view_614 = add_191 = None
        mul_277: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_102: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_277);  mul_277 = None
        mul_278: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_275, sub_102);  mul_275 = sub_102 = None
        mul_279: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_278, 0.7978845608028654);  mul_278 = None
        mul_280: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_279, 0.044715)
        pow_28: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_459, 2.0);  view_459 = None
        mul_281: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
        mul_282: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_280, mul_281);  mul_280 = mul_281 = None
        add_241: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_279, mul_282);  mul_279 = mul_282 = None
        mul_283: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_276, 0.5);  mul_276 = None
        add_242: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_241, mul_283);  add_241 = mul_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_615: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_242, [4096, 8192]);  add_242 = None
        permute_229: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_294, [1, 0]);  primals_294 = None
        permute_372: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        mm_113: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_615, permute_372);  permute_372 = None
        permute_373: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_114: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_373, view_458);  permute_373 = view_458 = None
        sum_70: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
        view_616: "f32[8192]" = torch.ops.aten.reshape.default(sum_70, [8192]);  sum_70 = None
        view_617: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_113, [32, 128, 2048]);  mm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_285: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_617, primals_292);  primals_292 = None
        mul_286: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_285, 2048)
        sum_71: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True)
        mul_287: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_285, mul_162);  mul_285 = None
        sum_72: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
        mul_288: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_162, sum_72);  sum_72 = None
        sub_104: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_286, sum_71);  mul_286 = sum_71 = None
        sub_105: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_104, mul_288);  sub_104 = mul_288 = None
        mul_289: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_33, sub_105);  div_33 = sub_105 = None
        mul_290: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_617, mul_162);  mul_162 = None
        sum_73: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1]);  mul_290 = None
        sum_74: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_617, [0, 1]);  view_617 = None
        add_243: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_240, mul_289);  add_240 = mul_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_618: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_243, [4096, 2048])
        permute_228: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_290, [1, 0]);  primals_290 = None
        permute_376: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        mm_115: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_618, permute_376);  permute_376 = None
        permute_377: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_116: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_377, view_456);  permute_377 = view_456 = None
        sum_75: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
        view_619: "f32[2048]" = torch.ops.aten.reshape.default(sum_75, [2048]);  sum_75 = None
        view_620: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_115, [32, 128, 2048]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_621: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_620, [32, 128, 16, 128]);  view_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_380: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_621, [0, 2, 1, 3]);  view_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_182: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        view_622: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_182, [512, 128, 128]);  clone_182 = None
        expand_84: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_20, [32, 16, 128, 128])
        view_452: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_84, [512, 128, 128]);  expand_84 = None
        permute_381: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_452, [0, 2, 1]);  view_452 = None
        bmm_60: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_381, view_622);  permute_381 = None
        bmm_61: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_622, permute_382);  view_622 = permute_382 = None
        view_623: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_60, [32, 16, 128, 128]);  bmm_60 = None
        view_624: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_61, [32, 16, 128, 128]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_291: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_624, div_20);  view_624 = None
        sum_76: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [-1], True)
        neg_4: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_3: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_4, sum_76, mul_291);  neg_4 = sum_76 = mul_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_44: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_289, 2, 0, 128);  primals_289 = None
        slice_45: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_44, 3, 0, 128);  slice_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_32: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_45, fma_3, full_default_1);  slice_45 = fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_625: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_32, [512, 128, 128]);  where_32 = None
        bmm_62: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_383, view_625);  permute_383 = None
        bmm_63: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_625, permute_384);  view_625 = permute_384 = None
        view_626: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_62, [32, 16, 128, 128]);  bmm_62 = None
        view_627: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_63, [32, 16, 128, 128]);  bmm_63 = None
        permute_385: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_626, [0, 1, 3, 2]);  view_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_386: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_623, [0, 2, 1, 3]);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_183: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_628: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_183, [32, 128, 2048]);  clone_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_387: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_385, [0, 2, 1, 3]);  permute_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_629: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_387, [32, 128, 2048]);  permute_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_388: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_184: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
        view_630: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_184, [32, 128, 2048]);  clone_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_631: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_628, [4096, 2048]);  view_628 = None
        permute_389: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_117: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_389, view_440);  permute_389 = None
        permute_222: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_288, [1, 0]);  primals_288 = None
        permute_391: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        mm_118: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_631, permute_391);  view_631 = permute_391 = None
        view_632: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_118, [32, 128, 2048]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_185: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_629, memory_format = torch.contiguous_format);  view_629 = None
        view_633: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_185, [4096, 2048]);  clone_185 = None
        permute_393: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_633, [1, 0])
        mm_119: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_393, view_440);  permute_393 = None
        permute_221: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
        permute_395: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        mm_120: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_633, permute_395);  view_633 = permute_395 = None
        view_634: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_120, [32, 128, 2048]);  mm_120 = None
        add_244: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_632, view_634);  view_632 = view_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_635: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_630, [4096, 2048]);  view_630 = None
        permute_397: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_635, [1, 0])
        mm_121: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_397, view_440);  permute_397 = view_440 = None
        permute_220: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_286, [1, 0]);  primals_286 = None
        permute_399: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        mm_122: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_635, permute_399);  view_635 = permute_399 = None
        view_636: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_122, [32, 128, 2048]);  mm_122 = None
        add_245: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_244, view_636);  add_244 = view_636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_293: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_245, primals_284);  primals_284 = None
        mul_294: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_293, 2048)
        sum_77: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True)
        mul_295: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_293, mul_160);  mul_293 = None
        sum_78: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True);  mul_295 = None
        mul_296: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_160, sum_78);  sum_78 = None
        sub_107: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_294, sum_77);  mul_294 = sum_77 = None
        sub_108: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_107, mul_296);  sub_107 = mul_296 = None
        mul_297: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_34, sub_108);  div_34 = sub_108 = None
        mul_298: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_245, mul_160);  mul_160 = None
        sum_79: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 1]);  mul_298 = None
        sum_80: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
        add_246: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_243, mul_297);  add_243 = mul_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_637: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_246, [4096, 2048])
        permute_219: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
        permute_401: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        mm_123: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_637, permute_401);  permute_401 = None
        permute_402: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_637, [1, 0])
        mm_124: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_402, view_438);  permute_402 = view_438 = None
        sum_81: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_637, [0], True);  view_637 = None
        view_638: "f32[2048]" = torch.ops.aten.reshape.default(sum_81, [2048]);  sum_81 = None
        view_639: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_123, [32, 128, 8192]);  mm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_437: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_58, [32, 128, 8192]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_437, 0.5)
        mul_299: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_639, mul_156);  mul_156 = None
        pow_20: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_437, 3.0)
        mul_157: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_181: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_437, mul_157);  mul_157 = None
        mul_158: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_181, 0.7978845608028654);  add_181 = None
        tanh_19: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_182: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_300: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_639, add_182);  view_639 = add_182 = None
        mul_301: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_109: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_301);  mul_301 = None
        mul_302: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_299, sub_109);  mul_299 = sub_109 = None
        mul_303: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_302, 0.7978845608028654);  mul_302 = None
        mul_304: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_303, 0.044715)
        pow_29: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_437, 2.0);  view_437 = None
        mul_305: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
        mul_306: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
        add_247: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_303, mul_306);  mul_303 = mul_306 = None
        mul_307: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_300, 0.5);  mul_300 = None
        add_248: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_247, mul_307);  add_247 = mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_640: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_248, [4096, 8192]);  add_248 = None
        permute_218: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
        permute_405: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        mm_125: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_640, permute_405);  permute_405 = None
        permute_406: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_126: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_406, view_436);  permute_406 = view_436 = None
        sum_82: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_640, [0], True);  view_640 = None
        view_641: "f32[8192]" = torch.ops.aten.reshape.default(sum_82, [8192]);  sum_82 = None
        view_642: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_125, [32, 128, 2048]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_309: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_642, primals_278);  primals_278 = None
        mul_310: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_309, 2048)
        sum_83: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
        mul_311: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_309, mul_154);  mul_309 = None
        sum_84: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
        mul_312: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_154, sum_84);  sum_84 = None
        sub_111: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_310, sum_83);  mul_310 = sum_83 = None
        sub_112: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_111, mul_312);  sub_111 = mul_312 = None
        mul_313: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_35, sub_112);  div_35 = sub_112 = None
        mul_314: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_642, mul_154);  mul_154 = None
        sum_85: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
        sum_86: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_642, [0, 1]);  view_642 = None
        add_249: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_246, mul_313);  add_246 = mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_643: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_249, [4096, 2048])
        permute_217: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_276, [1, 0]);  primals_276 = None
        permute_409: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        mm_127: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_643, permute_409);  permute_409 = None
        permute_410: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_643, [1, 0])
        mm_128: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_410, view_434);  permute_410 = view_434 = None
        sum_87: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_643, [0], True);  view_643 = None
        view_644: "f32[2048]" = torch.ops.aten.reshape.default(sum_87, [2048]);  sum_87 = None
        view_645: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_127, [32, 128, 2048]);  mm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_646: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_645, [32, 128, 16, 128]);  view_645 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_413: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_646, [0, 2, 1, 3]);  view_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_186: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_647: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_186, [512, 128, 128]);  clone_186 = None
        expand_80: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_19, [32, 16, 128, 128])
        view_430: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_80, [512, 128, 128]);  expand_80 = None
        permute_414: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None
        bmm_64: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_414, view_647);  permute_414 = None
        bmm_65: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_647, permute_415);  view_647 = permute_415 = None
        view_648: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_64, [32, 16, 128, 128]);  bmm_64 = None
        view_649: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_65, [32, 16, 128, 128]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_315: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_649, div_19);  view_649 = None
        sum_88: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [-1], True)
        neg_5: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_4: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_5, sum_88, mul_315);  neg_5 = sum_88 = mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_42: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_275, 2, 0, 128);  primals_275 = None
        slice_43: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_42, 3, 0, 128);  slice_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_33: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_43, fma_4, full_default_1);  slice_43 = fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_650: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_33, [512, 128, 128]);  where_33 = None
        bmm_66: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_416, view_650);  permute_416 = None
        bmm_67: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_650, permute_417);  view_650 = permute_417 = None
        view_651: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_66, [32, 16, 128, 128]);  bmm_66 = None
        view_652: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_67, [32, 16, 128, 128]);  bmm_67 = None
        permute_418: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_651, [0, 1, 3, 2]);  view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_419: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_187: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_653: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_187, [32, 128, 2048]);  clone_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_420: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_418, [0, 2, 1, 3]);  permute_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_654: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_420, [32, 128, 2048]);  permute_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_421: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_188: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
        view_655: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_188, [32, 128, 2048]);  clone_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_656: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_653, [4096, 2048]);  view_653 = None
        permute_422: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_129: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_422, view_418);  permute_422 = None
        permute_211: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
        permute_424: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        mm_130: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_656, permute_424);  view_656 = permute_424 = None
        view_657: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_130, [32, 128, 2048]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_189: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_654, memory_format = torch.contiguous_format);  view_654 = None
        view_658: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_189, [4096, 2048]);  clone_189 = None
        permute_426: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_658, [1, 0])
        mm_131: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_426, view_418);  permute_426 = None
        permute_210: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
        permute_428: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
        mm_132: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_658, permute_428);  view_658 = permute_428 = None
        view_659: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_132, [32, 128, 2048]);  mm_132 = None
        add_250: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_657, view_659);  view_657 = view_659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_660: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_655, [4096, 2048]);  view_655 = None
        permute_430: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_660, [1, 0])
        mm_133: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_430, view_418);  permute_430 = view_418 = None
        permute_209: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_272, [1, 0]);  primals_272 = None
        permute_432: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        mm_134: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_660, permute_432);  view_660 = permute_432 = None
        view_661: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_134, [32, 128, 2048]);  mm_134 = None
        add_251: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_250, view_661);  add_250 = view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_317: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_251, primals_270);  primals_270 = None
        mul_318: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_317, 2048)
        sum_89: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True)
        mul_319: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_317, mul_152);  mul_317 = None
        sum_90: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True);  mul_319 = None
        mul_320: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, sum_90);  sum_90 = None
        sub_114: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_318, sum_89);  mul_318 = sum_89 = None
        sub_115: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_114, mul_320);  sub_114 = mul_320 = None
        mul_321: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_36, sub_115);  div_36 = sub_115 = None
        mul_322: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_251, mul_152);  mul_152 = None
        sum_91: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 1]);  mul_322 = None
        sum_92: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None
        add_252: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_249, mul_321);  add_249 = mul_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_662: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_252, [4096, 2048])
        permute_208: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_268, [1, 0]);  primals_268 = None
        permute_434: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        mm_135: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_662, permute_434);  permute_434 = None
        permute_435: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_662, [1, 0])
        mm_136: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_435, view_416);  permute_435 = view_416 = None
        sum_93: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_662, [0], True);  view_662 = None
        view_663: "f32[2048]" = torch.ops.aten.reshape.default(sum_93, [2048]);  sum_93 = None
        view_664: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_135, [32, 128, 8192]);  mm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_415: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_55, [32, 128, 8192]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_148: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        mul_323: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_664, mul_148);  mul_148 = None
        pow_19: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_415, 3.0)
        mul_149: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_172: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_415, mul_149);  mul_149 = None
        mul_150: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_172, 0.7978845608028654);  add_172 = None
        tanh_18: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
        add_173: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_324: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_664, add_173);  view_664 = add_173 = None
        mul_325: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_116: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_325);  mul_325 = None
        mul_326: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_323, sub_116);  mul_323 = sub_116 = None
        mul_327: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_326, 0.7978845608028654);  mul_326 = None
        mul_328: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_327, 0.044715)
        pow_30: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_415, 2.0);  view_415 = None
        mul_329: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_330: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_328, mul_329);  mul_328 = mul_329 = None
        add_253: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_327, mul_330);  mul_327 = mul_330 = None
        mul_331: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_324, 0.5);  mul_324 = None
        add_254: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_253, mul_331);  add_253 = mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_665: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_254, [4096, 8192]);  add_254 = None
        permute_207: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_266, [1, 0]);  primals_266 = None
        permute_438: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        mm_137: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_665, permute_438);  permute_438 = None
        permute_439: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_665, [1, 0])
        mm_138: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_439, view_414);  permute_439 = view_414 = None
        sum_94: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_665, [0], True);  view_665 = None
        view_666: "f32[8192]" = torch.ops.aten.reshape.default(sum_94, [8192]);  sum_94 = None
        view_667: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_137, [32, 128, 2048]);  mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_333: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_667, primals_264);  primals_264 = None
        mul_334: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_333, 2048)
        sum_95: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [2], True)
        mul_335: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_333, mul_146);  mul_333 = None
        sum_96: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True);  mul_335 = None
        mul_336: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_146, sum_96);  sum_96 = None
        sub_118: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_334, sum_95);  mul_334 = sum_95 = None
        sub_119: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_118, mul_336);  sub_118 = mul_336 = None
        mul_337: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_37, sub_119);  div_37 = sub_119 = None
        mul_338: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_667, mul_146);  mul_146 = None
        sum_97: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1]);  mul_338 = None
        sum_98: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_667, [0, 1]);  view_667 = None
        add_255: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_252, mul_337);  add_252 = mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_668: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_255, [4096, 2048])
        permute_206: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
        permute_442: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        mm_139: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_668, permute_442);  permute_442 = None
        permute_443: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_668, [1, 0])
        mm_140: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_443, view_412);  permute_443 = view_412 = None
        sum_99: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_668, [0], True);  view_668 = None
        view_669: "f32[2048]" = torch.ops.aten.reshape.default(sum_99, [2048]);  sum_99 = None
        view_670: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_139, [32, 128, 2048]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_671: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_670, [32, 128, 16, 128]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_446: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_190: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
        view_672: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_190, [512, 128, 128]);  clone_190 = None
        expand_76: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_18, [32, 16, 128, 128])
        view_408: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_76, [512, 128, 128]);  expand_76 = None
        permute_447: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None
        bmm_68: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_447, view_672);  permute_447 = None
        bmm_69: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_672, permute_448);  view_672 = permute_448 = None
        view_673: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_68, [32, 16, 128, 128]);  bmm_68 = None
        view_674: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_69, [32, 16, 128, 128]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_339: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_674, div_18);  view_674 = None
        sum_100: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [-1], True)
        neg_6: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_5: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_6, sum_100, mul_339);  neg_6 = sum_100 = mul_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_40: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_261, 2, 0, 128);  primals_261 = None
        slice_41: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_40, 3, 0, 128);  slice_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_34: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_41, fma_5, full_default_1);  slice_41 = fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_675: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_34, [512, 128, 128]);  where_34 = None
        bmm_70: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_449, view_675);  permute_449 = None
        bmm_71: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_675, permute_450);  view_675 = permute_450 = None
        view_676: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_70, [32, 16, 128, 128]);  bmm_70 = None
        view_677: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_71, [32, 16, 128, 128]);  bmm_71 = None
        permute_451: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_676, [0, 1, 3, 2]);  view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_452: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_191: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_678: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_191, [32, 128, 2048]);  clone_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_453: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_451, [0, 2, 1, 3]);  permute_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_679: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_453, [32, 128, 2048]);  permute_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_454: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_677, [0, 2, 1, 3]);  view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_192: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
        view_680: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_192, [32, 128, 2048]);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_681: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_678, [4096, 2048]);  view_678 = None
        permute_455: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_141: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_455, view_396);  permute_455 = None
        permute_200: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_260, [1, 0]);  primals_260 = None
        permute_457: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        mm_142: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_681, permute_457);  view_681 = permute_457 = None
        view_682: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_142, [32, 128, 2048]);  mm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_193: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_679, memory_format = torch.contiguous_format);  view_679 = None
        view_683: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_193, [4096, 2048]);  clone_193 = None
        permute_459: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_683, [1, 0])
        mm_143: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_459, view_396);  permute_459 = None
        permute_199: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
        permute_461: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        mm_144: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_683, permute_461);  view_683 = permute_461 = None
        view_684: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_144, [32, 128, 2048]);  mm_144 = None
        add_256: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_682, view_684);  view_682 = view_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_685: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_680, [4096, 2048]);  view_680 = None
        permute_463: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_685, [1, 0])
        mm_145: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_463, view_396);  permute_463 = view_396 = None
        permute_198: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_258, [1, 0]);  primals_258 = None
        permute_465: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        mm_146: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_685, permute_465);  view_685 = permute_465 = None
        view_686: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_146, [32, 128, 2048]);  mm_146 = None
        add_257: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_256, view_686);  add_256 = view_686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_341: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_257, primals_256);  primals_256 = None
        mul_342: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_341, 2048)
        sum_101: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
        mul_343: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_341, mul_144);  mul_341 = None
        sum_102: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
        mul_344: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_144, sum_102);  sum_102 = None
        sub_121: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_342, sum_101);  mul_342 = sum_101 = None
        sub_122: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_121, mul_344);  sub_121 = mul_344 = None
        mul_345: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_38, sub_122);  div_38 = sub_122 = None
        mul_346: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_257, mul_144);  mul_144 = None
        sum_103: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
        sum_104: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None
        add_258: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_255, mul_345);  add_255 = mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_687: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_258, [4096, 2048])
        permute_197: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
        permute_467: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        mm_147: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_687, permute_467);  permute_467 = None
        permute_468: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_687, [1, 0])
        mm_148: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_468, view_394);  permute_468 = view_394 = None
        sum_105: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
        view_688: "f32[2048]" = torch.ops.aten.reshape.default(sum_105, [2048]);  sum_105 = None
        view_689: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_147, [32, 128, 8192]);  mm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_393: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_52, [32, 128, 8192]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_393, 0.5)
        mul_347: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_689, mul_140);  mul_140 = None
        pow_18: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_393, 3.0)
        mul_141: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_163: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_393, mul_141);  mul_141 = None
        mul_142: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_163, 0.7978845608028654);  add_163 = None
        tanh_17: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_164: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_348: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_689, add_164);  view_689 = add_164 = None
        mul_349: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_123: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_349);  mul_349 = None
        mul_350: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_347, sub_123);  mul_347 = sub_123 = None
        mul_351: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_350, 0.7978845608028654);  mul_350 = None
        mul_352: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_351, 0.044715)
        pow_31: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_393, 2.0);  view_393 = None
        mul_353: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
        mul_354: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_352, mul_353);  mul_352 = mul_353 = None
        add_259: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_351, mul_354);  mul_351 = mul_354 = None
        mul_355: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_348, 0.5);  mul_348 = None
        add_260: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_259, mul_355);  add_259 = mul_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_690: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_260, [4096, 8192]);  add_260 = None
        permute_196: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
        permute_471: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        mm_149: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_690, permute_471);  permute_471 = None
        permute_472: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_690, [1, 0])
        mm_150: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_472, view_392);  permute_472 = view_392 = None
        sum_106: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_690, [0], True);  view_690 = None
        view_691: "f32[8192]" = torch.ops.aten.reshape.default(sum_106, [8192]);  sum_106 = None
        view_692: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_149, [32, 128, 2048]);  mm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_357: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_692, primals_250);  primals_250 = None
        mul_358: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_357, 2048)
        sum_107: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True)
        mul_359: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_357, mul_138);  mul_357 = None
        sum_108: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
        mul_360: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_138, sum_108);  sum_108 = None
        sub_125: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_358, sum_107);  mul_358 = sum_107 = None
        sub_126: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_125, mul_360);  sub_125 = mul_360 = None
        mul_361: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_39, sub_126);  div_39 = sub_126 = None
        mul_362: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_692, mul_138);  mul_138 = None
        sum_109: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1]);  mul_362 = None
        sum_110: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_692, [0, 1]);  view_692 = None
        add_261: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_258, mul_361);  add_258 = mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_693: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_261, [4096, 2048])
        permute_195: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
        permute_475: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        mm_151: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_693, permute_475);  permute_475 = None
        permute_476: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_693, [1, 0])
        mm_152: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_476, view_390);  permute_476 = view_390 = None
        sum_111: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_693, [0], True);  view_693 = None
        view_694: "f32[2048]" = torch.ops.aten.reshape.default(sum_111, [2048]);  sum_111 = None
        view_695: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_151, [32, 128, 2048]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_696: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_695, [32, 128, 16, 128]);  view_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_479: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_696, [0, 2, 1, 3]);  view_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_194: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        view_697: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_194, [512, 128, 128]);  clone_194 = None
        expand_72: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_17, [32, 16, 128, 128])
        view_386: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_72, [512, 128, 128]);  expand_72 = None
        permute_480: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
        bmm_72: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_480, view_697);  permute_480 = None
        bmm_73: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_697, permute_481);  view_697 = permute_481 = None
        view_698: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_72, [32, 16, 128, 128]);  bmm_72 = None
        view_699: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_73, [32, 16, 128, 128]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_363: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_699, div_17);  view_699 = None
        sum_112: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
        neg_7: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_6: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_7, sum_112, mul_363);  neg_7 = sum_112 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_38: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_247, 2, 0, 128);  primals_247 = None
        slice_39: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_38, 3, 0, 128);  slice_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_35: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_39, fma_6, full_default_1);  slice_39 = fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_700: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_35, [512, 128, 128]);  where_35 = None
        bmm_74: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_482, view_700);  permute_482 = None
        bmm_75: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_700, permute_483);  view_700 = permute_483 = None
        view_701: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_74, [32, 16, 128, 128]);  bmm_74 = None
        view_702: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_75, [32, 16, 128, 128]);  bmm_75 = None
        permute_484: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_701, [0, 1, 3, 2]);  view_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_485: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_195: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_703: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_195, [32, 128, 2048]);  clone_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_486: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_484, [0, 2, 1, 3]);  permute_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_704: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_486, [32, 128, 2048]);  permute_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_487: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_702, [0, 2, 1, 3]);  view_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_196: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
        view_705: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_196, [32, 128, 2048]);  clone_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_706: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_703, [4096, 2048]);  view_703 = None
        permute_488: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_706, [1, 0])
        mm_153: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_488, view_374);  permute_488 = None
        permute_189: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_246, [1, 0]);  primals_246 = None
        permute_490: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        mm_154: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_706, permute_490);  view_706 = permute_490 = None
        view_707: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_154, [32, 128, 2048]);  mm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_197: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_704, memory_format = torch.contiguous_format);  view_704 = None
        view_708: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_197, [4096, 2048]);  clone_197 = None
        permute_492: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_708, [1, 0])
        mm_155: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_492, view_374);  permute_492 = None
        permute_188: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
        permute_494: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        mm_156: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_708, permute_494);  view_708 = permute_494 = None
        view_709: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_156, [32, 128, 2048]);  mm_156 = None
        add_262: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_707, view_709);  view_707 = view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_710: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_705, [4096, 2048]);  view_705 = None
        permute_496: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_710, [1, 0])
        mm_157: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_496, view_374);  permute_496 = view_374 = None
        permute_187: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_244, [1, 0]);  primals_244 = None
        permute_498: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        mm_158: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_710, permute_498);  view_710 = permute_498 = None
        view_711: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_158, [32, 128, 2048]);  mm_158 = None
        add_263: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_262, view_711);  add_262 = view_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_365: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_263, primals_242);  primals_242 = None
        mul_366: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_365, 2048)
        sum_113: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True)
        mul_367: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_365, mul_136);  mul_365 = None
        sum_114: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True);  mul_367 = None
        mul_368: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_136, sum_114);  sum_114 = None
        sub_128: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_366, sum_113);  mul_366 = sum_113 = None
        sub_129: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_128, mul_368);  sub_128 = mul_368 = None
        mul_369: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_40, sub_129);  div_40 = sub_129 = None
        mul_370: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_263, mul_136);  mul_136 = None
        sum_115: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1]);  mul_370 = None
        sum_116: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None
        add_264: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_261, mul_369);  add_261 = mul_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_712: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_264, [4096, 2048])
        permute_186: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
        permute_500: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        mm_159: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_712, permute_500);  permute_500 = None
        permute_501: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_160: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_501, view_372);  permute_501 = view_372 = None
        sum_117: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
        view_713: "f32[2048]" = torch.ops.aten.reshape.default(sum_117, [2048]);  sum_117 = None
        view_714: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_159, [32, 128, 8192]);  mm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_371: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_49, [32, 128, 8192]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_132: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        mul_371: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_714, mul_132);  mul_132 = None
        pow_17: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_371, 3.0)
        mul_133: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_154: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_371, mul_133);  mul_133 = None
        mul_134: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_154, 0.7978845608028654);  add_154 = None
        tanh_16: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
        add_155: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_372: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_714, add_155);  view_714 = add_155 = None
        mul_373: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_130: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_373);  mul_373 = None
        mul_374: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_371, sub_130);  mul_371 = sub_130 = None
        mul_375: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_374, 0.7978845608028654);  mul_374 = None
        mul_376: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_375, 0.044715)
        pow_32: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_371, 2.0);  view_371 = None
        mul_377: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_378: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
        add_265: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_375, mul_378);  mul_375 = mul_378 = None
        mul_379: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_372, 0.5);  mul_372 = None
        add_266: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_265, mul_379);  add_265 = mul_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_715: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_266, [4096, 8192]);  add_266 = None
        permute_185: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
        permute_504: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        mm_161: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_715, permute_504);  permute_504 = None
        permute_505: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_715, [1, 0])
        mm_162: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_505, view_370);  permute_505 = view_370 = None
        sum_118: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
        view_716: "f32[8192]" = torch.ops.aten.reshape.default(sum_118, [8192]);  sum_118 = None
        view_717: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_161, [32, 128, 2048]);  mm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_381: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_717, primals_236);  primals_236 = None
        mul_382: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_381, 2048)
        sum_119: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
        mul_383: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_381, mul_130);  mul_381 = None
        sum_120: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
        mul_384: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_130, sum_120);  sum_120 = None
        sub_132: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_382, sum_119);  mul_382 = sum_119 = None
        sub_133: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_132, mul_384);  sub_132 = mul_384 = None
        mul_385: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_41, sub_133);  div_41 = sub_133 = None
        mul_386: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_717, mul_130);  mul_130 = None
        sum_121: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
        sum_122: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_717, [0, 1]);  view_717 = None
        add_267: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_264, mul_385);  add_264 = mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_718: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_267, [4096, 2048])
        permute_184: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_234, [1, 0]);  primals_234 = None
        permute_508: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        mm_163: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_718, permute_508);  permute_508 = None
        permute_509: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_718, [1, 0])
        mm_164: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_509, view_368);  permute_509 = view_368 = None
        sum_123: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_718, [0], True);  view_718 = None
        view_719: "f32[2048]" = torch.ops.aten.reshape.default(sum_123, [2048]);  sum_123 = None
        view_720: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_163, [32, 128, 2048]);  mm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_721: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_720, [32, 128, 16, 128]);  view_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_512: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_721, [0, 2, 1, 3]);  view_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_198: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
        view_722: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_198, [512, 128, 128]);  clone_198 = None
        expand_68: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_16, [32, 16, 128, 128])
        view_364: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_68, [512, 128, 128]);  expand_68 = None
        permute_513: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_364, [0, 2, 1]);  view_364 = None
        bmm_76: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_513, view_722);  permute_513 = None
        bmm_77: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_722, permute_514);  view_722 = permute_514 = None
        view_723: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_76, [32, 16, 128, 128]);  bmm_76 = None
        view_724: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_77, [32, 16, 128, 128]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_387: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_724, div_16);  view_724 = None
        sum_124: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [-1], True)
        neg_8: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_7: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_8, sum_124, mul_387);  neg_8 = sum_124 = mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_36: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_233, 2, 0, 128);  primals_233 = None
        slice_37: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_36, 3, 0, 128);  slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_36: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_37, fma_7, full_default_1);  slice_37 = fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_725: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_36, [512, 128, 128]);  where_36 = None
        bmm_78: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_515, view_725);  permute_515 = None
        bmm_79: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_725, permute_516);  view_725 = permute_516 = None
        view_726: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_78, [32, 16, 128, 128]);  bmm_78 = None
        view_727: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_79, [32, 16, 128, 128]);  bmm_79 = None
        permute_517: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_726, [0, 1, 3, 2]);  view_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_518: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_199: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_728: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_199, [32, 128, 2048]);  clone_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_519: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_517, [0, 2, 1, 3]);  permute_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_729: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_519, [32, 128, 2048]);  permute_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_520: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_727, [0, 2, 1, 3]);  view_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_200: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
        view_730: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_200, [32, 128, 2048]);  clone_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_731: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_728, [4096, 2048]);  view_728 = None
        permute_521: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_731, [1, 0])
        mm_165: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_521, view_352);  permute_521 = None
        permute_178: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
        permute_523: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        mm_166: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_731, permute_523);  view_731 = permute_523 = None
        view_732: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_166, [32, 128, 2048]);  mm_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_201: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_729, memory_format = torch.contiguous_format);  view_729 = None
        view_733: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_201, [4096, 2048]);  clone_201 = None
        permute_525: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_733, [1, 0])
        mm_167: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_525, view_352);  permute_525 = None
        permute_177: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
        permute_527: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        mm_168: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_733, permute_527);  view_733 = permute_527 = None
        view_734: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_168, [32, 128, 2048]);  mm_168 = None
        add_268: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_732, view_734);  view_732 = view_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_735: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_730, [4096, 2048]);  view_730 = None
        permute_529: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_735, [1, 0])
        mm_169: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_529, view_352);  permute_529 = view_352 = None
        permute_176: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
        permute_531: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_170: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_735, permute_531);  view_735 = permute_531 = None
        view_736: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_170, [32, 128, 2048]);  mm_170 = None
        add_269: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_268, view_736);  add_268 = view_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_389: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_269, primals_228);  primals_228 = None
        mul_390: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_389, 2048)
        sum_125: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True)
        mul_391: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_389, mul_128);  mul_389 = None
        sum_126: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [2], True);  mul_391 = None
        mul_392: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_128, sum_126);  sum_126 = None
        sub_135: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_390, sum_125);  mul_390 = sum_125 = None
        sub_136: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_135, mul_392);  sub_135 = mul_392 = None
        mul_393: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_42, sub_136);  div_42 = sub_136 = None
        mul_394: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_269, mul_128);  mul_128 = None
        sum_127: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_394, [0, 1]);  mul_394 = None
        sum_128: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None
        add_270: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_267, mul_393);  add_267 = mul_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_737: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_270, [4096, 2048])
        permute_175: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
        permute_533: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        mm_171: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_737, permute_533);  permute_533 = None
        permute_534: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_737, [1, 0])
        mm_172: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_534, view_350);  permute_534 = view_350 = None
        sum_129: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
        view_738: "f32[2048]" = torch.ops.aten.reshape.default(sum_129, [2048]);  sum_129 = None
        view_739: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_171, [32, 128, 8192]);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_349: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_46, [32, 128, 8192]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_124: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_349, 0.5)
        mul_395: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_739, mul_124);  mul_124 = None
        pow_16: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_349, 3.0)
        mul_125: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_145: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_349, mul_125);  mul_125 = None
        mul_126: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_15: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_146: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_396: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_739, add_146);  view_739 = add_146 = None
        mul_397: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_137: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_397);  mul_397 = None
        mul_398: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_395, sub_137);  mul_395 = sub_137 = None
        mul_399: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_398, 0.7978845608028654);  mul_398 = None
        mul_400: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_399, 0.044715)
        pow_33: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_349, 2.0);  view_349 = None
        mul_401: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
        mul_402: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
        add_271: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_399, mul_402);  mul_399 = mul_402 = None
        mul_403: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_396, 0.5);  mul_396 = None
        add_272: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_271, mul_403);  add_271 = mul_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_740: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_272, [4096, 8192]);  add_272 = None
        permute_174: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_224, [1, 0]);  primals_224 = None
        permute_537: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_173: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_740, permute_537);  permute_537 = None
        permute_538: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_174: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_538, view_348);  permute_538 = view_348 = None
        sum_130: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
        view_741: "f32[8192]" = torch.ops.aten.reshape.default(sum_130, [8192]);  sum_130 = None
        view_742: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_173, [32, 128, 2048]);  mm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_405: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_742, primals_222);  primals_222 = None
        mul_406: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_405, 2048)
        sum_131: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [2], True)
        mul_407: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_405, mul_122);  mul_405 = None
        sum_132: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_407, [2], True);  mul_407 = None
        mul_408: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_122, sum_132);  sum_132 = None
        sub_139: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_406, sum_131);  mul_406 = sum_131 = None
        sub_140: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_139, mul_408);  sub_139 = mul_408 = None
        mul_409: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_43, sub_140);  div_43 = sub_140 = None
        mul_410: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_742, mul_122);  mul_122 = None
        sum_133: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_410, [0, 1]);  mul_410 = None
        sum_134: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_742, [0, 1]);  view_742 = None
        add_273: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_270, mul_409);  add_270 = mul_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_743: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_273, [4096, 2048])
        permute_173: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_220, [1, 0]);  primals_220 = None
        permute_541: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_175: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_743, permute_541);  permute_541 = None
        permute_542: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_176: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_542, view_346);  permute_542 = view_346 = None
        sum_135: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
        view_744: "f32[2048]" = torch.ops.aten.reshape.default(sum_135, [2048]);  sum_135 = None
        view_745: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_175, [32, 128, 2048]);  mm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_746: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_745, [32, 128, 16, 128]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_545: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_202: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_545, memory_format = torch.contiguous_format);  permute_545 = None
        view_747: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_202, [512, 128, 128]);  clone_202 = None
        expand_64: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_15, [32, 16, 128, 128])
        view_342: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_64, [512, 128, 128]);  expand_64 = None
        permute_546: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
        bmm_80: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_546, view_747);  permute_546 = None
        bmm_81: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_747, permute_547);  view_747 = permute_547 = None
        view_748: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_80, [32, 16, 128, 128]);  bmm_80 = None
        view_749: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_81, [32, 16, 128, 128]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_411: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_749, div_15);  view_749 = None
        sum_136: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [-1], True)
        neg_9: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_8: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_9, sum_136, mul_411);  neg_9 = sum_136 = mul_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_34: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_219, 2, 0, 128);  primals_219 = None
        slice_35: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_34, 3, 0, 128);  slice_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_37: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_35, fma_8, full_default_1);  slice_35 = fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_750: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_37, [512, 128, 128]);  where_37 = None
        bmm_82: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_548, view_750);  permute_548 = None
        bmm_83: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_750, permute_549);  view_750 = permute_549 = None
        view_751: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_82, [32, 16, 128, 128]);  bmm_82 = None
        view_752: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_83, [32, 16, 128, 128]);  bmm_83 = None
        permute_550: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_751, [0, 1, 3, 2]);  view_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_551: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_203: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        view_753: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_203, [32, 128, 2048]);  clone_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_552: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_550, [0, 2, 1, 3]);  permute_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_754: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_552, [32, 128, 2048]);  permute_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_553: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_752, [0, 2, 1, 3]);  view_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_204: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_553, memory_format = torch.contiguous_format);  permute_553 = None
        view_755: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_204, [32, 128, 2048]);  clone_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_756: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_753, [4096, 2048]);  view_753 = None
        permute_554: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_756, [1, 0])
        mm_177: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_554, view_330);  permute_554 = None
        permute_167: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
        permute_556: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_178: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_756, permute_556);  view_756 = permute_556 = None
        view_757: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_178, [32, 128, 2048]);  mm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_205: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_754, memory_format = torch.contiguous_format);  view_754 = None
        view_758: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_205, [4096, 2048]);  clone_205 = None
        permute_558: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_758, [1, 0])
        mm_179: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_558, view_330);  permute_558 = None
        permute_166: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
        permute_560: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        mm_180: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_758, permute_560);  view_758 = permute_560 = None
        view_759: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_180, [32, 128, 2048]);  mm_180 = None
        add_274: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_757, view_759);  view_757 = view_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_760: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_755, [4096, 2048]);  view_755 = None
        permute_562: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_760, [1, 0])
        mm_181: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_562, view_330);  permute_562 = view_330 = None
        permute_165: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
        permute_564: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_182: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_760, permute_564);  view_760 = permute_564 = None
        view_761: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_182, [32, 128, 2048]);  mm_182 = None
        add_275: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_274, view_761);  add_274 = view_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_413: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_275, primals_214);  primals_214 = None
        mul_414: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_413, 2048)
        sum_137: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
        mul_415: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_413, mul_120);  mul_413 = None
        sum_138: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
        mul_416: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, sum_138);  sum_138 = None
        sub_142: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_414, sum_137);  mul_414 = sum_137 = None
        sub_143: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_142, mul_416);  sub_142 = mul_416 = None
        mul_417: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_44, sub_143);  div_44 = sub_143 = None
        mul_418: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_275, mul_120);  mul_120 = None
        sum_139: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
        sum_140: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None
        add_276: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_273, mul_417);  add_273 = mul_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_762: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_276, [4096, 2048])
        permute_164: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
        permute_566: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        mm_183: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_762, permute_566);  permute_566 = None
        permute_567: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_184: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_567, view_328);  permute_567 = view_328 = None
        sum_141: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
        view_763: "f32[2048]" = torch.ops.aten.reshape.default(sum_141, [2048]);  sum_141 = None
        view_764: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_183, [32, 128, 8192]);  mm_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_327: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_43, [32, 128, 8192]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_327, 0.5)
        mul_419: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_764, mul_116);  mul_116 = None
        pow_15: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_327, 3.0)
        mul_117: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_136: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_327, mul_117);  mul_117 = None
        mul_118: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_136, 0.7978845608028654);  add_136 = None
        tanh_14: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_137: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_420: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_764, add_137);  view_764 = add_137 = None
        mul_421: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_144: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_421);  mul_421 = None
        mul_422: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_419, sub_144);  mul_419 = sub_144 = None
        mul_423: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_422, 0.7978845608028654);  mul_422 = None
        mul_424: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_423, 0.044715)
        pow_34: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_327, 2.0);  view_327 = None
        mul_425: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_426: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        add_277: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_423, mul_426);  mul_423 = mul_426 = None
        mul_427: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_420, 0.5);  mul_420 = None
        add_278: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_277, mul_427);  add_277 = mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_765: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_278, [4096, 8192]);  add_278 = None
        permute_163: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
        permute_570: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_185: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_765, permute_570);  permute_570 = None
        permute_571: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_765, [1, 0])
        mm_186: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_571, view_326);  permute_571 = view_326 = None
        sum_142: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
        view_766: "f32[8192]" = torch.ops.aten.reshape.default(sum_142, [8192]);  sum_142 = None
        view_767: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_185, [32, 128, 2048]);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_429: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_767, primals_208);  primals_208 = None
        mul_430: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_429, 2048)
        sum_143: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True)
        mul_431: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_429, mul_114);  mul_429 = None
        sum_144: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True);  mul_431 = None
        mul_432: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_114, sum_144);  sum_144 = None
        sub_146: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_430, sum_143);  mul_430 = sum_143 = None
        sub_147: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_146, mul_432);  sub_146 = mul_432 = None
        mul_433: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_45, sub_147);  div_45 = sub_147 = None
        mul_434: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_767, mul_114);  mul_114 = None
        sum_145: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 1]);  mul_434 = None
        sum_146: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_767, [0, 1]);  view_767 = None
        add_279: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_276, mul_433);  add_276 = mul_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_768: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_279, [4096, 2048])
        permute_162: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
        permute_574: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_187: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_768, permute_574);  permute_574 = None
        permute_575: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_188: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_575, view_324);  permute_575 = view_324 = None
        sum_147: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
        view_769: "f32[2048]" = torch.ops.aten.reshape.default(sum_147, [2048]);  sum_147 = None
        view_770: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_187, [32, 128, 2048]);  mm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_771: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_770, [32, 128, 16, 128]);  view_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_578: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_206: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_578, memory_format = torch.contiguous_format);  permute_578 = None
        view_772: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_206, [512, 128, 128]);  clone_206 = None
        expand_60: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_14, [32, 16, 128, 128])
        view_320: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_60, [512, 128, 128]);  expand_60 = None
        permute_579: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
        bmm_84: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_579, view_772);  permute_579 = None
        bmm_85: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_772, permute_580);  view_772 = permute_580 = None
        view_773: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_84, [32, 16, 128, 128]);  bmm_84 = None
        view_774: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_85, [32, 16, 128, 128]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_435: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_774, div_14);  view_774 = None
        sum_148: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [-1], True)
        neg_10: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_14);  div_14 = None
        fma_9: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_10, sum_148, mul_435);  neg_10 = sum_148 = mul_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_32: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_205, 2, 0, 128);  primals_205 = None
        slice_33: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_32, 3, 0, 128);  slice_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_38: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_33, fma_9, full_default_1);  slice_33 = fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_775: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_38, [512, 128, 128]);  where_38 = None
        bmm_86: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_581, view_775);  permute_581 = None
        bmm_87: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_775, permute_582);  view_775 = permute_582 = None
        view_776: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_86, [32, 16, 128, 128]);  bmm_86 = None
        view_777: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_87, [32, 16, 128, 128]);  bmm_87 = None
        permute_583: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_776, [0, 1, 3, 2]);  view_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_584: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_207: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_584, memory_format = torch.contiguous_format);  permute_584 = None
        view_778: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_207, [32, 128, 2048]);  clone_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_585: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_583, [0, 2, 1, 3]);  permute_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_779: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_585, [32, 128, 2048]);  permute_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_586: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_777, [0, 2, 1, 3]);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_208: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
        view_780: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_208, [32, 128, 2048]);  clone_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_781: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_778, [4096, 2048]);  view_778 = None
        permute_587: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_781, [1, 0])
        mm_189: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_587, view_308);  permute_587 = None
        permute_156: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
        permute_589: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_190: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_781, permute_589);  view_781 = permute_589 = None
        view_782: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_190, [32, 128, 2048]);  mm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_209: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_779, memory_format = torch.contiguous_format);  view_779 = None
        view_783: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_209, [4096, 2048]);  clone_209 = None
        permute_591: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_783, [1, 0])
        mm_191: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_591, view_308);  permute_591 = None
        permute_155: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
        permute_593: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        mm_192: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_783, permute_593);  view_783 = permute_593 = None
        view_784: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_192, [32, 128, 2048]);  mm_192 = None
        add_280: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_782, view_784);  view_782 = view_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_785: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_780, [4096, 2048]);  view_780 = None
        permute_595: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_785, [1, 0])
        mm_193: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_595, view_308);  permute_595 = view_308 = None
        permute_154: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
        permute_597: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_194: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_785, permute_597);  view_785 = permute_597 = None
        view_786: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_194, [32, 128, 2048]);  mm_194 = None
        add_281: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_280, view_786);  add_280 = view_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_437: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_281, primals_200);  primals_200 = None
        mul_438: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_437, 2048)
        sum_149: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True)
        mul_439: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_437, mul_112);  mul_437 = None
        sum_150: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True);  mul_439 = None
        mul_440: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_112, sum_150);  sum_150 = None
        sub_149: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_438, sum_149);  mul_438 = sum_149 = None
        sub_150: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_149, mul_440);  sub_149 = mul_440 = None
        mul_441: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_46, sub_150);  div_46 = sub_150 = None
        mul_442: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_281, mul_112);  mul_112 = None
        sum_151: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 1]);  mul_442 = None
        sum_152: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None
        add_282: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_279, mul_441);  add_279 = mul_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_787: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_282, [4096, 2048])
        permute_153: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
        permute_599: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_195: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_787, permute_599);  permute_599 = None
        permute_600: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_787, [1, 0])
        mm_196: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_600, view_306);  permute_600 = view_306 = None
        sum_153: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_787, [0], True);  view_787 = None
        view_788: "f32[2048]" = torch.ops.aten.reshape.default(sum_153, [2048]);  sum_153 = None
        view_789: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_195, [32, 128, 8192]);  mm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_305: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_40, [32, 128, 8192]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_108: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_443: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_789, mul_108);  mul_108 = None
        pow_14: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_305, 3.0)
        mul_109: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_127: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_305, mul_109);  mul_109 = None
        mul_110: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_127, 0.7978845608028654);  add_127 = None
        tanh_13: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
        add_128: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_444: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_789, add_128);  view_789 = add_128 = None
        mul_445: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_151: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_445);  mul_445 = None
        mul_446: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_443, sub_151);  mul_443 = sub_151 = None
        mul_447: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_446, 0.7978845608028654);  mul_446 = None
        mul_448: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_447, 0.044715)
        pow_35: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_305, 2.0);  view_305 = None
        mul_449: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
        mul_450: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_448, mul_449);  mul_448 = mul_449 = None
        add_283: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_447, mul_450);  mul_447 = mul_450 = None
        mul_451: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_444, 0.5);  mul_444 = None
        add_284: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_283, mul_451);  add_283 = mul_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_790: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_284, [4096, 8192]);  add_284 = None
        permute_152: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
        permute_603: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_197: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_790, permute_603);  permute_603 = None
        permute_604: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_198: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_604, view_304);  permute_604 = view_304 = None
        sum_154: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
        view_791: "f32[8192]" = torch.ops.aten.reshape.default(sum_154, [8192]);  sum_154 = None
        view_792: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_197, [32, 128, 2048]);  mm_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_453: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_792, primals_194);  primals_194 = None
        mul_454: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_453, 2048)
        sum_155: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
        mul_455: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_453, mul_106);  mul_453 = None
        sum_156: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
        mul_456: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_106, sum_156);  sum_156 = None
        sub_153: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_454, sum_155);  mul_454 = sum_155 = None
        sub_154: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_153, mul_456);  sub_153 = mul_456 = None
        mul_457: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_47, sub_154);  div_47 = sub_154 = None
        mul_458: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_792, mul_106);  mul_106 = None
        sum_157: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
        sum_158: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_792, [0, 1]);  view_792 = None
        add_285: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_282, mul_457);  add_282 = mul_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_793: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_285, [4096, 2048])
        permute_151: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        permute_607: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_199: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_793, permute_607);  permute_607 = None
        permute_608: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_793, [1, 0])
        mm_200: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_608, view_302);  permute_608 = view_302 = None
        sum_159: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
        view_794: "f32[2048]" = torch.ops.aten.reshape.default(sum_159, [2048]);  sum_159 = None
        view_795: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_199, [32, 128, 2048]);  mm_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_796: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_795, [32, 128, 16, 128]);  view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_611: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_796, [0, 2, 1, 3]);  view_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_210: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_611, memory_format = torch.contiguous_format);  permute_611 = None
        view_797: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_210, [512, 128, 128]);  clone_210 = None
        expand_56: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_13, [32, 16, 128, 128])
        view_298: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_56, [512, 128, 128]);  expand_56 = None
        permute_612: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_298, [0, 2, 1]);  view_298 = None
        bmm_88: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_612, view_797);  permute_612 = None
        bmm_89: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_797, permute_613);  view_797 = permute_613 = None
        view_798: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_88, [32, 16, 128, 128]);  bmm_88 = None
        view_799: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_89, [32, 16, 128, 128]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_459: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_799, div_13);  view_799 = None
        sum_160: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [-1], True)
        neg_11: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_10: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_11, sum_160, mul_459);  neg_11 = sum_160 = mul_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_30: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_191, 2, 0, 128);  primals_191 = None
        slice_31: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_30, 3, 0, 128);  slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_39: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_31, fma_10, full_default_1);  slice_31 = fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_800: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_39, [512, 128, 128]);  where_39 = None
        bmm_90: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_614, view_800);  permute_614 = None
        bmm_91: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_800, permute_615);  view_800 = permute_615 = None
        view_801: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_90, [32, 16, 128, 128]);  bmm_90 = None
        view_802: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_91, [32, 16, 128, 128]);  bmm_91 = None
        permute_616: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_801, [0, 1, 3, 2]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_617: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_798, [0, 2, 1, 3]);  view_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_211: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_617, memory_format = torch.contiguous_format);  permute_617 = None
        view_803: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_211, [32, 128, 2048]);  clone_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_618: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_616, [0, 2, 1, 3]);  permute_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_804: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_618, [32, 128, 2048]);  permute_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_619: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_212: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_619, memory_format = torch.contiguous_format);  permute_619 = None
        view_805: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_212, [32, 128, 2048]);  clone_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_806: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_803, [4096, 2048]);  view_803 = None
        permute_620: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_806, [1, 0])
        mm_201: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_620, view_286);  permute_620 = None
        permute_145: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_622: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_202: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_806, permute_622);  view_806 = permute_622 = None
        view_807: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_202, [32, 128, 2048]);  mm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_213: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_804, memory_format = torch.contiguous_format);  view_804 = None
        view_808: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_213, [4096, 2048]);  clone_213 = None
        permute_624: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_808, [1, 0])
        mm_203: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_624, view_286);  permute_624 = None
        permute_144: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        permute_626: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mm_204: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_808, permute_626);  view_808 = permute_626 = None
        view_809: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_204, [32, 128, 2048]);  mm_204 = None
        add_286: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_807, view_809);  view_807 = view_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_810: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_805, [4096, 2048]);  view_805 = None
        permute_628: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_810, [1, 0])
        mm_205: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_628, view_286);  permute_628 = view_286 = None
        permute_143: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        permute_630: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_206: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_810, permute_630);  view_810 = permute_630 = None
        view_811: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_206, [32, 128, 2048]);  mm_206 = None
        add_287: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_286, view_811);  add_286 = view_811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_461: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_287, primals_186);  primals_186 = None
        mul_462: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_461, 2048)
        sum_161: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True)
        mul_463: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_461, mul_104);  mul_461 = None
        sum_162: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_463, [2], True);  mul_463 = None
        mul_464: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_104, sum_162);  sum_162 = None
        sub_156: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_462, sum_161);  mul_462 = sum_161 = None
        sub_157: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_156, mul_464);  sub_156 = mul_464 = None
        mul_465: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_48, sub_157);  div_48 = sub_157 = None
        mul_466: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_287, mul_104);  mul_104 = None
        sum_163: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 1]);  mul_466 = None
        sum_164: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None
        add_288: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_285, mul_465);  add_285 = mul_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_812: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_288, [4096, 2048])
        permute_142: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_632: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_207: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_812, permute_632);  permute_632 = None
        permute_633: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_812, [1, 0])
        mm_208: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_633, view_284);  permute_633 = view_284 = None
        sum_165: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_812, [0], True);  view_812 = None
        view_813: "f32[2048]" = torch.ops.aten.reshape.default(sum_165, [2048]);  sum_165 = None
        view_814: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_207, [32, 128, 8192]);  mm_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_283: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_37, [32, 128, 8192]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_100: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_283, 0.5)
        mul_467: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_814, mul_100);  mul_100 = None
        pow_13: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_283, 3.0)
        mul_101: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_118: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_283, mul_101);  mul_101 = None
        mul_102: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_118, 0.7978845608028654);  add_118 = None
        tanh_12: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
        add_119: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_468: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_814, add_119);  view_814 = add_119 = None
        mul_469: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_158: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_469);  mul_469 = None
        mul_470: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_467, sub_158);  mul_467 = sub_158 = None
        mul_471: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_470, 0.7978845608028654);  mul_470 = None
        mul_472: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_471, 0.044715)
        pow_36: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_283, 2.0);  view_283 = None
        mul_473: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_474: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
        add_289: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_471, mul_474);  mul_471 = mul_474 = None
        mul_475: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_468, 0.5);  mul_468 = None
        add_290: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_289, mul_475);  add_289 = mul_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_815: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_290, [4096, 8192]);  add_290 = None
        permute_141: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        permute_636: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_209: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_815, permute_636);  permute_636 = None
        permute_637: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_815, [1, 0])
        mm_210: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_637, view_282);  permute_637 = view_282 = None
        sum_166: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_815, [0], True);  view_815 = None
        view_816: "f32[8192]" = torch.ops.aten.reshape.default(sum_166, [8192]);  sum_166 = None
        view_817: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_209, [32, 128, 2048]);  mm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_477: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_817, primals_180);  primals_180 = None
        mul_478: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_477, 2048)
        sum_167: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True)
        mul_479: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_477, mul_98);  mul_477 = None
        sum_168: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
        mul_480: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_98, sum_168);  sum_168 = None
        sub_160: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_478, sum_167);  mul_478 = sum_167 = None
        sub_161: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_160, mul_480);  sub_160 = mul_480 = None
        mul_481: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_49, sub_161);  div_49 = sub_161 = None
        mul_482: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_817, mul_98);  mul_98 = None
        sum_169: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1]);  mul_482 = None
        sum_170: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_817, [0, 1]);  view_817 = None
        add_291: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_288, mul_481);  add_288 = mul_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_818: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_291, [4096, 2048])
        permute_140: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        permute_640: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_211: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_818, permute_640);  permute_640 = None
        permute_641: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_212: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_641, view_280);  permute_641 = view_280 = None
        sum_171: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
        view_819: "f32[2048]" = torch.ops.aten.reshape.default(sum_171, [2048]);  sum_171 = None
        view_820: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_211, [32, 128, 2048]);  mm_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_821: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_820, [32, 128, 16, 128]);  view_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_644: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_821, [0, 2, 1, 3]);  view_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_214: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_644, memory_format = torch.contiguous_format);  permute_644 = None
        view_822: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_214, [512, 128, 128]);  clone_214 = None
        expand_52: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_12, [32, 16, 128, 128])
        view_276: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_52, [512, 128, 128]);  expand_52 = None
        permute_645: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_276, [0, 2, 1]);  view_276 = None
        bmm_92: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_645, view_822);  permute_645 = None
        bmm_93: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_822, permute_646);  view_822 = permute_646 = None
        view_823: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_92, [32, 16, 128, 128]);  bmm_92 = None
        view_824: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_93, [32, 16, 128, 128]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_483: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_824, div_12);  view_824 = None
        sum_172: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [-1], True)
        neg_12: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_11: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_12, sum_172, mul_483);  neg_12 = sum_172 = mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_28: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_177, 2, 0, 128);  primals_177 = None
        slice_29: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_28, 3, 0, 128);  slice_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_40: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_29, fma_11, full_default_1);  slice_29 = fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_825: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_40, [512, 128, 128]);  where_40 = None
        bmm_94: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_647, view_825);  permute_647 = None
        bmm_95: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_825, permute_648);  view_825 = permute_648 = None
        view_826: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_94, [32, 16, 128, 128]);  bmm_94 = None
        view_827: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_95, [32, 16, 128, 128]);  bmm_95 = None
        permute_649: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_826, [0, 1, 3, 2]);  view_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_650: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_823, [0, 2, 1, 3]);  view_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_215: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
        view_828: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_215, [32, 128, 2048]);  clone_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_651: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_649, [0, 2, 1, 3]);  permute_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_829: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_651, [32, 128, 2048]);  permute_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_652: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_827, [0, 2, 1, 3]);  view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_216: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_652, memory_format = torch.contiguous_format);  permute_652 = None
        view_830: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_216, [32, 128, 2048]);  clone_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_831: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_828, [4096, 2048]);  view_828 = None
        permute_653: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_831, [1, 0])
        mm_213: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_653, view_264);  permute_653 = None
        permute_134: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        permute_655: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_214: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_831, permute_655);  view_831 = permute_655 = None
        view_832: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_214, [32, 128, 2048]);  mm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_217: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_829, memory_format = torch.contiguous_format);  view_829 = None
        view_833: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_217, [4096, 2048]);  clone_217 = None
        permute_657: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_833, [1, 0])
        mm_215: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_657, view_264);  permute_657 = None
        permute_133: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        permute_659: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_216: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_833, permute_659);  view_833 = permute_659 = None
        view_834: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_216, [32, 128, 2048]);  mm_216 = None
        add_292: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_832, view_834);  view_832 = view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_835: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_830, [4096, 2048]);  view_830 = None
        permute_661: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_835, [1, 0])
        mm_217: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_661, view_264);  permute_661 = view_264 = None
        permute_132: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        permute_663: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_218: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_835, permute_663);  view_835 = permute_663 = None
        view_836: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_218, [32, 128, 2048]);  mm_218 = None
        add_293: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_292, view_836);  add_292 = view_836 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_485: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_293, primals_172);  primals_172 = None
        mul_486: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_485, 2048)
        sum_173: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True)
        mul_487: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_485, mul_96);  mul_485 = None
        sum_174: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True);  mul_487 = None
        mul_488: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_96, sum_174);  sum_174 = None
        sub_163: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_486, sum_173);  mul_486 = sum_173 = None
        sub_164: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_163, mul_488);  sub_163 = mul_488 = None
        mul_489: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_50, sub_164);  div_50 = sub_164 = None
        mul_490: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_293, mul_96);  mul_96 = None
        sum_175: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1]);  mul_490 = None
        sum_176: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None
        add_294: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_291, mul_489);  add_291 = mul_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_837: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_294, [4096, 2048])
        permute_131: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        permute_665: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_219: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_837, permute_665);  permute_665 = None
        permute_666: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_837, [1, 0])
        mm_220: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_666, view_262);  permute_666 = view_262 = None
        sum_177: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_837, [0], True);  view_837 = None
        view_838: "f32[2048]" = torch.ops.aten.reshape.default(sum_177, [2048]);  sum_177 = None
        view_839: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_219, [32, 128, 8192]);  mm_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_261: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_34, [32, 128, 8192]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_491: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_839, mul_92);  mul_92 = None
        pow_12: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_261, 3.0)
        mul_93: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_109: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_261, mul_93);  mul_93 = None
        mul_94: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_109, 0.7978845608028654);  add_109 = None
        tanh_11: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_110: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_492: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_839, add_110);  view_839 = add_110 = None
        mul_493: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_165: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_493);  mul_493 = None
        mul_494: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_491, sub_165);  mul_491 = sub_165 = None
        mul_495: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_494, 0.7978845608028654);  mul_494 = None
        mul_496: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_495, 0.044715)
        pow_37: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_261, 2.0);  view_261 = None
        mul_497: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
        mul_498: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_496, mul_497);  mul_496 = mul_497 = None
        add_295: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_495, mul_498);  mul_495 = mul_498 = None
        mul_499: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_492, 0.5);  mul_492 = None
        add_296: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_295, mul_499);  add_295 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_840: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_296, [4096, 8192]);  add_296 = None
        permute_130: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
        permute_669: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_221: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_840, permute_669);  permute_669 = None
        permute_670: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_840, [1, 0])
        mm_222: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_670, view_260);  permute_670 = view_260 = None
        sum_178: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_840, [0], True);  view_840 = None
        view_841: "f32[8192]" = torch.ops.aten.reshape.default(sum_178, [8192]);  sum_178 = None
        view_842: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_221, [32, 128, 2048]);  mm_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_501: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_842, primals_166);  primals_166 = None
        mul_502: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_501, 2048)
        sum_179: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [2], True)
        mul_503: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_501, mul_90);  mul_501 = None
        sum_180: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_503, [2], True);  mul_503 = None
        mul_504: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_90, sum_180);  sum_180 = None
        sub_167: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_502, sum_179);  mul_502 = sum_179 = None
        sub_168: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_167, mul_504);  sub_167 = mul_504 = None
        mul_505: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_51, sub_168);  div_51 = sub_168 = None
        mul_506: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_842, mul_90);  mul_90 = None
        sum_181: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_506, [0, 1]);  mul_506 = None
        sum_182: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_842, [0, 1]);  view_842 = None
        add_297: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_294, mul_505);  add_294 = mul_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_843: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_297, [4096, 2048])
        permute_129: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
        permute_673: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_223: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_843, permute_673);  permute_673 = None
        permute_674: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_843, [1, 0])
        mm_224: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_674, view_258);  permute_674 = view_258 = None
        sum_183: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_843, [0], True);  view_843 = None
        view_844: "f32[2048]" = torch.ops.aten.reshape.default(sum_183, [2048]);  sum_183 = None
        view_845: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_223, [32, 128, 2048]);  mm_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_846: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_845, [32, 128, 16, 128]);  view_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_677: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_218: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_677, memory_format = torch.contiguous_format);  permute_677 = None
        view_847: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_218, [512, 128, 128]);  clone_218 = None
        expand_48: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_11, [32, 16, 128, 128])
        view_254: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_48, [512, 128, 128]);  expand_48 = None
        permute_678: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        bmm_96: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_678, view_847);  permute_678 = None
        bmm_97: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_847, permute_679);  view_847 = permute_679 = None
        view_848: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_96, [32, 16, 128, 128]);  bmm_96 = None
        view_849: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_97, [32, 16, 128, 128]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_507: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_849, div_11);  view_849 = None
        sum_184: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [-1], True)
        neg_13: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_12: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_13, sum_184, mul_507);  neg_13 = sum_184 = mul_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_26: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_163, 2, 0, 128);  primals_163 = None
        slice_27: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_26, 3, 0, 128);  slice_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_41: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_27, fma_12, full_default_1);  slice_27 = fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_850: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_41, [512, 128, 128]);  where_41 = None
        bmm_98: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_680, view_850);  permute_680 = None
        bmm_99: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_850, permute_681);  view_850 = permute_681 = None
        view_851: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_98, [32, 16, 128, 128]);  bmm_98 = None
        view_852: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_99, [32, 16, 128, 128]);  bmm_99 = None
        permute_682: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_851, [0, 1, 3, 2]);  view_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_683: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_848, [0, 2, 1, 3]);  view_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_219: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_683, memory_format = torch.contiguous_format);  permute_683 = None
        view_853: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_219, [32, 128, 2048]);  clone_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_684: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_682, [0, 2, 1, 3]);  permute_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_854: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_684, [32, 128, 2048]);  permute_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_685: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_852, [0, 2, 1, 3]);  view_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_220: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_685, memory_format = torch.contiguous_format);  permute_685 = None
        view_855: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_220, [32, 128, 2048]);  clone_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_856: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_853, [4096, 2048]);  view_853 = None
        permute_686: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_856, [1, 0])
        mm_225: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_686, view_242);  permute_686 = None
        permute_123: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_688: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_226: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_856, permute_688);  view_856 = permute_688 = None
        view_857: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_226, [32, 128, 2048]);  mm_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_221: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_854, memory_format = torch.contiguous_format);  view_854 = None
        view_858: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_221, [4096, 2048]);  clone_221 = None
        permute_690: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_858, [1, 0])
        mm_227: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_690, view_242);  permute_690 = None
        permute_122: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
        permute_692: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_228: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_858, permute_692);  view_858 = permute_692 = None
        view_859: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_228, [32, 128, 2048]);  mm_228 = None
        add_298: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_857, view_859);  view_857 = view_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_860: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_855, [4096, 2048]);  view_855 = None
        permute_694: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_860, [1, 0])
        mm_229: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_694, view_242);  permute_694 = view_242 = None
        permute_121: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        permute_696: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_230: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_860, permute_696);  view_860 = permute_696 = None
        view_861: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_230, [32, 128, 2048]);  mm_230 = None
        add_299: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_298, view_861);  add_298 = view_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_509: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_299, primals_158);  primals_158 = None
        mul_510: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_509, 2048)
        sum_185: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True)
        mul_511: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_509, mul_88);  mul_509 = None
        sum_186: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [2], True);  mul_511 = None
        mul_512: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_88, sum_186);  sum_186 = None
        sub_170: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_510, sum_185);  mul_510 = sum_185 = None
        sub_171: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_170, mul_512);  sub_170 = mul_512 = None
        mul_513: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_52, sub_171);  div_52 = sub_171 = None
        mul_514: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_299, mul_88);  mul_88 = None
        sum_187: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_514, [0, 1]);  mul_514 = None
        sum_188: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None
        add_300: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_297, mul_513);  add_297 = mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_862: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_300, [4096, 2048])
        permute_120: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        permute_698: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_231: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_862, permute_698);  permute_698 = None
        permute_699: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_862, [1, 0])
        mm_232: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_699, view_240);  permute_699 = view_240 = None
        sum_189: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_862, [0], True);  view_862 = None
        view_863: "f32[2048]" = torch.ops.aten.reshape.default(sum_189, [2048]);  sum_189 = None
        view_864: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_231, [32, 128, 8192]);  mm_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_239: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_31, [32, 128, 8192]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_84: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_515: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_864, mul_84);  mul_84 = None
        pow_11: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_239, 3.0)
        mul_85: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_100: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_239, mul_85);  mul_85 = None
        mul_86: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_101: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_516: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_864, add_101);  view_864 = add_101 = None
        mul_517: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_172: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_517);  mul_517 = None
        mul_518: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_515, sub_172);  mul_515 = sub_172 = None
        mul_519: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_518, 0.7978845608028654);  mul_518 = None
        mul_520: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_519, 0.044715)
        pow_38: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_239, 2.0);  view_239 = None
        mul_521: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_522: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_520, mul_521);  mul_520 = mul_521 = None
        add_301: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_519, mul_522);  mul_519 = mul_522 = None
        mul_523: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_516, 0.5);  mul_516 = None
        add_302: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_301, mul_523);  add_301 = mul_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_865: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_302, [4096, 8192]);  add_302 = None
        permute_119: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
        permute_702: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_233: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_865, permute_702);  permute_702 = None
        permute_703: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_865, [1, 0])
        mm_234: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_703, view_238);  permute_703 = view_238 = None
        sum_190: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_865, [0], True);  view_865 = None
        view_866: "f32[8192]" = torch.ops.aten.reshape.default(sum_190, [8192]);  sum_190 = None
        view_867: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_233, [32, 128, 2048]);  mm_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_525: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_867, primals_152);  primals_152 = None
        mul_526: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_525, 2048)
        sum_191: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True)
        mul_527: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_525, mul_82);  mul_525 = None
        sum_192: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True);  mul_527 = None
        mul_528: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_82, sum_192);  sum_192 = None
        sub_174: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_526, sum_191);  mul_526 = sum_191 = None
        sub_175: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_174, mul_528);  sub_174 = mul_528 = None
        mul_529: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_53, sub_175);  div_53 = sub_175 = None
        mul_530: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_867, mul_82);  mul_82 = None
        sum_193: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 1]);  mul_530 = None
        sum_194: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_867, [0, 1]);  view_867 = None
        add_303: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_300, mul_529);  add_300 = mul_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_868: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_303, [4096, 2048])
        permute_118: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        permute_706: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_235: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_868, permute_706);  permute_706 = None
        permute_707: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_868, [1, 0])
        mm_236: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_707, view_236);  permute_707 = view_236 = None
        sum_195: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_868, [0], True);  view_868 = None
        view_869: "f32[2048]" = torch.ops.aten.reshape.default(sum_195, [2048]);  sum_195 = None
        view_870: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_235, [32, 128, 2048]);  mm_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_871: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_870, [32, 128, 16, 128]);  view_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_710: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_871, [0, 2, 1, 3]);  view_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_222: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_710, memory_format = torch.contiguous_format);  permute_710 = None
        view_872: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_222, [512, 128, 128]);  clone_222 = None
        expand_44: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_10, [32, 16, 128, 128])
        view_232: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_44, [512, 128, 128]);  expand_44 = None
        permute_711: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        bmm_100: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_711, view_872);  permute_711 = None
        bmm_101: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_872, permute_712);  view_872 = permute_712 = None
        view_873: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_100, [32, 16, 128, 128]);  bmm_100 = None
        view_874: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_101, [32, 16, 128, 128]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_531: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_874, div_10);  view_874 = None
        sum_196: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [-1], True)
        neg_14: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_13: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_14, sum_196, mul_531);  neg_14 = sum_196 = mul_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_24: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_149, 2, 0, 128);  primals_149 = None
        slice_25: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_24, 3, 0, 128);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_42: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_25, fma_13, full_default_1);  slice_25 = fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_875: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_42, [512, 128, 128]);  where_42 = None
        bmm_102: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_713, view_875);  permute_713 = None
        bmm_103: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_875, permute_714);  view_875 = permute_714 = None
        view_876: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_102, [32, 16, 128, 128]);  bmm_102 = None
        view_877: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_103, [32, 16, 128, 128]);  bmm_103 = None
        permute_715: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_876, [0, 1, 3, 2]);  view_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_716: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_873, [0, 2, 1, 3]);  view_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_223: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_716, memory_format = torch.contiguous_format);  permute_716 = None
        view_878: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_223, [32, 128, 2048]);  clone_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_717: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_715, [0, 2, 1, 3]);  permute_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_879: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_717, [32, 128, 2048]);  permute_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_718: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_877, [0, 2, 1, 3]);  view_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_224: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_718, memory_format = torch.contiguous_format);  permute_718 = None
        view_880: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_224, [32, 128, 2048]);  clone_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_881: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_878, [4096, 2048]);  view_878 = None
        permute_719: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_881, [1, 0])
        mm_237: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_719, view_220);  permute_719 = None
        permute_112: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        permute_721: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_238: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_881, permute_721);  view_881 = permute_721 = None
        view_882: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_238, [32, 128, 2048]);  mm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_225: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_879, memory_format = torch.contiguous_format);  view_879 = None
        view_883: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_225, [4096, 2048]);  clone_225 = None
        permute_723: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_239: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_723, view_220);  permute_723 = None
        permute_111: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        permute_725: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_240: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_883, permute_725);  view_883 = permute_725 = None
        view_884: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_240, [32, 128, 2048]);  mm_240 = None
        add_304: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_882, view_884);  view_882 = view_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_885: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_880, [4096, 2048]);  view_880 = None
        permute_727: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_885, [1, 0])
        mm_241: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_727, view_220);  permute_727 = view_220 = None
        permute_110: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        permute_729: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_242: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_885, permute_729);  view_885 = permute_729 = None
        view_886: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_242, [32, 128, 2048]);  mm_242 = None
        add_305: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_304, view_886);  add_304 = view_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_533: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_305, primals_144);  primals_144 = None
        mul_534: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_533, 2048)
        sum_197: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_533, [2], True)
        mul_535: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_533, mul_80);  mul_533 = None
        sum_198: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_535, [2], True);  mul_535 = None
        mul_536: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_80, sum_198);  sum_198 = None
        sub_177: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_534, sum_197);  mul_534 = sum_197 = None
        sub_178: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_177, mul_536);  sub_177 = mul_536 = None
        mul_537: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_54, sub_178);  div_54 = sub_178 = None
        mul_538: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_305, mul_80);  mul_80 = None
        sum_199: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_538, [0, 1]);  mul_538 = None
        sum_200: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None
        add_306: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_303, mul_537);  add_303 = mul_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_887: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_306, [4096, 2048])
        permute_109: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_731: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_243: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_887, permute_731);  permute_731 = None
        permute_732: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_887, [1, 0])
        mm_244: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_732, view_218);  permute_732 = view_218 = None
        sum_201: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_887, [0], True);  view_887 = None
        view_888: "f32[2048]" = torch.ops.aten.reshape.default(sum_201, [2048]);  sum_201 = None
        view_889: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_243, [32, 128, 8192]);  mm_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_217: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_28, [32, 128, 8192]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_539: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_889, mul_76);  mul_76 = None
        pow_10: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_217, 3.0)
        mul_77: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_91: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_217, mul_77);  mul_77 = None
        mul_78: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_91, 0.7978845608028654);  add_91 = None
        tanh_9: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_92: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_540: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_889, add_92);  view_889 = add_92 = None
        mul_541: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_179: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_541);  mul_541 = None
        mul_542: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_539, sub_179);  mul_539 = sub_179 = None
        mul_543: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_542, 0.7978845608028654);  mul_542 = None
        mul_544: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_543, 0.044715)
        pow_39: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_217, 2.0);  view_217 = None
        mul_545: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
        mul_546: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_544, mul_545);  mul_544 = mul_545 = None
        add_307: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_543, mul_546);  mul_543 = mul_546 = None
        mul_547: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_540, 0.5);  mul_540 = None
        add_308: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_307, mul_547);  add_307 = mul_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_890: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_308, [4096, 8192]);  add_308 = None
        permute_108: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
        permute_735: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_245: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_890, permute_735);  permute_735 = None
        permute_736: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_890, [1, 0])
        mm_246: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_736, view_216);  permute_736 = view_216 = None
        sum_202: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_890, [0], True);  view_890 = None
        view_891: "f32[8192]" = torch.ops.aten.reshape.default(sum_202, [8192]);  sum_202 = None
        view_892: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_245, [32, 128, 2048]);  mm_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_549: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_892, primals_138);  primals_138 = None
        mul_550: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_549, 2048)
        sum_203: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_549, [2], True)
        mul_551: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_549, mul_74);  mul_549 = None
        sum_204: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True);  mul_551 = None
        mul_552: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_74, sum_204);  sum_204 = None
        sub_181: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_550, sum_203);  mul_550 = sum_203 = None
        sub_182: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_181, mul_552);  sub_181 = mul_552 = None
        mul_553: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_55, sub_182);  div_55 = sub_182 = None
        mul_554: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_892, mul_74);  mul_74 = None
        sum_205: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 1]);  mul_554 = None
        sum_206: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_892, [0, 1]);  view_892 = None
        add_309: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_306, mul_553);  add_306 = mul_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_893: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_309, [4096, 2048])
        permute_107: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        permute_739: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_247: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_893, permute_739);  permute_739 = None
        permute_740: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_893, [1, 0])
        mm_248: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_740, view_214);  permute_740 = view_214 = None
        sum_207: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_893, [0], True);  view_893 = None
        view_894: "f32[2048]" = torch.ops.aten.reshape.default(sum_207, [2048]);  sum_207 = None
        view_895: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_247, [32, 128, 2048]);  mm_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_896: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_895, [32, 128, 16, 128]);  view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_743: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_896, [0, 2, 1, 3]);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_226: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_743, memory_format = torch.contiguous_format);  permute_743 = None
        view_897: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_226, [512, 128, 128]);  clone_226 = None
        expand_40: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_9, [32, 16, 128, 128])
        view_210: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_40, [512, 128, 128]);  expand_40 = None
        permute_744: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        bmm_104: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_744, view_897);  permute_744 = None
        bmm_105: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_897, permute_745);  view_897 = permute_745 = None
        view_898: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_104, [32, 16, 128, 128]);  bmm_104 = None
        view_899: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_105, [32, 16, 128, 128]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_555: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_899, div_9);  view_899 = None
        sum_208: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_555, [-1], True)
        neg_15: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_14: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_15, sum_208, mul_555);  neg_15 = sum_208 = mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_22: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_135, 2, 0, 128);  primals_135 = None
        slice_23: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_22, 3, 0, 128);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_43: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_23, fma_14, full_default_1);  slice_23 = fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_900: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_43, [512, 128, 128]);  where_43 = None
        bmm_106: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_746, view_900);  permute_746 = None
        bmm_107: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_900, permute_747);  view_900 = permute_747 = None
        view_901: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_106, [32, 16, 128, 128]);  bmm_106 = None
        view_902: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_107, [32, 16, 128, 128]);  bmm_107 = None
        permute_748: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_901, [0, 1, 3, 2]);  view_901 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_749: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_898, [0, 2, 1, 3]);  view_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_227: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_749, memory_format = torch.contiguous_format);  permute_749 = None
        view_903: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_227, [32, 128, 2048]);  clone_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_750: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_748, [0, 2, 1, 3]);  permute_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_904: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_750, [32, 128, 2048]);  permute_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_751: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_902, [0, 2, 1, 3]);  view_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_228: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_751, memory_format = torch.contiguous_format);  permute_751 = None
        view_905: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_228, [32, 128, 2048]);  clone_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_906: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_903, [4096, 2048]);  view_903 = None
        permute_752: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_906, [1, 0])
        mm_249: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_752, view_198);  permute_752 = None
        permute_101: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        permute_754: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_250: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_906, permute_754);  view_906 = permute_754 = None
        view_907: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_250, [32, 128, 2048]);  mm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_229: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_904, memory_format = torch.contiguous_format);  view_904 = None
        view_908: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_229, [4096, 2048]);  clone_229 = None
        permute_756: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_251: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_756, view_198);  permute_756 = None
        permute_100: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        permute_758: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_252: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_908, permute_758);  view_908 = permute_758 = None
        view_909: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_252, [32, 128, 2048]);  mm_252 = None
        add_310: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_907, view_909);  view_907 = view_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_910: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_905, [4096, 2048]);  view_905 = None
        permute_760: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_910, [1, 0])
        mm_253: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_760, view_198);  permute_760 = view_198 = None
        permute_99: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        permute_762: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_254: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_910, permute_762);  view_910 = permute_762 = None
        view_911: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_254, [32, 128, 2048]);  mm_254 = None
        add_311: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_310, view_911);  add_310 = view_911 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_557: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_311, primals_130);  primals_130 = None
        mul_558: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_557, 2048)
        sum_209: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True)
        mul_559: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_557, mul_72);  mul_557 = None
        sum_210: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_559, [2], True);  mul_559 = None
        mul_560: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_72, sum_210);  sum_210 = None
        sub_184: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_558, sum_209);  mul_558 = sum_209 = None
        sub_185: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_184, mul_560);  sub_184 = mul_560 = None
        mul_561: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_56, sub_185);  div_56 = sub_185 = None
        mul_562: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_311, mul_72);  mul_72 = None
        sum_211: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 1]);  mul_562 = None
        sum_212: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None
        add_312: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_309, mul_561);  add_309 = mul_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_912: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_312, [4096, 2048])
        permute_98: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_764: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_255: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_912, permute_764);  permute_764 = None
        permute_765: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_912, [1, 0])
        mm_256: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_765, view_196);  permute_765 = view_196 = None
        sum_213: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_912, [0], True);  view_912 = None
        view_913: "f32[2048]" = torch.ops.aten.reshape.default(sum_213, [2048]);  sum_213 = None
        view_914: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_255, [32, 128, 8192]);  mm_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_195: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_25, [32, 128, 8192]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_563: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_914, mul_68);  mul_68 = None
        pow_9: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_195, 3.0)
        mul_69: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_82: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_195, mul_69);  mul_69 = None
        mul_70: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
        tanh_8: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_83: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_564: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_914, add_83);  view_914 = add_83 = None
        mul_565: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_186: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_565);  mul_565 = None
        mul_566: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_563, sub_186);  mul_563 = sub_186 = None
        mul_567: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_566, 0.7978845608028654);  mul_566 = None
        mul_568: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_567, 0.044715)
        pow_40: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_195, 2.0);  view_195 = None
        mul_569: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_570: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
        add_313: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_567, mul_570);  mul_567 = mul_570 = None
        mul_571: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_564, 0.5);  mul_564 = None
        add_314: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_313, mul_571);  add_313 = mul_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_915: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_314, [4096, 8192]);  add_314 = None
        permute_97: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        permute_768: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_257: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_915, permute_768);  permute_768 = None
        permute_769: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_915, [1, 0])
        mm_258: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_769, view_194);  permute_769 = view_194 = None
        sum_214: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_915, [0], True);  view_915 = None
        view_916: "f32[8192]" = torch.ops.aten.reshape.default(sum_214, [8192]);  sum_214 = None
        view_917: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_257, [32, 128, 2048]);  mm_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_573: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_917, primals_124);  primals_124 = None
        mul_574: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_573, 2048)
        sum_215: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True)
        mul_575: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_573, mul_66);  mul_573 = None
        sum_216: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_575, [2], True);  mul_575 = None
        mul_576: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_66, sum_216);  sum_216 = None
        sub_188: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_574, sum_215);  mul_574 = sum_215 = None
        sub_189: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_188, mul_576);  sub_188 = mul_576 = None
        mul_577: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_57, sub_189);  div_57 = sub_189 = None
        mul_578: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_917, mul_66);  mul_66 = None
        sum_217: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 1]);  mul_578 = None
        sum_218: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_917, [0, 1]);  view_917 = None
        add_315: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_312, mul_577);  add_312 = mul_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_918: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_315, [4096, 2048])
        permute_96: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_772: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_259: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_918, permute_772);  permute_772 = None
        permute_773: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_918, [1, 0])
        mm_260: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_773, view_192);  permute_773 = view_192 = None
        sum_219: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_918, [0], True);  view_918 = None
        view_919: "f32[2048]" = torch.ops.aten.reshape.default(sum_219, [2048]);  sum_219 = None
        view_920: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_259, [32, 128, 2048]);  mm_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_921: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_920, [32, 128, 16, 128]);  view_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_776: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_921, [0, 2, 1, 3]);  view_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_230: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_776, memory_format = torch.contiguous_format);  permute_776 = None
        view_922: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_230, [512, 128, 128]);  clone_230 = None
        expand_36: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_8, [32, 16, 128, 128])
        view_188: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_36, [512, 128, 128]);  expand_36 = None
        permute_777: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
        bmm_108: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_777, view_922);  permute_777 = None
        bmm_109: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_922, permute_778);  view_922 = permute_778 = None
        view_923: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_108, [32, 16, 128, 128]);  bmm_108 = None
        view_924: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_109, [32, 16, 128, 128]);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_579: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_924, div_8);  view_924 = None
        sum_220: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [-1], True)
        neg_16: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_15: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_16, sum_220, mul_579);  neg_16 = sum_220 = mul_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_20: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_121, 2, 0, 128);  primals_121 = None
        slice_21: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_20, 3, 0, 128);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_44: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_21, fma_15, full_default_1);  slice_21 = fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_925: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_44, [512, 128, 128]);  where_44 = None
        bmm_110: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_779, view_925);  permute_779 = None
        bmm_111: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_925, permute_780);  view_925 = permute_780 = None
        view_926: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_110, [32, 16, 128, 128]);  bmm_110 = None
        view_927: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_111, [32, 16, 128, 128]);  bmm_111 = None
        permute_781: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_926, [0, 1, 3, 2]);  view_926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_782: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_923, [0, 2, 1, 3]);  view_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_231: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_782, memory_format = torch.contiguous_format);  permute_782 = None
        view_928: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_231, [32, 128, 2048]);  clone_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_783: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_781, [0, 2, 1, 3]);  permute_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_929: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_783, [32, 128, 2048]);  permute_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_784: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_927, [0, 2, 1, 3]);  view_927 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_232: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
        view_930: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_232, [32, 128, 2048]);  clone_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_931: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_928, [4096, 2048]);  view_928 = None
        permute_785: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_931, [1, 0])
        mm_261: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_785, view_176);  permute_785 = None
        permute_90: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_787: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_262: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_931, permute_787);  view_931 = permute_787 = None
        view_932: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_262, [32, 128, 2048]);  mm_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_233: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_929, memory_format = torch.contiguous_format);  view_929 = None
        view_933: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_233, [4096, 2048]);  clone_233 = None
        permute_789: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_933, [1, 0])
        mm_263: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_789, view_176);  permute_789 = None
        permute_89: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        permute_791: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_264: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_933, permute_791);  view_933 = permute_791 = None
        view_934: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_264, [32, 128, 2048]);  mm_264 = None
        add_316: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_932, view_934);  view_932 = view_934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_935: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_930, [4096, 2048]);  view_930 = None
        permute_793: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_935, [1, 0])
        mm_265: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_793, view_176);  permute_793 = view_176 = None
        permute_88: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        permute_795: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_266: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_935, permute_795);  view_935 = permute_795 = None
        view_936: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_266, [32, 128, 2048]);  mm_266 = None
        add_317: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_316, view_936);  add_316 = view_936 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_581: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_317, primals_116);  primals_116 = None
        mul_582: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_581, 2048)
        sum_221: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True)
        mul_583: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_581, mul_64);  mul_581 = None
        sum_222: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_583, [2], True);  mul_583 = None
        mul_584: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_64, sum_222);  sum_222 = None
        sub_191: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_582, sum_221);  mul_582 = sum_221 = None
        sub_192: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_191, mul_584);  sub_191 = mul_584 = None
        mul_585: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_58, sub_192);  div_58 = sub_192 = None
        mul_586: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_317, mul_64);  mul_64 = None
        sum_223: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 1]);  mul_586 = None
        sum_224: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None
        add_318: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_315, mul_585);  add_315 = mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_937: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_318, [4096, 2048])
        permute_87: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_797: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_267: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_937, permute_797);  permute_797 = None
        permute_798: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_937, [1, 0])
        mm_268: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_798, view_174);  permute_798 = view_174 = None
        sum_225: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_937, [0], True);  view_937 = None
        view_938: "f32[2048]" = torch.ops.aten.reshape.default(sum_225, [2048]);  sum_225 = None
        view_939: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_267, [32, 128, 8192]);  mm_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_173: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_22, [32, 128, 8192]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_60: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_587: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_939, mul_60);  mul_60 = None
        pow_8: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_173, 3.0)
        mul_61: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_73: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_173, mul_61);  mul_61 = None
        mul_62: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_7: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
        add_74: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_588: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_939, add_74);  view_939 = add_74 = None
        mul_589: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_193: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_589);  mul_589 = None
        mul_590: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_587, sub_193);  mul_587 = sub_193 = None
        mul_591: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_590, 0.7978845608028654);  mul_590 = None
        mul_592: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_591, 0.044715)
        pow_41: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_173, 2.0);  view_173 = None
        mul_593: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
        mul_594: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        add_319: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_591, mul_594);  mul_591 = mul_594 = None
        mul_595: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_588, 0.5);  mul_588 = None
        add_320: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_319, mul_595);  add_319 = mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_940: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_320, [4096, 8192]);  add_320 = None
        permute_86: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
        permute_801: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_269: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_940, permute_801);  permute_801 = None
        permute_802: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_940, [1, 0])
        mm_270: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_802, view_172);  permute_802 = view_172 = None
        sum_226: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_940, [0], True);  view_940 = None
        view_941: "f32[8192]" = torch.ops.aten.reshape.default(sum_226, [8192]);  sum_226 = None
        view_942: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_269, [32, 128, 2048]);  mm_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_597: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_942, primals_110);  primals_110 = None
        mul_598: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_597, 2048)
        sum_227: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True)
        mul_599: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_597, mul_58);  mul_597 = None
        sum_228: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True);  mul_599 = None
        mul_600: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_58, sum_228);  sum_228 = None
        sub_195: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_598, sum_227);  mul_598 = sum_227 = None
        sub_196: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_195, mul_600);  sub_195 = mul_600 = None
        mul_601: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_59, sub_196);  div_59 = sub_196 = None
        mul_602: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_942, mul_58);  mul_58 = None
        sum_229: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 1]);  mul_602 = None
        sum_230: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_942, [0, 1]);  view_942 = None
        add_321: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_318, mul_601);  add_318 = mul_601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_943: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_321, [4096, 2048])
        permute_85: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        permute_805: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_271: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_943, permute_805);  permute_805 = None
        permute_806: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_943, [1, 0])
        mm_272: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_806, view_170);  permute_806 = view_170 = None
        sum_231: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_943, [0], True);  view_943 = None
        view_944: "f32[2048]" = torch.ops.aten.reshape.default(sum_231, [2048]);  sum_231 = None
        view_945: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_271, [32, 128, 2048]);  mm_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_946: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_945, [32, 128, 16, 128]);  view_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_809: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_946, [0, 2, 1, 3]);  view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_234: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_809, memory_format = torch.contiguous_format);  permute_809 = None
        view_947: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_234, [512, 128, 128]);  clone_234 = None
        expand_32: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_7, [32, 16, 128, 128])
        view_166: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_32, [512, 128, 128]);  expand_32 = None
        permute_810: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
        bmm_112: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_810, view_947);  permute_810 = None
        bmm_113: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_947, permute_811);  view_947 = permute_811 = None
        view_948: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_112, [32, 16, 128, 128]);  bmm_112 = None
        view_949: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_113, [32, 16, 128, 128]);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_603: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_949, div_7);  view_949 = None
        sum_232: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_603, [-1], True)
        neg_17: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_16: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_17, sum_232, mul_603);  neg_17 = sum_232 = mul_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_18: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_107, 2, 0, 128);  primals_107 = None
        slice_19: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_18, 3, 0, 128);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_45: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_19, fma_16, full_default_1);  slice_19 = fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_950: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_45, [512, 128, 128]);  where_45 = None
        bmm_114: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_812, view_950);  permute_812 = None
        bmm_115: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_950, permute_813);  view_950 = permute_813 = None
        view_951: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_114, [32, 16, 128, 128]);  bmm_114 = None
        view_952: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_115, [32, 16, 128, 128]);  bmm_115 = None
        permute_814: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_951, [0, 1, 3, 2]);  view_951 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_815: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_948, [0, 2, 1, 3]);  view_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_235: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_815, memory_format = torch.contiguous_format);  permute_815 = None
        view_953: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_235, [32, 128, 2048]);  clone_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_816: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_814, [0, 2, 1, 3]);  permute_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_954: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_816, [32, 128, 2048]);  permute_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_817: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_952, [0, 2, 1, 3]);  view_952 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_236: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_817, memory_format = torch.contiguous_format);  permute_817 = None
        view_955: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_236, [32, 128, 2048]);  clone_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_956: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_953, [4096, 2048]);  view_953 = None
        permute_818: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_956, [1, 0])
        mm_273: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_818, view_154);  permute_818 = None
        permute_79: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_820: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_274: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_956, permute_820);  view_956 = permute_820 = None
        view_957: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_274, [32, 128, 2048]);  mm_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_237: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_954, memory_format = torch.contiguous_format);  view_954 = None
        view_958: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_237, [4096, 2048]);  clone_237 = None
        permute_822: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_958, [1, 0])
        mm_275: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_822, view_154);  permute_822 = None
        permute_78: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_824: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_276: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_958, permute_824);  view_958 = permute_824 = None
        view_959: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_276, [32, 128, 2048]);  mm_276 = None
        add_322: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_957, view_959);  view_957 = view_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_960: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_955, [4096, 2048]);  view_955 = None
        permute_826: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_960, [1, 0])
        mm_277: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_826, view_154);  permute_826 = view_154 = None
        permute_77: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_828: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_278: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_960, permute_828);  view_960 = permute_828 = None
        view_961: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_278, [32, 128, 2048]);  mm_278 = None
        add_323: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_322, view_961);  add_322 = view_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_605: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_323, primals_102);  primals_102 = None
        mul_606: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_605, 2048)
        sum_233: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_605, [2], True)
        mul_607: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_605, mul_56);  mul_605 = None
        sum_234: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_607, [2], True);  mul_607 = None
        mul_608: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_56, sum_234);  sum_234 = None
        sub_198: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_606, sum_233);  mul_606 = sum_233 = None
        sub_199: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_198, mul_608);  sub_198 = mul_608 = None
        mul_609: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_60, sub_199);  div_60 = sub_199 = None
        mul_610: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_323, mul_56);  mul_56 = None
        sum_235: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 1]);  mul_610 = None
        sum_236: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_323, [0, 1]);  add_323 = None
        add_324: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_321, mul_609);  add_321 = mul_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_962: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_324, [4096, 2048])
        permute_76: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_830: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_279: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_962, permute_830);  permute_830 = None
        permute_831: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_962, [1, 0])
        mm_280: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_831, view_152);  permute_831 = view_152 = None
        sum_237: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_962, [0], True);  view_962 = None
        view_963: "f32[2048]" = torch.ops.aten.reshape.default(sum_237, [2048]);  sum_237 = None
        view_964: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_279, [32, 128, 8192]);  mm_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_151: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_19, [32, 128, 8192]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_52: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_611: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_964, mul_52);  mul_52 = None
        pow_7: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_151, 3.0)
        mul_53: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_64: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_151, mul_53);  mul_53 = None
        mul_54: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_64, 0.7978845608028654);  add_64 = None
        tanh_6: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
        add_65: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_612: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_964, add_65);  view_964 = add_65 = None
        mul_613: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_200: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_613);  mul_613 = None
        mul_614: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_611, sub_200);  mul_611 = sub_200 = None
        mul_615: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_614, 0.7978845608028654);  mul_614 = None
        mul_616: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_615, 0.044715)
        pow_42: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_151, 2.0);  view_151 = None
        mul_617: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_618: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_616, mul_617);  mul_616 = mul_617 = None
        add_325: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_615, mul_618);  mul_615 = mul_618 = None
        mul_619: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_612, 0.5);  mul_612 = None
        add_326: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_325, mul_619);  add_325 = mul_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_965: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_326, [4096, 8192]);  add_326 = None
        permute_75: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        permute_834: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_281: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_965, permute_834);  permute_834 = None
        permute_835: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_965, [1, 0])
        mm_282: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_835, view_150);  permute_835 = view_150 = None
        sum_238: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
        view_966: "f32[8192]" = torch.ops.aten.reshape.default(sum_238, [8192]);  sum_238 = None
        view_967: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_281, [32, 128, 2048]);  mm_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_621: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_967, primals_96);  primals_96 = None
        mul_622: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_621, 2048)
        sum_239: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True)
        mul_623: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_621, mul_50);  mul_621 = None
        sum_240: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True);  mul_623 = None
        mul_624: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_50, sum_240);  sum_240 = None
        sub_202: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_622, sum_239);  mul_622 = sum_239 = None
        sub_203: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_202, mul_624);  sub_202 = mul_624 = None
        mul_625: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_61, sub_203);  div_61 = sub_203 = None
        mul_626: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_967, mul_50);  mul_50 = None
        sum_241: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 1]);  mul_626 = None
        sum_242: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_967, [0, 1]);  view_967 = None
        add_327: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_324, mul_625);  add_324 = mul_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_968: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_327, [4096, 2048])
        permute_74: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_838: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_283: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_968, permute_838);  permute_838 = None
        permute_839: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_968, [1, 0])
        mm_284: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_839, view_148);  permute_839 = view_148 = None
        sum_243: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_968, [0], True);  view_968 = None
        view_969: "f32[2048]" = torch.ops.aten.reshape.default(sum_243, [2048]);  sum_243 = None
        view_970: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_283, [32, 128, 2048]);  mm_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_971: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_970, [32, 128, 16, 128]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_842: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_971, [0, 2, 1, 3]);  view_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_238: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_842, memory_format = torch.contiguous_format);  permute_842 = None
        view_972: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_238, [512, 128, 128]);  clone_238 = None
        expand_28: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_6, [32, 16, 128, 128])
        view_144: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_28, [512, 128, 128]);  expand_28 = None
        permute_843: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        bmm_116: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_843, view_972);  permute_843 = None
        bmm_117: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_972, permute_844);  view_972 = permute_844 = None
        view_973: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_116, [32, 16, 128, 128]);  bmm_116 = None
        view_974: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_117, [32, 16, 128, 128]);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_627: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_974, div_6);  view_974 = None
        sum_244: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_627, [-1], True)
        neg_18: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_17: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_18, sum_244, mul_627);  neg_18 = sum_244 = mul_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_16: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_93, 2, 0, 128);  primals_93 = None
        slice_17: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_16, 3, 0, 128);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_46: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_17, fma_17, full_default_1);  slice_17 = fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_975: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_46, [512, 128, 128]);  where_46 = None
        bmm_118: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_845, view_975);  permute_845 = None
        bmm_119: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_975, permute_846);  view_975 = permute_846 = None
        view_976: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_118, [32, 16, 128, 128]);  bmm_118 = None
        view_977: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_119, [32, 16, 128, 128]);  bmm_119 = None
        permute_847: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_976, [0, 1, 3, 2]);  view_976 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_848: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_973, [0, 2, 1, 3]);  view_973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_239: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_848, memory_format = torch.contiguous_format);  permute_848 = None
        view_978: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_239, [32, 128, 2048]);  clone_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_849: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_847, [0, 2, 1, 3]);  permute_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_979: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_849, [32, 128, 2048]);  permute_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_850: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_977, [0, 2, 1, 3]);  view_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_240: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_850, memory_format = torch.contiguous_format);  permute_850 = None
        view_980: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_240, [32, 128, 2048]);  clone_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_981: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_978, [4096, 2048]);  view_978 = None
        permute_851: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_981, [1, 0])
        mm_285: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_851, view_132);  permute_851 = None
        permute_68: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        permute_853: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_286: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_981, permute_853);  view_981 = permute_853 = None
        view_982: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_286, [32, 128, 2048]);  mm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_241: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_979, memory_format = torch.contiguous_format);  view_979 = None
        view_983: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_241, [4096, 2048]);  clone_241 = None
        permute_855: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_983, [1, 0])
        mm_287: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_855, view_132);  permute_855 = None
        permute_67: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_857: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_288: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_983, permute_857);  view_983 = permute_857 = None
        view_984: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_288, [32, 128, 2048]);  mm_288 = None
        add_328: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_982, view_984);  view_982 = view_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_985: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_980, [4096, 2048]);  view_980 = None
        permute_859: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_985, [1, 0])
        mm_289: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_859, view_132);  permute_859 = view_132 = None
        permute_66: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_861: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_290: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_985, permute_861);  view_985 = permute_861 = None
        view_986: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_290, [32, 128, 2048]);  mm_290 = None
        add_329: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_328, view_986);  add_328 = view_986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_629: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_329, primals_88);  primals_88 = None
        mul_630: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_629, 2048)
        sum_245: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_629, [2], True)
        mul_631: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_629, mul_48);  mul_629 = None
        sum_246: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_631, [2], True);  mul_631 = None
        mul_632: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_48, sum_246);  sum_246 = None
        sub_205: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_630, sum_245);  mul_630 = sum_245 = None
        sub_206: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_205, mul_632);  sub_205 = mul_632 = None
        mul_633: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_62, sub_206);  div_62 = sub_206 = None
        mul_634: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_329, mul_48);  mul_48 = None
        sum_247: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 1]);  mul_634 = None
        sum_248: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_329, [0, 1]);  add_329 = None
        add_330: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_327, mul_633);  add_327 = mul_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_987: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_330, [4096, 2048])
        permute_65: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_863: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_291: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_987, permute_863);  permute_863 = None
        permute_864: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_987, [1, 0])
        mm_292: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_864, view_130);  permute_864 = view_130 = None
        sum_249: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_987, [0], True);  view_987 = None
        view_988: "f32[2048]" = torch.ops.aten.reshape.default(sum_249, [2048]);  sum_249 = None
        view_989: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_291, [32, 128, 8192]);  mm_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_129: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_16, [32, 128, 8192]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_635: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_989, mul_44);  mul_44 = None
        pow_6: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_129, 3.0)
        mul_45: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_55: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_129, mul_45);  mul_45 = None
        mul_46: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_55, 0.7978845608028654);  add_55 = None
        tanh_5: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_56: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_636: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_989, add_56);  view_989 = add_56 = None
        mul_637: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_207: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_637);  mul_637 = None
        mul_638: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_635, sub_207);  mul_635 = sub_207 = None
        mul_639: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_638, 0.7978845608028654);  mul_638 = None
        mul_640: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_639, 0.044715)
        pow_43: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_129, 2.0);  view_129 = None
        mul_641: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
        mul_642: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_640, mul_641);  mul_640 = mul_641 = None
        add_331: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_639, mul_642);  mul_639 = mul_642 = None
        mul_643: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_636, 0.5);  mul_636 = None
        add_332: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_331, mul_643);  add_331 = mul_643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_990: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_332, [4096, 8192]);  add_332 = None
        permute_64: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        permute_867: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_293: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_990, permute_867);  permute_867 = None
        permute_868: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_990, [1, 0])
        mm_294: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_868, view_128);  permute_868 = view_128 = None
        sum_250: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_990, [0], True);  view_990 = None
        view_991: "f32[8192]" = torch.ops.aten.reshape.default(sum_250, [8192]);  sum_250 = None
        view_992: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_293, [32, 128, 2048]);  mm_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_645: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_992, primals_82);  primals_82 = None
        mul_646: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_645, 2048)
        sum_251: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_645, [2], True)
        mul_647: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_645, mul_42);  mul_645 = None
        sum_252: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [2], True);  mul_647 = None
        mul_648: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_42, sum_252);  sum_252 = None
        sub_209: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_646, sum_251);  mul_646 = sum_251 = None
        sub_210: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_209, mul_648);  sub_209 = mul_648 = None
        mul_649: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_63, sub_210);  div_63 = sub_210 = None
        mul_650: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_992, mul_42);  mul_42 = None
        sum_253: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 1]);  mul_650 = None
        sum_254: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_992, [0, 1]);  view_992 = None
        add_333: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_330, mul_649);  add_330 = mul_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_993: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_333, [4096, 2048])
        permute_63: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_871: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_295: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_993, permute_871);  permute_871 = None
        permute_872: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_993, [1, 0])
        mm_296: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_872, view_126);  permute_872 = view_126 = None
        sum_255: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_993, [0], True);  view_993 = None
        view_994: "f32[2048]" = torch.ops.aten.reshape.default(sum_255, [2048]);  sum_255 = None
        view_995: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_295, [32, 128, 2048]);  mm_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_996: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_995, [32, 128, 16, 128]);  view_995 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_875: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_996, [0, 2, 1, 3]);  view_996 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_242: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_875, memory_format = torch.contiguous_format);  permute_875 = None
        view_997: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_242, [512, 128, 128]);  clone_242 = None
        expand_24: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_5, [32, 16, 128, 128])
        view_122: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_24, [512, 128, 128]);  expand_24 = None
        permute_876: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
        bmm_120: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_876, view_997);  permute_876 = None
        bmm_121: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_997, permute_877);  view_997 = permute_877 = None
        view_998: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_120, [32, 16, 128, 128]);  bmm_120 = None
        view_999: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_121, [32, 16, 128, 128]);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_651: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_999, div_5);  view_999 = None
        sum_256: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_651, [-1], True)
        neg_19: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_18: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_19, sum_256, mul_651);  neg_19 = sum_256 = mul_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_14: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_79, 2, 0, 128);  primals_79 = None
        slice_15: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_14, 3, 0, 128);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_47: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_15, fma_18, full_default_1);  slice_15 = fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1000: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_47, [512, 128, 128]);  where_47 = None
        bmm_122: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_878, view_1000);  permute_878 = None
        bmm_123: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1000, permute_879);  view_1000 = permute_879 = None
        view_1001: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_122, [32, 16, 128, 128]);  bmm_122 = None
        view_1002: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_123, [32, 16, 128, 128]);  bmm_123 = None
        permute_880: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1001, [0, 1, 3, 2]);  view_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_881: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_243: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
        view_1003: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_243, [32, 128, 2048]);  clone_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_882: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_880, [0, 2, 1, 3]);  permute_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1004: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_882, [32, 128, 2048]);  permute_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_883: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1002, [0, 2, 1, 3]);  view_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_244: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_883, memory_format = torch.contiguous_format);  permute_883 = None
        view_1005: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_244, [32, 128, 2048]);  clone_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1006: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1003, [4096, 2048]);  view_1003 = None
        permute_884: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1006, [1, 0])
        mm_297: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_884, view_110);  permute_884 = None
        permute_57: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_886: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_298: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1006, permute_886);  view_1006 = permute_886 = None
        view_1007: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_298, [32, 128, 2048]);  mm_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_245: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_1004, memory_format = torch.contiguous_format);  view_1004 = None
        view_1008: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_245, [4096, 2048]);  clone_245 = None
        permute_888: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1008, [1, 0])
        mm_299: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_888, view_110);  permute_888 = None
        permute_56: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_890: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_300: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1008, permute_890);  view_1008 = permute_890 = None
        view_1009: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_300, [32, 128, 2048]);  mm_300 = None
        add_334: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_1007, view_1009);  view_1007 = view_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1010: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1005, [4096, 2048]);  view_1005 = None
        permute_892: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1010, [1, 0])
        mm_301: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_892, view_110);  permute_892 = view_110 = None
        permute_55: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_894: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_302: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1010, permute_894);  view_1010 = permute_894 = None
        view_1011: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_302, [32, 128, 2048]);  mm_302 = None
        add_335: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_334, view_1011);  add_334 = view_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_653: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_335, primals_74);  primals_74 = None
        mul_654: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_653, 2048)
        sum_257: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_653, [2], True)
        mul_655: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_653, mul_40);  mul_653 = None
        sum_258: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_655, [2], True);  mul_655 = None
        mul_656: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_40, sum_258);  sum_258 = None
        sub_212: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_654, sum_257);  mul_654 = sum_257 = None
        sub_213: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_212, mul_656);  sub_212 = mul_656 = None
        mul_657: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_64, sub_213);  div_64 = sub_213 = None
        mul_658: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_335, mul_40);  mul_40 = None
        sum_259: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 1]);  mul_658 = None
        sum_260: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 1]);  add_335 = None
        add_336: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_333, mul_657);  add_333 = mul_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1012: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_336, [4096, 2048])
        permute_54: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_896: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_303: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_1012, permute_896);  permute_896 = None
        permute_897: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1012, [1, 0])
        mm_304: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_897, view_108);  permute_897 = view_108 = None
        sum_261: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True);  view_1012 = None
        view_1013: "f32[2048]" = torch.ops.aten.reshape.default(sum_261, [2048]);  sum_261 = None
        view_1014: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_303, [32, 128, 8192]);  mm_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_107: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_13, [32, 128, 8192]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_659: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1014, mul_36);  mul_36 = None
        pow_5: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 3.0)
        mul_37: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_46: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_107, mul_37);  mul_37 = None
        mul_38: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_4: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_47: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_660: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1014, add_47);  view_1014 = add_47 = None
        mul_661: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_214: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_661);  mul_661 = None
        mul_662: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_659, sub_214);  mul_659 = sub_214 = None
        mul_663: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_662, 0.7978845608028654);  mul_662 = None
        mul_664: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_663, 0.044715)
        pow_44: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_107, 2.0);  view_107 = None
        mul_665: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_666: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
        add_337: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_663, mul_666);  mul_663 = mul_666 = None
        mul_667: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_660, 0.5);  mul_660 = None
        add_338: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_337, mul_667);  add_337 = mul_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1015: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_338, [4096, 8192]);  add_338 = None
        permute_53: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_900: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_305: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1015, permute_900);  permute_900 = None
        permute_901: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1015, [1, 0])
        mm_306: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_901, view_106);  permute_901 = view_106 = None
        sum_262: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1015, [0], True);  view_1015 = None
        view_1016: "f32[8192]" = torch.ops.aten.reshape.default(sum_262, [8192]);  sum_262 = None
        view_1017: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_305, [32, 128, 2048]);  mm_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_669: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1017, primals_68);  primals_68 = None
        mul_670: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_669, 2048)
        sum_263: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_669, [2], True)
        mul_671: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_669, mul_34);  mul_669 = None
        sum_264: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True);  mul_671 = None
        mul_672: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_34, sum_264);  sum_264 = None
        sub_216: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_670, sum_263);  mul_670 = sum_263 = None
        sub_217: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_216, mul_672);  sub_216 = mul_672 = None
        mul_673: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_65, sub_217);  div_65 = sub_217 = None
        mul_674: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1017, mul_34);  mul_34 = None
        sum_265: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_674, [0, 1]);  mul_674 = None
        sum_266: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1017, [0, 1]);  view_1017 = None
        add_339: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_336, mul_673);  add_336 = mul_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1018: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_339, [4096, 2048])
        permute_52: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_904: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_307: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1018, permute_904);  permute_904 = None
        permute_905: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1018, [1, 0])
        mm_308: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_905, view_104);  permute_905 = view_104 = None
        sum_267: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1018, [0], True);  view_1018 = None
        view_1019: "f32[2048]" = torch.ops.aten.reshape.default(sum_267, [2048]);  sum_267 = None
        view_1020: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_307, [32, 128, 2048]);  mm_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1021: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_1020, [32, 128, 16, 128]);  view_1020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_908: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1021, [0, 2, 1, 3]);  view_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_246: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_908, memory_format = torch.contiguous_format);  permute_908 = None
        view_1022: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_246, [512, 128, 128]);  clone_246 = None
        expand_20: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_4, [32, 16, 128, 128])
        view_100: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_20, [512, 128, 128]);  expand_20 = None
        permute_909: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
        bmm_124: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_909, view_1022);  permute_909 = None
        bmm_125: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1022, permute_910);  view_1022 = permute_910 = None
        view_1023: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_124, [32, 16, 128, 128]);  bmm_124 = None
        view_1024: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_125, [32, 16, 128, 128]);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_675: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1024, div_4);  view_1024 = None
        sum_268: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_675, [-1], True)
        neg_20: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_19: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_20, sum_268, mul_675);  neg_20 = sum_268 = mul_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_12: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_65, 2, 0, 128);  primals_65 = None
        slice_13: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 128);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_48: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_13, fma_19, full_default_1);  slice_13 = fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1025: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_48, [512, 128, 128]);  where_48 = None
        bmm_126: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_911, view_1025);  permute_911 = None
        bmm_127: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1025, permute_912);  view_1025 = permute_912 = None
        view_1026: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_126, [32, 16, 128, 128]);  bmm_126 = None
        view_1027: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_127, [32, 16, 128, 128]);  bmm_127 = None
        permute_913: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1026, [0, 1, 3, 2]);  view_1026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_914: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1023, [0, 2, 1, 3]);  view_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_247: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_914, memory_format = torch.contiguous_format);  permute_914 = None
        view_1028: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_247, [32, 128, 2048]);  clone_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_915: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_913, [0, 2, 1, 3]);  permute_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1029: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_915, [32, 128, 2048]);  permute_915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_916: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1027, [0, 2, 1, 3]);  view_1027 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_248: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_916, memory_format = torch.contiguous_format);  permute_916 = None
        view_1030: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_248, [32, 128, 2048]);  clone_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1031: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1028, [4096, 2048]);  view_1028 = None
        permute_917: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1031, [1, 0])
        mm_309: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_917, view_88);  permute_917 = None
        permute_46: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_919: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_310: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1031, permute_919);  view_1031 = permute_919 = None
        view_1032: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_310, [32, 128, 2048]);  mm_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_249: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_1029, memory_format = torch.contiguous_format);  view_1029 = None
        view_1033: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_249, [4096, 2048]);  clone_249 = None
        permute_921: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1033, [1, 0])
        mm_311: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_921, view_88);  permute_921 = None
        permute_45: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_923: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_312: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1033, permute_923);  view_1033 = permute_923 = None
        view_1034: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_312, [32, 128, 2048]);  mm_312 = None
        add_340: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_1032, view_1034);  view_1032 = view_1034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1035: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1030, [4096, 2048]);  view_1030 = None
        permute_925: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1035, [1, 0])
        mm_313: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_925, view_88);  permute_925 = view_88 = None
        permute_44: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_927: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_314: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1035, permute_927);  view_1035 = permute_927 = None
        view_1036: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_314, [32, 128, 2048]);  mm_314 = None
        add_341: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_340, view_1036);  add_340 = view_1036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_677: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_341, primals_60);  primals_60 = None
        mul_678: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_677, 2048)
        sum_269: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [2], True)
        mul_679: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_677, mul_32);  mul_677 = None
        sum_270: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_679, [2], True);  mul_679 = None
        mul_680: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_32, sum_270);  sum_270 = None
        sub_219: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_678, sum_269);  mul_678 = sum_269 = None
        sub_220: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_219, mul_680);  sub_219 = mul_680 = None
        mul_681: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_66, sub_220);  div_66 = sub_220 = None
        mul_682: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_341, mul_32);  mul_32 = None
        sum_271: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 1]);  mul_682 = None
        sum_272: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_341, [0, 1]);  add_341 = None
        add_342: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_339, mul_681);  add_339 = mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1037: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_342, [4096, 2048])
        permute_43: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
        permute_929: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_315: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_1037, permute_929);  permute_929 = None
        permute_930: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1037, [1, 0])
        mm_316: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_930, view_86);  permute_930 = view_86 = None
        sum_273: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
        view_1038: "f32[2048]" = torch.ops.aten.reshape.default(sum_273, [2048]);  sum_273 = None
        view_1039: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_315, [32, 128, 8192]);  mm_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_85: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_10, [32, 128, 8192]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_28: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_683: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1039, mul_28);  mul_28 = None
        pow_4: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_85, 3.0)
        mul_29: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_37: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_85, mul_29);  mul_29 = None
        mul_30: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_37, 0.7978845608028654);  add_37 = None
        tanh_3: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
        add_38: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_684: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1039, add_38);  view_1039 = add_38 = None
        mul_685: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_221: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_685);  mul_685 = None
        mul_686: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_683, sub_221);  mul_683 = sub_221 = None
        mul_687: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_686, 0.7978845608028654);  mul_686 = None
        mul_688: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_687, 0.044715)
        pow_45: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_85, 2.0);  view_85 = None
        mul_689: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
        mul_690: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_688, mul_689);  mul_688 = mul_689 = None
        add_343: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_687, mul_690);  mul_687 = mul_690 = None
        mul_691: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_684, 0.5);  mul_684 = None
        add_344: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_343, mul_691);  add_343 = mul_691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1040: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_344, [4096, 8192]);  add_344 = None
        permute_42: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_933: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_317: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1040, permute_933);  permute_933 = None
        permute_934: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1040, [1, 0])
        mm_318: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_934, view_84);  permute_934 = view_84 = None
        sum_274: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1040, [0], True);  view_1040 = None
        view_1041: "f32[8192]" = torch.ops.aten.reshape.default(sum_274, [8192]);  sum_274 = None
        view_1042: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_317, [32, 128, 2048]);  mm_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_693: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1042, primals_54);  primals_54 = None
        mul_694: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_693, 2048)
        sum_275: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_693, [2], True)
        mul_695: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_693, mul_26);  mul_693 = None
        sum_276: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_695, [2], True);  mul_695 = None
        mul_696: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_26, sum_276);  sum_276 = None
        sub_223: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_694, sum_275);  mul_694 = sum_275 = None
        sub_224: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_223, mul_696);  sub_223 = mul_696 = None
        mul_697: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_67, sub_224);  div_67 = sub_224 = None
        mul_698: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1042, mul_26);  mul_26 = None
        sum_277: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 1]);  mul_698 = None
        sum_278: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1042, [0, 1]);  view_1042 = None
        add_345: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_342, mul_697);  add_342 = mul_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1043: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_345, [4096, 2048])
        permute_41: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_937: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_319: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1043, permute_937);  permute_937 = None
        permute_938: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1043, [1, 0])
        mm_320: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_938, view_82);  permute_938 = view_82 = None
        sum_279: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1043, [0], True);  view_1043 = None
        view_1044: "f32[2048]" = torch.ops.aten.reshape.default(sum_279, [2048]);  sum_279 = None
        view_1045: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_319, [32, 128, 2048]);  mm_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1046: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_1045, [32, 128, 16, 128]);  view_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_941: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1046, [0, 2, 1, 3]);  view_1046 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_250: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_941, memory_format = torch.contiguous_format);  permute_941 = None
        view_1047: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_250, [512, 128, 128]);  clone_250 = None
        expand_16: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_3, [32, 16, 128, 128])
        view_78: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_16, [512, 128, 128]);  expand_16 = None
        permute_942: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
        bmm_128: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_942, view_1047);  permute_942 = None
        bmm_129: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1047, permute_943);  view_1047 = permute_943 = None
        view_1048: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_128, [32, 16, 128, 128]);  bmm_128 = None
        view_1049: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_129, [32, 16, 128, 128]);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_699: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1049, div_3);  view_1049 = None
        sum_280: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_699, [-1], True)
        neg_21: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_20: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_21, sum_280, mul_699);  neg_21 = sum_280 = mul_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_10: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_51, 2, 0, 128);  primals_51 = None
        slice_11: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_10, 3, 0, 128);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_49: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_11, fma_20, full_default_1);  slice_11 = fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1050: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_49, [512, 128, 128]);  where_49 = None
        bmm_130: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_944, view_1050);  permute_944 = None
        bmm_131: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1050, permute_945);  view_1050 = permute_945 = None
        view_1051: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_130, [32, 16, 128, 128]);  bmm_130 = None
        view_1052: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_131, [32, 16, 128, 128]);  bmm_131 = None
        permute_946: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1051, [0, 1, 3, 2]);  view_1051 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_947: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1048, [0, 2, 1, 3]);  view_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_251: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_947, memory_format = torch.contiguous_format);  permute_947 = None
        view_1053: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_251, [32, 128, 2048]);  clone_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_948: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_946, [0, 2, 1, 3]);  permute_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1054: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_948, [32, 128, 2048]);  permute_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_949: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1052, [0, 2, 1, 3]);  view_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_252: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_949, memory_format = torch.contiguous_format);  permute_949 = None
        view_1055: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_252, [32, 128, 2048]);  clone_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1056: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1053, [4096, 2048]);  view_1053 = None
        permute_950: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1056, [1, 0])
        mm_321: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_950, view_66);  permute_950 = None
        permute_35: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_952: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_322: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1056, permute_952);  view_1056 = permute_952 = None
        view_1057: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_322, [32, 128, 2048]);  mm_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_253: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_1054, memory_format = torch.contiguous_format);  view_1054 = None
        view_1058: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_253, [4096, 2048]);  clone_253 = None
        permute_954: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1058, [1, 0])
        mm_323: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_954, view_66);  permute_954 = None
        permute_34: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        permute_956: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_324: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1058, permute_956);  view_1058 = permute_956 = None
        view_1059: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_324, [32, 128, 2048]);  mm_324 = None
        add_346: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_1057, view_1059);  view_1057 = view_1059 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1060: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1055, [4096, 2048]);  view_1055 = None
        permute_958: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1060, [1, 0])
        mm_325: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_958, view_66);  permute_958 = view_66 = None
        permute_33: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_960: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_326: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1060, permute_960);  view_1060 = permute_960 = None
        view_1061: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_326, [32, 128, 2048]);  mm_326 = None
        add_347: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_346, view_1061);  add_346 = view_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_701: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_347, primals_46);  primals_46 = None
        mul_702: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_701, 2048)
        sum_281: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_701, [2], True)
        mul_703: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_701, mul_24);  mul_701 = None
        sum_282: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_703, [2], True);  mul_703 = None
        mul_704: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_24, sum_282);  sum_282 = None
        sub_226: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_702, sum_281);  mul_702 = sum_281 = None
        sub_227: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_226, mul_704);  sub_226 = mul_704 = None
        mul_705: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_68, sub_227);  div_68 = sub_227 = None
        mul_706: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_347, mul_24);  mul_24 = None
        sum_283: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 1]);  mul_706 = None
        sum_284: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_347, [0, 1]);  add_347 = None
        add_348: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_345, mul_705);  add_345 = mul_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1062: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_348, [4096, 2048])
        permute_32: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_962: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_327: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_1062, permute_962);  permute_962 = None
        permute_963: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1062, [1, 0])
        mm_328: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_963, view_64);  permute_963 = view_64 = None
        sum_285: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True);  view_1062 = None
        view_1063: "f32[2048]" = torch.ops.aten.reshape.default(sum_285, [2048]);  sum_285 = None
        view_1064: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_327, [32, 128, 8192]);  mm_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_63: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_7, [32, 128, 8192]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_707: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1064, mul_20);  mul_20 = None
        pow_3: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_63, 3.0)
        mul_21: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_28: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_63, mul_21);  mul_21 = None
        mul_22: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
        tanh_2: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_29: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_708: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1064, add_29);  view_1064 = add_29 = None
        mul_709: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_228: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_709);  mul_709 = None
        mul_710: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_707, sub_228);  mul_707 = sub_228 = None
        mul_711: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_710, 0.7978845608028654);  mul_710 = None
        mul_712: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_711, 0.044715)
        pow_46: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_63, 2.0);  view_63 = None
        mul_713: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_714: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_712, mul_713);  mul_712 = mul_713 = None
        add_349: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_711, mul_714);  mul_711 = mul_714 = None
        mul_715: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_708, 0.5);  mul_708 = None
        add_350: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_349, mul_715);  add_349 = mul_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1065: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_350, [4096, 8192]);  add_350 = None
        permute_31: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_966: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_329: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1065, permute_966);  permute_966 = None
        permute_967: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1065, [1, 0])
        mm_330: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_967, view_62);  permute_967 = view_62 = None
        sum_286: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1065, [0], True);  view_1065 = None
        view_1066: "f32[8192]" = torch.ops.aten.reshape.default(sum_286, [8192]);  sum_286 = None
        view_1067: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_329, [32, 128, 2048]);  mm_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_717: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1067, primals_40);  primals_40 = None
        mul_718: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_717, 2048)
        sum_287: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_717, [2], True)
        mul_719: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_717, mul_18);  mul_717 = None
        sum_288: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_719, [2], True);  mul_719 = None
        mul_720: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_18, sum_288);  sum_288 = None
        sub_230: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_718, sum_287);  mul_718 = sum_287 = None
        sub_231: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_230, mul_720);  sub_230 = mul_720 = None
        mul_721: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_69, sub_231);  div_69 = sub_231 = None
        mul_722: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1067, mul_18);  mul_18 = None
        sum_289: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_722, [0, 1]);  mul_722 = None
        sum_290: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1067, [0, 1]);  view_1067 = None
        add_351: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_348, mul_721);  add_348 = mul_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1068: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_351, [4096, 2048])
        permute_30: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_970: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_331: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1068, permute_970);  permute_970 = None
        permute_971: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1068, [1, 0])
        mm_332: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_971, view_60);  permute_971 = view_60 = None
        sum_291: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1068, [0], True);  view_1068 = None
        view_1069: "f32[2048]" = torch.ops.aten.reshape.default(sum_291, [2048]);  sum_291 = None
        view_1070: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_331, [32, 128, 2048]);  mm_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1071: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_1070, [32, 128, 16, 128]);  view_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_974: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1071, [0, 2, 1, 3]);  view_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_254: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_974, memory_format = torch.contiguous_format);  permute_974 = None
        view_1072: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_254, [512, 128, 128]);  clone_254 = None
        expand_12: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_2, [32, 16, 128, 128])
        view_56: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_12, [512, 128, 128]);  expand_12 = None
        permute_975: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
        bmm_132: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_975, view_1072);  permute_975 = None
        bmm_133: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1072, permute_976);  view_1072 = permute_976 = None
        view_1073: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_132, [32, 16, 128, 128]);  bmm_132 = None
        view_1074: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_133, [32, 16, 128, 128]);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_723: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1074, div_2);  view_1074 = None
        sum_292: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_723, [-1], True)
        neg_22: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_21: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_22, sum_292, mul_723);  neg_22 = sum_292 = mul_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_8: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_37, 2, 0, 128);  primals_37 = None
        slice_9: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_8, 3, 0, 128);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_50: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_9, fma_21, full_default_1);  slice_9 = fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1075: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_50, [512, 128, 128]);  where_50 = None
        bmm_134: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_977, view_1075);  permute_977 = None
        bmm_135: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1075, permute_978);  view_1075 = permute_978 = None
        view_1076: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_134, [32, 16, 128, 128]);  bmm_134 = None
        view_1077: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_135, [32, 16, 128, 128]);  bmm_135 = None
        permute_979: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1076, [0, 1, 3, 2]);  view_1076 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_980: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1073, [0, 2, 1, 3]);  view_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_255: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_980, memory_format = torch.contiguous_format);  permute_980 = None
        view_1078: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_255, [32, 128, 2048]);  clone_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_981: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_979, [0, 2, 1, 3]);  permute_979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1079: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_981, [32, 128, 2048]);  permute_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_982: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1077, [0, 2, 1, 3]);  view_1077 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_256: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_982, memory_format = torch.contiguous_format);  permute_982 = None
        view_1080: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_256, [32, 128, 2048]);  clone_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1081: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1078, [4096, 2048]);  view_1078 = None
        permute_983: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1081, [1, 0])
        mm_333: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_983, view_44);  permute_983 = None
        permute_24: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_36, [1, 0]);  primals_36 = None
        permute_985: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_334: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1081, permute_985);  view_1081 = permute_985 = None
        view_1082: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_334, [32, 128, 2048]);  mm_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_257: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_1079, memory_format = torch.contiguous_format);  view_1079 = None
        view_1083: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_257, [4096, 2048]);  clone_257 = None
        permute_987: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1083, [1, 0])
        mm_335: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_987, view_44);  permute_987 = None
        permute_23: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_989: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_336: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1083, permute_989);  view_1083 = permute_989 = None
        view_1084: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_336, [32, 128, 2048]);  mm_336 = None
        add_352: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_1082, view_1084);  view_1082 = view_1084 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1085: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1080, [4096, 2048]);  view_1080 = None
        permute_991: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1085, [1, 0])
        mm_337: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_991, view_44);  permute_991 = view_44 = None
        permute_22: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_993: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_338: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1085, permute_993);  view_1085 = permute_993 = None
        view_1086: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_338, [32, 128, 2048]);  mm_338 = None
        add_353: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_352, view_1086);  add_352 = view_1086 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_725: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_353, primals_32);  primals_32 = None
        mul_726: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_725, 2048)
        sum_293: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_725, [2], True)
        mul_727: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_725, mul_16);  mul_725 = None
        sum_294: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_727, [2], True);  mul_727 = None
        mul_728: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_16, sum_294);  sum_294 = None
        sub_233: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_726, sum_293);  mul_726 = sum_293 = None
        sub_234: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_233, mul_728);  sub_233 = mul_728 = None
        mul_729: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_70, sub_234);  div_70 = sub_234 = None
        mul_730: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_353, mul_16);  mul_16 = None
        sum_295: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 1]);  mul_730 = None
        sum_296: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_353, [0, 1]);  add_353 = None
        add_354: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_351, mul_729);  add_351 = mul_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1087: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_354, [4096, 2048])
        permute_21: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_995: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_339: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_1087, permute_995);  permute_995 = None
        permute_996: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1087, [1, 0])
        mm_340: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_996, view_42);  permute_996 = view_42 = None
        sum_297: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1087, [0], True);  view_1087 = None
        view_1088: "f32[2048]" = torch.ops.aten.reshape.default(sum_297, [2048]);  sum_297 = None
        view_1089: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_339, [32, 128, 8192]);  mm_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_41: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_4, [32, 128, 8192]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_731: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1089, mul_12);  mul_12 = None
        pow_2: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_41, 3.0)
        mul_13: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_19: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_41, mul_13);  mul_13 = None
        mul_14: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_19, 0.7978845608028654);  add_19 = None
        tanh_1: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_20: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_732: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1089, add_20);  view_1089 = add_20 = None
        mul_733: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_235: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_733);  mul_733 = None
        mul_734: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_731, sub_235);  mul_731 = sub_235 = None
        mul_735: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_734, 0.7978845608028654);  mul_734 = None
        mul_736: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_735, 0.044715)
        pow_47: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_41, 2.0);  view_41 = None
        mul_737: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
        mul_738: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
        add_355: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_735, mul_738);  mul_735 = mul_738 = None
        mul_739: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_732, 0.5);  mul_732 = None
        add_356: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_355, mul_739);  add_355 = mul_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1090: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_356, [4096, 8192]);  add_356 = None
        permute_20: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_999: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_341: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1090, permute_999);  permute_999 = None
        permute_1000: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1090, [1, 0])
        mm_342: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_1000, view_40);  permute_1000 = view_40 = None
        sum_298: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True);  view_1090 = None
        view_1091: "f32[8192]" = torch.ops.aten.reshape.default(sum_298, [8192]);  sum_298 = None
        view_1092: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_341, [32, 128, 2048]);  mm_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_741: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1092, primals_26);  primals_26 = None
        mul_742: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_741, 2048)
        sum_299: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_741, [2], True)
        mul_743: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_741, mul_10);  mul_741 = None
        sum_300: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_743, [2], True);  mul_743 = None
        mul_744: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_10, sum_300);  sum_300 = None
        sub_237: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_742, sum_299);  mul_742 = sum_299 = None
        sub_238: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_237, mul_744);  sub_237 = mul_744 = None
        mul_745: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_71, sub_238);  div_71 = sub_238 = None
        mul_746: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1092, mul_10);  mul_10 = None
        sum_301: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_746, [0, 1]);  mul_746 = None
        sum_302: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1092, [0, 1]);  view_1092 = None
        add_357: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_354, mul_745);  add_354 = mul_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1093: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_357, [4096, 2048])
        permute_19: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_1003: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_343: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1093, permute_1003);  permute_1003 = None
        permute_1004: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1093, [1, 0])
        mm_344: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1004, view_38);  permute_1004 = view_38 = None
        sum_303: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1093, [0], True);  view_1093 = None
        view_1094: "f32[2048]" = torch.ops.aten.reshape.default(sum_303, [2048]);  sum_303 = None
        view_1095: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_343, [32, 128, 2048]);  mm_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1096: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_1095, [32, 128, 16, 128]);  view_1095 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1007: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1096, [0, 2, 1, 3]);  view_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_258: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_1007, memory_format = torch.contiguous_format);  permute_1007 = None
        view_1097: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_258, [512, 128, 128]);  clone_258 = None
        expand_8: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div_1, [32, 16, 128, 128])
        view_34: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_8, [512, 128, 128]);  expand_8 = None
        permute_1008: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
        bmm_136: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_1008, view_1097);  permute_1008 = None
        bmm_137: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1097, permute_1009);  view_1097 = permute_1009 = None
        view_1098: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_136, [32, 16, 128, 128]);  bmm_136 = None
        view_1099: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_137, [32, 16, 128, 128]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_747: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1099, div_1);  view_1099 = None
        sum_304: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [-1], True)
        neg_23: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_22: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_23, sum_304, mul_747);  neg_23 = sum_304 = mul_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_6: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_23, 2, 0, 128);  primals_23 = None
        slice_7: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_6, 3, 0, 128);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_51: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_7, fma_22, full_default_1);  slice_7 = fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1100: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_51, [512, 128, 128]);  where_51 = None
        bmm_138: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_1010, view_1100);  permute_1010 = None
        bmm_139: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1100, permute_1011);  view_1100 = permute_1011 = None
        view_1101: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_138, [32, 16, 128, 128]);  bmm_138 = None
        view_1102: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_139, [32, 16, 128, 128]);  bmm_139 = None
        permute_1012: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1101, [0, 1, 3, 2]);  view_1101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1013: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1098, [0, 2, 1, 3]);  view_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_259: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1013, memory_format = torch.contiguous_format);  permute_1013 = None
        view_1103: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_259, [32, 128, 2048]);  clone_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1014: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_1012, [0, 2, 1, 3]);  permute_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1104: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_1014, [32, 128, 2048]);  permute_1014 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1015: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1102, [0, 2, 1, 3]);  view_1102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_260: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1015, memory_format = torch.contiguous_format);  permute_1015 = None
        view_1105: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_260, [32, 128, 2048]);  clone_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1106: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1103, [4096, 2048]);  view_1103 = None
        permute_1016: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1106, [1, 0])
        mm_345: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1016, view_22);  permute_1016 = None
        permute_13: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_1018: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_346: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1106, permute_1018);  view_1106 = permute_1018 = None
        view_1107: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_346, [32, 128, 2048]);  mm_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_261: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_1104, memory_format = torch.contiguous_format);  view_1104 = None
        view_1108: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_261, [4096, 2048]);  clone_261 = None
        permute_1020: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1108, [1, 0])
        mm_347: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1020, view_22);  permute_1020 = None
        permute_12: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_1022: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_348: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1108, permute_1022);  view_1108 = permute_1022 = None
        view_1109: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_348, [32, 128, 2048]);  mm_348 = None
        add_358: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_1107, view_1109);  view_1107 = view_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1110: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1105, [4096, 2048]);  view_1105 = None
        permute_1024: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1110, [1, 0])
        mm_349: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1024, view_22);  permute_1024 = view_22 = None
        permute_11: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_1026: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_350: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1110, permute_1026);  view_1110 = permute_1026 = None
        view_1111: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_350, [32, 128, 2048]);  mm_350 = None
        add_359: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_358, view_1111);  add_358 = view_1111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_749: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_359, primals_18);  primals_18 = None
        mul_750: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_749, 2048)
        sum_305: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True)
        mul_751: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_749, mul_8);  mul_749 = None
        sum_306: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_751, [2], True);  mul_751 = None
        mul_752: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_8, sum_306);  sum_306 = None
        sub_240: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_750, sum_305);  mul_750 = sum_305 = None
        sub_241: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_240, mul_752);  sub_240 = mul_752 = None
        mul_753: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_72, sub_241);  div_72 = sub_241 = None
        mul_754: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_359, mul_8);  mul_8 = None
        sum_307: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 1]);  mul_754 = None
        sum_308: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_359, [0, 1]);  add_359 = None
        add_360: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_357, mul_753);  add_357 = mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:307 in forward, code: hidden_states = self.c_proj(hidden_states)
        view_1112: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_360, [4096, 2048])
        permute_10: "f32[8192, 2048]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_1028: "f32[2048, 8192]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_351: "f32[4096, 8192]" = torch.ops.aten.mm.default(view_1112, permute_1028);  permute_1028 = None
        permute_1029: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1112, [1, 0])
        mm_352: "f32[2048, 8192]" = torch.ops.aten.mm.default(permute_1029, view_20);  permute_1029 = view_20 = None
        sum_309: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1112, [0], True);  view_1112 = None
        view_1113: "f32[2048]" = torch.ops.aten.reshape.default(sum_309, [2048]);  sum_309 = None
        view_1114: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(mm_351, [32, 128, 8192]);  mm_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_19: "f32[32, 128, 8192]" = torch.ops.aten.reshape.default(addmm_1, [32, 128, 8192]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_4: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_755: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1114, mul_4);  mul_4 = None
        pow_1: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 3.0)
        mul_5: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_10: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_19, mul_5);  mul_5 = None
        mul_6: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_10, 0.7978845608028654);  add_10 = None
        tanh: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
        add_11: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_756: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_1114, add_11);  view_1114 = add_11 = None
        mul_757: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_242: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_757);  mul_757 = None
        mul_758: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_755, sub_242);  mul_755 = sub_242 = None
        mul_759: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_758, 0.7978845608028654);  mul_758 = None
        mul_760: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_759, 0.044715)
        pow_48: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_19, 2.0);  view_19 = None
        mul_761: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_762: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
        add_361: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(mul_759, mul_762);  mul_759 = mul_762 = None
        mul_763: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_756, 0.5);  mul_756 = None
        add_362: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_361, mul_763);  add_361 = mul_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:305 in forward, code: hidden_states = self.c_fc(hidden_states)
        view_1115: "f32[4096, 8192]" = torch.ops.aten.reshape.default(add_362, [4096, 8192]);  add_362 = None
        permute_9: "f32[2048, 8192]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_1032: "f32[8192, 2048]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_353: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1115, permute_1032);  permute_1032 = None
        permute_1033: "f32[8192, 4096]" = torch.ops.aten.permute.default(view_1115, [1, 0])
        mm_354: "f32[8192, 2048]" = torch.ops.aten.mm.default(permute_1033, view_18);  permute_1033 = view_18 = None
        sum_310: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_1115, [0], True);  view_1115 = None
        view_1116: "f32[8192]" = torch.ops.aten.reshape.default(sum_310, [8192]);  sum_310 = None
        view_1117: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_353, [32, 128, 2048]);  mm_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_765: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1117, primals_12);  primals_12 = None
        mul_766: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_765, 2048)
        sum_311: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_765, [2], True)
        mul_767: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_765, mul_2);  mul_765 = None
        sum_312: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_767, [2], True);  mul_767 = None
        mul_768: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, sum_312);  sum_312 = None
        sub_244: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_766, sum_311);  mul_766 = sum_311 = None
        sub_245: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_244, mul_768);  sub_244 = mul_768 = None
        mul_769: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_73, sub_245);  div_73 = sub_245 = None
        mul_770: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(view_1117, mul_2);  mul_2 = None
        sum_313: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 1]);  mul_770 = None
        sum_314: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_1117, [0, 1]);  view_1117 = None
        add_363: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_360, mul_769);  add_360 = mul_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:155 in forward, code: attn_output = self.out_proj(attn_output)
        view_1118: "f32[4096, 2048]" = torch.ops.aten.reshape.default(add_363, [4096, 2048])
        permute_8: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_1036: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_355: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1118, permute_1036);  permute_1036 = None
        permute_1037: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1118, [1, 0])
        mm_356: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1037, view_16);  permute_1037 = view_16 = None
        sum_315: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True);  view_1118 = None
        view_1119: "f32[2048]" = torch.ops.aten.reshape.default(sum_315, [2048]);  sum_315 = None
        view_1120: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_355, [32, 128, 2048]);  mm_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:103 in _merge_heads, code: return tensor.view(new_shape)
        view_1121: "f32[32, 128, 16, 128]" = torch.ops.aten.reshape.default(view_1120, [32, 128, 16, 128]);  view_1120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:101 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1040: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1121, [0, 2, 1, 3]);  view_1121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:128 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        clone_262: "f32[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_1040, memory_format = torch.contiguous_format);  permute_1040 = None
        view_1122: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(clone_262, [512, 128, 128]);  clone_262 = None
        expand_4: "f32[32, 16, 128, 128]" = torch.ops.aten.expand.default(div, [32, 16, 128, 128])
        view_12: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_4, [512, 128, 128]);  expand_4 = None
        permute_1041: "f32[512, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm_140: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_1041, view_1122);  permute_1041 = None
        bmm_141: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1122, permute_1042);  view_1122 = permute_1042 = None
        view_1123: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_140, [32, 16, 128, 128]);  bmm_140 = None
        view_1124: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_141, [32, 16, 128, 128]);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:124 in _attn, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        mul_771: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_1124, div);  view_1124 = None
        sum_316: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_771, [-1], True)
        neg_24: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div);  div = None
        fma_23: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_24, sum_316, mul_771);  neg_24 = sum_316 = mul_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:114 in _attn, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        slice_4: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(primals_9, 2, 0, 128);  primals_9 = None
        slice_5: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 128);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:119 in _attn, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        where_52: "f32[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_5, fma_23, full_default_1);  slice_5 = fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:110 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        view_1125: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(where_52, [512, 128, 128]);  where_52 = None
        bmm_142: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(permute_1043, view_1125);  permute_1043 = None
        bmm_143: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_1125, permute_1044);  view_1125 = permute_1044 = None
        view_1126: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_142, [32, 16, 128, 128]);  bmm_142 = None
        view_1127: "f32[32, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_143, [32, 16, 128, 128]);  bmm_143 = None
        permute_1045: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_1126, [0, 1, 3, 2]);  view_1126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1046: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1123, [0, 2, 1, 3]);  view_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_263: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1046, memory_format = torch.contiguous_format);  permute_1046 = None
        view_1128: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_263, [32, 128, 2048]);  clone_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1047: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(permute_1045, [0, 2, 1, 3]);  permute_1045 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        view_1129: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(permute_1047, [32, 128, 2048]);  permute_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:95 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1048: "f32[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_1127, [0, 2, 1, 3]);  view_1127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:94 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_264: "f32[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_1048, memory_format = torch.contiguous_format);  permute_1048 = None
        view_1130: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(clone_264, [32, 128, 2048]);  clone_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:143 in forward, code: value = self.v_proj(hidden_states)
        view_1131: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1128, [4096, 2048]);  view_1128 = None
        permute_1049: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1131, [1, 0])
        mm_357: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1049, view);  permute_1049 = None
        permute_2: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_1051: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_358: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1131, permute_1051);  view_1131 = permute_1051 = None
        view_1132: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_358, [32, 128, 2048]);  mm_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:142 in forward, code: key = self.k_proj(hidden_states)
        clone_265: "f32[32, 128, 2048]" = torch.ops.aten.clone.default(view_1129, memory_format = torch.contiguous_format);  view_1129 = None
        view_1133: "f32[4096, 2048]" = torch.ops.aten.reshape.default(clone_265, [4096, 2048]);  clone_265 = None
        permute_1053: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1133, [1, 0])
        mm_359: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1053, view);  permute_1053 = None
        permute_1: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_1055: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_360: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1133, permute_1055);  view_1133 = permute_1055 = None
        view_1134: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_360, [32, 128, 2048]);  mm_360 = None
        add_364: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(view_1132, view_1134);  view_1132 = view_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:141 in forward, code: query = self.q_proj(hidden_states)
        view_1135: "f32[4096, 2048]" = torch.ops.aten.reshape.default(view_1130, [4096, 2048]);  view_1130 = None
        permute_1057: "f32[2048, 4096]" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_361: "f32[2048, 2048]" = torch.ops.aten.mm.default(permute_1057, view);  permute_1057 = view = None
        permute: "f32[2048, 2048]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_1059: "f32[2048, 2048]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_362: "f32[4096, 2048]" = torch.ops.aten.mm.default(view_1135, permute_1059);  view_1135 = permute_1059 = None
        view_1136: "f32[32, 128, 2048]" = torch.ops.aten.reshape.default(mm_362, [32, 128, 2048]);  mm_362 = None
        add_365: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_364, view_1136);  add_364 = view_1136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_773: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_365, primals_4);  primals_4 = None
        mul_774: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_773, 2048)
        sum_317: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_773, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        add_3: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:332 in forward, code: hidden_states = self.ln_1(hidden_states)
        sub_2: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_3, getitem_1);  add_3 = getitem_1 = None
        mul: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_775: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_773, mul);  mul_773 = None
        sum_318: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True);  mul_775 = None
        mul_776: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, sum_318);  sum_318 = None
        sub_247: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_774, sum_317);  mul_774 = sum_317 = None
        sub_248: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_247, mul_776);  sub_247 = mul_776 = None
        div_74: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 2048);  rsqrt = None
        mul_777: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_74, sub_248);  div_74 = sub_248 = None
        mul_778: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_365, mul);  mul = None
        sum_319: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 1]);  mul_778 = None
        sum_320: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_365, [0, 1]);  add_365 = None
        add_366: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_363, mul_777);  add_363 = mul_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:463 in forward, code: hidden_states = inputs_embeds + position_embeds
        sum_321: "f32[1, 128, 2048]" = torch.ops.aten.sum.dim_IntList(add_366, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:462 in forward, code: position_embeds = self.wpe(position_ids)
        eq_1: "b8[1, 128]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_12: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_53: "f32[1, 128, 2048]" = torch.ops.aten.where.self(unsqueeze_12, full_default_1, sum_321);  unsqueeze_12 = sum_321 = None
        full_default_57: "f32[2048, 2048]" = torch.ops.aten.full.default([2048, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[2048, 2048]" = torch.ops.aten.index_put.default(full_default_57, [unsqueeze], where_53, True);  full_default_57 = unsqueeze = where_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:444 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_2: "b8[32, 128]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_13: "b8[32, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_54: "f32[32, 128, 2048]" = torch.ops.aten.where.self(unsqueeze_13, full_default_1, add_366);  unsqueeze_13 = full_default_1 = add_366 = None
        full_default_59: "f32[50257, 2048]" = torch.ops.aten.full.default([50257, 2048], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[50257, 2048]" = torch.ops.aten.index_put.default(full_default_59, [primals_1], where_54, True);  full_default_59 = primals_1 = where_54 = None
        add_367: "f32[50257, 2048]" = torch.ops.aten.add.Tensor(slice_tensor, index_put_1);  slice_tensor = index_put_1 = None
        return (None, add_367, index_put, sum_319, sum_320, mm_361, mm_359, mm_357, None, mm_356, view_1119, sum_313, sum_314, mm_354, view_1116, mm_352, view_1113, sum_307, sum_308, mm_349, mm_347, mm_345, None, mm_344, view_1094, sum_301, sum_302, mm_342, view_1091, mm_340, view_1088, sum_295, sum_296, mm_337, mm_335, mm_333, None, mm_332, view_1069, sum_289, sum_290, mm_330, view_1066, mm_328, view_1063, sum_283, sum_284, mm_325, mm_323, mm_321, None, mm_320, view_1044, sum_277, sum_278, mm_318, view_1041, mm_316, view_1038, sum_271, sum_272, mm_313, mm_311, mm_309, None, mm_308, view_1019, sum_265, sum_266, mm_306, view_1016, mm_304, view_1013, sum_259, sum_260, mm_301, mm_299, mm_297, None, mm_296, view_994, sum_253, sum_254, mm_294, view_991, mm_292, view_988, sum_247, sum_248, mm_289, mm_287, mm_285, None, mm_284, view_969, sum_241, sum_242, mm_282, view_966, mm_280, view_963, sum_235, sum_236, mm_277, mm_275, mm_273, None, mm_272, view_944, sum_229, sum_230, mm_270, view_941, mm_268, view_938, sum_223, sum_224, mm_265, mm_263, mm_261, None, mm_260, view_919, sum_217, sum_218, mm_258, view_916, mm_256, view_913, sum_211, sum_212, mm_253, mm_251, mm_249, None, mm_248, view_894, sum_205, sum_206, mm_246, view_891, mm_244, view_888, sum_199, sum_200, mm_241, mm_239, mm_237, None, mm_236, view_869, sum_193, sum_194, mm_234, view_866, mm_232, view_863, sum_187, sum_188, mm_229, mm_227, mm_225, None, mm_224, view_844, sum_181, sum_182, mm_222, view_841, mm_220, view_838, sum_175, sum_176, mm_217, mm_215, mm_213, None, mm_212, view_819, sum_169, sum_170, mm_210, view_816, mm_208, view_813, sum_163, sum_164, mm_205, mm_203, mm_201, None, mm_200, view_794, sum_157, sum_158, mm_198, view_791, mm_196, view_788, sum_151, sum_152, mm_193, mm_191, mm_189, None, mm_188, view_769, sum_145, sum_146, mm_186, view_766, mm_184, view_763, sum_139, sum_140, mm_181, mm_179, mm_177, None, mm_176, view_744, sum_133, sum_134, mm_174, view_741, mm_172, view_738, sum_127, sum_128, mm_169, mm_167, mm_165, None, mm_164, view_719, sum_121, sum_122, mm_162, view_716, mm_160, view_713, sum_115, sum_116, mm_157, mm_155, mm_153, None, mm_152, view_694, sum_109, sum_110, mm_150, view_691, mm_148, view_688, sum_103, sum_104, mm_145, mm_143, mm_141, None, mm_140, view_669, sum_97, sum_98, mm_138, view_666, mm_136, view_663, sum_91, sum_92, mm_133, mm_131, mm_129, None, mm_128, view_644, sum_85, sum_86, mm_126, view_641, mm_124, view_638, sum_79, sum_80, mm_121, mm_119, mm_117, None, mm_116, view_619, sum_73, sum_74, mm_114, view_616, mm_112, view_613, sum_67, sum_68, mm_109, mm_107, mm_105, None, mm_104, view_594, sum_61, sum_62, mm_102, view_591, mm_100, view_588, sum_55, sum_56, mm_97, mm_95, mm_93, None, mm_92, view_569, sum_49, sum_50, mm_90, view_566, mm_88, view_563, sum_43, sum_44, mm_85, mm_83, mm_81, None, mm_80, view_544, sum_37, sum_38, mm_78, view_541, mm_76, view_538, sum_31, sum_32, None)
