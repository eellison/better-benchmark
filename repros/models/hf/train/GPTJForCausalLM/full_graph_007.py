class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[1, 128]", primals_3: "f32[4096]", primals_5: "f32[4096, 4096]", primals_6: "f32[4096, 4096]", primals_7: "f32[4096, 4096]", primals_9: "f32[4096, 4096]", primals_10: "f32[16384, 4096]", primals_12: "f32[4096, 16384]", primals_14: "f32[4096]", primals_16: "f32[4096, 4096]", primals_17: "f32[4096, 4096]", primals_18: "f32[4096, 4096]", primals_20: "f32[4096, 4096]", primals_21: "f32[16384, 4096]", primals_23: "f32[4096, 16384]", primals_25: "f32[4096]", primals_27: "f32[4096, 4096]", primals_28: "f32[4096, 4096]", primals_29: "f32[4096, 4096]", primals_31: "f32[4096, 4096]", primals_32: "f32[16384, 4096]", primals_34: "f32[4096, 16384]", primals_36: "f32[4096]", primals_38: "f32[4096, 4096]", primals_39: "f32[4096, 4096]", primals_40: "f32[4096, 4096]", primals_42: "f32[4096, 4096]", primals_43: "f32[16384, 4096]", primals_45: "f32[4096, 16384]", primals_47: "f32[4096]", primals_49: "f32[4096, 4096]", primals_50: "f32[4096, 4096]", primals_51: "f32[4096, 4096]", primals_53: "f32[4096, 4096]", primals_54: "f32[16384, 4096]", primals_56: "f32[4096, 16384]", primals_58: "f32[4096]", primals_60: "f32[4096, 4096]", primals_61: "f32[4096, 4096]", primals_62: "f32[4096, 4096]", primals_64: "f32[4096, 4096]", primals_65: "f32[16384, 4096]", primals_67: "f32[4096, 16384]", primals_69: "f32[4096]", primals_71: "f32[4096, 4096]", primals_72: "f32[4096, 4096]", primals_73: "f32[4096, 4096]", primals_75: "f32[4096, 4096]", primals_76: "f32[16384, 4096]", primals_78: "f32[4096, 16384]", primals_80: "f32[4096]", primals_82: "f32[4096, 4096]", primals_83: "f32[4096, 4096]", primals_84: "f32[4096, 4096]", primals_86: "f32[4096, 4096]", primals_87: "f32[16384, 4096]", primals_89: "f32[4096, 16384]", primals_91: "f32[4096]", primals_93: "f32[4096, 4096]", primals_94: "f32[4096, 4096]", primals_95: "f32[4096, 4096]", primals_97: "f32[4096, 4096]", primals_98: "f32[16384, 4096]", primals_100: "f32[4096, 16384]", primals_102: "f32[4096]", primals_104: "f32[4096, 4096]", primals_105: "f32[4096, 4096]", primals_106: "f32[4096, 4096]", primals_108: "f32[4096, 4096]", primals_109: "f32[16384, 4096]", primals_111: "f32[4096, 16384]", primals_113: "f32[4096]", primals_115: "f32[4096, 4096]", primals_116: "f32[4096, 4096]", primals_117: "f32[4096, 4096]", primals_119: "f32[4096, 4096]", primals_120: "f32[16384, 4096]", primals_122: "f32[4096, 16384]", primals_124: "f32[4096]", primals_126: "f32[4096, 4096]", primals_127: "f32[4096, 4096]", primals_128: "f32[4096, 4096]", primals_130: "f32[4096, 4096]", primals_131: "f32[16384, 4096]", primals_133: "f32[4096, 16384]", primals_135: "f32[4096]", primals_137: "f32[4096, 4096]", primals_138: "f32[4096, 4096]", primals_139: "f32[4096, 4096]", primals_141: "f32[4096, 4096]", primals_142: "f32[16384, 4096]", primals_144: "f32[4096, 16384]", primals_146: "f32[4096]", primals_148: "f32[4096, 4096]", primals_149: "f32[4096, 4096]", primals_150: "f32[4096, 4096]", primals_152: "f32[4096, 4096]", primals_153: "f32[16384, 4096]", primals_155: "f32[4096, 16384]", primals_157: "f32[4096]", primals_159: "f32[4096, 4096]", primals_160: "f32[4096, 4096]", primals_161: "f32[4096, 4096]", primals_163: "f32[4096, 4096]", primals_164: "f32[16384, 4096]", primals_166: "f32[4096, 16384]", primals_168: "f32[4096]", primals_170: "f32[4096, 4096]", primals_171: "f32[4096, 4096]", primals_172: "f32[4096, 4096]", primals_174: "f32[4096, 4096]", primals_175: "f32[16384, 4096]", primals_177: "f32[4096, 16384]", primals_179: "f32[4096]", primals_181: "f32[4096, 4096]", primals_182: "f32[4096, 4096]", primals_183: "f32[4096, 4096]", primals_185: "f32[4096, 4096]", primals_186: "f32[16384, 4096]", primals_188: "f32[4096, 16384]", primals_190: "f32[4096]", primals_192: "f32[4096, 4096]", primals_193: "f32[4096, 4096]", primals_194: "f32[4096, 4096]", primals_196: "f32[4096, 4096]", primals_197: "f32[16384, 4096]", primals_199: "f32[4096, 16384]", primals_201: "f32[4096]", primals_203: "f32[4096, 4096]", primals_204: "f32[4096, 4096]", primals_205: "f32[4096, 4096]", primals_207: "f32[4096, 4096]", primals_208: "f32[16384, 4096]", primals_210: "f32[4096, 16384]", primals_212: "f32[4096]", primals_214: "f32[4096, 4096]", primals_215: "f32[4096, 4096]", primals_216: "f32[4096, 4096]", primals_218: "f32[4096, 4096]", primals_219: "f32[16384, 4096]", primals_221: "f32[4096, 16384]", primals_223: "f32[4096]", primals_225: "f32[4096, 4096]", primals_226: "f32[4096, 4096]", primals_227: "f32[4096, 4096]", primals_229: "f32[4096, 4096]", primals_230: "f32[16384, 4096]", primals_232: "f32[4096, 16384]", primals_234: "f32[4096]", primals_236: "f32[4096, 4096]", primals_237: "f32[4096, 4096]", primals_238: "f32[4096, 4096]", primals_240: "f32[4096, 4096]", primals_241: "f32[16384, 4096]", primals_243: "f32[4096, 16384]", primals_245: "f32[4096]", primals_247: "f32[4096, 4096]", primals_248: "f32[4096, 4096]", primals_249: "f32[4096, 4096]", primals_251: "f32[4096, 4096]", primals_252: "f32[16384, 4096]", primals_254: "f32[4096, 16384]", primals_256: "f32[4096]", primals_258: "f32[4096, 4096]", primals_259: "f32[4096, 4096]", primals_260: "f32[4096, 4096]", primals_262: "f32[4096, 4096]", primals_263: "f32[16384, 4096]", primals_265: "f32[4096, 16384]", primals_267: "f32[4096]", primals_269: "f32[4096, 4096]", primals_270: "f32[4096, 4096]", primals_271: "f32[4096, 4096]", primals_273: "f32[4096, 4096]", primals_274: "f32[16384, 4096]", primals_276: "f32[4096, 16384]", primals_278: "f32[4096]", primals_280: "f32[4096, 4096]", primals_281: "f32[4096, 4096]", primals_282: "f32[4096, 4096]", primals_284: "f32[4096, 4096]", primals_285: "f32[16384, 4096]", primals_287: "f32[4096, 16384]", primals_289: "f32[4096]", primals_291: "f32[4096, 4096]", primals_292: "f32[4096, 4096]", primals_293: "f32[4096, 4096]", primals_295: "f32[4096, 4096]", primals_296: "f32[16384, 4096]", primals_298: "f32[4096, 16384]", primals_300: "f32[4096]", primals_302: "f32[4096, 4096]", primals_303: "f32[4096, 4096]", primals_304: "f32[4096, 4096]", primals_306: "f32[4096, 4096]", primals_307: "f32[16384, 4096]", primals_309: "f32[4096, 16384]", primals_311: "f32[4096]", primals_313: "f32[50400, 4096]", embedding: "f32[1, 128, 4096]", where: "f32[1, 1, 128, 128]", getitem_1: "f32[1, 128, 1]", rsqrt: "f32[1, 128, 1]", view: "f32[128, 4096]", permute_3: "f32[1, 16, 128, 256]", unsqueeze_12: "f32[1, 128, 1, 32, 1]", unsqueeze_14: "f32[1, 128, 1, 32, 1]", permute_4: "f32[1, 16, 128, 256]", permute_5: "f32[1, 16, 128, 256]", getitem_303: "f32[1, 16, 128, 256]", getitem_304: "f32[1, 16, 128]", getitem_305: "i64[]", getitem_306: "i64[]", view_22: "f32[128, 4096]", addmm: "f32[128, 16384]", view_26: "f32[128, 16384]", mul_10: "f32[1, 128, 4096]", view_28: "f32[128, 4096]", permute_14: "f32[1, 16, 128, 256]", unsqueeze_25: "f32[1, 128, 1, 32, 1]", unsqueeze_27: "f32[1, 128, 1, 32, 1]", permute_15: "f32[1, 16, 128, 256]", permute_16: "f32[1, 16, 128, 256]", getitem_296: "f32[1, 16, 128, 256]", getitem_297: "f32[1, 16, 128]", getitem_298: "i64[]", getitem_299: "i64[]", view_50: "f32[128, 4096]", addmm_2: "f32[128, 16384]", view_54: "f32[128, 16384]", mul_20: "f32[1, 128, 4096]", view_56: "f32[128, 4096]", permute_25: "f32[1, 16, 128, 256]", unsqueeze_38: "f32[1, 128, 1, 32, 1]", unsqueeze_40: "f32[1, 128, 1, 32, 1]", permute_26: "f32[1, 16, 128, 256]", permute_27: "f32[1, 16, 128, 256]", getitem_289: "f32[1, 16, 128, 256]", getitem_290: "f32[1, 16, 128]", getitem_291: "i64[]", getitem_292: "i64[]", view_78: "f32[128, 4096]", addmm_4: "f32[128, 16384]", view_82: "f32[128, 16384]", mul_30: "f32[1, 128, 4096]", view_84: "f32[128, 4096]", permute_36: "f32[1, 16, 128, 256]", unsqueeze_51: "f32[1, 128, 1, 32, 1]", unsqueeze_53: "f32[1, 128, 1, 32, 1]", permute_37: "f32[1, 16, 128, 256]", permute_38: "f32[1, 16, 128, 256]", getitem_282: "f32[1, 16, 128, 256]", getitem_283: "f32[1, 16, 128]", getitem_284: "i64[]", getitem_285: "i64[]", view_106: "f32[128, 4096]", addmm_6: "f32[128, 16384]", view_110: "f32[128, 16384]", mul_40: "f32[1, 128, 4096]", view_112: "f32[128, 4096]", permute_47: "f32[1, 16, 128, 256]", unsqueeze_64: "f32[1, 128, 1, 32, 1]", unsqueeze_66: "f32[1, 128, 1, 32, 1]", permute_48: "f32[1, 16, 128, 256]", permute_49: "f32[1, 16, 128, 256]", getitem_275: "f32[1, 16, 128, 256]", getitem_276: "f32[1, 16, 128]", getitem_277: "i64[]", getitem_278: "i64[]", view_134: "f32[128, 4096]", addmm_8: "f32[128, 16384]", view_138: "f32[128, 16384]", mul_50: "f32[1, 128, 4096]", view_140: "f32[128, 4096]", permute_58: "f32[1, 16, 128, 256]", unsqueeze_77: "f32[1, 128, 1, 32, 1]", unsqueeze_79: "f32[1, 128, 1, 32, 1]", permute_59: "f32[1, 16, 128, 256]", permute_60: "f32[1, 16, 128, 256]", getitem_268: "f32[1, 16, 128, 256]", getitem_269: "f32[1, 16, 128]", getitem_270: "i64[]", getitem_271: "i64[]", view_162: "f32[128, 4096]", addmm_10: "f32[128, 16384]", view_166: "f32[128, 16384]", mul_60: "f32[1, 128, 4096]", view_168: "f32[128, 4096]", permute_69: "f32[1, 16, 128, 256]", unsqueeze_90: "f32[1, 128, 1, 32, 1]", unsqueeze_92: "f32[1, 128, 1, 32, 1]", permute_70: "f32[1, 16, 128, 256]", permute_71: "f32[1, 16, 128, 256]", getitem_261: "f32[1, 16, 128, 256]", getitem_262: "f32[1, 16, 128]", getitem_263: "i64[]", getitem_264: "i64[]", view_190: "f32[128, 4096]", addmm_12: "f32[128, 16384]", view_194: "f32[128, 16384]", mul_70: "f32[1, 128, 4096]", view_196: "f32[128, 4096]", permute_80: "f32[1, 16, 128, 256]", unsqueeze_103: "f32[1, 128, 1, 32, 1]", unsqueeze_105: "f32[1, 128, 1, 32, 1]", permute_81: "f32[1, 16, 128, 256]", permute_82: "f32[1, 16, 128, 256]", getitem_254: "f32[1, 16, 128, 256]", getitem_255: "f32[1, 16, 128]", getitem_256: "i64[]", getitem_257: "i64[]", view_218: "f32[128, 4096]", addmm_14: "f32[128, 16384]", view_222: "f32[128, 16384]", mul_80: "f32[1, 128, 4096]", view_224: "f32[128, 4096]", permute_91: "f32[1, 16, 128, 256]", unsqueeze_116: "f32[1, 128, 1, 32, 1]", unsqueeze_118: "f32[1, 128, 1, 32, 1]", permute_92: "f32[1, 16, 128, 256]", permute_93: "f32[1, 16, 128, 256]", getitem_247: "f32[1, 16, 128, 256]", getitem_248: "f32[1, 16, 128]", getitem_249: "i64[]", getitem_250: "i64[]", view_246: "f32[128, 4096]", addmm_16: "f32[128, 16384]", view_250: "f32[128, 16384]", mul_90: "f32[1, 128, 4096]", view_252: "f32[128, 4096]", permute_102: "f32[1, 16, 128, 256]", unsqueeze_129: "f32[1, 128, 1, 32, 1]", unsqueeze_131: "f32[1, 128, 1, 32, 1]", permute_103: "f32[1, 16, 128, 256]", permute_104: "f32[1, 16, 128, 256]", getitem_240: "f32[1, 16, 128, 256]", getitem_241: "f32[1, 16, 128]", getitem_242: "i64[]", getitem_243: "i64[]", view_274: "f32[128, 4096]", addmm_18: "f32[128, 16384]", view_278: "f32[128, 16384]", mul_100: "f32[1, 128, 4096]", view_280: "f32[128, 4096]", permute_113: "f32[1, 16, 128, 256]", unsqueeze_142: "f32[1, 128, 1, 32, 1]", unsqueeze_144: "f32[1, 128, 1, 32, 1]", permute_114: "f32[1, 16, 128, 256]", permute_115: "f32[1, 16, 128, 256]", getitem_233: "f32[1, 16, 128, 256]", getitem_234: "f32[1, 16, 128]", getitem_235: "i64[]", getitem_236: "i64[]", view_302: "f32[128, 4096]", addmm_20: "f32[128, 16384]", view_306: "f32[128, 16384]", mul_110: "f32[1, 128, 4096]", view_308: "f32[128, 4096]", permute_124: "f32[1, 16, 128, 256]", unsqueeze_155: "f32[1, 128, 1, 32, 1]", unsqueeze_157: "f32[1, 128, 1, 32, 1]", permute_125: "f32[1, 16, 128, 256]", permute_126: "f32[1, 16, 128, 256]", getitem_226: "f32[1, 16, 128, 256]", getitem_227: "f32[1, 16, 128]", getitem_228: "i64[]", getitem_229: "i64[]", view_330: "f32[128, 4096]", addmm_22: "f32[128, 16384]", view_334: "f32[128, 16384]", mul_120: "f32[1, 128, 4096]", view_336: "f32[128, 4096]", permute_135: "f32[1, 16, 128, 256]", unsqueeze_168: "f32[1, 128, 1, 32, 1]", unsqueeze_170: "f32[1, 128, 1, 32, 1]", permute_136: "f32[1, 16, 128, 256]", permute_137: "f32[1, 16, 128, 256]", getitem_219: "f32[1, 16, 128, 256]", getitem_220: "f32[1, 16, 128]", getitem_221: "i64[]", getitem_222: "i64[]", view_358: "f32[128, 4096]", addmm_24: "f32[128, 16384]", view_362: "f32[128, 16384]", mul_130: "f32[1, 128, 4096]", view_364: "f32[128, 4096]", permute_146: "f32[1, 16, 128, 256]", unsqueeze_181: "f32[1, 128, 1, 32, 1]", unsqueeze_183: "f32[1, 128, 1, 32, 1]", permute_147: "f32[1, 16, 128, 256]", permute_148: "f32[1, 16, 128, 256]", getitem_212: "f32[1, 16, 128, 256]", getitem_213: "f32[1, 16, 128]", getitem_214: "i64[]", getitem_215: "i64[]", view_386: "f32[128, 4096]", addmm_26: "f32[128, 16384]", view_390: "f32[128, 16384]", mul_140: "f32[1, 128, 4096]", view_392: "f32[128, 4096]", permute_157: "f32[1, 16, 128, 256]", unsqueeze_194: "f32[1, 128, 1, 32, 1]", unsqueeze_196: "f32[1, 128, 1, 32, 1]", permute_158: "f32[1, 16, 128, 256]", permute_159: "f32[1, 16, 128, 256]", getitem_205: "f32[1, 16, 128, 256]", getitem_206: "f32[1, 16, 128]", getitem_207: "i64[]", getitem_208: "i64[]", view_414: "f32[128, 4096]", addmm_28: "f32[128, 16384]", view_418: "f32[128, 16384]", mul_150: "f32[1, 128, 4096]", view_420: "f32[128, 4096]", permute_168: "f32[1, 16, 128, 256]", unsqueeze_207: "f32[1, 128, 1, 32, 1]", unsqueeze_209: "f32[1, 128, 1, 32, 1]", permute_169: "f32[1, 16, 128, 256]", permute_170: "f32[1, 16, 128, 256]", getitem_198: "f32[1, 16, 128, 256]", getitem_199: "f32[1, 16, 128]", getitem_200: "i64[]", getitem_201: "i64[]", view_442: "f32[128, 4096]", addmm_30: "f32[128, 16384]", view_446: "f32[128, 16384]", mul_160: "f32[1, 128, 4096]", view_448: "f32[128, 4096]", permute_179: "f32[1, 16, 128, 256]", unsqueeze_220: "f32[1, 128, 1, 32, 1]", unsqueeze_222: "f32[1, 128, 1, 32, 1]", permute_180: "f32[1, 16, 128, 256]", permute_181: "f32[1, 16, 128, 256]", getitem_191: "f32[1, 16, 128, 256]", getitem_192: "f32[1, 16, 128]", getitem_193: "i64[]", getitem_194: "i64[]", view_470: "f32[128, 4096]", addmm_32: "f32[128, 16384]", view_474: "f32[128, 16384]", mul_170: "f32[1, 128, 4096]", view_476: "f32[128, 4096]", permute_190: "f32[1, 16, 128, 256]", unsqueeze_233: "f32[1, 128, 1, 32, 1]", unsqueeze_235: "f32[1, 128, 1, 32, 1]", permute_191: "f32[1, 16, 128, 256]", permute_192: "f32[1, 16, 128, 256]", getitem_184: "f32[1, 16, 128, 256]", getitem_185: "f32[1, 16, 128]", getitem_186: "i64[]", getitem_187: "i64[]", view_498: "f32[128, 4096]", addmm_34: "f32[128, 16384]", view_502: "f32[128, 16384]", mul_180: "f32[1, 128, 4096]", view_504: "f32[128, 4096]", permute_201: "f32[1, 16, 128, 256]", unsqueeze_246: "f32[1, 128, 1, 32, 1]", unsqueeze_248: "f32[1, 128, 1, 32, 1]", permute_202: "f32[1, 16, 128, 256]", permute_203: "f32[1, 16, 128, 256]", getitem_177: "f32[1, 16, 128, 256]", getitem_178: "f32[1, 16, 128]", getitem_179: "i64[]", getitem_180: "i64[]", view_526: "f32[128, 4096]", addmm_36: "f32[128, 16384]", view_530: "f32[128, 16384]", mul_190: "f32[1, 128, 4096]", view_532: "f32[128, 4096]", permute_212: "f32[1, 16, 128, 256]", unsqueeze_259: "f32[1, 128, 1, 32, 1]", unsqueeze_261: "f32[1, 128, 1, 32, 1]", permute_213: "f32[1, 16, 128, 256]", permute_214: "f32[1, 16, 128, 256]", getitem_170: "f32[1, 16, 128, 256]", getitem_171: "f32[1, 16, 128]", getitem_172: "i64[]", getitem_173: "i64[]", view_554: "f32[128, 4096]", addmm_38: "f32[128, 16384]", view_558: "f32[128, 16384]", mul_200: "f32[1, 128, 4096]", view_560: "f32[128, 4096]", permute_223: "f32[1, 16, 128, 256]", unsqueeze_272: "f32[1, 128, 1, 32, 1]", unsqueeze_274: "f32[1, 128, 1, 32, 1]", permute_224: "f32[1, 16, 128, 256]", permute_225: "f32[1, 16, 128, 256]", getitem_163: "f32[1, 16, 128, 256]", getitem_164: "f32[1, 16, 128]", getitem_165: "i64[]", getitem_166: "i64[]", view_582: "f32[128, 4096]", addmm_40: "f32[128, 16384]", view_586: "f32[128, 16384]", mul_210: "f32[1, 128, 4096]", view_588: "f32[128, 4096]", permute_234: "f32[1, 16, 128, 256]", unsqueeze_285: "f32[1, 128, 1, 32, 1]", unsqueeze_287: "f32[1, 128, 1, 32, 1]", permute_235: "f32[1, 16, 128, 256]", permute_236: "f32[1, 16, 128, 256]", getitem_156: "f32[1, 16, 128, 256]", getitem_157: "f32[1, 16, 128]", getitem_158: "i64[]", getitem_159: "i64[]", view_610: "f32[128, 4096]", addmm_42: "f32[128, 16384]", view_614: "f32[128, 16384]", mul_220: "f32[1, 128, 4096]", view_616: "f32[128, 4096]", permute_245: "f32[1, 16, 128, 256]", unsqueeze_298: "f32[1, 128, 1, 32, 1]", unsqueeze_300: "f32[1, 128, 1, 32, 1]", permute_246: "f32[1, 16, 128, 256]", permute_247: "f32[1, 16, 128, 256]", getitem_149: "f32[1, 16, 128, 256]", getitem_150: "f32[1, 16, 128]", getitem_151: "i64[]", getitem_152: "i64[]", view_638: "f32[128, 4096]", addmm_44: "f32[128, 16384]", view_642: "f32[128, 16384]", mul_230: "f32[1, 128, 4096]", view_644: "f32[128, 4096]", permute_256: "f32[1, 16, 128, 256]", unsqueeze_311: "f32[1, 128, 1, 32, 1]", unsqueeze_313: "f32[1, 128, 1, 32, 1]", permute_257: "f32[1, 16, 128, 256]", permute_258: "f32[1, 16, 128, 256]", getitem_142: "f32[1, 16, 128, 256]", getitem_143: "f32[1, 16, 128]", getitem_144: "i64[]", getitem_145: "i64[]", view_666: "f32[128, 4096]", addmm_46: "f32[128, 16384]", view_670: "f32[128, 16384]", mul_240: "f32[1, 128, 4096]", view_672: "f32[128, 4096]", permute_267: "f32[1, 16, 128, 256]", unsqueeze_324: "f32[1, 128, 1, 32, 1]", unsqueeze_326: "f32[1, 128, 1, 32, 1]", permute_268: "f32[1, 16, 128, 256]", permute_269: "f32[1, 16, 128, 256]", getitem_135: "f32[1, 16, 128, 256]", getitem_136: "f32[1, 16, 128]", getitem_137: "i64[]", getitem_138: "i64[]", view_694: "f32[128, 4096]", addmm_48: "f32[128, 16384]", view_698: "f32[128, 16384]", mul_250: "f32[1, 128, 4096]", view_700: "f32[128, 4096]", permute_278: "f32[1, 16, 128, 256]", unsqueeze_337: "f32[1, 128, 1, 32, 1]", unsqueeze_339: "f32[1, 128, 1, 32, 1]", permute_279: "f32[1, 16, 128, 256]", permute_280: "f32[1, 16, 128, 256]", getitem_128: "f32[1, 16, 128, 256]", getitem_129: "f32[1, 16, 128]", getitem_130: "i64[]", getitem_131: "i64[]", view_722: "f32[128, 4096]", addmm_50: "f32[128, 16384]", view_726: "f32[128, 16384]", mul_260: "f32[1, 128, 4096]", view_728: "f32[128, 4096]", permute_289: "f32[1, 16, 128, 256]", unsqueeze_350: "f32[1, 128, 1, 32, 1]", unsqueeze_352: "f32[1, 128, 1, 32, 1]", permute_290: "f32[1, 16, 128, 256]", permute_291: "f32[1, 16, 128, 256]", getitem_121: "f32[1, 16, 128, 256]", getitem_122: "f32[1, 16, 128]", getitem_123: "i64[]", getitem_124: "i64[]", view_750: "f32[128, 4096]", addmm_52: "f32[128, 16384]", view_754: "f32[128, 16384]", mul_270: "f32[1, 128, 4096]", view_756: "f32[128, 4096]", permute_300: "f32[1, 16, 128, 256]", unsqueeze_363: "f32[1, 128, 1, 32, 1]", unsqueeze_365: "f32[1, 128, 1, 32, 1]", permute_301: "f32[1, 16, 128, 256]", permute_302: "f32[1, 16, 128, 256]", getitem_114: "f32[1, 16, 128, 256]", getitem_115: "f32[1, 16, 128]", getitem_116: "i64[]", getitem_117: "i64[]", view_778: "f32[128, 4096]", addmm_54: "f32[128, 16384]", view_782: "f32[128, 16384]", mul_280: "f32[1, 128, 4096]", view_785: "f32[128, 4096]", view_786: "f32[1, 128, 50400]", constant_pad_nd: "i64[1, 129]", amax_28: "f32[128, 1]", log: "f32[128, 1]", convert_element_type: "f32[]", div_58: "f32[1, 128, 1]", div_60: "f32[1, 128, 1]", div_62: "f32[1, 128, 1]", div_64: "f32[1, 128, 1]", div_66: "f32[1, 128, 1]", div_68: "f32[1, 128, 1]", div_70: "f32[1, 128, 1]", div_72: "f32[1, 128, 1]", div_74: "f32[1, 128, 1]", div_76: "f32[1, 128, 1]", div_78: "f32[1, 128, 1]", div_80: "f32[1, 128, 1]", div_82: "f32[1, 128, 1]", div_84: "f32[1, 128, 1]", div_86: "f32[1, 128, 1]", div_88: "f32[1, 128, 1]", div_90: "f32[1, 128, 1]", div_92: "f32[1, 128, 1]", div_94: "f32[1, 128, 1]", div_96: "f32[1, 128, 1]", div_98: "f32[1, 128, 1]", div_100: "f32[1, 128, 1]", div_102: "f32[1, 128, 1]", div_104: "f32[1, 128, 1]", div_106: "f32[1, 128, 1]", div_108: "f32[1, 128, 1]", div_110: "f32[1, 128, 1]", div_112: "f32[1, 128, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 128, 50400]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_57: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_228: "i64[1, 128]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_788: "i64[128]" = torch.ops.aten.reshape.default(slice_228, [-1]);  slice_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_375: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(view_788, 1);  view_788 = None
        ne_4: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_375, -100)
        full_default_3: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "i64[128, 1]" = torch.ops.aten.where.self(ne_4, unsqueeze_375, full_default_3);  unsqueeze_375 = full_default_3 = None

        # No stacktrace found for following nodes
        iota_default: "i64[50400]" = torch.ops.prims.iota.default(50400, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 50400]" = torch.ops.aten.reshape.default(iota_default, [1, 50400]);  iota_default = None
        expand_default: "i64[128, 50400]" = torch.ops.aten.expand.default(where_3, [128, 50400]);  where_3 = None
        eq_tensor: "b8[128, 50400]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[128, 50400]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_4: "f32[128, 1]" = torch.ops.aten.where.self(ne_4, div_57, full_default_1);  ne_4 = div_57 = None
        mul_282: "f32[128, 50400]" = torch.ops.aten.mul.Tensor(where_self, where_4);  where_self = where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_787: "f32[128, 50400]" = torch.ops.aten.reshape.default(view_786, [-1, 50400]);  view_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_59: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(view_787, amax_28);  view_787 = amax_28 = None
        sub_60: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(sub_59, log);  sub_59 = log = None
        exp_29: "f32[128, 50400]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        sum_32: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [1], True)
        mul_283: "f32[128, 50400]" = torch.ops.aten.mul.Tensor(exp_29, sum_32);  exp_29 = sum_32 = None
        sub_61: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_789: "f32[1, 128, 50400]" = torch.ops.aten.reshape.default(sub_61, [1, 128, 50400]);  sub_61 = None
        add_257: "f32[1, 128, 50400]" = torch.ops.aten.add.Tensor(tangents_2, view_789);  tangents_2 = view_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:623 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_790: "f32[128, 50400]" = torch.ops.aten.reshape.default(add_257, [128, 50400]);  add_257 = None
        permute_308: "f32[4096, 50400]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
        permute_309: "f32[50400, 4096]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
        mm_112: "f32[128, 4096]" = torch.ops.aten.mm.default(view_790, permute_309);  permute_309 = None
        permute_310: "f32[50400, 128]" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_113: "f32[50400, 4096]" = torch.ops.aten.mm.default(permute_310, view_785);  permute_310 = view_785 = None
        sum_33: "f32[1, 50400]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
        view_791: "f32[50400]" = torch.ops.aten.reshape.default(sum_33, [50400]);  sum_33 = None
        view_792: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_112, [1, 128, 4096]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_285: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_792, primals_311);  primals_311 = None
        mul_286: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_285, 4096)
        sum_34: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True)
        mul_287: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_285, mul_280);  mul_285 = None
        sum_35: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True);  mul_287 = None
        mul_288: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, sum_35);  sum_35 = None
        sub_63: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_286, sum_34);  mul_286 = sum_34 = None
        sub_64: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_63, mul_288);  sub_63 = mul_288 = None
        mul_289: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_58, sub_64);  div_58 = sub_64 = None
        mul_290: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_792, mul_280);  mul_280 = None
        sum_36: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1]);  mul_290 = None
        sum_37: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_792, [0, 1]);  view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_794: "f32[128, 4096]" = torch.ops.aten.reshape.default(mul_289, [128, 4096])
        permute_307: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
        permute_313: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
        mm_114: "f32[128, 16384]" = torch.ops.aten.mm.default(view_794, permute_313);  permute_313 = None
        permute_314: "f32[4096, 128]" = torch.ops.aten.permute.default(view_794, [1, 0])
        mm_115: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_314, view_782);  view_782 = None
        sum_38: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_794, [0], True)
        view_795: "f32[4096]" = torch.ops.aten.reshape.default(sum_38, [4096]);  sum_38 = None
        view_796: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_114, [1, 128, 16384]);  mm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_781: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 16384]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_276: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_781, 0.5)
        mul_291: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_796, mul_276);  mul_276 = None
        pow_28: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_781, 3.0)
        mul_277: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_251: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_781, mul_277);  mul_277 = None
        mul_278: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_251, 0.7978845608028654);  add_251 = None
        tanh_27: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_252: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_27, 1.0)
        mul_292: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_796, add_252);  view_796 = add_252 = None
        mul_293: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_27, tanh_27);  tanh_27 = None
        sub_65: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_293);  mul_293 = None
        mul_294: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_291, sub_65);  mul_291 = sub_65 = None
        mul_295: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_294, 0.7978845608028654);  mul_294 = None
        mul_296: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_295, 0.044715)
        pow_29: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_781, 2.0);  view_781 = None
        mul_297: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_29, 3.0);  pow_29 = None
        mul_298: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
        add_258: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_295, mul_298);  mul_295 = mul_298 = None
        mul_299: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_292, 0.5);  mul_292 = None
        add_259: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_258, mul_299);  add_258 = mul_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_797: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_259, [128, 16384]);  add_259 = None
        permute_306: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
        permute_317: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
        mm_116: "f32[128, 4096]" = torch.ops.aten.mm.default(view_797, permute_317);  permute_317 = None
        permute_318: "f32[16384, 128]" = torch.ops.aten.permute.default(view_797, [1, 0])
        mm_117: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_318, view_756);  permute_318 = None
        sum_39: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_797, [0], True);  view_797 = None
        view_798: "f32[16384]" = torch.ops.aten.reshape.default(sum_39, [16384]);  sum_39 = None
        view_799: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_116, [1, 128, 4096]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_118: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_314, view_778);  permute_314 = view_778 = None
        permute_305: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_306, [1, 0]);  primals_306 = None
        permute_323: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        mm_119: "f32[128, 4096]" = torch.ops.aten.mm.default(view_794, permute_323);  view_794 = permute_323 = None
        view_801: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_119, [1, 128, 4096]);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_802: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_801, [1, 128, 16, 256]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_325: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None

        # No stacktrace found for following nodes
        expand_default_28: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(where, [1, 16, 128, 128]);  where = None
        _scaled_dot_product_efficient_attention_backward_default = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_325, permute_302, permute_301, permute_300, expand_default_28, getitem_114, getitem_115, getitem_116, getitem_117, 0.0, [True, True, True, False], scale = 0.0625);  permute_325 = permute_302 = permute_301 = permute_300 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_118: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default[0]
        getitem_119: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_120: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default[2];  _scaled_dot_product_efficient_attention_backward_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_331: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_118, [0, 2, 1, 3]);  getitem_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_332: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_229: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_331, 3, 0, 64)
        slice_230: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_331, 3, 64, 256);  permute_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_231: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_332, 3, 0, 64)
        slice_232: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_332, 3, 64, 256);  permute_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_363, [1, 128, 1, 32, 2]);  unsqueeze_363 = None
        clone_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
        view_765: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_217, [1, 128, 1, 64]);  clone_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_301: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_229, view_765)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_809: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_301, [1, 128, 16, 32, 2]);  mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_809, -1, 0)
        select_1: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_809, -1, 1);  view_809 = None
        neg_58: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select);  select = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        full_default_8: "f32[1, 128, 16, 64]" = torch.ops.aten.full.default([1, 128, 16, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_58, 3, 1, 9223372036854775807, 2);  neg_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_1: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_1, 3, 0, 9223372036854775807, 2);  select_1 = None
        add_260: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_218: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_365, [1, 128, 1, 32, 2]);  unsqueeze_365 = None
        clone_218: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_218, memory_format = torch.contiguous_format);  expand_218 = None
        view_766: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_218, [1, 128, 1, 64]);  clone_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_302: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_229, view_766);  slice_229 = None
        add_261: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_260, mul_302);  add_260 = mul_302 = None
        mul_303: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_231, view_765);  view_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_810: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_303, [1, 128, 16, 32, 2]);  mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_2: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_810, -1, 0)
        select_3: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_810, -1, 1);  view_810 = None
        neg_59: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_2);  select_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_2: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_59, 3, 1, 9223372036854775807, 2);  neg_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_3: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_3, 3, 0, 9223372036854775807, 2);  select_3 = None
        add_262: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_304: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_231, view_766);  slice_231 = view_766 = None
        add_263: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_262, mul_304);  add_262 = mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        full_default_12: "f32[1, 128, 16, 256]" = torch.ops.aten.full.default([1, 128, 16, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_4: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_230, 3, 64, 9223372036854775807);  slice_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_5: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_261, 3, 0, 64);  add_261 = None
        add_264: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_6: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_232, 3, 64, 9223372036854775807);  slice_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_7: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_263, 3, 0, 64);  add_263 = None
        add_265: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_6, slice_scatter_7);  slice_scatter_6 = slice_scatter_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_333: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_225: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
        view_811: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_225, [1, 128, 4096]);  clone_225 = None
        view_812: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_265, [1, 128, 4096]);  add_265 = None
        view_813: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_264, [1, 128, 4096]);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_814: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_811, [128, 4096]);  view_811 = None
        permute_334: "f32[4096, 128]" = torch.ops.aten.permute.default(view_814, [1, 0])
        mm_120: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_334, view_756);  permute_334 = None
        permute_299: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_304, [1, 0]);  primals_304 = None
        permute_336: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
        mm_121: "f32[128, 4096]" = torch.ops.aten.mm.default(view_814, permute_336);  view_814 = permute_336 = None
        view_815: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_121, [1, 128, 4096]);  mm_121 = None
        add_266: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_799, view_815);  view_799 = view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_816: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_812, [128, 4096]);  view_812 = None
        permute_338: "f32[4096, 128]" = torch.ops.aten.permute.default(view_816, [1, 0])
        mm_122: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_338, view_756);  permute_338 = None
        permute_298: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
        permute_340: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
        mm_123: "f32[128, 4096]" = torch.ops.aten.mm.default(view_816, permute_340);  view_816 = permute_340 = None
        view_817: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_123, [1, 128, 4096]);  mm_123 = None
        add_267: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_266, view_817);  add_266 = view_817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_818: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_813, [128, 4096]);  view_813 = None
        permute_342: "f32[4096, 128]" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_124: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_342, view_756);  permute_342 = view_756 = None
        permute_297: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_302, [1, 0]);  primals_302 = None
        permute_344: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
        mm_125: "f32[128, 4096]" = torch.ops.aten.mm.default(view_818, permute_344);  view_818 = permute_344 = None
        view_819: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_125, [1, 128, 4096]);  mm_125 = None
        add_268: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_267, view_819);  add_267 = view_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_306: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_268, primals_300);  primals_300 = None
        mul_307: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_306, 4096)
        sum_41: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [2], True)
        mul_308: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_306, mul_270);  mul_306 = None
        sum_42: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_308, [2], True);  mul_308 = None
        mul_309: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_270, sum_42);  sum_42 = None
        sub_67: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_307, sum_41);  mul_307 = sum_41 = None
        sub_68: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_67, mul_309);  sub_67 = mul_309 = None
        mul_310: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_60, sub_68);  div_60 = sub_68 = None
        mul_311: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_268, mul_270);  mul_270 = None
        sum_43: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 1]);  mul_311 = None
        sum_44: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_268, [0, 1]);  add_268 = None
        add_269: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_289, mul_310);  mul_289 = mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_820: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_269, [128, 4096])
        permute_296: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_298, [1, 0]);  primals_298 = None
        permute_346: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
        mm_126: "f32[128, 16384]" = torch.ops.aten.mm.default(view_820, permute_346);  permute_346 = None
        permute_347: "f32[4096, 128]" = torch.ops.aten.permute.default(view_820, [1, 0])
        mm_127: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_347, view_754);  view_754 = None
        sum_45: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_820, [0], True)
        view_821: "f32[4096]" = torch.ops.aten.reshape.default(sum_45, [4096]);  sum_45 = None
        view_822: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_126, [1, 128, 16384]);  mm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_753: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 16384]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_266: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_753, 0.5)
        mul_312: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_822, mul_266);  mul_266 = None
        pow_27: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_753, 3.0)
        mul_267: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_242: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_753, mul_267);  mul_267 = None
        mul_268: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_242, 0.7978845608028654);  add_242 = None
        tanh_26: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_268);  mul_268 = None
        add_243: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_26, 1.0)
        mul_313: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_822, add_243);  view_822 = add_243 = None
        mul_314: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_26, tanh_26);  tanh_26 = None
        sub_69: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_314);  mul_314 = None
        mul_315: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_312, sub_69);  mul_312 = sub_69 = None
        mul_316: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_315, 0.7978845608028654);  mul_315 = None
        mul_317: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_316, 0.044715)
        pow_30: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_753, 2.0);  view_753 = None
        mul_318: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_319: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
        add_270: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_316, mul_319);  mul_316 = mul_319 = None
        mul_320: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_313, 0.5);  mul_313 = None
        add_271: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_270, mul_320);  add_270 = mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_823: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_271, [128, 16384]);  add_271 = None
        permute_295: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_296, [1, 0]);  primals_296 = None
        permute_350: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
        mm_128: "f32[128, 4096]" = torch.ops.aten.mm.default(view_823, permute_350);  permute_350 = None
        permute_351: "f32[16384, 128]" = torch.ops.aten.permute.default(view_823, [1, 0])
        mm_129: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_351, view_728);  permute_351 = None
        sum_46: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_823, [0], True);  view_823 = None
        view_824: "f32[16384]" = torch.ops.aten.reshape.default(sum_46, [16384]);  sum_46 = None
        view_825: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_128, [1, 128, 4096]);  mm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_130: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_347, view_750);  permute_347 = view_750 = None
        permute_294: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
        permute_356: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
        mm_131: "f32[128, 4096]" = torch.ops.aten.mm.default(view_820, permute_356);  view_820 = permute_356 = None
        view_827: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_131, [1, 128, 4096]);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_828: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_827, [1, 128, 16, 256]);  view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_358: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_828, [0, 2, 1, 3]);  view_828 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_358, permute_291, permute_290, permute_289, expand_default_28, getitem_121, getitem_122, getitem_123, getitem_124, 0.0, [True, True, True, False], scale = 0.0625);  permute_358 = permute_291 = permute_290 = permute_289 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_125: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_1[0]
        getitem_126: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_1[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_127: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_1[2];  _scaled_dot_product_efficient_attention_backward_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_364: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_125, [0, 2, 1, 3]);  getitem_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_365: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_233: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_364, 3, 0, 64)
        slice_234: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_364, 3, 64, 256);  permute_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_235: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_365, 3, 0, 64)
        slice_236: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_365, 3, 64, 256);  permute_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_350, [1, 128, 1, 32, 2]);  unsqueeze_350 = None
        clone_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
        view_737: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_209, [1, 128, 1, 64]);  clone_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_322: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_233, view_737)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_835: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_322, [1, 128, 16, 32, 2]);  mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_4: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_835, -1, 0)
        select_5: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_835, -1, 1);  view_835 = None
        neg_61: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_4);  select_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_8: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_61, 3, 1, 9223372036854775807, 2);  neg_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_9: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_5, 3, 0, 9223372036854775807, 2);  select_5 = None
        add_272: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_9);  slice_scatter_8 = slice_scatter_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_210: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_352, [1, 128, 1, 32, 2]);  unsqueeze_352 = None
        clone_210: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
        view_738: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_210, [1, 128, 1, 64]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_323: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_233, view_738);  slice_233 = None
        add_273: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_272, mul_323);  add_272 = mul_323 = None
        mul_324: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_235, view_737);  view_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_836: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_324, [1, 128, 16, 32, 2]);  mul_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_6: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_836, -1, 0)
        select_7: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_836, -1, 1);  view_836 = None
        neg_62: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_6);  select_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_10: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_62, 3, 1, 9223372036854775807, 2);  neg_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_11: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_7, 3, 0, 9223372036854775807, 2);  select_7 = None
        add_274: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_11);  slice_scatter_10 = slice_scatter_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_325: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_235, view_738);  slice_235 = view_738 = None
        add_275: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_274, mul_325);  add_274 = mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_12: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_234, 3, 64, 9223372036854775807);  slice_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_13: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_273, 3, 0, 64);  add_273 = None
        add_276: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_12, slice_scatter_13);  slice_scatter_12 = slice_scatter_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_14: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_236, 3, 64, 9223372036854775807);  slice_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_15: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_275, 3, 0, 64);  add_275 = None
        add_277: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_14, slice_scatter_15);  slice_scatter_14 = slice_scatter_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_366: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_127, [0, 2, 1, 3]);  getitem_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_226: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_366, memory_format = torch.contiguous_format);  permute_366 = None
        view_837: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_226, [1, 128, 4096]);  clone_226 = None
        view_838: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_277, [1, 128, 4096]);  add_277 = None
        view_839: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_276, [1, 128, 4096]);  add_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_840: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_837, [128, 4096]);  view_837 = None
        permute_367: "f32[4096, 128]" = torch.ops.aten.permute.default(view_840, [1, 0])
        mm_132: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_367, view_728);  permute_367 = None
        permute_288: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
        permute_369: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
        mm_133: "f32[128, 4096]" = torch.ops.aten.mm.default(view_840, permute_369);  view_840 = permute_369 = None
        view_841: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_133, [1, 128, 4096]);  mm_133 = None
        add_278: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_825, view_841);  view_825 = view_841 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_842: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_838, [128, 4096]);  view_838 = None
        permute_371: "f32[4096, 128]" = torch.ops.aten.permute.default(view_842, [1, 0])
        mm_134: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_371, view_728);  permute_371 = None
        permute_287: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_292, [1, 0]);  primals_292 = None
        permute_373: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
        mm_135: "f32[128, 4096]" = torch.ops.aten.mm.default(view_842, permute_373);  view_842 = permute_373 = None
        view_843: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_135, [1, 128, 4096]);  mm_135 = None
        add_279: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_278, view_843);  add_278 = view_843 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_844: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_839, [128, 4096]);  view_839 = None
        permute_375: "f32[4096, 128]" = torch.ops.aten.permute.default(view_844, [1, 0])
        mm_136: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_375, view_728);  permute_375 = view_728 = None
        permute_286: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
        permute_377: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
        mm_137: "f32[128, 4096]" = torch.ops.aten.mm.default(view_844, permute_377);  view_844 = permute_377 = None
        view_845: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_137, [1, 128, 4096]);  mm_137 = None
        add_280: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_279, view_845);  add_279 = view_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_327: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_280, primals_289);  primals_289 = None
        mul_328: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_327, 4096)
        sum_48: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
        mul_329: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_327, mul_260);  mul_327 = None
        sum_49: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
        mul_330: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_260, sum_49);  sum_49 = None
        sub_71: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_328, sum_48);  mul_328 = sum_48 = None
        sub_72: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_71, mul_330);  sub_71 = mul_330 = None
        mul_331: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_62, sub_72);  div_62 = sub_72 = None
        mul_332: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_280, mul_260);  mul_260 = None
        sum_50: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
        sum_51: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_280, [0, 1]);  add_280 = None
        add_281: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_269, mul_331);  add_269 = mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_846: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_281, [128, 4096])
        permute_285: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
        permute_379: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
        mm_138: "f32[128, 16384]" = torch.ops.aten.mm.default(view_846, permute_379);  permute_379 = None
        permute_380: "f32[4096, 128]" = torch.ops.aten.permute.default(view_846, [1, 0])
        mm_139: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_380, view_726);  view_726 = None
        sum_52: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True)
        view_847: "f32[4096]" = torch.ops.aten.reshape.default(sum_52, [4096]);  sum_52 = None
        view_848: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_138, [1, 128, 16384]);  mm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_725: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 16384]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_256: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_725, 0.5)
        mul_333: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_848, mul_256);  mul_256 = None
        pow_26: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_725, 3.0)
        mul_257: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_233: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_725, mul_257);  mul_257 = None
        mul_258: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_233, 0.7978845608028654);  add_233 = None
        tanh_25: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_258);  mul_258 = None
        add_234: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_25, 1.0)
        mul_334: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_848, add_234);  view_848 = add_234 = None
        mul_335: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_25, tanh_25);  tanh_25 = None
        sub_73: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_335);  mul_335 = None
        mul_336: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_333, sub_73);  mul_333 = sub_73 = None
        mul_337: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_336, 0.7978845608028654);  mul_336 = None
        mul_338: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_337, 0.044715)
        pow_31: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_725, 2.0);  view_725 = None
        mul_339: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_31, 3.0);  pow_31 = None
        mul_340: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
        add_282: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_337, mul_340);  mul_337 = mul_340 = None
        mul_341: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_334, 0.5);  mul_334 = None
        add_283: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_282, mul_341);  add_282 = mul_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_849: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_283, [128, 16384]);  add_283 = None
        permute_284: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
        permute_383: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
        mm_140: "f32[128, 4096]" = torch.ops.aten.mm.default(view_849, permute_383);  permute_383 = None
        permute_384: "f32[16384, 128]" = torch.ops.aten.permute.default(view_849, [1, 0])
        mm_141: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_384, view_700);  permute_384 = None
        sum_53: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
        view_850: "f32[16384]" = torch.ops.aten.reshape.default(sum_53, [16384]);  sum_53 = None
        view_851: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_140, [1, 128, 4096]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_142: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_380, view_722);  permute_380 = view_722 = None
        permute_283: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_284, [1, 0]);  primals_284 = None
        permute_389: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
        mm_143: "f32[128, 4096]" = torch.ops.aten.mm.default(view_846, permute_389);  view_846 = permute_389 = None
        view_853: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_143, [1, 128, 4096]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_854: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_853, [1, 128, 16, 256]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_391: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_391, permute_280, permute_279, permute_278, expand_default_28, getitem_128, getitem_129, getitem_130, getitem_131, 0.0, [True, True, True, False], scale = 0.0625);  permute_391 = permute_280 = permute_279 = permute_278 = getitem_128 = getitem_129 = getitem_130 = getitem_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_132: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_2[0]
        getitem_133: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_2[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_134: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_2[2];  _scaled_dot_product_efficient_attention_backward_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_397: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_398: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_237: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_397, 3, 0, 64)
        slice_238: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_397, 3, 64, 256);  permute_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_239: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_398, 3, 0, 64)
        slice_240: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_398, 3, 64, 256);  permute_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_337, [1, 128, 1, 32, 2]);  unsqueeze_337 = None
        clone_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
        view_709: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_201, [1, 128, 1, 64]);  clone_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_343: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_237, view_709)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_861: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_343, [1, 128, 16, 32, 2]);  mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_8: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_861, -1, 0)
        select_9: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_861, -1, 1);  view_861 = None
        neg_64: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_8);  select_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_16: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_64, 3, 1, 9223372036854775807, 2);  neg_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_17: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_9, 3, 0, 9223372036854775807, 2);  select_9 = None
        add_284: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_17);  slice_scatter_16 = slice_scatter_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_202: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_339, [1, 128, 1, 32, 2]);  unsqueeze_339 = None
        clone_202: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
        view_710: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_202, [1, 128, 1, 64]);  clone_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_344: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_237, view_710);  slice_237 = None
        add_285: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_284, mul_344);  add_284 = mul_344 = None
        mul_345: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_239, view_709);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_862: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_345, [1, 128, 16, 32, 2]);  mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_10: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_862, -1, 0)
        select_11: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_862, -1, 1);  view_862 = None
        neg_65: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_10);  select_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_18: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_65, 3, 1, 9223372036854775807, 2);  neg_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_19: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_11, 3, 0, 9223372036854775807, 2);  select_11 = None
        add_286: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_19);  slice_scatter_18 = slice_scatter_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_346: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_239, view_710);  slice_239 = view_710 = None
        add_287: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_286, mul_346);  add_286 = mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_20: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_238, 3, 64, 9223372036854775807);  slice_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_21: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_285, 3, 0, 64);  add_285 = None
        add_288: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_20, slice_scatter_21);  slice_scatter_20 = slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_22: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_240, 3, 64, 9223372036854775807);  slice_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_23: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_287, 3, 0, 64);  add_287 = None
        add_289: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_23);  slice_scatter_22 = slice_scatter_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_399: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_227: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_399, memory_format = torch.contiguous_format);  permute_399 = None
        view_863: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_227, [1, 128, 4096]);  clone_227 = None
        view_864: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_289, [1, 128, 4096]);  add_289 = None
        view_865: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_288, [1, 128, 4096]);  add_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_866: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_863, [128, 4096]);  view_863 = None
        permute_400: "f32[4096, 128]" = torch.ops.aten.permute.default(view_866, [1, 0])
        mm_144: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_400, view_700);  permute_400 = None
        permute_277: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_282, [1, 0]);  primals_282 = None
        permute_402: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
        mm_145: "f32[128, 4096]" = torch.ops.aten.mm.default(view_866, permute_402);  view_866 = permute_402 = None
        view_867: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_145, [1, 128, 4096]);  mm_145 = None
        add_290: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_851, view_867);  view_851 = view_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_868: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_864, [128, 4096]);  view_864 = None
        permute_404: "f32[4096, 128]" = torch.ops.aten.permute.default(view_868, [1, 0])
        mm_146: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_404, view_700);  permute_404 = None
        permute_276: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
        permute_406: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
        mm_147: "f32[128, 4096]" = torch.ops.aten.mm.default(view_868, permute_406);  view_868 = permute_406 = None
        view_869: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_147, [1, 128, 4096]);  mm_147 = None
        add_291: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_290, view_869);  add_290 = view_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_870: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_865, [128, 4096]);  view_865 = None
        permute_408: "f32[4096, 128]" = torch.ops.aten.permute.default(view_870, [1, 0])
        mm_148: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_408, view_700);  permute_408 = view_700 = None
        permute_275: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_280, [1, 0]);  primals_280 = None
        permute_410: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
        mm_149: "f32[128, 4096]" = torch.ops.aten.mm.default(view_870, permute_410);  view_870 = permute_410 = None
        view_871: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_149, [1, 128, 4096]);  mm_149 = None
        add_292: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_291, view_871);  add_291 = view_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_348: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_292, primals_278);  primals_278 = None
        mul_349: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_348, 4096)
        sum_55: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True)
        mul_350: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_348, mul_250);  mul_348 = None
        sum_56: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True);  mul_350 = None
        mul_351: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_250, sum_56);  sum_56 = None
        sub_75: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_349, sum_55);  mul_349 = sum_55 = None
        sub_76: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_75, mul_351);  sub_75 = mul_351 = None
        mul_352: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_64, sub_76);  div_64 = sub_76 = None
        mul_353: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_292, mul_250);  mul_250 = None
        sum_57: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_353, [0, 1]);  mul_353 = None
        sum_58: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_292, [0, 1]);  add_292 = None
        add_293: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_281, mul_352);  add_281 = mul_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_872: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_293, [128, 4096])
        permute_274: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_276, [1, 0]);  primals_276 = None
        permute_412: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
        mm_150: "f32[128, 16384]" = torch.ops.aten.mm.default(view_872, permute_412);  permute_412 = None
        permute_413: "f32[4096, 128]" = torch.ops.aten.permute.default(view_872, [1, 0])
        mm_151: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_413, view_698);  view_698 = None
        sum_59: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_872, [0], True)
        view_873: "f32[4096]" = torch.ops.aten.reshape.default(sum_59, [4096]);  sum_59 = None
        view_874: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_150, [1, 128, 16384]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_697: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 16384]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_246: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_697, 0.5)
        mul_354: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_874, mul_246);  mul_246 = None
        pow_25: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_697, 3.0)
        mul_247: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_224: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_697, mul_247);  mul_247 = None
        mul_248: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_224, 0.7978845608028654);  add_224 = None
        tanh_24: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_248);  mul_248 = None
        add_225: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_24, 1.0)
        mul_355: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_874, add_225);  view_874 = add_225 = None
        mul_356: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_24, tanh_24);  tanh_24 = None
        sub_77: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_356);  mul_356 = None
        mul_357: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_354, sub_77);  mul_354 = sub_77 = None
        mul_358: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_357, 0.7978845608028654);  mul_357 = None
        mul_359: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_358, 0.044715)
        pow_32: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_697, 2.0);  view_697 = None
        mul_360: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_361: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
        add_294: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_358, mul_361);  mul_358 = mul_361 = None
        mul_362: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_355, 0.5);  mul_355 = None
        add_295: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_294, mul_362);  add_294 = mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_875: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_295, [128, 16384]);  add_295 = None
        permute_273: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_274, [1, 0]);  primals_274 = None
        permute_416: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
        mm_152: "f32[128, 4096]" = torch.ops.aten.mm.default(view_875, permute_416);  permute_416 = None
        permute_417: "f32[16384, 128]" = torch.ops.aten.permute.default(view_875, [1, 0])
        mm_153: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_417, view_672);  permute_417 = None
        sum_60: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_875, [0], True);  view_875 = None
        view_876: "f32[16384]" = torch.ops.aten.reshape.default(sum_60, [16384]);  sum_60 = None
        view_877: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_152, [1, 128, 4096]);  mm_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_154: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_413, view_694);  permute_413 = view_694 = None
        permute_272: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
        permute_422: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
        mm_155: "f32[128, 4096]" = torch.ops.aten.mm.default(view_872, permute_422);  view_872 = permute_422 = None
        view_879: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_155, [1, 128, 4096]);  mm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_880: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_879, [1, 128, 16, 256]);  view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_424: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_880, [0, 2, 1, 3]);  view_880 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_424, permute_269, permute_268, permute_267, expand_default_28, getitem_135, getitem_136, getitem_137, getitem_138, 0.0, [True, True, True, False], scale = 0.0625);  permute_424 = permute_269 = permute_268 = permute_267 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_139: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_3[0]
        getitem_140: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_3[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_141: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_3[2];  _scaled_dot_product_efficient_attention_backward_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_430: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_431: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1, 3]);  getitem_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_241: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_430, 3, 0, 64)
        slice_242: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_430, 3, 64, 256);  permute_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_243: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_431, 3, 0, 64)
        slice_244: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_431, 3, 64, 256);  permute_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_324, [1, 128, 1, 32, 2]);  unsqueeze_324 = None
        clone_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
        view_681: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_193, [1, 128, 1, 64]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_364: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_241, view_681)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_887: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_364, [1, 128, 16, 32, 2]);  mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_12: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_887, -1, 0)
        select_13: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_887, -1, 1);  view_887 = None
        neg_67: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_12);  select_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_24: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_67, 3, 1, 9223372036854775807, 2);  neg_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_25: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_13, 3, 0, 9223372036854775807, 2);  select_13 = None
        add_296: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_24, slice_scatter_25);  slice_scatter_24 = slice_scatter_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_194: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_326, [1, 128, 1, 32, 2]);  unsqueeze_326 = None
        clone_194: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_194, memory_format = torch.contiguous_format);  expand_194 = None
        view_682: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_194, [1, 128, 1, 64]);  clone_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_365: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_241, view_682);  slice_241 = None
        add_297: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_296, mul_365);  add_296 = mul_365 = None
        mul_366: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_243, view_681);  view_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_888: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_366, [1, 128, 16, 32, 2]);  mul_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_14: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_888, -1, 0)
        select_15: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_888, -1, 1);  view_888 = None
        neg_68: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_14);  select_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_26: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_68, 3, 1, 9223372036854775807, 2);  neg_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_27: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_15, 3, 0, 9223372036854775807, 2);  select_15 = None
        add_298: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_26, slice_scatter_27);  slice_scatter_26 = slice_scatter_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_367: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_243, view_682);  slice_243 = view_682 = None
        add_299: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_298, mul_367);  add_298 = mul_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_28: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_242, 3, 64, 9223372036854775807);  slice_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_29: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_297, 3, 0, 64);  add_297 = None
        add_300: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_28, slice_scatter_29);  slice_scatter_28 = slice_scatter_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_30: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_244, 3, 64, 9223372036854775807);  slice_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_31: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_299, 3, 0, 64);  add_299 = None
        add_301: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_30, slice_scatter_31);  slice_scatter_30 = slice_scatter_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_432: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_141, [0, 2, 1, 3]);  getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_228: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_432, memory_format = torch.contiguous_format);  permute_432 = None
        view_889: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_228, [1, 128, 4096]);  clone_228 = None
        view_890: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_301, [1, 128, 4096]);  add_301 = None
        view_891: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_300, [1, 128, 4096]);  add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_892: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_889, [128, 4096]);  view_889 = None
        permute_433: "f32[4096, 128]" = torch.ops.aten.permute.default(view_892, [1, 0])
        mm_156: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_433, view_672);  permute_433 = None
        permute_266: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
        permute_435: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
        mm_157: "f32[128, 4096]" = torch.ops.aten.mm.default(view_892, permute_435);  view_892 = permute_435 = None
        view_893: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_157, [1, 128, 4096]);  mm_157 = None
        add_302: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_877, view_893);  view_877 = view_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_894: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_890, [128, 4096]);  view_890 = None
        permute_437: "f32[4096, 128]" = torch.ops.aten.permute.default(view_894, [1, 0])
        mm_158: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_437, view_672);  permute_437 = None
        permute_265: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_270, [1, 0]);  primals_270 = None
        permute_439: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        mm_159: "f32[128, 4096]" = torch.ops.aten.mm.default(view_894, permute_439);  view_894 = permute_439 = None
        view_895: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_159, [1, 128, 4096]);  mm_159 = None
        add_303: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_302, view_895);  add_302 = view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_896: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_891, [128, 4096]);  view_891 = None
        permute_441: "f32[4096, 128]" = torch.ops.aten.permute.default(view_896, [1, 0])
        mm_160: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_441, view_672);  permute_441 = view_672 = None
        permute_264: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
        permute_443: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        mm_161: "f32[128, 4096]" = torch.ops.aten.mm.default(view_896, permute_443);  view_896 = permute_443 = None
        view_897: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_161, [1, 128, 4096]);  mm_161 = None
        add_304: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_303, view_897);  add_303 = view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_369: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_304, primals_267);  primals_267 = None
        mul_370: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_369, 4096)
        sum_62: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
        mul_371: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_369, mul_240);  mul_369 = None
        sum_63: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
        mul_372: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_240, sum_63);  sum_63 = None
        sub_79: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_370, sum_62);  mul_370 = sum_62 = None
        sub_80: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_79, mul_372);  sub_79 = mul_372 = None
        mul_373: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_66, sub_80);  div_66 = sub_80 = None
        mul_374: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_304, mul_240);  mul_240 = None
        sum_64: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
        sum_65: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_304, [0, 1]);  add_304 = None
        add_305: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_293, mul_373);  add_293 = mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_898: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_305, [128, 4096])
        permute_263: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
        permute_445: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        mm_162: "f32[128, 16384]" = torch.ops.aten.mm.default(view_898, permute_445);  permute_445 = None
        permute_446: "f32[4096, 128]" = torch.ops.aten.permute.default(view_898, [1, 0])
        mm_163: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_446, view_670);  view_670 = None
        sum_66: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True)
        view_899: "f32[4096]" = torch.ops.aten.reshape.default(sum_66, [4096]);  sum_66 = None
        view_900: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_162, [1, 128, 16384]);  mm_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_669: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 16384]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_236: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_669, 0.5)
        mul_375: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_900, mul_236);  mul_236 = None
        pow_24: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_669, 3.0)
        mul_237: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_215: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_669, mul_237);  mul_237 = None
        mul_238: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_215, 0.7978845608028654);  add_215 = None
        tanh_23: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_216: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_376: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_900, add_216);  view_900 = add_216 = None
        mul_377: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_81: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_377);  mul_377 = None
        mul_378: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_375, sub_81);  mul_375 = sub_81 = None
        mul_379: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_378, 0.7978845608028654);  mul_378 = None
        mul_380: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_379, 0.044715)
        pow_33: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_669, 2.0);  view_669 = None
        mul_381: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_33, 3.0);  pow_33 = None
        mul_382: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
        add_306: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_379, mul_382);  mul_379 = mul_382 = None
        mul_383: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_376, 0.5);  mul_376 = None
        add_307: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_306, mul_383);  add_306 = mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_901: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_307, [128, 16384]);  add_307 = None
        permute_262: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
        permute_449: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        mm_164: "f32[128, 4096]" = torch.ops.aten.mm.default(view_901, permute_449);  permute_449 = None
        permute_450: "f32[16384, 128]" = torch.ops.aten.permute.default(view_901, [1, 0])
        mm_165: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_450, view_644);  permute_450 = None
        sum_67: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_901, [0], True);  view_901 = None
        view_902: "f32[16384]" = torch.ops.aten.reshape.default(sum_67, [16384]);  sum_67 = None
        view_903: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_164, [1, 128, 4096]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_166: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_446, view_666);  permute_446 = view_666 = None
        permute_261: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_262, [1, 0]);  primals_262 = None
        permute_455: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        mm_167: "f32[128, 4096]" = torch.ops.aten.mm.default(view_898, permute_455);  view_898 = permute_455 = None
        view_905: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_167, [1, 128, 4096]);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_906: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_905, [1, 128, 16, 256]);  view_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_457: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_906, [0, 2, 1, 3]);  view_906 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_457, permute_258, permute_257, permute_256, expand_default_28, getitem_142, getitem_143, getitem_144, getitem_145, 0.0, [True, True, True, False], scale = 0.0625);  permute_457 = permute_258 = permute_257 = permute_256 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_146: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_4[0]
        getitem_147: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_4[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_148: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_4[2];  _scaled_dot_product_efficient_attention_backward_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_463: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_464: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1, 3]);  getitem_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_245: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_463, 3, 0, 64)
        slice_246: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_463, 3, 64, 256);  permute_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_247: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_464, 3, 0, 64)
        slice_248: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_464, 3, 64, 256);  permute_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_311, [1, 128, 1, 32, 2]);  unsqueeze_311 = None
        clone_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
        view_653: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_185, [1, 128, 1, 64]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_385: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_245, view_653)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_913: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_385, [1, 128, 16, 32, 2]);  mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_16: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_913, -1, 0)
        select_17: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_913, -1, 1);  view_913 = None
        neg_70: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_16);  select_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_32: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_70, 3, 1, 9223372036854775807, 2);  neg_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_33: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_17, 3, 0, 9223372036854775807, 2);  select_17 = None
        add_308: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_33);  slice_scatter_32 = slice_scatter_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_186: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_313, [1, 128, 1, 32, 2]);  unsqueeze_313 = None
        clone_186: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
        view_654: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1, 64]);  clone_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_386: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_245, view_654);  slice_245 = None
        add_309: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_308, mul_386);  add_308 = mul_386 = None
        mul_387: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_247, view_653);  view_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_914: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_387, [1, 128, 16, 32, 2]);  mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_18: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_914, -1, 0)
        select_19: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_914, -1, 1);  view_914 = None
        neg_71: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_18);  select_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_34: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_71, 3, 1, 9223372036854775807, 2);  neg_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_35: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_19, 3, 0, 9223372036854775807, 2);  select_19 = None
        add_310: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_34, slice_scatter_35);  slice_scatter_34 = slice_scatter_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_388: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_247, view_654);  slice_247 = view_654 = None
        add_311: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_310, mul_388);  add_310 = mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_36: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_246, 3, 64, 9223372036854775807);  slice_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_37: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_309, 3, 0, 64);  add_309 = None
        add_312: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_36, slice_scatter_37);  slice_scatter_36 = slice_scatter_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_38: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_248, 3, 64, 9223372036854775807);  slice_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_39: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_311, 3, 0, 64);  add_311 = None
        add_313: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_38, slice_scatter_39);  slice_scatter_38 = slice_scatter_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_465: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_229: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_465, memory_format = torch.contiguous_format);  permute_465 = None
        view_915: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_229, [1, 128, 4096]);  clone_229 = None
        view_916: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_313, [1, 128, 4096]);  add_313 = None
        view_917: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_312, [1, 128, 4096]);  add_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_918: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_915, [128, 4096]);  view_915 = None
        permute_466: "f32[4096, 128]" = torch.ops.aten.permute.default(view_918, [1, 0])
        mm_168: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_466, view_644);  permute_466 = None
        permute_255: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_260, [1, 0]);  primals_260 = None
        permute_468: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        mm_169: "f32[128, 4096]" = torch.ops.aten.mm.default(view_918, permute_468);  view_918 = permute_468 = None
        view_919: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_169, [1, 128, 4096]);  mm_169 = None
        add_314: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_903, view_919);  view_903 = view_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_920: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_916, [128, 4096]);  view_916 = None
        permute_470: "f32[4096, 128]" = torch.ops.aten.permute.default(view_920, [1, 0])
        mm_170: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_470, view_644);  permute_470 = None
        permute_254: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
        permute_472: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        mm_171: "f32[128, 4096]" = torch.ops.aten.mm.default(view_920, permute_472);  view_920 = permute_472 = None
        view_921: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_171, [1, 128, 4096]);  mm_171 = None
        add_315: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_314, view_921);  add_314 = view_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_922: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_917, [128, 4096]);  view_917 = None
        permute_474: "f32[4096, 128]" = torch.ops.aten.permute.default(view_922, [1, 0])
        mm_172: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_474, view_644);  permute_474 = view_644 = None
        permute_253: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_258, [1, 0]);  primals_258 = None
        permute_476: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        mm_173: "f32[128, 4096]" = torch.ops.aten.mm.default(view_922, permute_476);  view_922 = permute_476 = None
        view_923: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_173, [1, 128, 4096]);  mm_173 = None
        add_316: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_315, view_923);  add_315 = view_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_390: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_316, primals_256);  primals_256 = None
        mul_391: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_390, 4096)
        sum_69: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True)
        mul_392: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_390, mul_230);  mul_390 = None
        sum_70: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True);  mul_392 = None
        mul_393: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_230, sum_70);  sum_70 = None
        sub_83: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_391, sum_69);  mul_391 = sum_69 = None
        sub_84: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_83, mul_393);  sub_83 = mul_393 = None
        mul_394: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_68, sub_84);  div_68 = sub_84 = None
        mul_395: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_316, mul_230);  mul_230 = None
        sum_71: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1]);  mul_395 = None
        sum_72: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_316, [0, 1]);  add_316 = None
        add_317: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_305, mul_394);  add_305 = mul_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_924: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_317, [128, 4096])
        permute_252: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_254, [1, 0]);  primals_254 = None
        permute_478: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        mm_174: "f32[128, 16384]" = torch.ops.aten.mm.default(view_924, permute_478);  permute_478 = None
        permute_479: "f32[4096, 128]" = torch.ops.aten.permute.default(view_924, [1, 0])
        mm_175: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_479, view_642);  view_642 = None
        sum_73: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_924, [0], True)
        view_925: "f32[4096]" = torch.ops.aten.reshape.default(sum_73, [4096]);  sum_73 = None
        view_926: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_174, [1, 128, 16384]);  mm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_641: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 16384]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_226: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_641, 0.5)
        mul_396: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_926, mul_226);  mul_226 = None
        pow_23: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_641, 3.0)
        mul_227: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_206: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_641, mul_227);  mul_227 = None
        mul_228: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_206, 0.7978845608028654);  add_206 = None
        tanh_22: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_228);  mul_228 = None
        add_207: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_397: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_926, add_207);  view_926 = add_207 = None
        mul_398: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_85: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_398);  mul_398 = None
        mul_399: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_396, sub_85);  mul_396 = sub_85 = None
        mul_400: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_399, 0.7978845608028654);  mul_399 = None
        mul_401: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_400, 0.044715)
        pow_34: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_641, 2.0);  view_641 = None
        mul_402: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_403: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
        add_318: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_400, mul_403);  mul_400 = mul_403 = None
        mul_404: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_397, 0.5);  mul_397 = None
        add_319: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_318, mul_404);  add_318 = mul_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_927: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_319, [128, 16384]);  add_319 = None
        permute_251: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_252, [1, 0]);  primals_252 = None
        permute_482: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        mm_176: "f32[128, 4096]" = torch.ops.aten.mm.default(view_927, permute_482);  permute_482 = None
        permute_483: "f32[16384, 128]" = torch.ops.aten.permute.default(view_927, [1, 0])
        mm_177: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_483, view_616);  permute_483 = None
        sum_74: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_927, [0], True);  view_927 = None
        view_928: "f32[16384]" = torch.ops.aten.reshape.default(sum_74, [16384]);  sum_74 = None
        view_929: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_176, [1, 128, 4096]);  mm_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_178: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_479, view_638);  permute_479 = view_638 = None
        permute_250: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
        permute_488: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        mm_179: "f32[128, 4096]" = torch.ops.aten.mm.default(view_924, permute_488);  view_924 = permute_488 = None
        view_931: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_179, [1, 128, 4096]);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_932: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_931, [1, 128, 16, 256]);  view_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_490: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_932, [0, 2, 1, 3]);  view_932 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_490, permute_247, permute_246, permute_245, expand_default_28, getitem_149, getitem_150, getitem_151, getitem_152, 0.0, [True, True, True, False], scale = 0.0625);  permute_490 = permute_247 = permute_246 = permute_245 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_153: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_5[0]
        getitem_154: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_5[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_155: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_5[2];  _scaled_dot_product_efficient_attention_backward_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_496: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_497: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_154, [0, 2, 1, 3]);  getitem_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_249: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_496, 3, 0, 64)
        slice_250: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_496, 3, 64, 256);  permute_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_251: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_497, 3, 0, 64)
        slice_252: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_497, 3, 64, 256);  permute_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_298, [1, 128, 1, 32, 2]);  unsqueeze_298 = None
        clone_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
        view_625: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_177, [1, 128, 1, 64]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_406: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_249, view_625)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_939: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_406, [1, 128, 16, 32, 2]);  mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_20: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_939, -1, 0)
        select_21: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_939, -1, 1);  view_939 = None
        neg_73: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_20);  select_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_40: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_73, 3, 1, 9223372036854775807, 2);  neg_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_41: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_21, 3, 0, 9223372036854775807, 2);  select_21 = None
        add_320: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_40, slice_scatter_41);  slice_scatter_40 = slice_scatter_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_178: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_300, [1, 128, 1, 32, 2]);  unsqueeze_300 = None
        clone_178: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
        view_626: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1, 64]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_407: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_249, view_626);  slice_249 = None
        add_321: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_320, mul_407);  add_320 = mul_407 = None
        mul_408: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_251, view_625);  view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_940: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_408, [1, 128, 16, 32, 2]);  mul_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_22: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_940, -1, 0)
        select_23: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_940, -1, 1);  view_940 = None
        neg_74: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_22);  select_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_42: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_74, 3, 1, 9223372036854775807, 2);  neg_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_43: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_23, 3, 0, 9223372036854775807, 2);  select_23 = None
        add_322: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_42, slice_scatter_43);  slice_scatter_42 = slice_scatter_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_409: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_251, view_626);  slice_251 = view_626 = None
        add_323: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_322, mul_409);  add_322 = mul_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_44: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_250, 3, 64, 9223372036854775807);  slice_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_45: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_321, 3, 0, 64);  add_321 = None
        add_324: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_44, slice_scatter_45);  slice_scatter_44 = slice_scatter_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_46: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_252, 3, 64, 9223372036854775807);  slice_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_47: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_323, 3, 0, 64);  add_323 = None
        add_325: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_46, slice_scatter_47);  slice_scatter_46 = slice_scatter_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_498: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_230: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_498, memory_format = torch.contiguous_format);  permute_498 = None
        view_941: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_230, [1, 128, 4096]);  clone_230 = None
        view_942: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_325, [1, 128, 4096]);  add_325 = None
        view_943: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_324, [1, 128, 4096]);  add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_944: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_941, [128, 4096]);  view_941 = None
        permute_499: "f32[4096, 128]" = torch.ops.aten.permute.default(view_944, [1, 0])
        mm_180: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_499, view_616);  permute_499 = None
        permute_244: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
        permute_501: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        mm_181: "f32[128, 4096]" = torch.ops.aten.mm.default(view_944, permute_501);  view_944 = permute_501 = None
        view_945: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_181, [1, 128, 4096]);  mm_181 = None
        add_326: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_929, view_945);  view_929 = view_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_946: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_942, [128, 4096]);  view_942 = None
        permute_503: "f32[4096, 128]" = torch.ops.aten.permute.default(view_946, [1, 0])
        mm_182: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_503, view_616);  permute_503 = None
        permute_243: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_248, [1, 0]);  primals_248 = None
        permute_505: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        mm_183: "f32[128, 4096]" = torch.ops.aten.mm.default(view_946, permute_505);  view_946 = permute_505 = None
        view_947: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_183, [1, 128, 4096]);  mm_183 = None
        add_327: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_326, view_947);  add_326 = view_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_948: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_943, [128, 4096]);  view_943 = None
        permute_507: "f32[4096, 128]" = torch.ops.aten.permute.default(view_948, [1, 0])
        mm_184: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_507, view_616);  permute_507 = view_616 = None
        permute_242: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
        permute_509: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        mm_185: "f32[128, 4096]" = torch.ops.aten.mm.default(view_948, permute_509);  view_948 = permute_509 = None
        view_949: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_185, [1, 128, 4096]);  mm_185 = None
        add_328: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_327, view_949);  add_327 = view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_411: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_328, primals_245);  primals_245 = None
        mul_412: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_411, 4096)
        sum_76: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
        mul_413: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_411, mul_220);  mul_411 = None
        sum_77: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
        mul_414: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_220, sum_77);  sum_77 = None
        sub_87: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_412, sum_76);  mul_412 = sum_76 = None
        sub_88: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_87, mul_414);  sub_87 = mul_414 = None
        mul_415: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_70, sub_88);  div_70 = sub_88 = None
        mul_416: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_328, mul_220);  mul_220 = None
        sum_78: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
        sum_79: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_328, [0, 1]);  add_328 = None
        add_329: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_317, mul_415);  add_317 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_950: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_329, [128, 4096])
        permute_241: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_243, [1, 0]);  primals_243 = None
        permute_511: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        mm_186: "f32[128, 16384]" = torch.ops.aten.mm.default(view_950, permute_511);  permute_511 = None
        permute_512: "f32[4096, 128]" = torch.ops.aten.permute.default(view_950, [1, 0])
        mm_187: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_512, view_614);  view_614 = None
        sum_80: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_950, [0], True)
        view_951: "f32[4096]" = torch.ops.aten.reshape.default(sum_80, [4096]);  sum_80 = None
        view_952: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_186, [1, 128, 16384]);  mm_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_613: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 16384]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_216: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_613, 0.5)
        mul_417: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_952, mul_216);  mul_216 = None
        pow_22: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_613, 3.0)
        mul_217: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_197: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_613, mul_217);  mul_217 = None
        mul_218: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_197, 0.7978845608028654);  add_197 = None
        tanh_21: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
        add_198: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_418: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_952, add_198);  view_952 = add_198 = None
        mul_419: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_89: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_419);  mul_419 = None
        mul_420: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_417, sub_89);  mul_417 = sub_89 = None
        mul_421: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_420, 0.7978845608028654);  mul_420 = None
        mul_422: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_421, 0.044715)
        pow_35: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_613, 2.0);  view_613 = None
        mul_423: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_35, 3.0);  pow_35 = None
        mul_424: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
        add_330: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_421, mul_424);  mul_421 = mul_424 = None
        mul_425: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_418, 0.5);  mul_418 = None
        add_331: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_330, mul_425);  add_330 = mul_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_953: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_331, [128, 16384]);  add_331 = None
        permute_240: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
        permute_515: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        mm_188: "f32[128, 4096]" = torch.ops.aten.mm.default(view_953, permute_515);  permute_515 = None
        permute_516: "f32[16384, 128]" = torch.ops.aten.permute.default(view_953, [1, 0])
        mm_189: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_516, view_588);  permute_516 = None
        sum_81: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_953, [0], True);  view_953 = None
        view_954: "f32[16384]" = torch.ops.aten.reshape.default(sum_81, [16384]);  sum_81 = None
        view_955: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_188, [1, 128, 4096]);  mm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_190: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_512, view_610);  permute_512 = view_610 = None
        permute_239: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_240, [1, 0]);  primals_240 = None
        permute_521: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        mm_191: "f32[128, 4096]" = torch.ops.aten.mm.default(view_950, permute_521);  view_950 = permute_521 = None
        view_957: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_191, [1, 128, 4096]);  mm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_958: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_957, [1, 128, 16, 256]);  view_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_523: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_958, [0, 2, 1, 3]);  view_958 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_523, permute_236, permute_235, permute_234, expand_default_28, getitem_156, getitem_157, getitem_158, getitem_159, 0.0, [True, True, True, False], scale = 0.0625);  permute_523 = permute_236 = permute_235 = permute_234 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_160: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_6[0]
        getitem_161: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_6[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_162: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_6[2];  _scaled_dot_product_efficient_attention_backward_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_529: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_530: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_253: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_529, 3, 0, 64)
        slice_254: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_529, 3, 64, 256);  permute_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_255: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_530, 3, 0, 64)
        slice_256: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_530, 3, 64, 256);  permute_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_285, [1, 128, 1, 32, 2]);  unsqueeze_285 = None
        clone_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
        view_597: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_169, [1, 128, 1, 64]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_427: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_253, view_597)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_965: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_427, [1, 128, 16, 32, 2]);  mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_24: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_965, -1, 0)
        select_25: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_965, -1, 1);  view_965 = None
        neg_76: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_24);  select_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_48: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_76, 3, 1, 9223372036854775807, 2);  neg_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_49: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_25, 3, 0, 9223372036854775807, 2);  select_25 = None
        add_332: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_48, slice_scatter_49);  slice_scatter_48 = slice_scatter_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_170: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_287, [1, 128, 1, 32, 2]);  unsqueeze_287 = None
        clone_170: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
        view_598: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_170, [1, 128, 1, 64]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_428: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_253, view_598);  slice_253 = None
        add_333: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_332, mul_428);  add_332 = mul_428 = None
        mul_429: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_255, view_597);  view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_966: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_429, [1, 128, 16, 32, 2]);  mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_26: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_966, -1, 0)
        select_27: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_966, -1, 1);  view_966 = None
        neg_77: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_26);  select_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_50: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_77, 3, 1, 9223372036854775807, 2);  neg_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_51: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_27, 3, 0, 9223372036854775807, 2);  select_27 = None
        add_334: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_50, slice_scatter_51);  slice_scatter_50 = slice_scatter_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_430: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_255, view_598);  slice_255 = view_598 = None
        add_335: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_334, mul_430);  add_334 = mul_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_52: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_254, 3, 64, 9223372036854775807);  slice_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_53: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_333, 3, 0, 64);  add_333 = None
        add_336: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_52, slice_scatter_53);  slice_scatter_52 = slice_scatter_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_54: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_256, 3, 64, 9223372036854775807);  slice_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_55: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_335, 3, 0, 64);  add_335 = None
        add_337: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_54, slice_scatter_55);  slice_scatter_54 = slice_scatter_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_531: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_231: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_531, memory_format = torch.contiguous_format);  permute_531 = None
        view_967: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_231, [1, 128, 4096]);  clone_231 = None
        view_968: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_337, [1, 128, 4096]);  add_337 = None
        view_969: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_336, [1, 128, 4096]);  add_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_970: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_967, [128, 4096]);  view_967 = None
        permute_532: "f32[4096, 128]" = torch.ops.aten.permute.default(view_970, [1, 0])
        mm_192: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_532, view_588);  permute_532 = None
        permute_233: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_238, [1, 0]);  primals_238 = None
        permute_534: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        mm_193: "f32[128, 4096]" = torch.ops.aten.mm.default(view_970, permute_534);  view_970 = permute_534 = None
        view_971: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_193, [1, 128, 4096]);  mm_193 = None
        add_338: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_955, view_971);  view_955 = view_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_972: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_968, [128, 4096]);  view_968 = None
        permute_536: "f32[4096, 128]" = torch.ops.aten.permute.default(view_972, [1, 0])
        mm_194: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_536, view_588);  permute_536 = None
        permute_232: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
        permute_538: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        mm_195: "f32[128, 4096]" = torch.ops.aten.mm.default(view_972, permute_538);  view_972 = permute_538 = None
        view_973: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_195, [1, 128, 4096]);  mm_195 = None
        add_339: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_338, view_973);  add_338 = view_973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_974: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_969, [128, 4096]);  view_969 = None
        permute_540: "f32[4096, 128]" = torch.ops.aten.permute.default(view_974, [1, 0])
        mm_196: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_540, view_588);  permute_540 = view_588 = None
        permute_231: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
        permute_542: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        mm_197: "f32[128, 4096]" = torch.ops.aten.mm.default(view_974, permute_542);  view_974 = permute_542 = None
        view_975: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_197, [1, 128, 4096]);  mm_197 = None
        add_340: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_339, view_975);  add_339 = view_975 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_432: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_340, primals_234);  primals_234 = None
        mul_433: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_432, 4096)
        sum_83: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
        mul_434: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_432, mul_210);  mul_432 = None
        sum_84: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
        mul_435: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_210, sum_84);  sum_84 = None
        sub_91: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_433, sum_83);  mul_433 = sum_83 = None
        sub_92: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_91, mul_435);  sub_91 = mul_435 = None
        mul_436: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_72, sub_92);  div_72 = sub_92 = None
        mul_437: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_340, mul_210);  mul_210 = None
        sum_85: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 1]);  mul_437 = None
        sum_86: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_340, [0, 1]);  add_340 = None
        add_341: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_329, mul_436);  add_329 = mul_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_976: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_341, [128, 4096])
        permute_230: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_232, [1, 0]);  primals_232 = None
        permute_544: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        mm_198: "f32[128, 16384]" = torch.ops.aten.mm.default(view_976, permute_544);  permute_544 = None
        permute_545: "f32[4096, 128]" = torch.ops.aten.permute.default(view_976, [1, 0])
        mm_199: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_545, view_586);  view_586 = None
        sum_87: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True)
        view_977: "f32[4096]" = torch.ops.aten.reshape.default(sum_87, [4096]);  sum_87 = None
        view_978: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_198, [1, 128, 16384]);  mm_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_585: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 16384]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_206: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_585, 0.5)
        mul_438: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_978, mul_206);  mul_206 = None
        pow_21: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_585, 3.0)
        mul_207: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_188: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_585, mul_207);  mul_207 = None
        mul_208: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_188, 0.7978845608028654);  add_188 = None
        tanh_20: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_208);  mul_208 = None
        add_189: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_439: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_978, add_189);  view_978 = add_189 = None
        mul_440: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_93: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_440);  mul_440 = None
        mul_441: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_438, sub_93);  mul_438 = sub_93 = None
        mul_442: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_441, 0.7978845608028654);  mul_441 = None
        mul_443: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_442, 0.044715)
        pow_36: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_585, 2.0);  view_585 = None
        mul_444: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_445: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None
        add_342: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_442, mul_445);  mul_442 = mul_445 = None
        mul_446: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_439, 0.5);  mul_439 = None
        add_343: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_342, mul_446);  add_342 = mul_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_979: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_343, [128, 16384]);  add_343 = None
        permute_229: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_230, [1, 0]);  primals_230 = None
        permute_548: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        mm_200: "f32[128, 4096]" = torch.ops.aten.mm.default(view_979, permute_548);  permute_548 = None
        permute_549: "f32[16384, 128]" = torch.ops.aten.permute.default(view_979, [1, 0])
        mm_201: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_549, view_560);  permute_549 = None
        sum_88: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_979, [0], True);  view_979 = None
        view_980: "f32[16384]" = torch.ops.aten.reshape.default(sum_88, [16384]);  sum_88 = None
        view_981: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_200, [1, 128, 4096]);  mm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_202: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_545, view_582);  permute_545 = view_582 = None
        permute_228: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
        permute_554: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        mm_203: "f32[128, 4096]" = torch.ops.aten.mm.default(view_976, permute_554);  view_976 = permute_554 = None
        view_983: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_203, [1, 128, 4096]);  mm_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_984: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_983, [1, 128, 16, 256]);  view_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_556: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_984, [0, 2, 1, 3]);  view_984 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_556, permute_225, permute_224, permute_223, expand_default_28, getitem_163, getitem_164, getitem_165, getitem_166, 0.0, [True, True, True, False], scale = 0.0625);  permute_556 = permute_225 = permute_224 = permute_223 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_167: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_7[0]
        getitem_168: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_7[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_169: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_7[2];  _scaled_dot_product_efficient_attention_backward_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_562: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_563: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_257: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_562, 3, 0, 64)
        slice_258: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_562, 3, 64, 256);  permute_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_259: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_563, 3, 0, 64)
        slice_260: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_563, 3, 64, 256);  permute_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_272, [1, 128, 1, 32, 2]);  unsqueeze_272 = None
        clone_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
        view_569: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_161, [1, 128, 1, 64]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_448: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_257, view_569)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_991: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_448, [1, 128, 16, 32, 2]);  mul_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_28: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_991, -1, 0)
        select_29: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_991, -1, 1);  view_991 = None
        neg_79: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_28);  select_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_56: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_79, 3, 1, 9223372036854775807, 2);  neg_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_57: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_29, 3, 0, 9223372036854775807, 2);  select_29 = None
        add_344: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_56, slice_scatter_57);  slice_scatter_56 = slice_scatter_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_162: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_274, [1, 128, 1, 32, 2]);  unsqueeze_274 = None
        clone_162: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
        view_570: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_162, [1, 128, 1, 64]);  clone_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_449: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_257, view_570);  slice_257 = None
        add_345: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_344, mul_449);  add_344 = mul_449 = None
        mul_450: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_259, view_569);  view_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_992: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_450, [1, 128, 16, 32, 2]);  mul_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_30: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_992, -1, 0)
        select_31: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_992, -1, 1);  view_992 = None
        neg_80: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_30);  select_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_58: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_80, 3, 1, 9223372036854775807, 2);  neg_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_59: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_31, 3, 0, 9223372036854775807, 2);  select_31 = None
        add_346: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_58, slice_scatter_59);  slice_scatter_58 = slice_scatter_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_451: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_259, view_570);  slice_259 = view_570 = None
        add_347: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_346, mul_451);  add_346 = mul_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_60: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_258, 3, 64, 9223372036854775807);  slice_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_61: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_345, 3, 0, 64);  add_345 = None
        add_348: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_60, slice_scatter_61);  slice_scatter_60 = slice_scatter_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_62: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_260, 3, 64, 9223372036854775807);  slice_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_63: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_347, 3, 0, 64);  add_347 = None
        add_349: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_62, slice_scatter_63);  slice_scatter_62 = slice_scatter_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_564: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_169, [0, 2, 1, 3]);  getitem_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_232: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_564, memory_format = torch.contiguous_format);  permute_564 = None
        view_993: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_232, [1, 128, 4096]);  clone_232 = None
        view_994: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_349, [1, 128, 4096]);  add_349 = None
        view_995: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_348, [1, 128, 4096]);  add_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_996: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_993, [128, 4096]);  view_993 = None
        permute_565: "f32[4096, 128]" = torch.ops.aten.permute.default(view_996, [1, 0])
        mm_204: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_565, view_560);  permute_565 = None
        permute_222: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
        permute_567: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        mm_205: "f32[128, 4096]" = torch.ops.aten.mm.default(view_996, permute_567);  view_996 = permute_567 = None
        view_997: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_205, [1, 128, 4096]);  mm_205 = None
        add_350: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_981, view_997);  view_981 = view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_998: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_994, [128, 4096]);  view_994 = None
        permute_569: "f32[4096, 128]" = torch.ops.aten.permute.default(view_998, [1, 0])
        mm_206: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_569, view_560);  permute_569 = None
        permute_221: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_226, [1, 0]);  primals_226 = None
        permute_571: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        mm_207: "f32[128, 4096]" = torch.ops.aten.mm.default(view_998, permute_571);  view_998 = permute_571 = None
        view_999: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_207, [1, 128, 4096]);  mm_207 = None
        add_351: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_350, view_999);  add_350 = view_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1000: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_995, [128, 4096]);  view_995 = None
        permute_573: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1000, [1, 0])
        mm_208: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_573, view_560);  permute_573 = view_560 = None
        permute_220: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
        permute_575: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        mm_209: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1000, permute_575);  view_1000 = permute_575 = None
        view_1001: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_209, [1, 128, 4096]);  mm_209 = None
        add_352: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_351, view_1001);  add_351 = view_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_453: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_352, primals_223);  primals_223 = None
        mul_454: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_453, 4096)
        sum_90: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True)
        mul_455: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_453, mul_200);  mul_453 = None
        sum_91: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
        mul_456: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_200, sum_91);  sum_91 = None
        sub_95: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_454, sum_90);  mul_454 = sum_90 = None
        sub_96: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_95, mul_456);  sub_95 = mul_456 = None
        mul_457: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_74, sub_96);  div_74 = sub_96 = None
        mul_458: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_352, mul_200);  mul_200 = None
        sum_92: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
        sum_93: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_352, [0, 1]);  add_352 = None
        add_353: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_341, mul_457);  add_341 = mul_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1002: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_353, [128, 4096])
        permute_219: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
        permute_577: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        mm_210: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1002, permute_577);  permute_577 = None
        permute_578: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1002, [1, 0])
        mm_211: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_578, view_558);  view_558 = None
        sum_94: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1002, [0], True)
        view_1003: "f32[4096]" = torch.ops.aten.reshape.default(sum_94, [4096]);  sum_94 = None
        view_1004: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_210, [1, 128, 16384]);  mm_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_557: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 16384]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_196: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_557, 0.5)
        mul_459: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1004, mul_196);  mul_196 = None
        pow_20: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_557, 3.0)
        mul_197: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_179: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_557, mul_197);  mul_197 = None
        mul_198: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_179, 0.7978845608028654);  add_179 = None
        tanh_19: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
        add_180: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_460: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1004, add_180);  view_1004 = add_180 = None
        mul_461: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_97: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_461);  mul_461 = None
        mul_462: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_459, sub_97);  mul_459 = sub_97 = None
        mul_463: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_462, 0.7978845608028654);  mul_462 = None
        mul_464: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_463, 0.044715)
        pow_37: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_557, 2.0);  view_557 = None
        mul_465: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
        mul_466: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
        add_354: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_463, mul_466);  mul_463 = mul_466 = None
        mul_467: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_460, 0.5);  mul_460 = None
        add_355: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_354, mul_467);  add_354 = mul_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1005: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_355, [128, 16384]);  add_355 = None
        permute_218: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
        permute_581: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        mm_212: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1005, permute_581);  permute_581 = None
        permute_582: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1005, [1, 0])
        mm_213: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_582, view_532);  permute_582 = None
        sum_95: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1005, [0], True);  view_1005 = None
        view_1006: "f32[16384]" = torch.ops.aten.reshape.default(sum_95, [16384]);  sum_95 = None
        view_1007: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_212, [1, 128, 4096]);  mm_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_214: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_578, view_554);  permute_578 = view_554 = None
        permute_217: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_218, [1, 0]);  primals_218 = None
        permute_587: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        mm_215: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1002, permute_587);  view_1002 = permute_587 = None
        view_1009: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_215, [1, 128, 4096]);  mm_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1010: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1009, [1, 128, 16, 256]);  view_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_589: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1010, [0, 2, 1, 3]);  view_1010 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_589, permute_214, permute_213, permute_212, expand_default_28, getitem_170, getitem_171, getitem_172, getitem_173, 0.0, [True, True, True, False], scale = 0.0625);  permute_589 = permute_214 = permute_213 = permute_212 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_174: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_8[0]
        getitem_175: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_8[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_176: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_8[2];  _scaled_dot_product_efficient_attention_backward_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_595: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_596: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_175, [0, 2, 1, 3]);  getitem_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_261: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_595, 3, 0, 64)
        slice_262: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_595, 3, 64, 256);  permute_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_263: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_596, 3, 0, 64)
        slice_264: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_596, 3, 64, 256);  permute_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_259, [1, 128, 1, 32, 2]);  unsqueeze_259 = None
        clone_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
        view_541: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_153, [1, 128, 1, 64]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_469: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_261, view_541)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1017: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_469, [1, 128, 16, 32, 2]);  mul_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_32: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1017, -1, 0)
        select_33: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1017, -1, 1);  view_1017 = None
        neg_82: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_32);  select_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_64: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_82, 3, 1, 9223372036854775807, 2);  neg_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_65: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_33, 3, 0, 9223372036854775807, 2);  select_33 = None
        add_356: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_64, slice_scatter_65);  slice_scatter_64 = slice_scatter_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_154: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_261, [1, 128, 1, 32, 2]);  unsqueeze_261 = None
        clone_154: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
        view_542: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_154, [1, 128, 1, 64]);  clone_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_470: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_261, view_542);  slice_261 = None
        add_357: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_356, mul_470);  add_356 = mul_470 = None
        mul_471: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_263, view_541);  view_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1018: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_471, [1, 128, 16, 32, 2]);  mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_34: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1018, -1, 0)
        select_35: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1018, -1, 1);  view_1018 = None
        neg_83: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_34);  select_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_66: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_83, 3, 1, 9223372036854775807, 2);  neg_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_67: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_35, 3, 0, 9223372036854775807, 2);  select_35 = None
        add_358: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_66, slice_scatter_67);  slice_scatter_66 = slice_scatter_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_472: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_263, view_542);  slice_263 = view_542 = None
        add_359: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_358, mul_472);  add_358 = mul_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_68: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_262, 3, 64, 9223372036854775807);  slice_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_69: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_357, 3, 0, 64);  add_357 = None
        add_360: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_68, slice_scatter_69);  slice_scatter_68 = slice_scatter_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_70: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_264, 3, 64, 9223372036854775807);  slice_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_71: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_359, 3, 0, 64);  add_359 = None
        add_361: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_70, slice_scatter_71);  slice_scatter_70 = slice_scatter_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_597: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_233: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_597, memory_format = torch.contiguous_format);  permute_597 = None
        view_1019: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_233, [1, 128, 4096]);  clone_233 = None
        view_1020: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_361, [1, 128, 4096]);  add_361 = None
        view_1021: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_360, [1, 128, 4096]);  add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1022: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1019, [128, 4096]);  view_1019 = None
        permute_598: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1022, [1, 0])
        mm_216: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_598, view_532);  permute_598 = None
        permute_211: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_216, [1, 0]);  primals_216 = None
        permute_600: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        mm_217: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1022, permute_600);  view_1022 = permute_600 = None
        view_1023: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_217, [1, 128, 4096]);  mm_217 = None
        add_362: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1007, view_1023);  view_1007 = view_1023 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1024: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1020, [128, 4096]);  view_1020 = None
        permute_602: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1024, [1, 0])
        mm_218: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_602, view_532);  permute_602 = None
        permute_210: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
        permute_604: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
        mm_219: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1024, permute_604);  view_1024 = permute_604 = None
        view_1025: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_219, [1, 128, 4096]);  mm_219 = None
        add_363: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_362, view_1025);  add_362 = view_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1026: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1021, [128, 4096]);  view_1021 = None
        permute_606: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1026, [1, 0])
        mm_220: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_606, view_532);  permute_606 = view_532 = None
        permute_209: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
        permute_608: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        mm_221: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1026, permute_608);  view_1026 = permute_608 = None
        view_1027: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_221, [1, 128, 4096]);  mm_221 = None
        add_364: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_363, view_1027);  add_363 = view_1027 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_474: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_364, primals_212);  primals_212 = None
        mul_475: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_474, 4096)
        sum_97: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [2], True)
        mul_476: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_474, mul_190);  mul_474 = None
        sum_98: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True);  mul_476 = None
        mul_477: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, sum_98);  sum_98 = None
        sub_99: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_475, sum_97);  mul_475 = sum_97 = None
        sub_100: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_99, mul_477);  sub_99 = mul_477 = None
        mul_478: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_76, sub_100);  div_76 = sub_100 = None
        mul_479: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_364, mul_190);  mul_190 = None
        sum_99: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1]);  mul_479 = None
        sum_100: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_364, [0, 1]);  add_364 = None
        add_365: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_353, mul_478);  add_353 = mul_478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1028: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_365, [128, 4096])
        permute_208: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_210, [1, 0]);  primals_210 = None
        permute_610: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        mm_222: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1028, permute_610);  permute_610 = None
        permute_611: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1028, [1, 0])
        mm_223: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_611, view_530);  view_530 = None
        sum_101: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1028, [0], True)
        view_1029: "f32[4096]" = torch.ops.aten.reshape.default(sum_101, [4096]);  sum_101 = None
        view_1030: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_222, [1, 128, 16384]);  mm_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_529: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 16384]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_186: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_480: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1030, mul_186);  mul_186 = None
        pow_19: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_529, 3.0)
        mul_187: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_170: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_529, mul_187);  mul_187 = None
        mul_188: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_170, 0.7978845608028654);  add_170 = None
        tanh_18: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
        add_171: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_481: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1030, add_171);  view_1030 = add_171 = None
        mul_482: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_101: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_482);  mul_482 = None
        mul_483: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_480, sub_101);  mul_480 = sub_101 = None
        mul_484: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_483, 0.7978845608028654);  mul_483 = None
        mul_485: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_484, 0.044715)
        pow_38: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_529, 2.0);  view_529 = None
        mul_486: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_487: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_485, mul_486);  mul_485 = mul_486 = None
        add_366: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_484, mul_487);  mul_484 = mul_487 = None
        mul_488: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_481, 0.5);  mul_481 = None
        add_367: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_366, mul_488);  add_366 = mul_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1031: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_367, [128, 16384]);  add_367 = None
        permute_207: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_208, [1, 0]);  primals_208 = None
        permute_614: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        mm_224: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1031, permute_614);  permute_614 = None
        permute_615: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1031, [1, 0])
        mm_225: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_615, view_504);  permute_615 = None
        sum_102: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1031, [0], True);  view_1031 = None
        view_1032: "f32[16384]" = torch.ops.aten.reshape.default(sum_102, [16384]);  sum_102 = None
        view_1033: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_224, [1, 128, 4096]);  mm_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_226: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_611, view_526);  permute_611 = view_526 = None
        permute_206: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
        permute_620: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        mm_227: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1028, permute_620);  view_1028 = permute_620 = None
        view_1035: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_227, [1, 128, 4096]);  mm_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1036: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1035, [1, 128, 16, 256]);  view_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_622: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1036, [0, 2, 1, 3]);  view_1036 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_622, permute_203, permute_202, permute_201, expand_default_28, getitem_177, getitem_178, getitem_179, getitem_180, 0.0, [True, True, True, False], scale = 0.0625);  permute_622 = permute_203 = permute_202 = permute_201 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_181: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_9[0]
        getitem_182: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_9[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_183: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_9[2];  _scaled_dot_product_efficient_attention_backward_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_628: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_629: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_182, [0, 2, 1, 3]);  getitem_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_265: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_628, 3, 0, 64)
        slice_266: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_628, 3, 64, 256);  permute_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_267: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_629, 3, 0, 64)
        slice_268: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_629, 3, 64, 256);  permute_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_246, [1, 128, 1, 32, 2]);  unsqueeze_246 = None
        clone_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
        view_513: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_145, [1, 128, 1, 64]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_490: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_265, view_513)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1043: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_490, [1, 128, 16, 32, 2]);  mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_36: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1043, -1, 0)
        select_37: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1043, -1, 1);  view_1043 = None
        neg_85: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_36);  select_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_72: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_85, 3, 1, 9223372036854775807, 2);  neg_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_73: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_37, 3, 0, 9223372036854775807, 2);  select_37 = None
        add_368: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_72, slice_scatter_73);  slice_scatter_72 = slice_scatter_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_146: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_248, [1, 128, 1, 32, 2]);  unsqueeze_248 = None
        clone_146: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
        view_514: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_146, [1, 128, 1, 64]);  clone_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_491: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_265, view_514);  slice_265 = None
        add_369: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_368, mul_491);  add_368 = mul_491 = None
        mul_492: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_267, view_513);  view_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1044: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_492, [1, 128, 16, 32, 2]);  mul_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_38: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1044, -1, 0)
        select_39: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1044, -1, 1);  view_1044 = None
        neg_86: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_38);  select_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_74: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_86, 3, 1, 9223372036854775807, 2);  neg_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_75: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_39, 3, 0, 9223372036854775807, 2);  select_39 = None
        add_370: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_74, slice_scatter_75);  slice_scatter_74 = slice_scatter_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_493: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_267, view_514);  slice_267 = view_514 = None
        add_371: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_370, mul_493);  add_370 = mul_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_76: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_266, 3, 64, 9223372036854775807);  slice_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_77: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_369, 3, 0, 64);  add_369 = None
        add_372: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_76, slice_scatter_77);  slice_scatter_76 = slice_scatter_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_78: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_268, 3, 64, 9223372036854775807);  slice_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_79: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_371, 3, 0, 64);  add_371 = None
        add_373: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_78, slice_scatter_79);  slice_scatter_78 = slice_scatter_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_630: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_234: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_630, memory_format = torch.contiguous_format);  permute_630 = None
        view_1045: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_234, [1, 128, 4096]);  clone_234 = None
        view_1046: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_373, [1, 128, 4096]);  add_373 = None
        view_1047: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_372, [1, 128, 4096]);  add_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1048: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1045, [128, 4096]);  view_1045 = None
        permute_631: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1048, [1, 0])
        mm_228: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_631, view_504);  permute_631 = None
        permute_200: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
        permute_633: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        mm_229: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1048, permute_633);  view_1048 = permute_633 = None
        view_1049: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_229, [1, 128, 4096]);  mm_229 = None
        add_374: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1033, view_1049);  view_1033 = view_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1050: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1046, [128, 4096]);  view_1046 = None
        permute_635: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1050, [1, 0])
        mm_230: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_635, view_504);  permute_635 = None
        permute_199: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_204, [1, 0]);  primals_204 = None
        permute_637: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        mm_231: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1050, permute_637);  view_1050 = permute_637 = None
        view_1051: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_231, [1, 128, 4096]);  mm_231 = None
        add_375: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_374, view_1051);  add_374 = view_1051 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1052: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1047, [128, 4096]);  view_1047 = None
        permute_639: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1052, [1, 0])
        mm_232: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_639, view_504);  permute_639 = view_504 = None
        permute_198: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
        permute_641: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        mm_233: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1052, permute_641);  view_1052 = permute_641 = None
        view_1053: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_233, [1, 128, 4096]);  mm_233 = None
        add_376: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_375, view_1053);  add_375 = view_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_495: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_376, primals_201);  primals_201 = None
        mul_496: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_495, 4096)
        sum_104: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
        mul_497: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_495, mul_180);  mul_495 = None
        sum_105: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
        mul_498: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_180, sum_105);  sum_105 = None
        sub_103: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_496, sum_104);  mul_496 = sum_104 = None
        sub_104: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_103, mul_498);  sub_103 = mul_498 = None
        mul_499: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_78, sub_104);  div_78 = sub_104 = None
        mul_500: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_376, mul_180);  mul_180 = None
        sum_106: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
        sum_107: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_376, [0, 1]);  add_376 = None
        add_377: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_365, mul_499);  add_365 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1054: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_377, [128, 4096])
        permute_197: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
        permute_643: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        mm_234: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1054, permute_643);  permute_643 = None
        permute_644: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1054, [1, 0])
        mm_235: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_644, view_502);  view_502 = None
        sum_108: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1054, [0], True)
        view_1055: "f32[4096]" = torch.ops.aten.reshape.default(sum_108, [4096]);  sum_108 = None
        view_1056: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_234, [1, 128, 16384]);  mm_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_501: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 16384]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_176: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        mul_501: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1056, mul_176);  mul_176 = None
        pow_18: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_501, 3.0)
        mul_177: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_161: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_501, mul_177);  mul_177 = None
        mul_178: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_17: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
        add_162: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_502: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1056, add_162);  view_1056 = add_162 = None
        mul_503: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_105: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_503);  mul_503 = None
        mul_504: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_501, sub_105);  mul_501 = sub_105 = None
        mul_505: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_504, 0.7978845608028654);  mul_504 = None
        mul_506: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_505, 0.044715)
        pow_39: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_501, 2.0);  view_501 = None
        mul_507: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
        mul_508: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_506, mul_507);  mul_506 = mul_507 = None
        add_378: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_505, mul_508);  mul_505 = mul_508 = None
        mul_509: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_502, 0.5);  mul_502 = None
        add_379: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_378, mul_509);  add_378 = mul_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1057: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_379, [128, 16384]);  add_379 = None
        permute_196: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
        permute_647: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        mm_236: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1057, permute_647);  permute_647 = None
        permute_648: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1057, [1, 0])
        mm_237: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_648, view_476);  permute_648 = None
        sum_109: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1057, [0], True);  view_1057 = None
        view_1058: "f32[16384]" = torch.ops.aten.reshape.default(sum_109, [16384]);  sum_109 = None
        view_1059: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_236, [1, 128, 4096]);  mm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_238: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_644, view_498);  permute_644 = view_498 = None
        permute_195: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
        permute_653: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        mm_239: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1054, permute_653);  view_1054 = permute_653 = None
        view_1061: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_239, [1, 128, 4096]);  mm_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1062: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1061, [1, 128, 16, 256]);  view_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_655: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1062, [0, 2, 1, 3]);  view_1062 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_655, permute_192, permute_191, permute_190, expand_default_28, getitem_184, getitem_185, getitem_186, getitem_187, 0.0, [True, True, True, False], scale = 0.0625);  permute_655 = permute_192 = permute_191 = permute_190 = getitem_184 = getitem_185 = getitem_186 = getitem_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_188: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_10[0]
        getitem_189: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_10[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_190: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_10[2];  _scaled_dot_product_efficient_attention_backward_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_661: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3]);  getitem_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_662: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_269: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_661, 3, 0, 64)
        slice_270: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_661, 3, 64, 256);  permute_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_271: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_662, 3, 0, 64)
        slice_272: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_662, 3, 64, 256);  permute_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_233, [1, 128, 1, 32, 2]);  unsqueeze_233 = None
        clone_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
        view_485: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_137, [1, 128, 1, 64]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_511: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_269, view_485)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1069: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_511, [1, 128, 16, 32, 2]);  mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_40: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1069, -1, 0)
        select_41: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1069, -1, 1);  view_1069 = None
        neg_88: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_40);  select_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_80: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_88, 3, 1, 9223372036854775807, 2);  neg_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_81: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_41, 3, 0, 9223372036854775807, 2);  select_41 = None
        add_380: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_80, slice_scatter_81);  slice_scatter_80 = slice_scatter_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_138: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_235, [1, 128, 1, 32, 2]);  unsqueeze_235 = None
        clone_138: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
        view_486: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_138, [1, 128, 1, 64]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_512: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_269, view_486);  slice_269 = None
        add_381: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_380, mul_512);  add_380 = mul_512 = None
        mul_513: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_271, view_485);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1070: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_513, [1, 128, 16, 32, 2]);  mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_42: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1070, -1, 0)
        select_43: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1070, -1, 1);  view_1070 = None
        neg_89: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_42);  select_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_82: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_89, 3, 1, 9223372036854775807, 2);  neg_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_83: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_43, 3, 0, 9223372036854775807, 2);  select_43 = None
        add_382: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_82, slice_scatter_83);  slice_scatter_82 = slice_scatter_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_514: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_271, view_486);  slice_271 = view_486 = None
        add_383: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_382, mul_514);  add_382 = mul_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_84: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_270, 3, 64, 9223372036854775807);  slice_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_85: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_381, 3, 0, 64);  add_381 = None
        add_384: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_84, slice_scatter_85);  slice_scatter_84 = slice_scatter_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_86: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_272, 3, 64, 9223372036854775807);  slice_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_87: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_383, 3, 0, 64);  add_383 = None
        add_385: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_86, slice_scatter_87);  slice_scatter_86 = slice_scatter_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_663: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_190, [0, 2, 1, 3]);  getitem_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_235: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_663, memory_format = torch.contiguous_format);  permute_663 = None
        view_1071: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_235, [1, 128, 4096]);  clone_235 = None
        view_1072: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_385, [1, 128, 4096]);  add_385 = None
        view_1073: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_384, [1, 128, 4096]);  add_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1074: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1071, [128, 4096]);  view_1071 = None
        permute_664: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1074, [1, 0])
        mm_240: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_664, view_476);  permute_664 = None
        permute_189: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        permute_666: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        mm_241: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1074, permute_666);  view_1074 = permute_666 = None
        view_1075: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_241, [1, 128, 4096]);  mm_241 = None
        add_386: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1059, view_1075);  view_1059 = view_1075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1076: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1072, [128, 4096]);  view_1072 = None
        permute_668: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1076, [1, 0])
        mm_242: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_668, view_476);  permute_668 = None
        permute_188: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
        permute_670: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        mm_243: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1076, permute_670);  view_1076 = permute_670 = None
        view_1077: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_243, [1, 128, 4096]);  mm_243 = None
        add_387: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_386, view_1077);  add_386 = view_1077 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1078: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1073, [128, 4096]);  view_1073 = None
        permute_672: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1078, [1, 0])
        mm_244: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_672, view_476);  permute_672 = view_476 = None
        permute_187: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
        permute_674: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        mm_245: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1078, permute_674);  view_1078 = permute_674 = None
        view_1079: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_245, [1, 128, 4096]);  mm_245 = None
        add_388: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_387, view_1079);  add_387 = view_1079 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_516: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_388, primals_190);  primals_190 = None
        mul_517: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_516, 4096)
        sum_111: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_516, [2], True)
        mul_518: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_516, mul_170);  mul_516 = None
        sum_112: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True);  mul_518 = None
        mul_519: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_170, sum_112);  sum_112 = None
        sub_107: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_517, sum_111);  mul_517 = sum_111 = None
        sub_108: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_107, mul_519);  sub_107 = mul_519 = None
        mul_520: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_80, sub_108);  div_80 = sub_108 = None
        mul_521: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_388, mul_170);  mul_170 = None
        sum_113: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 1]);  mul_521 = None
        sum_114: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_388, [0, 1]);  add_388 = None
        add_389: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_377, mul_520);  add_377 = mul_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1080: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_389, [128, 4096])
        permute_186: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        permute_676: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        mm_246: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1080, permute_676);  permute_676 = None
        permute_677: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1080, [1, 0])
        mm_247: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_677, view_474);  view_474 = None
        sum_115: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1080, [0], True)
        view_1081: "f32[4096]" = torch.ops.aten.reshape.default(sum_115, [4096]);  sum_115 = None
        view_1082: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_246, [1, 128, 16384]);  mm_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_473: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 16384]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_166: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_473, 0.5)
        mul_522: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1082, mul_166);  mul_166 = None
        pow_17: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_473, 3.0)
        mul_167: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_152: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_473, mul_167);  mul_167 = None
        mul_168: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_152, 0.7978845608028654);  add_152 = None
        tanh_16: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_168);  mul_168 = None
        add_153: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_523: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1082, add_153);  view_1082 = add_153 = None
        mul_524: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_109: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_524);  mul_524 = None
        mul_525: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_522, sub_109);  mul_522 = sub_109 = None
        mul_526: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_525, 0.7978845608028654);  mul_525 = None
        mul_527: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_526, 0.044715)
        pow_40: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_473, 2.0);  view_473 = None
        mul_528: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_529: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_527, mul_528);  mul_527 = mul_528 = None
        add_390: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_526, mul_529);  mul_526 = mul_529 = None
        mul_530: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_523, 0.5);  mul_523 = None
        add_391: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_390, mul_530);  add_390 = mul_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1083: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_391, [128, 16384]);  add_391 = None
        permute_185: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
        permute_680: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        mm_248: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1083, permute_680);  permute_680 = None
        permute_681: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1083, [1, 0])
        mm_249: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_681, view_448);  permute_681 = None
        sum_116: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1083, [0], True);  view_1083 = None
        view_1084: "f32[16384]" = torch.ops.aten.reshape.default(sum_116, [16384]);  sum_116 = None
        view_1085: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_248, [1, 128, 4096]);  mm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_250: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_677, view_470);  permute_677 = view_470 = None
        permute_184: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        permute_686: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        mm_251: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1080, permute_686);  view_1080 = permute_686 = None
        view_1087: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_251, [1, 128, 4096]);  mm_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1088: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1087, [1, 128, 16, 256]);  view_1087 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_688: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1088, [0, 2, 1, 3]);  view_1088 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_688, permute_181, permute_180, permute_179, expand_default_28, getitem_191, getitem_192, getitem_193, getitem_194, 0.0, [True, True, True, False], scale = 0.0625);  permute_688 = permute_181 = permute_180 = permute_179 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_195: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_11[0]
        getitem_196: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_11[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_197: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_11[2];  _scaled_dot_product_efficient_attention_backward_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_694: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_695: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_196, [0, 2, 1, 3]);  getitem_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_273: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_694, 3, 0, 64)
        slice_274: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_694, 3, 64, 256);  permute_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_275: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_695, 3, 0, 64)
        slice_276: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_695, 3, 64, 256);  permute_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_220, [1, 128, 1, 32, 2]);  unsqueeze_220 = None
        clone_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
        view_457: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_129, [1, 128, 1, 64]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_532: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_273, view_457)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1095: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_532, [1, 128, 16, 32, 2]);  mul_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_44: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1095, -1, 0)
        select_45: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1095, -1, 1);  view_1095 = None
        neg_91: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_44);  select_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_88: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_91, 3, 1, 9223372036854775807, 2);  neg_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_89: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_45, 3, 0, 9223372036854775807, 2);  select_45 = None
        add_392: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_88, slice_scatter_89);  slice_scatter_88 = slice_scatter_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_130: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_222, [1, 128, 1, 32, 2]);  unsqueeze_222 = None
        clone_130: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_458: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1, 64]);  clone_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_533: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_273, view_458);  slice_273 = None
        add_393: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_392, mul_533);  add_392 = mul_533 = None
        mul_534: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_275, view_457);  view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1096: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_534, [1, 128, 16, 32, 2]);  mul_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_46: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1096, -1, 0)
        select_47: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1096, -1, 1);  view_1096 = None
        neg_92: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_46);  select_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_90: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_92, 3, 1, 9223372036854775807, 2);  neg_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_91: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_47, 3, 0, 9223372036854775807, 2);  select_47 = None
        add_394: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_90, slice_scatter_91);  slice_scatter_90 = slice_scatter_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_535: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_275, view_458);  slice_275 = view_458 = None
        add_395: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_394, mul_535);  add_394 = mul_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_92: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_274, 3, 64, 9223372036854775807);  slice_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_93: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_393, 3, 0, 64);  add_393 = None
        add_396: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_92, slice_scatter_93);  slice_scatter_92 = slice_scatter_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_94: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_276, 3, 64, 9223372036854775807);  slice_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_95: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_395, 3, 0, 64);  add_395 = None
        add_397: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_94, slice_scatter_95);  slice_scatter_94 = slice_scatter_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_696: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_236: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_696, memory_format = torch.contiguous_format);  permute_696 = None
        view_1097: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_236, [1, 128, 4096]);  clone_236 = None
        view_1098: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_397, [1, 128, 4096]);  add_397 = None
        view_1099: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_396, [1, 128, 4096]);  add_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1100: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1097, [128, 4096]);  view_1097 = None
        permute_697: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1100, [1, 0])
        mm_252: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_697, view_448);  permute_697 = None
        permute_178: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
        permute_699: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        mm_253: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1100, permute_699);  view_1100 = permute_699 = None
        view_1101: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_253, [1, 128, 4096]);  mm_253 = None
        add_398: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1085, view_1101);  view_1085 = view_1101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1102: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1098, [128, 4096]);  view_1098 = None
        permute_701: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1102, [1, 0])
        mm_254: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_701, view_448);  permute_701 = None
        permute_177: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        permute_703: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        mm_255: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1102, permute_703);  view_1102 = permute_703 = None
        view_1103: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_255, [1, 128, 4096]);  mm_255 = None
        add_399: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_398, view_1103);  add_398 = view_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1104: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1099, [128, 4096]);  view_1099 = None
        permute_705: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1104, [1, 0])
        mm_256: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_705, view_448);  permute_705 = view_448 = None
        permute_176: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
        permute_707: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_257: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1104, permute_707);  view_1104 = permute_707 = None
        view_1105: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_257, [1, 128, 4096]);  mm_257 = None
        add_400: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_399, view_1105);  add_399 = view_1105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_537: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_400, primals_179);  primals_179 = None
        mul_538: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_537, 4096)
        sum_118: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [2], True)
        mul_539: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_537, mul_160);  mul_537 = None
        sum_119: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_539, [2], True);  mul_539 = None
        mul_540: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_160, sum_119);  sum_119 = None
        sub_111: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_538, sum_118);  mul_538 = sum_118 = None
        sub_112: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_111, mul_540);  sub_111 = mul_540 = None
        mul_541: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_82, sub_112);  div_82 = sub_112 = None
        mul_542: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_400, mul_160);  mul_160 = None
        sum_120: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 1]);  mul_542 = None
        sum_121: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_400, [0, 1]);  add_400 = None
        add_401: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_389, mul_541);  add_389 = mul_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1106: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_401, [128, 4096])
        permute_175: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
        permute_709: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        mm_258: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1106, permute_709);  permute_709 = None
        permute_710: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1106, [1, 0])
        mm_259: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_710, view_446);  view_446 = None
        sum_122: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True)
        view_1107: "f32[4096]" = torch.ops.aten.reshape.default(sum_122, [4096]);  sum_122 = None
        view_1108: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_258, [1, 128, 16384]);  mm_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_445: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 16384]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_156: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_445, 0.5)
        mul_543: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1108, mul_156);  mul_156 = None
        pow_16: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_445, 3.0)
        mul_157: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_143: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_445, mul_157);  mul_157 = None
        mul_158: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_143, 0.7978845608028654);  add_143 = None
        tanh_15: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
        add_144: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_544: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1108, add_144);  view_1108 = add_144 = None
        mul_545: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_113: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_545);  mul_545 = None
        mul_546: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_543, sub_113);  mul_543 = sub_113 = None
        mul_547: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_546, 0.7978845608028654);  mul_546 = None
        mul_548: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_547, 0.044715)
        pow_41: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_445, 2.0);  view_445 = None
        mul_549: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
        mul_550: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_548, mul_549);  mul_548 = mul_549 = None
        add_402: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_547, mul_550);  mul_547 = mul_550 = None
        mul_551: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_544, 0.5);  mul_544 = None
        add_403: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_402, mul_551);  add_402 = mul_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1109: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_403, [128, 16384]);  add_403 = None
        permute_174: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        permute_713: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_260: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1109, permute_713);  permute_713 = None
        permute_714: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1109, [1, 0])
        mm_261: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_714, view_420);  permute_714 = None
        sum_123: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1109, [0], True);  view_1109 = None
        view_1110: "f32[16384]" = torch.ops.aten.reshape.default(sum_123, [16384]);  sum_123 = None
        view_1111: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_260, [1, 128, 4096]);  mm_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_262: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_710, view_442);  permute_710 = view_442 = None
        permute_173: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        permute_719: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_263: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1106, permute_719);  view_1106 = permute_719 = None
        view_1113: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_263, [1, 128, 4096]);  mm_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1114: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1113, [1, 128, 16, 256]);  view_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_721: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1114, [0, 2, 1, 3]);  view_1114 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_721, permute_170, permute_169, permute_168, expand_default_28, getitem_198, getitem_199, getitem_200, getitem_201, 0.0, [True, True, True, False], scale = 0.0625);  permute_721 = permute_170 = permute_169 = permute_168 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_202: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_12[0]
        getitem_203: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_12[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_204: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_12[2];  _scaled_dot_product_efficient_attention_backward_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_727: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_728: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3]);  getitem_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_277: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_727, 3, 0, 64)
        slice_278: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_727, 3, 64, 256);  permute_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_279: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_728, 3, 0, 64)
        slice_280: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_728, 3, 64, 256);  permute_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_207, [1, 128, 1, 32, 2]);  unsqueeze_207 = None
        clone_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
        view_429: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_121, [1, 128, 1, 64]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_553: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_277, view_429)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1121: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_553, [1, 128, 16, 32, 2]);  mul_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_48: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1121, -1, 0)
        select_49: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1121, -1, 1);  view_1121 = None
        neg_94: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_48);  select_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_96: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_94, 3, 1, 9223372036854775807, 2);  neg_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_97: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_49, 3, 0, 9223372036854775807, 2);  select_49 = None
        add_404: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_96, slice_scatter_97);  slice_scatter_96 = slice_scatter_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_122: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_209, [1, 128, 1, 32, 2]);  unsqueeze_209 = None
        clone_122: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_430: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1, 64]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_554: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_277, view_430);  slice_277 = None
        add_405: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_404, mul_554);  add_404 = mul_554 = None
        mul_555: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_279, view_429);  view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1122: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_555, [1, 128, 16, 32, 2]);  mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_50: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1122, -1, 0)
        select_51: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1122, -1, 1);  view_1122 = None
        neg_95: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_50);  select_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_98: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_95, 3, 1, 9223372036854775807, 2);  neg_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_99: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_51, 3, 0, 9223372036854775807, 2);  select_51 = None
        add_406: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_98, slice_scatter_99);  slice_scatter_98 = slice_scatter_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_556: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_279, view_430);  slice_279 = view_430 = None
        add_407: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_406, mul_556);  add_406 = mul_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_100: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_278, 3, 64, 9223372036854775807);  slice_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_101: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_405, 3, 0, 64);  add_405 = None
        add_408: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_100, slice_scatter_101);  slice_scatter_100 = slice_scatter_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_102: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_280, 3, 64, 9223372036854775807);  slice_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_103: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_407, 3, 0, 64);  add_407 = None
        add_409: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_102, slice_scatter_103);  slice_scatter_102 = slice_scatter_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_729: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_204, [0, 2, 1, 3]);  getitem_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_237: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_729, memory_format = torch.contiguous_format);  permute_729 = None
        view_1123: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_237, [1, 128, 4096]);  clone_237 = None
        view_1124: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_409, [1, 128, 4096]);  add_409 = None
        view_1125: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_408, [1, 128, 4096]);  add_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1126: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1123, [128, 4096]);  view_1123 = None
        permute_730: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1126, [1, 0])
        mm_264: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_730, view_420);  permute_730 = None
        permute_167: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        permute_732: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_265: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1126, permute_732);  view_1126 = permute_732 = None
        view_1127: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_265, [1, 128, 4096]);  mm_265 = None
        add_410: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1111, view_1127);  view_1111 = view_1127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1128: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1124, [128, 4096]);  view_1124 = None
        permute_734: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1128, [1, 0])
        mm_266: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_734, view_420);  permute_734 = None
        permute_166: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_736: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        mm_267: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1128, permute_736);  view_1128 = permute_736 = None
        view_1129: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_267, [1, 128, 4096]);  mm_267 = None
        add_411: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_410, view_1129);  add_410 = view_1129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1130: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1125, [128, 4096]);  view_1125 = None
        permute_738: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1130, [1, 0])
        mm_268: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_738, view_420);  permute_738 = view_420 = None
        permute_165: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        permute_740: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_269: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1130, permute_740);  view_1130 = permute_740 = None
        view_1131: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_269, [1, 128, 4096]);  mm_269 = None
        add_412: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_411, view_1131);  add_411 = view_1131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_558: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_412, primals_168);  primals_168 = None
        mul_559: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_558, 4096)
        sum_125: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_558, [2], True)
        mul_560: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_558, mul_150);  mul_558 = None
        sum_126: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True);  mul_560 = None
        mul_561: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, sum_126);  sum_126 = None
        sub_115: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_559, sum_125);  mul_559 = sum_125 = None
        sub_116: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_115, mul_561);  sub_115 = mul_561 = None
        mul_562: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_84, sub_116);  div_84 = sub_116 = None
        mul_563: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_412, mul_150);  mul_150 = None
        sum_127: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 1]);  mul_563 = None
        sum_128: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_412, [0, 1]);  add_412 = None
        add_413: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_401, mul_562);  add_401 = mul_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1132: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_413, [128, 4096])
        permute_164: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        permute_742: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        mm_270: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1132, permute_742);  permute_742 = None
        permute_743: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1132, [1, 0])
        mm_271: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_743, view_418);  view_418 = None
        sum_129: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True)
        view_1133: "f32[4096]" = torch.ops.aten.reshape.default(sum_129, [4096]);  sum_129 = None
        view_1134: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_270, [1, 128, 16384]);  mm_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_417: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 16384]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_146: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
        mul_564: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1134, mul_146);  mul_146 = None
        pow_15: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_417, 3.0)
        mul_147: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_134: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_417, mul_147);  mul_147 = None
        mul_148: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_134, 0.7978845608028654);  add_134 = None
        tanh_14: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_148);  mul_148 = None
        add_135: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_565: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1134, add_135);  view_1134 = add_135 = None
        mul_566: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_117: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_566);  mul_566 = None
        mul_567: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_564, sub_117);  mul_564 = sub_117 = None
        mul_568: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_567, 0.7978845608028654);  mul_567 = None
        mul_569: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_568, 0.044715)
        pow_42: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_417, 2.0);  view_417 = None
        mul_570: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_571: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
        add_414: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_568, mul_571);  mul_568 = mul_571 = None
        mul_572: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_565, 0.5);  mul_565 = None
        add_415: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_414, mul_572);  add_414 = mul_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1135: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_415, [128, 16384]);  add_415 = None
        permute_163: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
        permute_746: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_272: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1135, permute_746);  permute_746 = None
        permute_747: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_273: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_747, view_392);  permute_747 = None
        sum_130: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
        view_1136: "f32[16384]" = torch.ops.aten.reshape.default(sum_130, [16384]);  sum_130 = None
        view_1137: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_272, [1, 128, 4096]);  mm_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_274: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_743, view_414);  permute_743 = view_414 = None
        permute_162: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        permute_752: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_275: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1132, permute_752);  view_1132 = permute_752 = None
        view_1139: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_275, [1, 128, 4096]);  mm_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1140: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1139, [1, 128, 16, 256]);  view_1139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_754: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1140, [0, 2, 1, 3]);  view_1140 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_754, permute_159, permute_158, permute_157, expand_default_28, getitem_205, getitem_206, getitem_207, getitem_208, 0.0, [True, True, True, False], scale = 0.0625);  permute_754 = permute_159 = permute_158 = permute_157 = getitem_205 = getitem_206 = getitem_207 = getitem_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_209: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_13[0]
        getitem_210: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_13[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_211: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_13[2];  _scaled_dot_product_efficient_attention_backward_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_760: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_761: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_210, [0, 2, 1, 3]);  getitem_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_281: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_760, 3, 0, 64)
        slice_282: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_760, 3, 64, 256);  permute_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_283: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_761, 3, 0, 64)
        slice_284: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_761, 3, 64, 256);  permute_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_194, [1, 128, 1, 32, 2]);  unsqueeze_194 = None
        clone_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_401: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_113, [1, 128, 1, 64]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_574: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_281, view_401)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1147: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_574, [1, 128, 16, 32, 2]);  mul_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_52: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1147, -1, 0)
        select_53: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1147, -1, 1);  view_1147 = None
        neg_97: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_52);  select_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_104: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_97, 3, 1, 9223372036854775807, 2);  neg_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_105: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_53, 3, 0, 9223372036854775807, 2);  select_53 = None
        add_416: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_104, slice_scatter_105);  slice_scatter_104 = slice_scatter_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_114: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_196, [1, 128, 1, 32, 2]);  unsqueeze_196 = None
        clone_114: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
        view_402: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_114, [1, 128, 1, 64]);  clone_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_575: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_281, view_402);  slice_281 = None
        add_417: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_416, mul_575);  add_416 = mul_575 = None
        mul_576: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_283, view_401);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1148: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_576, [1, 128, 16, 32, 2]);  mul_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_54: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1148, -1, 0)
        select_55: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1148, -1, 1);  view_1148 = None
        neg_98: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_54);  select_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_106: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_98, 3, 1, 9223372036854775807, 2);  neg_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_107: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_55, 3, 0, 9223372036854775807, 2);  select_55 = None
        add_418: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_106, slice_scatter_107);  slice_scatter_106 = slice_scatter_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_577: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_283, view_402);  slice_283 = view_402 = None
        add_419: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_418, mul_577);  add_418 = mul_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_108: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_282, 3, 64, 9223372036854775807);  slice_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_109: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_417, 3, 0, 64);  add_417 = None
        add_420: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_108, slice_scatter_109);  slice_scatter_108 = slice_scatter_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_110: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_284, 3, 64, 9223372036854775807);  slice_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_111: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_419, 3, 0, 64);  add_419 = None
        add_421: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_110, slice_scatter_111);  slice_scatter_110 = slice_scatter_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_762: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_211, [0, 2, 1, 3]);  getitem_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_238: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_762, memory_format = torch.contiguous_format);  permute_762 = None
        view_1149: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_238, [1, 128, 4096]);  clone_238 = None
        view_1150: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_421, [1, 128, 4096]);  add_421 = None
        view_1151: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_420, [1, 128, 4096]);  add_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1152: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1149, [128, 4096]);  view_1149 = None
        permute_763: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1152, [1, 0])
        mm_276: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_763, view_392);  permute_763 = None
        permute_156: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
        permute_765: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_277: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1152, permute_765);  view_1152 = permute_765 = None
        view_1153: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_277, [1, 128, 4096]);  mm_277 = None
        add_422: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1137, view_1153);  view_1137 = view_1153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1154: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1150, [128, 4096]);  view_1150 = None
        permute_767: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1154, [1, 0])
        mm_278: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_767, view_392);  permute_767 = None
        permute_155: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        permute_769: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        mm_279: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1154, permute_769);  view_1154 = permute_769 = None
        view_1155: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_279, [1, 128, 4096]);  mm_279 = None
        add_423: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_422, view_1155);  add_422 = view_1155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1156: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1151, [128, 4096]);  view_1151 = None
        permute_771: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1156, [1, 0])
        mm_280: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_771, view_392);  permute_771 = view_392 = None
        permute_154: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        permute_773: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_281: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1156, permute_773);  view_1156 = permute_773 = None
        view_1157: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_281, [1, 128, 4096]);  mm_281 = None
        add_424: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_423, view_1157);  add_423 = view_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_579: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_424, primals_157);  primals_157 = None
        mul_580: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_579, 4096)
        sum_132: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True)
        mul_581: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_579, mul_140);  mul_579 = None
        sum_133: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
        mul_582: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, sum_133);  sum_133 = None
        sub_119: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_580, sum_132);  mul_580 = sum_132 = None
        sub_120: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_119, mul_582);  sub_119 = mul_582 = None
        mul_583: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_86, sub_120);  div_86 = sub_120 = None
        mul_584: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_424, mul_140);  mul_140 = None
        sum_134: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1]);  mul_584 = None
        sum_135: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_424, [0, 1]);  add_424 = None
        add_425: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_413, mul_583);  add_413 = mul_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1158: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_425, [128, 4096])
        permute_153: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_775: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_282: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1158, permute_775);  permute_775 = None
        permute_776: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1158, [1, 0])
        mm_283: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_776, view_390);  view_390 = None
        sum_136: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True)
        view_1159: "f32[4096]" = torch.ops.aten.reshape.default(sum_136, [4096]);  sum_136 = None
        view_1160: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_282, [1, 128, 16384]);  mm_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_389: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 16384]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_136: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_389, 0.5)
        mul_585: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1160, mul_136);  mul_136 = None
        pow_14: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_389, 3.0)
        mul_137: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_125: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_389, mul_137);  mul_137 = None
        mul_138: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_125, 0.7978845608028654);  add_125 = None
        tanh_13: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_138);  mul_138 = None
        add_126: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_586: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1160, add_126);  view_1160 = add_126 = None
        mul_587: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_121: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_587);  mul_587 = None
        mul_588: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_585, sub_121);  mul_585 = sub_121 = None
        mul_589: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_588, 0.7978845608028654);  mul_588 = None
        mul_590: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_589, 0.044715)
        pow_43: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_389, 2.0);  view_389 = None
        mul_591: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
        mul_592: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_590, mul_591);  mul_590 = mul_591 = None
        add_426: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_589, mul_592);  mul_589 = mul_592 = None
        mul_593: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_586, 0.5);  mul_586 = None
        add_427: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_426, mul_593);  add_426 = mul_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1161: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_427, [128, 16384]);  add_427 = None
        permute_152: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_779: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_284: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1161, permute_779);  permute_779 = None
        permute_780: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1161, [1, 0])
        mm_285: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_780, view_364);  permute_780 = None
        sum_137: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1161, [0], True);  view_1161 = None
        view_1162: "f32[16384]" = torch.ops.aten.reshape.default(sum_137, [16384]);  sum_137 = None
        view_1163: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_284, [1, 128, 4096]);  mm_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_286: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_776, view_386);  permute_776 = view_386 = None
        permute_151: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_785: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_287: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1158, permute_785);  view_1158 = permute_785 = None
        view_1165: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_287, [1, 128, 4096]);  mm_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1166: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1165, [1, 128, 16, 256]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_787: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_787, permute_148, permute_147, permute_146, expand_default_28, getitem_212, getitem_213, getitem_214, getitem_215, 0.0, [True, True, True, False], scale = 0.0625);  permute_787 = permute_148 = permute_147 = permute_146 = getitem_212 = getitem_213 = getitem_214 = getitem_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_216: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_14[0]
        getitem_217: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_14[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_218: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_14[2];  _scaled_dot_product_efficient_attention_backward_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_793: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_794: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_217, [0, 2, 1, 3]);  getitem_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_285: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_793, 3, 0, 64)
        slice_286: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_793, 3, 64, 256);  permute_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_287: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_794, 3, 0, 64)
        slice_288: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_794, 3, 64, 256);  permute_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_181, [1, 128, 1, 32, 2]);  unsqueeze_181 = None
        clone_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_373: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_105, [1, 128, 1, 64]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_595: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_285, view_373)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1173: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_595, [1, 128, 16, 32, 2]);  mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_56: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1173, -1, 0)
        select_57: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1173, -1, 1);  view_1173 = None
        neg_100: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_56);  select_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_112: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_100, 3, 1, 9223372036854775807, 2);  neg_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_113: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_57, 3, 0, 9223372036854775807, 2);  select_57 = None
        add_428: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_112, slice_scatter_113);  slice_scatter_112 = slice_scatter_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_106: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_183, [1, 128, 1, 32, 2]);  unsqueeze_183 = None
        clone_106: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
        view_374: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_106, [1, 128, 1, 64]);  clone_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_596: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_285, view_374);  slice_285 = None
        add_429: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_428, mul_596);  add_428 = mul_596 = None
        mul_597: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_287, view_373);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1174: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_597, [1, 128, 16, 32, 2]);  mul_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_58: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1174, -1, 0)
        select_59: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1174, -1, 1);  view_1174 = None
        neg_101: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_58);  select_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_114: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_101, 3, 1, 9223372036854775807, 2);  neg_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_115: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_59, 3, 0, 9223372036854775807, 2);  select_59 = None
        add_430: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_114, slice_scatter_115);  slice_scatter_114 = slice_scatter_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_598: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_287, view_374);  slice_287 = view_374 = None
        add_431: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_430, mul_598);  add_430 = mul_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_116: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_286, 3, 64, 9223372036854775807);  slice_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_117: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_429, 3, 0, 64);  add_429 = None
        add_432: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_116, slice_scatter_117);  slice_scatter_116 = slice_scatter_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_118: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_288, 3, 64, 9223372036854775807);  slice_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_119: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_431, 3, 0, 64);  add_431 = None
        add_433: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_118, slice_scatter_119);  slice_scatter_118 = slice_scatter_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_795: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_239: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_795, memory_format = torch.contiguous_format);  permute_795 = None
        view_1175: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_239, [1, 128, 4096]);  clone_239 = None
        view_1176: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_433, [1, 128, 4096]);  add_433 = None
        view_1177: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_432, [1, 128, 4096]);  add_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1178: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1175, [128, 4096]);  view_1175 = None
        permute_796: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1178, [1, 0])
        mm_288: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_796, view_364);  permute_796 = None
        permute_145: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        permute_798: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_289: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1178, permute_798);  view_1178 = permute_798 = None
        view_1179: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_289, [1, 128, 4096]);  mm_289 = None
        add_434: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1163, view_1179);  view_1163 = view_1179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1180: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1176, [128, 4096]);  view_1176 = None
        permute_800: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1180, [1, 0])
        mm_290: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_800, view_364);  permute_800 = None
        permute_144: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        permute_802: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mm_291: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1180, permute_802);  view_1180 = permute_802 = None
        view_1181: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_291, [1, 128, 4096]);  mm_291 = None
        add_435: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_434, view_1181);  add_434 = view_1181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1182: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1177, [128, 4096]);  view_1177 = None
        permute_804: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1182, [1, 0])
        mm_292: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_804, view_364);  permute_804 = view_364 = None
        permute_143: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        permute_806: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_293: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1182, permute_806);  view_1182 = permute_806 = None
        view_1183: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_293, [1, 128, 4096]);  mm_293 = None
        add_436: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_435, view_1183);  add_435 = view_1183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_600: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_436, primals_146);  primals_146 = None
        mul_601: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_600, 4096)
        sum_139: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True)
        mul_602: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_600, mul_130);  mul_600 = None
        sum_140: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True);  mul_602 = None
        mul_603: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_130, sum_140);  sum_140 = None
        sub_123: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_601, sum_139);  mul_601 = sum_139 = None
        sub_124: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_123, mul_603);  sub_123 = mul_603 = None
        mul_604: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_88, sub_124);  div_88 = sub_124 = None
        mul_605: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_436, mul_130);  mul_130 = None
        sum_141: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 1]);  mul_605 = None
        sum_142: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_436, [0, 1]);  add_436 = None
        add_437: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_425, mul_604);  add_425 = mul_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1184: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_437, [128, 4096])
        permute_142: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        permute_808: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_294: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1184, permute_808);  permute_808 = None
        permute_809: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1184, [1, 0])
        mm_295: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_809, view_362);  view_362 = None
        sum_143: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1184, [0], True)
        view_1185: "f32[4096]" = torch.ops.aten.reshape.default(sum_143, [4096]);  sum_143 = None
        view_1186: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_294, [1, 128, 16384]);  mm_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_361: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 16384]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_126: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_361, 0.5)
        mul_606: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1186, mul_126);  mul_126 = None
        pow_13: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_361, 3.0)
        mul_127: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_116: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_361, mul_127);  mul_127 = None
        mul_128: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_116, 0.7978845608028654);  add_116 = None
        tanh_12: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
        add_117: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_607: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1186, add_117);  view_1186 = add_117 = None
        mul_608: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_125: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_608);  mul_608 = None
        mul_609: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_606, sub_125);  mul_606 = sub_125 = None
        mul_610: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_609, 0.7978845608028654);  mul_609 = None
        mul_611: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_610, 0.044715)
        pow_44: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_361, 2.0);  view_361 = None
        mul_612: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_613: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_611, mul_612);  mul_611 = mul_612 = None
        add_438: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_610, mul_613);  mul_610 = mul_613 = None
        mul_614: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_607, 0.5);  mul_607 = None
        add_439: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_438, mul_614);  add_438 = mul_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1187: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_439, [128, 16384]);  add_439 = None
        permute_141: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_812: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_296: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1187, permute_812);  permute_812 = None
        permute_813: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1187, [1, 0])
        mm_297: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_813, view_336);  permute_813 = None
        sum_144: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1187, [0], True);  view_1187 = None
        view_1188: "f32[16384]" = torch.ops.aten.reshape.default(sum_144, [16384]);  sum_144 = None
        view_1189: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_296, [1, 128, 4096]);  mm_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_298: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_809, view_358);  permute_809 = view_358 = None
        permute_140: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_818: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_299: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1184, permute_818);  view_1184 = permute_818 = None
        view_1191: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_299, [1, 128, 4096]);  mm_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1192: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1191, [1, 128, 16, 256]);  view_1191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_820: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1192, [0, 2, 1, 3]);  view_1192 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_820, permute_137, permute_136, permute_135, expand_default_28, getitem_219, getitem_220, getitem_221, getitem_222, 0.0, [True, True, True, False], scale = 0.0625);  permute_820 = permute_137 = permute_136 = permute_135 = getitem_219 = getitem_220 = getitem_221 = getitem_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_223: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_15[0]
        getitem_224: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_15[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_225: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_15[2];  _scaled_dot_product_efficient_attention_backward_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_826: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_223, [0, 2, 1, 3]);  getitem_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_827: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_224, [0, 2, 1, 3]);  getitem_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_289: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_826, 3, 0, 64)
        slice_290: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_826, 3, 64, 256);  permute_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_291: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_827, 3, 0, 64)
        slice_292: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_827, 3, 64, 256);  permute_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_168, [1, 128, 1, 32, 2]);  unsqueeze_168 = None
        clone_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_345: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_97, [1, 128, 1, 64]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_616: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_289, view_345)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1199: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_616, [1, 128, 16, 32, 2]);  mul_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_60: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1199, -1, 0)
        select_61: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1199, -1, 1);  view_1199 = None
        neg_103: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_60);  select_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_120: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_103, 3, 1, 9223372036854775807, 2);  neg_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_121: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_61, 3, 0, 9223372036854775807, 2);  select_61 = None
        add_440: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_120, slice_scatter_121);  slice_scatter_120 = slice_scatter_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_98: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_170, [1, 128, 1, 32, 2]);  unsqueeze_170 = None
        clone_98: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_346: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_98, [1, 128, 1, 64]);  clone_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_617: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_289, view_346);  slice_289 = None
        add_441: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_440, mul_617);  add_440 = mul_617 = None
        mul_618: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_291, view_345);  view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1200: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_618, [1, 128, 16, 32, 2]);  mul_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_62: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1200, -1, 0)
        select_63: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1200, -1, 1);  view_1200 = None
        neg_104: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_62);  select_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_122: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_104, 3, 1, 9223372036854775807, 2);  neg_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_123: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_63, 3, 0, 9223372036854775807, 2);  select_63 = None
        add_442: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_122, slice_scatter_123);  slice_scatter_122 = slice_scatter_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_619: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_291, view_346);  slice_291 = view_346 = None
        add_443: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_442, mul_619);  add_442 = mul_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_124: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_290, 3, 64, 9223372036854775807);  slice_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_125: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_441, 3, 0, 64);  add_441 = None
        add_444: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_124, slice_scatter_125);  slice_scatter_124 = slice_scatter_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_126: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_292, 3, 64, 9223372036854775807);  slice_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_127: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_443, 3, 0, 64);  add_443 = None
        add_445: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_126, slice_scatter_127);  slice_scatter_126 = slice_scatter_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_828: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_240: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_828, memory_format = torch.contiguous_format);  permute_828 = None
        view_1201: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_240, [1, 128, 4096]);  clone_240 = None
        view_1202: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_445, [1, 128, 4096]);  add_445 = None
        view_1203: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_444, [1, 128, 4096]);  add_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1204: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1201, [128, 4096]);  view_1201 = None
        permute_829: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1204, [1, 0])
        mm_300: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_829, view_336);  permute_829 = None
        permute_134: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        permute_831: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_301: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1204, permute_831);  view_1204 = permute_831 = None
        view_1205: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_301, [1, 128, 4096]);  mm_301 = None
        add_446: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1189, view_1205);  view_1189 = view_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1206: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1202, [128, 4096]);  view_1202 = None
        permute_833: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1206, [1, 0])
        mm_302: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_833, view_336);  permute_833 = None
        permute_133: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        permute_835: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        mm_303: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1206, permute_835);  view_1206 = permute_835 = None
        view_1207: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_303, [1, 128, 4096]);  mm_303 = None
        add_447: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_446, view_1207);  add_446 = view_1207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1208: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1203, [128, 4096]);  view_1203 = None
        permute_837: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1208, [1, 0])
        mm_304: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_837, view_336);  permute_837 = view_336 = None
        permute_132: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        permute_839: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_305: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1208, permute_839);  view_1208 = permute_839 = None
        view_1209: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_305, [1, 128, 4096]);  mm_305 = None
        add_448: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_447, view_1209);  add_447 = view_1209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_621: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_448, primals_135);  primals_135 = None
        mul_622: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_621, 4096)
        sum_146: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True)
        mul_623: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_621, mul_120);  mul_621 = None
        sum_147: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True);  mul_623 = None
        mul_624: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_120, sum_147);  sum_147 = None
        sub_127: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_622, sum_146);  mul_622 = sum_146 = None
        sub_128: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_127, mul_624);  sub_127 = mul_624 = None
        mul_625: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_90, sub_128);  div_90 = sub_128 = None
        mul_626: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_448, mul_120);  mul_120 = None
        sum_148: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 1]);  mul_626 = None
        sum_149: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_448, [0, 1]);  add_448 = None
        add_449: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_437, mul_625);  add_437 = mul_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1210: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_449, [128, 4096])
        permute_131: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        permute_841: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_306: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1210, permute_841);  permute_841 = None
        permute_842: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1210, [1, 0])
        mm_307: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_842, view_334);  view_334 = None
        sum_150: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True)
        view_1211: "f32[4096]" = torch.ops.aten.reshape.default(sum_150, [4096]);  sum_150 = None
        view_1212: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_306, [1, 128, 16384]);  mm_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_333: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 16384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_333, 0.5)
        mul_627: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1212, mul_116);  mul_116 = None
        pow_12: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_333, 3.0)
        mul_117: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_107: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_333, mul_117);  mul_117 = None
        mul_118: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_107, 0.7978845608028654);  add_107 = None
        tanh_11: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_108: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_628: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1212, add_108);  view_1212 = add_108 = None
        mul_629: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_129: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_629);  mul_629 = None
        mul_630: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_627, sub_129);  mul_627 = sub_129 = None
        mul_631: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_630, 0.7978845608028654);  mul_630 = None
        mul_632: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_631, 0.044715)
        pow_45: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_333, 2.0);  view_333 = None
        mul_633: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
        mul_634: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_632, mul_633);  mul_632 = mul_633 = None
        add_450: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_631, mul_634);  mul_631 = mul_634 = None
        mul_635: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_628, 0.5);  mul_628 = None
        add_451: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_450, mul_635);  add_450 = mul_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1213: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_451, [128, 16384]);  add_451 = None
        permute_130: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        permute_845: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_308: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1213, permute_845);  permute_845 = None
        permute_846: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1213, [1, 0])
        mm_309: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_846, view_308);  permute_846 = None
        sum_151: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1213, [0], True);  view_1213 = None
        view_1214: "f32[16384]" = torch.ops.aten.reshape.default(sum_151, [16384]);  sum_151 = None
        view_1215: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_308, [1, 128, 4096]);  mm_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_310: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_842, view_330);  permute_842 = view_330 = None
        permute_129: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        permute_851: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_311: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1210, permute_851);  view_1210 = permute_851 = None
        view_1217: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_311, [1, 128, 4096]);  mm_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1218: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1217, [1, 128, 16, 256]);  view_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_853: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1218, [0, 2, 1, 3]);  view_1218 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_853, permute_126, permute_125, permute_124, expand_default_28, getitem_226, getitem_227, getitem_228, getitem_229, 0.0, [True, True, True, False], scale = 0.0625);  permute_853 = permute_126 = permute_125 = permute_124 = getitem_226 = getitem_227 = getitem_228 = getitem_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_230: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_16[0]
        getitem_231: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_16[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_232: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_16[2];  _scaled_dot_product_efficient_attention_backward_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_859: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_230, [0, 2, 1, 3]);  getitem_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_860: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_231, [0, 2, 1, 3]);  getitem_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_293: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_859, 3, 0, 64)
        slice_294: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_859, 3, 64, 256);  permute_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_295: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_860, 3, 0, 64)
        slice_296: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_860, 3, 64, 256);  permute_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_155, [1, 128, 1, 32, 2]);  unsqueeze_155 = None
        clone_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_317: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_89, [1, 128, 1, 64]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_637: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_293, view_317)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1225: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_637, [1, 128, 16, 32, 2]);  mul_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_64: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1225, -1, 0)
        select_65: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1225, -1, 1);  view_1225 = None
        neg_106: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_64);  select_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_128: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_106, 3, 1, 9223372036854775807, 2);  neg_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_129: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_65, 3, 0, 9223372036854775807, 2);  select_65 = None
        add_452: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_128, slice_scatter_129);  slice_scatter_128 = slice_scatter_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_90: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_157, [1, 128, 1, 32, 2]);  unsqueeze_157 = None
        clone_90: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_318: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_90, [1, 128, 1, 64]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_638: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_293, view_318);  slice_293 = None
        add_453: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_452, mul_638);  add_452 = mul_638 = None
        mul_639: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_295, view_317);  view_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1226: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_639, [1, 128, 16, 32, 2]);  mul_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_66: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1226, -1, 0)
        select_67: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1226, -1, 1);  view_1226 = None
        neg_107: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_66);  select_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_130: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_107, 3, 1, 9223372036854775807, 2);  neg_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_131: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_67, 3, 0, 9223372036854775807, 2);  select_67 = None
        add_454: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_130, slice_scatter_131);  slice_scatter_130 = slice_scatter_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_640: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_295, view_318);  slice_295 = view_318 = None
        add_455: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_454, mul_640);  add_454 = mul_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_132: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_294, 3, 64, 9223372036854775807);  slice_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_133: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_453, 3, 0, 64);  add_453 = None
        add_456: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_132, slice_scatter_133);  slice_scatter_132 = slice_scatter_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_134: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_296, 3, 64, 9223372036854775807);  slice_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_135: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_455, 3, 0, 64);  add_455 = None
        add_457: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_134, slice_scatter_135);  slice_scatter_134 = slice_scatter_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_861: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_232, [0, 2, 1, 3]);  getitem_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_241: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_861, memory_format = torch.contiguous_format);  permute_861 = None
        view_1227: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_241, [1, 128, 4096]);  clone_241 = None
        view_1228: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_457, [1, 128, 4096]);  add_457 = None
        view_1229: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_456, [1, 128, 4096]);  add_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1230: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1227, [128, 4096]);  view_1227 = None
        permute_862: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1230, [1, 0])
        mm_312: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_862, view_308);  permute_862 = None
        permute_123: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_864: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_313: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1230, permute_864);  view_1230 = permute_864 = None
        view_1231: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_313, [1, 128, 4096]);  mm_313 = None
        add_458: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1215, view_1231);  view_1215 = view_1231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1232: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1228, [128, 4096]);  view_1228 = None
        permute_866: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1232, [1, 0])
        mm_314: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_866, view_308);  permute_866 = None
        permute_122: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_868: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        mm_315: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1232, permute_868);  view_1232 = permute_868 = None
        view_1233: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_315, [1, 128, 4096]);  mm_315 = None
        add_459: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_458, view_1233);  add_458 = view_1233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1234: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1229, [128, 4096]);  view_1229 = None
        permute_870: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1234, [1, 0])
        mm_316: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_870, view_308);  permute_870 = view_308 = None
        permute_121: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        permute_872: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_317: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1234, permute_872);  view_1234 = permute_872 = None
        view_1235: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_317, [1, 128, 4096]);  mm_317 = None
        add_460: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_459, view_1235);  add_459 = view_1235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_642: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_460, primals_124);  primals_124 = None
        mul_643: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_642, 4096)
        sum_153: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [2], True)
        mul_644: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_642, mul_110);  mul_642 = None
        sum_154: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True);  mul_644 = None
        mul_645: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, sum_154);  sum_154 = None
        sub_131: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_643, sum_153);  mul_643 = sum_153 = None
        sub_132: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_131, mul_645);  sub_131 = mul_645 = None
        mul_646: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_92, sub_132);  div_92 = sub_132 = None
        mul_647: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_460, mul_110);  mul_110 = None
        sum_155: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 1]);  mul_647 = None
        sum_156: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_460, [0, 1]);  add_460 = None
        add_461: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_449, mul_646);  add_449 = mul_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1236: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_461, [128, 4096])
        permute_120: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_874: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_318: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1236, permute_874);  permute_874 = None
        permute_875: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1236, [1, 0])
        mm_319: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_875, view_306);  view_306 = None
        sum_157: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1236, [0], True)
        view_1237: "f32[4096]" = torch.ops.aten.reshape.default(sum_157, [4096]);  sum_157 = None
        view_1238: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_318, [1, 128, 16384]);  mm_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_305: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 16384]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_106: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_305, 0.5)
        mul_648: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1238, mul_106);  mul_106 = None
        pow_11: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_305, 3.0)
        mul_107: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_98: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_305, mul_107);  mul_107 = None
        mul_108: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_98, 0.7978845608028654);  add_98 = None
        tanh_10: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
        add_99: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_649: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1238, add_99);  view_1238 = add_99 = None
        mul_650: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_133: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_650);  mul_650 = None
        mul_651: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_648, sub_133);  mul_648 = sub_133 = None
        mul_652: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_651, 0.7978845608028654);  mul_651 = None
        mul_653: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_652, 0.044715)
        pow_46: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_305, 2.0);  view_305 = None
        mul_654: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_655: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
        add_462: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_652, mul_655);  mul_652 = mul_655 = None
        mul_656: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_649, 0.5);  mul_649 = None
        add_463: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_462, mul_656);  add_462 = mul_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1239: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_463, [128, 16384]);  add_463 = None
        permute_119: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_878: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_320: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1239, permute_878);  permute_878 = None
        permute_879: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1239, [1, 0])
        mm_321: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_879, view_280);  permute_879 = None
        sum_158: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1239, [0], True);  view_1239 = None
        view_1240: "f32[16384]" = torch.ops.aten.reshape.default(sum_158, [16384]);  sum_158 = None
        view_1241: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_320, [1, 128, 4096]);  mm_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_322: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_875, view_302);  permute_875 = view_302 = None
        permute_118: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        permute_884: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_323: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1236, permute_884);  view_1236 = permute_884 = None
        view_1243: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_323, [1, 128, 4096]);  mm_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1244: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1243, [1, 128, 16, 256]);  view_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_886: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1244, [0, 2, 1, 3]);  view_1244 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_886, permute_115, permute_114, permute_113, expand_default_28, getitem_233, getitem_234, getitem_235, getitem_236, 0.0, [True, True, True, False], scale = 0.0625);  permute_886 = permute_115 = permute_114 = permute_113 = getitem_233 = getitem_234 = getitem_235 = getitem_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_237: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_17[0]
        getitem_238: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_17[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_239: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_17[2];  _scaled_dot_product_efficient_attention_backward_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_892: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_237, [0, 2, 1, 3]);  getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_893: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_238, [0, 2, 1, 3]);  getitem_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_297: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_892, 3, 0, 64)
        slice_298: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_892, 3, 64, 256);  permute_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_299: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_893, 3, 0, 64)
        slice_300: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_893, 3, 64, 256);  permute_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_142, [1, 128, 1, 32, 2]);  unsqueeze_142 = None
        clone_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_289: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_81, [1, 128, 1, 64]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_658: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_297, view_289)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1251: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_658, [1, 128, 16, 32, 2]);  mul_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_68: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1251, -1, 0)
        select_69: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1251, -1, 1);  view_1251 = None
        neg_109: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_68);  select_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_136: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_109, 3, 1, 9223372036854775807, 2);  neg_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_137: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_69, 3, 0, 9223372036854775807, 2);  select_69 = None
        add_464: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_136, slice_scatter_137);  slice_scatter_136 = slice_scatter_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_82: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_144, [1, 128, 1, 32, 2]);  unsqueeze_144 = None
        clone_82: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_290: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_82, [1, 128, 1, 64]);  clone_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_659: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_297, view_290);  slice_297 = None
        add_465: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_464, mul_659);  add_464 = mul_659 = None
        mul_660: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_299, view_289);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1252: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_660, [1, 128, 16, 32, 2]);  mul_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_70: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1252, -1, 0)
        select_71: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1252, -1, 1);  view_1252 = None
        neg_110: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_70);  select_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_138: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_110, 3, 1, 9223372036854775807, 2);  neg_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_139: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_71, 3, 0, 9223372036854775807, 2);  select_71 = None
        add_466: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_138, slice_scatter_139);  slice_scatter_138 = slice_scatter_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_661: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_299, view_290);  slice_299 = view_290 = None
        add_467: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_466, mul_661);  add_466 = mul_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_140: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_298, 3, 64, 9223372036854775807);  slice_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_141: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_465, 3, 0, 64);  add_465 = None
        add_468: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_140, slice_scatter_141);  slice_scatter_140 = slice_scatter_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_142: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_300, 3, 64, 9223372036854775807);  slice_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_143: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_467, 3, 0, 64);  add_467 = None
        add_469: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_142, slice_scatter_143);  slice_scatter_142 = slice_scatter_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_894: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_239, [0, 2, 1, 3]);  getitem_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_242: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_894, memory_format = torch.contiguous_format);  permute_894 = None
        view_1253: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_242, [1, 128, 4096]);  clone_242 = None
        view_1254: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_469, [1, 128, 4096]);  add_469 = None
        view_1255: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_468, [1, 128, 4096]);  add_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1256: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1253, [128, 4096]);  view_1253 = None
        permute_895: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1256, [1, 0])
        mm_324: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_895, view_280);  permute_895 = None
        permute_112: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        permute_897: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_325: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1256, permute_897);  view_1256 = permute_897 = None
        view_1257: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_325, [1, 128, 4096]);  mm_325 = None
        add_470: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1241, view_1257);  view_1241 = view_1257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1258: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1254, [128, 4096]);  view_1254 = None
        permute_899: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1258, [1, 0])
        mm_326: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_899, view_280);  permute_899 = None
        permute_111: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        permute_901: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_327: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1258, permute_901);  view_1258 = permute_901 = None
        view_1259: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_327, [1, 128, 4096]);  mm_327 = None
        add_471: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_470, view_1259);  add_470 = view_1259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1260: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1255, [128, 4096]);  view_1255 = None
        permute_903: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1260, [1, 0])
        mm_328: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_903, view_280);  permute_903 = view_280 = None
        permute_110: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_905: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_329: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1260, permute_905);  view_1260 = permute_905 = None
        view_1261: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_329, [1, 128, 4096]);  mm_329 = None
        add_472: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_471, view_1261);  add_471 = view_1261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_663: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_472, primals_113);  primals_113 = None
        mul_664: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_663, 4096)
        sum_160: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True)
        mul_665: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_663, mul_100);  mul_663 = None
        sum_161: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True);  mul_665 = None
        mul_666: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_100, sum_161);  sum_161 = None
        sub_135: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_664, sum_160);  mul_664 = sum_160 = None
        sub_136: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_135, mul_666);  sub_135 = mul_666 = None
        mul_667: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_94, sub_136);  div_94 = sub_136 = None
        mul_668: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_472, mul_100);  mul_100 = None
        sum_162: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1]);  mul_668 = None
        sum_163: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_472, [0, 1]);  add_472 = None
        add_473: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_461, mul_667);  add_461 = mul_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1262: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_473, [128, 4096])
        permute_109: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_907: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_330: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1262, permute_907);  permute_907 = None
        permute_908: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1262, [1, 0])
        mm_331: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_908, view_278);  view_278 = None
        sum_164: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1262, [0], True)
        view_1263: "f32[4096]" = torch.ops.aten.reshape.default(sum_164, [4096]);  sum_164 = None
        view_1264: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_330, [1, 128, 16384]);  mm_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_277: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 16384]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_96: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_277, 0.5)
        mul_669: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1264, mul_96);  mul_96 = None
        pow_10: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_277, 3.0)
        mul_97: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_89: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_277, mul_97);  mul_97 = None
        mul_98: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_9: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
        add_90: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_670: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1264, add_90);  view_1264 = add_90 = None
        mul_671: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_137: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_671);  mul_671 = None
        mul_672: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_669, sub_137);  mul_669 = sub_137 = None
        mul_673: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_672, 0.7978845608028654);  mul_672 = None
        mul_674: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_673, 0.044715)
        pow_47: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_277, 2.0);  view_277 = None
        mul_675: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
        mul_676: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
        add_474: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_673, mul_676);  mul_673 = mul_676 = None
        mul_677: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_670, 0.5);  mul_670 = None
        add_475: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_474, mul_677);  add_474 = mul_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1265: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_475, [128, 16384]);  add_475 = None
        permute_108: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_911: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_332: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1265, permute_911);  permute_911 = None
        permute_912: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1265, [1, 0])
        mm_333: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_912, view_252);  permute_912 = None
        sum_165: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1265, [0], True);  view_1265 = None
        view_1266: "f32[16384]" = torch.ops.aten.reshape.default(sum_165, [16384]);  sum_165 = None
        view_1267: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_332, [1, 128, 4096]);  mm_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_334: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_908, view_274);  permute_908 = view_274 = None
        permute_107: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        permute_917: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_335: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1262, permute_917);  view_1262 = permute_917 = None
        view_1269: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_335, [1, 128, 4096]);  mm_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1270: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1269, [1, 128, 16, 256]);  view_1269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_919: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1270, [0, 2, 1, 3]);  view_1270 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_919, permute_104, permute_103, permute_102, expand_default_28, getitem_240, getitem_241, getitem_242, getitem_243, 0.0, [True, True, True, False], scale = 0.0625);  permute_919 = permute_104 = permute_103 = permute_102 = getitem_240 = getitem_241 = getitem_242 = getitem_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_244: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_18[0]
        getitem_245: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_18[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_246: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_18[2];  _scaled_dot_product_efficient_attention_backward_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_925: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_244, [0, 2, 1, 3]);  getitem_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_926: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_245, [0, 2, 1, 3]);  getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_301: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_925, 3, 0, 64)
        slice_302: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_925, 3, 64, 256);  permute_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_303: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_926, 3, 0, 64)
        slice_304: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_926, 3, 64, 256);  permute_926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_129, [1, 128, 1, 32, 2]);  unsqueeze_129 = None
        clone_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_261: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_73, [1, 128, 1, 64]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_679: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_301, view_261)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1277: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_679, [1, 128, 16, 32, 2]);  mul_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_72: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1277, -1, 0)
        select_73: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1277, -1, 1);  view_1277 = None
        neg_112: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_72);  select_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_144: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_112, 3, 1, 9223372036854775807, 2);  neg_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_145: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_73, 3, 0, 9223372036854775807, 2);  select_73 = None
        add_476: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_144, slice_scatter_145);  slice_scatter_144 = slice_scatter_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_74: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_131, [1, 128, 1, 32, 2]);  unsqueeze_131 = None
        clone_74: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
        view_262: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_74, [1, 128, 1, 64]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_680: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_301, view_262);  slice_301 = None
        add_477: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_476, mul_680);  add_476 = mul_680 = None
        mul_681: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_303, view_261);  view_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1278: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_681, [1, 128, 16, 32, 2]);  mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_74: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1278, -1, 0)
        select_75: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1278, -1, 1);  view_1278 = None
        neg_113: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_74);  select_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_146: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_113, 3, 1, 9223372036854775807, 2);  neg_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_147: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_75, 3, 0, 9223372036854775807, 2);  select_75 = None
        add_478: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_146, slice_scatter_147);  slice_scatter_146 = slice_scatter_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_682: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_303, view_262);  slice_303 = view_262 = None
        add_479: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_478, mul_682);  add_478 = mul_682 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_148: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_302, 3, 64, 9223372036854775807);  slice_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_149: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_477, 3, 0, 64);  add_477 = None
        add_480: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_148, slice_scatter_149);  slice_scatter_148 = slice_scatter_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_150: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_304, 3, 64, 9223372036854775807);  slice_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_151: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_479, 3, 0, 64);  add_479 = None
        add_481: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_150, slice_scatter_151);  slice_scatter_150 = slice_scatter_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_927: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_246, [0, 2, 1, 3]);  getitem_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_243: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_927, memory_format = torch.contiguous_format);  permute_927 = None
        view_1279: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_243, [1, 128, 4096]);  clone_243 = None
        view_1280: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_481, [1, 128, 4096]);  add_481 = None
        view_1281: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_480, [1, 128, 4096]);  add_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1282: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1279, [128, 4096]);  view_1279 = None
        permute_928: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1282, [1, 0])
        mm_336: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_928, view_252);  permute_928 = None
        permute_101: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_930: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_337: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1282, permute_930);  view_1282 = permute_930 = None
        view_1283: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_337, [1, 128, 4096]);  mm_337 = None
        add_482: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1267, view_1283);  view_1267 = view_1283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1284: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1280, [128, 4096]);  view_1280 = None
        permute_932: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1284, [1, 0])
        mm_338: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_932, view_252);  permute_932 = None
        permute_100: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_934: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        mm_339: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1284, permute_934);  view_1284 = permute_934 = None
        view_1285: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_339, [1, 128, 4096]);  mm_339 = None
        add_483: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_482, view_1285);  add_482 = view_1285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1286: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1281, [128, 4096]);  view_1281 = None
        permute_936: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1286, [1, 0])
        mm_340: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_936, view_252);  permute_936 = view_252 = None
        permute_99: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_938: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_341: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1286, permute_938);  view_1286 = permute_938 = None
        view_1287: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_341, [1, 128, 4096]);  mm_341 = None
        add_484: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_483, view_1287);  add_483 = view_1287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_684: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_484, primals_102);  primals_102 = None
        mul_685: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_684, 4096)
        sum_167: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_684, [2], True)
        mul_686: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_684, mul_90);  mul_684 = None
        sum_168: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_686, [2], True);  mul_686 = None
        mul_687: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_90, sum_168);  sum_168 = None
        sub_139: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_685, sum_167);  mul_685 = sum_167 = None
        sub_140: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_139, mul_687);  sub_139 = mul_687 = None
        mul_688: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_96, sub_140);  div_96 = sub_140 = None
        mul_689: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_484, mul_90);  mul_90 = None
        sum_169: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 1]);  mul_689 = None
        sum_170: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_484, [0, 1]);  add_484 = None
        add_485: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_473, mul_688);  add_473 = mul_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1288: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_485, [128, 4096])
        permute_98: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_940: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_342: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1288, permute_940);  permute_940 = None
        permute_941: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1288, [1, 0])
        mm_343: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_941, view_250);  view_250 = None
        sum_171: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1288, [0], True)
        view_1289: "f32[4096]" = torch.ops.aten.reshape.default(sum_171, [4096]);  sum_171 = None
        view_1290: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_342, [1, 128, 16384]);  mm_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_249: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 16384]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_86: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_249, 0.5)
        mul_690: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1290, mul_86);  mul_86 = None
        pow_9: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_249, 3.0)
        mul_87: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_80: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_249, mul_87);  mul_87 = None
        mul_88: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
        add_81: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_691: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1290, add_81);  view_1290 = add_81 = None
        mul_692: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_141: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_692);  mul_692 = None
        mul_693: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_690, sub_141);  mul_690 = sub_141 = None
        mul_694: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_693, 0.7978845608028654);  mul_693 = None
        mul_695: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_694, 0.044715)
        pow_48: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_249, 2.0);  view_249 = None
        mul_696: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_697: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_695, mul_696);  mul_695 = mul_696 = None
        add_486: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_694, mul_697);  mul_694 = mul_697 = None
        mul_698: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_691, 0.5);  mul_691 = None
        add_487: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_486, mul_698);  add_486 = mul_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1291: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_487, [128, 16384]);  add_487 = None
        permute_97: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        permute_944: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_344: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1291, permute_944);  permute_944 = None
        permute_945: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1291, [1, 0])
        mm_345: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_945, view_224);  permute_945 = None
        sum_172: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1291, [0], True);  view_1291 = None
        view_1292: "f32[16384]" = torch.ops.aten.reshape.default(sum_172, [16384]);  sum_172 = None
        view_1293: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_344, [1, 128, 4096]);  mm_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_346: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_941, view_246);  permute_941 = view_246 = None
        permute_96: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_950: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_347: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1288, permute_950);  view_1288 = permute_950 = None
        view_1295: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_347, [1, 128, 4096]);  mm_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1296: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1295, [1, 128, 16, 256]);  view_1295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_952: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1296, [0, 2, 1, 3]);  view_1296 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_952, permute_93, permute_92, permute_91, expand_default_28, getitem_247, getitem_248, getitem_249, getitem_250, 0.0, [True, True, True, False], scale = 0.0625);  permute_952 = permute_93 = permute_92 = permute_91 = getitem_247 = getitem_248 = getitem_249 = getitem_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_251: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_19[0]
        getitem_252: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_19[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_253: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_19[2];  _scaled_dot_product_efficient_attention_backward_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_958: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_251, [0, 2, 1, 3]);  getitem_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_959: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3]);  getitem_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_305: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_958, 3, 0, 64)
        slice_306: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_958, 3, 64, 256);  permute_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_307: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_959, 3, 0, 64)
        slice_308: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_959, 3, 64, 256);  permute_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_116, [1, 128, 1, 32, 2]);  unsqueeze_116 = None
        clone_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_233: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_65, [1, 128, 1, 64]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_700: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_305, view_233)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1303: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_700, [1, 128, 16, 32, 2]);  mul_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_76: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1303, -1, 0)
        select_77: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1303, -1, 1);  view_1303 = None
        neg_115: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_76);  select_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_152: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_115, 3, 1, 9223372036854775807, 2);  neg_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_153: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_77, 3, 0, 9223372036854775807, 2);  select_77 = None
        add_488: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_152, slice_scatter_153);  slice_scatter_152 = slice_scatter_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_66: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_118, [1, 128, 1, 32, 2]);  unsqueeze_118 = None
        clone_66: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
        view_234: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_66, [1, 128, 1, 64]);  clone_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_701: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_305, view_234);  slice_305 = None
        add_489: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_488, mul_701);  add_488 = mul_701 = None
        mul_702: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_307, view_233);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1304: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_702, [1, 128, 16, 32, 2]);  mul_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_78: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1304, -1, 0)
        select_79: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1304, -1, 1);  view_1304 = None
        neg_116: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_78);  select_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_154: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_116, 3, 1, 9223372036854775807, 2);  neg_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_155: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_79, 3, 0, 9223372036854775807, 2);  select_79 = None
        add_490: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_154, slice_scatter_155);  slice_scatter_154 = slice_scatter_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_703: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_307, view_234);  slice_307 = view_234 = None
        add_491: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_490, mul_703);  add_490 = mul_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_156: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_306, 3, 64, 9223372036854775807);  slice_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_157: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_489, 3, 0, 64);  add_489 = None
        add_492: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_156, slice_scatter_157);  slice_scatter_156 = slice_scatter_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_158: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_308, 3, 64, 9223372036854775807);  slice_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_159: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_491, 3, 0, 64);  add_491 = None
        add_493: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_158, slice_scatter_159);  slice_scatter_158 = slice_scatter_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_960: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_253, [0, 2, 1, 3]);  getitem_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_244: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_960, memory_format = torch.contiguous_format);  permute_960 = None
        view_1305: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_244, [1, 128, 4096]);  clone_244 = None
        view_1306: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_493, [1, 128, 4096]);  add_493 = None
        view_1307: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_492, [1, 128, 4096]);  add_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1308: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1305, [128, 4096]);  view_1305 = None
        permute_961: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1308, [1, 0])
        mm_348: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_961, view_224);  permute_961 = None
        permute_90: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_963: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_349: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1308, permute_963);  view_1308 = permute_963 = None
        view_1309: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_349, [1, 128, 4096]);  mm_349 = None
        add_494: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1293, view_1309);  view_1293 = view_1309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1310: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1306, [128, 4096]);  view_1306 = None
        permute_965: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1310, [1, 0])
        mm_350: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_965, view_224);  permute_965 = None
        permute_89: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_967: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_351: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1310, permute_967);  view_1310 = permute_967 = None
        view_1311: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_351, [1, 128, 4096]);  mm_351 = None
        add_495: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_494, view_1311);  add_494 = view_1311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1312: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1307, [128, 4096]);  view_1307 = None
        permute_969: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1312, [1, 0])
        mm_352: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_969, view_224);  permute_969 = view_224 = None
        permute_88: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_971: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_353: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1312, permute_971);  view_1312 = permute_971 = None
        view_1313: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_353, [1, 128, 4096]);  mm_353 = None
        add_496: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_495, view_1313);  add_495 = view_1313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_705: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_496, primals_91);  primals_91 = None
        mul_706: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_705, 4096)
        sum_174: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True)
        mul_707: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_705, mul_80);  mul_705 = None
        sum_175: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [2], True);  mul_707 = None
        mul_708: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_80, sum_175);  sum_175 = None
        sub_143: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_706, sum_174);  mul_706 = sum_174 = None
        sub_144: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_143, mul_708);  sub_143 = mul_708 = None
        mul_709: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_98, sub_144);  div_98 = sub_144 = None
        mul_710: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_496, mul_80);  mul_80 = None
        sum_176: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 1]);  mul_710 = None
        sum_177: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_496, [0, 1]);  add_496 = None
        add_497: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_485, mul_709);  add_485 = mul_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1314: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_497, [128, 4096])
        permute_87: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_973: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_354: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1314, permute_973);  permute_973 = None
        permute_974: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1314, [1, 0])
        mm_355: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_974, view_222);  view_222 = None
        sum_178: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1314, [0], True)
        view_1315: "f32[4096]" = torch.ops.aten.reshape.default(sum_178, [4096]);  sum_178 = None
        view_1316: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_354, [1, 128, 16384]);  mm_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_221: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 16384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_76: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_221, 0.5)
        mul_711: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1316, mul_76);  mul_76 = None
        pow_8: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_221, 3.0)
        mul_77: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_71: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_221, mul_77);  mul_77 = None
        mul_78: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_7: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
        add_72: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_712: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1316, add_72);  view_1316 = add_72 = None
        mul_713: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_145: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_713);  mul_713 = None
        mul_714: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_711, sub_145);  mul_711 = sub_145 = None
        mul_715: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_714, 0.7978845608028654);  mul_714 = None
        mul_716: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_715, 0.044715)
        pow_49: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_221, 2.0);  view_221 = None
        mul_717: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_49, 3.0);  pow_49 = None
        mul_718: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_716, mul_717);  mul_716 = mul_717 = None
        add_498: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_715, mul_718);  mul_715 = mul_718 = None
        mul_719: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_712, 0.5);  mul_712 = None
        add_499: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_498, mul_719);  add_498 = mul_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1317: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_499, [128, 16384]);  add_499 = None
        permute_86: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_977: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_356: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1317, permute_977);  permute_977 = None
        permute_978: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1317, [1, 0])
        mm_357: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_978, view_196);  permute_978 = None
        sum_179: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1317, [0], True);  view_1317 = None
        view_1318: "f32[16384]" = torch.ops.aten.reshape.default(sum_179, [16384]);  sum_179 = None
        view_1319: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_356, [1, 128, 4096]);  mm_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_358: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_974, view_218);  permute_974 = view_218 = None
        permute_85: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_983: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_359: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1314, permute_983);  view_1314 = permute_983 = None
        view_1321: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_359, [1, 128, 4096]);  mm_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1322: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1321, [1, 128, 16, 256]);  view_1321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_985: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1322, [0, 2, 1, 3]);  view_1322 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_985, permute_82, permute_81, permute_80, expand_default_28, getitem_254, getitem_255, getitem_256, getitem_257, 0.0, [True, True, True, False], scale = 0.0625);  permute_985 = permute_82 = permute_81 = permute_80 = getitem_254 = getitem_255 = getitem_256 = getitem_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_258: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_20[0]
        getitem_259: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_20[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_260: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_20[2];  _scaled_dot_product_efficient_attention_backward_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_991: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_258, [0, 2, 1, 3]);  getitem_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_992: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_259, [0, 2, 1, 3]);  getitem_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_309: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_991, 3, 0, 64)
        slice_310: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_991, 3, 64, 256);  permute_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_311: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_992, 3, 0, 64)
        slice_312: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_992, 3, 64, 256);  permute_992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_103, [1, 128, 1, 32, 2]);  unsqueeze_103 = None
        clone_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_205: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_57, [1, 128, 1, 64]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_721: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_309, view_205)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1329: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_721, [1, 128, 16, 32, 2]);  mul_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_80: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1329, -1, 0)
        select_81: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1329, -1, 1);  view_1329 = None
        neg_118: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_80);  select_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_160: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_118, 3, 1, 9223372036854775807, 2);  neg_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_161: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_81, 3, 0, 9223372036854775807, 2);  select_81 = None
        add_500: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_160, slice_scatter_161);  slice_scatter_160 = slice_scatter_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_58: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_105, [1, 128, 1, 32, 2]);  unsqueeze_105 = None
        clone_58: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_206: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_58, [1, 128, 1, 64]);  clone_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_722: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_309, view_206);  slice_309 = None
        add_501: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_500, mul_722);  add_500 = mul_722 = None
        mul_723: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_311, view_205);  view_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1330: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_723, [1, 128, 16, 32, 2]);  mul_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_82: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1330, -1, 0)
        select_83: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1330, -1, 1);  view_1330 = None
        neg_119: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_82);  select_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_162: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_119, 3, 1, 9223372036854775807, 2);  neg_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_163: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_83, 3, 0, 9223372036854775807, 2);  select_83 = None
        add_502: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_162, slice_scatter_163);  slice_scatter_162 = slice_scatter_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_724: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_311, view_206);  slice_311 = view_206 = None
        add_503: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_502, mul_724);  add_502 = mul_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_164: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_310, 3, 64, 9223372036854775807);  slice_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_165: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_501, 3, 0, 64);  add_501 = None
        add_504: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_164, slice_scatter_165);  slice_scatter_164 = slice_scatter_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_166: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_312, 3, 64, 9223372036854775807);  slice_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_167: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_503, 3, 0, 64);  add_503 = None
        add_505: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_166, slice_scatter_167);  slice_scatter_166 = slice_scatter_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_993: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_245: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_993, memory_format = torch.contiguous_format);  permute_993 = None
        view_1331: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_245, [1, 128, 4096]);  clone_245 = None
        view_1332: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_505, [1, 128, 4096]);  add_505 = None
        view_1333: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_504, [1, 128, 4096]);  add_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1334: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1331, [128, 4096]);  view_1331 = None
        permute_994: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1334, [1, 0])
        mm_360: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_994, view_196);  permute_994 = None
        permute_79: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        permute_996: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_361: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1334, permute_996);  view_1334 = permute_996 = None
        view_1335: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_361, [1, 128, 4096]);  mm_361 = None
        add_506: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1319, view_1335);  view_1319 = view_1335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1336: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1332, [128, 4096]);  view_1332 = None
        permute_998: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1336, [1, 0])
        mm_362: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_998, view_196);  permute_998 = None
        permute_78: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_1000: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        mm_363: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1336, permute_1000);  view_1336 = permute_1000 = None
        view_1337: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_363, [1, 128, 4096]);  mm_363 = None
        add_507: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_506, view_1337);  add_506 = view_1337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1338: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1333, [128, 4096]);  view_1333 = None
        permute_1002: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1338, [1, 0])
        mm_364: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1002, view_196);  permute_1002 = view_196 = None
        permute_77: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        permute_1004: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_365: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1338, permute_1004);  view_1338 = permute_1004 = None
        view_1339: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_365, [1, 128, 4096]);  mm_365 = None
        add_508: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_507, view_1339);  add_507 = view_1339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_726: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_508, primals_80);  primals_80 = None
        mul_727: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_726, 4096)
        sum_181: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_726, [2], True)
        mul_728: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_726, mul_70);  mul_726 = None
        sum_182: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_728, [2], True);  mul_728 = None
        mul_729: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, sum_182);  sum_182 = None
        sub_147: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_727, sum_181);  mul_727 = sum_181 = None
        sub_148: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_147, mul_729);  sub_147 = mul_729 = None
        mul_730: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_100, sub_148);  div_100 = sub_148 = None
        mul_731: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_508, mul_70);  mul_70 = None
        sum_183: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 1]);  mul_731 = None
        sum_184: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_508, [0, 1]);  add_508 = None
        add_509: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_497, mul_730);  add_497 = mul_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1340: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_509, [128, 4096])
        permute_76: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
        permute_1006: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_366: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1340, permute_1006);  permute_1006 = None
        permute_1007: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1340, [1, 0])
        mm_367: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1007, view_194);  view_194 = None
        sum_185: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1340, [0], True)
        view_1341: "f32[4096]" = torch.ops.aten.reshape.default(sum_185, [4096]);  sum_185 = None
        view_1342: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_366, [1, 128, 16384]);  mm_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_193: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 16384]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_66: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_193, 0.5)
        mul_732: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1342, mul_66);  mul_66 = None
        pow_7: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_193, 3.0)
        mul_67: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_62: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_193, mul_67);  mul_67 = None
        mul_68: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
        tanh_6: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
        add_63: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_733: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1342, add_63);  view_1342 = add_63 = None
        mul_734: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_149: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_734);  mul_734 = None
        mul_735: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_732, sub_149);  mul_732 = sub_149 = None
        mul_736: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_735, 0.7978845608028654);  mul_735 = None
        mul_737: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_736, 0.044715)
        pow_50: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_193, 2.0);  view_193 = None
        mul_738: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_50, 3.0);  pow_50 = None
        mul_739: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
        add_510: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_736, mul_739);  mul_736 = mul_739 = None
        mul_740: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_733, 0.5);  mul_733 = None
        add_511: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_510, mul_740);  add_510 = mul_740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1343: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_511, [128, 16384]);  add_511 = None
        permute_75: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_1010: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_368: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1343, permute_1010);  permute_1010 = None
        permute_1011: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1343, [1, 0])
        mm_369: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1011, view_168);  permute_1011 = None
        sum_186: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1343, [0], True);  view_1343 = None
        view_1344: "f32[16384]" = torch.ops.aten.reshape.default(sum_186, [16384]);  sum_186 = None
        view_1345: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_368, [1, 128, 4096]);  mm_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_370: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1007, view_190);  permute_1007 = view_190 = None
        permute_74: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_1016: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_371: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1340, permute_1016);  view_1340 = permute_1016 = None
        view_1347: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_371, [1, 128, 4096]);  mm_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1348: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1347, [1, 128, 16, 256]);  view_1347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1018: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1348, [0, 2, 1, 3]);  view_1348 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_1018, permute_71, permute_70, permute_69, expand_default_28, getitem_261, getitem_262, getitem_263, getitem_264, 0.0, [True, True, True, False], scale = 0.0625);  permute_1018 = permute_71 = permute_70 = permute_69 = getitem_261 = getitem_262 = getitem_263 = getitem_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_265: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_21[0]
        getitem_266: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_21[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_267: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_21[2];  _scaled_dot_product_efficient_attention_backward_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1024: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_265, [0, 2, 1, 3]);  getitem_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1025: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_266, [0, 2, 1, 3]);  getitem_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_313: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1024, 3, 0, 64)
        slice_314: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1024, 3, 64, 256);  permute_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_315: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1025, 3, 0, 64)
        slice_316: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1025, 3, 64, 256);  permute_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_90, [1, 128, 1, 32, 2]);  unsqueeze_90 = None
        clone_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_177: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_49, [1, 128, 1, 64]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_742: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_313, view_177)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1355: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_742, [1, 128, 16, 32, 2]);  mul_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_84: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1355, -1, 0)
        select_85: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1355, -1, 1);  view_1355 = None
        neg_121: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_84);  select_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_168: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_121, 3, 1, 9223372036854775807, 2);  neg_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_169: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_85, 3, 0, 9223372036854775807, 2);  select_85 = None
        add_512: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_168, slice_scatter_169);  slice_scatter_168 = slice_scatter_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_50: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_92, [1, 128, 1, 32, 2]);  unsqueeze_92 = None
        clone_50: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_178: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_50, [1, 128, 1, 64]);  clone_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_743: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_313, view_178);  slice_313 = None
        add_513: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_512, mul_743);  add_512 = mul_743 = None
        mul_744: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_315, view_177);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1356: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_744, [1, 128, 16, 32, 2]);  mul_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_86: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1356, -1, 0)
        select_87: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1356, -1, 1);  view_1356 = None
        neg_122: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_86);  select_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_170: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_122, 3, 1, 9223372036854775807, 2);  neg_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_171: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_87, 3, 0, 9223372036854775807, 2);  select_87 = None
        add_514: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_170, slice_scatter_171);  slice_scatter_170 = slice_scatter_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_745: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_315, view_178);  slice_315 = view_178 = None
        add_515: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_514, mul_745);  add_514 = mul_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_172: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_314, 3, 64, 9223372036854775807);  slice_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_173: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_513, 3, 0, 64);  add_513 = None
        add_516: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_172, slice_scatter_173);  slice_scatter_172 = slice_scatter_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_174: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_316, 3, 64, 9223372036854775807);  slice_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_175: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_515, 3, 0, 64);  add_515 = None
        add_517: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_174, slice_scatter_175);  slice_scatter_174 = slice_scatter_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1026: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_267, [0, 2, 1, 3]);  getitem_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_246: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1026, memory_format = torch.contiguous_format);  permute_1026 = None
        view_1357: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_246, [1, 128, 4096]);  clone_246 = None
        view_1358: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_517, [1, 128, 4096]);  add_517 = None
        view_1359: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_516, [1, 128, 4096]);  add_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1360: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1357, [128, 4096]);  view_1357 = None
        permute_1027: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1360, [1, 0])
        mm_372: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1027, view_168);  permute_1027 = None
        permute_68: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_1029: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_373: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1360, permute_1029);  view_1360 = permute_1029 = None
        view_1361: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_373, [1, 128, 4096]);  mm_373 = None
        add_518: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1345, view_1361);  view_1345 = view_1361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1362: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1358, [128, 4096]);  view_1358 = None
        permute_1031: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1362, [1, 0])
        mm_374: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1031, view_168);  permute_1031 = None
        permute_67: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_1033: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_375: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1362, permute_1033);  view_1362 = permute_1033 = None
        view_1363: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_375, [1, 128, 4096]);  mm_375 = None
        add_519: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_518, view_1363);  add_518 = view_1363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1364: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1359, [128, 4096]);  view_1359 = None
        permute_1035: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1364, [1, 0])
        mm_376: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1035, view_168);  permute_1035 = view_168 = None
        permute_66: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_1037: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_377: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1364, permute_1037);  view_1364 = permute_1037 = None
        view_1365: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_377, [1, 128, 4096]);  mm_377 = None
        add_520: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_519, view_1365);  add_519 = view_1365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_747: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_520, primals_69);  primals_69 = None
        mul_748: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_747, 4096)
        sum_188: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True)
        mul_749: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_747, mul_60);  mul_747 = None
        sum_189: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True);  mul_749 = None
        mul_750: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_60, sum_189);  sum_189 = None
        sub_151: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_748, sum_188);  mul_748 = sum_188 = None
        sub_152: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_151, mul_750);  sub_151 = mul_750 = None
        mul_751: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_102, sub_152);  div_102 = sub_152 = None
        mul_752: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_520, mul_60);  mul_60 = None
        sum_190: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 1]);  mul_752 = None
        sum_191: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_520, [0, 1]);  add_520 = None
        add_521: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_509, mul_751);  add_509 = mul_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1366: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_521, [128, 4096])
        permute_65: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_1039: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_378: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1366, permute_1039);  permute_1039 = None
        permute_1040: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1366, [1, 0])
        mm_379: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1040, view_166);  view_166 = None
        sum_192: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1366, [0], True)
        view_1367: "f32[4096]" = torch.ops.aten.reshape.default(sum_192, [4096]);  sum_192 = None
        view_1368: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_378, [1, 128, 16384]);  mm_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_165: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 16384]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_165, 0.5)
        mul_753: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1368, mul_56);  mul_56 = None
        pow_6: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_165, 3.0)
        mul_57: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_53: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_165, mul_57);  mul_57 = None
        mul_58: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_5: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_54: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_754: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1368, add_54);  view_1368 = add_54 = None
        mul_755: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_153: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_755);  mul_755 = None
        mul_756: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_753, sub_153);  mul_753 = sub_153 = None
        mul_757: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_756, 0.7978845608028654);  mul_756 = None
        mul_758: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_757, 0.044715)
        pow_51: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_165, 2.0);  view_165 = None
        mul_759: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_51, 3.0);  pow_51 = None
        mul_760: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_758, mul_759);  mul_758 = mul_759 = None
        add_522: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_757, mul_760);  mul_757 = mul_760 = None
        mul_761: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_754, 0.5);  mul_754 = None
        add_523: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_522, mul_761);  add_522 = mul_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1369: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_523, [128, 16384]);  add_523 = None
        permute_64: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        permute_1043: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_380: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1369, permute_1043);  permute_1043 = None
        permute_1044: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1369, [1, 0])
        mm_381: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1044, view_140);  permute_1044 = None
        sum_193: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1369, [0], True);  view_1369 = None
        view_1370: "f32[16384]" = torch.ops.aten.reshape.default(sum_193, [16384]);  sum_193 = None
        view_1371: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_380, [1, 128, 4096]);  mm_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_382: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1040, view_162);  permute_1040 = view_162 = None
        permute_63: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_1049: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_383: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1366, permute_1049);  view_1366 = permute_1049 = None
        view_1373: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_383, [1, 128, 4096]);  mm_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1374: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1373, [1, 128, 16, 256]);  view_1373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1051: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1374, [0, 2, 1, 3]);  view_1374 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_1051, permute_60, permute_59, permute_58, expand_default_28, getitem_268, getitem_269, getitem_270, getitem_271, 0.0, [True, True, True, False], scale = 0.0625);  permute_1051 = permute_60 = permute_59 = permute_58 = getitem_268 = getitem_269 = getitem_270 = getitem_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_272: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_22[0]
        getitem_273: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_22[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_274: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_22[2];  _scaled_dot_product_efficient_attention_backward_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1057: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_272, [0, 2, 1, 3]);  getitem_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1058: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_273, [0, 2, 1, 3]);  getitem_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_317: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1057, 3, 0, 64)
        slice_318: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1057, 3, 64, 256);  permute_1057 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_319: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1058, 3, 0, 64)
        slice_320: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1058, 3, 64, 256);  permute_1058 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_77, [1, 128, 1, 32, 2]);  unsqueeze_77 = None
        clone_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_149: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_41, [1, 128, 1, 64]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_763: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_317, view_149)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1381: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_763, [1, 128, 16, 32, 2]);  mul_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_88: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1381, -1, 0)
        select_89: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1381, -1, 1);  view_1381 = None
        neg_124: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_88);  select_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_176: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_124, 3, 1, 9223372036854775807, 2);  neg_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_177: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_89, 3, 0, 9223372036854775807, 2);  select_89 = None
        add_524: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_176, slice_scatter_177);  slice_scatter_176 = slice_scatter_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_42: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_79, [1, 128, 1, 32, 2]);  unsqueeze_79 = None
        clone_42: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_150: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_42, [1, 128, 1, 64]);  clone_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_764: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_317, view_150);  slice_317 = None
        add_525: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_524, mul_764);  add_524 = mul_764 = None
        mul_765: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_319, view_149);  view_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1382: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_765, [1, 128, 16, 32, 2]);  mul_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_90: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1382, -1, 0)
        select_91: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1382, -1, 1);  view_1382 = None
        neg_125: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_90);  select_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_178: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_125, 3, 1, 9223372036854775807, 2);  neg_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_179: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_91, 3, 0, 9223372036854775807, 2);  select_91 = None
        add_526: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_178, slice_scatter_179);  slice_scatter_178 = slice_scatter_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_766: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_319, view_150);  slice_319 = view_150 = None
        add_527: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_526, mul_766);  add_526 = mul_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_180: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_318, 3, 64, 9223372036854775807);  slice_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_181: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_525, 3, 0, 64);  add_525 = None
        add_528: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_180, slice_scatter_181);  slice_scatter_180 = slice_scatter_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_182: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_320, 3, 64, 9223372036854775807);  slice_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_183: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_527, 3, 0, 64);  add_527 = None
        add_529: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_182, slice_scatter_183);  slice_scatter_182 = slice_scatter_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1059: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_274, [0, 2, 1, 3]);  getitem_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_247: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1059, memory_format = torch.contiguous_format);  permute_1059 = None
        view_1383: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_247, [1, 128, 4096]);  clone_247 = None
        view_1384: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_529, [1, 128, 4096]);  add_529 = None
        view_1385: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_528, [1, 128, 4096]);  add_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1386: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1383, [128, 4096]);  view_1383 = None
        permute_1060: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1386, [1, 0])
        mm_384: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1060, view_140);  permute_1060 = None
        permute_57: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_1062: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_385: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1386, permute_1062);  view_1386 = permute_1062 = None
        view_1387: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_385, [1, 128, 4096]);  mm_385 = None
        add_530: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1371, view_1387);  view_1371 = view_1387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1388: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1384, [128, 4096]);  view_1384 = None
        permute_1064: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1388, [1, 0])
        mm_386: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1064, view_140);  permute_1064 = None
        permute_56: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_1066: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_387: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1388, permute_1066);  view_1388 = permute_1066 = None
        view_1389: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_387, [1, 128, 4096]);  mm_387 = None
        add_531: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_530, view_1389);  add_530 = view_1389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1390: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1385, [128, 4096]);  view_1385 = None
        permute_1068: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1390, [1, 0])
        mm_388: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1068, view_140);  permute_1068 = view_140 = None
        permute_55: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        permute_1070: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_389: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1390, permute_1070);  view_1390 = permute_1070 = None
        view_1391: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_389, [1, 128, 4096]);  mm_389 = None
        add_532: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_531, view_1391);  add_531 = view_1391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_768: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_532, primals_58);  primals_58 = None
        mul_769: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_768, 4096)
        sum_195: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2], True)
        mul_770: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_768, mul_50);  mul_768 = None
        sum_196: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_770, [2], True);  mul_770 = None
        mul_771: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, sum_196);  sum_196 = None
        sub_155: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_769, sum_195);  mul_769 = sum_195 = None
        sub_156: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_155, mul_771);  sub_155 = mul_771 = None
        mul_772: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_104, sub_156);  div_104 = sub_156 = None
        mul_773: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_532, mul_50);  mul_50 = None
        sum_197: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_773, [0, 1]);  mul_773 = None
        sum_198: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_532, [0, 1]);  add_532 = None
        add_533: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_521, mul_772);  add_521 = mul_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1392: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_533, [128, 4096])
        permute_54: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_1072: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_390: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1392, permute_1072);  permute_1072 = None
        permute_1073: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1392, [1, 0])
        mm_391: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1073, view_138);  view_138 = None
        sum_199: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1392, [0], True)
        view_1393: "f32[4096]" = torch.ops.aten.reshape.default(sum_199, [4096]);  sum_199 = None
        view_1394: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_390, [1, 128, 16384]);  mm_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_137: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 16384]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_137, 0.5)
        mul_774: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1394, mul_46);  mul_46 = None
        pow_5: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_137, 3.0)
        mul_47: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_44: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_137, mul_47);  mul_47 = None
        mul_48: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
        tanh_4: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_45: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_775: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1394, add_45);  view_1394 = add_45 = None
        mul_776: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_157: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_776);  mul_776 = None
        mul_777: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_774, sub_157);  mul_774 = sub_157 = None
        mul_778: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_777, 0.7978845608028654);  mul_777 = None
        mul_779: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_778, 0.044715)
        pow_52: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_137, 2.0);  view_137 = None
        mul_780: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_52, 3.0);  pow_52 = None
        mul_781: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_779, mul_780);  mul_779 = mul_780 = None
        add_534: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_778, mul_781);  mul_778 = mul_781 = None
        mul_782: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_775, 0.5);  mul_775 = None
        add_535: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_534, mul_782);  add_534 = mul_782 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1395: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_535, [128, 16384]);  add_535 = None
        permute_53: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
        permute_1076: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_392: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1395, permute_1076);  permute_1076 = None
        permute_1077: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1395, [1, 0])
        mm_393: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1077, view_112);  permute_1077 = None
        sum_200: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1395, [0], True);  view_1395 = None
        view_1396: "f32[16384]" = torch.ops.aten.reshape.default(sum_200, [16384]);  sum_200 = None
        view_1397: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_392, [1, 128, 4096]);  mm_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_394: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1073, view_134);  permute_1073 = view_134 = None
        permute_52: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        permute_1082: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_395: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1392, permute_1082);  view_1392 = permute_1082 = None
        view_1399: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_395, [1, 128, 4096]);  mm_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1400: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1399, [1, 128, 16, 256]);  view_1399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1084: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1400, [0, 2, 1, 3]);  view_1400 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_1084, permute_49, permute_48, permute_47, expand_default_28, getitem_275, getitem_276, getitem_277, getitem_278, 0.0, [True, True, True, False], scale = 0.0625);  permute_1084 = permute_49 = permute_48 = permute_47 = getitem_275 = getitem_276 = getitem_277 = getitem_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_279: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_23[0]
        getitem_280: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_23[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_281: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_23[2];  _scaled_dot_product_efficient_attention_backward_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1090: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_279, [0, 2, 1, 3]);  getitem_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1091: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_280, [0, 2, 1, 3]);  getitem_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_321: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1090, 3, 0, 64)
        slice_322: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1090, 3, 64, 256);  permute_1090 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_323: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1091, 3, 0, 64)
        slice_324: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1091, 3, 64, 256);  permute_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_64, [1, 128, 1, 32, 2]);  unsqueeze_64 = None
        clone_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_121: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_33, [1, 128, 1, 64]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_784: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_321, view_121)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1407: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_784, [1, 128, 16, 32, 2]);  mul_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_92: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1407, -1, 0)
        select_93: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1407, -1, 1);  view_1407 = None
        neg_127: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_92);  select_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_184: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_127, 3, 1, 9223372036854775807, 2);  neg_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_185: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_93, 3, 0, 9223372036854775807, 2);  select_93 = None
        add_536: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_184, slice_scatter_185);  slice_scatter_184 = slice_scatter_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_34: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_66, [1, 128, 1, 32, 2]);  unsqueeze_66 = None
        clone_34: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_122: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_34, [1, 128, 1, 64]);  clone_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_785: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_321, view_122);  slice_321 = None
        add_537: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_536, mul_785);  add_536 = mul_785 = None
        mul_786: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_323, view_121);  view_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1408: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_786, [1, 128, 16, 32, 2]);  mul_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_94: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1408, -1, 0)
        select_95: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1408, -1, 1);  view_1408 = None
        neg_128: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_94);  select_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_186: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_128, 3, 1, 9223372036854775807, 2);  neg_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_187: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_95, 3, 0, 9223372036854775807, 2);  select_95 = None
        add_538: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_186, slice_scatter_187);  slice_scatter_186 = slice_scatter_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_787: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_323, view_122);  slice_323 = view_122 = None
        add_539: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_538, mul_787);  add_538 = mul_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_188: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_322, 3, 64, 9223372036854775807);  slice_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_189: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_537, 3, 0, 64);  add_537 = None
        add_540: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_188, slice_scatter_189);  slice_scatter_188 = slice_scatter_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_190: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_324, 3, 64, 9223372036854775807);  slice_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_191: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_539, 3, 0, 64);  add_539 = None
        add_541: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_190, slice_scatter_191);  slice_scatter_190 = slice_scatter_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1092: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_281, [0, 2, 1, 3]);  getitem_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_248: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1092, memory_format = torch.contiguous_format);  permute_1092 = None
        view_1409: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_248, [1, 128, 4096]);  clone_248 = None
        view_1410: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_541, [1, 128, 4096]);  add_541 = None
        view_1411: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_540, [1, 128, 4096]);  add_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1412: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1409, [128, 4096]);  view_1409 = None
        permute_1093: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1412, [1, 0])
        mm_396: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1093, view_112);  permute_1093 = None
        permute_46: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_1095: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_397: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1412, permute_1095);  view_1412 = permute_1095 = None
        view_1413: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_397, [1, 128, 4096]);  mm_397 = None
        add_542: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1397, view_1413);  view_1397 = view_1413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1414: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1410, [128, 4096]);  view_1410 = None
        permute_1097: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1414, [1, 0])
        mm_398: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1097, view_112);  permute_1097 = None
        permute_45: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_1099: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_399: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1414, permute_1099);  view_1414 = permute_1099 = None
        view_1415: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_399, [1, 128, 4096]);  mm_399 = None
        add_543: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_542, view_1415);  add_542 = view_1415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1416: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1411, [128, 4096]);  view_1411 = None
        permute_1101: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1416, [1, 0])
        mm_400: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1101, view_112);  permute_1101 = view_112 = None
        permute_44: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        permute_1103: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_401: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1416, permute_1103);  view_1416 = permute_1103 = None
        view_1417: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_401, [1, 128, 4096]);  mm_401 = None
        add_544: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_543, view_1417);  add_543 = view_1417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_789: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_544, primals_47);  primals_47 = None
        mul_790: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_789, 4096)
        sum_202: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_789, [2], True)
        mul_791: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_789, mul_40);  mul_789 = None
        sum_203: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_791, [2], True);  mul_791 = None
        mul_792: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_40, sum_203);  sum_203 = None
        sub_159: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_790, sum_202);  mul_790 = sum_202 = None
        sub_160: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_159, mul_792);  sub_159 = mul_792 = None
        mul_793: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_106, sub_160);  div_106 = sub_160 = None
        mul_794: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_544, mul_40);  mul_40 = None
        sum_204: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_794, [0, 1]);  mul_794 = None
        sum_205: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_544, [0, 1]);  add_544 = None
        add_545: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_533, mul_793);  add_533 = mul_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1418: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_545, [128, 4096])
        permute_43: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_1105: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_402: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1418, permute_1105);  permute_1105 = None
        permute_1106: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1418, [1, 0])
        mm_403: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1106, view_110);  view_110 = None
        sum_206: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1418, [0], True)
        view_1419: "f32[4096]" = torch.ops.aten.reshape.default(sum_206, [4096]);  sum_206 = None
        view_1420: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_402, [1, 128, 16384]);  mm_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_109: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 16384]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_36: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_795: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1420, mul_36);  mul_36 = None
        pow_4: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
        mul_37: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_35: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_109, mul_37);  mul_37 = None
        mul_38: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_35, 0.7978845608028654);  add_35 = None
        tanh_3: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
        add_36: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_796: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1420, add_36);  view_1420 = add_36 = None
        mul_797: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_161: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_797);  mul_797 = None
        mul_798: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_795, sub_161);  mul_795 = sub_161 = None
        mul_799: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_798, 0.7978845608028654);  mul_798 = None
        mul_800: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_799, 0.044715)
        pow_53: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 2.0);  view_109 = None
        mul_801: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_53, 3.0);  pow_53 = None
        mul_802: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
        add_546: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_799, mul_802);  mul_799 = mul_802 = None
        mul_803: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_796, 0.5);  mul_796 = None
        add_547: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_546, mul_803);  add_546 = mul_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1421: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_547, [128, 16384]);  add_547 = None
        permute_42: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_1109: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_404: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1421, permute_1109);  permute_1109 = None
        permute_1110: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1421, [1, 0])
        mm_405: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1110, view_84);  permute_1110 = None
        sum_207: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1421, [0], True);  view_1421 = None
        view_1422: "f32[16384]" = torch.ops.aten.reshape.default(sum_207, [16384]);  sum_207 = None
        view_1423: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_404, [1, 128, 4096]);  mm_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_406: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1106, view_106);  permute_1106 = view_106 = None
        permute_41: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_1115: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_407: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1418, permute_1115);  view_1418 = permute_1115 = None
        view_1425: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_407, [1, 128, 4096]);  mm_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1426: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1425, [1, 128, 16, 256]);  view_1425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1117: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1426, [0, 2, 1, 3]);  view_1426 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_24 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_1117, permute_38, permute_37, permute_36, expand_default_28, getitem_282, getitem_283, getitem_284, getitem_285, 0.0, [True, True, True, False], scale = 0.0625);  permute_1117 = permute_38 = permute_37 = permute_36 = getitem_282 = getitem_283 = getitem_284 = getitem_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_286: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_24[0]
        getitem_287: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_24[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_288: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_24[2];  _scaled_dot_product_efficient_attention_backward_default_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1123: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_286, [0, 2, 1, 3]);  getitem_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1124: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_287, [0, 2, 1, 3]);  getitem_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_325: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1123, 3, 0, 64)
        slice_326: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1123, 3, 64, 256);  permute_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_327: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1124, 3, 0, 64)
        slice_328: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1124, 3, 64, 256);  permute_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_51, [1, 128, 1, 32, 2]);  unsqueeze_51 = None
        clone_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_93: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_25, [1, 128, 1, 64]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_805: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_325, view_93)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1433: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_805, [1, 128, 16, 32, 2]);  mul_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_96: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1433, -1, 0)
        select_97: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1433, -1, 1);  view_1433 = None
        neg_130: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_96);  select_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_192: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_130, 3, 1, 9223372036854775807, 2);  neg_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_193: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_97, 3, 0, 9223372036854775807, 2);  select_97 = None
        add_548: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_192, slice_scatter_193);  slice_scatter_192 = slice_scatter_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_26: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_53, [1, 128, 1, 32, 2]);  unsqueeze_53 = None
        clone_26: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_94: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_26, [1, 128, 1, 64]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_806: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_325, view_94);  slice_325 = None
        add_549: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_548, mul_806);  add_548 = mul_806 = None
        mul_807: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_327, view_93);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1434: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_807, [1, 128, 16, 32, 2]);  mul_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_98: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1434, -1, 0)
        select_99: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1434, -1, 1);  view_1434 = None
        neg_131: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_98);  select_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_194: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_131, 3, 1, 9223372036854775807, 2);  neg_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_195: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_99, 3, 0, 9223372036854775807, 2);  select_99 = None
        add_550: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_194, slice_scatter_195);  slice_scatter_194 = slice_scatter_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_808: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_327, view_94);  slice_327 = view_94 = None
        add_551: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_550, mul_808);  add_550 = mul_808 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_196: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_326, 3, 64, 9223372036854775807);  slice_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_197: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_549, 3, 0, 64);  add_549 = None
        add_552: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_196, slice_scatter_197);  slice_scatter_196 = slice_scatter_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_198: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_328, 3, 64, 9223372036854775807);  slice_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_199: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_551, 3, 0, 64);  add_551 = None
        add_553: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_198, slice_scatter_199);  slice_scatter_198 = slice_scatter_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1125: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_288, [0, 2, 1, 3]);  getitem_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_249: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1125, memory_format = torch.contiguous_format);  permute_1125 = None
        view_1435: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_249, [1, 128, 4096]);  clone_249 = None
        view_1436: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_553, [1, 128, 4096]);  add_553 = None
        view_1437: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_552, [1, 128, 4096]);  add_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1438: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1435, [128, 4096]);  view_1435 = None
        permute_1126: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1438, [1, 0])
        mm_408: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1126, view_84);  permute_1126 = None
        permute_35: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_1128: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_409: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1438, permute_1128);  view_1438 = permute_1128 = None
        view_1439: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_409, [1, 128, 4096]);  mm_409 = None
        add_554: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1423, view_1439);  view_1423 = view_1439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1440: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1436, [128, 4096]);  view_1436 = None
        permute_1130: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1440, [1, 0])
        mm_410: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1130, view_84);  permute_1130 = None
        permute_34: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_1132: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_411: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1440, permute_1132);  view_1440 = permute_1132 = None
        view_1441: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_411, [1, 128, 4096]);  mm_411 = None
        add_555: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_554, view_1441);  add_554 = view_1441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1442: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1437, [128, 4096]);  view_1437 = None
        permute_1134: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1442, [1, 0])
        mm_412: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1134, view_84);  permute_1134 = view_84 = None
        permute_33: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_1136: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_413: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1442, permute_1136);  view_1442 = permute_1136 = None
        view_1443: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_413, [1, 128, 4096]);  mm_413 = None
        add_556: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_555, view_1443);  add_555 = view_1443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_810: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_556, primals_36);  primals_36 = None
        mul_811: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_810, 4096)
        sum_209: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_810, [2], True)
        mul_812: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_810, mul_30);  mul_810 = None
        sum_210: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_812, [2], True);  mul_812 = None
        mul_813: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, sum_210);  sum_210 = None
        sub_163: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_811, sum_209);  mul_811 = sum_209 = None
        sub_164: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_163, mul_813);  sub_163 = mul_813 = None
        mul_814: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_108, sub_164);  div_108 = sub_164 = None
        mul_815: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_556, mul_30);  mul_30 = None
        sum_211: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 1]);  mul_815 = None
        sum_212: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_556, [0, 1]);  add_556 = None
        add_557: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_545, mul_814);  add_545 = mul_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1444: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_557, [128, 4096])
        permute_32: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_1138: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_414: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1444, permute_1138);  permute_1138 = None
        permute_1139: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1444, [1, 0])
        mm_415: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1139, view_82);  view_82 = None
        sum_213: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1444, [0], True)
        view_1445: "f32[4096]" = torch.ops.aten.reshape.default(sum_213, [4096]);  sum_213 = None
        view_1446: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_414, [1, 128, 16384]);  mm_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_81: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 16384]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_26: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_81, 0.5)
        mul_816: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1446, mul_26);  mul_26 = None
        pow_3: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_81, 3.0)
        mul_27: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_26: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_81, mul_27);  mul_27 = None
        mul_28: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_26, 0.7978845608028654);  add_26 = None
        tanh_2: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
        add_27: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_817: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1446, add_27);  view_1446 = add_27 = None
        mul_818: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_165: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_818);  mul_818 = None
        mul_819: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_816, sub_165);  mul_816 = sub_165 = None
        mul_820: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_819, 0.7978845608028654);  mul_819 = None
        mul_821: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_820, 0.044715)
        pow_54: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_81, 2.0);  view_81 = None
        mul_822: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_54, 3.0);  pow_54 = None
        mul_823: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_821, mul_822);  mul_821 = mul_822 = None
        add_558: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_820, mul_823);  mul_820 = mul_823 = None
        mul_824: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_817, 0.5);  mul_817 = None
        add_559: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_558, mul_824);  add_558 = mul_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1447: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_559, [128, 16384]);  add_559 = None
        permute_31: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_1142: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_416: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1447, permute_1142);  permute_1142 = None
        permute_1143: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1447, [1, 0])
        mm_417: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1143, view_56);  permute_1143 = None
        sum_214: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1447, [0], True);  view_1447 = None
        view_1448: "f32[16384]" = torch.ops.aten.reshape.default(sum_214, [16384]);  sum_214 = None
        view_1449: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_416, [1, 128, 4096]);  mm_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_418: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1139, view_78);  permute_1139 = view_78 = None
        permute_30: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_1148: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_419: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1444, permute_1148);  view_1444 = permute_1148 = None
        view_1451: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_419, [1, 128, 4096]);  mm_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1452: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1451, [1, 128, 16, 256]);  view_1451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1150: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1452, [0, 2, 1, 3]);  view_1452 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_25 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_1150, permute_27, permute_26, permute_25, expand_default_28, getitem_289, getitem_290, getitem_291, getitem_292, 0.0, [True, True, True, False], scale = 0.0625);  permute_1150 = permute_27 = permute_26 = permute_25 = getitem_289 = getitem_290 = getitem_291 = getitem_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_293: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_25[0]
        getitem_294: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_25[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_295: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_25[2];  _scaled_dot_product_efficient_attention_backward_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1156: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_293, [0, 2, 1, 3]);  getitem_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1157: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_294, [0, 2, 1, 3]);  getitem_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_329: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1156, 3, 0, 64)
        slice_330: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1156, 3, 64, 256);  permute_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_331: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1157, 3, 0, 64)
        slice_332: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1157, 3, 64, 256);  permute_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_38, [1, 128, 1, 32, 2]);  unsqueeze_38 = None
        clone_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_65: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_17, [1, 128, 1, 64]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_826: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_329, view_65)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1459: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_826, [1, 128, 16, 32, 2]);  mul_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_100: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1459, -1, 0)
        select_101: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1459, -1, 1);  view_1459 = None
        neg_133: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_100);  select_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_200: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_133, 3, 1, 9223372036854775807, 2);  neg_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_201: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_101, 3, 0, 9223372036854775807, 2);  select_101 = None
        add_560: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_200, slice_scatter_201);  slice_scatter_200 = slice_scatter_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_18: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_40, [1, 128, 1, 32, 2]);  unsqueeze_40 = None
        clone_18: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_66: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_18, [1, 128, 1, 64]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_827: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_329, view_66);  slice_329 = None
        add_561: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_560, mul_827);  add_560 = mul_827 = None
        mul_828: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_331, view_65);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1460: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_828, [1, 128, 16, 32, 2]);  mul_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_102: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1460, -1, 0)
        select_103: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1460, -1, 1);  view_1460 = None
        neg_134: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_102);  select_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_202: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_134, 3, 1, 9223372036854775807, 2);  neg_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_203: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_103, 3, 0, 9223372036854775807, 2);  select_103 = None
        add_562: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_202, slice_scatter_203);  slice_scatter_202 = slice_scatter_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_829: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_331, view_66);  slice_331 = view_66 = None
        add_563: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_562, mul_829);  add_562 = mul_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_204: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_330, 3, 64, 9223372036854775807);  slice_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_205: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_561, 3, 0, 64);  add_561 = None
        add_564: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_204, slice_scatter_205);  slice_scatter_204 = slice_scatter_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_206: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_332, 3, 64, 9223372036854775807);  slice_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_207: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_563, 3, 0, 64);  add_563 = None
        add_565: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_206, slice_scatter_207);  slice_scatter_206 = slice_scatter_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1158: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_295, [0, 2, 1, 3]);  getitem_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_250: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1158, memory_format = torch.contiguous_format);  permute_1158 = None
        view_1461: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_250, [1, 128, 4096]);  clone_250 = None
        view_1462: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_565, [1, 128, 4096]);  add_565 = None
        view_1463: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_564, [1, 128, 4096]);  add_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1464: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1461, [128, 4096]);  view_1461 = None
        permute_1159: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1464, [1, 0])
        mm_420: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1159, view_56);  permute_1159 = None
        permute_24: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_1161: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_421: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1464, permute_1161);  view_1464 = permute_1161 = None
        view_1465: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_421, [1, 128, 4096]);  mm_421 = None
        add_566: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1449, view_1465);  view_1449 = view_1465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1466: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1462, [128, 4096]);  view_1462 = None
        permute_1163: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1466, [1, 0])
        mm_422: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1163, view_56);  permute_1163 = None
        permute_23: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_1165: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_423: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1466, permute_1165);  view_1466 = permute_1165 = None
        view_1467: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_423, [1, 128, 4096]);  mm_423 = None
        add_567: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_566, view_1467);  add_566 = view_1467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1468: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1463, [128, 4096]);  view_1463 = None
        permute_1167: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1468, [1, 0])
        mm_424: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1167, view_56);  permute_1167 = view_56 = None
        permute_22: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_1169: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_425: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1468, permute_1169);  view_1468 = permute_1169 = None
        view_1469: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_425, [1, 128, 4096]);  mm_425 = None
        add_568: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_567, view_1469);  add_567 = view_1469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_831: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_568, primals_25);  primals_25 = None
        mul_832: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_831, 4096)
        sum_216: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [2], True)
        mul_833: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_831, mul_20);  mul_831 = None
        sum_217: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True);  mul_833 = None
        mul_834: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_20, sum_217);  sum_217 = None
        sub_167: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_832, sum_216);  mul_832 = sum_216 = None
        sub_168: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_167, mul_834);  sub_167 = mul_834 = None
        mul_835: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_110, sub_168);  div_110 = sub_168 = None
        mul_836: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_568, mul_20);  mul_20 = None
        sum_218: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1]);  mul_836 = None
        sum_219: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_568, [0, 1]);  add_568 = None
        add_569: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_557, mul_835);  add_557 = mul_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1470: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_569, [128, 4096])
        permute_21: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_1171: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_426: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1470, permute_1171);  permute_1171 = None
        permute_1172: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1470, [1, 0])
        mm_427: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1172, view_54);  view_54 = None
        sum_220: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1470, [0], True)
        view_1471: "f32[4096]" = torch.ops.aten.reshape.default(sum_220, [4096]);  sum_220 = None
        view_1472: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_426, [1, 128, 16384]);  mm_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_53: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 16384]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_16: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_53, 0.5)
        mul_837: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1472, mul_16);  mul_16 = None
        pow_2: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 3.0)
        mul_17: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_53, mul_17);  mul_17 = None
        mul_18: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
        add_18: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_838: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1472, add_18);  view_1472 = add_18 = None
        mul_839: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_169: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_839);  mul_839 = None
        mul_840: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_837, sub_169);  mul_837 = sub_169 = None
        mul_841: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_840, 0.7978845608028654);  mul_840 = None
        mul_842: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_841, 0.044715)
        pow_55: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_53, 2.0);  view_53 = None
        mul_843: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_55, 3.0);  pow_55 = None
        mul_844: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_842, mul_843);  mul_842 = mul_843 = None
        add_570: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_841, mul_844);  mul_841 = mul_844 = None
        mul_845: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_838, 0.5);  mul_838 = None
        add_571: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_570, mul_845);  add_570 = mul_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1473: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_571, [128, 16384]);  add_571 = None
        permute_20: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_1175: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_428: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1473, permute_1175);  permute_1175 = None
        permute_1176: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1473, [1, 0])
        mm_429: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1176, view_28);  permute_1176 = None
        sum_221: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1473, [0], True);  view_1473 = None
        view_1474: "f32[16384]" = torch.ops.aten.reshape.default(sum_221, [16384]);  sum_221 = None
        view_1475: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_428, [1, 128, 4096]);  mm_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_430: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1172, view_50);  permute_1172 = view_50 = None
        permute_19: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_1181: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_431: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1470, permute_1181);  view_1470 = permute_1181 = None
        view_1477: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_431, [1, 128, 4096]);  mm_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1478: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1477, [1, 128, 16, 256]);  view_1477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1183: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1478, [0, 2, 1, 3]);  view_1478 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_26 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_1183, permute_16, permute_15, permute_14, expand_default_28, getitem_296, getitem_297, getitem_298, getitem_299, 0.0, [True, True, True, False], scale = 0.0625);  permute_1183 = permute_16 = permute_15 = permute_14 = getitem_296 = getitem_297 = getitem_298 = getitem_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_300: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_26[0]
        getitem_301: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_26[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_302: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_26[2];  _scaled_dot_product_efficient_attention_backward_default_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1189: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_300, [0, 2, 1, 3]);  getitem_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1190: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_301, [0, 2, 1, 3]);  getitem_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_333: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1189, 3, 0, 64)
        slice_334: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1189, 3, 64, 256);  permute_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_335: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1190, 3, 0, 64)
        slice_336: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1190, 3, 64, 256);  permute_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_25, [1, 128, 1, 32, 2]);  unsqueeze_25 = None
        clone_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_37: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_9, [1, 128, 1, 64]);  clone_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_847: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_333, view_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1485: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_847, [1, 128, 16, 32, 2]);  mul_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_104: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1485, -1, 0)
        select_105: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1485, -1, 1);  view_1485 = None
        neg_136: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_104);  select_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_208: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_136, 3, 1, 9223372036854775807, 2);  neg_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_209: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_105, 3, 0, 9223372036854775807, 2);  select_105 = None
        add_572: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_208, slice_scatter_209);  slice_scatter_208 = slice_scatter_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_10: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_27, [1, 128, 1, 32, 2]);  unsqueeze_27 = None
        clone_10: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_38: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_10, [1, 128, 1, 64]);  clone_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_848: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_333, view_38);  slice_333 = None
        add_573: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_572, mul_848);  add_572 = mul_848 = None
        mul_849: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_335, view_37);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1486: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_849, [1, 128, 16, 32, 2]);  mul_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_106: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1486, -1, 0)
        select_107: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1486, -1, 1);  view_1486 = None
        neg_137: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_106);  select_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_210: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_137, 3, 1, 9223372036854775807, 2);  neg_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_211: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_107, 3, 0, 9223372036854775807, 2);  select_107 = None
        add_574: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_210, slice_scatter_211);  slice_scatter_210 = slice_scatter_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_850: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_335, view_38);  slice_335 = view_38 = None
        add_575: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_574, mul_850);  add_574 = mul_850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_212: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_334, 3, 64, 9223372036854775807);  slice_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_213: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_573, 3, 0, 64);  add_573 = None
        add_576: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_212, slice_scatter_213);  slice_scatter_212 = slice_scatter_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_214: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_336, 3, 64, 9223372036854775807);  slice_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_215: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_575, 3, 0, 64);  add_575 = None
        add_577: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_214, slice_scatter_215);  slice_scatter_214 = slice_scatter_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1191: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_302, [0, 2, 1, 3]);  getitem_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_251: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1191, memory_format = torch.contiguous_format);  permute_1191 = None
        view_1487: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_251, [1, 128, 4096]);  clone_251 = None
        view_1488: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_577, [1, 128, 4096]);  add_577 = None
        view_1489: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_576, [1, 128, 4096]);  add_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1490: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1487, [128, 4096]);  view_1487 = None
        permute_1192: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1490, [1, 0])
        mm_432: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1192, view_28);  permute_1192 = None
        permute_13: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_1194: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_433: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1490, permute_1194);  view_1490 = permute_1194 = None
        view_1491: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_433, [1, 128, 4096]);  mm_433 = None
        add_578: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1475, view_1491);  view_1475 = view_1491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1492: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1488, [128, 4096]);  view_1488 = None
        permute_1196: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1492, [1, 0])
        mm_434: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1196, view_28);  permute_1196 = None
        permute_12: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_1198: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_435: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1492, permute_1198);  view_1492 = permute_1198 = None
        view_1493: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_435, [1, 128, 4096]);  mm_435 = None
        add_579: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_578, view_1493);  add_578 = view_1493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1494: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1489, [128, 4096]);  view_1489 = None
        permute_1200: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1494, [1, 0])
        mm_436: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1200, view_28);  permute_1200 = view_28 = None
        permute_11: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_1202: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_437: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1494, permute_1202);  view_1494 = permute_1202 = None
        view_1495: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_437, [1, 128, 4096]);  mm_437 = None
        add_580: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_579, view_1495);  add_579 = view_1495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_852: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_580, primals_14);  primals_14 = None
        mul_853: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_852, 4096)
        sum_223: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [2], True)
        mul_854: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_852, mul_10);  mul_852 = None
        sum_224: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_854, [2], True);  mul_854 = None
        mul_855: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_10, sum_224);  sum_224 = None
        sub_171: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_853, sum_223);  mul_853 = sum_223 = None
        sub_172: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_171, mul_855);  sub_171 = mul_855 = None
        mul_856: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_112, sub_172);  div_112 = sub_172 = None
        mul_857: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_580, mul_10);  mul_10 = None
        sum_225: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 1]);  mul_857 = None
        sum_226: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_580, [0, 1]);  add_580 = None
        add_581: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_569, mul_856);  add_569 = mul_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        view_1496: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_581, [128, 4096])
        permute_10: "f32[16384, 4096]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_1204: "f32[4096, 16384]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_438: "f32[128, 16384]" = torch.ops.aten.mm.default(view_1496, permute_1204);  permute_1204 = None
        permute_1205: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1496, [1, 0])
        mm_439: "f32[4096, 16384]" = torch.ops.aten.mm.default(permute_1205, view_26);  view_26 = None
        sum_227: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1496, [0], True)
        view_1497: "f32[4096]" = torch.ops.aten.reshape.default(sum_227, [4096]);  sum_227 = None
        view_1498: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(mm_438, [1, 128, 16384]);  mm_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_25: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm, [1, 128, 16384]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_6: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_25, 0.5)
        mul_858: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1498, mul_6);  mul_6 = None
        pow_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_25, 3.0)
        mul_7: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_8: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_25, mul_7);  mul_7 = None
        mul_8: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
        tanh: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
        add_9: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_859: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_1498, add_9);  view_1498 = add_9 = None
        mul_860: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_173: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_860);  mul_860 = None
        mul_861: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_858, sub_173);  mul_858 = sub_173 = None
        mul_862: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_861, 0.7978845608028654);  mul_861 = None
        mul_863: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_862, 0.044715)
        pow_56: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_25, 2.0);  view_25 = None
        mul_864: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_56, 3.0);  pow_56 = None
        mul_865: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_863, mul_864);  mul_863 = mul_864 = None
        add_582: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(mul_862, mul_865);  mul_862 = mul_865 = None
        mul_866: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_859, 0.5);  mul_859 = None
        add_583: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_582, mul_866);  add_582 = mul_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        view_1499: "f32[128, 16384]" = torch.ops.aten.reshape.default(add_583, [128, 16384]);  add_583 = None
        permute_9: "f32[4096, 16384]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_1208: "f32[16384, 4096]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_440: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1499, permute_1208);  permute_1208 = None
        permute_1209: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1499, [1, 0])
        mm_441: "f32[16384, 4096]" = torch.ops.aten.mm.default(permute_1209, view);  permute_1209 = None
        sum_228: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1499, [0], True);  view_1499 = None
        view_1500: "f32[16384]" = torch.ops.aten.reshape.default(sum_228, [16384]);  sum_228 = None
        view_1501: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_440, [1, 128, 4096]);  mm_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:222 in forward, code: attn_output = self.out_proj(attn_output)
        mm_442: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1205, view_22);  permute_1205 = view_22 = None
        permute_8: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_1214: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_443: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1496, permute_1214);  view_1496 = permute_1214 = None
        view_1503: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_443, [1, 128, 4096]);  mm_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:134 in _merge_heads, code: return tensor.view(new_shape)
        view_1504: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_1503, [1, 128, 16, 256]);  view_1503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:130 in _merge_heads, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
        permute_1216: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_1504, [0, 2, 1, 3]);  view_1504 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_27 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_1216, permute_5, permute_4, permute_3, expand_default_28, getitem_303, getitem_304, getitem_305, getitem_306, 0.0, [True, True, True, False], scale = 0.0625);  permute_1216 = permute_5 = permute_4 = permute_3 = expand_default_28 = getitem_303 = getitem_304 = getitem_305 = getitem_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:147 in _attn, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
        getitem_307: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_27[0]
        getitem_308: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_27[1]

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:157 in _attn, code: attn_output = torch.matmul(attn_weights, value)
        getitem_309: "f32[1, 16, 128, 256]" = _scaled_dot_product_efficient_attention_backward_default_27[2];  _scaled_dot_product_efficient_attention_backward_default_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:213 in forward, code: query = query.permute(0, 2, 1, 3)
        permute_1222: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_307, [0, 2, 1, 3]);  getitem_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:212 in forward, code: key = key.permute(0, 2, 1, 3)
        permute_1223: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_308, [0, 2, 1, 3]);  getitem_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:207 in forward, code: query = torch.cat([q_rot, q_pass], dim=-1)
        slice_337: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1222, 3, 0, 64)
        slice_338: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1222, 3, 64, 256);  permute_1222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:206 in forward, code: key = torch.cat([k_rot, k_pass], dim=-1)
        slice_339: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_1223, 3, 0, 64)
        slice_340: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_1223, 3, 64, 256);  permute_1223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:65 in apply_rotary_pos_emb, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
        expand_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_12, [1, 128, 1, 32, 2]);  unsqueeze_12 = None
        clone_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_1, [1, 128, 1, 64]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_868: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_337, view_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1511: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_868, [1, 128, 16, 32, 2]);  mul_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_108: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1511, -1, 0)
        select_109: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1511, -1, 1);  view_1511 = None
        neg_139: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_108);  select_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_216: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_139, 3, 1, 9223372036854775807, 2);  neg_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_217: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_109, 3, 0, 9223372036854775807, 2);  select_109 = None
        add_584: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_216, slice_scatter_217);  slice_scatter_216 = slice_scatter_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:66 in apply_rotary_pos_emb, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
        expand_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_14, [1, 128, 1, 32, 2]);  unsqueeze_14 = None
        clone_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_2, [1, 128, 1, 64]);  clone_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_869: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_337, view_10);  slice_337 = None
        add_585: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_584, mul_869);  add_584 = mul_869 = None
        mul_870: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_339, view_9);  view_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:61 in rotate_every_two, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
        view_1512: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.reshape.default(mul_870, [1, 128, 16, 32, 2]);  mul_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:60 in rotate_every_two, code: x = torch.stack((-x2, x1), dim=-1)
        select_110: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1512, -1, 0)
        select_111: "f32[1, 128, 16, 32]" = torch.ops.aten.select.int(view_1512, -1, 1);  view_1512 = None
        neg_140: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_110);  select_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:59 in rotate_every_two, code: x2 = x[:, :, :, 1::2]
        slice_scatter_218: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, neg_140, 3, 1, 9223372036854775807, 2);  neg_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:58 in rotate_every_two, code: x1 = x[:, :, :, ::2]
        slice_scatter_219: "f32[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_default_8, select_111, 3, 0, 9223372036854775807, 2);  full_default_8 = select_111 = None
        add_586: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_218, slice_scatter_219);  slice_scatter_218 = slice_scatter_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:67 in apply_rotary_pos_emb, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
        mul_871: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_339, view_10);  slice_339 = view_10 = None
        add_587: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_586, mul_871);  add_586 = mul_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:201 in forward, code: q_pass = query[:, :, :, self.rotary_dim :]
        slice_scatter_220: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_338, 3, 64, 9223372036854775807);  slice_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:200 in forward, code: q_rot = query[:, :, :, : self.rotary_dim]
        slice_scatter_221: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_585, 3, 0, 64);  add_585 = None
        add_588: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_220, slice_scatter_221);  slice_scatter_220 = slice_scatter_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:198 in forward, code: k_pass = key[:, :, :, self.rotary_dim :]
        slice_scatter_222: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_340, 3, 64, 9223372036854775807);  slice_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:197 in forward, code: k_rot = key[:, :, :, : self.rotary_dim]
        slice_scatter_223: "f32[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_default_12, add_587, 3, 0, 64);  full_default_12 = add_587 = None
        add_589: "f32[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_222, slice_scatter_223);  slice_scatter_222 = slice_scatter_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:119 in _split_heads, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        permute_1224: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(getitem_309, [0, 2, 1, 3]);  getitem_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:113 in _split_heads, code: tensor = tensor.view(new_shape)
        clone_252: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_1224, memory_format = torch.contiguous_format);  permute_1224 = None
        view_1513: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_252, [1, 128, 4096]);  clone_252 = None
        view_1514: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_589, [1, 128, 4096]);  add_589 = None
        view_1515: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_588, [1, 128, 4096]);  add_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        view_1516: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1513, [128, 4096]);  view_1513 = None
        permute_1225: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1516, [1, 0])
        mm_444: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1225, view);  permute_1225 = None
        permute_2: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        permute_1227: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_445: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1516, permute_1227);  view_1516 = permute_1227 = None
        view_1517: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_445, [1, 128, 4096]);  mm_445 = None
        add_590: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_1501, view_1517);  view_1501 = view_1517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        view_1518: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1514, [128, 4096]);  view_1514 = None
        permute_1229: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1518, [1, 0])
        mm_446: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1229, view);  permute_1229 = None
        permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_1231: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        mm_447: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1518, permute_1231);  view_1518 = permute_1231 = None
        view_1519: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_447, [1, 128, 4096]);  mm_447 = None
        add_591: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_590, view_1519);  add_590 = view_1519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        view_1520: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_1515, [128, 4096]);  view_1515 = None
        permute_1233: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1520, [1, 0])
        mm_448: "f32[4096, 4096]" = torch.ops.aten.mm.default(permute_1233, view);  permute_1233 = view = None
        permute: "f32[4096, 4096]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_1235: "f32[4096, 4096]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_449: "f32[128, 4096]" = torch.ops.aten.mm.default(view_1520, permute_1235);  view_1520 = permute_1235 = None
        view_1521: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_449, [1, 128, 4096]);  mm_449 = None
        add_592: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_591, view_1521);  add_591 = view_1521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_873: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_592, primals_3);  primals_3 = None
        mul_874: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_873, 4096)
        sum_230: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_873, [2], True)
        sub_2: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding, getitem_1);  embedding = getitem_1 = None
        mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        mul_875: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_873, mul);  mul_873 = None
        sum_231: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_875, [2], True);  mul_875 = None
        mul_876: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, sum_231);  sum_231 = None
        sub_175: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_874, sum_230);  mul_874 = sum_230 = None
        sub_176: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_175, mul_876);  sub_175 = mul_876 = None
        div_114: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(rsqrt, 4096);  rsqrt = None
        mul_877: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_114, sub_176);  div_114 = sub_176 = None
        mul_878: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_592, mul);  mul = None
        sum_232: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 1]);  mul_878 = None
        sum_233: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_592, [0, 1]);  add_592 = None
        add_593: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_581, mul_877);  add_581 = mul_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:494 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_1: "b8[1, 128]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_376: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_5: "f32[1, 128, 4096]" = torch.ops.aten.where.self(unsqueeze_376, full_default_1, add_593);  unsqueeze_376 = full_default_1 = add_593 = None
        full_default_233: "f32[50400, 4096]" = torch.ops.aten.full.default([50400, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[50400, 4096]" = torch.ops.aten.index_put.default(full_default_233, [primals_1], where_5, True);  full_default_233 = primals_1 = where_5 = None
        return (None, index_put, sum_232, sum_233, mm_448, mm_446, mm_444, None, mm_442, mm_441, view_1500, mm_439, view_1497, sum_225, sum_226, mm_436, mm_434, mm_432, None, mm_430, mm_429, view_1474, mm_427, view_1471, sum_218, sum_219, mm_424, mm_422, mm_420, None, mm_418, mm_417, view_1448, mm_415, view_1445, sum_211, sum_212, mm_412, mm_410, mm_408, None, mm_406, mm_405, view_1422, mm_403, view_1419, sum_204, sum_205, mm_400, mm_398, mm_396, None, mm_394, mm_393, view_1396, mm_391, view_1393, sum_197, sum_198, mm_388, mm_386, mm_384, None, mm_382, mm_381, view_1370, mm_379, view_1367, sum_190, sum_191, mm_376, mm_374, mm_372, None, mm_370, mm_369, view_1344, mm_367, view_1341, sum_183, sum_184, mm_364, mm_362, mm_360, None, mm_358, mm_357, view_1318, mm_355, view_1315, sum_176, sum_177, mm_352, mm_350, mm_348, None, mm_346, mm_345, view_1292, mm_343, view_1289, sum_169, sum_170, mm_340, mm_338, mm_336, None, mm_334, mm_333, view_1266, mm_331, view_1263, sum_162, sum_163, mm_328, mm_326, mm_324, None, mm_322, mm_321, view_1240, mm_319, view_1237, sum_155, sum_156, mm_316, mm_314, mm_312, None, mm_310, mm_309, view_1214, mm_307, view_1211, sum_148, sum_149, mm_304, mm_302, mm_300, None, mm_298, mm_297, view_1188, mm_295, view_1185, sum_141, sum_142, mm_292, mm_290, mm_288, None, mm_286, mm_285, view_1162, mm_283, view_1159, sum_134, sum_135, mm_280, mm_278, mm_276, None, mm_274, mm_273, view_1136, mm_271, view_1133, sum_127, sum_128, mm_268, mm_266, mm_264, None, mm_262, mm_261, view_1110, mm_259, view_1107, sum_120, sum_121, mm_256, mm_254, mm_252, None, mm_250, mm_249, view_1084, mm_247, view_1081, sum_113, sum_114, mm_244, mm_242, mm_240, None, mm_238, mm_237, view_1058, mm_235, view_1055, sum_106, sum_107, mm_232, mm_230, mm_228, None, mm_226, mm_225, view_1032, mm_223, view_1029, sum_99, sum_100, mm_220, mm_218, mm_216, None, mm_214, mm_213, view_1006, mm_211, view_1003, sum_92, sum_93, mm_208, mm_206, mm_204, None, mm_202, mm_201, view_980, mm_199, view_977, sum_85, sum_86, mm_196, mm_194, mm_192, None, mm_190, mm_189, view_954, mm_187, view_951, sum_78, sum_79, mm_184, mm_182, mm_180, None, mm_178, mm_177, view_928, mm_175, view_925, sum_71, sum_72, mm_172, mm_170, mm_168, None, mm_166, mm_165, view_902, mm_163, view_899, sum_64, sum_65, mm_160, mm_158, mm_156, None, mm_154, mm_153, view_876, mm_151, view_873, sum_57, sum_58, mm_148, mm_146, mm_144, None, mm_142, mm_141, view_850, mm_139, view_847, sum_50, sum_51, mm_136, mm_134, mm_132, None, mm_130, mm_129, view_824, mm_127, view_821, sum_43, sum_44, mm_124, mm_122, mm_120, None, mm_118, mm_117, view_798, mm_115, view_795, sum_36, sum_37, mm_113, view_791, None)
