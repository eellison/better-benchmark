class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[16, 512][512, 1]cuda:0", primals_2: "i64[16, 512][512, 1]cuda:0", primals_3: "f32[29056, 1024][1024, 1]cuda:0", primals_4: "i64[1, 512][512, 1]cuda:0", primals_5: "f32[2, 1024][1024, 1]cuda:0", primals_6: "f32[512, 1024][1024, 1]cuda:0", primals_7: "f32[1024][1]cuda:0", primals_8: "f32[1024][1]cuda:0", primals_9: "f32[1024, 1024][1024, 1]cuda:0", primals_10: "f32[1024][1]cuda:0", primals_11: "f32[1024, 1024][1024, 1]cuda:0", primals_12: "f32[1024][1]cuda:0", primals_13: "f32[1024, 1024][1024, 1]cuda:0", primals_14: "f32[1024][1]cuda:0", primals_15: "f32[1024, 1024][1024, 1]cuda:0", primals_16: "f32[1024][1]cuda:0", primals_17: "f32[1024][1]cuda:0", primals_18: "f32[1024][1]cuda:0", primals_19: "f32[4096, 1024][1024, 1]cuda:0", primals_20: "f32[4096][1]cuda:0", primals_21: "f32[1024, 4096][4096, 1]cuda:0", primals_22: "f32[1024][1]cuda:0", primals_23: "f32[1024][1]cuda:0", primals_24: "f32[1024][1]cuda:0", primals_25: "f32[1024, 1024][1024, 1]cuda:0", primals_26: "f32[1024][1]cuda:0", primals_27: "f32[1024, 1024][1024, 1]cuda:0", primals_28: "f32[1024][1]cuda:0", primals_29: "f32[1024, 1024][1024, 1]cuda:0", primals_30: "f32[1024][1]cuda:0", primals_31: "f32[1024, 1024][1024, 1]cuda:0", primals_32: "f32[1024][1]cuda:0", primals_33: "f32[1024][1]cuda:0", primals_34: "f32[1024][1]cuda:0", primals_35: "f32[4096, 1024][1024, 1]cuda:0", primals_36: "f32[4096][1]cuda:0", primals_37: "f32[1024, 4096][4096, 1]cuda:0", primals_38: "f32[1024][1]cuda:0", primals_39: "f32[1024][1]cuda:0", primals_40: "f32[1024][1]cuda:0", primals_41: "f32[1024, 1024][1024, 1]cuda:0", primals_42: "f32[1024][1]cuda:0", primals_43: "f32[1024, 1024][1024, 1]cuda:0", primals_44: "f32[1024][1]cuda:0", primals_45: "f32[1024, 1024][1024, 1]cuda:0", primals_46: "f32[1024][1]cuda:0", primals_47: "f32[1024, 1024][1024, 1]cuda:0", primals_48: "f32[1024][1]cuda:0", primals_49: "f32[1024][1]cuda:0", primals_50: "f32[1024][1]cuda:0", primals_51: "f32[4096, 1024][1024, 1]cuda:0", primals_52: "f32[4096][1]cuda:0", primals_53: "f32[1024, 4096][4096, 1]cuda:0", primals_54: "f32[1024][1]cuda:0", primals_55: "f32[1024][1]cuda:0", primals_56: "f32[1024][1]cuda:0", primals_57: "f32[1024, 1024][1024, 1]cuda:0", primals_58: "f32[1024][1]cuda:0", primals_59: "f32[1024, 1024][1024, 1]cuda:0", primals_60: "f32[1024][1]cuda:0", primals_61: "f32[1024, 1024][1024, 1]cuda:0", primals_62: "f32[1024][1]cuda:0", primals_63: "f32[1024, 1024][1024, 1]cuda:0", primals_64: "f32[1024][1]cuda:0", primals_65: "f32[1024][1]cuda:0", primals_66: "f32[1024][1]cuda:0", primals_67: "f32[4096, 1024][1024, 1]cuda:0", primals_68: "f32[4096][1]cuda:0", primals_69: "f32[1024, 4096][4096, 1]cuda:0", primals_70: "f32[1024][1]cuda:0", primals_71: "f32[1024][1]cuda:0", primals_72: "f32[1024][1]cuda:0", primals_73: "f32[1024, 1024][1024, 1]cuda:0", primals_74: "f32[1024][1]cuda:0", primals_75: "f32[1024, 1024][1024, 1]cuda:0", primals_76: "f32[1024][1]cuda:0", primals_77: "f32[1024, 1024][1024, 1]cuda:0", primals_78: "f32[1024][1]cuda:0", primals_79: "f32[1024, 1024][1024, 1]cuda:0", primals_80: "f32[1024][1]cuda:0", primals_81: "f32[1024][1]cuda:0", primals_82: "f32[1024][1]cuda:0", primals_83: "f32[4096, 1024][1024, 1]cuda:0", primals_84: "f32[4096][1]cuda:0", primals_85: "f32[1024, 4096][4096, 1]cuda:0", primals_86: "f32[1024][1]cuda:0", primals_87: "f32[1024][1]cuda:0", primals_88: "f32[1024][1]cuda:0", primals_89: "f32[1024, 1024][1024, 1]cuda:0", primals_90: "f32[1024][1]cuda:0", primals_91: "f32[1024, 1024][1024, 1]cuda:0", primals_92: "f32[1024][1]cuda:0", primals_93: "f32[1024, 1024][1024, 1]cuda:0", primals_94: "f32[1024][1]cuda:0", primals_95: "f32[1024, 1024][1024, 1]cuda:0", primals_96: "f32[1024][1]cuda:0", primals_97: "f32[1024][1]cuda:0", primals_98: "f32[1024][1]cuda:0", primals_99: "f32[4096, 1024][1024, 1]cuda:0", primals_100: "f32[4096][1]cuda:0", primals_101: "f32[1024, 4096][4096, 1]cuda:0", primals_102: "f32[1024][1]cuda:0", primals_103: "f32[1024][1]cuda:0", primals_104: "f32[1024][1]cuda:0", primals_105: "f32[1024, 1024][1024, 1]cuda:0", primals_106: "f32[1024][1]cuda:0", primals_107: "f32[1024, 1024][1024, 1]cuda:0", primals_108: "f32[1024][1]cuda:0", primals_109: "f32[1024, 1024][1024, 1]cuda:0", primals_110: "f32[1024][1]cuda:0", primals_111: "f32[1024, 1024][1024, 1]cuda:0", primals_112: "f32[1024][1]cuda:0", primals_113: "f32[1024][1]cuda:0", primals_114: "f32[1024][1]cuda:0", primals_115: "f32[4096, 1024][1024, 1]cuda:0", primals_116: "f32[4096][1]cuda:0", primals_117: "f32[1024, 4096][4096, 1]cuda:0", primals_118: "f32[1024][1]cuda:0", primals_119: "f32[1024][1]cuda:0", primals_120: "f32[1024][1]cuda:0", primals_121: "f32[1024, 1024][1024, 1]cuda:0", primals_122: "f32[1024][1]cuda:0", primals_123: "f32[1024, 1024][1024, 1]cuda:0", primals_124: "f32[1024][1]cuda:0", primals_125: "f32[1024, 1024][1024, 1]cuda:0", primals_126: "f32[1024][1]cuda:0", primals_127: "f32[1024, 1024][1024, 1]cuda:0", primals_128: "f32[1024][1]cuda:0", primals_129: "f32[1024][1]cuda:0", primals_130: "f32[1024][1]cuda:0", primals_131: "f32[4096, 1024][1024, 1]cuda:0", primals_132: "f32[4096][1]cuda:0", primals_133: "f32[1024, 4096][4096, 1]cuda:0", primals_134: "f32[1024][1]cuda:0", primals_135: "f32[1024][1]cuda:0", primals_136: "f32[1024][1]cuda:0", primals_137: "f32[1024, 1024][1024, 1]cuda:0", primals_138: "f32[1024][1]cuda:0", primals_139: "f32[1024, 1024][1024, 1]cuda:0", primals_140: "f32[1024][1]cuda:0", primals_141: "f32[1024, 1024][1024, 1]cuda:0", primals_142: "f32[1024][1]cuda:0", primals_143: "f32[1024, 1024][1024, 1]cuda:0", primals_144: "f32[1024][1]cuda:0", primals_145: "f32[1024][1]cuda:0", primals_146: "f32[1024][1]cuda:0", primals_147: "f32[4096, 1024][1024, 1]cuda:0", primals_148: "f32[4096][1]cuda:0", primals_149: "f32[1024, 4096][4096, 1]cuda:0", primals_150: "f32[1024][1]cuda:0", primals_151: "f32[1024][1]cuda:0", primals_152: "f32[1024][1]cuda:0", primals_153: "f32[1024, 1024][1024, 1]cuda:0", primals_154: "f32[1024][1]cuda:0", primals_155: "f32[1024, 1024][1024, 1]cuda:0", primals_156: "f32[1024][1]cuda:0", primals_157: "f32[1024, 1024][1024, 1]cuda:0", primals_158: "f32[1024][1]cuda:0", primals_159: "f32[1024, 1024][1024, 1]cuda:0", primals_160: "f32[1024][1]cuda:0", primals_161: "f32[1024][1]cuda:0", primals_162: "f32[1024][1]cuda:0", primals_163: "f32[4096, 1024][1024, 1]cuda:0", primals_164: "f32[4096][1]cuda:0", primals_165: "f32[1024, 4096][4096, 1]cuda:0", primals_166: "f32[1024][1]cuda:0", primals_167: "f32[1024][1]cuda:0", primals_168: "f32[1024][1]cuda:0", primals_169: "f32[1024, 1024][1024, 1]cuda:0", primals_170: "f32[1024][1]cuda:0", primals_171: "f32[1024, 1024][1024, 1]cuda:0", primals_172: "f32[1024][1]cuda:0", primals_173: "f32[1024, 1024][1024, 1]cuda:0", primals_174: "f32[1024][1]cuda:0", primals_175: "f32[1024, 1024][1024, 1]cuda:0", primals_176: "f32[1024][1]cuda:0", primals_177: "f32[1024][1]cuda:0", primals_178: "f32[1024][1]cuda:0", primals_179: "f32[4096, 1024][1024, 1]cuda:0", primals_180: "f32[4096][1]cuda:0", primals_181: "f32[1024, 4096][4096, 1]cuda:0", primals_182: "f32[1024][1]cuda:0", primals_183: "f32[1024][1]cuda:0", primals_184: "f32[1024][1]cuda:0", primals_185: "f32[1024, 1024][1024, 1]cuda:0", primals_186: "f32[1024][1]cuda:0", primals_187: "f32[1024, 1024][1024, 1]cuda:0", primals_188: "f32[1024][1]cuda:0", primals_189: "f32[1024, 1024][1024, 1]cuda:0", primals_190: "f32[1024][1]cuda:0", primals_191: "f32[1024, 1024][1024, 1]cuda:0", primals_192: "f32[1024][1]cuda:0", primals_193: "f32[1024][1]cuda:0", primals_194: "f32[1024][1]cuda:0", primals_195: "f32[4096, 1024][1024, 1]cuda:0", primals_196: "f32[4096][1]cuda:0", primals_197: "f32[1024, 4096][4096, 1]cuda:0", primals_198: "f32[1024][1]cuda:0", primals_199: "f32[1024][1]cuda:0", primals_200: "f32[1024][1]cuda:0", primals_201: "f32[1024, 1024][1024, 1]cuda:0", primals_202: "f32[1024][1]cuda:0", primals_203: "f32[1024, 1024][1024, 1]cuda:0", primals_204: "f32[1024][1]cuda:0", primals_205: "f32[1024, 1024][1024, 1]cuda:0", primals_206: "f32[1024][1]cuda:0", primals_207: "f32[1024, 1024][1024, 1]cuda:0", primals_208: "f32[1024][1]cuda:0", primals_209: "f32[1024][1]cuda:0", primals_210: "f32[1024][1]cuda:0", primals_211: "f32[4096, 1024][1024, 1]cuda:0", primals_212: "f32[4096][1]cuda:0", primals_213: "f32[1024, 4096][4096, 1]cuda:0", primals_214: "f32[1024][1]cuda:0", primals_215: "f32[1024][1]cuda:0", primals_216: "f32[1024][1]cuda:0", primals_217: "f32[1024, 1024][1024, 1]cuda:0", primals_218: "f32[1024][1]cuda:0", primals_219: "f32[1024, 1024][1024, 1]cuda:0", primals_220: "f32[1024][1]cuda:0", primals_221: "f32[1024, 1024][1024, 1]cuda:0", primals_222: "f32[1024][1]cuda:0", primals_223: "f32[1024, 1024][1024, 1]cuda:0", primals_224: "f32[1024][1]cuda:0", primals_225: "f32[1024][1]cuda:0", primals_226: "f32[1024][1]cuda:0", primals_227: "f32[4096, 1024][1024, 1]cuda:0", primals_228: "f32[4096][1]cuda:0", primals_229: "f32[1024, 4096][4096, 1]cuda:0", primals_230: "f32[1024][1]cuda:0", primals_231: "f32[1024][1]cuda:0", primals_232: "f32[1024][1]cuda:0", primals_233: "f32[1024, 1024][1024, 1]cuda:0", primals_234: "f32[1024][1]cuda:0", primals_235: "f32[1024, 1024][1024, 1]cuda:0", primals_236: "f32[1024][1]cuda:0", primals_237: "f32[1024, 1024][1024, 1]cuda:0", primals_238: "f32[1024][1]cuda:0", primals_239: "f32[1024, 1024][1024, 1]cuda:0", primals_240: "f32[1024][1]cuda:0", primals_241: "f32[1024][1]cuda:0", primals_242: "f32[1024][1]cuda:0", primals_243: "f32[4096, 1024][1024, 1]cuda:0", primals_244: "f32[4096][1]cuda:0", primals_245: "f32[1024, 4096][4096, 1]cuda:0", primals_246: "f32[1024][1]cuda:0", primals_247: "f32[1024][1]cuda:0", primals_248: "f32[1024][1]cuda:0", primals_249: "f32[1024, 1024][1024, 1]cuda:0", primals_250: "f32[1024][1]cuda:0", primals_251: "f32[1024, 1024][1024, 1]cuda:0", primals_252: "f32[1024][1]cuda:0", primals_253: "f32[1024, 1024][1024, 1]cuda:0", primals_254: "f32[1024][1]cuda:0", primals_255: "f32[1024, 1024][1024, 1]cuda:0", primals_256: "f32[1024][1]cuda:0", primals_257: "f32[1024][1]cuda:0", primals_258: "f32[1024][1]cuda:0", primals_259: "f32[4096, 1024][1024, 1]cuda:0", primals_260: "f32[4096][1]cuda:0", primals_261: "f32[1024, 4096][4096, 1]cuda:0", primals_262: "f32[1024][1]cuda:0", primals_263: "f32[1024][1]cuda:0", primals_264: "f32[1024][1]cuda:0", primals_265: "f32[1024, 1024][1024, 1]cuda:0", primals_266: "f32[1024][1]cuda:0", primals_267: "f32[1024, 1024][1024, 1]cuda:0", primals_268: "f32[1024][1]cuda:0", primals_269: "f32[1024, 1024][1024, 1]cuda:0", primals_270: "f32[1024][1]cuda:0", primals_271: "f32[1024, 1024][1024, 1]cuda:0", primals_272: "f32[1024][1]cuda:0", primals_273: "f32[1024][1]cuda:0", primals_274: "f32[1024][1]cuda:0", primals_275: "f32[4096, 1024][1024, 1]cuda:0", primals_276: "f32[4096][1]cuda:0", primals_277: "f32[1024, 4096][4096, 1]cuda:0", primals_278: "f32[1024][1]cuda:0", primals_279: "f32[1024][1]cuda:0", primals_280: "f32[1024][1]cuda:0", primals_281: "f32[1024, 1024][1024, 1]cuda:0", primals_282: "f32[1024][1]cuda:0", primals_283: "f32[1024, 1024][1024, 1]cuda:0", primals_284: "f32[1024][1]cuda:0", primals_285: "f32[1024, 1024][1024, 1]cuda:0", primals_286: "f32[1024][1]cuda:0", primals_287: "f32[1024, 1024][1024, 1]cuda:0", primals_288: "f32[1024][1]cuda:0", primals_289: "f32[1024][1]cuda:0", primals_290: "f32[1024][1]cuda:0", primals_291: "f32[4096, 1024][1024, 1]cuda:0", primals_292: "f32[4096][1]cuda:0", primals_293: "f32[1024, 4096][4096, 1]cuda:0", primals_294: "f32[1024][1]cuda:0", primals_295: "f32[1024][1]cuda:0", primals_296: "f32[1024][1]cuda:0", primals_297: "f32[1024, 1024][1024, 1]cuda:0", primals_298: "f32[1024][1]cuda:0", primals_299: "f32[1024, 1024][1024, 1]cuda:0", primals_300: "f32[1024][1]cuda:0", primals_301: "f32[1024, 1024][1024, 1]cuda:0", primals_302: "f32[1024][1]cuda:0", primals_303: "f32[1024, 1024][1024, 1]cuda:0", primals_304: "f32[1024][1]cuda:0", primals_305: "f32[1024][1]cuda:0", primals_306: "f32[1024][1]cuda:0", primals_307: "f32[4096, 1024][1024, 1]cuda:0", primals_308: "f32[4096][1]cuda:0", primals_309: "f32[1024, 4096][4096, 1]cuda:0", primals_310: "f32[1024][1]cuda:0", primals_311: "f32[1024][1]cuda:0", primals_312: "f32[1024][1]cuda:0", primals_313: "f32[1024, 1024][1024, 1]cuda:0", primals_314: "f32[1024][1]cuda:0", primals_315: "f32[1024, 1024][1024, 1]cuda:0", primals_316: "f32[1024][1]cuda:0", primals_317: "f32[1024, 1024][1024, 1]cuda:0", primals_318: "f32[1024][1]cuda:0", primals_319: "f32[1024, 1024][1024, 1]cuda:0", primals_320: "f32[1024][1]cuda:0", primals_321: "f32[1024][1]cuda:0", primals_322: "f32[1024][1]cuda:0", primals_323: "f32[4096, 1024][1024, 1]cuda:0", primals_324: "f32[4096][1]cuda:0", primals_325: "f32[1024, 4096][4096, 1]cuda:0", primals_326: "f32[1024][1]cuda:0", primals_327: "f32[1024][1]cuda:0", primals_328: "f32[1024][1]cuda:0", primals_329: "f32[1024, 1024][1024, 1]cuda:0", primals_330: "f32[1024][1]cuda:0", primals_331: "f32[1024, 1024][1024, 1]cuda:0", primals_332: "f32[1024][1]cuda:0", primals_333: "f32[1024, 1024][1024, 1]cuda:0", primals_334: "f32[1024][1]cuda:0", primals_335: "f32[1024, 1024][1024, 1]cuda:0", primals_336: "f32[1024][1]cuda:0", primals_337: "f32[1024][1]cuda:0", primals_338: "f32[1024][1]cuda:0", primals_339: "f32[4096, 1024][1024, 1]cuda:0", primals_340: "f32[4096][1]cuda:0", primals_341: "f32[1024, 4096][4096, 1]cuda:0", primals_342: "f32[1024][1]cuda:0", primals_343: "f32[1024][1]cuda:0", primals_344: "f32[1024][1]cuda:0", primals_345: "f32[1024, 1024][1024, 1]cuda:0", primals_346: "f32[1024][1]cuda:0", primals_347: "f32[1024, 1024][1024, 1]cuda:0", primals_348: "f32[1024][1]cuda:0", primals_349: "f32[1024, 1024][1024, 1]cuda:0", primals_350: "f32[1024][1]cuda:0", primals_351: "f32[1024, 1024][1024, 1]cuda:0", primals_352: "f32[1024][1]cuda:0", primals_353: "f32[1024][1]cuda:0", primals_354: "f32[1024][1]cuda:0", primals_355: "f32[4096, 1024][1024, 1]cuda:0", primals_356: "f32[4096][1]cuda:0", primals_357: "f32[1024, 4096][4096, 1]cuda:0", primals_358: "f32[1024][1]cuda:0", primals_359: "f32[1024][1]cuda:0", primals_360: "f32[1024][1]cuda:0", primals_361: "f32[1024, 1024][1024, 1]cuda:0", primals_362: "f32[1024][1]cuda:0", primals_363: "f32[1024, 1024][1024, 1]cuda:0", primals_364: "f32[1024][1]cuda:0", primals_365: "f32[1024, 1024][1024, 1]cuda:0", primals_366: "f32[1024][1]cuda:0", primals_367: "f32[1024, 1024][1024, 1]cuda:0", primals_368: "f32[1024][1]cuda:0", primals_369: "f32[1024][1]cuda:0", primals_370: "f32[1024][1]cuda:0", primals_371: "f32[4096, 1024][1024, 1]cuda:0", primals_372: "f32[4096][1]cuda:0", primals_373: "f32[1024, 4096][4096, 1]cuda:0", primals_374: "f32[1024][1]cuda:0", primals_375: "f32[1024][1]cuda:0", primals_376: "f32[1024][1]cuda:0", primals_377: "f32[1024, 1024][1024, 1]cuda:0", primals_378: "f32[1024][1]cuda:0", primals_379: "f32[1024, 1024][1024, 1]cuda:0", primals_380: "f32[1024][1]cuda:0", primals_381: "f32[1024, 1024][1024, 1]cuda:0", primals_382: "f32[1024][1]cuda:0", primals_383: "f32[1024, 1024][1024, 1]cuda:0", primals_384: "f32[1024][1]cuda:0", primals_385: "f32[1024][1]cuda:0", primals_386: "f32[1024][1]cuda:0", primals_387: "f32[4096, 1024][1024, 1]cuda:0", primals_388: "f32[4096][1]cuda:0", primals_389: "f32[1024, 4096][4096, 1]cuda:0", primals_390: "f32[1024][1]cuda:0", primals_391: "f32[1024][1]cuda:0", primals_392: "f32[1024][1]cuda:0", primals_393: "f32[1024, 1024][1024, 1]cuda:0", primals_394: "f32[1024][1]cuda:0", primals_395: "f32[1024][1]cuda:0", primals_396: "f32[1024][1]cuda:0", primals_397: "f32[29056][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:628 in forward, code: token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        full_default: "i64[16, 512][512, 1]cuda:0" = torch.ops.aten.full.default([16, 512], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:89 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        embedding: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(primals_3, primals_2, 0)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:90 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embedding_1: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(primals_5, full_default);  primals_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:91 in forward, code: embeddings = inputs_embeds + token_type_embeddings
        add: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:93 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_2: "f32[1, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(primals_6, primals_4);  primals_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:94 in forward, code: embeddings += position_embeddings
        add_1: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[49][1]cuda:0" = torch.ops.prims.inductor_seeds.default(49, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:98 in forward, code: embeddings = self.dropout(embeddings)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_48: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_48, 0.1);  inductor_random_default_48 = None
        mul_1: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, add_1);  add_1 = None
        mul_2: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, 1.1111111111111112);  mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(mul_2, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_2: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        sub_1: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_2, getitem_1);  getitem_1 = None
        mul_3: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_4: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, primals_7)
        add_3: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, primals_8);  mul_4 = primals_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_10, torch.bfloat16);  primals_10 = None
        convert_element_type_1: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        convert_element_type_2: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16);  add_3 = None
        view: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2, [8192, 1024]);  convert_element_type_2 = None
        permute: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [16, 512, 1024]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_2: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1, [16, 512, -1, 64]);  view_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_6: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_7: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_11, torch.bfloat16);  primals_11 = None
        permute_2: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_1: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_6, view, permute_2);  convert_element_type_6 = None
        view_4: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [16, 512, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_5: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_4, [16, 512, -1, 64]);  view_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_12: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convert_element_type_13: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        permute_4: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_13, [1, 0]);  convert_element_type_13 = None
        addmm_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_12, view, permute_4);  convert_element_type_12 = None
        view_7: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [16, 512, 1024]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_8: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_7, [16, 512, -1, 64]);  view_7 = None

        # No stacktrace found for following nodes
        permute_default_138: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
        permute_default_139: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
        permute_default_140: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
        _scaled_dot_product_flash_attention_default_23 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_138, permute_default_139, permute_default_140, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_261: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_23[0]

        # No stacktrace found for following nodes
        getitem_262: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_23[1]
        getitem_263: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_23[6]
        getitem_264: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_23[7];  _scaled_dot_product_flash_attention_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_7: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_261, [0, 2, 1, 3])
        clone_3: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_15: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [16, 512, 1024]);  clone_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_23: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_16, torch.bfloat16);  primals_16 = None
        convert_element_type_24: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        view_16: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_15, [8192, 1024]);  view_15 = None
        permute_8: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_3: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_23, view_16, permute_8);  convert_element_type_23 = None
        view_17: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [16, 512, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_47: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_71: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_47, torch.bfloat16);  inductor_random_default_47 = None
        gt_2: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_71, 0.1);  convert_element_type_default_71 = None
        mul_7: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, view_17);  view_17 = None
        mul_8: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_5: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2, mul_8);  mul_2 = mul_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_5, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_3: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_5, getitem_3);  getitem_3 = None
        mul_9: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_1);  sub_3 = None
        mul_10: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, primals_17)
        add_7: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, primals_18);  mul_10 = primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_28: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convert_element_type_29: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convert_element_type_30: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16);  add_7 = None
        view_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_30, [8192, 1024]);  convert_element_type_30 = None
        permute_9: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        addmm_4: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_28, view_18, permute_9);  convert_element_type_28 = None
        view_19: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_34: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_11: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.5)
        mul_12: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476);  convert_element_type_34 = None
        erf: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_13: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, add_8);  mul_11 = add_8 = None
        convert_element_type_35: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_13, torch.bfloat16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_36: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_22, torch.bfloat16);  primals_22 = None
        convert_element_type_37: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        view_20: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_35, [8192, 4096]);  convert_element_type_35 = None
        permute_10: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_37, [1, 0]);  convert_element_type_37 = None
        addmm_5: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_36, view_20, permute_10);  convert_element_type_36 = None
        view_21: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [16, 512, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_46: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        convert_element_type_default_70: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_46, torch.bfloat16);  inductor_random_default_46 = None
        gt_3: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_70, 0.1);  convert_element_type_default_70 = None
        mul_14: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, view_21);  view_21 = None
        mul_15: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, 1.1111111111111112);  mul_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_9: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_5, mul_15);  add_5 = mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
        getitem_4: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_10: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_4: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_9, getitem_5);  getitem_5 = None
        mul_16: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_17: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, primals_23)
        add_11: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, primals_24);  mul_17 = primals_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_41: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convert_element_type_42: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convert_element_type_43: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        view_22: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_43, [8192, 1024]);  convert_element_type_43 = None
        permute_11: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_42, [1, 0]);  convert_element_type_42 = None
        addmm_6: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_41, view_22, permute_11);  convert_element_type_41 = None
        view_23: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [16, 512, 1024]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_24: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [16, 512, -1, 64]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_47: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_48: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        permute_13: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_48, [1, 0]);  convert_element_type_48 = None
        addmm_7: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_47, view_22, permute_13);  convert_element_type_47 = None
        view_26: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [16, 512, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_27: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_26, [16, 512, -1, 64]);  view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_53: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convert_element_type_54: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        permute_15: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0]);  convert_element_type_54 = None
        addmm_8: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_53, view_22, permute_15);  convert_element_type_53 = None
        view_29: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [16, 512, 1024]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_30: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_29, [16, 512, -1, 64]);  view_29 = None

        # No stacktrace found for following nodes
        permute_default_132: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
        permute_default_133: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
        permute_default_134: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        _scaled_dot_product_flash_attention_default_22 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_132, permute_default_133, permute_default_134, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_254: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_22[0]

        # No stacktrace found for following nodes
        getitem_255: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_22[1]
        getitem_256: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_22[6]
        getitem_257: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_22[7];  _scaled_dot_product_flash_attention_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_18: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_254, [0, 2, 1, 3])
        clone_7: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_37: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [16, 512, 1024]);  clone_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_64: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convert_element_type_65: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        view_38: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_37, [8192, 1024]);  view_37 = None
        permute_19: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_65, [1, 0]);  convert_element_type_65 = None
        addmm_9: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_64, view_38, permute_19);  convert_element_type_64 = None
        view_39: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [16, 512, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_45: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_69: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_45, torch.bfloat16);  inductor_random_default_45 = None
        gt_5: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_69, 0.1);  convert_element_type_default_69 = None
        mul_20: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_5, view_39);  view_39 = None
        mul_21: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, 1.1111111111111112);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_13: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_9, mul_21);  add_9 = mul_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
        getitem_6: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_14: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_6: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_13, getitem_7);  getitem_7 = None
        mul_22: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = None
        mul_23: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, primals_33)
        add_15: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_23, primals_34);  mul_23 = primals_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_69: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        convert_element_type_70: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        convert_element_type_71: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        view_40: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_71, [8192, 1024]);  convert_element_type_71 = None
        permute_20: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_70, [1, 0]);  convert_element_type_70 = None
        addmm_10: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_69, view_40, permute_20);  convert_element_type_69 = None
        view_41: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_75: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_24: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.5)
        mul_25: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.7071067811865476);  convert_element_type_75 = None
        erf_1: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_26: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_16);  mul_24 = add_16 = None
        convert_element_type_76: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_77: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convert_element_type_78: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        view_42: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_76, [8192, 4096]);  convert_element_type_76 = None
        permute_21: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_78, [1, 0]);  convert_element_type_78 = None
        addmm_11: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_77, view_42, permute_21);  convert_element_type_77 = None
        view_43: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [16, 512, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_44: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_68: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_44, torch.bfloat16);  inductor_random_default_44 = None
        gt_6: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_68, 0.1);  convert_element_type_default_68 = None
        mul_27: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_6, view_43);  view_43 = None
        mul_28: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, 1.1111111111111112);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_17: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, mul_28);  add_13 = mul_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
        getitem_8: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_18: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        sub_7: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_17, getitem_9);  getitem_9 = None
        mul_29: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
        mul_30: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, primals_39)
        add_19: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_30, primals_40);  mul_30 = primals_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_82: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        convert_element_type_83: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_41, torch.bfloat16);  primals_41 = None
        convert_element_type_84: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        view_44: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_84, [8192, 1024]);  convert_element_type_84 = None
        permute_22: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_83, [1, 0]);  convert_element_type_83 = None
        addmm_12: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_82, view_44, permute_22);  convert_element_type_82 = None
        view_45: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [16, 512, 1024]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_46: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_45, [16, 512, -1, 64]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_88: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convert_element_type_89: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        permute_24: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_89, [1, 0]);  convert_element_type_89 = None
        addmm_13: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_88, view_44, permute_24);  convert_element_type_88 = None
        view_48: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [16, 512, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_49: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_48, [16, 512, -1, 64]);  view_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_94: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_46, torch.bfloat16);  primals_46 = None
        convert_element_type_95: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        permute_26: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_95, [1, 0]);  convert_element_type_95 = None
        addmm_14: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_94, view_44, permute_26);  convert_element_type_94 = None
        view_51: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [16, 512, 1024]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_52: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_51, [16, 512, -1, 64]);  view_51 = None

        # No stacktrace found for following nodes
        permute_default_126: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        permute_default_127: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        permute_default_128: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
        _scaled_dot_product_flash_attention_default_21 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_126, permute_default_127, permute_default_128, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_247: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_21[0]

        # No stacktrace found for following nodes
        getitem_248: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_21[1]
        getitem_249: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_21[6]
        getitem_250: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_21[7];  _scaled_dot_product_flash_attention_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_29: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_247, [0, 2, 1, 3])
        clone_11: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_59: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [16, 512, 1024]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_105: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        convert_element_type_106: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_47, torch.bfloat16);  primals_47 = None
        view_60: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_59, [8192, 1024]);  view_59 = None
        permute_30: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_106, [1, 0]);  convert_element_type_106 = None
        addmm_15: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_105, view_60, permute_30);  convert_element_type_105 = None
        view_61: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [16, 512, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_43: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_67: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_43, torch.bfloat16);  inductor_random_default_43 = None
        gt_8: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_67, 0.1);  convert_element_type_default_67 = None
        mul_33: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_8, view_61);  view_61 = None
        mul_34: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, 1.1111111111111112);  mul_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_21: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_17, mul_34);  add_17 = mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
        getitem_10: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_22: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        sub_9: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_21, getitem_11);  getitem_11 = None
        mul_35: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_5);  sub_9 = None
        mul_36: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, primals_49)
        add_23: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, primals_50);  mul_36 = primals_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_110: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_52, torch.bfloat16);  primals_52 = None
        convert_element_type_111: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        convert_element_type_112: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        view_62: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_112, [8192, 1024]);  convert_element_type_112 = None
        permute_31: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_111, [1, 0]);  convert_element_type_111 = None
        addmm_16: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_110, view_62, permute_31);  convert_element_type_110 = None
        view_63: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_116: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_37: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.5)
        mul_38: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.7071067811865476);  convert_element_type_116 = None
        erf_2: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_24: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_39: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, add_24);  mul_37 = add_24 = None
        convert_element_type_117: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_39, torch.bfloat16);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_118: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        convert_element_type_119: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_53, torch.bfloat16);  primals_53 = None
        view_64: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_117, [8192, 4096]);  convert_element_type_117 = None
        permute_32: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        addmm_17: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_118, view_64, permute_32);  convert_element_type_118 = None
        view_65: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [16, 512, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_42: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        convert_element_type_default_66: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_42, torch.bfloat16);  inductor_random_default_42 = None
        gt_9: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_66, 0.1);  convert_element_type_default_66 = None
        mul_40: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_9, view_65);  view_65 = None
        mul_41: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, 1.1111111111111112);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_25: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_21, mul_41);  add_21 = mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
        getitem_12: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_26: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_10: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_25, getitem_13);  getitem_13 = None
        mul_42: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
        mul_43: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, primals_55)
        add_27: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, primals_56);  mul_43 = primals_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_123: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        convert_element_type_124: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_125: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16);  add_27 = None
        view_66: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_125, [8192, 1024]);  convert_element_type_125 = None
        permute_33: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_124, [1, 0]);  convert_element_type_124 = None
        addmm_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_123, view_66, permute_33);  convert_element_type_123 = None
        view_67: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [16, 512, 1024]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_68: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_67, [16, 512, -1, 64]);  view_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_129: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_130: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        permute_35: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_130, [1, 0]);  convert_element_type_130 = None
        addmm_19: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_129, view_66, permute_35);  convert_element_type_129 = None
        view_70: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [16, 512, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_71: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_70, [16, 512, -1, 64]);  view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_135: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convert_element_type_136: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        permute_37: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_136, [1, 0]);  convert_element_type_136 = None
        addmm_20: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_135, view_66, permute_37);  convert_element_type_135 = None
        view_73: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [16, 512, 1024]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_74: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_73, [16, 512, -1, 64]);  view_73 = None

        # No stacktrace found for following nodes
        permute_default_120: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        permute_default_121: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        permute_default_122: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
        _scaled_dot_product_flash_attention_default_20 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_120, permute_default_121, permute_default_122, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_240: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_20[0]

        # No stacktrace found for following nodes
        getitem_241: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_20[1]
        getitem_242: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_20[6]
        getitem_243: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_20[7];  _scaled_dot_product_flash_attention_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_40: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_240, [0, 2, 1, 3])
        clone_15: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_81: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [16, 512, 1024]);  clone_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_146: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        convert_element_type_147: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        view_82: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_81, [8192, 1024]);  view_81 = None
        permute_41: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_147, [1, 0]);  convert_element_type_147 = None
        addmm_21: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_146, view_82, permute_41);  convert_element_type_146 = None
        view_83: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [16, 512, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_41: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_65: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_41, torch.bfloat16);  inductor_random_default_41 = None
        gt_11: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_65, 0.1);  convert_element_type_default_65 = None
        mul_46: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_11, view_83);  view_83 = None
        mul_47: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, 1.1111111111111112);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_29: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_25, mul_47);  add_25 = mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_14: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_30: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        sub_12: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_29, getitem_15);  getitem_15 = None
        mul_48: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_7);  sub_12 = None
        mul_49: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, primals_65)
        add_31: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_49, primals_66);  mul_49 = primals_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_151: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convert_element_type_152: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convert_element_type_153: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_31, torch.bfloat16);  add_31 = None
        view_84: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_153, [8192, 1024]);  convert_element_type_153 = None
        permute_42: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_152, [1, 0]);  convert_element_type_152 = None
        addmm_22: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_151, view_84, permute_42);  convert_element_type_151 = None
        view_85: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_157: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_50: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.5)
        mul_51: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.7071067811865476);  convert_element_type_157 = None
        erf_3: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_32: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_52: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_32);  mul_50 = add_32 = None
        convert_element_type_158: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_159: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_70, torch.bfloat16);  primals_70 = None
        convert_element_type_160: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        view_86: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_158, [8192, 4096]);  convert_element_type_158 = None
        permute_43: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_160, [1, 0]);  convert_element_type_160 = None
        addmm_23: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_159, view_86, permute_43);  convert_element_type_159 = None
        view_87: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [16, 512, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_40: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_64: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_40, torch.bfloat16);  inductor_random_default_40 = None
        gt_12: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_64, 0.1);  convert_element_type_default_64 = None
        mul_53: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_12, view_87);  view_87 = None
        mul_54: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_33: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_29, mul_54);  add_29 = mul_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_16: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_34: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        sub_13: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_33, getitem_17);  getitem_17 = None
        mul_55: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
        mul_56: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, primals_71)
        add_35: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_56, primals_72);  mul_56 = primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_164: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convert_element_type_165: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_166: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16);  add_35 = None
        view_88: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_166, [8192, 1024]);  convert_element_type_166 = None
        permute_44: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_165, [1, 0]);  convert_element_type_165 = None
        addmm_24: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_164, view_88, permute_44);  convert_element_type_164 = None
        view_89: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [16, 512, 1024]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_90: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_89, [16, 512, -1, 64]);  view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_170: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_76, torch.bfloat16);  primals_76 = None
        convert_element_type_171: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        permute_46: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_171, [1, 0]);  convert_element_type_171 = None
        addmm_25: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_170, view_88, permute_46);  convert_element_type_170 = None
        view_92: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [16, 512, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_93: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_92, [16, 512, -1, 64]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_176: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        convert_element_type_177: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_77, torch.bfloat16);  primals_77 = None
        permute_48: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_177, [1, 0]);  convert_element_type_177 = None
        addmm_26: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_176, view_88, permute_48);  convert_element_type_176 = None
        view_95: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [16, 512, 1024]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_96: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_95, [16, 512, -1, 64]);  view_95 = None

        # No stacktrace found for following nodes
        permute_default_114: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        permute_default_115: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
        permute_default_116: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        _scaled_dot_product_flash_attention_default_19 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_114, permute_default_115, permute_default_116, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_233: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_19[0]

        # No stacktrace found for following nodes
        getitem_234: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_19[1]
        getitem_235: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_19[6]
        getitem_236: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_19[7];  _scaled_dot_product_flash_attention_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_51: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_233, [0, 2, 1, 3])
        clone_19: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_103: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [16, 512, 1024]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_187: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convert_element_type_188: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        view_104: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_103, [8192, 1024]);  view_103 = None
        permute_52: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_188, [1, 0]);  convert_element_type_188 = None
        addmm_27: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_187, view_104, permute_52);  convert_element_type_187 = None
        view_105: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [16, 512, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_39: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_63: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_39, torch.bfloat16);  inductor_random_default_39 = None
        gt_14: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_63, 0.1);  convert_element_type_default_63 = None
        mul_59: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_14, view_105);  view_105 = None
        mul_60: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, 1.1111111111111112);  mul_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_37: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_33, mul_60);  add_33 = mul_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_18: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_38: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_15: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_37, getitem_19);  getitem_19 = None
        mul_61: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_9);  sub_15 = None
        mul_62: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, primals_81)
        add_39: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, primals_82);  mul_62 = primals_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_192: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        convert_element_type_193: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_83, torch.bfloat16);  primals_83 = None
        convert_element_type_194: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        view_106: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_194, [8192, 1024]);  convert_element_type_194 = None
        permute_53: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_193, [1, 0]);  convert_element_type_193 = None
        addmm_28: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_192, view_106, permute_53);  convert_element_type_192 = None
        view_107: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_198: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_63: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.5)
        mul_64: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.7071067811865476);  convert_element_type_198 = None
        erf_4: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_40: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_65: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, add_40);  mul_63 = add_40 = None
        convert_element_type_199: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16);  mul_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_200: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convert_element_type_201: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        view_108: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_199, [8192, 4096]);  convert_element_type_199 = None
        permute_54: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_201, [1, 0]);  convert_element_type_201 = None
        addmm_29: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_200, view_108, permute_54);  convert_element_type_200 = None
        view_109: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [16, 512, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_38: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        convert_element_type_default_62: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_38, torch.bfloat16);  inductor_random_default_38 = None
        gt_15: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_62, 0.1);  convert_element_type_default_62 = None
        mul_66: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_15, view_109);  view_109 = None
        mul_67: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_41: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_37, mul_67);  add_37 = mul_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_20: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_42: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_16: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_41, getitem_21);  getitem_21 = None
        mul_68: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
        mul_69: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, primals_87)
        add_43: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, primals_88);  mul_69 = primals_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_205: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convert_element_type_206: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        convert_element_type_207: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        view_110: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_207, [8192, 1024]);  convert_element_type_207 = None
        permute_55: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_206, [1, 0]);  convert_element_type_206 = None
        addmm_30: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_205, view_110, permute_55);  convert_element_type_205 = None
        view_111: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [16, 512, 1024]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_112: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_111, [16, 512, -1, 64]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_211: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convert_element_type_212: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        permute_57: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_212, [1, 0]);  convert_element_type_212 = None
        addmm_31: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_211, view_110, permute_57);  convert_element_type_211 = None
        view_114: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [16, 512, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_115: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_114, [16, 512, -1, 64]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_217: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        convert_element_type_218: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        permute_59: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_218, [1, 0]);  convert_element_type_218 = None
        addmm_32: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_217, view_110, permute_59);  convert_element_type_217 = None
        view_117: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [16, 512, 1024]);  addmm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_118: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_117, [16, 512, -1, 64]);  view_117 = None

        # No stacktrace found for following nodes
        permute_default_108: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
        permute_default_109: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
        permute_default_110: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        _scaled_dot_product_flash_attention_default_18 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_108, permute_default_109, permute_default_110, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_226: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_18[0]

        # No stacktrace found for following nodes
        getitem_227: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_18[1]
        getitem_228: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_18[6]
        getitem_229: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_18[7];  _scaled_dot_product_flash_attention_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_62: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_226, [0, 2, 1, 3])
        clone_23: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_125: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [16, 512, 1024]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_228: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        convert_element_type_229: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        view_126: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [8192, 1024]);  view_125 = None
        permute_63: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_229, [1, 0]);  convert_element_type_229 = None
        addmm_33: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_228, view_126, permute_63);  convert_element_type_228 = None
        view_127: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [16, 512, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_37: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_61: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_37, torch.bfloat16);  inductor_random_default_37 = None
        gt_17: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_61, 0.1);  convert_element_type_default_61 = None
        mul_72: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_17, view_127);  view_127 = None
        mul_73: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, 1.1111111111111112);  mul_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_45: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_41, mul_73);  add_41 = mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
        getitem_22: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_46: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_18: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_45, getitem_23);  getitem_23 = None
        mul_74: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_11);  sub_18 = None
        mul_75: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, primals_97)
        add_47: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_75, primals_98);  mul_75 = primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_233: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_100, torch.bfloat16);  primals_100 = None
        convert_element_type_234: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        convert_element_type_235: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None
        view_128: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_235, [8192, 1024]);  convert_element_type_235 = None
        permute_64: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        addmm_34: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_233, view_128, permute_64);  convert_element_type_233 = None
        view_129: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_239: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_76: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, 0.5)
        mul_77: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, 0.7071067811865476);  convert_element_type_239 = None
        erf_5: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_48: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_78: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, add_48);  mul_76 = add_48 = None
        convert_element_type_240: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_78, torch.bfloat16);  mul_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_241: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_242: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_101, torch.bfloat16);  primals_101 = None
        view_130: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_240, [8192, 4096]);  convert_element_type_240 = None
        permute_65: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_242, [1, 0]);  convert_element_type_242 = None
        addmm_35: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_241, view_130, permute_65);  convert_element_type_241 = None
        view_131: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [16, 512, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_36: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        convert_element_type_default_60: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_36, torch.bfloat16);  inductor_random_default_36 = None
        gt_18: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_60, 0.1);  convert_element_type_default_60 = None
        mul_79: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_18, view_131);  view_131 = None
        mul_80: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_79, 1.1111111111111112);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_49: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_45, mul_80);  add_45 = mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
        getitem_24: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_50: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        sub_19: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_49, getitem_25);  getitem_25 = None
        mul_81: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
        mul_82: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, primals_103)
        add_51: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, primals_104);  mul_82 = primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_246: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_106, torch.bfloat16);  primals_106 = None
        convert_element_type_247: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_248: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_51, torch.bfloat16);  add_51 = None
        view_132: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_248, [8192, 1024]);  convert_element_type_248 = None
        permute_66: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_247, [1, 0]);  convert_element_type_247 = None
        addmm_36: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_246, view_132, permute_66);  convert_element_type_246 = None
        view_133: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [16, 512, 1024]);  addmm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_134: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_133, [16, 512, -1, 64]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_252: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        convert_element_type_253: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_107, torch.bfloat16);  primals_107 = None
        permute_68: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_253, [1, 0]);  convert_element_type_253 = None
        addmm_37: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_252, view_132, permute_68);  convert_element_type_252 = None
        view_136: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [16, 512, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_137: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_136, [16, 512, -1, 64]);  view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_258: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convert_element_type_259: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        permute_70: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        addmm_38: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_258, view_132, permute_70);  convert_element_type_258 = None
        view_139: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [16, 512, 1024]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_140: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_139, [16, 512, -1, 64]);  view_139 = None

        # No stacktrace found for following nodes
        permute_default_102: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        permute_default_103: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
        permute_default_104: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        _scaled_dot_product_flash_attention_default_17 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_102, permute_default_103, permute_default_104, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_219: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_17[0]

        # No stacktrace found for following nodes
        getitem_220: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_17[1]
        getitem_221: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_17[6]
        getitem_222: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_17[7];  _scaled_dot_product_flash_attention_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_73: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_219, [0, 2, 1, 3])
        clone_27: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_147: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [16, 512, 1024]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_269: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_112, torch.bfloat16);  primals_112 = None
        convert_element_type_270: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        view_148: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_147, [8192, 1024]);  view_147 = None
        permute_74: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_270, [1, 0]);  convert_element_type_270 = None
        addmm_39: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_269, view_148, permute_74);  convert_element_type_269 = None
        view_149: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [16, 512, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_35: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        convert_element_type_default_59: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_35, torch.bfloat16);  inductor_random_default_35 = None
        gt_20: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_59, 0.1);  convert_element_type_default_59 = None
        mul_85: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_20, view_149);  view_149 = None
        mul_86: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, 1.1111111111111112);  mul_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_53: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_49, mul_86);  add_49 = mul_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
        getitem_26: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_54: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
        sub_21: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_53, getitem_27);  getitem_27 = None
        mul_87: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_13);  sub_21 = None
        mul_88: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, primals_113)
        add_55: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, primals_114);  mul_88 = primals_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_274: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convert_element_type_275: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convert_element_type_276: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        view_150: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_276, [8192, 1024]);  convert_element_type_276 = None
        permute_75: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_275, [1, 0]);  convert_element_type_275 = None
        addmm_40: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_274, view_150, permute_75);  convert_element_type_274 = None
        view_151: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_280: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_89: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, 0.5)
        mul_90: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, 0.7071067811865476);  convert_element_type_280 = None
        erf_6: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_56: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_91: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, add_56);  mul_89 = add_56 = None
        convert_element_type_281: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_91, torch.bfloat16);  mul_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_282: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_283: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        view_152: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_281, [8192, 4096]);  convert_element_type_281 = None
        permute_76: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_283, [1, 0]);  convert_element_type_283 = None
        addmm_41: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_282, view_152, permute_76);  convert_element_type_282 = None
        view_153: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [16, 512, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_34: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        convert_element_type_default_58: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_34, torch.bfloat16);  inductor_random_default_34 = None
        gt_21: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_58, 0.1);  convert_element_type_default_58 = None
        mul_92: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_21, view_153);  view_153 = None
        mul_93: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, 1.1111111111111112);  mul_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_57: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_53, mul_93);  add_53 = mul_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
        getitem_28: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_58: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_22: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_57, getitem_29);  getitem_29 = None
        mul_94: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
        mul_95: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, primals_119)
        add_59: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, primals_120);  mul_95 = primals_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_287: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convert_element_type_288: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convert_element_type_289: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        view_154: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_289, [8192, 1024]);  convert_element_type_289 = None
        permute_77: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_288, [1, 0]);  convert_element_type_288 = None
        addmm_42: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_287, view_154, permute_77);  convert_element_type_287 = None
        view_155: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [16, 512, 1024]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_156: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [16, 512, -1, 64]);  view_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_293: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        convert_element_type_294: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        permute_79: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_294, [1, 0]);  convert_element_type_294 = None
        addmm_43: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_293, view_154, permute_79);  convert_element_type_293 = None
        view_158: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [16, 512, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_159: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_158, [16, 512, -1, 64]);  view_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_299: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        convert_element_type_300: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        permute_81: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_300, [1, 0]);  convert_element_type_300 = None
        addmm_44: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_299, view_154, permute_81);  convert_element_type_299 = None
        view_161: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [16, 512, 1024]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_162: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_161, [16, 512, -1, 64]);  view_161 = None

        # No stacktrace found for following nodes
        permute_default_96: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        permute_default_97: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        permute_default_98: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
        _scaled_dot_product_flash_attention_default_16 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_96, permute_default_97, permute_default_98, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_212: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_16[0]

        # No stacktrace found for following nodes
        getitem_213: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_16[1]
        getitem_214: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_16[6]
        getitem_215: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_16[7];  _scaled_dot_product_flash_attention_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_84: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_212, [0, 2, 1, 3])
        clone_31: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_169: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [16, 512, 1024]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_310: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convert_element_type_311: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        view_170: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_169, [8192, 1024]);  view_169 = None
        permute_85: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_311, [1, 0]);  convert_element_type_311 = None
        addmm_45: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_310, view_170, permute_85);  convert_element_type_310 = None
        view_171: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [16, 512, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_33: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        convert_element_type_default_57: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_33, torch.bfloat16);  inductor_random_default_33 = None
        gt_23: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_57, 0.1);  convert_element_type_default_57 = None
        mul_98: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_23, view_171);  view_171 = None
        mul_99: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_61: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, mul_99);  add_57 = mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
        getitem_30: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_62: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
        sub_24: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_61, getitem_31);  getitem_31 = None
        mul_100: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_15);  sub_24 = None
        mul_101: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, primals_129)
        add_63: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, primals_130);  mul_101 = primals_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_315: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_316: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_131, torch.bfloat16);  primals_131 = None
        convert_element_type_317: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None
        view_172: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_317, [8192, 1024]);  convert_element_type_317 = None
        permute_86: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_316, [1, 0]);  convert_element_type_316 = None
        addmm_46: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_315, view_172, permute_86);  convert_element_type_315 = None
        view_173: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_321: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_102: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, 0.5)
        mul_103: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, 0.7071067811865476);  convert_element_type_321 = None
        erf_7: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_64: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_104: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_102, add_64);  mul_102 = add_64 = None
        convert_element_type_322: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_323: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convert_element_type_324: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        view_174: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_322, [8192, 4096]);  convert_element_type_322 = None
        permute_87: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_324, [1, 0]);  convert_element_type_324 = None
        addmm_47: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_323, view_174, permute_87);  convert_element_type_323 = None
        view_175: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [16, 512, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_32: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        convert_element_type_default_56: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_32, torch.bfloat16);  inductor_random_default_32 = None
        gt_24: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_56, 0.1);  convert_element_type_default_56 = None
        mul_105: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_24, view_175);  view_175 = None
        mul_106: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_65: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_61, mul_106);  add_61 = mul_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
        getitem_32: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_66: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_25: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_65, getitem_33);  getitem_33 = None
        mul_107: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
        mul_108: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, primals_135)
        add_67: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_108, primals_136);  mul_108 = primals_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_328: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        convert_element_type_329: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_137, torch.bfloat16);  primals_137 = None
        convert_element_type_330: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None
        view_176: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_330, [8192, 1024]);  convert_element_type_330 = None
        permute_88: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_329, [1, 0]);  convert_element_type_329 = None
        addmm_48: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_328, view_176, permute_88);  convert_element_type_328 = None
        view_177: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [16, 512, 1024]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_178: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_177, [16, 512, -1, 64]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_334: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convert_element_type_335: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        permute_90: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_335, [1, 0]);  convert_element_type_335 = None
        addmm_49: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_334, view_176, permute_90);  convert_element_type_334 = None
        view_180: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_49, [16, 512, 1024]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_181: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_180, [16, 512, -1, 64]);  view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_340: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_142, torch.bfloat16);  primals_142 = None
        convert_element_type_341: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        permute_92: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_341, [1, 0]);  convert_element_type_341 = None
        addmm_50: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_340, view_176, permute_92);  convert_element_type_340 = None
        view_183: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [16, 512, 1024]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_184: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_183, [16, 512, -1, 64]);  view_183 = None

        # No stacktrace found for following nodes
        permute_default_90: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        permute_default_91: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
        permute_default_92: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
        _scaled_dot_product_flash_attention_default_15 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_90, permute_default_91, permute_default_92, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_205: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_15[0]

        # No stacktrace found for following nodes
        getitem_206: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_15[1]
        getitem_207: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_15[6]
        getitem_208: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_15[7];  _scaled_dot_product_flash_attention_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_95: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3])
        clone_35: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_191: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [16, 512, 1024]);  clone_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_351: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        convert_element_type_352: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_143, torch.bfloat16);  primals_143 = None
        view_192: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [8192, 1024]);  view_191 = None
        permute_96: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_352, [1, 0]);  convert_element_type_352 = None
        addmm_51: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_351, view_192, permute_96);  convert_element_type_351 = None
        view_193: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_51, [16, 512, 1024]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_31: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        convert_element_type_default_55: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_31, torch.bfloat16);  inductor_random_default_31 = None
        gt_26: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_55, 0.1);  convert_element_type_default_55 = None
        mul_111: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_26, view_193);  view_193 = None
        mul_112: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, 1.1111111111111112);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_69: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_65, mul_112);  add_65 = mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
        getitem_34: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_70: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_27: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_69, getitem_35);  getitem_35 = None
        mul_113: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_17);  sub_27 = None
        mul_114: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, primals_145)
        add_71: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, primals_146);  mul_114 = primals_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_356: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convert_element_type_357: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convert_element_type_358: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None
        view_194: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_358, [8192, 1024]);  convert_element_type_358 = None
        permute_97: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_357, [1, 0]);  convert_element_type_357 = None
        addmm_52: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_356, view_194, permute_97);  convert_element_type_356 = None
        view_195: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_362: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_115: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, 0.5)
        mul_116: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, 0.7071067811865476);  convert_element_type_362 = None
        erf_8: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_72: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_117: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, add_72);  mul_115 = add_72 = None
        convert_element_type_363: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_117, torch.bfloat16);  mul_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_364: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_365: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        view_196: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_363, [8192, 4096]);  convert_element_type_363 = None
        permute_98: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_365, [1, 0]);  convert_element_type_365 = None
        addmm_53: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_364, view_196, permute_98);  convert_element_type_364 = None
        view_197: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_53, [16, 512, 1024]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_30: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        convert_element_type_default_54: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_30, torch.bfloat16);  inductor_random_default_30 = None
        gt_27: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_54, 0.1);  convert_element_type_default_54 = None
        mul_118: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_27, view_197);  view_197 = None
        mul_119: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, 1.1111111111111112);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_73: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_69, mul_119);  add_69 = mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_36: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_74: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_28: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_73, getitem_37);  getitem_37 = None
        mul_120: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
        mul_121: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, primals_151)
        add_75: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_121, primals_152);  mul_121 = primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_369: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        convert_element_type_370: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        convert_element_type_371: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16);  add_75 = None
        view_198: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_371, [8192, 1024]);  convert_element_type_371 = None
        permute_99: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_370, [1, 0]);  convert_element_type_370 = None
        addmm_54: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_369, view_198, permute_99);  convert_element_type_369 = None
        view_199: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [16, 512, 1024]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_200: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_199, [16, 512, -1, 64]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_375: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        convert_element_type_376: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        permute_101: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_376, [1, 0]);  convert_element_type_376 = None
        addmm_55: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_375, view_198, permute_101);  convert_element_type_375 = None
        view_202: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_55, [16, 512, 1024]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_203: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_202, [16, 512, -1, 64]);  view_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_381: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        convert_element_type_382: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        permute_103: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_382, [1, 0]);  convert_element_type_382 = None
        addmm_56: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_381, view_198, permute_103);  convert_element_type_381 = None
        view_205: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_56, [16, 512, 1024]);  addmm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_206: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_205, [16, 512, -1, 64]);  view_205 = None

        # No stacktrace found for following nodes
        permute_default_84: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        permute_default_85: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
        permute_default_86: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        _scaled_dot_product_flash_attention_default_14 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_84, permute_default_85, permute_default_86, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_198: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_14[0]

        # No stacktrace found for following nodes
        getitem_199: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_14[1]
        getitem_200: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_14[6]
        getitem_201: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_14[7];  _scaled_dot_product_flash_attention_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_106: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3])
        clone_39: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_213: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [16, 512, 1024]);  clone_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_392: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_160, torch.bfloat16);  primals_160 = None
        convert_element_type_393: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        view_214: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [8192, 1024]);  view_213 = None
        permute_107: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_393, [1, 0]);  convert_element_type_393 = None
        addmm_57: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_392, view_214, permute_107);  convert_element_type_392 = None
        view_215: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_57, [16, 512, 1024]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_29: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        convert_element_type_default_53: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_29, torch.bfloat16);  inductor_random_default_29 = None
        gt_29: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_53, 0.1);  convert_element_type_default_53 = None
        mul_124: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_29, view_215);  view_215 = None
        mul_125: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, 1.1111111111111112);  mul_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_77: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_73, mul_125);  add_73 = mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_38: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_78: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        sub_30: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_77, getitem_39);  getitem_39 = None
        mul_126: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_19);  sub_30 = None
        mul_127: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, primals_161)
        add_79: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, primals_162);  mul_127 = primals_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_397: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convert_element_type_398: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convert_element_type_399: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None
        view_216: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_399, [8192, 1024]);  convert_element_type_399 = None
        permute_108: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_398, [1, 0]);  convert_element_type_398 = None
        addmm_58: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_397, view_216, permute_108);  convert_element_type_397 = None
        view_217: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_403: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_128: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, 0.5)
        mul_129: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, 0.7071067811865476);  convert_element_type_403 = None
        erf_9: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_80: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_130: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, add_80);  mul_128 = add_80 = None
        convert_element_type_404: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_130, torch.bfloat16);  mul_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_405: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_166, torch.bfloat16);  primals_166 = None
        convert_element_type_406: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        view_218: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_404, [8192, 4096]);  convert_element_type_404 = None
        permute_109: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_406, [1, 0]);  convert_element_type_406 = None
        addmm_59: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_405, view_218, permute_109);  convert_element_type_405 = None
        view_219: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_59, [16, 512, 1024]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_28: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        convert_element_type_default_52: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_28, torch.bfloat16);  inductor_random_default_28 = None
        gt_30: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_52, 0.1);  convert_element_type_default_52 = None
        mul_131: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_30, view_219);  view_219 = None
        mul_132: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, 1.1111111111111112);  mul_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_81: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_77, mul_132);  add_77 = mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_40: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_82: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_31: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_81, getitem_41);  getitem_41 = None
        mul_133: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
        mul_134: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, primals_167)
        add_83: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, primals_168);  mul_134 = primals_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_410: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convert_element_type_411: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        convert_element_type_412: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16);  add_83 = None
        view_220: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_412, [8192, 1024]);  convert_element_type_412 = None
        permute_110: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_411, [1, 0]);  convert_element_type_411 = None
        addmm_60: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_410, view_220, permute_110);  convert_element_type_410 = None
        view_221: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_60, [16, 512, 1024]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_222: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [16, 512, -1, 64]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_416: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_172, torch.bfloat16);  primals_172 = None
        convert_element_type_417: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        permute_112: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_417, [1, 0]);  convert_element_type_417 = None
        addmm_61: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_416, view_220, permute_112);  convert_element_type_416 = None
        view_224: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_61, [16, 512, 1024]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_225: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_224, [16, 512, -1, 64]);  view_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_422: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        convert_element_type_423: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_173, torch.bfloat16);  primals_173 = None
        permute_114: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_423, [1, 0]);  convert_element_type_423 = None
        addmm_62: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_422, view_220, permute_114);  convert_element_type_422 = None
        view_227: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [16, 512, 1024]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_228: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_227, [16, 512, -1, 64]);  view_227 = None

        # No stacktrace found for following nodes
        permute_default_78: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        permute_default_79: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        permute_default_80: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
        _scaled_dot_product_flash_attention_default_13 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_78, permute_default_79, permute_default_80, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_191: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_13[0]

        # No stacktrace found for following nodes
        getitem_192: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_13[1]
        getitem_193: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_13[6]
        getitem_194: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_13[7];  _scaled_dot_product_flash_attention_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_117: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_191, [0, 2, 1, 3])
        clone_43: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_235: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [16, 512, 1024]);  clone_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_433: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convert_element_type_434: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        view_236: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_235, [8192, 1024]);  view_235 = None
        permute_118: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_434, [1, 0]);  convert_element_type_434 = None
        addmm_63: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_433, view_236, permute_118);  convert_element_type_433 = None
        view_237: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_63, [16, 512, 1024]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_27: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        convert_element_type_default_51: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_27, torch.bfloat16);  inductor_random_default_27 = None
        gt_32: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_51, 0.1);  convert_element_type_default_51 = None
        mul_137: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_32, view_237);  view_237 = None
        mul_138: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, 1.1111111111111112);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_85: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_81, mul_138);  add_81 = mul_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_42: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_86: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_33: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_85, getitem_43);  getitem_43 = None
        mul_139: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_21);  sub_33 = None
        mul_140: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, primals_177)
        add_87: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_140, primals_178);  mul_140 = primals_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_438: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16);  primals_180 = None
        convert_element_type_439: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        convert_element_type_440: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16);  add_87 = None
        view_238: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_440, [8192, 1024]);  convert_element_type_440 = None
        permute_119: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_439, [1, 0]);  convert_element_type_439 = None
        addmm_64: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_438, view_238, permute_119);  convert_element_type_438 = None
        view_239: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_444: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_141: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, 0.5)
        mul_142: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, 0.7071067811865476);  convert_element_type_444 = None
        erf_10: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_88: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_143: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, add_88);  mul_141 = add_88 = None
        convert_element_type_445: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_446: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convert_element_type_447: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        view_240: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_445, [8192, 4096]);  convert_element_type_445 = None
        permute_120: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_447, [1, 0]);  convert_element_type_447 = None
        addmm_65: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_446, view_240, permute_120);  convert_element_type_446 = None
        view_241: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_65, [16, 512, 1024]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_26: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        convert_element_type_default_50: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_26, torch.bfloat16);  inductor_random_default_26 = None
        gt_33: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_50, 0.1);  convert_element_type_default_50 = None
        mul_144: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_33, view_241);  view_241 = None
        mul_145: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_89: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_85, mul_145);  add_85 = mul_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
        getitem_44: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_90: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_34: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_89, getitem_45);  getitem_45 = None
        mul_146: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
        mul_147: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, primals_183)
        add_91: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_147, primals_184);  mul_147 = primals_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_451: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16);  primals_186 = None
        convert_element_type_452: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        convert_element_type_453: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16);  add_91 = None
        view_242: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_453, [8192, 1024]);  convert_element_type_453 = None
        permute_121: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_452, [1, 0]);  convert_element_type_452 = None
        addmm_66: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_451, view_242, permute_121);  convert_element_type_451 = None
        view_243: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [16, 512, 1024]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_244: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_243, [16, 512, -1, 64]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_457: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convert_element_type_458: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16);  primals_187 = None
        permute_123: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_458, [1, 0]);  convert_element_type_458 = None
        addmm_67: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_457, view_242, permute_123);  convert_element_type_457 = None
        view_246: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_67, [16, 512, 1024]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_247: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_246, [16, 512, -1, 64]);  view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_463: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_190, torch.bfloat16);  primals_190 = None
        convert_element_type_464: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        permute_125: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_464, [1, 0]);  convert_element_type_464 = None
        addmm_68: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_463, view_242, permute_125);  convert_element_type_463 = None
        view_249: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_68, [16, 512, 1024]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_250: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_249, [16, 512, -1, 64]);  view_249 = None

        # No stacktrace found for following nodes
        permute_default_72: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        permute_default_73: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
        permute_default_74: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        _scaled_dot_product_flash_attention_default_12 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_72, permute_default_73, permute_default_74, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_184: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_12[0]

        # No stacktrace found for following nodes
        getitem_185: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_12[1]
        getitem_186: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_12[6]
        getitem_187: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_12[7];  _scaled_dot_product_flash_attention_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_128: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_184, [0, 2, 1, 3])
        clone_47: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_257: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [16, 512, 1024]);  clone_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_474: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16);  primals_192 = None
        convert_element_type_475: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_191, torch.bfloat16);  primals_191 = None
        view_258: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [8192, 1024]);  view_257 = None
        permute_129: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_475, [1, 0]);  convert_element_type_475 = None
        addmm_69: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_474, view_258, permute_129);  convert_element_type_474 = None
        view_259: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_69, [16, 512, 1024]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_25: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        convert_element_type_default_49: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_25, torch.bfloat16);  inductor_random_default_25 = None
        gt_35: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_49, 0.1);  convert_element_type_default_49 = None
        mul_150: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_35, view_259);  view_259 = None
        mul_151: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 1.1111111111111112);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_93: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_89, mul_151);  add_89 = mul_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
        getitem_46: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_94: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
        sub_36: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_93, getitem_47);  getitem_47 = None
        mul_152: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_23);  sub_36 = None
        mul_153: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, primals_193)
        add_95: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_153, primals_194);  mul_153 = primals_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_479: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_196, torch.bfloat16);  primals_196 = None
        convert_element_type_480: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16);  primals_195 = None
        convert_element_type_481: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_95, torch.bfloat16);  add_95 = None
        view_260: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_481, [8192, 1024]);  convert_element_type_481 = None
        permute_130: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_480, [1, 0]);  convert_element_type_480 = None
        addmm_70: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_479, view_260, permute_130);  convert_element_type_479 = None
        view_261: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_485: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_154: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_485, 0.5)
        mul_155: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_485, 0.7071067811865476);  convert_element_type_485 = None
        erf_11: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
        add_96: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_156: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, add_96);  mul_154 = add_96 = None
        convert_element_type_486: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_487: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        convert_element_type_488: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_197, torch.bfloat16);  primals_197 = None
        view_262: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_486, [8192, 4096]);  convert_element_type_486 = None
        permute_131: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_488, [1, 0]);  convert_element_type_488 = None
        addmm_71: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_487, view_262, permute_131);  convert_element_type_487 = None
        view_263: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_71, [16, 512, 1024]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_24: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        convert_element_type_default_48: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_24, torch.bfloat16);  inductor_random_default_24 = None
        gt_36: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_48, 0.1);  convert_element_type_default_48 = None
        mul_157: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_36, view_263);  view_263 = None
        mul_158: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, 1.1111111111111112);  mul_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_97: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_93, mul_158);  add_93 = mul_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
        getitem_48: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_98: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
        sub_37: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_97, getitem_49);  getitem_49 = None
        mul_159: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
        mul_160: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_159, primals_199)
        add_99: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_160, primals_200);  mul_160 = primals_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_492: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_202, torch.bfloat16);  primals_202 = None
        convert_element_type_493: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16);  primals_201 = None
        convert_element_type_494: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        view_264: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_494, [8192, 1024]);  convert_element_type_494 = None
        permute_132: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_493, [1, 0]);  convert_element_type_493 = None
        addmm_72: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_492, view_264, permute_132);  convert_element_type_492 = None
        view_265: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_72, [16, 512, 1024]);  addmm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_266: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_265, [16, 512, -1, 64]);  view_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_498: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16);  primals_204 = None
        convert_element_type_499: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_203, torch.bfloat16);  primals_203 = None
        permute_134: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_499, [1, 0]);  convert_element_type_499 = None
        addmm_73: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_498, view_264, permute_134);  convert_element_type_498 = None
        view_268: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_73, [16, 512, 1024]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_269: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_268, [16, 512, -1, 64]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_504: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        convert_element_type_505: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16);  primals_205 = None
        permute_136: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_505, [1, 0]);  convert_element_type_505 = None
        addmm_74: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_504, view_264, permute_136);  convert_element_type_504 = None
        view_271: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [16, 512, 1024]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_272: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_271, [16, 512, -1, 64]);  view_271 = None

        # No stacktrace found for following nodes
        permute_default_66: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        permute_default_67: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
        permute_default_68: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        _scaled_dot_product_flash_attention_default_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_66, permute_default_67, permute_default_68, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_177: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_11[0]

        # No stacktrace found for following nodes
        getitem_178: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_11[1]
        getitem_179: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_11[6]
        getitem_180: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_11[7];  _scaled_dot_product_flash_attention_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_139: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_177, [0, 2, 1, 3])
        clone_51: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_279: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_51, [16, 512, 1024]);  clone_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_515: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16);  primals_208 = None
        convert_element_type_516: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16);  primals_207 = None
        view_280: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_279, [8192, 1024]);  view_279 = None
        permute_140: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_516, [1, 0]);  convert_element_type_516 = None
        addmm_75: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_515, view_280, permute_140);  convert_element_type_515 = None
        view_281: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_75, [16, 512, 1024]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_25: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_23: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        convert_element_type_default_47: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_23, torch.bfloat16);  inductor_random_default_23 = None
        gt_38: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_47, 0.1);  convert_element_type_default_47 = None
        mul_163: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_38, view_281);  view_281 = None
        mul_164: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, 1.1111111111111112);  mul_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_101: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, mul_164);  add_97 = mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_101, [2], correction = 0, keepdim = True)
        getitem_50: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_102: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        sub_39: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_101, getitem_51);  getitem_51 = None
        mul_165: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = None
        mul_166: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, primals_209)
        add_103: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_166, primals_210);  mul_166 = primals_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_520: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        convert_element_type_521: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_211, torch.bfloat16);  primals_211 = None
        convert_element_type_522: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16);  add_103 = None
        view_282: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_522, [8192, 1024]);  convert_element_type_522 = None
        permute_141: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_521, [1, 0]);  convert_element_type_521 = None
        addmm_76: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_520, view_282, permute_141);  convert_element_type_520 = None
        view_283: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_526: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        mul_167: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, 0.5)
        mul_168: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, 0.7071067811865476);  convert_element_type_526 = None
        erf_12: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_104: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_169: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, add_104);  mul_167 = add_104 = None
        convert_element_type_527: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_169, torch.bfloat16);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_528: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16);  primals_214 = None
        convert_element_type_529: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16);  primals_213 = None
        view_284: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_527, [8192, 4096]);  convert_element_type_527 = None
        permute_142: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_529, [1, 0]);  convert_element_type_529 = None
        addmm_77: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_528, view_284, permute_142);  convert_element_type_528 = None
        view_285: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_77, [16, 512, 1024]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_26: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_22: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        convert_element_type_default_46: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_22, torch.bfloat16);  inductor_random_default_22 = None
        gt_39: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_46, 0.1);  convert_element_type_default_46 = None
        mul_170: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_39, view_285);  view_285 = None
        mul_171: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, 1.1111111111111112);  mul_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_105: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, mul_171);  add_101 = mul_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
        getitem_52: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_106: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_40: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_105, getitem_53);  getitem_53 = None
        mul_172: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = None
        mul_173: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, primals_215)
        add_107: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_173, primals_216);  mul_173 = primals_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_533: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        convert_element_type_534: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_217, torch.bfloat16);  primals_217 = None
        convert_element_type_535: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16);  add_107 = None
        view_286: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_535, [8192, 1024]);  convert_element_type_535 = None
        permute_143: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_534, [1, 0]);  convert_element_type_534 = None
        addmm_78: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_533, view_286, permute_143);  convert_element_type_533 = None
        view_287: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [16, 512, 1024]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_288: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_287, [16, 512, -1, 64]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_539: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_220, torch.bfloat16);  primals_220 = None
        convert_element_type_540: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_219, torch.bfloat16);  primals_219 = None
        permute_145: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_540, [1, 0]);  convert_element_type_540 = None
        addmm_79: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_539, view_286, permute_145);  convert_element_type_539 = None
        view_290: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_79, [16, 512, 1024]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_291: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_290, [16, 512, -1, 64]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_545: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16);  primals_222 = None
        convert_element_type_546: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_221, torch.bfloat16);  primals_221 = None
        permute_147: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_546, [1, 0]);  convert_element_type_546 = None
        addmm_80: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_545, view_286, permute_147);  convert_element_type_545 = None
        view_293: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_80, [16, 512, 1024]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_294: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_293, [16, 512, -1, 64]);  view_293 = None

        # No stacktrace found for following nodes
        permute_default_60: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        permute_default_61: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
        permute_default_62: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        _scaled_dot_product_flash_attention_default_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_60, permute_default_61, permute_default_62, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_170: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_10[0]

        # No stacktrace found for following nodes
        getitem_171: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_10[1]
        getitem_172: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_10[6]
        getitem_173: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_10[7];  _scaled_dot_product_flash_attention_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_150: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3])
        clone_55: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_301: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_55, [16, 512, 1024]);  clone_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_556: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        convert_element_type_557: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_223, torch.bfloat16);  primals_223 = None
        view_302: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [8192, 1024]);  view_301 = None
        permute_151: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_557, [1, 0]);  convert_element_type_557 = None
        addmm_81: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_556, view_302, permute_151);  convert_element_type_556 = None
        view_303: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_81, [16, 512, 1024]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_27: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_21: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        convert_element_type_default_45: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_21, torch.bfloat16);  inductor_random_default_21 = None
        gt_41: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_45, 0.1);  convert_element_type_default_45 = None
        mul_176: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_41, view_303);  view_303 = None
        mul_177: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, 1.1111111111111112);  mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_109: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, mul_177);  add_105 = mul_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
        getitem_54: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_110: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
        sub_42: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_109, getitem_55);  getitem_55 = None
        mul_178: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_27);  sub_42 = None
        mul_179: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, primals_225)
        add_111: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, primals_226);  mul_179 = primals_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_561: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_228, torch.bfloat16);  primals_228 = None
        convert_element_type_562: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_227, torch.bfloat16);  primals_227 = None
        convert_element_type_563: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None
        view_304: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_563, [8192, 1024]);  convert_element_type_563 = None
        permute_152: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_562, [1, 0]);  convert_element_type_562 = None
        addmm_82: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_561, view_304, permute_152);  convert_element_type_561 = None
        view_305: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_567: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        mul_180: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, 0.5)
        mul_181: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, 0.7071067811865476);  convert_element_type_567 = None
        erf_13: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_112: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_182: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, add_112);  mul_180 = add_112 = None
        convert_element_type_568: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_182, torch.bfloat16);  mul_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_569: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        convert_element_type_570: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_229, torch.bfloat16);  primals_229 = None
        view_306: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_568, [8192, 4096]);  convert_element_type_568 = None
        permute_153: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_570, [1, 0]);  convert_element_type_570 = None
        addmm_83: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_569, view_306, permute_153);  convert_element_type_569 = None
        view_307: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_83, [16, 512, 1024]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_28: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_20: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        convert_element_type_default_44: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_20, torch.bfloat16);  inductor_random_default_20 = None
        gt_42: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_44, 0.1);  convert_element_type_default_44 = None
        mul_183: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_42, view_307);  view_307 = None
        mul_184: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, 1.1111111111111112);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_113: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, mul_184);  add_109 = mul_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_113, [2], correction = 0, keepdim = True)
        getitem_56: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_114: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_114);  add_114 = None
        sub_43: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_113, getitem_57);  getitem_57 = None
        mul_185: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = None
        mul_186: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, primals_231)
        add_115: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_186, primals_232);  mul_186 = primals_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_574: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_234, torch.bfloat16);  primals_234 = None
        convert_element_type_575: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_233, torch.bfloat16);  primals_233 = None
        convert_element_type_576: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_115, torch.bfloat16);  add_115 = None
        view_308: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_576, [8192, 1024]);  convert_element_type_576 = None
        permute_154: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_575, [1, 0]);  convert_element_type_575 = None
        addmm_84: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_574, view_308, permute_154);  convert_element_type_574 = None
        view_309: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_84, [16, 512, 1024]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_310: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [16, 512, -1, 64]);  view_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_580: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        convert_element_type_581: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_235, torch.bfloat16);  primals_235 = None
        permute_156: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_581, [1, 0]);  convert_element_type_581 = None
        addmm_85: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_580, view_308, permute_156);  convert_element_type_580 = None
        view_312: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_85, [16, 512, 1024]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_313: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_312, [16, 512, -1, 64]);  view_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_586: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_238, torch.bfloat16);  primals_238 = None
        convert_element_type_587: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_237, torch.bfloat16);  primals_237 = None
        permute_158: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_587, [1, 0]);  convert_element_type_587 = None
        addmm_86: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_586, view_308, permute_158);  convert_element_type_586 = None
        view_315: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_86, [16, 512, 1024]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_316: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_315, [16, 512, -1, 64]);  view_315 = None

        # No stacktrace found for following nodes
        permute_default_54: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
        permute_default_55: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
        permute_default_56: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
        _scaled_dot_product_flash_attention_default_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_54, permute_default_55, permute_default_56, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_163: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_9[0]

        # No stacktrace found for following nodes
        getitem_164: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_9[1]
        getitem_165: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_9[6]
        getitem_166: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_9[7];  _scaled_dot_product_flash_attention_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_161: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_163, [0, 2, 1, 3])
        clone_59: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_323: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_59, [16, 512, 1024]);  clone_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_597: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_240, torch.bfloat16);  primals_240 = None
        convert_element_type_598: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_239, torch.bfloat16);  primals_239 = None
        view_324: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_323, [8192, 1024]);  view_323 = None
        permute_162: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_598, [1, 0]);  convert_element_type_598 = None
        addmm_87: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_597, view_324, permute_162);  convert_element_type_597 = None
        view_325: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_87, [16, 512, 1024]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_29: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_19: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        convert_element_type_default_43: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_19, torch.bfloat16);  inductor_random_default_19 = None
        gt_44: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_43, 0.1);  convert_element_type_default_43 = None
        mul_189: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_44, view_325);  view_325 = None
        mul_190: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, 1.1111111111111112);  mul_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_117: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_113, mul_190);  add_113 = mul_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_29 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_58: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_118: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        sub_45: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_117, getitem_59);  getitem_59 = None
        mul_191: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_29);  sub_45 = None
        mul_192: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_191, primals_241)
        add_119: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_192, primals_242);  mul_192 = primals_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_602: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_244, torch.bfloat16);  primals_244 = None
        convert_element_type_603: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_243, torch.bfloat16);  primals_243 = None
        convert_element_type_604: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None
        view_326: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_604, [8192, 1024]);  convert_element_type_604 = None
        permute_163: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_603, [1, 0]);  convert_element_type_603 = None
        addmm_88: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_602, view_326, permute_163);  convert_element_type_602 = None
        view_327: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_608: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        mul_193: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_608, 0.5)
        mul_194: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_608, 0.7071067811865476);  convert_element_type_608 = None
        erf_14: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_194);  mul_194 = None
        add_120: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_195: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_193, add_120);  mul_193 = add_120 = None
        convert_element_type_609: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_195, torch.bfloat16);  mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_610: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_246, torch.bfloat16);  primals_246 = None
        convert_element_type_611: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_245, torch.bfloat16);  primals_245 = None
        view_328: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_609, [8192, 4096]);  convert_element_type_609 = None
        permute_164: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_611, [1, 0]);  convert_element_type_611 = None
        addmm_89: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_610, view_328, permute_164);  convert_element_type_610 = None
        view_329: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_89, [16, 512, 1024]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_30: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_18: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        convert_element_type_default_42: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_18, torch.bfloat16);  inductor_random_default_18 = None
        gt_45: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_42, 0.1);  convert_element_type_default_42 = None
        mul_196: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_45, view_329);  view_329 = None
        mul_197: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, 1.1111111111111112);  mul_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_121: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, mul_197);  add_117 = mul_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_30 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_60: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_122: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_46: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_121, getitem_61);  getitem_61 = None
        mul_198: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = None
        mul_199: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, primals_247)
        add_123: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, primals_248);  mul_199 = primals_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_615: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_250, torch.bfloat16);  primals_250 = None
        convert_element_type_616: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_249, torch.bfloat16);  primals_249 = None
        convert_element_type_617: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16);  add_123 = None
        view_330: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_617, [8192, 1024]);  convert_element_type_617 = None
        permute_165: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_616, [1, 0]);  convert_element_type_616 = None
        addmm_90: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_615, view_330, permute_165);  convert_element_type_615 = None
        view_331: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_90, [16, 512, 1024]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_332: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_331, [16, 512, -1, 64]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_621: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_252, torch.bfloat16);  primals_252 = None
        convert_element_type_622: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_251, torch.bfloat16);  primals_251 = None
        permute_167: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_622, [1, 0]);  convert_element_type_622 = None
        addmm_91: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_621, view_330, permute_167);  convert_element_type_621 = None
        view_334: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_91, [16, 512, 1024]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_335: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_334, [16, 512, -1, 64]);  view_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_627: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        convert_element_type_628: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_253, torch.bfloat16);  primals_253 = None
        permute_169: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_628, [1, 0]);  convert_element_type_628 = None
        addmm_92: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_627, view_330, permute_169);  convert_element_type_627 = None
        view_337: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_92, [16, 512, 1024]);  addmm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_338: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_337, [16, 512, -1, 64]);  view_337 = None

        # No stacktrace found for following nodes
        permute_default_48: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None
        permute_default_49: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None
        permute_default_50: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        _scaled_dot_product_flash_attention_default_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_48, permute_default_49, permute_default_50, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_156: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_8[0]

        # No stacktrace found for following nodes
        getitem_157: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_8[1]
        getitem_158: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_8[6]
        getitem_159: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_8[7];  _scaled_dot_product_flash_attention_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_172: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_156, [0, 2, 1, 3])
        clone_63: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_345: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_63, [16, 512, 1024]);  clone_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_638: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_256, torch.bfloat16);  primals_256 = None
        convert_element_type_639: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_255, torch.bfloat16);  primals_255 = None
        view_346: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_345, [8192, 1024]);  view_345 = None
        permute_173: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_639, [1, 0]);  convert_element_type_639 = None
        addmm_93: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_638, view_346, permute_173);  convert_element_type_638 = None
        view_347: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_93, [16, 512, 1024]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_31: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_17: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        convert_element_type_default_41: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_17, torch.bfloat16);  inductor_random_default_17 = None
        gt_47: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_41, 0.1);  convert_element_type_default_41 = None
        mul_202: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_47, view_347);  view_347 = None
        mul_203: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_202, 1.1111111111111112);  mul_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_125: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, mul_203);  add_121 = mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_31 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_62: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_126: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_48: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_125, getitem_63);  getitem_63 = None
        mul_204: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_31);  sub_48 = None
        mul_205: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, primals_257)
        add_127: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_205, primals_258);  mul_205 = primals_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_643: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        convert_element_type_644: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16);  primals_259 = None
        convert_element_type_645: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16);  add_127 = None
        view_348: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_645, [8192, 1024]);  convert_element_type_645 = None
        permute_174: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_644, [1, 0]);  convert_element_type_644 = None
        addmm_94: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_643, view_348, permute_174);  convert_element_type_643 = None
        view_349: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_649: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        mul_206: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_649, 0.5)
        mul_207: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_649, 0.7071067811865476);  convert_element_type_649 = None
        erf_15: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_128: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_208: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, add_128);  mul_206 = add_128 = None
        convert_element_type_650: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_208, torch.bfloat16);  mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_651: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_262, torch.bfloat16);  primals_262 = None
        convert_element_type_652: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_261, torch.bfloat16);  primals_261 = None
        view_350: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_650, [8192, 4096]);  convert_element_type_650 = None
        permute_175: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_652, [1, 0]);  convert_element_type_652 = None
        addmm_95: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_651, view_350, permute_175);  convert_element_type_651 = None
        view_351: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_95, [16, 512, 1024]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_32: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_16: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        convert_element_type_default_40: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_16, torch.bfloat16);  inductor_random_default_16 = None
        gt_48: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_40, 0.1);  convert_element_type_default_40 = None
        mul_209: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_48, view_351);  view_351 = None
        mul_210: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_209, 1.1111111111111112);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_129: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, mul_210);  add_125 = mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_32 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_64: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_130: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_49: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_129, getitem_65);  getitem_65 = None
        mul_211: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = None
        mul_212: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, primals_263)
        add_131: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_212, primals_264);  mul_212 = primals_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_656: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convert_element_type_657: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_265, torch.bfloat16);  primals_265 = None
        convert_element_type_658: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16);  add_131 = None
        view_352: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_658, [8192, 1024]);  convert_element_type_658 = None
        permute_176: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_657, [1, 0]);  convert_element_type_657 = None
        addmm_96: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_656, view_352, permute_176);  convert_element_type_656 = None
        view_353: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_96, [16, 512, 1024]);  addmm_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_354: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_353, [16, 512, -1, 64]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_662: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_268, torch.bfloat16);  primals_268 = None
        convert_element_type_663: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_267, torch.bfloat16);  primals_267 = None
        permute_178: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_663, [1, 0]);  convert_element_type_663 = None
        addmm_97: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_662, view_352, permute_178);  convert_element_type_662 = None
        view_356: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_97, [16, 512, 1024]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_357: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_356, [16, 512, -1, 64]);  view_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_668: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_270, torch.bfloat16);  primals_270 = None
        convert_element_type_669: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_269, torch.bfloat16);  primals_269 = None
        permute_180: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_669, [1, 0]);  convert_element_type_669 = None
        addmm_98: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_668, view_352, permute_180);  convert_element_type_668 = None
        view_359: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_98, [16, 512, 1024]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_360: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_359, [16, 512, -1, 64]);  view_359 = None

        # No stacktrace found for following nodes
        permute_default_42: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        permute_default_43: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
        permute_default_44: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
        _scaled_dot_product_flash_attention_default_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_42, permute_default_43, permute_default_44, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_149: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_7[0]

        # No stacktrace found for following nodes
        getitem_150: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_7[1]
        getitem_151: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_7[6]
        getitem_152: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_7[7];  _scaled_dot_product_flash_attention_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_183: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3])
        clone_67: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_367: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_67, [16, 512, 1024]);  clone_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_679: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        convert_element_type_680: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_271, torch.bfloat16);  primals_271 = None
        view_368: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_367, [8192, 1024]);  view_367 = None
        permute_184: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_680, [1, 0]);  convert_element_type_680 = None
        addmm_99: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_679, view_368, permute_184);  convert_element_type_679 = None
        view_369: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_99, [16, 512, 1024]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_33: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_15: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        convert_element_type_default_39: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_15, torch.bfloat16);  inductor_random_default_15 = None
        gt_50: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_39, 0.1);  convert_element_type_default_39 = None
        mul_215: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_50, view_369);  view_369 = None
        mul_216: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_215, 1.1111111111111112);  mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_133: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, mul_216);  add_129 = mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_33 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
        getitem_66: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_134: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        sub_51: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_133, getitem_67);  getitem_67 = None
        mul_217: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_33);  sub_51 = None
        mul_218: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, primals_273)
        add_135: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, primals_274);  mul_218 = primals_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_684: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16);  primals_276 = None
        convert_element_type_685: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16);  primals_275 = None
        convert_element_type_686: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16);  add_135 = None
        view_370: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_686, [8192, 1024]);  convert_element_type_686 = None
        permute_185: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_685, [1, 0]);  convert_element_type_685 = None
        addmm_100: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_684, view_370, permute_185);  convert_element_type_684 = None
        view_371: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_100, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_690: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        mul_219: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, 0.5)
        mul_220: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, 0.7071067811865476);  convert_element_type_690 = None
        erf_16: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_220);  mul_220 = None
        add_136: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_221: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_219, add_136);  mul_219 = add_136 = None
        convert_element_type_691: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_692: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16);  primals_278 = None
        convert_element_type_693: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_277, torch.bfloat16);  primals_277 = None
        view_372: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_691, [8192, 4096]);  convert_element_type_691 = None
        permute_186: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_693, [1, 0]);  convert_element_type_693 = None
        addmm_101: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_692, view_372, permute_186);  convert_element_type_692 = None
        view_373: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_101, [16, 512, 1024]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_34: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_14: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        convert_element_type_default_38: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_14, torch.bfloat16);  inductor_random_default_14 = None
        gt_51: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_38, 0.1);  convert_element_type_default_38 = None
        mul_222: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_51, view_373);  view_373 = None
        mul_223: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_222, 1.1111111111111112);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_137: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, mul_223);  add_133 = mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_34 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
        getitem_68: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_138: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_52: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_137, getitem_69);  getitem_69 = None
        mul_224: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = None
        mul_225: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, primals_279)
        add_139: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_225, primals_280);  mul_225 = primals_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_697: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16);  primals_282 = None
        convert_element_type_698: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_281, torch.bfloat16);  primals_281 = None
        convert_element_type_699: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None
        view_374: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_699, [8192, 1024]);  convert_element_type_699 = None
        permute_187: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_698, [1, 0]);  convert_element_type_698 = None
        addmm_102: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_697, view_374, permute_187);  convert_element_type_697 = None
        view_375: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_102, [16, 512, 1024]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_376: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_375, [16, 512, -1, 64]);  view_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_703: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        convert_element_type_704: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_283, torch.bfloat16);  primals_283 = None
        permute_189: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_704, [1, 0]);  convert_element_type_704 = None
        addmm_103: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_703, view_374, permute_189);  convert_element_type_703 = None
        view_378: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_103, [16, 512, 1024]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_379: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_378, [16, 512, -1, 64]);  view_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_709: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_286, torch.bfloat16);  primals_286 = None
        convert_element_type_710: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_285, torch.bfloat16);  primals_285 = None
        permute_191: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_710, [1, 0]);  convert_element_type_710 = None
        addmm_104: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_709, view_374, permute_191);  convert_element_type_709 = None
        view_381: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_104, [16, 512, 1024]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_382: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_381, [16, 512, -1, 64]);  view_381 = None

        # No stacktrace found for following nodes
        permute_default_36: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        permute_default_37: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        permute_default_38: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        _scaled_dot_product_flash_attention_default_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_36, permute_default_37, permute_default_38, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_142: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_6[0]

        # No stacktrace found for following nodes
        getitem_143: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_6[1]
        getitem_144: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_6[6]
        getitem_145: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_6[7];  _scaled_dot_product_flash_attention_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_194: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3])
        clone_71: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_389: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [16, 512, 1024]);  clone_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_720: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_288, torch.bfloat16);  primals_288 = None
        convert_element_type_721: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_287, torch.bfloat16);  primals_287 = None
        view_390: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_389, [8192, 1024]);  view_389 = None
        permute_195: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_721, [1, 0]);  convert_element_type_721 = None
        addmm_105: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_720, view_390, permute_195);  convert_element_type_720 = None
        view_391: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_105, [16, 512, 1024]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_35: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_13: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        convert_element_type_default_37: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_13, torch.bfloat16);  inductor_random_default_13 = None
        gt_53: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_37, 0.1);  convert_element_type_default_37 = None
        mul_228: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_53, view_391);  view_391 = None
        mul_229: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_228, 1.1111111111111112);  mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_141: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_137, mul_229);  add_137 = mul_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_35 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
        getitem_70: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_142: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
        sub_54: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_141, getitem_71);  getitem_71 = None
        mul_230: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_35);  sub_54 = None
        mul_231: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, primals_289)
        add_143: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_231, primals_290);  mul_231 = primals_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_725: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_292, torch.bfloat16);  primals_292 = None
        convert_element_type_726: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_291, torch.bfloat16);  primals_291 = None
        convert_element_type_727: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        view_392: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_727, [8192, 1024]);  convert_element_type_727 = None
        permute_196: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_726, [1, 0]);  convert_element_type_726 = None
        addmm_106: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_725, view_392, permute_196);  convert_element_type_725 = None
        view_393: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_106, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_731: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_232: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, 0.5)
        mul_233: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, 0.7071067811865476);  convert_element_type_731 = None
        erf_17: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
        add_144: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_234: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_232, add_144);  mul_232 = add_144 = None
        convert_element_type_732: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_234, torch.bfloat16);  mul_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_733: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_294, torch.bfloat16);  primals_294 = None
        convert_element_type_734: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_293, torch.bfloat16);  primals_293 = None
        view_394: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_732, [8192, 4096]);  convert_element_type_732 = None
        permute_197: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_734, [1, 0]);  convert_element_type_734 = None
        addmm_107: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_733, view_394, permute_197);  convert_element_type_733 = None
        view_395: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_107, [16, 512, 1024]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_36: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_12: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        convert_element_type_default_36: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_12, torch.bfloat16);  inductor_random_default_12 = None
        gt_54: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_36, 0.1);  convert_element_type_default_36 = None
        mul_235: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_54, view_395);  view_395 = None
        mul_236: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, 1.1111111111111112);  mul_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_145: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, mul_236);  add_141 = mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_36 = torch.ops.aten.var_mean.correction(add_145, [2], correction = 0, keepdim = True)
        getitem_72: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_146: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_55: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_145, getitem_73);  getitem_73 = None
        mul_237: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = None
        mul_238: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, primals_295)
        add_147: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_238, primals_296);  mul_238 = primals_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_738: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_298, torch.bfloat16);  primals_298 = None
        convert_element_type_739: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_297, torch.bfloat16);  primals_297 = None
        convert_element_type_740: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_147, torch.bfloat16);  add_147 = None
        view_396: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_740, [8192, 1024]);  convert_element_type_740 = None
        permute_198: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_739, [1, 0]);  convert_element_type_739 = None
        addmm_108: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_738, view_396, permute_198);  convert_element_type_738 = None
        view_397: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_108, [16, 512, 1024]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_398: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_397, [16, 512, -1, 64]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_744: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_300, torch.bfloat16);  primals_300 = None
        convert_element_type_745: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_299, torch.bfloat16);  primals_299 = None
        permute_200: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_745, [1, 0]);  convert_element_type_745 = None
        addmm_109: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_744, view_396, permute_200);  convert_element_type_744 = None
        view_400: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_109, [16, 512, 1024]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_401: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_400, [16, 512, -1, 64]);  view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_750: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_302, torch.bfloat16);  primals_302 = None
        convert_element_type_751: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_301, torch.bfloat16);  primals_301 = None
        permute_202: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_751, [1, 0]);  convert_element_type_751 = None
        addmm_110: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_750, view_396, permute_202);  convert_element_type_750 = None
        view_403: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_110, [16, 512, 1024]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_404: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_403, [16, 512, -1, 64]);  view_403 = None

        # No stacktrace found for following nodes
        permute_default_30: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        permute_default_31: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None
        permute_default_32: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        _scaled_dot_product_flash_attention_default_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_30, permute_default_31, permute_default_32, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_135: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_5[0]

        # No stacktrace found for following nodes
        getitem_136: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_5[1]
        getitem_137: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_5[6]
        getitem_138: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_5[7];  _scaled_dot_product_flash_attention_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_205: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3])
        clone_75: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_411: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [16, 512, 1024]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_761: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_304, torch.bfloat16);  primals_304 = None
        convert_element_type_762: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_303, torch.bfloat16);  primals_303 = None
        view_412: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_411, [8192, 1024]);  view_411 = None
        permute_206: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_762, [1, 0]);  convert_element_type_762 = None
        addmm_111: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_761, view_412, permute_206);  convert_element_type_761 = None
        view_413: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_111, [16, 512, 1024]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_37: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_11: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        convert_element_type_default_35: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.bfloat16);  inductor_random_default_11 = None
        gt_56: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_35, 0.1);  convert_element_type_default_35 = None
        mul_241: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_56, view_413);  view_413 = None
        mul_242: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, 1.1111111111111112);  mul_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_149: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, mul_242);  add_145 = mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_37 = torch.ops.aten.var_mean.correction(add_149, [2], correction = 0, keepdim = True)
        getitem_74: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_150: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_150);  add_150 = None
        sub_57: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_149, getitem_75);  getitem_75 = None
        mul_243: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_37);  sub_57 = None
        mul_244: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, primals_305)
        add_151: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_244, primals_306);  mul_244 = primals_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_766: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_308, torch.bfloat16);  primals_308 = None
        convert_element_type_767: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_307, torch.bfloat16);  primals_307 = None
        convert_element_type_768: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.bfloat16);  add_151 = None
        view_414: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_768, [8192, 1024]);  convert_element_type_768 = None
        permute_207: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_767, [1, 0]);  convert_element_type_767 = None
        addmm_112: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_766, view_414, permute_207);  convert_element_type_766 = None
        view_415: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_112, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_772: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_245: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_772, 0.5)
        mul_246: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_772, 0.7071067811865476);  convert_element_type_772 = None
        erf_18: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_246);  mul_246 = None
        add_152: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_247: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, add_152);  mul_245 = add_152 = None
        convert_element_type_773: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_247, torch.bfloat16);  mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_774: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_310, torch.bfloat16);  primals_310 = None
        convert_element_type_775: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_309, torch.bfloat16);  primals_309 = None
        view_416: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_773, [8192, 4096]);  convert_element_type_773 = None
        permute_208: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_775, [1, 0]);  convert_element_type_775 = None
        addmm_113: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_774, view_416, permute_208);  convert_element_type_774 = None
        view_417: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_113, [16, 512, 1024]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_38: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_10: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        convert_element_type_default_34: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.bfloat16);  inductor_random_default_10 = None
        gt_57: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_34, 0.1);  convert_element_type_default_34 = None
        mul_248: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_57, view_417);  view_417 = None
        mul_249: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, 1.1111111111111112);  mul_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_153: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_149, mul_249);  add_149 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_38 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
        getitem_76: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_154: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_58: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_153, getitem_77);  getitem_77 = None
        mul_250: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = None
        mul_251: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, primals_311)
        add_155: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_251, primals_312);  mul_251 = primals_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_779: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_314, torch.bfloat16);  primals_314 = None
        convert_element_type_780: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_313, torch.bfloat16);  primals_313 = None
        convert_element_type_781: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16);  add_155 = None
        view_418: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_781, [8192, 1024]);  convert_element_type_781 = None
        permute_209: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_780, [1, 0]);  convert_element_type_780 = None
        addmm_114: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_779, view_418, permute_209);  convert_element_type_779 = None
        view_419: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_114, [16, 512, 1024]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_420: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_419, [16, 512, -1, 64]);  view_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_785: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_316, torch.bfloat16);  primals_316 = None
        convert_element_type_786: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_315, torch.bfloat16);  primals_315 = None
        permute_211: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_786, [1, 0]);  convert_element_type_786 = None
        addmm_115: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_785, view_418, permute_211);  convert_element_type_785 = None
        view_422: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_115, [16, 512, 1024]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_423: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_422, [16, 512, -1, 64]);  view_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_791: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_318, torch.bfloat16);  primals_318 = None
        convert_element_type_792: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_317, torch.bfloat16);  primals_317 = None
        permute_213: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_792, [1, 0]);  convert_element_type_792 = None
        addmm_116: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_791, view_418, permute_213);  convert_element_type_791 = None
        view_425: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_116, [16, 512, 1024]);  addmm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_426: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_425, [16, 512, -1, 64]);  view_425 = None

        # No stacktrace found for following nodes
        permute_default_24: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        permute_default_25: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None
        permute_default_26: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
        _scaled_dot_product_flash_attention_default_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_24, permute_default_25, permute_default_26, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_128: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_4[0]

        # No stacktrace found for following nodes
        getitem_129: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_4[1]
        getitem_130: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_4[6]
        getitem_131: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_4[7];  _scaled_dot_product_flash_attention_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_216: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3])
        clone_79: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_433: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [16, 512, 1024]);  clone_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_802: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_320, torch.bfloat16);  primals_320 = None
        convert_element_type_803: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_319, torch.bfloat16);  primals_319 = None
        view_434: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_433, [8192, 1024]);  view_433 = None
        permute_217: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_803, [1, 0]);  convert_element_type_803 = None
        addmm_117: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_802, view_434, permute_217);  convert_element_type_802 = None
        view_435: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_117, [16, 512, 1024]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_39: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_9: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        convert_element_type_default_33: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.bfloat16);  inductor_random_default_9 = None
        gt_59: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_33, 0.1);  convert_element_type_default_33 = None
        mul_254: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_59, view_435);  view_435 = None
        mul_255: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_254, 1.1111111111111112);  mul_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_157: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_153, mul_255);  add_153 = mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_39 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
        getitem_78: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_158: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        sub_60: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_157, getitem_79);  getitem_79 = None
        mul_256: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_39);  sub_60 = None
        mul_257: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, primals_321)
        add_159: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_257, primals_322);  mul_257 = primals_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_807: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_324, torch.bfloat16);  primals_324 = None
        convert_element_type_808: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_323, torch.bfloat16);  primals_323 = None
        convert_element_type_809: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None
        view_436: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_809, [8192, 1024]);  convert_element_type_809 = None
        permute_218: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_808, [1, 0]);  convert_element_type_808 = None
        addmm_118: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_807, view_436, permute_218);  convert_element_type_807 = None
        view_437: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_118, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_813: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        mul_258: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_813, 0.5)
        mul_259: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_813, 0.7071067811865476);  convert_element_type_813 = None
        erf_19: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_259);  mul_259 = None
        add_160: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_260: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_258, add_160);  mul_258 = add_160 = None
        convert_element_type_814: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_260, torch.bfloat16);  mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_815: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_326, torch.bfloat16);  primals_326 = None
        convert_element_type_816: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_325, torch.bfloat16);  primals_325 = None
        view_438: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_814, [8192, 4096]);  convert_element_type_814 = None
        permute_219: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_816, [1, 0]);  convert_element_type_816 = None
        addmm_119: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_815, view_438, permute_219);  convert_element_type_815 = None
        view_439: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_119, [16, 512, 1024]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_40: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_8: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        convert_element_type_default_32: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_8, torch.bfloat16);  inductor_random_default_8 = None
        gt_60: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_32, 0.1);  convert_element_type_default_32 = None
        mul_261: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_60, view_439);  view_439 = None
        mul_262: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, 1.1111111111111112);  mul_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_161: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_157, mul_262);  add_157 = mul_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_40 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_80: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_162: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        sub_61: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_161, getitem_81);  getitem_81 = None
        mul_263: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = None
        mul_264: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, primals_327)
        add_163: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_264, primals_328);  mul_264 = primals_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_820: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_330, torch.bfloat16);  primals_330 = None
        convert_element_type_821: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_329, torch.bfloat16);  primals_329 = None
        convert_element_type_822: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_163, torch.bfloat16);  add_163 = None
        view_440: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_822, [8192, 1024]);  convert_element_type_822 = None
        permute_220: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_821, [1, 0]);  convert_element_type_821 = None
        addmm_120: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_820, view_440, permute_220);  convert_element_type_820 = None
        view_441: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_120, [16, 512, 1024]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_442: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [16, 512, -1, 64]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_826: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_332, torch.bfloat16);  primals_332 = None
        convert_element_type_827: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_331, torch.bfloat16);  primals_331 = None
        permute_222: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_827, [1, 0]);  convert_element_type_827 = None
        addmm_121: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_826, view_440, permute_222);  convert_element_type_826 = None
        view_444: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_121, [16, 512, 1024]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_445: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_444, [16, 512, -1, 64]);  view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_832: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_334, torch.bfloat16);  primals_334 = None
        convert_element_type_833: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_333, torch.bfloat16);  primals_333 = None
        permute_224: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_833, [1, 0]);  convert_element_type_833 = None
        addmm_122: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_832, view_440, permute_224);  convert_element_type_832 = None
        view_447: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_122, [16, 512, 1024]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_448: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_447, [16, 512, -1, 64]);  view_447 = None

        # No stacktrace found for following nodes
        permute_default_18: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
        permute_default_19: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
        permute_default_20: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
        _scaled_dot_product_flash_attention_default_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_18, permute_default_19, permute_default_20, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_121: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_3[0]

        # No stacktrace found for following nodes
        getitem_122: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_3[1]
        getitem_123: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_3[6]
        getitem_124: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_3[7];  _scaled_dot_product_flash_attention_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_227: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_121, [0, 2, 1, 3])
        clone_83: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_455: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [16, 512, 1024]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_843: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_336, torch.bfloat16);  primals_336 = None
        convert_element_type_844: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_335, torch.bfloat16);  primals_335 = None
        view_456: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_455, [8192, 1024]);  view_455 = None
        permute_228: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_844, [1, 0]);  convert_element_type_844 = None
        addmm_123: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_843, view_456, permute_228);  convert_element_type_843 = None
        view_457: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_123, [16, 512, 1024]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_41: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_7: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        convert_element_type_default_31: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.bfloat16);  inductor_random_default_7 = None
        gt_62: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_31, 0.1);  convert_element_type_default_31 = None
        mul_267: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_62, view_457);  view_457 = None
        mul_268: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_267, 1.1111111111111112);  mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_165: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_161, mul_268);  add_161 = mul_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_41 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_82: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_166: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_63: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_165, getitem_83);  getitem_83 = None
        mul_269: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_41);  sub_63 = None
        mul_270: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, primals_337)
        add_167: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_270, primals_338);  mul_270 = primals_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_848: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_340, torch.bfloat16);  primals_340 = None
        convert_element_type_849: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_339, torch.bfloat16);  primals_339 = None
        convert_element_type_850: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.bfloat16);  add_167 = None
        view_458: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_850, [8192, 1024]);  convert_element_type_850 = None
        permute_229: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_849, [1, 0]);  convert_element_type_849 = None
        addmm_124: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_848, view_458, permute_229);  convert_element_type_848 = None
        view_459: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_124, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_854: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        mul_271: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_854, 0.5)
        mul_272: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_854, 0.7071067811865476);  convert_element_type_854 = None
        erf_20: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_272);  mul_272 = None
        add_168: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_273: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_271, add_168);  mul_271 = add_168 = None
        convert_element_type_855: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16);  mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_856: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_342, torch.bfloat16);  primals_342 = None
        convert_element_type_857: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_341, torch.bfloat16);  primals_341 = None
        view_460: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_855, [8192, 4096]);  convert_element_type_855 = None
        permute_230: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_857, [1, 0]);  convert_element_type_857 = None
        addmm_125: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_856, view_460, permute_230);  convert_element_type_856 = None
        view_461: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_125, [16, 512, 1024]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_42: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_6: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        convert_element_type_default_30: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.bfloat16);  inductor_random_default_6 = None
        gt_63: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_30, 0.1);  convert_element_type_default_30 = None
        mul_274: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_63, view_461);  view_461 = None
        mul_275: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_274, 1.1111111111111112);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_169: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_165, mul_275);  add_165 = mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_42 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_84: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_170: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_64: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_169, getitem_85);  getitem_85 = None
        mul_276: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = None
        mul_277: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, primals_343)
        add_171: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_277, primals_344);  mul_277 = primals_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_861: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_346, torch.bfloat16);  primals_346 = None
        convert_element_type_862: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_345, torch.bfloat16);  primals_345 = None
        convert_element_type_863: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16);  add_171 = None
        view_462: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_863, [8192, 1024]);  convert_element_type_863 = None
        permute_231: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_862, [1, 0]);  convert_element_type_862 = None
        addmm_126: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_861, view_462, permute_231);  convert_element_type_861 = None
        view_463: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_126, [16, 512, 1024]);  addmm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_464: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_463, [16, 512, -1, 64]);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_867: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_348, torch.bfloat16);  primals_348 = None
        convert_element_type_868: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_347, torch.bfloat16);  primals_347 = None
        permute_233: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_868, [1, 0]);  convert_element_type_868 = None
        addmm_127: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_867, view_462, permute_233);  convert_element_type_867 = None
        view_466: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_127, [16, 512, 1024]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_467: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_466, [16, 512, -1, 64]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_873: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_350, torch.bfloat16);  primals_350 = None
        convert_element_type_874: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_349, torch.bfloat16);  primals_349 = None
        permute_235: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_874, [1, 0]);  convert_element_type_874 = None
        addmm_128: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_873, view_462, permute_235);  convert_element_type_873 = None
        view_469: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_128, [16, 512, 1024]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_470: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_469, [16, 512, -1, 64]);  view_469 = None

        # No stacktrace found for following nodes
        permute_default_12: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None
        permute_default_13: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
        permute_default_14: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
        _scaled_dot_product_flash_attention_default_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_12, permute_default_13, permute_default_14, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_114: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_2[0]

        # No stacktrace found for following nodes
        getitem_115: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_2[1]
        getitem_116: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_2[6]
        getitem_117: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_2[7];  _scaled_dot_product_flash_attention_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_238: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_114, [0, 2, 1, 3])
        clone_87: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_477: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [16, 512, 1024]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_884: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_352, torch.bfloat16);  primals_352 = None
        convert_element_type_885: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_351, torch.bfloat16);  primals_351 = None
        view_478: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_477, [8192, 1024]);  view_477 = None
        permute_239: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_885, [1, 0]);  convert_element_type_885 = None
        addmm_129: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_884, view_478, permute_239);  convert_element_type_884 = None
        view_479: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_129, [16, 512, 1024]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_43: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_5: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        convert_element_type_default_29: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.bfloat16);  inductor_random_default_5 = None
        gt_65: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_29, 0.1);  convert_element_type_default_29 = None
        mul_280: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_65, view_479);  view_479 = None
        mul_281: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, 1.1111111111111112);  mul_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_173: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_169, mul_281);  add_169 = mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_43 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_86: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_174: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_66: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_173, getitem_87);  getitem_87 = None
        mul_282: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_43);  sub_66 = None
        mul_283: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, primals_353)
        add_175: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_283, primals_354);  mul_283 = primals_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_889: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_356, torch.bfloat16);  primals_356 = None
        convert_element_type_890: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_355, torch.bfloat16);  primals_355 = None
        convert_element_type_891: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16);  add_175 = None
        view_480: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_891, [8192, 1024]);  convert_element_type_891 = None
        permute_240: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_890, [1, 0]);  convert_element_type_890 = None
        addmm_130: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_889, view_480, permute_240);  convert_element_type_889 = None
        view_481: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_130, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_895: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_481, torch.float32);  view_481 = None
        mul_284: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_895, 0.5)
        mul_285: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_895, 0.7071067811865476);  convert_element_type_895 = None
        erf_21: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_285);  mul_285 = None
        add_176: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_286: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_284, add_176);  mul_284 = add_176 = None
        convert_element_type_896: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_286, torch.bfloat16);  mul_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_897: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_358, torch.bfloat16);  primals_358 = None
        convert_element_type_898: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_357, torch.bfloat16);  primals_357 = None
        view_482: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_896, [8192, 4096]);  convert_element_type_896 = None
        permute_241: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_898, [1, 0]);  convert_element_type_898 = None
        addmm_131: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_897, view_482, permute_241);  convert_element_type_897 = None
        view_483: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_131, [16, 512, 1024]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_44: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_4: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        convert_element_type_default_28: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_4, torch.bfloat16);  inductor_random_default_4 = None
        gt_66: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_28, 0.1);  convert_element_type_default_28 = None
        mul_287: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_66, view_483);  view_483 = None
        mul_288: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, 1.1111111111111112);  mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_177: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_173, mul_288);  add_173 = mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_44 = torch.ops.aten.var_mean.correction(add_177, [2], correction = 0, keepdim = True)
        getitem_88: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_178: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
        sub_67: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_177, getitem_89);  getitem_89 = None
        mul_289: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = None
        mul_290: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, primals_359)
        add_179: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_290, primals_360);  mul_290 = primals_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_902: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_362, torch.bfloat16);  primals_362 = None
        convert_element_type_903: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_361, torch.bfloat16);  primals_361 = None
        convert_element_type_904: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None
        view_484: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_904, [8192, 1024]);  convert_element_type_904 = None
        permute_242: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_903, [1, 0]);  convert_element_type_903 = None
        addmm_132: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_902, view_484, permute_242);  convert_element_type_902 = None
        view_485: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_132, [16, 512, 1024]);  addmm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_486: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_485, [16, 512, -1, 64]);  view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_908: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_364, torch.bfloat16);  primals_364 = None
        convert_element_type_909: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_363, torch.bfloat16);  primals_363 = None
        permute_244: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_909, [1, 0]);  convert_element_type_909 = None
        addmm_133: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_908, view_484, permute_244);  convert_element_type_908 = None
        view_488: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_133, [16, 512, 1024]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_489: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_488, [16, 512, -1, 64]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_914: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_366, torch.bfloat16);  primals_366 = None
        convert_element_type_915: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_365, torch.bfloat16);  primals_365 = None
        permute_246: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_915, [1, 0]);  convert_element_type_915 = None
        addmm_134: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_914, view_484, permute_246);  convert_element_type_914 = None
        view_491: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_134, [16, 512, 1024]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_492: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_491, [16, 512, -1, 64]);  view_491 = None

        # No stacktrace found for following nodes
        permute_default_6: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        permute_default_7: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_489, [0, 2, 1, 3]);  view_489 = None
        permute_default_8: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
        _scaled_dot_product_flash_attention_default_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default_6, permute_default_7, permute_default_8, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_107: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default_1[0]

        # No stacktrace found for following nodes
        getitem_108: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default_1[1]
        getitem_109: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default_1[6]
        getitem_110: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default_1[7];  _scaled_dot_product_flash_attention_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_249: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_107, [0, 2, 1, 3])
        clone_91: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_499: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [16, 512, 1024]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_925: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_368, torch.bfloat16);  primals_368 = None
        convert_element_type_926: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_367, torch.bfloat16);  primals_367 = None
        view_500: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_499, [8192, 1024]);  view_499 = None
        permute_250: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_926, [1, 0]);  convert_element_type_926 = None
        addmm_135: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_925, view_500, permute_250);  convert_element_type_925 = None
        view_501: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_135, [16, 512, 1024]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_45: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_3: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        convert_element_type_default_27: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.bfloat16);  inductor_random_default_3 = None
        gt_68: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_27, 0.1);  convert_element_type_default_27 = None
        mul_293: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_68, view_501);  view_501 = None
        mul_294: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, 1.1111111111111112);  mul_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_181: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_177, mul_294);  add_177 = mul_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_45 = torch.ops.aten.var_mean.correction(add_181, [2], correction = 0, keepdim = True)
        getitem_90: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_182: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
        sub_69: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_181, getitem_91);  getitem_91 = None
        mul_295: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_45);  sub_69 = None
        mul_296: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, primals_369)
        add_183: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_296, primals_370);  mul_296 = primals_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_930: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_372, torch.bfloat16);  primals_372 = None
        convert_element_type_931: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_371, torch.bfloat16);  primals_371 = None
        convert_element_type_932: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_183, torch.bfloat16);  add_183 = None
        view_502: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_932, [8192, 1024]);  convert_element_type_932 = None
        permute_251: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_931, [1, 0]);  convert_element_type_931 = None
        addmm_136: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_930, view_502, permute_251);  convert_element_type_930 = None
        view_503: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_136, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_936: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        mul_297: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_936, 0.5)
        mul_298: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_936, 0.7071067811865476);  convert_element_type_936 = None
        erf_22: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_298);  mul_298 = None
        add_184: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_299: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, add_184);  mul_297 = add_184 = None
        convert_element_type_937: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_299, torch.bfloat16);  mul_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_938: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_374, torch.bfloat16);  primals_374 = None
        convert_element_type_939: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_373, torch.bfloat16);  primals_373 = None
        view_504: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_937, [8192, 4096]);  convert_element_type_937 = None
        permute_252: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_939, [1, 0]);  convert_element_type_939 = None
        addmm_137: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_938, view_504, permute_252);  convert_element_type_938 = None
        view_505: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_137, [16, 512, 1024]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_46: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_2: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        convert_element_type_default_26: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.bfloat16);  inductor_random_default_2 = None
        gt_69: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_26, 0.1);  convert_element_type_default_26 = None
        mul_300: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_69, view_505);  view_505 = None
        mul_301: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, 1.1111111111111112);  mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_185: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_181, mul_301);  add_181 = mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        var_mean_46 = torch.ops.aten.var_mean.correction(add_185, [2], correction = 0, keepdim = True)
        getitem_92: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_46[0]
        getitem_93: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_186: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_70: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_185, getitem_93);  getitem_93 = None
        mul_302: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = None
        mul_303: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, primals_375)
        add_187: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_303, primals_376);  mul_303 = primals_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        convert_element_type_943: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_378, torch.bfloat16);  primals_378 = None
        convert_element_type_944: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_377, torch.bfloat16);  primals_377 = None
        convert_element_type_945: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_187, torch.bfloat16);  add_187 = None
        view_506: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_945, [8192, 1024]);  convert_element_type_945 = None
        permute_253: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_944, [1, 0]);  convert_element_type_944 = None
        addmm_138: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_943, view_506, permute_253);  convert_element_type_943 = None
        view_507: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_138, [16, 512, 1024]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        view_508: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_507, [16, 512, -1, 64]);  view_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        convert_element_type_949: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_380, torch.bfloat16);  primals_380 = None
        convert_element_type_950: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_379, torch.bfloat16);  primals_379 = None
        permute_255: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_950, [1, 0]);  convert_element_type_950 = None
        addmm_139: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_949, view_506, permute_255);  convert_element_type_949 = None
        view_510: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_139, [16, 512, 1024]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_511: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_510, [16, 512, -1, 64]);  view_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        convert_element_type_955: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_382, torch.bfloat16);  primals_382 = None
        convert_element_type_956: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_381, torch.bfloat16);  primals_381 = None
        permute_257: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_956, [1, 0]);  convert_element_type_956 = None
        addmm_140: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_955, view_506, permute_257);  convert_element_type_955 = None
        view_513: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_140, [16, 512, 1024]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        view_514: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_513, [16, 512, -1, 64]);  view_513 = None

        # No stacktrace found for following nodes
        permute_default: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        permute_default_1: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
        permute_default_2: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
        _scaled_dot_product_flash_attention_default = torch.ops.aten._scaled_dot_product_flash_attention.default(permute_default, permute_default_1, permute_default_2, 0.1, scale = 0.125)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        getitem_100: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_default[0]

        # No stacktrace found for following nodes
        getitem_101: "f32[16, 16, 512][8192, 512, 1]cuda:0" = _scaled_dot_product_flash_attention_default[1]
        getitem_102: "u64[2][1]cuda:0" = _scaled_dot_product_flash_attention_default[6]
        getitem_103: "u64[][]cuda:0" = _scaled_dot_product_flash_attention_default[7];  _scaled_dot_product_flash_attention_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_260: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_100, [0, 2, 1, 3])
        clone_95: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_521: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [16, 512, 1024]);  clone_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_966: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_384, torch.bfloat16);  primals_384 = None
        convert_element_type_967: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_383, torch.bfloat16);  primals_383 = None
        view_522: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_521, [8192, 1024]);  view_521 = None
        permute_261: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_967, [1, 0]);  convert_element_type_967 = None
        addmm_141: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_966, view_522, permute_261);  convert_element_type_966 = None
        view_523: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_141, [16, 512, 1024]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_47: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_1: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        convert_element_type_default_25: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_71: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_25, 0.1);  convert_element_type_default_25 = None
        mul_306: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_71, view_523);  view_523 = None
        mul_307: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, 1.1111111111111112);  mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        add_189: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_185, mul_307);  add_185 = mul_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        var_mean_47 = torch.ops.aten.var_mean.correction(add_189, [2], correction = 0, keepdim = True)
        getitem_94: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_47[0]
        getitem_95: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_190: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        sub_72: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_189, getitem_95);  getitem_95 = None
        mul_308: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_47);  sub_72 = None
        mul_309: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, primals_385)
        add_191: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_309, primals_386);  mul_309 = primals_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_971: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_388, torch.bfloat16);  primals_388 = None
        convert_element_type_972: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_387, torch.bfloat16);  primals_387 = None
        convert_element_type_973: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16);  add_191 = None
        view_524: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_973, [8192, 1024]);  convert_element_type_973 = None
        permute_262: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_972, [1, 0]);  convert_element_type_972 = None
        addmm_142: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_971, view_524, permute_262);  convert_element_type_971 = None
        view_525: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_142, [16, 512, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_977: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_525, torch.float32);  view_525 = None
        mul_310: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_977, 0.5)
        mul_311: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_977, 0.7071067811865476);  convert_element_type_977 = None
        erf_23: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_311);  mul_311 = None
        add_192: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_312: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_310, add_192);  mul_310 = add_192 = None
        convert_element_type_978: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_979: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_390, torch.bfloat16);  primals_390 = None
        convert_element_type_980: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_389, torch.bfloat16);  primals_389 = None
        view_526: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_978, [8192, 4096]);  convert_element_type_978 = None
        permute_263: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_980, [1, 0]);  convert_element_type_980 = None
        addmm_143: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_979, view_526, permute_263);  convert_element_type_979 = None
        view_527: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_143, [16, 512, 1024]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        inductor_lookup_seed_default_48: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48);  inductor_seeds_default = None
        inductor_random_default: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 512, 1024], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        convert_element_type_default_24: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default, torch.bfloat16);  inductor_random_default = None
        gt_72: "b8[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_24, 0.1);  convert_element_type_default_24 = None
        mul_313: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_72, view_527);  view_527 = None
        mul_314: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_313, 1.1111111111111112);  mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        add_193: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_189, mul_314);  add_189 = mul_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        var_mean_48 = torch.ops.aten.var_mean.correction(add_193, [2], correction = 0, keepdim = True)
        getitem_96: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_48[0]
        getitem_97: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_194: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-12);  getitem_96 = None
        rsqrt_48: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
        sub_73: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_193, getitem_97);  add_193 = getitem_97 = None
        mul_315: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_48);  sub_73 = None
        mul_316: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, primals_391)
        add_195: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_316, primals_392);  mul_316 = primals_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        convert_element_type_984: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_394, torch.bfloat16);  primals_394 = None
        convert_element_type_985: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_393, torch.bfloat16);  primals_393 = None
        convert_element_type_986: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_195, torch.bfloat16);  add_195 = None
        view_528: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_986, [8192, 1024]);  convert_element_type_986 = None
        permute_264: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_985, [1, 0]);  convert_element_type_985 = None
        addmm_144: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_984, view_528, permute_264);  convert_element_type_984 = None
        view_529: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_144, [16, 512, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_990: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        mul_317: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_990, 0.5)
        mul_318: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_990, 0.7071067811865476);  convert_element_type_990 = None
        erf_24: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_318);  mul_318 = None
        add_196: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_319: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_317, add_196);  mul_317 = add_196 = None
        convert_element_type_991: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_319, torch.bfloat16);  mul_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_992: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_991, torch.float32);  convert_element_type_991 = None
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_992, [2], correction = 0, keepdim = True)
        getitem_98: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_49[0]
        getitem_99: "f32[16, 512, 1][512, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_197: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-12);  getitem_98 = None
        rsqrt_49: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_74: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_992, getitem_99);  convert_element_type_992 = None
        mul_320: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = None
        mul_321: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_320, primals_395);  mul_320 = None
        add_198: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_321, primals_396);  mul_321 = primals_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        convert_element_type_993: "bf16[29056][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_397, torch.bfloat16);  primals_397 = None
        convert_element_type_994: "bf16[29056, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        convert_element_type_995: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16);  add_198 = None
        view_530: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_995, [8192, 1024]);  convert_element_type_995 = None
        permute_265: "bf16[1024, 29056][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_994, [1, 0]);  convert_element_type_994 = None
        addmm_145: "bf16[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_993, view_530, permute_265);  convert_element_type_993 = None
        view_531: "bf16[16, 512, 29056][14876672, 29056, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_145, [16, 512, 29056]);  addmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_999: "f32[16, 512, 29056][14876672, 29056, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_531, torch.float32)

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "i64[16, 513][513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(primals_1, [0, 1], -100.0);  primals_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[16, 512][513, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807)
        clone_96: "i64[16, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_532: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_999, [-1, 29056]);  convert_element_type_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_533: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [-1]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        amax_24: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(view_532, [1], True)
        sub_75: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = None
        exp_24: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.exp.default(sub_75)
        sum_25: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_76: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = None
        ne: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_533, -100)
        full_default_2: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne, view_533, full_default_2);  view_533 = full_default_2 = None
        unsqueeze_2: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(sub_76, 1, unsqueeze_2);  sub_76 = unsqueeze_2 = None
        squeeze: "f32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_3: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_3);  neg = full_default_3 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_1000: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_48: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_1000);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        permute_266: "bf16[29056, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        permute_270: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        div_51: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_48, 1024);  rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_274: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_278: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_52: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_282: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_293: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_298: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_303: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_54: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_307: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_311: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_55: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_315: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_326: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_331: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_336: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_57: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_340: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_344: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_58: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_348: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_359: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_364: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_369: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_60: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_373: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_377: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_61: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_381: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_392: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_397: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_402: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_63: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_406: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_410: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_64: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_414: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_425: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_430: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_435: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_66: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_439: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_443: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_67: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_447: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_458: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_463: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_468: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_69: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_472: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_476: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_70: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_480: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_491: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_496: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_501: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_72: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_505: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_509: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_73: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_513: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_524: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_529: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_534: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_75: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_538: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_542: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_76: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_546: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_557: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_562: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_567: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_78: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_571: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_575: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_79: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_579: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_590: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_595: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_600: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_81: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_604: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_608: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_82: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_612: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_623: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_628: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_633: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_84: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_637: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_641: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_85: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_645: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_656: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_661: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_666: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_87: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_670: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_674: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_88: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_678: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_689: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_694: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_699: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_90: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_703: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_707: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_91: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_711: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_722: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_727: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_732: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_93: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_736: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_740: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_94: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_744: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_755: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_760: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_765: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_96: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_769: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_773: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_97: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_777: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_788: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_793: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_798: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_99: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_802: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_806: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_100: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_810: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_821: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_826: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_831: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_102: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_835: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_839: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_103: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_843: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_854: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_859: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_864: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_105: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_868: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_872: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_106: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_876: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_887: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_892: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_897: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_108: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_901: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_905: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_109: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_909: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_920: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_925: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_930: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_111: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_934: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_938: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_112: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_942: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_953: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_958: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_963: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_114: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_967: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_971: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_115: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_975: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_986: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_991: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_996: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_117: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_1000: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_1004: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_118: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_1008: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_1019: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_1024: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_1029: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_120: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        permute_1033: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        permute_1037: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        div_121: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        permute_1041: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        permute_1052: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        permute_1057: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        permute_1062: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        div_123: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None
        return (div_48, view_531, primals_2, primals_4, primals_7, primals_17, primals_23, primals_33, primals_39, primals_49, primals_55, primals_65, primals_71, primals_81, primals_87, primals_97, primals_103, primals_113, primals_119, primals_129, primals_135, primals_145, primals_151, primals_161, primals_167, primals_177, primals_183, primals_193, primals_199, primals_209, primals_215, primals_225, primals_231, primals_241, primals_247, primals_257, primals_263, primals_273, primals_279, primals_289, primals_295, primals_305, primals_311, primals_321, primals_327, primals_337, primals_343, primals_353, primals_359, primals_369, primals_375, primals_385, primals_391, primals_395, full_default, gt, mul_3, view, permute_default_138, permute_default_139, permute_default_140, getitem_261, getitem_262, getitem_263, getitem_264, view_16, gt_2, mul_9, view_18, addmm_4, view_20, gt_3, mul_16, view_22, permute_default_132, permute_default_133, permute_default_134, getitem_254, getitem_255, getitem_256, getitem_257, view_38, gt_5, mul_22, view_40, addmm_10, view_42, gt_6, mul_29, view_44, permute_default_126, permute_default_127, permute_default_128, getitem_247, getitem_248, getitem_249, getitem_250, view_60, gt_8, mul_35, view_62, addmm_16, view_64, gt_9, mul_42, view_66, permute_default_120, permute_default_121, permute_default_122, getitem_240, getitem_241, getitem_242, getitem_243, view_82, gt_11, mul_48, view_84, addmm_22, view_86, gt_12, mul_55, view_88, permute_default_114, permute_default_115, permute_default_116, getitem_233, getitem_234, getitem_235, getitem_236, view_104, gt_14, mul_61, view_106, addmm_28, view_108, gt_15, mul_68, view_110, permute_default_108, permute_default_109, permute_default_110, getitem_226, getitem_227, getitem_228, getitem_229, view_126, gt_17, mul_74, view_128, addmm_34, view_130, gt_18, mul_81, view_132, permute_default_102, permute_default_103, permute_default_104, getitem_219, getitem_220, getitem_221, getitem_222, view_148, gt_20, mul_87, view_150, addmm_40, view_152, gt_21, mul_94, view_154, permute_default_96, permute_default_97, permute_default_98, getitem_212, getitem_213, getitem_214, getitem_215, view_170, gt_23, mul_100, view_172, addmm_46, view_174, gt_24, mul_107, view_176, permute_default_90, permute_default_91, permute_default_92, getitem_205, getitem_206, getitem_207, getitem_208, view_192, gt_26, mul_113, view_194, addmm_52, view_196, gt_27, mul_120, view_198, permute_default_84, permute_default_85, permute_default_86, getitem_198, getitem_199, getitem_200, getitem_201, view_214, gt_29, mul_126, view_216, addmm_58, view_218, gt_30, mul_133, view_220, permute_default_78, permute_default_79, permute_default_80, getitem_191, getitem_192, getitem_193, getitem_194, view_236, gt_32, mul_139, view_238, addmm_64, view_240, gt_33, mul_146, view_242, permute_default_72, permute_default_73, permute_default_74, getitem_184, getitem_185, getitem_186, getitem_187, view_258, gt_35, mul_152, view_260, addmm_70, view_262, gt_36, mul_159, view_264, permute_default_66, permute_default_67, permute_default_68, getitem_177, getitem_178, getitem_179, getitem_180, view_280, gt_38, mul_165, view_282, addmm_76, view_284, gt_39, mul_172, view_286, permute_default_60, permute_default_61, permute_default_62, getitem_170, getitem_171, getitem_172, getitem_173, view_302, gt_41, mul_178, view_304, addmm_82, view_306, gt_42, mul_185, view_308, permute_default_54, permute_default_55, permute_default_56, getitem_163, getitem_164, getitem_165, getitem_166, view_324, gt_44, mul_191, view_326, addmm_88, view_328, gt_45, mul_198, view_330, permute_default_48, permute_default_49, permute_default_50, getitem_156, getitem_157, getitem_158, getitem_159, view_346, gt_47, mul_204, view_348, addmm_94, view_350, gt_48, mul_211, view_352, permute_default_42, permute_default_43, permute_default_44, getitem_149, getitem_150, getitem_151, getitem_152, view_368, gt_50, mul_217, view_370, addmm_100, view_372, gt_51, mul_224, view_374, permute_default_36, permute_default_37, permute_default_38, getitem_142, getitem_143, getitem_144, getitem_145, view_390, gt_53, mul_230, view_392, addmm_106, view_394, gt_54, mul_237, view_396, permute_default_30, permute_default_31, permute_default_32, getitem_135, getitem_136, getitem_137, getitem_138, view_412, gt_56, mul_243, view_414, addmm_112, view_416, gt_57, mul_250, view_418, permute_default_24, permute_default_25, permute_default_26, getitem_128, getitem_129, getitem_130, getitem_131, view_434, gt_59, mul_256, view_436, addmm_118, view_438, gt_60, mul_263, view_440, permute_default_18, permute_default_19, permute_default_20, getitem_121, getitem_122, getitem_123, getitem_124, view_456, gt_62, mul_269, view_458, addmm_124, view_460, gt_63, mul_276, view_462, permute_default_12, permute_default_13, permute_default_14, getitem_114, getitem_115, getitem_116, getitem_117, view_478, gt_65, mul_282, view_480, addmm_130, view_482, gt_66, mul_289, view_484, permute_default_6, permute_default_7, permute_default_8, getitem_107, getitem_108, getitem_109, getitem_110, view_500, gt_68, mul_295, view_502, addmm_136, view_504, gt_69, mul_302, view_506, permute_default, permute_default_1, permute_default_2, getitem_100, getitem_101, getitem_102, getitem_103, view_522, gt_71, mul_308, view_524, addmm_142, view_526, gt_72, mul_315, view_528, addmm_144, getitem_99, rsqrt_49, view_530, view_531, constant_pad_nd, amax_24, log, convert_element_type_1000, permute_266, permute_270, div_51, permute_274, permute_278, div_52, permute_282, permute_293, permute_298, permute_303, div_54, permute_307, permute_311, div_55, permute_315, permute_326, permute_331, permute_336, div_57, permute_340, permute_344, div_58, permute_348, permute_359, permute_364, permute_369, div_60, permute_373, permute_377, div_61, permute_381, permute_392, permute_397, permute_402, div_63, permute_406, permute_410, div_64, permute_414, permute_425, permute_430, permute_435, div_66, permute_439, permute_443, div_67, permute_447, permute_458, permute_463, permute_468, div_69, permute_472, permute_476, div_70, permute_480, permute_491, permute_496, permute_501, div_72, permute_505, permute_509, div_73, permute_513, permute_524, permute_529, permute_534, div_75, permute_538, permute_542, div_76, permute_546, permute_557, permute_562, permute_567, div_78, permute_571, permute_575, div_79, permute_579, permute_590, permute_595, permute_600, div_81, permute_604, permute_608, div_82, permute_612, permute_623, permute_628, permute_633, div_84, permute_637, permute_641, div_85, permute_645, permute_656, permute_661, permute_666, div_87, permute_670, permute_674, div_88, permute_678, permute_689, permute_694, permute_699, div_90, permute_703, permute_707, div_91, permute_711, permute_722, permute_727, permute_732, div_93, permute_736, permute_740, div_94, permute_744, permute_755, permute_760, permute_765, div_96, permute_769, permute_773, div_97, permute_777, permute_788, permute_793, permute_798, div_99, permute_802, permute_806, div_100, permute_810, permute_821, permute_826, permute_831, div_102, permute_835, permute_839, div_103, permute_843, permute_854, permute_859, permute_864, div_105, permute_868, permute_872, div_106, permute_876, permute_887, permute_892, permute_897, div_108, permute_901, permute_905, div_109, permute_909, permute_920, permute_925, permute_930, div_111, permute_934, permute_938, div_112, permute_942, permute_953, permute_958, permute_963, div_114, permute_967, permute_971, div_115, permute_975, permute_986, permute_991, permute_996, div_117, permute_1000, permute_1004, div_118, permute_1008, permute_1019, permute_1024, permute_1029, div_120, permute_1033, permute_1037, div_121, permute_1041, permute_1052, permute_1057, permute_1062, div_123)
