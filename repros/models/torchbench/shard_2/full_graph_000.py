class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0", primals_2: "Sym(s28)", primals_3: "f32[s28, 3, 224, 224][150528, 50176, 224, 1]cuda:0", primals_4: "i64[][]cuda:0", primals_5: "f32[64][1]cuda:0", primals_6: "f32[64][1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_8: "f32[64][1]cuda:0", primals_9: "i64[][]cuda:0", primals_10: "f32[64][1]cuda:0", primals_11: "f32[64][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_13: "f32[64][1]cuda:0", primals_14: "f32[128, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_15: "i64[][]cuda:0", primals_16: "f32[128][1]cuda:0", primals_17: "f32[128][1]cuda:0", primals_18: "f32[128][1]cuda:0", primals_19: "f32[128][1]cuda:0", primals_20: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_21: "i64[][]cuda:0", primals_22: "f32[96][1]cuda:0", primals_23: "f32[96][1]cuda:0", primals_24: "f32[96][1]cuda:0", primals_25: "f32[96][1]cuda:0", primals_26: "f32[128, 96, 1, 1][96, 1, 1, 1]cuda:0", primals_27: "i64[][]cuda:0", primals_28: "f32[128][1]cuda:0", primals_29: "f32[128][1]cuda:0", primals_30: "f32[128][1]cuda:0", primals_31: "f32[128][1]cuda:0", primals_32: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_33: "i64[][]cuda:0", primals_34: "f32[128][1]cuda:0", primals_35: "f32[128][1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_37: "f32[128][1]cuda:0", primals_38: "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_39: "i64[][]cuda:0", primals_40: "f32[128][1]cuda:0", primals_41: "f32[128][1]cuda:0", primals_42: "f32[128][1]cuda:0", primals_43: "f32[128][1]cuda:0", primals_44: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_45: "i64[][]cuda:0", primals_46: "f32[160][1]cuda:0", primals_47: "f32[160][1]cuda:0", primals_48: "f32[160][1]cuda:0", primals_49: "f32[160][1]cuda:0", primals_50: "f32[128, 160, 1, 1][160, 1, 1, 1]cuda:0", primals_51: "i64[][]cuda:0", primals_52: "f32[128][1]cuda:0", primals_53: "f32[128][1]cuda:0", primals_54: "f32[128][1]cuda:0", primals_55: "f32[128][1]cuda:0", primals_56: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_57: "i64[][]cuda:0", primals_58: "f32[192][1]cuda:0", primals_59: "f32[192][1]cuda:0", primals_60: "f32[192][1]cuda:0", primals_61: "f32[192][1]cuda:0", primals_62: "f32[128, 192, 1, 1][192, 1, 1, 1]cuda:0", primals_63: "i64[][]cuda:0", primals_64: "f32[128][1]cuda:0", primals_65: "f32[128][1]cuda:0", primals_66: "f32[128][1]cuda:0", primals_67: "f32[128][1]cuda:0", primals_68: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_69: "i64[][]cuda:0", primals_70: "f32[224][1]cuda:0", primals_71: "f32[224][1]cuda:0", primals_72: "f32[224][1]cuda:0", primals_73: "f32[224][1]cuda:0", primals_74: "f32[128, 224, 1, 1][224, 1, 1, 1]cuda:0", primals_75: "i64[][]cuda:0", primals_76: "f32[128][1]cuda:0", primals_77: "f32[128][1]cuda:0", primals_78: "f32[128][1]cuda:0", primals_79: "f32[128][1]cuda:0", primals_80: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_81: "i64[][]cuda:0", primals_82: "f32[256][1]cuda:0", primals_83: "f32[256][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_85: "f32[256][1]cuda:0", primals_86: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_87: "i64[][]cuda:0", primals_88: "f32[128][1]cuda:0", primals_89: "f32[128][1]cuda:0", primals_90: "f32[128][1]cuda:0", primals_91: "f32[128][1]cuda:0", primals_92: "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_93: "i64[][]cuda:0", primals_94: "f32[128][1]cuda:0", primals_95: "f32[128][1]cuda:0", primals_96: "f32[128][1]cuda:0", primals_97: "f32[128][1]cuda:0", primals_98: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_99: "i64[][]cuda:0", primals_100: "f32[160][1]cuda:0", primals_101: "f32[160][1]cuda:0", primals_102: "f32[160][1]cuda:0", primals_103: "f32[160][1]cuda:0", primals_104: "f32[128, 160, 1, 1][160, 1, 1, 1]cuda:0", primals_105: "i64[][]cuda:0", primals_106: "f32[128][1]cuda:0", primals_107: "f32[128][1]cuda:0", primals_108: "f32[128][1]cuda:0", primals_109: "f32[128][1]cuda:0", primals_110: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_111: "i64[][]cuda:0", primals_112: "f32[192][1]cuda:0", primals_113: "f32[192][1]cuda:0", primals_114: "f32[192][1]cuda:0", primals_115: "f32[192][1]cuda:0", primals_116: "f32[128, 192, 1, 1][192, 1, 1, 1]cuda:0", primals_117: "i64[][]cuda:0", primals_118: "f32[128][1]cuda:0", primals_119: "f32[128][1]cuda:0", primals_120: "f32[128][1]cuda:0", primals_121: "f32[128][1]cuda:0", primals_122: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_123: "i64[][]cuda:0", primals_124: "f32[224][1]cuda:0", primals_125: "f32[224][1]cuda:0", primals_126: "f32[224][1]cuda:0", primals_127: "f32[224][1]cuda:0", primals_128: "f32[128, 224, 1, 1][224, 1, 1, 1]cuda:0", primals_129: "i64[][]cuda:0", primals_130: "f32[128][1]cuda:0", primals_131: "f32[128][1]cuda:0", primals_132: "f32[128][1]cuda:0", primals_133: "f32[128][1]cuda:0", primals_134: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_135: "i64[][]cuda:0", primals_136: "f32[256][1]cuda:0", primals_137: "f32[256][1]cuda:0", primals_138: "f32[256][1]cuda:0", primals_139: "f32[256][1]cuda:0", primals_140: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_141: "i64[][]cuda:0", primals_142: "f32[128][1]cuda:0", primals_143: "f32[128][1]cuda:0", primals_144: "f32[128][1]cuda:0", primals_145: "f32[128][1]cuda:0", primals_146: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_147: "i64[][]cuda:0", primals_148: "f32[288][1]cuda:0", primals_149: "f32[288][1]cuda:0", primals_150: "f32[288][1]cuda:0", primals_151: "f32[288][1]cuda:0", primals_152: "f32[128, 288, 1, 1][288, 1, 1, 1]cuda:0", primals_153: "i64[][]cuda:0", primals_154: "f32[128][1]cuda:0", primals_155: "f32[128][1]cuda:0", primals_156: "f32[128][1]cuda:0", primals_157: "f32[128][1]cuda:0", primals_158: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_159: "i64[][]cuda:0", primals_160: "f32[320][1]cuda:0", primals_161: "f32[320][1]cuda:0", primals_162: "f32[320][1]cuda:0", primals_163: "f32[320][1]cuda:0", primals_164: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0", primals_165: "i64[][]cuda:0", primals_166: "f32[128][1]cuda:0", primals_167: "f32[128][1]cuda:0", primals_168: "f32[128][1]cuda:0", primals_169: "f32[128][1]cuda:0", primals_170: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_171: "i64[][]cuda:0", primals_172: "f32[352][1]cuda:0", primals_173: "f32[352][1]cuda:0", primals_174: "f32[352][1]cuda:0", primals_175: "f32[352][1]cuda:0", primals_176: "f32[128, 352, 1, 1][352, 1, 1, 1]cuda:0", primals_177: "i64[][]cuda:0", primals_178: "f32[128][1]cuda:0", primals_179: "f32[128][1]cuda:0", primals_180: "f32[128][1]cuda:0", primals_181: "f32[128][1]cuda:0", primals_182: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_183: "i64[][]cuda:0", primals_184: "f32[384][1]cuda:0", primals_185: "f32[384][1]cuda:0", primals_186: "f32[384][1]cuda:0", primals_187: "f32[384][1]cuda:0", primals_188: "f32[128, 384, 1, 1][384, 1, 1, 1]cuda:0", primals_189: "i64[][]cuda:0", primals_190: "f32[128][1]cuda:0", primals_191: "f32[128][1]cuda:0", primals_192: "f32[128][1]cuda:0", primals_193: "f32[128][1]cuda:0", primals_194: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_195: "i64[][]cuda:0", primals_196: "f32[416][1]cuda:0", primals_197: "f32[416][1]cuda:0", primals_198: "f32[416][1]cuda:0", primals_199: "f32[416][1]cuda:0", primals_200: "f32[128, 416, 1, 1][416, 1, 1, 1]cuda:0", primals_201: "i64[][]cuda:0", primals_202: "f32[128][1]cuda:0", primals_203: "f32[128][1]cuda:0", primals_204: "f32[128][1]cuda:0", primals_205: "f32[128][1]cuda:0", primals_206: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_207: "i64[][]cuda:0", primals_208: "f32[448][1]cuda:0", primals_209: "f32[448][1]cuda:0", primals_210: "f32[448][1]cuda:0", primals_211: "f32[448][1]cuda:0", primals_212: "f32[128, 448, 1, 1][448, 1, 1, 1]cuda:0", primals_213: "i64[][]cuda:0", primals_214: "f32[128][1]cuda:0", primals_215: "f32[128][1]cuda:0", primals_216: "f32[128][1]cuda:0", primals_217: "f32[128][1]cuda:0", primals_218: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_219: "i64[][]cuda:0", primals_220: "f32[480][1]cuda:0", primals_221: "f32[480][1]cuda:0", primals_222: "f32[480][1]cuda:0", primals_223: "f32[480][1]cuda:0", primals_224: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0", primals_225: "i64[][]cuda:0", primals_226: "f32[128][1]cuda:0", primals_227: "f32[128][1]cuda:0", primals_228: "f32[128][1]cuda:0", primals_229: "f32[128][1]cuda:0", primals_230: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_231: "i64[][]cuda:0", primals_232: "f32[512][1]cuda:0", primals_233: "f32[512][1]cuda:0", primals_234: "f32[512][1]cuda:0", primals_235: "f32[512][1]cuda:0", primals_236: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_237: "i64[][]cuda:0", primals_238: "f32[256][1]cuda:0", primals_239: "f32[256][1]cuda:0", primals_240: "f32[256][1]cuda:0", primals_241: "f32[256][1]cuda:0", primals_242: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_243: "i64[][]cuda:0", primals_244: "f32[128][1]cuda:0", primals_245: "f32[128][1]cuda:0", primals_246: "f32[128][1]cuda:0", primals_247: "f32[128][1]cuda:0", primals_248: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_249: "i64[][]cuda:0", primals_250: "f32[288][1]cuda:0", primals_251: "f32[288][1]cuda:0", primals_252: "f32[288][1]cuda:0", primals_253: "f32[288][1]cuda:0", primals_254: "f32[128, 288, 1, 1][288, 1, 1, 1]cuda:0", primals_255: "i64[][]cuda:0", primals_256: "f32[128][1]cuda:0", primals_257: "f32[128][1]cuda:0", primals_258: "f32[128][1]cuda:0", primals_259: "f32[128][1]cuda:0", primals_260: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_261: "i64[][]cuda:0", primals_262: "f32[320][1]cuda:0", primals_263: "f32[320][1]cuda:0", primals_264: "f32[320][1]cuda:0", primals_265: "f32[320][1]cuda:0", primals_266: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0", primals_267: "i64[][]cuda:0", primals_268: "f32[128][1]cuda:0", primals_269: "f32[128][1]cuda:0", primals_270: "f32[128][1]cuda:0", primals_271: "f32[128][1]cuda:0", primals_272: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_273: "i64[][]cuda:0", primals_274: "f32[352][1]cuda:0", primals_275: "f32[352][1]cuda:0", primals_276: "f32[352][1]cuda:0", primals_277: "f32[352][1]cuda:0", primals_278: "f32[128, 352, 1, 1][352, 1, 1, 1]cuda:0", primals_279: "i64[][]cuda:0", primals_280: "f32[128][1]cuda:0", primals_281: "f32[128][1]cuda:0", primals_282: "f32[128][1]cuda:0", primals_283: "f32[128][1]cuda:0", primals_284: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_285: "i64[][]cuda:0", primals_286: "f32[384][1]cuda:0", primals_287: "f32[384][1]cuda:0", primals_288: "f32[384][1]cuda:0", primals_289: "f32[384][1]cuda:0", primals_290: "f32[128, 384, 1, 1][384, 1, 1, 1]cuda:0", primals_291: "i64[][]cuda:0", primals_292: "f32[128][1]cuda:0", primals_293: "f32[128][1]cuda:0", primals_294: "f32[128][1]cuda:0", primals_295: "f32[128][1]cuda:0", primals_296: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_297: "i64[][]cuda:0", primals_298: "f32[416][1]cuda:0", primals_299: "f32[416][1]cuda:0", primals_300: "f32[416][1]cuda:0", primals_301: "f32[416][1]cuda:0", primals_302: "f32[128, 416, 1, 1][416, 1, 1, 1]cuda:0", primals_303: "i64[][]cuda:0", primals_304: "f32[128][1]cuda:0", primals_305: "f32[128][1]cuda:0", primals_306: "f32[128][1]cuda:0", primals_307: "f32[128][1]cuda:0", primals_308: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_309: "i64[][]cuda:0", primals_310: "f32[448][1]cuda:0", primals_311: "f32[448][1]cuda:0", primals_312: "f32[448][1]cuda:0", primals_313: "f32[448][1]cuda:0", primals_314: "f32[128, 448, 1, 1][448, 1, 1, 1]cuda:0", primals_315: "i64[][]cuda:0", primals_316: "f32[128][1]cuda:0", primals_317: "f32[128][1]cuda:0", primals_318: "f32[128][1]cuda:0", primals_319: "f32[128][1]cuda:0", primals_320: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_321: "i64[][]cuda:0", primals_322: "f32[480][1]cuda:0", primals_323: "f32[480][1]cuda:0", primals_324: "f32[480][1]cuda:0", primals_325: "f32[480][1]cuda:0", primals_326: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0", primals_327: "i64[][]cuda:0", primals_328: "f32[128][1]cuda:0", primals_329: "f32[128][1]cuda:0", primals_330: "f32[128][1]cuda:0", primals_331: "f32[128][1]cuda:0", primals_332: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_333: "i64[][]cuda:0", primals_334: "f32[512][1]cuda:0", primals_335: "f32[512][1]cuda:0", primals_336: "f32[512][1]cuda:0", primals_337: "f32[512][1]cuda:0", primals_338: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_339: "i64[][]cuda:0", primals_340: "f32[128][1]cuda:0", primals_341: "f32[128][1]cuda:0", primals_342: "f32[128][1]cuda:0", primals_343: "f32[128][1]cuda:0", primals_344: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_345: "i64[][]cuda:0", primals_346: "f32[544][1]cuda:0", primals_347: "f32[544][1]cuda:0", primals_348: "f32[544][1]cuda:0", primals_349: "f32[544][1]cuda:0", primals_350: "f32[128, 544, 1, 1][544, 1, 1, 1]cuda:0", primals_351: "i64[][]cuda:0", primals_352: "f32[128][1]cuda:0", primals_353: "f32[128][1]cuda:0", primals_354: "f32[128][1]cuda:0", primals_355: "f32[128][1]cuda:0", primals_356: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_357: "i64[][]cuda:0", primals_358: "f32[576][1]cuda:0", primals_359: "f32[576][1]cuda:0", primals_360: "f32[576][1]cuda:0", primals_361: "f32[576][1]cuda:0", primals_362: "f32[128, 576, 1, 1][576, 1, 1, 1]cuda:0", primals_363: "i64[][]cuda:0", primals_364: "f32[128][1]cuda:0", primals_365: "f32[128][1]cuda:0", primals_366: "f32[128][1]cuda:0", primals_367: "f32[128][1]cuda:0", primals_368: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_369: "i64[][]cuda:0", primals_370: "f32[608][1]cuda:0", primals_371: "f32[608][1]cuda:0", primals_372: "f32[608][1]cuda:0", primals_373: "f32[608][1]cuda:0", primals_374: "f32[128, 608, 1, 1][608, 1, 1, 1]cuda:0", primals_375: "i64[][]cuda:0", primals_376: "f32[128][1]cuda:0", primals_377: "f32[128][1]cuda:0", primals_378: "f32[128][1]cuda:0", primals_379: "f32[128][1]cuda:0", primals_380: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_381: "i64[][]cuda:0", primals_382: "f32[640][1]cuda:0", primals_383: "f32[640][1]cuda:0", primals_384: "f32[640][1]cuda:0", primals_385: "f32[640][1]cuda:0", primals_386: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0", primals_387: "i64[][]cuda:0", primals_388: "f32[128][1]cuda:0", primals_389: "f32[128][1]cuda:0", primals_390: "f32[128][1]cuda:0", primals_391: "f32[128][1]cuda:0", primals_392: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_393: "i64[][]cuda:0", primals_394: "f32[672][1]cuda:0", primals_395: "f32[672][1]cuda:0", primals_396: "f32[672][1]cuda:0", primals_397: "f32[672][1]cuda:0", primals_398: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0", primals_399: "i64[][]cuda:0", primals_400: "f32[128][1]cuda:0", primals_401: "f32[128][1]cuda:0", primals_402: "f32[128][1]cuda:0", primals_403: "f32[128][1]cuda:0", primals_404: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_405: "i64[][]cuda:0", primals_406: "f32[704][1]cuda:0", primals_407: "f32[704][1]cuda:0", primals_408: "f32[704][1]cuda:0", primals_409: "f32[704][1]cuda:0", primals_410: "f32[128, 704, 1, 1][704, 1, 1, 1]cuda:0", primals_411: "i64[][]cuda:0", primals_412: "f32[128][1]cuda:0", primals_413: "f32[128][1]cuda:0", primals_414: "f32[128][1]cuda:0", primals_415: "f32[128][1]cuda:0", primals_416: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_417: "i64[][]cuda:0", primals_418: "f32[736][1]cuda:0", primals_419: "f32[736][1]cuda:0", primals_420: "f32[736][1]cuda:0", primals_421: "f32[736][1]cuda:0", primals_422: "f32[128, 736, 1, 1][736, 1, 1, 1]cuda:0", primals_423: "i64[][]cuda:0", primals_424: "f32[128][1]cuda:0", primals_425: "f32[128][1]cuda:0", primals_426: "f32[128][1]cuda:0", primals_427: "f32[128][1]cuda:0", primals_428: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_429: "i64[][]cuda:0", primals_430: "f32[768][1]cuda:0", primals_431: "f32[768][1]cuda:0", primals_432: "f32[768][1]cuda:0", primals_433: "f32[768][1]cuda:0", primals_434: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0", primals_435: "i64[][]cuda:0", primals_436: "f32[128][1]cuda:0", primals_437: "f32[128][1]cuda:0", primals_438: "f32[128][1]cuda:0", primals_439: "f32[128][1]cuda:0", primals_440: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_441: "i64[][]cuda:0", primals_442: "f32[800][1]cuda:0", primals_443: "f32[800][1]cuda:0", primals_444: "f32[800][1]cuda:0", primals_445: "f32[800][1]cuda:0", primals_446: "f32[128, 800, 1, 1][800, 1, 1, 1]cuda:0", primals_447: "i64[][]cuda:0", primals_448: "f32[128][1]cuda:0", primals_449: "f32[128][1]cuda:0", primals_450: "f32[128][1]cuda:0", primals_451: "f32[128][1]cuda:0", primals_452: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_453: "i64[][]cuda:0", primals_454: "f32[832][1]cuda:0", primals_455: "f32[832][1]cuda:0", primals_456: "f32[832][1]cuda:0", primals_457: "f32[832][1]cuda:0", primals_458: "f32[128, 832, 1, 1][832, 1, 1, 1]cuda:0", primals_459: "i64[][]cuda:0", primals_460: "f32[128][1]cuda:0", primals_461: "f32[128][1]cuda:0", primals_462: "f32[128][1]cuda:0", primals_463: "f32[128][1]cuda:0", primals_464: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_465: "i64[][]cuda:0", primals_466: "f32[864][1]cuda:0", primals_467: "f32[864][1]cuda:0", primals_468: "f32[864][1]cuda:0", primals_469: "f32[864][1]cuda:0", primals_470: "f32[128, 864, 1, 1][864, 1, 1, 1]cuda:0", primals_471: "i64[][]cuda:0", primals_472: "f32[128][1]cuda:0", primals_473: "f32[128][1]cuda:0", primals_474: "f32[128][1]cuda:0", primals_475: "f32[128][1]cuda:0", primals_476: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_477: "i64[][]cuda:0", primals_478: "f32[896][1]cuda:0", primals_479: "f32[896][1]cuda:0", primals_480: "f32[896][1]cuda:0", primals_481: "f32[896][1]cuda:0", primals_482: "f32[128, 896, 1, 1][896, 1, 1, 1]cuda:0", primals_483: "i64[][]cuda:0", primals_484: "f32[128][1]cuda:0", primals_485: "f32[128][1]cuda:0", primals_486: "f32[128][1]cuda:0", primals_487: "f32[128][1]cuda:0", primals_488: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_489: "i64[][]cuda:0", primals_490: "f32[928][1]cuda:0", primals_491: "f32[928][1]cuda:0", primals_492: "f32[928][1]cuda:0", primals_493: "f32[928][1]cuda:0", primals_494: "f32[128, 928, 1, 1][928, 1, 1, 1]cuda:0", primals_495: "i64[][]cuda:0", primals_496: "f32[128][1]cuda:0", primals_497: "f32[128][1]cuda:0", primals_498: "f32[128][1]cuda:0", primals_499: "f32[128][1]cuda:0", primals_500: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_501: "i64[][]cuda:0", primals_502: "f32[960][1]cuda:0", primals_503: "f32[960][1]cuda:0", primals_504: "f32[960][1]cuda:0", primals_505: "f32[960][1]cuda:0", primals_506: "f32[128, 960, 1, 1][960, 1, 1, 1]cuda:0", primals_507: "i64[][]cuda:0", primals_508: "f32[128][1]cuda:0", primals_509: "f32[128][1]cuda:0", primals_510: "f32[128][1]cuda:0", primals_511: "f32[128][1]cuda:0", primals_512: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_513: "i64[][]cuda:0", primals_514: "f32[992][1]cuda:0", primals_515: "f32[992][1]cuda:0", primals_516: "f32[992][1]cuda:0", primals_517: "f32[992][1]cuda:0", primals_518: "f32[128, 992, 1, 1][992, 1, 1, 1]cuda:0", primals_519: "i64[][]cuda:0", primals_520: "f32[128][1]cuda:0", primals_521: "f32[128][1]cuda:0", primals_522: "f32[128][1]cuda:0", primals_523: "f32[128][1]cuda:0", primals_524: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_525: "i64[][]cuda:0", primals_526: "f32[1024][1]cuda:0", primals_527: "f32[1024][1]cuda:0", primals_528: "f32[1024][1]cuda:0", primals_529: "f32[1024][1]cuda:0", primals_530: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_531: "i64[][]cuda:0", primals_532: "f32[512][1]cuda:0", primals_533: "f32[512][1]cuda:0", primals_534: "f32[512][1]cuda:0", primals_535: "f32[512][1]cuda:0", primals_536: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_537: "i64[][]cuda:0", primals_538: "f32[128][1]cuda:0", primals_539: "f32[128][1]cuda:0", primals_540: "f32[128][1]cuda:0", primals_541: "f32[128][1]cuda:0", primals_542: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_543: "i64[][]cuda:0", primals_544: "f32[544][1]cuda:0", primals_545: "f32[544][1]cuda:0", primals_546: "f32[544][1]cuda:0", primals_547: "f32[544][1]cuda:0", primals_548: "f32[128, 544, 1, 1][544, 1, 1, 1]cuda:0", primals_549: "i64[][]cuda:0", primals_550: "f32[128][1]cuda:0", primals_551: "f32[128][1]cuda:0", primals_552: "f32[128][1]cuda:0", primals_553: "f32[128][1]cuda:0", primals_554: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_555: "i64[][]cuda:0", primals_556: "f32[576][1]cuda:0", primals_557: "f32[576][1]cuda:0", primals_558: "f32[576][1]cuda:0", primals_559: "f32[576][1]cuda:0", primals_560: "f32[128, 576, 1, 1][576, 1, 1, 1]cuda:0", primals_561: "i64[][]cuda:0", primals_562: "f32[128][1]cuda:0", primals_563: "f32[128][1]cuda:0", primals_564: "f32[128][1]cuda:0", primals_565: "f32[128][1]cuda:0", primals_566: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_567: "i64[][]cuda:0", primals_568: "f32[608][1]cuda:0", primals_569: "f32[608][1]cuda:0", primals_570: "f32[608][1]cuda:0", primals_571: "f32[608][1]cuda:0", primals_572: "f32[128, 608, 1, 1][608, 1, 1, 1]cuda:0", primals_573: "i64[][]cuda:0", primals_574: "f32[128][1]cuda:0", primals_575: "f32[128][1]cuda:0", primals_576: "f32[128][1]cuda:0", primals_577: "f32[128][1]cuda:0", primals_578: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_579: "i64[][]cuda:0", primals_580: "f32[640][1]cuda:0", primals_581: "f32[640][1]cuda:0", primals_582: "f32[640][1]cuda:0", primals_583: "f32[640][1]cuda:0", primals_584: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0", primals_585: "i64[][]cuda:0", primals_586: "f32[128][1]cuda:0", primals_587: "f32[128][1]cuda:0", primals_588: "f32[128][1]cuda:0", primals_589: "f32[128][1]cuda:0", primals_590: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_591: "i64[][]cuda:0", primals_592: "f32[672][1]cuda:0", primals_593: "f32[672][1]cuda:0", primals_594: "f32[672][1]cuda:0", primals_595: "f32[672][1]cuda:0", primals_596: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0", primals_597: "i64[][]cuda:0", primals_598: "f32[128][1]cuda:0", primals_599: "f32[128][1]cuda:0", primals_600: "f32[128][1]cuda:0", primals_601: "f32[128][1]cuda:0", primals_602: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_603: "i64[][]cuda:0", primals_604: "f32[704][1]cuda:0", primals_605: "f32[704][1]cuda:0", primals_606: "f32[704][1]cuda:0", primals_607: "f32[704][1]cuda:0", primals_608: "f32[128, 704, 1, 1][704, 1, 1, 1]cuda:0", primals_609: "i64[][]cuda:0", primals_610: "f32[128][1]cuda:0", primals_611: "f32[128][1]cuda:0", primals_612: "f32[128][1]cuda:0", primals_613: "f32[128][1]cuda:0", primals_614: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_615: "i64[][]cuda:0", primals_616: "f32[736][1]cuda:0", primals_617: "f32[736][1]cuda:0", primals_618: "f32[736][1]cuda:0", primals_619: "f32[736][1]cuda:0", primals_620: "f32[128, 736, 1, 1][736, 1, 1, 1]cuda:0", primals_621: "i64[][]cuda:0", primals_622: "f32[128][1]cuda:0", primals_623: "f32[128][1]cuda:0", primals_624: "f32[128][1]cuda:0", primals_625: "f32[128][1]cuda:0", primals_626: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_627: "i64[][]cuda:0", primals_628: "f32[768][1]cuda:0", primals_629: "f32[768][1]cuda:0", primals_630: "f32[768][1]cuda:0", primals_631: "f32[768][1]cuda:0", primals_632: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0", primals_633: "i64[][]cuda:0", primals_634: "f32[128][1]cuda:0", primals_635: "f32[128][1]cuda:0", primals_636: "f32[128][1]cuda:0", primals_637: "f32[128][1]cuda:0", primals_638: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_639: "i64[][]cuda:0", primals_640: "f32[800][1]cuda:0", primals_641: "f32[800][1]cuda:0", primals_642: "f32[800][1]cuda:0", primals_643: "f32[800][1]cuda:0", primals_644: "f32[128, 800, 1, 1][800, 1, 1, 1]cuda:0", primals_645: "i64[][]cuda:0", primals_646: "f32[128][1]cuda:0", primals_647: "f32[128][1]cuda:0", primals_648: "f32[128][1]cuda:0", primals_649: "f32[128][1]cuda:0", primals_650: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_651: "i64[][]cuda:0", primals_652: "f32[832][1]cuda:0", primals_653: "f32[832][1]cuda:0", primals_654: "f32[832][1]cuda:0", primals_655: "f32[832][1]cuda:0", primals_656: "f32[128, 832, 1, 1][832, 1, 1, 1]cuda:0", primals_657: "i64[][]cuda:0", primals_658: "f32[128][1]cuda:0", primals_659: "f32[128][1]cuda:0", primals_660: "f32[128][1]cuda:0", primals_661: "f32[128][1]cuda:0", primals_662: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_663: "i64[][]cuda:0", primals_664: "f32[864][1]cuda:0", primals_665: "f32[864][1]cuda:0", primals_666: "f32[864][1]cuda:0", primals_667: "f32[864][1]cuda:0", primals_668: "f32[128, 864, 1, 1][864, 1, 1, 1]cuda:0", primals_669: "i64[][]cuda:0", primals_670: "f32[128][1]cuda:0", primals_671: "f32[128][1]cuda:0", primals_672: "f32[128][1]cuda:0", primals_673: "f32[128][1]cuda:0", primals_674: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_675: "i64[][]cuda:0", primals_676: "f32[896][1]cuda:0", primals_677: "f32[896][1]cuda:0", primals_678: "f32[896][1]cuda:0", primals_679: "f32[896][1]cuda:0", primals_680: "f32[128, 896, 1, 1][896, 1, 1, 1]cuda:0", primals_681: "i64[][]cuda:0", primals_682: "f32[128][1]cuda:0", primals_683: "f32[128][1]cuda:0", primals_684: "f32[128][1]cuda:0", primals_685: "f32[128][1]cuda:0", primals_686: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_687: "i64[][]cuda:0", primals_688: "f32[928][1]cuda:0", primals_689: "f32[928][1]cuda:0", primals_690: "f32[928][1]cuda:0", primals_691: "f32[928][1]cuda:0", primals_692: "f32[128, 928, 1, 1][928, 1, 1, 1]cuda:0", primals_693: "i64[][]cuda:0", primals_694: "f32[128][1]cuda:0", primals_695: "f32[128][1]cuda:0", primals_696: "f32[128][1]cuda:0", primals_697: "f32[128][1]cuda:0", primals_698: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_699: "i64[][]cuda:0", primals_700: "f32[960][1]cuda:0", primals_701: "f32[960][1]cuda:0", primals_702: "f32[960][1]cuda:0", primals_703: "f32[960][1]cuda:0", primals_704: "f32[128, 960, 1, 1][960, 1, 1, 1]cuda:0", primals_705: "i64[][]cuda:0", primals_706: "f32[128][1]cuda:0", primals_707: "f32[128][1]cuda:0", primals_708: "f32[128][1]cuda:0", primals_709: "f32[128][1]cuda:0", primals_710: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_711: "i64[][]cuda:0", primals_712: "f32[992][1]cuda:0", primals_713: "f32[992][1]cuda:0", primals_714: "f32[992][1]cuda:0", primals_715: "f32[992][1]cuda:0", primals_716: "f32[128, 992, 1, 1][992, 1, 1, 1]cuda:0", primals_717: "i64[][]cuda:0", primals_718: "f32[128][1]cuda:0", primals_719: "f32[128][1]cuda:0", primals_720: "f32[128][1]cuda:0", primals_721: "f32[128][1]cuda:0", primals_722: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_723: "i64[][]cuda:0", primals_724: "f32[1024][1]cuda:0", primals_725: "f32[1024][1]cuda:0", primals_726: "f32[1024][1]cuda:0", primals_727: "f32[1024][1]cuda:0", primals_728: "f32[1000, 1024][1024, 1]cuda:0", primals_729: "f32[1000][1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        convert_element_type: "f16[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.float16);  primals_1 = None
        convert_element_type_1: "f16[s28, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.float16);  primals_3 = None
        convolution: "f16[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
        add_10: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_4, 1)
        convert_element_type_2: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_11: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul_11: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_12: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_13: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_12: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
        sym_numel_default: "Sym(802816 * s28)" = torch.ops.aten.sym_numel.default(convolution)
        truediv: "Sym(IntTrueDiv(802816*s28, 64))" = sym_numel_default / 64;  sym_numel_default = None
        squeeze_2: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        sub_3: "Sym(-1.00000000000000 + IntTrueDiv(802816*s28, 64))" = truediv - 1.0
        truediv_1: "Sym(FloatTrueDiv(IntTrueDiv(802816*s28, 64), (IntTrueDiv(802816*s28, 64)) - 1.0))" = truediv / sub_3;  truediv = sub_3 = None
        mul_14: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, truediv_1);  squeeze_2 = truediv_1 = None
        mul_15: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, 0.1);  mul_14 = None
        mul_16: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_6, 0.9)
        add_13: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_17: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, unsqueeze_1);  mul_11 = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_8, -1)
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_14: "f32[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_3);  mul_17 = unsqueeze_3 = None
        convert_element_type_3: "f16[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.float16);  add_14 = None
        relu: "f16[s28, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem_2: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[0]
        getitem_3: "i8[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_50: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_9, 1)
        convert_element_type_4: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_4 = None
        getitem_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_51: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_11: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(getitem_2, getitem_5)
        mul_39: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_1);  sub_11 = None
        squeeze_3: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_40: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_41: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_52: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
        sym_numel_default_1: "Sym(200704 * s28)" = torch.ops.aten.sym_numel.default(getitem_2)
        truediv_2: "Sym(IntTrueDiv(200704*s28, 64))" = sym_numel_default_1 / 64;  sym_numel_default_1 = None
        squeeze_5: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        sub_12: "Sym(-1.00000000000000 + IntTrueDiv(200704*s28, 64))" = truediv_2 - 1.0
        truediv_3: "Sym(FloatTrueDiv(IntTrueDiv(200704*s28, 64), (IntTrueDiv(200704*s28, 64)) - 1.0))" = truediv_2 / sub_12;  truediv_2 = sub_12 = None
        mul_42: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, truediv_3);  squeeze_5 = truediv_3 = None
        mul_43: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, 0.1);  mul_42 = None
        mul_44: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_53: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        unsqueeze_4: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_45: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, unsqueeze_5);  mul_39 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_54: "f32[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_45, unsqueeze_7);  mul_45 = unsqueeze_7 = None
        convert_element_type_5: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.float16);  add_54 = None
        relu_1: "f16[s28, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_5);  convert_element_type_5 = None
        convert_element_type_6: "f16[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.float16);  primals_14 = None
        convolution_1: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_80: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_15, 1)
        convert_element_type_7: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_7, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_7 = None
        getitem_6: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_7: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_81: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_18: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_7)
        mul_63: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_2);  sub_18 = None
        squeeze_6: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_64: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_65: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_82: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        sym_numel_default_2: "Sym(401408 * s28)" = torch.ops.aten.sym_numel.default(convolution_1)
        truediv_4: "Sym(IntTrueDiv(401408*s28, 128))" = sym_numel_default_2 / 128;  sym_numel_default_2 = None
        squeeze_8: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        sub_19: "Sym(-1.00000000000000 + IntTrueDiv(401408*s28, 128))" = truediv_4 - 1.0
        truediv_5: "Sym(FloatTrueDiv(IntTrueDiv(401408*s28, 128), (IntTrueDiv(401408*s28, 128)) - 1.0))" = truediv_4 / sub_19;  truediv_4 = sub_19 = None
        mul_66: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, truediv_5);  squeeze_8 = truediv_5 = None
        mul_67: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_83: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_8: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_69: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_9);  mul_63 = unsqueeze_9 = None
        unsqueeze_10: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_84: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_11);  mul_69 = unsqueeze_11 = None
        convert_element_type_8: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.float16);  add_84 = None
        relu_2: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_8);  convert_element_type_8 = None
        convert_element_type_9: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.float16);  primals_20 = None
        convolution_2: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat: "f16[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_115: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_21, 1)
        convert_element_type_10: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_10, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_10 = None
        getitem_8: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_9: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_116: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_26: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat, getitem_9)
        mul_89: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_3);  sub_26 = None
        squeeze_9: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_90: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_91: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_117: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, mul_91);  mul_90 = mul_91 = None
        sym_numel_default_3: "Sym(301056 * s28)" = torch.ops.aten.sym_numel.default(cat)
        truediv_6: "Sym(IntTrueDiv(301056*s28, 96))" = sym_numel_default_3 / 96;  sym_numel_default_3 = None
        squeeze_11: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        sub_27: "Sym(-1.00000000000000 + IntTrueDiv(301056*s28, 96))" = truediv_6 - 1.0
        truediv_7: "Sym(FloatTrueDiv(IntTrueDiv(301056*s28, 96), (IntTrueDiv(301056*s28, 96)) - 1.0))" = truediv_6 / sub_27;  truediv_6 = sub_27 = None
        mul_92: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, truediv_7);  squeeze_11 = truediv_7 = None
        mul_93: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, 0.1);  mul_92 = None
        mul_94: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_118: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
        unsqueeze_12: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_95: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, unsqueeze_13);  mul_89 = unsqueeze_13 = None
        unsqueeze_14: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_119: "f32[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_15);  mul_95 = unsqueeze_15 = None
        convert_element_type_11: "f16[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.float16);  add_119 = None
        relu_3: "f16[s28, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_11);  convert_element_type_11 = None
        convert_element_type_12: "f16[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.float16);  primals_26 = None
        convolution_3: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_145: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_27, 1)
        convert_element_type_13: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_13, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_13 = None
        getitem_10: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_11: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_146: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_33: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_11)
        mul_113: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_4);  sub_33 = None
        squeeze_12: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_114: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_115: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_147: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
        sym_numel_default_4: "Sym(401408 * s28)" = torch.ops.aten.sym_numel.default(convolution_3)
        truediv_8: "Sym(IntTrueDiv(401408*s28, 128))" = sym_numel_default_4 / 128;  sym_numel_default_4 = None
        squeeze_14: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        sub_34: "Sym(-1.00000000000000 + IntTrueDiv(401408*s28, 128))" = truediv_8 - 1.0
        truediv_9: "Sym(FloatTrueDiv(IntTrueDiv(401408*s28, 128), (IntTrueDiv(401408*s28, 128)) - 1.0))" = truediv_8 / sub_34;  truediv_8 = sub_34 = None
        mul_116: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, truediv_9);  squeeze_14 = truediv_9 = None
        mul_117: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, 0.1);  mul_116 = None
        mul_118: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_148: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
        unsqueeze_16: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_119: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, unsqueeze_17);  mul_113 = unsqueeze_17 = None
        unsqueeze_18: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_149: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_19);  mul_119 = unsqueeze_19 = None
        convert_element_type_14: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.float16);  add_149 = None
        relu_4: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_14);  convert_element_type_14 = None
        convert_element_type_15: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.float16);  primals_32 = None
        convolution_4: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_15, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_1: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_180: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_33, 1)
        convert_element_type_16: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_1, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_16, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_16 = None
        getitem_12: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_13: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_181: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_41: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_1, getitem_13)
        mul_139: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_5);  sub_41 = None
        squeeze_15: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_140: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_141: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_182: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        sym_numel_default_5: "Sym(401408 * s28)" = torch.ops.aten.sym_numel.default(cat_1)
        truediv_10: "Sym(IntTrueDiv(401408*s28, 128))" = sym_numel_default_5 / 128;  sym_numel_default_5 = None
        squeeze_17: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        sub_42: "Sym(-1.00000000000000 + IntTrueDiv(401408*s28, 128))" = truediv_10 - 1.0
        truediv_11: "Sym(FloatTrueDiv(IntTrueDiv(401408*s28, 128), (IntTrueDiv(401408*s28, 128)) - 1.0))" = truediv_10 / sub_42;  truediv_10 = sub_42 = None
        mul_142: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, truediv_11);  squeeze_17 = truediv_11 = None
        mul_143: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_142, 0.1);  mul_142 = None
        mul_144: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_183: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_143, mul_144);  mul_143 = mul_144 = None
        unsqueeze_20: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_145: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_21);  mul_139 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_184: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_23);  mul_145 = unsqueeze_23 = None
        convert_element_type_17: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.float16);  add_184 = None
        relu_5: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_17);  convert_element_type_17 = None
        convert_element_type_18: "f16[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.float16);  primals_38 = None
        convolution_5: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_210: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_39, 1)
        convert_element_type_19: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_19, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_19 = None
        getitem_14: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_15: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_211: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_6: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        sub_48: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
        mul_163: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_6);  sub_48 = None
        squeeze_18: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_19: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_164: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_165: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_212: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
        sym_numel_default_6: "Sym(401408 * s28)" = torch.ops.aten.sym_numel.default(convolution_5)
        truediv_12: "Sym(IntTrueDiv(401408*s28, 128))" = sym_numel_default_6 / 128;  sym_numel_default_6 = None
        squeeze_20: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        sub_49: "Sym(-1.00000000000000 + IntTrueDiv(401408*s28, 128))" = truediv_12 - 1.0
        truediv_13: "Sym(FloatTrueDiv(IntTrueDiv(401408*s28, 128), (IntTrueDiv(401408*s28, 128)) - 1.0))" = truediv_12 / sub_49;  truediv_12 = sub_49 = None
        mul_166: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, truediv_13);  squeeze_20 = truediv_13 = None
        mul_167: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, 0.1);  mul_166 = None
        mul_168: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_213: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_167, mul_168);  mul_167 = mul_168 = None
        unsqueeze_24: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_169: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_25);  mul_163 = unsqueeze_25 = None
        unsqueeze_26: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_214: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, unsqueeze_27);  mul_169 = unsqueeze_27 = None
        convert_element_type_20: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.float16);  add_214 = None
        relu_6: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_20);  convert_element_type_20 = None
        convert_element_type_21: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.float16);  primals_44 = None
        convolution_6: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_6, convert_element_type_21, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_2: "f16[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_245: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_45, 1)
        convert_element_type_22: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_2, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_22, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_22 = None
        getitem_16: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_17: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_246: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_7: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        sub_56: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_2, getitem_17)
        mul_189: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_7);  sub_56 = None
        squeeze_21: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_22: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_190: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_191: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_247: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        sym_numel_default_7: "Sym(501760 * s28)" = torch.ops.aten.sym_numel.default(cat_2)
        truediv_14: "Sym(IntTrueDiv(501760*s28, 160))" = sym_numel_default_7 / 160;  sym_numel_default_7 = None
        squeeze_23: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        sub_57: "Sym(-1.00000000000000 + IntTrueDiv(501760*s28, 160))" = truediv_14 - 1.0
        truediv_15: "Sym(FloatTrueDiv(IntTrueDiv(501760*s28, 160), (IntTrueDiv(501760*s28, 160)) - 1.0))" = truediv_14 / sub_57;  truediv_14 = sub_57 = None
        mul_192: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, truediv_15);  squeeze_23 = truediv_15 = None
        mul_193: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_248: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_28: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_195: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_29);  mul_189 = unsqueeze_29 = None
        unsqueeze_30: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_249: "f32[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_31);  mul_195 = unsqueeze_31 = None
        convert_element_type_23: "f16[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.float16);  add_249 = None
        relu_7: "f16[s28, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_23);  convert_element_type_23 = None
        convert_element_type_24: "f16[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.float16);  primals_50 = None
        convolution_7: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_275: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_51, 1)
        convert_element_type_25: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_25, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_25 = None
        getitem_18: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_19: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_276: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_8: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_63: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_19)
        mul_213: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_8);  sub_63 = None
        squeeze_24: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_25: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_214: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_215: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_277: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        sym_numel_default_8: "Sym(401408 * s28)" = torch.ops.aten.sym_numel.default(convolution_7)
        truediv_16: "Sym(IntTrueDiv(401408*s28, 128))" = sym_numel_default_8 / 128;  sym_numel_default_8 = None
        squeeze_26: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        sub_64: "Sym(-1.00000000000000 + IntTrueDiv(401408*s28, 128))" = truediv_16 - 1.0
        truediv_17: "Sym(FloatTrueDiv(IntTrueDiv(401408*s28, 128), (IntTrueDiv(401408*s28, 128)) - 1.0))" = truediv_16 / sub_64;  truediv_16 = sub_64 = None
        mul_216: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, truediv_17);  squeeze_26 = truediv_17 = None
        mul_217: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_216, 0.1);  mul_216 = None
        mul_218: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_278: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
        unsqueeze_32: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_219: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, unsqueeze_33);  mul_213 = unsqueeze_33 = None
        unsqueeze_34: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_279: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_219, unsqueeze_35);  mul_219 = unsqueeze_35 = None
        convert_element_type_26: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.float16);  add_279 = None
        relu_8: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_26);  convert_element_type_26 = None
        convert_element_type_27: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.float16);  primals_56 = None
        convolution_8: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_27, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_3: "f16[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6, convolution_8], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_310: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_57, 1)
        convert_element_type_28: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_3, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_28, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_28 = None
        getitem_20: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_21: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_311: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_311);  add_311 = None
        sub_71: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_3, getitem_21)
        mul_239: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_9);  sub_71 = None
        squeeze_27: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_28: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_240: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_241: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_312: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_240, mul_241);  mul_240 = mul_241 = None
        sym_numel_default_9: "Sym(602112 * s28)" = torch.ops.aten.sym_numel.default(cat_3)
        truediv_18: "Sym(IntTrueDiv(602112*s28, 192))" = sym_numel_default_9 / 192;  sym_numel_default_9 = None
        squeeze_29: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        sub_72: "Sym(-1.00000000000000 + IntTrueDiv(602112*s28, 192))" = truediv_18 - 1.0
        truediv_19: "Sym(FloatTrueDiv(IntTrueDiv(602112*s28, 192), (IntTrueDiv(602112*s28, 192)) - 1.0))" = truediv_18 / sub_72;  truediv_18 = sub_72 = None
        mul_242: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, truediv_19);  squeeze_29 = truediv_19 = None
        mul_243: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, 0.1);  mul_242 = None
        mul_244: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_313: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
        unsqueeze_36: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_245: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_239, unsqueeze_37);  mul_239 = unsqueeze_37 = None
        unsqueeze_38: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_314: "f32[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_39);  mul_245 = unsqueeze_39 = None
        convert_element_type_29: "f16[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_314, torch.float16);  add_314 = None
        relu_9: "f16[s28, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_29);  convert_element_type_29 = None
        convert_element_type_30: "f16[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.float16);  primals_62 = None
        convolution_9: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_9, convert_element_type_30, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_340: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_63, 1)
        convert_element_type_31: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_31, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_31 = None
        getitem_22: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_23: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_341: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_78: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_23)
        mul_263: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_10);  sub_78 = None
        squeeze_30: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_264: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_265: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_342: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
        sym_numel_default_10: "Sym(401408 * s28)" = torch.ops.aten.sym_numel.default(convolution_9)
        truediv_20: "Sym(IntTrueDiv(401408*s28, 128))" = sym_numel_default_10 / 128;  sym_numel_default_10 = None
        squeeze_32: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        sub_79: "Sym(-1.00000000000000 + IntTrueDiv(401408*s28, 128))" = truediv_20 - 1.0
        truediv_21: "Sym(FloatTrueDiv(IntTrueDiv(401408*s28, 128), (IntTrueDiv(401408*s28, 128)) - 1.0))" = truediv_20 / sub_79;  truediv_20 = sub_79 = None
        mul_266: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, truediv_21);  squeeze_32 = truediv_21 = None
        mul_267: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, 0.1);  mul_266 = None
        mul_268: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_343: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        unsqueeze_40: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_269: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_41);  mul_263 = unsqueeze_41 = None
        unsqueeze_42: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_344: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_43);  mul_269 = unsqueeze_43 = None
        convert_element_type_32: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_344, torch.float16);  add_344 = None
        relu_10: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_32);  convert_element_type_32 = None
        convert_element_type_33: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.float16);  primals_68 = None
        convolution_10: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_4: "f16[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6, convolution_8, convolution_10], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_375: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_69, 1)
        convert_element_type_34: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_4, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_34, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_34 = None
        getitem_24: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_25: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_376: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
        sub_86: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_4, getitem_25)
        mul_289: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_11);  sub_86 = None
        squeeze_33: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_290: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_291: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_377: "f32[224][1]cuda:0" = torch.ops.aten.add.Tensor(mul_290, mul_291);  mul_290 = mul_291 = None
        sym_numel_default_11: "Sym(702464 * s28)" = torch.ops.aten.sym_numel.default(cat_4)
        truediv_22: "Sym(IntTrueDiv(702464*s28, 224))" = sym_numel_default_11 / 224;  sym_numel_default_11 = None
        squeeze_35: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        sub_87: "Sym(-1.00000000000000 + IntTrueDiv(702464*s28, 224))" = truediv_22 - 1.0
        truediv_23: "Sym(FloatTrueDiv(IntTrueDiv(702464*s28, 224), (IntTrueDiv(702464*s28, 224)) - 1.0))" = truediv_22 / sub_87;  truediv_22 = sub_87 = None
        mul_292: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, truediv_23);  squeeze_35 = truediv_23 = None
        mul_293: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_292, 0.1);  mul_292 = None
        mul_294: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_378: "f32[224][1]cuda:0" = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
        unsqueeze_44: "f32[224, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[224, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_295: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_45);  mul_289 = unsqueeze_45 = None
        unsqueeze_46: "f32[224, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[224, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_379: "f32[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_295, unsqueeze_47);  mul_295 = unsqueeze_47 = None
        convert_element_type_35: "f16[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_379, torch.float16);  add_379 = None
        relu_11: "f16[s28, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_35);  convert_element_type_35 = None
        convert_element_type_36: "f16[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.float16);  primals_74 = None
        convolution_11: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, convert_element_type_36, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_405: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_75, 1)
        convert_element_type_37: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_37, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_37 = None
        getitem_26: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_27: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_406: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_406);  add_406 = None
        sub_93: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_27)
        mul_313: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_12);  sub_93 = None
        squeeze_36: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_314: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_315: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_407: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_314, mul_315);  mul_314 = mul_315 = None
        sym_numel_default_12: "Sym(401408 * s28)" = torch.ops.aten.sym_numel.default(convolution_11)
        truediv_24: "Sym(IntTrueDiv(401408*s28, 128))" = sym_numel_default_12 / 128;  sym_numel_default_12 = None
        squeeze_38: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        sub_94: "Sym(-1.00000000000000 + IntTrueDiv(401408*s28, 128))" = truediv_24 - 1.0
        truediv_25: "Sym(FloatTrueDiv(IntTrueDiv(401408*s28, 128), (IntTrueDiv(401408*s28, 128)) - 1.0))" = truediv_24 / sub_94;  truediv_24 = sub_94 = None
        mul_316: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, truediv_25);  squeeze_38 = truediv_25 = None
        mul_317: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_316, 0.1);  mul_316 = None
        mul_318: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_408: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
        unsqueeze_48: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_319: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_49);  mul_313 = unsqueeze_49 = None
        unsqueeze_50: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_51: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_409: "f32[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_319, unsqueeze_51);  mul_319 = unsqueeze_51 = None
        convert_element_type_38: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_409, torch.float16);  add_409 = None
        relu_12: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_38);  convert_element_type_38 = None
        convert_element_type_39: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.float16);  primals_80 = None
        convolution_12: "f16[s28, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        cat_5: "f16[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6, convolution_8, convolution_10, convolution_12], 1);  convolution_2 = convolution_4 = convolution_6 = convolution_8 = convolution_10 = convolution_12 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        add_440: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_81, 1)
        convert_element_type_40: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_5, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_40, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_40 = None
        getitem_28: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_29: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_441: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_441);  add_441 = None
        sub_101: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_5, getitem_29)
        mul_339: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_13);  sub_101 = None
        squeeze_39: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_40: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_340: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_341: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_442: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        sym_numel_default_13: "Sym(802816 * s28)" = torch.ops.aten.sym_numel.default(cat_5)
        truediv_26: "Sym(IntTrueDiv(802816*s28, 256))" = sym_numel_default_13 / 256;  sym_numel_default_13 = None
        squeeze_41: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        sub_102: "Sym(-1.00000000000000 + IntTrueDiv(802816*s28, 256))" = truediv_26 - 1.0
        truediv_27: "Sym(FloatTrueDiv(IntTrueDiv(802816*s28, 256), (IntTrueDiv(802816*s28, 256)) - 1.0))" = truediv_26 / sub_102;  truediv_26 = sub_102 = None
        mul_342: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, truediv_27);  squeeze_41 = truediv_27 = None
        mul_343: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, 0.1);  mul_342 = None
        mul_344: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_443: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_343, mul_344);  mul_343 = mul_344 = None
        unsqueeze_52: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_345: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, unsqueeze_53);  mul_339 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_444: "f32[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_345, unsqueeze_55);  mul_345 = unsqueeze_55 = None
        convert_element_type_41: "f16[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_444, torch.float16);  add_444 = None
        relu_13: "f16[s28, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_41);  convert_element_type_41 = None
        convert_element_type_42: "f16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.float16);  primals_86 = None
        convolution_13: "f16[s28, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_42, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        avg_pool2d: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.avg_pool2d.default(convolution_13, [2, 2], [2, 2])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_480: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_87, 1)
        convert_element_type_43: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_43, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_43 = None
        getitem_30: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_31: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_481: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
        sub_110: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(avg_pool2d, getitem_31)
        mul_367: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_14);  sub_110 = None
        squeeze_42: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_43: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_368: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_369: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_482: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        sym_numel_default_14: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(avg_pool2d)
        truediv_28: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_14 / 128;  sym_numel_default_14 = None
        squeeze_44: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        sub_111: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_28 - 1.0
        truediv_29: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_28 / sub_111;  truediv_28 = sub_111 = None
        mul_370: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, truediv_29);  squeeze_44 = truediv_29 = None
        mul_371: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_370, 0.1);  mul_370 = None
        mul_372: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_483: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
        unsqueeze_56: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_373: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_57);  mul_367 = unsqueeze_57 = None
        unsqueeze_58: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_484: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_373, unsqueeze_59);  mul_373 = unsqueeze_59 = None
        convert_element_type_44: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_484, torch.float16);  add_484 = None
        relu_14: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_44);  convert_element_type_44 = None
        convert_element_type_45: "f16[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.float16);  primals_92 = None
        convolution_14: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_14, convert_element_type_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_510: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_93, 1)
        convert_element_type_46: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_46, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_46 = None
        getitem_32: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_33: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_511: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_15: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_511);  add_511 = None
        sub_117: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_33)
        mul_391: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_15);  sub_117 = None
        squeeze_45: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_46: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_392: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_393: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_512: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
        sym_numel_default_15: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_14)
        truediv_30: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_15 / 128;  sym_numel_default_15 = None
        squeeze_47: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        sub_118: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_30 - 1.0
        truediv_31: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_30 / sub_118;  truediv_30 = sub_118 = None
        mul_394: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, truediv_31);  squeeze_47 = truediv_31 = None
        mul_395: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_394, 0.1);  mul_394 = None
        mul_396: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_513: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_395, mul_396);  mul_395 = mul_396 = None
        unsqueeze_60: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_397: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_61);  mul_391 = unsqueeze_61 = None
        unsqueeze_62: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_514: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_397, unsqueeze_63);  mul_397 = unsqueeze_63 = None
        convert_element_type_47: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_514, torch.float16);  add_514 = None
        relu_15: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_47);  convert_element_type_47 = None
        convert_element_type_48: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.float16);  primals_98 = None
        convolution_15: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_15, convert_element_type_48, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_6: "f16[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_545: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_99, 1)
        convert_element_type_49: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_6, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_49, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_49 = None
        getitem_34: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_35: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_546: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_16: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_546);  add_546 = None
        sub_125: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_6, getitem_35)
        mul_417: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_16);  sub_125 = None
        squeeze_48: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_49: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_418: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_419: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_547: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_418, mul_419);  mul_418 = mul_419 = None
        sym_numel_default_16: "Sym(125440 * s28)" = torch.ops.aten.sym_numel.default(cat_6)
        truediv_32: "Sym(IntTrueDiv(125440*s28, 160))" = sym_numel_default_16 / 160;  sym_numel_default_16 = None
        squeeze_50: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        sub_126: "Sym(-1.00000000000000 + IntTrueDiv(125440*s28, 160))" = truediv_32 - 1.0
        truediv_33: "Sym(FloatTrueDiv(IntTrueDiv(125440*s28, 160), (IntTrueDiv(125440*s28, 160)) - 1.0))" = truediv_32 / sub_126;  truediv_32 = sub_126 = None
        mul_420: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, truediv_33);  squeeze_50 = truediv_33 = None
        mul_421: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, 0.1);  mul_420 = None
        mul_422: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_548: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
        unsqueeze_64: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_423: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_417, unsqueeze_65);  mul_417 = unsqueeze_65 = None
        unsqueeze_66: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_549: "f32[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_423, unsqueeze_67);  mul_423 = unsqueeze_67 = None
        convert_element_type_50: "f16[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_549, torch.float16);  add_549 = None
        relu_16: "f16[s28, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_50);  convert_element_type_50 = None
        convert_element_type_51: "f16[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.float16);  primals_104 = None
        convolution_16: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_16, convert_element_type_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_575: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_105, 1)
        convert_element_type_52: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_52, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_52 = None
        getitem_36: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_37: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_576: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_17: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_576);  add_576 = None
        sub_132: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_37)
        mul_441: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_132, rsqrt_17);  sub_132 = None
        squeeze_51: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_52: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_442: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_443: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_577: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
        sym_numel_default_17: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_16)
        truediv_34: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_17 / 128;  sym_numel_default_17 = None
        squeeze_53: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        sub_133: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_34 - 1.0
        truediv_35: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_34 / sub_133;  truediv_34 = sub_133 = None
        mul_444: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, truediv_35);  squeeze_53 = truediv_35 = None
        mul_445: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
        mul_446: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_578: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
        unsqueeze_68: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_447: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_69);  mul_441 = unsqueeze_69 = None
        unsqueeze_70: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_109, -1);  primals_109 = None
        unsqueeze_71: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_579: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_71);  mul_447 = unsqueeze_71 = None
        convert_element_type_53: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_579, torch.float16);  add_579 = None
        relu_17: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_53);  convert_element_type_53 = None
        convert_element_type_54: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.float16);  primals_110 = None
        convolution_17: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_17, convert_element_type_54, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_7: "f16[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_610: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_111, 1)
        convert_element_type_55: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_7, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_55, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_55 = None
        getitem_38: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_39: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_611: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_18: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_611);  add_611 = None
        sub_140: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_7, getitem_39)
        mul_467: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_18);  sub_140 = None
        squeeze_54: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_55: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_468: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_469: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_612: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
        sym_numel_default_18: "Sym(150528 * s28)" = torch.ops.aten.sym_numel.default(cat_7)
        truediv_36: "Sym(IntTrueDiv(150528*s28, 192))" = sym_numel_default_18 / 192;  sym_numel_default_18 = None
        squeeze_56: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        sub_141: "Sym(-1.00000000000000 + IntTrueDiv(150528*s28, 192))" = truediv_36 - 1.0
        truediv_37: "Sym(FloatTrueDiv(IntTrueDiv(150528*s28, 192), (IntTrueDiv(150528*s28, 192)) - 1.0))" = truediv_36 / sub_141;  truediv_36 = sub_141 = None
        mul_470: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, truediv_37);  squeeze_56 = truediv_37 = None
        mul_471: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_470, 0.1);  mul_470 = None
        mul_472: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_613: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_471, mul_472);  mul_471 = mul_472 = None
        unsqueeze_72: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_473: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_467, unsqueeze_73);  mul_467 = unsqueeze_73 = None
        unsqueeze_74: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, -1);  primals_115 = None
        unsqueeze_75: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_614: "f32[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_473, unsqueeze_75);  mul_473 = unsqueeze_75 = None
        convert_element_type_56: "f16[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_614, torch.float16);  add_614 = None
        relu_18: "f16[s28, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_56);  convert_element_type_56 = None
        convert_element_type_57: "f16[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.float16);  primals_116 = None
        convolution_18: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_18, convert_element_type_57, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_640: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_117, 1)
        convert_element_type_58: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_58, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_58 = None
        getitem_40: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_41: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_641: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_19: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_641);  add_641 = None
        sub_147: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_41)
        mul_491: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_147, rsqrt_19);  sub_147 = None
        squeeze_57: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_58: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_492: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_493: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_642: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
        sym_numel_default_19: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_18)
        truediv_38: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_19 / 128;  sym_numel_default_19 = None
        squeeze_59: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        sub_148: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_38 - 1.0
        truediv_39: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_38 / sub_148;  truediv_38 = sub_148 = None
        mul_494: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, truediv_39);  squeeze_59 = truediv_39 = None
        mul_495: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_494, 0.1);  mul_494 = None
        mul_496: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_643: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_495, mul_496);  mul_495 = mul_496 = None
        unsqueeze_76: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_497: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_491, unsqueeze_77);  mul_491 = unsqueeze_77 = None
        unsqueeze_78: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_644: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_497, unsqueeze_79);  mul_497 = unsqueeze_79 = None
        convert_element_type_59: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_644, torch.float16);  add_644 = None
        relu_19: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_59);  convert_element_type_59 = None
        convert_element_type_60: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.float16);  primals_122 = None
        convolution_19: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_19, convert_element_type_60, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_8: "f16[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_675: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_123, 1)
        convert_element_type_61: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_8, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_61, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_61 = None
        getitem_42: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_43: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_676: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_20: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_676);  add_676 = None
        sub_155: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_8, getitem_43)
        mul_517: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_155, rsqrt_20);  sub_155 = None
        squeeze_60: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_61: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_518: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_519: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_677: "f32[224][1]cuda:0" = torch.ops.aten.add.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
        sym_numel_default_20: "Sym(175616 * s28)" = torch.ops.aten.sym_numel.default(cat_8)
        truediv_40: "Sym(IntTrueDiv(175616*s28, 224))" = sym_numel_default_20 / 224;  sym_numel_default_20 = None
        squeeze_62: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        sub_156: "Sym(-1.00000000000000 + IntTrueDiv(175616*s28, 224))" = truediv_40 - 1.0
        truediv_41: "Sym(FloatTrueDiv(IntTrueDiv(175616*s28, 224), (IntTrueDiv(175616*s28, 224)) - 1.0))" = truediv_40 / sub_156;  truediv_40 = sub_156 = None
        mul_520: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, truediv_41);  squeeze_62 = truediv_41 = None
        mul_521: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_520, 0.1);  mul_520 = None
        mul_522: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_678: "f32[224][1]cuda:0" = torch.ops.aten.add.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
        unsqueeze_80: "f32[224, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_81: "f32[224, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_523: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_517, unsqueeze_81);  mul_517 = unsqueeze_81 = None
        unsqueeze_82: "f32[224, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_83: "f32[224, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_679: "f32[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_523, unsqueeze_83);  mul_523 = unsqueeze_83 = None
        convert_element_type_62: "f16[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_679, torch.float16);  add_679 = None
        relu_20: "f16[s28, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_62);  convert_element_type_62 = None
        convert_element_type_63: "f16[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.float16);  primals_128 = None
        convolution_20: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_20, convert_element_type_63, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_705: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_129, 1)
        convert_element_type_64: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_64, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_64 = None
        getitem_44: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_45: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_706: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_21: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_706);  add_706 = None
        sub_162: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_45)
        mul_541: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_162, rsqrt_21);  sub_162 = None
        squeeze_63: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_64: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_542: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_543: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_707: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_542, mul_543);  mul_542 = mul_543 = None
        sym_numel_default_21: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_20)
        truediv_42: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_21 / 128;  sym_numel_default_21 = None
        squeeze_65: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        sub_163: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_42 - 1.0
        truediv_43: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_42 / sub_163;  truediv_42 = sub_163 = None
        mul_544: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, truediv_43);  squeeze_65 = truediv_43 = None
        mul_545: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_544, 0.1);  mul_544 = None
        mul_546: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_708: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
        unsqueeze_84: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_85: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_547: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_541, unsqueeze_85);  mul_541 = unsqueeze_85 = None
        unsqueeze_86: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1);  primals_133 = None
        unsqueeze_87: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_709: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_547, unsqueeze_87);  mul_547 = unsqueeze_87 = None
        convert_element_type_65: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_709, torch.float16);  add_709 = None
        relu_21: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_65);  convert_element_type_65 = None
        convert_element_type_66: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.float16);  primals_134 = None
        convolution_21: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_21, convert_element_type_66, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_9: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_740: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_135, 1)
        convert_element_type_67: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_9, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_67, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_67 = None
        getitem_46: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_47: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_741: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_22: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_741);  add_741 = None
        sub_170: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_9, getitem_47)
        mul_567: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_170, rsqrt_22);  sub_170 = None
        squeeze_66: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_67: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_568: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_569: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_742: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
        sym_numel_default_22: "Sym(200704 * s28)" = torch.ops.aten.sym_numel.default(cat_9)
        truediv_44: "Sym(IntTrueDiv(200704*s28, 256))" = sym_numel_default_22 / 256;  sym_numel_default_22 = None
        squeeze_68: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        sub_171: "Sym(-1.00000000000000 + IntTrueDiv(200704*s28, 256))" = truediv_44 - 1.0
        truediv_45: "Sym(FloatTrueDiv(IntTrueDiv(200704*s28, 256), (IntTrueDiv(200704*s28, 256)) - 1.0))" = truediv_44 / sub_171;  truediv_44 = sub_171 = None
        mul_570: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_68, truediv_45);  squeeze_68 = truediv_45 = None
        mul_571: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
        mul_572: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_743: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
        unsqueeze_88: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_89: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_573: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_89);  mul_567 = unsqueeze_89 = None
        unsqueeze_90: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_91: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_744: "f32[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_91);  mul_573 = unsqueeze_91 = None
        convert_element_type_68: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_744, torch.float16);  add_744 = None
        relu_22: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_68);  convert_element_type_68 = None
        convert_element_type_69: "f16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.float16);  primals_140 = None
        convolution_22: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_22, convert_element_type_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_770: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_141, 1)
        convert_element_type_70: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_70, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_70 = None
        getitem_48: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_23[0]
        getitem_49: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_771: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_23: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_771);  add_771 = None
        sub_177: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, getitem_49)
        mul_591: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, rsqrt_23);  sub_177 = None
        squeeze_69: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_70: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_592: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_593: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_772: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        sym_numel_default_23: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_22)
        truediv_46: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_23 / 128;  sym_numel_default_23 = None
        squeeze_71: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        sub_178: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_46 - 1.0
        truediv_47: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_46 / sub_178;  truediv_46 = sub_178 = None
        mul_594: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, truediv_47);  squeeze_71 = truediv_47 = None
        mul_595: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_594, 0.1);  mul_594 = None
        mul_596: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_773: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_595, mul_596);  mul_595 = mul_596 = None
        unsqueeze_92: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_93: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_597: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, unsqueeze_93);  mul_591 = unsqueeze_93 = None
        unsqueeze_94: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_95: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_774: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_597, unsqueeze_95);  mul_597 = unsqueeze_95 = None
        convert_element_type_71: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_774, torch.float16);  add_774 = None
        relu_23: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_71);  convert_element_type_71 = None
        convert_element_type_72: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.float16);  primals_146 = None
        convolution_23: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_23, convert_element_type_72, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_10: "f16[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_805: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_147, 1)
        convert_element_type_73: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_10, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_73, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_73 = None
        getitem_50: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = var_mean_24[0]
        getitem_51: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_806: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_24: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_806);  add_806 = None
        sub_185: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_10, getitem_51)
        mul_617: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_185, rsqrt_24);  sub_185 = None
        squeeze_72: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_73: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_618: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_619: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_807: "f32[288][1]cuda:0" = torch.ops.aten.add.Tensor(mul_618, mul_619);  mul_618 = mul_619 = None
        sym_numel_default_24: "Sym(225792 * s28)" = torch.ops.aten.sym_numel.default(cat_10)
        truediv_48: "Sym(IntTrueDiv(225792*s28, 288))" = sym_numel_default_24 / 288;  sym_numel_default_24 = None
        squeeze_74: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        sub_186: "Sym(-1.00000000000000 + IntTrueDiv(225792*s28, 288))" = truediv_48 - 1.0
        truediv_49: "Sym(FloatTrueDiv(IntTrueDiv(225792*s28, 288), (IntTrueDiv(225792*s28, 288)) - 1.0))" = truediv_48 / sub_186;  truediv_48 = sub_186 = None
        mul_620: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_74, truediv_49);  squeeze_74 = truediv_49 = None
        mul_621: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_620, 0.1);  mul_620 = None
        mul_622: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_808: "f32[288][1]cuda:0" = torch.ops.aten.add.Tensor(mul_621, mul_622);  mul_621 = mul_622 = None
        unsqueeze_96: "f32[288, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_97: "f32[288, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_623: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_617, unsqueeze_97);  mul_617 = unsqueeze_97 = None
        unsqueeze_98: "f32[288, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_99: "f32[288, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_809: "f32[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_623, unsqueeze_99);  mul_623 = unsqueeze_99 = None
        convert_element_type_74: "f16[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_809, torch.float16);  add_809 = None
        relu_24: "f16[s28, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_74);  convert_element_type_74 = None
        convert_element_type_75: "f16[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.float16);  primals_152 = None
        convolution_24: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_24, convert_element_type_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_835: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_153, 1)
        convert_element_type_76: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_76, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_76 = None
        getitem_52: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_25[0]
        getitem_53: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_836: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_25: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_836);  add_836 = None
        sub_192: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_53)
        mul_641: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_192, rsqrt_25);  sub_192 = None
        squeeze_75: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_76: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_642: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
        mul_643: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_837: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
        sym_numel_default_25: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_24)
        truediv_50: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_25 / 128;  sym_numel_default_25 = None
        squeeze_77: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        sub_193: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_50 - 1.0
        truediv_51: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_50 / sub_193;  truediv_50 = sub_193 = None
        mul_644: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, truediv_51);  squeeze_77 = truediv_51 = None
        mul_645: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, 0.1);  mul_644 = None
        mul_646: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
        add_838: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
        unsqueeze_100: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_647: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_641, unsqueeze_101);  mul_641 = unsqueeze_101 = None
        unsqueeze_102: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_103: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_839: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_647, unsqueeze_103);  mul_647 = unsqueeze_103 = None
        convert_element_type_77: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_839, torch.float16);  add_839 = None
        relu_25: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_77);  convert_element_type_77 = None
        convert_element_type_78: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.float16);  primals_158 = None
        convolution_25: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_25, convert_element_type_78, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_11: "f16[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_870: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_159, 1)
        convert_element_type_79: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_11, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_79, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_79 = None
        getitem_54: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_26[0]
        getitem_55: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_871: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_26: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_871);  add_871 = None
        sub_200: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_11, getitem_55)
        mul_667: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_200, rsqrt_26);  sub_200 = None
        squeeze_78: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_79: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_668: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_669: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_872: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_668, mul_669);  mul_668 = mul_669 = None
        sym_numel_default_26: "Sym(250880 * s28)" = torch.ops.aten.sym_numel.default(cat_11)
        truediv_52: "Sym(IntTrueDiv(250880*s28, 320))" = sym_numel_default_26 / 320;  sym_numel_default_26 = None
        squeeze_80: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        sub_201: "Sym(-1.00000000000000 + IntTrueDiv(250880*s28, 320))" = truediv_52 - 1.0
        truediv_53: "Sym(FloatTrueDiv(IntTrueDiv(250880*s28, 320), (IntTrueDiv(250880*s28, 320)) - 1.0))" = truediv_52 / sub_201;  truediv_52 = sub_201 = None
        mul_670: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_80, truediv_53);  squeeze_80 = truediv_53 = None
        mul_671: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_670, 0.1);  mul_670 = None
        mul_672: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
        add_873: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_671, mul_672);  mul_671 = mul_672 = None
        unsqueeze_104: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_105: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_673: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_667, unsqueeze_105);  mul_667 = unsqueeze_105 = None
        unsqueeze_106: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_107: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_874: "f32[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_673, unsqueeze_107);  mul_673 = unsqueeze_107 = None
        convert_element_type_80: "f16[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_874, torch.float16);  add_874 = None
        relu_26: "f16[s28, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_80);  convert_element_type_80 = None
        convert_element_type_81: "f16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.float16);  primals_164 = None
        convolution_26: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_26, convert_element_type_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_900: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_165, 1)
        convert_element_type_82: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_82, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_82 = None
        getitem_56: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_27[0]
        getitem_57: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_901: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_27: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_901);  add_901 = None
        sub_207: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_57)
        mul_691: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_207, rsqrt_27);  sub_207 = None
        squeeze_81: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_82: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_692: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_693: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_902: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_692, mul_693);  mul_692 = mul_693 = None
        sym_numel_default_27: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_26)
        truediv_54: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_27 / 128;  sym_numel_default_27 = None
        squeeze_83: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        sub_208: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_54 - 1.0
        truediv_55: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_54 / sub_208;  truediv_54 = sub_208 = None
        mul_694: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, truediv_55);  squeeze_83 = truediv_55 = None
        mul_695: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_694, 0.1);  mul_694 = None
        mul_696: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
        add_903: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_695, mul_696);  mul_695 = mul_696 = None
        unsqueeze_108: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_109: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_697: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_691, unsqueeze_109);  mul_691 = unsqueeze_109 = None
        unsqueeze_110: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
        unsqueeze_111: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_904: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_697, unsqueeze_111);  mul_697 = unsqueeze_111 = None
        convert_element_type_83: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_904, torch.float16);  add_904 = None
        relu_27: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_83);  convert_element_type_83 = None
        convert_element_type_84: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.float16);  primals_170 = None
        convolution_27: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_27, convert_element_type_84, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_12: "f16[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_935: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_171, 1)
        convert_element_type_85: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_12, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_85, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_85 = None
        getitem_58: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = var_mean_28[0]
        getitem_59: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_936: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_28: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_936);  add_936 = None
        sub_215: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_12, getitem_59)
        mul_717: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_215, rsqrt_28);  sub_215 = None
        squeeze_84: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_85: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_718: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_719: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
        add_937: "f32[352][1]cuda:0" = torch.ops.aten.add.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
        sym_numel_default_28: "Sym(275968 * s28)" = torch.ops.aten.sym_numel.default(cat_12)
        truediv_56: "Sym(IntTrueDiv(275968*s28, 352))" = sym_numel_default_28 / 352;  sym_numel_default_28 = None
        squeeze_86: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        sub_216: "Sym(-1.00000000000000 + IntTrueDiv(275968*s28, 352))" = truediv_56 - 1.0
        truediv_57: "Sym(FloatTrueDiv(IntTrueDiv(275968*s28, 352), (IntTrueDiv(275968*s28, 352)) - 1.0))" = truediv_56 / sub_216;  truediv_56 = sub_216 = None
        mul_720: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_86, truediv_57);  squeeze_86 = truediv_57 = None
        mul_721: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_720, 0.1);  mul_720 = None
        mul_722: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
        add_938: "f32[352][1]cuda:0" = torch.ops.aten.add.Tensor(mul_721, mul_722);  mul_721 = mul_722 = None
        unsqueeze_112: "f32[352, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_113: "f32[352, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_723: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_717, unsqueeze_113);  mul_717 = unsqueeze_113 = None
        unsqueeze_114: "f32[352, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_115: "f32[352, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_939: "f32[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_723, unsqueeze_115);  mul_723 = unsqueeze_115 = None
        convert_element_type_86: "f16[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_939, torch.float16);  add_939 = None
        relu_28: "f16[s28, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_86);  convert_element_type_86 = None
        convert_element_type_87: "f16[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.float16);  primals_176 = None
        convolution_28: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_28, convert_element_type_87, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_965: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_177, 1)
        convert_element_type_88: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_88, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_88 = None
        getitem_60: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_29[0]
        getitem_61: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_966: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_29: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_966);  add_966 = None
        sub_222: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_61)
        mul_741: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_222, rsqrt_29);  sub_222 = None
        squeeze_87: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_88: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_742: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_743: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
        add_967: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_742, mul_743);  mul_742 = mul_743 = None
        sym_numel_default_29: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_28)
        truediv_58: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_29 / 128;  sym_numel_default_29 = None
        squeeze_89: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        sub_223: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_58 - 1.0
        truediv_59: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_58 / sub_223;  truediv_58 = sub_223 = None
        mul_744: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, truediv_59);  squeeze_89 = truediv_59 = None
        mul_745: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_744, 0.1);  mul_744 = None
        mul_746: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
        add_968: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_745, mul_746);  mul_745 = mul_746 = None
        unsqueeze_116: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_180, -1)
        unsqueeze_117: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_747: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_741, unsqueeze_117);  mul_741 = unsqueeze_117 = None
        unsqueeze_118: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_181, -1);  primals_181 = None
        unsqueeze_119: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_969: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_747, unsqueeze_119);  mul_747 = unsqueeze_119 = None
        convert_element_type_89: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_969, torch.float16);  add_969 = None
        relu_29: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_89);  convert_element_type_89 = None
        convert_element_type_90: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.float16);  primals_182 = None
        convolution_29: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_29, convert_element_type_90, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_13: "f16[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1000: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_183, 1)
        convert_element_type_91: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_13, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_91, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_91 = None
        getitem_62: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_30[0]
        getitem_63: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_1001: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_30: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1001);  add_1001 = None
        sub_230: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_13, getitem_63)
        mul_767: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_230, rsqrt_30);  sub_230 = None
        squeeze_90: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_91: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_768: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_769: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
        add_1002: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_768, mul_769);  mul_768 = mul_769 = None
        sym_numel_default_30: "Sym(301056 * s28)" = torch.ops.aten.sym_numel.default(cat_13)
        truediv_60: "Sym(IntTrueDiv(301056*s28, 384))" = sym_numel_default_30 / 384;  sym_numel_default_30 = None
        squeeze_92: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        sub_231: "Sym(-1.00000000000000 + IntTrueDiv(301056*s28, 384))" = truediv_60 - 1.0
        truediv_61: "Sym(FloatTrueDiv(IntTrueDiv(301056*s28, 384), (IntTrueDiv(301056*s28, 384)) - 1.0))" = truediv_60 / sub_231;  truediv_60 = sub_231 = None
        mul_770: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_92, truediv_61);  squeeze_92 = truediv_61 = None
        mul_771: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_770, 0.1);  mul_770 = None
        mul_772: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
        add_1003: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
        unsqueeze_120: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_121: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_773: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_767, unsqueeze_121);  mul_767 = unsqueeze_121 = None
        unsqueeze_122: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_187, -1);  primals_187 = None
        unsqueeze_123: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_1004: "f32[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_773, unsqueeze_123);  mul_773 = unsqueeze_123 = None
        convert_element_type_92: "f16[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1004, torch.float16);  add_1004 = None
        relu_30: "f16[s28, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_92);  convert_element_type_92 = None
        convert_element_type_93: "f16[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.float16);  primals_188 = None
        convolution_30: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_30, convert_element_type_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1030: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_189, 1)
        convert_element_type_94: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_94, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_94 = None
        getitem_64: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_31[0]
        getitem_65: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_1031: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_31: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1031);  add_1031 = None
        sub_237: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_65)
        mul_791: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_237, rsqrt_31);  sub_237 = None
        squeeze_93: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_94: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_792: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_793: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
        add_1032: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
        sym_numel_default_31: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_30)
        truediv_62: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_31 / 128;  sym_numel_default_31 = None
        squeeze_95: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        sub_238: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_62 - 1.0
        truediv_63: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_62 / sub_238;  truediv_62 = sub_238 = None
        mul_794: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, truediv_63);  squeeze_95 = truediv_63 = None
        mul_795: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_794, 0.1);  mul_794 = None
        mul_796: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
        add_1033: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
        unsqueeze_124: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_797: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_791, unsqueeze_125);  mul_791 = unsqueeze_125 = None
        unsqueeze_126: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
        unsqueeze_127: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_1034: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_797, unsqueeze_127);  mul_797 = unsqueeze_127 = None
        convert_element_type_95: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1034, torch.float16);  add_1034 = None
        relu_31: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_95);  convert_element_type_95 = None
        convert_element_type_96: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.float16);  primals_194 = None
        convolution_31: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_31, convert_element_type_96, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_14: "f16[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1065: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_195, 1)
        convert_element_type_97: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_14, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_97, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_97 = None
        getitem_66: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = var_mean_32[0]
        getitem_67: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_1066: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_32: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1066);  add_1066 = None
        sub_245: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_14, getitem_67)
        mul_817: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, rsqrt_32);  sub_245 = None
        squeeze_96: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_97: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_818: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_819: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_1067: "f32[416][1]cuda:0" = torch.ops.aten.add.Tensor(mul_818, mul_819);  mul_818 = mul_819 = None
        sym_numel_default_32: "Sym(326144 * s28)" = torch.ops.aten.sym_numel.default(cat_14)
        truediv_64: "Sym(IntTrueDiv(326144*s28, 416))" = sym_numel_default_32 / 416;  sym_numel_default_32 = None
        squeeze_98: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        sub_246: "Sym(-1.00000000000000 + IntTrueDiv(326144*s28, 416))" = truediv_64 - 1.0
        truediv_65: "Sym(FloatTrueDiv(IntTrueDiv(326144*s28, 416), (IntTrueDiv(326144*s28, 416)) - 1.0))" = truediv_64 / sub_246;  truediv_64 = sub_246 = None
        mul_820: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_98, truediv_65);  squeeze_98 = truediv_65 = None
        mul_821: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_820, 0.1);  mul_820 = None
        mul_822: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
        add_1068: "f32[416][1]cuda:0" = torch.ops.aten.add.Tensor(mul_821, mul_822);  mul_821 = mul_822 = None
        unsqueeze_128: "f32[416, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_129: "f32[416, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_823: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_817, unsqueeze_129);  mul_817 = unsqueeze_129 = None
        unsqueeze_130: "f32[416, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
        unsqueeze_131: "f32[416, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_1069: "f32[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_823, unsqueeze_131);  mul_823 = unsqueeze_131 = None
        convert_element_type_98: "f16[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1069, torch.float16);  add_1069 = None
        relu_32: "f16[s28, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_98);  convert_element_type_98 = None
        convert_element_type_99: "f16[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.float16);  primals_200 = None
        convolution_32: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_32, convert_element_type_99, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1095: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_201, 1)
        convert_element_type_100: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_100, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_100 = None
        getitem_68: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_33[0]
        getitem_69: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_1096: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_33: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1096);  add_1096 = None
        sub_252: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_69)
        mul_841: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_252, rsqrt_33);  sub_252 = None
        squeeze_99: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_100: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_842: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_843: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_1097: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_842, mul_843);  mul_842 = mul_843 = None
        sym_numel_default_33: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_32)
        truediv_66: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_33 / 128;  sym_numel_default_33 = None
        squeeze_101: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        sub_253: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_66 - 1.0
        truediv_67: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_66 / sub_253;  truediv_66 = sub_253 = None
        mul_844: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, truediv_67);  squeeze_101 = truediv_67 = None
        mul_845: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_844, 0.1);  mul_844 = None
        mul_846: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
        add_1098: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_845, mul_846);  mul_845 = mul_846 = None
        unsqueeze_132: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_133: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_847: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_841, unsqueeze_133);  mul_841 = unsqueeze_133 = None
        unsqueeze_134: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_135: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_1099: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_847, unsqueeze_135);  mul_847 = unsqueeze_135 = None
        convert_element_type_101: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1099, torch.float16);  add_1099 = None
        relu_33: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_101);  convert_element_type_101 = None
        convert_element_type_102: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.float16);  primals_206 = None
        convolution_33: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_33, convert_element_type_102, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_15: "f16[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1130: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_207, 1)
        convert_element_type_103: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_15, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_103, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_103 = None
        getitem_70: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_34[0]
        getitem_71: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_1131: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_34: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1131);  add_1131 = None
        sub_260: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_15, getitem_71)
        mul_867: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_260, rsqrt_34);  sub_260 = None
        squeeze_102: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_103: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_868: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_869: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_1132: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_868, mul_869);  mul_868 = mul_869 = None
        sym_numel_default_34: "Sym(351232 * s28)" = torch.ops.aten.sym_numel.default(cat_15)
        truediv_68: "Sym(IntTrueDiv(351232*s28, 448))" = sym_numel_default_34 / 448;  sym_numel_default_34 = None
        squeeze_104: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        sub_261: "Sym(-1.00000000000000 + IntTrueDiv(351232*s28, 448))" = truediv_68 - 1.0
        truediv_69: "Sym(FloatTrueDiv(IntTrueDiv(351232*s28, 448), (IntTrueDiv(351232*s28, 448)) - 1.0))" = truediv_68 / sub_261;  truediv_68 = sub_261 = None
        mul_870: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_104, truediv_69);  squeeze_104 = truediv_69 = None
        mul_871: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_870, 0.1);  mul_870 = None
        mul_872: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
        add_1133: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_871, mul_872);  mul_871 = mul_872 = None
        unsqueeze_136: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_137: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_873: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_867, unsqueeze_137);  mul_867 = unsqueeze_137 = None
        unsqueeze_138: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_139: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_1134: "f32[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_873, unsqueeze_139);  mul_873 = unsqueeze_139 = None
        convert_element_type_104: "f16[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1134, torch.float16);  add_1134 = None
        relu_34: "f16[s28, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_104);  convert_element_type_104 = None
        convert_element_type_105: "f16[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_212, torch.float16);  primals_212 = None
        convolution_34: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_34, convert_element_type_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1160: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_213, 1)
        convert_element_type_106: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_106, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_106 = None
        getitem_72: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_35[0]
        getitem_73: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_1161: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_35: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1161);  add_1161 = None
        sub_267: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_73)
        mul_891: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_267, rsqrt_35);  sub_267 = None
        squeeze_105: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_106: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_892: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_893: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
        add_1162: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_892, mul_893);  mul_892 = mul_893 = None
        sym_numel_default_35: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_34)
        truediv_70: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_35 / 128;  sym_numel_default_35 = None
        squeeze_107: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        sub_268: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_70 - 1.0
        truediv_71: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_70 / sub_268;  truediv_70 = sub_268 = None
        mul_894: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, truediv_71);  squeeze_107 = truediv_71 = None
        mul_895: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_894, 0.1);  mul_894 = None
        mul_896: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
        add_1163: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_895, mul_896);  mul_895 = mul_896 = None
        unsqueeze_140: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_216, -1)
        unsqueeze_141: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_897: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_891, unsqueeze_141);  mul_891 = unsqueeze_141 = None
        unsqueeze_142: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_217, -1);  primals_217 = None
        unsqueeze_143: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_1164: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_897, unsqueeze_143);  mul_897 = unsqueeze_143 = None
        convert_element_type_107: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1164, torch.float16);  add_1164 = None
        relu_35: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_107);  convert_element_type_107 = None
        convert_element_type_108: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_218, torch.float16);  primals_218 = None
        convolution_35: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_35, convert_element_type_108, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_16: "f16[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33, convolution_35], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1195: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_219, 1)
        convert_element_type_109: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_16, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_109, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_109 = None
        getitem_74: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_36[0]
        getitem_75: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_1196: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_36: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1196);  add_1196 = None
        sub_275: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_16, getitem_75)
        mul_917: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_275, rsqrt_36);  sub_275 = None
        squeeze_108: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_109: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_918: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_919: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_1197: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_918, mul_919);  mul_918 = mul_919 = None
        sym_numel_default_36: "Sym(376320 * s28)" = torch.ops.aten.sym_numel.default(cat_16)
        truediv_72: "Sym(IntTrueDiv(376320*s28, 480))" = sym_numel_default_36 / 480;  sym_numel_default_36 = None
        squeeze_110: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        sub_276: "Sym(-1.00000000000000 + IntTrueDiv(376320*s28, 480))" = truediv_72 - 1.0
        truediv_73: "Sym(FloatTrueDiv(IntTrueDiv(376320*s28, 480), (IntTrueDiv(376320*s28, 480)) - 1.0))" = truediv_72 / sub_276;  truediv_72 = sub_276 = None
        mul_920: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_110, truediv_73);  squeeze_110 = truediv_73 = None
        mul_921: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_920, 0.1);  mul_920 = None
        mul_922: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_1198: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
        unsqueeze_144: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_145: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_923: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_917, unsqueeze_145);  mul_917 = unsqueeze_145 = None
        unsqueeze_146: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_147: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_1199: "f32[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_923, unsqueeze_147);  mul_923 = unsqueeze_147 = None
        convert_element_type_110: "f16[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1199, torch.float16);  add_1199 = None
        relu_36: "f16[s28, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_110);  convert_element_type_110 = None
        convert_element_type_111: "f16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_224, torch.float16);  primals_224 = None
        convolution_36: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_36, convert_element_type_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1225: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_225, 1)
        convert_element_type_112: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_112, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_112 = None
        getitem_76: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_37[0]
        getitem_77: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_1226: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_37: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1226);  add_1226 = None
        sub_282: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_77)
        mul_941: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_282, rsqrt_37);  sub_282 = None
        squeeze_111: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_112: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_942: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_943: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_1227: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
        sym_numel_default_37: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(convolution_36)
        truediv_74: "Sym(IntTrueDiv(100352*s28, 128))" = sym_numel_default_37 / 128;  sym_numel_default_37 = None
        squeeze_113: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        sub_283: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 128))" = truediv_74 - 1.0
        truediv_75: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 128), (IntTrueDiv(100352*s28, 128)) - 1.0))" = truediv_74 / sub_283;  truediv_74 = sub_283 = None
        mul_944: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, truediv_75);  squeeze_113 = truediv_75 = None
        mul_945: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_944, 0.1);  mul_944 = None
        mul_946: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_1228: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_945, mul_946);  mul_945 = mul_946 = None
        unsqueeze_148: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_947: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_941, unsqueeze_149);  mul_941 = unsqueeze_149 = None
        unsqueeze_150: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_151: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_1229: "f32[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_947, unsqueeze_151);  mul_947 = unsqueeze_151 = None
        convert_element_type_113: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1229, torch.float16);  add_1229 = None
        relu_37: "f16[s28, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_113);  convert_element_type_113 = None
        convert_element_type_114: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.float16);  primals_230 = None
        convolution_37: "f16[s28, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_37, convert_element_type_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        cat_17: "f16[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33, convolution_35, convolution_37], 1);  convolution_15 = convolution_17 = convolution_19 = convolution_21 = convolution_23 = convolution_25 = convolution_27 = convolution_29 = convolution_31 = convolution_33 = convolution_35 = convolution_37 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        add_1260: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_231, 1)
        convert_element_type_115: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_17, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_115, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_115 = None
        getitem_78: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_38[0]
        getitem_79: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_1261: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_38: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1261);  add_1261 = None
        sub_290: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_17, getitem_79)
        mul_967: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_290, rsqrt_38);  sub_290 = None
        squeeze_114: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_115: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_968: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_969: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_1262: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_968, mul_969);  mul_968 = mul_969 = None
        sym_numel_default_38: "Sym(401408 * s28)" = torch.ops.aten.sym_numel.default(cat_17)
        truediv_76: "Sym(IntTrueDiv(401408*s28, 512))" = sym_numel_default_38 / 512;  sym_numel_default_38 = None
        squeeze_116: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        sub_291: "Sym(-1.00000000000000 + IntTrueDiv(401408*s28, 512))" = truediv_76 - 1.0
        truediv_77: "Sym(FloatTrueDiv(IntTrueDiv(401408*s28, 512), (IntTrueDiv(401408*s28, 512)) - 1.0))" = truediv_76 / sub_291;  truediv_76 = sub_291 = None
        mul_970: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_116, truediv_77);  squeeze_116 = truediv_77 = None
        mul_971: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_970, 0.1);  mul_970 = None
        mul_972: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_1263: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_971, mul_972);  mul_971 = mul_972 = None
        unsqueeze_152: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_153: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_973: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_967, unsqueeze_153);  mul_967 = unsqueeze_153 = None
        unsqueeze_154: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_155: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_1264: "f32[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_973, unsqueeze_155);  mul_973 = unsqueeze_155 = None
        convert_element_type_116: "f16[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1264, torch.float16);  add_1264 = None
        relu_38: "f16[s28, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_116);  convert_element_type_116 = None
        convert_element_type_117: "f16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.float16);  primals_236 = None
        convolution_38: "f16[s28, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_38, convert_element_type_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        avg_pool2d_1: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.avg_pool2d.default(convolution_38, [2, 2], [2, 2])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1300: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_237, 1)
        convert_element_type_118: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_1, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_118, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_118 = None
        getitem_80: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_39[0]
        getitem_81: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_1301: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_39: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1301);  add_1301 = None
        sub_299: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(avg_pool2d_1, getitem_81)
        mul_995: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_299, rsqrt_39);  sub_299 = None
        squeeze_117: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_118: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_996: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_997: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_1302: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_996, mul_997);  mul_996 = mul_997 = None
        sym_numel_default_39: "Sym(50176 * s28)" = torch.ops.aten.sym_numel.default(avg_pool2d_1)
        truediv_78: "Sym(IntTrueDiv(50176*s28, 256))" = sym_numel_default_39 / 256;  sym_numel_default_39 = None
        squeeze_119: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        sub_300: "Sym(-1.00000000000000 + IntTrueDiv(50176*s28, 256))" = truediv_78 - 1.0
        truediv_79: "Sym(FloatTrueDiv(IntTrueDiv(50176*s28, 256), (IntTrueDiv(50176*s28, 256)) - 1.0))" = truediv_78 / sub_300;  truediv_78 = sub_300 = None
        mul_998: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_119, truediv_79);  squeeze_119 = truediv_79 = None
        mul_999: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_998, 0.1);  mul_998 = None
        mul_1000: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
        add_1303: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_999, mul_1000);  mul_999 = mul_1000 = None
        unsqueeze_156: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_157: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_1001: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_995, unsqueeze_157);  mul_995 = unsqueeze_157 = None
        unsqueeze_158: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
        unsqueeze_159: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_1304: "f32[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1001, unsqueeze_159);  mul_1001 = unsqueeze_159 = None
        convert_element_type_119: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1304, torch.float16);  add_1304 = None
        relu_39: "f16[s28, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_119);  convert_element_type_119 = None
        convert_element_type_120: "f16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_242, torch.float16);  primals_242 = None
        convolution_39: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_39, convert_element_type_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1330: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_243, 1)
        convert_element_type_121: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_121, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_121 = None
        getitem_82: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_40[0]
        getitem_83: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_1331: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_40: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1331);  add_1331 = None
        sub_306: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, getitem_83)
        mul_1019: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_306, rsqrt_40);  sub_306 = None
        squeeze_120: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_121: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_1020: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
        mul_1021: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_1332: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1020, mul_1021);  mul_1020 = mul_1021 = None
        sym_numel_default_40: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_39)
        truediv_80: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_40 / 128;  sym_numel_default_40 = None
        squeeze_122: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        sub_307: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_80 - 1.0
        truediv_81: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_80 / sub_307;  truediv_80 = sub_307 = None
        mul_1022: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_122, truediv_81);  squeeze_122 = truediv_81 = None
        mul_1023: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1022, 0.1);  mul_1022 = None
        mul_1024: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
        add_1333: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1023, mul_1024);  mul_1023 = mul_1024 = None
        unsqueeze_160: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_161: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_1025: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1019, unsqueeze_161);  mul_1019 = unsqueeze_161 = None
        unsqueeze_162: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_163: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_1334: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1025, unsqueeze_163);  mul_1025 = unsqueeze_163 = None
        convert_element_type_122: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1334, torch.float16);  add_1334 = None
        relu_40: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_122);  convert_element_type_122 = None
        convert_element_type_123: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_248, torch.float16);  primals_248 = None
        convolution_40: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_40, convert_element_type_123, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_18: "f16[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1365: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_249, 1)
        convert_element_type_124: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_18, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_124, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_124 = None
        getitem_84: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = var_mean_41[0]
        getitem_85: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_1366: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_41: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1366);  add_1366 = None
        sub_314: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_18, getitem_85)
        mul_1045: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_314, rsqrt_41);  sub_314 = None
        squeeze_123: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_124: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_1046: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_1047: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_1367: "f32[288][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1046, mul_1047);  mul_1046 = mul_1047 = None
        sym_numel_default_41: "Sym(56448 * s28)" = torch.ops.aten.sym_numel.default(cat_18)
        truediv_82: "Sym(IntTrueDiv(56448*s28, 288))" = sym_numel_default_41 / 288;  sym_numel_default_41 = None
        squeeze_125: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        sub_315: "Sym(-1.00000000000000 + IntTrueDiv(56448*s28, 288))" = truediv_82 - 1.0
        truediv_83: "Sym(FloatTrueDiv(IntTrueDiv(56448*s28, 288), (IntTrueDiv(56448*s28, 288)) - 1.0))" = truediv_82 / sub_315;  truediv_82 = sub_315 = None
        mul_1048: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_125, truediv_83);  squeeze_125 = truediv_83 = None
        mul_1049: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1048, 0.1);  mul_1048 = None
        mul_1050: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
        add_1368: "f32[288][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1049, mul_1050);  mul_1049 = mul_1050 = None
        unsqueeze_164: "f32[288, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_252, -1)
        unsqueeze_165: "f32[288, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_1051: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1045, unsqueeze_165);  mul_1045 = unsqueeze_165 = None
        unsqueeze_166: "f32[288, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
        unsqueeze_167: "f32[288, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_1369: "f32[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1051, unsqueeze_167);  mul_1051 = unsqueeze_167 = None
        convert_element_type_125: "f16[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1369, torch.float16);  add_1369 = None
        relu_41: "f16[s28, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_125);  convert_element_type_125 = None
        convert_element_type_126: "f16[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.float16);  primals_254 = None
        convolution_41: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_41, convert_element_type_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1395: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_255, 1)
        convert_element_type_127: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_127, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_127 = None
        getitem_86: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_42[0]
        getitem_87: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_1396: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_42: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1396);  add_1396 = None
        sub_321: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, getitem_87)
        mul_1069: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_321, rsqrt_42);  sub_321 = None
        squeeze_126: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_127: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_1070: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_1071: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
        add_1397: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1070, mul_1071);  mul_1070 = mul_1071 = None
        sym_numel_default_42: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_41)
        truediv_84: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_42 / 128;  sym_numel_default_42 = None
        squeeze_128: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        sub_322: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_84 - 1.0
        truediv_85: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_84 / sub_322;  truediv_84 = sub_322 = None
        mul_1072: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_128, truediv_85);  squeeze_128 = truediv_85 = None
        mul_1073: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1072, 0.1);  mul_1072 = None
        mul_1074: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
        add_1398: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1073, mul_1074);  mul_1073 = mul_1074 = None
        unsqueeze_168: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_258, -1)
        unsqueeze_169: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_1075: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1069, unsqueeze_169);  mul_1069 = unsqueeze_169 = None
        unsqueeze_170: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
        unsqueeze_171: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_1399: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1075, unsqueeze_171);  mul_1075 = unsqueeze_171 = None
        convert_element_type_128: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1399, torch.float16);  add_1399 = None
        relu_42: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_128);  convert_element_type_128 = None
        convert_element_type_129: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.float16);  primals_260 = None
        convolution_42: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_42, convert_element_type_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_19: "f16[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1430: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_261, 1)
        convert_element_type_130: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_19, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_130, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_130 = None
        getitem_88: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_43[0]
        getitem_89: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_1431: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_43: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1431);  add_1431 = None
        sub_329: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_19, getitem_89)
        mul_1095: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_329, rsqrt_43);  sub_329 = None
        squeeze_129: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_130: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_1096: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_1097: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_1432: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1096, mul_1097);  mul_1096 = mul_1097 = None
        sym_numel_default_43: "Sym(62720 * s28)" = torch.ops.aten.sym_numel.default(cat_19)
        truediv_86: "Sym(IntTrueDiv(62720*s28, 320))" = sym_numel_default_43 / 320;  sym_numel_default_43 = None
        squeeze_131: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        sub_330: "Sym(-1.00000000000000 + IntTrueDiv(62720*s28, 320))" = truediv_86 - 1.0
        truediv_87: "Sym(FloatTrueDiv(IntTrueDiv(62720*s28, 320), (IntTrueDiv(62720*s28, 320)) - 1.0))" = truediv_86 / sub_330;  truediv_86 = sub_330 = None
        mul_1098: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_131, truediv_87);  squeeze_131 = truediv_87 = None
        mul_1099: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1098, 0.1);  mul_1098 = None
        mul_1100: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
        add_1433: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1099, mul_1100);  mul_1099 = mul_1100 = None
        unsqueeze_172: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_173: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_1101: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1095, unsqueeze_173);  mul_1095 = unsqueeze_173 = None
        unsqueeze_174: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_265, -1);  primals_265 = None
        unsqueeze_175: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_1434: "f32[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1101, unsqueeze_175);  mul_1101 = unsqueeze_175 = None
        convert_element_type_131: "f16[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1434, torch.float16);  add_1434 = None
        relu_43: "f16[s28, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_131);  convert_element_type_131 = None
        convert_element_type_132: "f16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.float16);  primals_266 = None
        convolution_43: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_43, convert_element_type_132, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1460: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_267, 1)
        convert_element_type_133: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_133, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_133 = None
        getitem_90: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_44[0]
        getitem_91: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_1461: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_44: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1461);  add_1461 = None
        sub_336: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, getitem_91)
        mul_1119: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_336, rsqrt_44);  sub_336 = None
        squeeze_132: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_133: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_1120: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_1121: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_1462: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1120, mul_1121);  mul_1120 = mul_1121 = None
        sym_numel_default_44: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_43)
        truediv_88: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_44 / 128;  sym_numel_default_44 = None
        squeeze_134: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        sub_337: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_88 - 1.0
        truediv_89: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_88 / sub_337;  truediv_88 = sub_337 = None
        mul_1122: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_134, truediv_89);  squeeze_134 = truediv_89 = None
        mul_1123: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1122, 0.1);  mul_1122 = None
        mul_1124: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
        add_1463: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1123, mul_1124);  mul_1123 = mul_1124 = None
        unsqueeze_176: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_177: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_1125: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1119, unsqueeze_177);  mul_1119 = unsqueeze_177 = None
        unsqueeze_178: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_271, -1);  primals_271 = None
        unsqueeze_179: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_1464: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1125, unsqueeze_179);  mul_1125 = unsqueeze_179 = None
        convert_element_type_134: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1464, torch.float16);  add_1464 = None
        relu_44: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_134);  convert_element_type_134 = None
        convert_element_type_135: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.float16);  primals_272 = None
        convolution_44: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_44, convert_element_type_135, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_20: "f16[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1495: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_273, 1)
        convert_element_type_136: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_20, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_136, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_136 = None
        getitem_92: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = var_mean_45[0]
        getitem_93: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_1496: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_45: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1496);  add_1496 = None
        sub_344: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_20, getitem_93)
        mul_1145: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_344, rsqrt_45);  sub_344 = None
        squeeze_135: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_136: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_1146: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
        mul_1147: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
        add_1497: "f32[352][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1146, mul_1147);  mul_1146 = mul_1147 = None
        sym_numel_default_45: "Sym(68992 * s28)" = torch.ops.aten.sym_numel.default(cat_20)
        truediv_90: "Sym(IntTrueDiv(68992*s28, 352))" = sym_numel_default_45 / 352;  sym_numel_default_45 = None
        squeeze_137: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        sub_345: "Sym(-1.00000000000000 + IntTrueDiv(68992*s28, 352))" = truediv_90 - 1.0
        truediv_91: "Sym(FloatTrueDiv(IntTrueDiv(68992*s28, 352), (IntTrueDiv(68992*s28, 352)) - 1.0))" = truediv_90 / sub_345;  truediv_90 = sub_345 = None
        mul_1148: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_137, truediv_91);  squeeze_137 = truediv_91 = None
        mul_1149: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1148, 0.1);  mul_1148 = None
        mul_1150: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
        add_1498: "f32[352][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1149, mul_1150);  mul_1149 = mul_1150 = None
        unsqueeze_180: "f32[352, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_181: "f32[352, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_1151: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1145, unsqueeze_181);  mul_1145 = unsqueeze_181 = None
        unsqueeze_182: "f32[352, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_183: "f32[352, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_1499: "f32[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1151, unsqueeze_183);  mul_1151 = unsqueeze_183 = None
        convert_element_type_137: "f16[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1499, torch.float16);  add_1499 = None
        relu_45: "f16[s28, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_137);  convert_element_type_137 = None
        convert_element_type_138: "f16[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_278, torch.float16);  primals_278 = None
        convolution_45: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_45, convert_element_type_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1525: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_279, 1)
        convert_element_type_139: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_139, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_139 = None
        getitem_94: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_46[0]
        getitem_95: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_1526: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_46: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1526);  add_1526 = None
        sub_351: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_95)
        mul_1169: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_351, rsqrt_46);  sub_351 = None
        squeeze_138: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_139: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_1170: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_1171: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
        add_1527: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1170, mul_1171);  mul_1170 = mul_1171 = None
        sym_numel_default_46: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_45)
        truediv_92: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_46 / 128;  sym_numel_default_46 = None
        squeeze_140: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        sub_352: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_92 - 1.0
        truediv_93: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_92 / sub_352;  truediv_92 = sub_352 = None
        mul_1172: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_140, truediv_93);  squeeze_140 = truediv_93 = None
        mul_1173: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1172, 0.1);  mul_1172 = None
        mul_1174: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
        add_1528: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1173, mul_1174);  mul_1173 = mul_1174 = None
        unsqueeze_184: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_282, -1)
        unsqueeze_185: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_1175: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1169, unsqueeze_185);  mul_1169 = unsqueeze_185 = None
        unsqueeze_186: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_283, -1);  primals_283 = None
        unsqueeze_187: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_1529: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1175, unsqueeze_187);  mul_1175 = unsqueeze_187 = None
        convert_element_type_140: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1529, torch.float16);  add_1529 = None
        relu_46: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_140);  convert_element_type_140 = None
        convert_element_type_141: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_284, torch.float16);  primals_284 = None
        convolution_46: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_46, convert_element_type_141, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_21: "f16[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1560: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_285, 1)
        convert_element_type_142: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_21, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_142, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_142 = None
        getitem_96: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_47[0]
        getitem_97: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_1561: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
        rsqrt_47: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1561);  add_1561 = None
        sub_359: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_21, getitem_97)
        mul_1195: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_359, rsqrt_47);  sub_359 = None
        squeeze_141: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_142: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_1196: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_1197: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
        add_1562: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1196, mul_1197);  mul_1196 = mul_1197 = None
        sym_numel_default_47: "Sym(75264 * s28)" = torch.ops.aten.sym_numel.default(cat_21)
        truediv_94: "Sym(IntTrueDiv(75264*s28, 384))" = sym_numel_default_47 / 384;  sym_numel_default_47 = None
        squeeze_143: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        sub_360: "Sym(-1.00000000000000 + IntTrueDiv(75264*s28, 384))" = truediv_94 - 1.0
        truediv_95: "Sym(FloatTrueDiv(IntTrueDiv(75264*s28, 384), (IntTrueDiv(75264*s28, 384)) - 1.0))" = truediv_94 / sub_360;  truediv_94 = sub_360 = None
        mul_1198: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_143, truediv_95);  squeeze_143 = truediv_95 = None
        mul_1199: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1198, 0.1);  mul_1198 = None
        mul_1200: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
        add_1563: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1199, mul_1200);  mul_1199 = mul_1200 = None
        unsqueeze_188: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_288, -1)
        unsqueeze_189: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_1201: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1195, unsqueeze_189);  mul_1195 = unsqueeze_189 = None
        unsqueeze_190: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_289, -1);  primals_289 = None
        unsqueeze_191: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_1564: "f32[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1201, unsqueeze_191);  mul_1201 = unsqueeze_191 = None
        convert_element_type_143: "f16[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1564, torch.float16);  add_1564 = None
        relu_47: "f16[s28, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_143);  convert_element_type_143 = None
        convert_element_type_144: "f16[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_290, torch.float16);  primals_290 = None
        convolution_47: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_47, convert_element_type_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1590: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_291, 1)
        convert_element_type_145: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_145, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_145 = None
        getitem_98: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_48[0]
        getitem_99: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_1591: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_48: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1591);  add_1591 = None
        sub_366: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, getitem_99)
        mul_1219: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_366, rsqrt_48);  sub_366 = None
        squeeze_144: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_145: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_1220: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
        mul_1221: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
        add_1592: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1220, mul_1221);  mul_1220 = mul_1221 = None
        sym_numel_default_48: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_47)
        truediv_96: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_48 / 128;  sym_numel_default_48 = None
        squeeze_146: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        sub_367: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_96 - 1.0
        truediv_97: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_96 / sub_367;  truediv_96 = sub_367 = None
        mul_1222: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_146, truediv_97);  squeeze_146 = truediv_97 = None
        mul_1223: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1222, 0.1);  mul_1222 = None
        mul_1224: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
        add_1593: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1223, mul_1224);  mul_1223 = mul_1224 = None
        unsqueeze_192: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_193: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_1225: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1219, unsqueeze_193);  mul_1219 = unsqueeze_193 = None
        unsqueeze_194: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_295, -1);  primals_295 = None
        unsqueeze_195: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_1594: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1225, unsqueeze_195);  mul_1225 = unsqueeze_195 = None
        convert_element_type_146: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1594, torch.float16);  add_1594 = None
        relu_48: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_146);  convert_element_type_146 = None
        convert_element_type_147: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_296, torch.float16);  primals_296 = None
        convolution_48: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_48, convert_element_type_147, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_22: "f16[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1625: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_297, 1)
        convert_element_type_148: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_22, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_148, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_148 = None
        getitem_100: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = var_mean_49[0]
        getitem_101: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_1626: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_49: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1626);  add_1626 = None
        sub_374: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_22, getitem_101)
        mul_1245: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_374, rsqrt_49);  sub_374 = None
        squeeze_147: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_148: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_1246: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
        mul_1247: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
        add_1627: "f32[416][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1246, mul_1247);  mul_1246 = mul_1247 = None
        sym_numel_default_49: "Sym(81536 * s28)" = torch.ops.aten.sym_numel.default(cat_22)
        truediv_98: "Sym(IntTrueDiv(81536*s28, 416))" = sym_numel_default_49 / 416;  sym_numel_default_49 = None
        squeeze_149: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        sub_375: "Sym(-1.00000000000000 + IntTrueDiv(81536*s28, 416))" = truediv_98 - 1.0
        truediv_99: "Sym(FloatTrueDiv(IntTrueDiv(81536*s28, 416), (IntTrueDiv(81536*s28, 416)) - 1.0))" = truediv_98 / sub_375;  truediv_98 = sub_375 = None
        mul_1248: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_149, truediv_99);  squeeze_149 = truediv_99 = None
        mul_1249: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1248, 0.1);  mul_1248 = None
        mul_1250: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
        add_1628: "f32[416][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1249, mul_1250);  mul_1249 = mul_1250 = None
        unsqueeze_196: "f32[416, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_197: "f32[416, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_1251: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1245, unsqueeze_197);  mul_1245 = unsqueeze_197 = None
        unsqueeze_198: "f32[416, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_301, -1);  primals_301 = None
        unsqueeze_199: "f32[416, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_1629: "f32[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1251, unsqueeze_199);  mul_1251 = unsqueeze_199 = None
        convert_element_type_149: "f16[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1629, torch.float16);  add_1629 = None
        relu_49: "f16[s28, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_149);  convert_element_type_149 = None
        convert_element_type_150: "f16[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_302, torch.float16);  primals_302 = None
        convolution_49: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_49, convert_element_type_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1655: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_303, 1)
        convert_element_type_151: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_151, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_151 = None
        getitem_102: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_50[0]
        getitem_103: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        add_1656: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_50: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1656);  add_1656 = None
        sub_381: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, getitem_103)
        mul_1269: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_381, rsqrt_50);  sub_381 = None
        squeeze_150: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_151: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_1270: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
        mul_1271: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
        add_1657: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1270, mul_1271);  mul_1270 = mul_1271 = None
        sym_numel_default_50: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_49)
        truediv_100: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_50 / 128;  sym_numel_default_50 = None
        squeeze_152: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        sub_382: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_100 - 1.0
        truediv_101: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_100 / sub_382;  truediv_100 = sub_382 = None
        mul_1272: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_152, truediv_101);  squeeze_152 = truediv_101 = None
        mul_1273: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1272, 0.1);  mul_1272 = None
        mul_1274: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
        add_1658: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1273, mul_1274);  mul_1273 = mul_1274 = None
        unsqueeze_200: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_306, -1)
        unsqueeze_201: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_1275: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1269, unsqueeze_201);  mul_1269 = unsqueeze_201 = None
        unsqueeze_202: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_307, -1);  primals_307 = None
        unsqueeze_203: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_1659: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1275, unsqueeze_203);  mul_1275 = unsqueeze_203 = None
        convert_element_type_152: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1659, torch.float16);  add_1659 = None
        relu_50: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_152);  convert_element_type_152 = None
        convert_element_type_153: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_308, torch.float16);  primals_308 = None
        convolution_50: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_50, convert_element_type_153, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_23: "f16[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1690: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_309, 1)
        convert_element_type_154: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_23, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_154, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_154 = None
        getitem_104: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_51[0]
        getitem_105: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        add_1691: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_51: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1691);  add_1691 = None
        sub_389: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_23, getitem_105)
        mul_1295: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_389, rsqrt_51);  sub_389 = None
        squeeze_153: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_154: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_1296: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
        mul_1297: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
        add_1692: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1296, mul_1297);  mul_1296 = mul_1297 = None
        sym_numel_default_51: "Sym(87808 * s28)" = torch.ops.aten.sym_numel.default(cat_23)
        truediv_102: "Sym(IntTrueDiv(87808*s28, 448))" = sym_numel_default_51 / 448;  sym_numel_default_51 = None
        squeeze_155: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        sub_390: "Sym(-1.00000000000000 + IntTrueDiv(87808*s28, 448))" = truediv_102 - 1.0
        truediv_103: "Sym(FloatTrueDiv(IntTrueDiv(87808*s28, 448), (IntTrueDiv(87808*s28, 448)) - 1.0))" = truediv_102 / sub_390;  truediv_102 = sub_390 = None
        mul_1298: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_155, truediv_103);  squeeze_155 = truediv_103 = None
        mul_1299: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1298, 0.1);  mul_1298 = None
        mul_1300: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
        add_1693: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1299, mul_1300);  mul_1299 = mul_1300 = None
        unsqueeze_204: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_205: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_1301: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1295, unsqueeze_205);  mul_1295 = unsqueeze_205 = None
        unsqueeze_206: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_207: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_1694: "f32[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1301, unsqueeze_207);  mul_1301 = unsqueeze_207 = None
        convert_element_type_155: "f16[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1694, torch.float16);  add_1694 = None
        relu_51: "f16[s28, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_155);  convert_element_type_155 = None
        convert_element_type_156: "f16[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_314, torch.float16);  primals_314 = None
        convolution_51: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_51, convert_element_type_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1720: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_315, 1)
        convert_element_type_157: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_157, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_157 = None
        getitem_106: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_52[0]
        getitem_107: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        add_1721: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_52: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1721);  add_1721 = None
        sub_396: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_107)
        mul_1319: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_396, rsqrt_52);  sub_396 = None
        squeeze_156: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_157: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_1320: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_1321: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
        add_1722: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1320, mul_1321);  mul_1320 = mul_1321 = None
        sym_numel_default_52: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_51)
        truediv_104: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_52 / 128;  sym_numel_default_52 = None
        squeeze_158: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        sub_397: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_104 - 1.0
        truediv_105: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_104 / sub_397;  truediv_104 = sub_397 = None
        mul_1322: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_158, truediv_105);  squeeze_158 = truediv_105 = None
        mul_1323: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1322, 0.1);  mul_1322 = None
        mul_1324: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
        add_1723: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1323, mul_1324);  mul_1323 = mul_1324 = None
        unsqueeze_208: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_318, -1)
        unsqueeze_209: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        mul_1325: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1319, unsqueeze_209);  mul_1319 = unsqueeze_209 = None
        unsqueeze_210: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_319, -1);  primals_319 = None
        unsqueeze_211: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        add_1724: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1325, unsqueeze_211);  mul_1325 = unsqueeze_211 = None
        convert_element_type_158: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1724, torch.float16);  add_1724 = None
        relu_52: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_158);  convert_element_type_158 = None
        convert_element_type_159: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_320, torch.float16);  primals_320 = None
        convolution_52: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_52, convert_element_type_159, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_24: "f16[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1755: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_321, 1)
        convert_element_type_160: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_24, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_160, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_160 = None
        getitem_108: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_53[0]
        getitem_109: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_53[1];  var_mean_53 = None
        add_1756: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
        rsqrt_53: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1756);  add_1756 = None
        sub_404: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_24, getitem_109)
        mul_1345: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_404, rsqrt_53);  sub_404 = None
        squeeze_159: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_160: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_1346: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
        mul_1347: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
        add_1757: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1346, mul_1347);  mul_1346 = mul_1347 = None
        sym_numel_default_53: "Sym(94080 * s28)" = torch.ops.aten.sym_numel.default(cat_24)
        truediv_106: "Sym(IntTrueDiv(94080*s28, 480))" = sym_numel_default_53 / 480;  sym_numel_default_53 = None
        squeeze_161: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        sub_405: "Sym(-1.00000000000000 + IntTrueDiv(94080*s28, 480))" = truediv_106 - 1.0
        truediv_107: "Sym(FloatTrueDiv(IntTrueDiv(94080*s28, 480), (IntTrueDiv(94080*s28, 480)) - 1.0))" = truediv_106 / sub_405;  truediv_106 = sub_405 = None
        mul_1348: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_161, truediv_107);  squeeze_161 = truediv_107 = None
        mul_1349: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1348, 0.1);  mul_1348 = None
        mul_1350: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
        add_1758: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1349, mul_1350);  mul_1349 = mul_1350 = None
        unsqueeze_212: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_324, -1)
        unsqueeze_213: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_1351: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1345, unsqueeze_213);  mul_1345 = unsqueeze_213 = None
        unsqueeze_214: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_325, -1);  primals_325 = None
        unsqueeze_215: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_1759: "f32[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1351, unsqueeze_215);  mul_1351 = unsqueeze_215 = None
        convert_element_type_161: "f16[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1759, torch.float16);  add_1759 = None
        relu_53: "f16[s28, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_161);  convert_element_type_161 = None
        convert_element_type_162: "f16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_326, torch.float16);  primals_326 = None
        convolution_53: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_53, convert_element_type_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1785: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_327, 1)
        convert_element_type_163: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_53, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_163, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_163 = None
        getitem_110: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_54[0]
        getitem_111: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_54[1];  var_mean_54 = None
        add_1786: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
        rsqrt_54: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1786);  add_1786 = None
        sub_411: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_53, getitem_111)
        mul_1369: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_411, rsqrt_54);  sub_411 = None
        squeeze_162: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_163: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_1370: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
        mul_1371: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
        add_1787: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1370, mul_1371);  mul_1370 = mul_1371 = None
        sym_numel_default_54: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_53)
        truediv_108: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_54 / 128;  sym_numel_default_54 = None
        squeeze_164: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        sub_412: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_108 - 1.0
        truediv_109: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_108 / sub_412;  truediv_108 = sub_412 = None
        mul_1372: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_164, truediv_109);  squeeze_164 = truediv_109 = None
        mul_1373: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1372, 0.1);  mul_1372 = None
        mul_1374: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
        add_1788: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1373, mul_1374);  mul_1373 = mul_1374 = None
        unsqueeze_216: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_330, -1)
        unsqueeze_217: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        mul_1375: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1369, unsqueeze_217);  mul_1369 = unsqueeze_217 = None
        unsqueeze_218: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_331, -1);  primals_331 = None
        unsqueeze_219: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        add_1789: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1375, unsqueeze_219);  mul_1375 = unsqueeze_219 = None
        convert_element_type_164: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1789, torch.float16);  add_1789 = None
        relu_54: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_164);  convert_element_type_164 = None
        convert_element_type_165: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_332, torch.float16);  primals_332 = None
        convolution_54: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_54, convert_element_type_165, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_25: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1820: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_333, 1)
        convert_element_type_166: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_25, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_166, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_166 = None
        getitem_112: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_55[0]
        getitem_113: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_55[1];  var_mean_55 = None
        add_1821: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
        rsqrt_55: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1821);  add_1821 = None
        sub_419: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_25, getitem_113)
        mul_1395: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_419, rsqrt_55);  sub_419 = None
        squeeze_165: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_166: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_1396: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
        mul_1397: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
        add_1822: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1396, mul_1397);  mul_1396 = mul_1397 = None
        sym_numel_default_55: "Sym(100352 * s28)" = torch.ops.aten.sym_numel.default(cat_25)
        truediv_110: "Sym(IntTrueDiv(100352*s28, 512))" = sym_numel_default_55 / 512;  sym_numel_default_55 = None
        squeeze_167: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        sub_420: "Sym(-1.00000000000000 + IntTrueDiv(100352*s28, 512))" = truediv_110 - 1.0
        truediv_111: "Sym(FloatTrueDiv(IntTrueDiv(100352*s28, 512), (IntTrueDiv(100352*s28, 512)) - 1.0))" = truediv_110 / sub_420;  truediv_110 = sub_420 = None
        mul_1398: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_167, truediv_111);  squeeze_167 = truediv_111 = None
        mul_1399: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1398, 0.1);  mul_1398 = None
        mul_1400: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
        add_1823: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1399, mul_1400);  mul_1399 = mul_1400 = None
        unsqueeze_220: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_221: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_1401: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1395, unsqueeze_221);  mul_1395 = unsqueeze_221 = None
        unsqueeze_222: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_223: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_1824: "f32[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1401, unsqueeze_223);  mul_1401 = unsqueeze_223 = None
        convert_element_type_167: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1824, torch.float16);  add_1824 = None
        relu_55: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_167);  convert_element_type_167 = None
        convert_element_type_168: "f16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_338, torch.float16);  primals_338 = None
        convolution_55: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_55, convert_element_type_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1850: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_339, 1)
        convert_element_type_169: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32)
        var_mean_56 = torch.ops.aten.var_mean.correction(convert_element_type_169, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_169 = None
        getitem_114: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_56[0]
        getitem_115: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_56[1];  var_mean_56 = None
        add_1851: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
        rsqrt_56: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1851);  add_1851 = None
        sub_426: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, getitem_115)
        mul_1419: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_426, rsqrt_56);  sub_426 = None
        squeeze_168: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        squeeze_169: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_1420: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
        mul_1421: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
        add_1852: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1420, mul_1421);  mul_1420 = mul_1421 = None
        sym_numel_default_56: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_55)
        truediv_112: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_56 / 128;  sym_numel_default_56 = None
        squeeze_170: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        sub_427: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_112 - 1.0
        truediv_113: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_112 / sub_427;  truediv_112 = sub_427 = None
        mul_1422: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_170, truediv_113);  squeeze_170 = truediv_113 = None
        mul_1423: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1422, 0.1);  mul_1422 = None
        mul_1424: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
        add_1853: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1423, mul_1424);  mul_1423 = mul_1424 = None
        unsqueeze_224: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_342, -1)
        unsqueeze_225: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        mul_1425: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1419, unsqueeze_225);  mul_1419 = unsqueeze_225 = None
        unsqueeze_226: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_343, -1);  primals_343 = None
        unsqueeze_227: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        add_1854: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1425, unsqueeze_227);  mul_1425 = unsqueeze_227 = None
        convert_element_type_170: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1854, torch.float16);  add_1854 = None
        relu_56: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_170);  convert_element_type_170 = None
        convert_element_type_171: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_344, torch.float16);  primals_344 = None
        convolution_56: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_56, convert_element_type_171, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_26: "f16[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1885: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_345, 1)
        convert_element_type_172: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_26, torch.float32)
        var_mean_57 = torch.ops.aten.var_mean.correction(convert_element_type_172, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_172 = None
        getitem_116: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = var_mean_57[0]
        getitem_117: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = var_mean_57[1];  var_mean_57 = None
        add_1886: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
        rsqrt_57: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1886);  add_1886 = None
        sub_434: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_26, getitem_117)
        mul_1445: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_434, rsqrt_57);  sub_434 = None
        squeeze_171: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_172: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_1446: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
        mul_1447: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
        add_1887: "f32[544][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1446, mul_1447);  mul_1446 = mul_1447 = None
        sym_numel_default_57: "Sym(106624 * s28)" = torch.ops.aten.sym_numel.default(cat_26)
        truediv_114: "Sym(IntTrueDiv(106624*s28, 544))" = sym_numel_default_57 / 544;  sym_numel_default_57 = None
        squeeze_173: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        sub_435: "Sym(-1.00000000000000 + IntTrueDiv(106624*s28, 544))" = truediv_114 - 1.0
        truediv_115: "Sym(FloatTrueDiv(IntTrueDiv(106624*s28, 544), (IntTrueDiv(106624*s28, 544)) - 1.0))" = truediv_114 / sub_435;  truediv_114 = sub_435 = None
        mul_1448: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_173, truediv_115);  squeeze_173 = truediv_115 = None
        mul_1449: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1448, 0.1);  mul_1448 = None
        mul_1450: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
        add_1888: "f32[544][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1449, mul_1450);  mul_1449 = mul_1450 = None
        unsqueeze_228: "f32[544, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_348, -1)
        unsqueeze_229: "f32[544, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_1451: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1445, unsqueeze_229);  mul_1445 = unsqueeze_229 = None
        unsqueeze_230: "f32[544, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_349, -1);  primals_349 = None
        unsqueeze_231: "f32[544, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_1889: "f32[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1451, unsqueeze_231);  mul_1451 = unsqueeze_231 = None
        convert_element_type_173: "f16[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1889, torch.float16);  add_1889 = None
        relu_57: "f16[s28, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_173);  convert_element_type_173 = None
        convert_element_type_174: "f16[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_350, torch.float16);  primals_350 = None
        convolution_57: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_57, convert_element_type_174, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1915: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_351, 1)
        convert_element_type_175: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32)
        var_mean_58 = torch.ops.aten.var_mean.correction(convert_element_type_175, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_175 = None
        getitem_118: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_58[0]
        getitem_119: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_58[1];  var_mean_58 = None
        add_1916: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
        rsqrt_58: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1916);  add_1916 = None
        sub_441: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_57, getitem_119)
        mul_1469: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_441, rsqrt_58);  sub_441 = None
        squeeze_174: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        squeeze_175: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_1470: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
        mul_1471: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
        add_1917: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1470, mul_1471);  mul_1470 = mul_1471 = None
        sym_numel_default_58: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_57)
        truediv_116: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_58 / 128;  sym_numel_default_58 = None
        squeeze_176: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        sub_442: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_116 - 1.0
        truediv_117: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_116 / sub_442;  truediv_116 = sub_442 = None
        mul_1472: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_176, truediv_117);  squeeze_176 = truediv_117 = None
        mul_1473: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1472, 0.1);  mul_1472 = None
        mul_1474: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
        add_1918: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1473, mul_1474);  mul_1473 = mul_1474 = None
        unsqueeze_232: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_354, -1)
        unsqueeze_233: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        mul_1475: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1469, unsqueeze_233);  mul_1469 = unsqueeze_233 = None
        unsqueeze_234: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_355, -1);  primals_355 = None
        unsqueeze_235: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        add_1919: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1475, unsqueeze_235);  mul_1475 = unsqueeze_235 = None
        convert_element_type_176: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1919, torch.float16);  add_1919 = None
        relu_58: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_176);  convert_element_type_176 = None
        convert_element_type_177: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_356, torch.float16);  primals_356 = None
        convolution_58: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_58, convert_element_type_177, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_27: "f16[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_1950: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_357, 1)
        convert_element_type_178: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_27, torch.float32)
        var_mean_59 = torch.ops.aten.var_mean.correction(convert_element_type_178, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_178 = None
        getitem_120: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = var_mean_59[0]
        getitem_121: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = var_mean_59[1];  var_mean_59 = None
        add_1951: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
        rsqrt_59: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1951);  add_1951 = None
        sub_449: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_27, getitem_121)
        mul_1495: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_449, rsqrt_59);  sub_449 = None
        squeeze_177: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        squeeze_178: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_1496: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
        mul_1497: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
        add_1952: "f32[576][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1496, mul_1497);  mul_1496 = mul_1497 = None
        sym_numel_default_59: "Sym(112896 * s28)" = torch.ops.aten.sym_numel.default(cat_27)
        truediv_118: "Sym(IntTrueDiv(112896*s28, 576))" = sym_numel_default_59 / 576;  sym_numel_default_59 = None
        squeeze_179: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        sub_450: "Sym(-1.00000000000000 + IntTrueDiv(112896*s28, 576))" = truediv_118 - 1.0
        truediv_119: "Sym(FloatTrueDiv(IntTrueDiv(112896*s28, 576), (IntTrueDiv(112896*s28, 576)) - 1.0))" = truediv_118 / sub_450;  truediv_118 = sub_450 = None
        mul_1498: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_179, truediv_119);  squeeze_179 = truediv_119 = None
        mul_1499: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1498, 0.1);  mul_1498 = None
        mul_1500: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
        add_1953: "f32[576][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1499, mul_1500);  mul_1499 = mul_1500 = None
        unsqueeze_236: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_360, -1)
        unsqueeze_237: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_1501: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1495, unsqueeze_237);  mul_1495 = unsqueeze_237 = None
        unsqueeze_238: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_361, -1);  primals_361 = None
        unsqueeze_239: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_1954: "f32[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1501, unsqueeze_239);  mul_1501 = unsqueeze_239 = None
        convert_element_type_179: "f16[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1954, torch.float16);  add_1954 = None
        relu_59: "f16[s28, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_179);  convert_element_type_179 = None
        convert_element_type_180: "f16[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_362, torch.float16);  primals_362 = None
        convolution_59: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_59, convert_element_type_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_1980: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_363, 1)
        convert_element_type_181: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32)
        var_mean_60 = torch.ops.aten.var_mean.correction(convert_element_type_181, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_181 = None
        getitem_122: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_60[0]
        getitem_123: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_60[1];  var_mean_60 = None
        add_1981: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
        rsqrt_60: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1981);  add_1981 = None
        sub_456: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_59, getitem_123)
        mul_1519: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_456, rsqrt_60);  sub_456 = None
        squeeze_180: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        squeeze_181: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_1520: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
        mul_1521: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
        add_1982: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1520, mul_1521);  mul_1520 = mul_1521 = None
        sym_numel_default_60: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_59)
        truediv_120: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_60 / 128;  sym_numel_default_60 = None
        squeeze_182: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        sub_457: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_120 - 1.0
        truediv_121: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_120 / sub_457;  truediv_120 = sub_457 = None
        mul_1522: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_182, truediv_121);  squeeze_182 = truediv_121 = None
        mul_1523: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1522, 0.1);  mul_1522 = None
        mul_1524: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
        add_1983: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1523, mul_1524);  mul_1523 = mul_1524 = None
        unsqueeze_240: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_366, -1)
        unsqueeze_241: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        mul_1525: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1519, unsqueeze_241);  mul_1519 = unsqueeze_241 = None
        unsqueeze_242: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_367, -1);  primals_367 = None
        unsqueeze_243: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        add_1984: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1525, unsqueeze_243);  mul_1525 = unsqueeze_243 = None
        convert_element_type_182: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1984, torch.float16);  add_1984 = None
        relu_60: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_182);  convert_element_type_182 = None
        convert_element_type_183: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_368, torch.float16);  primals_368 = None
        convolution_60: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_60, convert_element_type_183, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_28: "f16[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2015: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_369, 1)
        convert_element_type_184: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_28, torch.float32)
        var_mean_61 = torch.ops.aten.var_mean.correction(convert_element_type_184, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_184 = None
        getitem_124: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = var_mean_61[0]
        getitem_125: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = var_mean_61[1];  var_mean_61 = None
        add_2016: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
        rsqrt_61: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2016);  add_2016 = None
        sub_464: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_28, getitem_125)
        mul_1545: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_464, rsqrt_61);  sub_464 = None
        squeeze_183: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_184: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_1546: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
        mul_1547: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
        add_2017: "f32[608][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1546, mul_1547);  mul_1546 = mul_1547 = None
        sym_numel_default_61: "Sym(119168 * s28)" = torch.ops.aten.sym_numel.default(cat_28)
        truediv_122: "Sym(IntTrueDiv(119168*s28, 608))" = sym_numel_default_61 / 608;  sym_numel_default_61 = None
        squeeze_185: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        sub_465: "Sym(-1.00000000000000 + IntTrueDiv(119168*s28, 608))" = truediv_122 - 1.0
        truediv_123: "Sym(FloatTrueDiv(IntTrueDiv(119168*s28, 608), (IntTrueDiv(119168*s28, 608)) - 1.0))" = truediv_122 / sub_465;  truediv_122 = sub_465 = None
        mul_1548: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_185, truediv_123);  squeeze_185 = truediv_123 = None
        mul_1549: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1548, 0.1);  mul_1548 = None
        mul_1550: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
        add_2018: "f32[608][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1549, mul_1550);  mul_1549 = mul_1550 = None
        unsqueeze_244: "f32[608, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_372, -1)
        unsqueeze_245: "f32[608, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_1551: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1545, unsqueeze_245);  mul_1545 = unsqueeze_245 = None
        unsqueeze_246: "f32[608, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_373, -1);  primals_373 = None
        unsqueeze_247: "f32[608, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_2019: "f32[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1551, unsqueeze_247);  mul_1551 = unsqueeze_247 = None
        convert_element_type_185: "f16[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2019, torch.float16);  add_2019 = None
        relu_61: "f16[s28, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_185);  convert_element_type_185 = None
        convert_element_type_186: "f16[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_374, torch.float16);  primals_374 = None
        convolution_61: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_61, convert_element_type_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2045: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_375, 1)
        convert_element_type_187: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32)
        var_mean_62 = torch.ops.aten.var_mean.correction(convert_element_type_187, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_187 = None
        getitem_126: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_62[0]
        getitem_127: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_62[1];  var_mean_62 = None
        add_2046: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
        rsqrt_62: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2046);  add_2046 = None
        sub_471: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, getitem_127)
        mul_1569: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_471, rsqrt_62);  sub_471 = None
        squeeze_186: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        squeeze_187: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
        mul_1570: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
        mul_1571: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
        add_2047: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1570, mul_1571);  mul_1570 = mul_1571 = None
        sym_numel_default_62: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_61)
        truediv_124: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_62 / 128;  sym_numel_default_62 = None
        squeeze_188: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        sub_472: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_124 - 1.0
        truediv_125: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_124 / sub_472;  truediv_124 = sub_472 = None
        mul_1572: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_188, truediv_125);  squeeze_188 = truediv_125 = None
        mul_1573: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1572, 0.1);  mul_1572 = None
        mul_1574: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
        add_2048: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1573, mul_1574);  mul_1573 = mul_1574 = None
        unsqueeze_248: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_378, -1)
        unsqueeze_249: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        mul_1575: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1569, unsqueeze_249);  mul_1569 = unsqueeze_249 = None
        unsqueeze_250: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_379, -1);  primals_379 = None
        unsqueeze_251: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        add_2049: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1575, unsqueeze_251);  mul_1575 = unsqueeze_251 = None
        convert_element_type_188: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2049, torch.float16);  add_2049 = None
        relu_62: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_188);  convert_element_type_188 = None
        convert_element_type_189: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_380, torch.float16);  primals_380 = None
        convolution_62: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_62, convert_element_type_189, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_29: "f16[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2080: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_381, 1)
        convert_element_type_190: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_29, torch.float32)
        var_mean_63 = torch.ops.aten.var_mean.correction(convert_element_type_190, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_190 = None
        getitem_128: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_63[0]
        getitem_129: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_63[1];  var_mean_63 = None
        add_2081: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
        rsqrt_63: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2081);  add_2081 = None
        sub_479: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_29, getitem_129)
        mul_1595: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_479, rsqrt_63);  sub_479 = None
        squeeze_189: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        squeeze_190: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
        mul_1596: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
        mul_1597: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
        add_2082: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1596, mul_1597);  mul_1596 = mul_1597 = None
        sym_numel_default_63: "Sym(125440 * s28)" = torch.ops.aten.sym_numel.default(cat_29)
        truediv_126: "Sym(IntTrueDiv(125440*s28, 640))" = sym_numel_default_63 / 640;  sym_numel_default_63 = None
        squeeze_191: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
        sub_480: "Sym(-1.00000000000000 + IntTrueDiv(125440*s28, 640))" = truediv_126 - 1.0
        truediv_127: "Sym(FloatTrueDiv(IntTrueDiv(125440*s28, 640), (IntTrueDiv(125440*s28, 640)) - 1.0))" = truediv_126 / sub_480;  truediv_126 = sub_480 = None
        mul_1598: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_191, truediv_127);  squeeze_191 = truediv_127 = None
        mul_1599: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1598, 0.1);  mul_1598 = None
        mul_1600: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
        add_2083: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1599, mul_1600);  mul_1599 = mul_1600 = None
        unsqueeze_252: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_384, -1)
        unsqueeze_253: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_1601: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1595, unsqueeze_253);  mul_1595 = unsqueeze_253 = None
        unsqueeze_254: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_385, -1);  primals_385 = None
        unsqueeze_255: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_2084: "f32[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1601, unsqueeze_255);  mul_1601 = unsqueeze_255 = None
        convert_element_type_191: "f16[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2084, torch.float16);  add_2084 = None
        relu_63: "f16[s28, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_191);  convert_element_type_191 = None
        convert_element_type_192: "f16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_386, torch.float16);  primals_386 = None
        convolution_63: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_63, convert_element_type_192, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2110: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_387, 1)
        convert_element_type_193: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32)
        var_mean_64 = torch.ops.aten.var_mean.correction(convert_element_type_193, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_193 = None
        getitem_130: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_64[0]
        getitem_131: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_64[1];  var_mean_64 = None
        add_2111: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05)
        rsqrt_64: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2111);  add_2111 = None
        sub_486: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_63, getitem_131)
        mul_1619: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_486, rsqrt_64);  sub_486 = None
        squeeze_192: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
        squeeze_193: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
        mul_1620: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
        mul_1621: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
        add_2112: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1620, mul_1621);  mul_1620 = mul_1621 = None
        sym_numel_default_64: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_63)
        truediv_128: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_64 / 128;  sym_numel_default_64 = None
        squeeze_194: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
        sub_487: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_128 - 1.0
        truediv_129: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_128 / sub_487;  truediv_128 = sub_487 = None
        mul_1622: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_194, truediv_129);  squeeze_194 = truediv_129 = None
        mul_1623: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1622, 0.1);  mul_1622 = None
        mul_1624: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
        add_2113: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1623, mul_1624);  mul_1623 = mul_1624 = None
        unsqueeze_256: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_390, -1)
        unsqueeze_257: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        mul_1625: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1619, unsqueeze_257);  mul_1619 = unsqueeze_257 = None
        unsqueeze_258: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_391, -1);  primals_391 = None
        unsqueeze_259: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        add_2114: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1625, unsqueeze_259);  mul_1625 = unsqueeze_259 = None
        convert_element_type_194: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2114, torch.float16);  add_2114 = None
        relu_64: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_194);  convert_element_type_194 = None
        convert_element_type_195: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_392, torch.float16);  primals_392 = None
        convolution_64: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_64, convert_element_type_195, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_30: "f16[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2145: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_393, 1)
        convert_element_type_196: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_30, torch.float32)
        var_mean_65 = torch.ops.aten.var_mean.correction(convert_element_type_196, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_196 = None
        getitem_132: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_65[0]
        getitem_133: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_65[1];  var_mean_65 = None
        add_2146: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
        rsqrt_65: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2146);  add_2146 = None
        sub_494: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_30, getitem_133)
        mul_1645: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_494, rsqrt_65);  sub_494 = None
        squeeze_195: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        squeeze_196: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
        mul_1646: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
        mul_1647: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
        add_2147: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1646, mul_1647);  mul_1646 = mul_1647 = None
        sym_numel_default_65: "Sym(131712 * s28)" = torch.ops.aten.sym_numel.default(cat_30)
        truediv_130: "Sym(IntTrueDiv(131712*s28, 672))" = sym_numel_default_65 / 672;  sym_numel_default_65 = None
        squeeze_197: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
        sub_495: "Sym(-1.00000000000000 + IntTrueDiv(131712*s28, 672))" = truediv_130 - 1.0
        truediv_131: "Sym(FloatTrueDiv(IntTrueDiv(131712*s28, 672), (IntTrueDiv(131712*s28, 672)) - 1.0))" = truediv_130 / sub_495;  truediv_130 = sub_495 = None
        mul_1648: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_197, truediv_131);  squeeze_197 = truediv_131 = None
        mul_1649: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1648, 0.1);  mul_1648 = None
        mul_1650: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
        add_2148: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1649, mul_1650);  mul_1649 = mul_1650 = None
        unsqueeze_260: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_396, -1)
        unsqueeze_261: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_1651: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1645, unsqueeze_261);  mul_1645 = unsqueeze_261 = None
        unsqueeze_262: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_397, -1);  primals_397 = None
        unsqueeze_263: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_2149: "f32[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1651, unsqueeze_263);  mul_1651 = unsqueeze_263 = None
        convert_element_type_197: "f16[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2149, torch.float16);  add_2149 = None
        relu_65: "f16[s28, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_197);  convert_element_type_197 = None
        convert_element_type_198: "f16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_398, torch.float16);  primals_398 = None
        convolution_65: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_65, convert_element_type_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2175: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_399, 1)
        convert_element_type_199: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_65, torch.float32)
        var_mean_66 = torch.ops.aten.var_mean.correction(convert_element_type_199, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_199 = None
        getitem_134: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_66[0]
        getitem_135: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_66[1];  var_mean_66 = None
        add_2176: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
        rsqrt_66: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2176);  add_2176 = None
        sub_501: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_65, getitem_135)
        mul_1669: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_501, rsqrt_66);  sub_501 = None
        squeeze_198: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
        squeeze_199: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
        mul_1670: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
        mul_1671: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
        add_2177: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1670, mul_1671);  mul_1670 = mul_1671 = None
        sym_numel_default_66: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_65)
        truediv_132: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_66 / 128;  sym_numel_default_66 = None
        squeeze_200: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
        sub_502: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_132 - 1.0
        truediv_133: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_132 / sub_502;  truediv_132 = sub_502 = None
        mul_1672: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_200, truediv_133);  squeeze_200 = truediv_133 = None
        mul_1673: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1672, 0.1);  mul_1672 = None
        mul_1674: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
        add_2178: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1673, mul_1674);  mul_1673 = mul_1674 = None
        unsqueeze_264: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_402, -1)
        unsqueeze_265: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        mul_1675: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1669, unsqueeze_265);  mul_1669 = unsqueeze_265 = None
        unsqueeze_266: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_403, -1);  primals_403 = None
        unsqueeze_267: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        add_2179: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1675, unsqueeze_267);  mul_1675 = unsqueeze_267 = None
        convert_element_type_200: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2179, torch.float16);  add_2179 = None
        relu_66: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_200);  convert_element_type_200 = None
        convert_element_type_201: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_404, torch.float16);  primals_404 = None
        convolution_66: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_66, convert_element_type_201, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_31: "f16[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2210: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_405, 1)
        convert_element_type_202: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_31, torch.float32)
        var_mean_67 = torch.ops.aten.var_mean.correction(convert_element_type_202, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_202 = None
        getitem_136: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = var_mean_67[0]
        getitem_137: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = var_mean_67[1];  var_mean_67 = None
        add_2211: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
        rsqrt_67: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2211);  add_2211 = None
        sub_509: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_31, getitem_137)
        mul_1695: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_509, rsqrt_67);  sub_509 = None
        squeeze_201: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        squeeze_202: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
        mul_1696: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
        mul_1697: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
        add_2212: "f32[704][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1696, mul_1697);  mul_1696 = mul_1697 = None
        sym_numel_default_67: "Sym(137984 * s28)" = torch.ops.aten.sym_numel.default(cat_31)
        truediv_134: "Sym(IntTrueDiv(137984*s28, 704))" = sym_numel_default_67 / 704;  sym_numel_default_67 = None
        squeeze_203: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
        sub_510: "Sym(-1.00000000000000 + IntTrueDiv(137984*s28, 704))" = truediv_134 - 1.0
        truediv_135: "Sym(FloatTrueDiv(IntTrueDiv(137984*s28, 704), (IntTrueDiv(137984*s28, 704)) - 1.0))" = truediv_134 / sub_510;  truediv_134 = sub_510 = None
        mul_1698: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_203, truediv_135);  squeeze_203 = truediv_135 = None
        mul_1699: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1698, 0.1);  mul_1698 = None
        mul_1700: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
        add_2213: "f32[704][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1699, mul_1700);  mul_1699 = mul_1700 = None
        unsqueeze_268: "f32[704, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_408, -1)
        unsqueeze_269: "f32[704, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_1701: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1695, unsqueeze_269);  mul_1695 = unsqueeze_269 = None
        unsqueeze_270: "f32[704, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_409, -1);  primals_409 = None
        unsqueeze_271: "f32[704, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_2214: "f32[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1701, unsqueeze_271);  mul_1701 = unsqueeze_271 = None
        convert_element_type_203: "f16[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2214, torch.float16);  add_2214 = None
        relu_67: "f16[s28, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_203);  convert_element_type_203 = None
        convert_element_type_204: "f16[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_410, torch.float16);  primals_410 = None
        convolution_67: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_67, convert_element_type_204, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2240: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_411, 1)
        convert_element_type_205: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32)
        var_mean_68 = torch.ops.aten.var_mean.correction(convert_element_type_205, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_205 = None
        getitem_138: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_68[0]
        getitem_139: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_68[1];  var_mean_68 = None
        add_2241: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
        rsqrt_68: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2241);  add_2241 = None
        sub_516: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_67, getitem_139)
        mul_1719: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_516, rsqrt_68);  sub_516 = None
        squeeze_204: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        squeeze_205: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
        mul_1720: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
        mul_1721: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
        add_2242: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1720, mul_1721);  mul_1720 = mul_1721 = None
        sym_numel_default_68: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_67)
        truediv_136: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_68 / 128;  sym_numel_default_68 = None
        squeeze_206: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
        sub_517: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_136 - 1.0
        truediv_137: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_136 / sub_517;  truediv_136 = sub_517 = None
        mul_1722: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_206, truediv_137);  squeeze_206 = truediv_137 = None
        mul_1723: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1722, 0.1);  mul_1722 = None
        mul_1724: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
        add_2243: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1723, mul_1724);  mul_1723 = mul_1724 = None
        unsqueeze_272: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_414, -1)
        unsqueeze_273: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        mul_1725: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1719, unsqueeze_273);  mul_1719 = unsqueeze_273 = None
        unsqueeze_274: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_415, -1);  primals_415 = None
        unsqueeze_275: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        add_2244: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1725, unsqueeze_275);  mul_1725 = unsqueeze_275 = None
        convert_element_type_206: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2244, torch.float16);  add_2244 = None
        relu_68: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_206);  convert_element_type_206 = None
        convert_element_type_207: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_416, torch.float16);  primals_416 = None
        convolution_68: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_68, convert_element_type_207, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_32: "f16[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2275: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_417, 1)
        convert_element_type_208: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_32, torch.float32)
        var_mean_69 = torch.ops.aten.var_mean.correction(convert_element_type_208, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_208 = None
        getitem_140: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = var_mean_69[0]
        getitem_141: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = var_mean_69[1];  var_mean_69 = None
        add_2276: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
        rsqrt_69: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2276);  add_2276 = None
        sub_524: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_32, getitem_141)
        mul_1745: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_524, rsqrt_69);  sub_524 = None
        squeeze_207: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
        squeeze_208: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
        mul_1746: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
        mul_1747: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
        add_2277: "f32[736][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1746, mul_1747);  mul_1746 = mul_1747 = None
        sym_numel_default_69: "Sym(144256 * s28)" = torch.ops.aten.sym_numel.default(cat_32)
        truediv_138: "Sym(IntTrueDiv(144256*s28, 736))" = sym_numel_default_69 / 736;  sym_numel_default_69 = None
        squeeze_209: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
        sub_525: "Sym(-1.00000000000000 + IntTrueDiv(144256*s28, 736))" = truediv_138 - 1.0
        truediv_139: "Sym(FloatTrueDiv(IntTrueDiv(144256*s28, 736), (IntTrueDiv(144256*s28, 736)) - 1.0))" = truediv_138 / sub_525;  truediv_138 = sub_525 = None
        mul_1748: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_209, truediv_139);  squeeze_209 = truediv_139 = None
        mul_1749: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1748, 0.1);  mul_1748 = None
        mul_1750: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
        add_2278: "f32[736][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1749, mul_1750);  mul_1749 = mul_1750 = None
        unsqueeze_276: "f32[736, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_420, -1)
        unsqueeze_277: "f32[736, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_1751: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1745, unsqueeze_277);  mul_1745 = unsqueeze_277 = None
        unsqueeze_278: "f32[736, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_421, -1);  primals_421 = None
        unsqueeze_279: "f32[736, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_2279: "f32[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1751, unsqueeze_279);  mul_1751 = unsqueeze_279 = None
        convert_element_type_209: "f16[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2279, torch.float16);  add_2279 = None
        relu_69: "f16[s28, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_209);  convert_element_type_209 = None
        convert_element_type_210: "f16[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_422, torch.float16);  primals_422 = None
        convolution_69: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_69, convert_element_type_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2305: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_423, 1)
        convert_element_type_211: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32)
        var_mean_70 = torch.ops.aten.var_mean.correction(convert_element_type_211, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_211 = None
        getitem_142: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_70[0]
        getitem_143: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_70[1];  var_mean_70 = None
        add_2306: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
        rsqrt_70: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2306);  add_2306 = None
        sub_531: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_69, getitem_143)
        mul_1769: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_531, rsqrt_70);  sub_531 = None
        squeeze_210: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
        squeeze_211: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
        mul_1770: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
        mul_1771: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
        add_2307: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1770, mul_1771);  mul_1770 = mul_1771 = None
        sym_numel_default_70: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_69)
        truediv_140: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_70 / 128;  sym_numel_default_70 = None
        squeeze_212: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
        sub_532: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_140 - 1.0
        truediv_141: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_140 / sub_532;  truediv_140 = sub_532 = None
        mul_1772: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_212, truediv_141);  squeeze_212 = truediv_141 = None
        mul_1773: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1772, 0.1);  mul_1772 = None
        mul_1774: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
        add_2308: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1773, mul_1774);  mul_1773 = mul_1774 = None
        unsqueeze_280: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_426, -1)
        unsqueeze_281: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        mul_1775: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1769, unsqueeze_281);  mul_1769 = unsqueeze_281 = None
        unsqueeze_282: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_427, -1);  primals_427 = None
        unsqueeze_283: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        add_2309: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1775, unsqueeze_283);  mul_1775 = unsqueeze_283 = None
        convert_element_type_212: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2309, torch.float16);  add_2309 = None
        relu_70: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_212);  convert_element_type_212 = None
        convert_element_type_213: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_428, torch.float16);  primals_428 = None
        convolution_70: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_70, convert_element_type_213, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_33: "f16[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2340: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_429, 1)
        convert_element_type_214: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_33, torch.float32)
        var_mean_71 = torch.ops.aten.var_mean.correction(convert_element_type_214, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_214 = None
        getitem_144: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_71[0]
        getitem_145: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_71[1];  var_mean_71 = None
        add_2341: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
        rsqrt_71: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2341);  add_2341 = None
        sub_539: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_33, getitem_145)
        mul_1795: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_539, rsqrt_71);  sub_539 = None
        squeeze_213: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        squeeze_214: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
        mul_1796: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
        mul_1797: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
        add_2342: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1796, mul_1797);  mul_1796 = mul_1797 = None
        sym_numel_default_71: "Sym(150528 * s28)" = torch.ops.aten.sym_numel.default(cat_33)
        truediv_142: "Sym(IntTrueDiv(150528*s28, 768))" = sym_numel_default_71 / 768;  sym_numel_default_71 = None
        squeeze_215: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
        sub_540: "Sym(-1.00000000000000 + IntTrueDiv(150528*s28, 768))" = truediv_142 - 1.0
        truediv_143: "Sym(FloatTrueDiv(IntTrueDiv(150528*s28, 768), (IntTrueDiv(150528*s28, 768)) - 1.0))" = truediv_142 / sub_540;  truediv_142 = sub_540 = None
        mul_1798: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_215, truediv_143);  squeeze_215 = truediv_143 = None
        mul_1799: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1798, 0.1);  mul_1798 = None
        mul_1800: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
        add_2343: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1799, mul_1800);  mul_1799 = mul_1800 = None
        unsqueeze_284: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_432, -1)
        unsqueeze_285: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_1801: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1795, unsqueeze_285);  mul_1795 = unsqueeze_285 = None
        unsqueeze_286: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_433, -1);  primals_433 = None
        unsqueeze_287: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_2344: "f32[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1801, unsqueeze_287);  mul_1801 = unsqueeze_287 = None
        convert_element_type_215: "f16[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2344, torch.float16);  add_2344 = None
        relu_71: "f16[s28, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_215);  convert_element_type_215 = None
        convert_element_type_216: "f16[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_434, torch.float16);  primals_434 = None
        convolution_71: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_71, convert_element_type_216, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2370: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_435, 1)
        convert_element_type_217: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_71, torch.float32)
        var_mean_72 = torch.ops.aten.var_mean.correction(convert_element_type_217, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_217 = None
        getitem_146: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_72[0]
        getitem_147: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_72[1];  var_mean_72 = None
        add_2371: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
        rsqrt_72: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2371);  add_2371 = None
        sub_546: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_71, getitem_147)
        mul_1819: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_546, rsqrt_72);  sub_546 = None
        squeeze_216: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
        squeeze_217: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
        mul_1820: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
        mul_1821: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
        add_2372: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1820, mul_1821);  mul_1820 = mul_1821 = None
        sym_numel_default_72: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_71)
        truediv_144: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_72 / 128;  sym_numel_default_72 = None
        squeeze_218: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
        sub_547: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_144 - 1.0
        truediv_145: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_144 / sub_547;  truediv_144 = sub_547 = None
        mul_1822: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_218, truediv_145);  squeeze_218 = truediv_145 = None
        mul_1823: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1822, 0.1);  mul_1822 = None
        mul_1824: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_437, 0.9)
        add_2373: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1823, mul_1824);  mul_1823 = mul_1824 = None
        unsqueeze_288: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_438, -1)
        unsqueeze_289: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        mul_1825: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1819, unsqueeze_289);  mul_1819 = unsqueeze_289 = None
        unsqueeze_290: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_439, -1);  primals_439 = None
        unsqueeze_291: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        add_2374: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1825, unsqueeze_291);  mul_1825 = unsqueeze_291 = None
        convert_element_type_218: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2374, torch.float16);  add_2374 = None
        relu_72: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_218);  convert_element_type_218 = None
        convert_element_type_219: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_440, torch.float16);  primals_440 = None
        convolution_72: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_72, convert_element_type_219, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_34: "f16[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2405: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_441, 1)
        convert_element_type_220: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_34, torch.float32)
        var_mean_73 = torch.ops.aten.var_mean.correction(convert_element_type_220, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_220 = None
        getitem_148: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = var_mean_73[0]
        getitem_149: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = var_mean_73[1];  var_mean_73 = None
        add_2406: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
        rsqrt_73: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2406);  add_2406 = None
        sub_554: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_34, getitem_149)
        mul_1845: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_554, rsqrt_73);  sub_554 = None
        squeeze_219: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
        squeeze_220: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
        mul_1846: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
        mul_1847: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
        add_2407: "f32[800][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1846, mul_1847);  mul_1846 = mul_1847 = None
        sym_numel_default_73: "Sym(156800 * s28)" = torch.ops.aten.sym_numel.default(cat_34)
        truediv_146: "Sym(IntTrueDiv(156800*s28, 800))" = sym_numel_default_73 / 800;  sym_numel_default_73 = None
        squeeze_221: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
        sub_555: "Sym(-1.00000000000000 + IntTrueDiv(156800*s28, 800))" = truediv_146 - 1.0
        truediv_147: "Sym(FloatTrueDiv(IntTrueDiv(156800*s28, 800), (IntTrueDiv(156800*s28, 800)) - 1.0))" = truediv_146 / sub_555;  truediv_146 = sub_555 = None
        mul_1848: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_221, truediv_147);  squeeze_221 = truediv_147 = None
        mul_1849: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1848, 0.1);  mul_1848 = None
        mul_1850: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
        add_2408: "f32[800][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1849, mul_1850);  mul_1849 = mul_1850 = None
        unsqueeze_292: "f32[800, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_444, -1)
        unsqueeze_293: "f32[800, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_1851: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1845, unsqueeze_293);  mul_1845 = unsqueeze_293 = None
        unsqueeze_294: "f32[800, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_445, -1);  primals_445 = None
        unsqueeze_295: "f32[800, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_2409: "f32[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1851, unsqueeze_295);  mul_1851 = unsqueeze_295 = None
        convert_element_type_221: "f16[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2409, torch.float16);  add_2409 = None
        relu_73: "f16[s28, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_221);  convert_element_type_221 = None
        convert_element_type_222: "f16[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_446, torch.float16);  primals_446 = None
        convolution_73: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_73, convert_element_type_222, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2435: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_447, 1)
        convert_element_type_223: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_73, torch.float32)
        var_mean_74 = torch.ops.aten.var_mean.correction(convert_element_type_223, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_223 = None
        getitem_150: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_74[0]
        getitem_151: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_74[1];  var_mean_74 = None
        add_2436: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
        rsqrt_74: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2436);  add_2436 = None
        sub_561: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_73, getitem_151)
        mul_1869: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_561, rsqrt_74);  sub_561 = None
        squeeze_222: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
        squeeze_223: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
        mul_1870: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
        mul_1871: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
        add_2437: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1870, mul_1871);  mul_1870 = mul_1871 = None
        sym_numel_default_74: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_73)
        truediv_148: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_74 / 128;  sym_numel_default_74 = None
        squeeze_224: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
        sub_562: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_148 - 1.0
        truediv_149: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_148 / sub_562;  truediv_148 = sub_562 = None
        mul_1872: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_224, truediv_149);  squeeze_224 = truediv_149 = None
        mul_1873: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1872, 0.1);  mul_1872 = None
        mul_1874: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
        add_2438: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1873, mul_1874);  mul_1873 = mul_1874 = None
        unsqueeze_296: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_450, -1)
        unsqueeze_297: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        mul_1875: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1869, unsqueeze_297);  mul_1869 = unsqueeze_297 = None
        unsqueeze_298: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_451, -1);  primals_451 = None
        unsqueeze_299: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        add_2439: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1875, unsqueeze_299);  mul_1875 = unsqueeze_299 = None
        convert_element_type_224: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2439, torch.float16);  add_2439 = None
        relu_74: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_224);  convert_element_type_224 = None
        convert_element_type_225: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_452, torch.float16);  primals_452 = None
        convolution_74: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_74, convert_element_type_225, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_35: "f16[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2470: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_453, 1)
        convert_element_type_226: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_35, torch.float32)
        var_mean_75 = torch.ops.aten.var_mean.correction(convert_element_type_226, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_226 = None
        getitem_152: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = var_mean_75[0]
        getitem_153: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = var_mean_75[1];  var_mean_75 = None
        add_2471: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
        rsqrt_75: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2471);  add_2471 = None
        sub_569: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_35, getitem_153)
        mul_1895: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_569, rsqrt_75);  sub_569 = None
        squeeze_225: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        squeeze_226: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
        mul_1896: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
        mul_1897: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
        add_2472: "f32[832][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1896, mul_1897);  mul_1896 = mul_1897 = None
        sym_numel_default_75: "Sym(163072 * s28)" = torch.ops.aten.sym_numel.default(cat_35)
        truediv_150: "Sym(IntTrueDiv(163072*s28, 832))" = sym_numel_default_75 / 832;  sym_numel_default_75 = None
        squeeze_227: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
        sub_570: "Sym(-1.00000000000000 + IntTrueDiv(163072*s28, 832))" = truediv_150 - 1.0
        truediv_151: "Sym(FloatTrueDiv(IntTrueDiv(163072*s28, 832), (IntTrueDiv(163072*s28, 832)) - 1.0))" = truediv_150 / sub_570;  truediv_150 = sub_570 = None
        mul_1898: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_227, truediv_151);  squeeze_227 = truediv_151 = None
        mul_1899: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1898, 0.1);  mul_1898 = None
        mul_1900: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
        add_2473: "f32[832][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1899, mul_1900);  mul_1899 = mul_1900 = None
        unsqueeze_300: "f32[832, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_456, -1)
        unsqueeze_301: "f32[832, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_1901: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1895, unsqueeze_301);  mul_1895 = unsqueeze_301 = None
        unsqueeze_302: "f32[832, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_457, -1);  primals_457 = None
        unsqueeze_303: "f32[832, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_2474: "f32[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1901, unsqueeze_303);  mul_1901 = unsqueeze_303 = None
        convert_element_type_227: "f16[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2474, torch.float16);  add_2474 = None
        relu_75: "f16[s28, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_227);  convert_element_type_227 = None
        convert_element_type_228: "f16[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_458, torch.float16);  primals_458 = None
        convolution_75: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_75, convert_element_type_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2500: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_459, 1)
        convert_element_type_229: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32)
        var_mean_76 = torch.ops.aten.var_mean.correction(convert_element_type_229, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_229 = None
        getitem_154: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_76[0]
        getitem_155: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_76[1];  var_mean_76 = None
        add_2501: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
        rsqrt_76: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2501);  add_2501 = None
        sub_576: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_75, getitem_155)
        mul_1919: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_576, rsqrt_76);  sub_576 = None
        squeeze_228: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
        squeeze_229: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
        mul_1920: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
        mul_1921: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
        add_2502: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1920, mul_1921);  mul_1920 = mul_1921 = None
        sym_numel_default_76: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_75)
        truediv_152: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_76 / 128;  sym_numel_default_76 = None
        squeeze_230: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
        sub_577: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_152 - 1.0
        truediv_153: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_152 / sub_577;  truediv_152 = sub_577 = None
        mul_1922: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_230, truediv_153);  squeeze_230 = truediv_153 = None
        mul_1923: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1922, 0.1);  mul_1922 = None
        mul_1924: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_461, 0.9)
        add_2503: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1923, mul_1924);  mul_1923 = mul_1924 = None
        unsqueeze_304: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_462, -1)
        unsqueeze_305: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        mul_1925: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1919, unsqueeze_305);  mul_1919 = unsqueeze_305 = None
        unsqueeze_306: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_463, -1);  primals_463 = None
        unsqueeze_307: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        add_2504: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1925, unsqueeze_307);  mul_1925 = unsqueeze_307 = None
        convert_element_type_230: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2504, torch.float16);  add_2504 = None
        relu_76: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_230);  convert_element_type_230 = None
        convert_element_type_231: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_464, torch.float16);  primals_464 = None
        convolution_76: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_76, convert_element_type_231, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_36: "f16[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2535: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_465, 1)
        convert_element_type_232: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_36, torch.float32)
        var_mean_77 = torch.ops.aten.var_mean.correction(convert_element_type_232, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_232 = None
        getitem_156: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = var_mean_77[0]
        getitem_157: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = var_mean_77[1];  var_mean_77 = None
        add_2536: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
        rsqrt_77: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2536);  add_2536 = None
        sub_584: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_36, getitem_157)
        mul_1945: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_584, rsqrt_77);  sub_584 = None
        squeeze_231: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        squeeze_232: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
        mul_1946: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
        mul_1947: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
        add_2537: "f32[864][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1946, mul_1947);  mul_1946 = mul_1947 = None
        sym_numel_default_77: "Sym(169344 * s28)" = torch.ops.aten.sym_numel.default(cat_36)
        truediv_154: "Sym(IntTrueDiv(169344*s28, 864))" = sym_numel_default_77 / 864;  sym_numel_default_77 = None
        squeeze_233: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
        sub_585: "Sym(-1.00000000000000 + IntTrueDiv(169344*s28, 864))" = truediv_154 - 1.0
        truediv_155: "Sym(FloatTrueDiv(IntTrueDiv(169344*s28, 864), (IntTrueDiv(169344*s28, 864)) - 1.0))" = truediv_154 / sub_585;  truediv_154 = sub_585 = None
        mul_1948: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_233, truediv_155);  squeeze_233 = truediv_155 = None
        mul_1949: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1948, 0.1);  mul_1948 = None
        mul_1950: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_467, 0.9)
        add_2538: "f32[864][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1949, mul_1950);  mul_1949 = mul_1950 = None
        unsqueeze_308: "f32[864, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_468, -1)
        unsqueeze_309: "f32[864, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_1951: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1945, unsqueeze_309);  mul_1945 = unsqueeze_309 = None
        unsqueeze_310: "f32[864, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_469, -1);  primals_469 = None
        unsqueeze_311: "f32[864, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_2539: "f32[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1951, unsqueeze_311);  mul_1951 = unsqueeze_311 = None
        convert_element_type_233: "f16[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2539, torch.float16);  add_2539 = None
        relu_77: "f16[s28, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_233);  convert_element_type_233 = None
        convert_element_type_234: "f16[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_470, torch.float16);  primals_470 = None
        convolution_77: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_77, convert_element_type_234, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2565: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_471, 1)
        convert_element_type_235: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32)
        var_mean_78 = torch.ops.aten.var_mean.correction(convert_element_type_235, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_235 = None
        getitem_158: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_78[0]
        getitem_159: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_78[1];  var_mean_78 = None
        add_2566: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
        rsqrt_78: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2566);  add_2566 = None
        sub_591: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_77, getitem_159)
        mul_1969: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_591, rsqrt_78);  sub_591 = None
        squeeze_234: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
        squeeze_235: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
        mul_1970: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
        mul_1971: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
        add_2567: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1970, mul_1971);  mul_1970 = mul_1971 = None
        sym_numel_default_78: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_77)
        truediv_156: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_78 / 128;  sym_numel_default_78 = None
        squeeze_236: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
        sub_592: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_156 - 1.0
        truediv_157: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_156 / sub_592;  truediv_156 = sub_592 = None
        mul_1972: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_236, truediv_157);  squeeze_236 = truediv_157 = None
        mul_1973: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1972, 0.1);  mul_1972 = None
        mul_1974: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_473, 0.9)
        add_2568: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1973, mul_1974);  mul_1973 = mul_1974 = None
        unsqueeze_312: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_474, -1)
        unsqueeze_313: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        mul_1975: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1969, unsqueeze_313);  mul_1969 = unsqueeze_313 = None
        unsqueeze_314: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_475, -1);  primals_475 = None
        unsqueeze_315: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        add_2569: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1975, unsqueeze_315);  mul_1975 = unsqueeze_315 = None
        convert_element_type_236: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2569, torch.float16);  add_2569 = None
        relu_78: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_236);  convert_element_type_236 = None
        convert_element_type_237: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_476, torch.float16);  primals_476 = None
        convolution_78: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_78, convert_element_type_237, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_37: "f16[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2600: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_477, 1)
        convert_element_type_238: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_37, torch.float32)
        var_mean_79 = torch.ops.aten.var_mean.correction(convert_element_type_238, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_238 = None
        getitem_160: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = var_mean_79[0]
        getitem_161: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = var_mean_79[1];  var_mean_79 = None
        add_2601: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_160, 1e-05)
        rsqrt_79: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2601);  add_2601 = None
        sub_599: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_37, getitem_161)
        mul_1995: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_599, rsqrt_79);  sub_599 = None
        squeeze_237: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
        squeeze_238: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
        mul_1996: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
        mul_1997: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
        add_2602: "f32[896][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1996, mul_1997);  mul_1996 = mul_1997 = None
        sym_numel_default_79: "Sym(175616 * s28)" = torch.ops.aten.sym_numel.default(cat_37)
        truediv_158: "Sym(IntTrueDiv(175616*s28, 896))" = sym_numel_default_79 / 896;  sym_numel_default_79 = None
        squeeze_239: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
        sub_600: "Sym(-1.00000000000000 + IntTrueDiv(175616*s28, 896))" = truediv_158 - 1.0
        truediv_159: "Sym(FloatTrueDiv(IntTrueDiv(175616*s28, 896), (IntTrueDiv(175616*s28, 896)) - 1.0))" = truediv_158 / sub_600;  truediv_158 = sub_600 = None
        mul_1998: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_239, truediv_159);  squeeze_239 = truediv_159 = None
        mul_1999: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1998, 0.1);  mul_1998 = None
        mul_2000: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_479, 0.9)
        add_2603: "f32[896][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1999, mul_2000);  mul_1999 = mul_2000 = None
        unsqueeze_316: "f32[896, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_480, -1)
        unsqueeze_317: "f32[896, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_2001: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1995, unsqueeze_317);  mul_1995 = unsqueeze_317 = None
        unsqueeze_318: "f32[896, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_481, -1);  primals_481 = None
        unsqueeze_319: "f32[896, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_2604: "f32[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2001, unsqueeze_319);  mul_2001 = unsqueeze_319 = None
        convert_element_type_239: "f16[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2604, torch.float16);  add_2604 = None
        relu_79: "f16[s28, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_239);  convert_element_type_239 = None
        convert_element_type_240: "f16[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_482, torch.float16);  primals_482 = None
        convolution_79: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_79, convert_element_type_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2630: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_483, 1)
        convert_element_type_241: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_79, torch.float32)
        var_mean_80 = torch.ops.aten.var_mean.correction(convert_element_type_241, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_241 = None
        getitem_162: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_80[0]
        getitem_163: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_80[1];  var_mean_80 = None
        add_2631: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
        rsqrt_80: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2631);  add_2631 = None
        sub_606: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_79, getitem_163)
        mul_2019: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_606, rsqrt_80);  sub_606 = None
        squeeze_240: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
        squeeze_241: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
        mul_2020: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
        mul_2021: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
        add_2632: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2020, mul_2021);  mul_2020 = mul_2021 = None
        sym_numel_default_80: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_79)
        truediv_160: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_80 / 128;  sym_numel_default_80 = None
        squeeze_242: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
        sub_607: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_160 - 1.0
        truediv_161: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_160 / sub_607;  truediv_160 = sub_607 = None
        mul_2022: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_242, truediv_161);  squeeze_242 = truediv_161 = None
        mul_2023: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2022, 0.1);  mul_2022 = None
        mul_2024: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_485, 0.9)
        add_2633: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2023, mul_2024);  mul_2023 = mul_2024 = None
        unsqueeze_320: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_486, -1)
        unsqueeze_321: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        mul_2025: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2019, unsqueeze_321);  mul_2019 = unsqueeze_321 = None
        unsqueeze_322: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_487, -1);  primals_487 = None
        unsqueeze_323: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        add_2634: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2025, unsqueeze_323);  mul_2025 = unsqueeze_323 = None
        convert_element_type_242: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2634, torch.float16);  add_2634 = None
        relu_80: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_242);  convert_element_type_242 = None
        convert_element_type_243: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_488, torch.float16);  primals_488 = None
        convolution_80: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_80, convert_element_type_243, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_38: "f16[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2665: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_489, 1)
        convert_element_type_244: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_38, torch.float32)
        var_mean_81 = torch.ops.aten.var_mean.correction(convert_element_type_244, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_244 = None
        getitem_164: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = var_mean_81[0]
        getitem_165: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = var_mean_81[1];  var_mean_81 = None
        add_2666: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_164, 1e-05)
        rsqrt_81: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2666);  add_2666 = None
        sub_614: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_38, getitem_165)
        mul_2045: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_614, rsqrt_81);  sub_614 = None
        squeeze_243: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
        squeeze_244: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
        mul_2046: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
        mul_2047: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_490, 0.9)
        add_2667: "f32[928][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2046, mul_2047);  mul_2046 = mul_2047 = None
        sym_numel_default_81: "Sym(181888 * s28)" = torch.ops.aten.sym_numel.default(cat_38)
        truediv_162: "Sym(IntTrueDiv(181888*s28, 928))" = sym_numel_default_81 / 928;  sym_numel_default_81 = None
        squeeze_245: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
        sub_615: "Sym(-1.00000000000000 + IntTrueDiv(181888*s28, 928))" = truediv_162 - 1.0
        truediv_163: "Sym(FloatTrueDiv(IntTrueDiv(181888*s28, 928), (IntTrueDiv(181888*s28, 928)) - 1.0))" = truediv_162 / sub_615;  truediv_162 = sub_615 = None
        mul_2048: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_245, truediv_163);  squeeze_245 = truediv_163 = None
        mul_2049: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2048, 0.1);  mul_2048 = None
        mul_2050: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_491, 0.9)
        add_2668: "f32[928][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2049, mul_2050);  mul_2049 = mul_2050 = None
        unsqueeze_324: "f32[928, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_492, -1)
        unsqueeze_325: "f32[928, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_2051: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2045, unsqueeze_325);  mul_2045 = unsqueeze_325 = None
        unsqueeze_326: "f32[928, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_493, -1);  primals_493 = None
        unsqueeze_327: "f32[928, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_2669: "f32[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2051, unsqueeze_327);  mul_2051 = unsqueeze_327 = None
        convert_element_type_245: "f16[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2669, torch.float16);  add_2669 = None
        relu_81: "f16[s28, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_245);  convert_element_type_245 = None
        convert_element_type_246: "f16[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_494, torch.float16);  primals_494 = None
        convolution_81: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_81, convert_element_type_246, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2695: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_495, 1)
        convert_element_type_247: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_81, torch.float32)
        var_mean_82 = torch.ops.aten.var_mean.correction(convert_element_type_247, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_247 = None
        getitem_166: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_82[0]
        getitem_167: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_82[1];  var_mean_82 = None
        add_2696: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_166, 1e-05)
        rsqrt_82: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2696);  add_2696 = None
        sub_621: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_81, getitem_167)
        mul_2069: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_621, rsqrt_82);  sub_621 = None
        squeeze_246: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
        squeeze_247: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
        mul_2070: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
        mul_2071: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_496, 0.9)
        add_2697: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2070, mul_2071);  mul_2070 = mul_2071 = None
        sym_numel_default_82: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_81)
        truediv_164: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_82 / 128;  sym_numel_default_82 = None
        squeeze_248: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
        sub_622: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_164 - 1.0
        truediv_165: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_164 / sub_622;  truediv_164 = sub_622 = None
        mul_2072: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_248, truediv_165);  squeeze_248 = truediv_165 = None
        mul_2073: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2072, 0.1);  mul_2072 = None
        mul_2074: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_497, 0.9)
        add_2698: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2073, mul_2074);  mul_2073 = mul_2074 = None
        unsqueeze_328: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_498, -1)
        unsqueeze_329: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        mul_2075: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2069, unsqueeze_329);  mul_2069 = unsqueeze_329 = None
        unsqueeze_330: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_499, -1);  primals_499 = None
        unsqueeze_331: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        add_2699: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2075, unsqueeze_331);  mul_2075 = unsqueeze_331 = None
        convert_element_type_248: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2699, torch.float16);  add_2699 = None
        relu_82: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_248);  convert_element_type_248 = None
        convert_element_type_249: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_500, torch.float16);  primals_500 = None
        convolution_82: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_82, convert_element_type_249, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_39: "f16[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80, convolution_82], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2730: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_501, 1)
        convert_element_type_250: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_39, torch.float32)
        var_mean_83 = torch.ops.aten.var_mean.correction(convert_element_type_250, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_250 = None
        getitem_168: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_83[0]
        getitem_169: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_83[1];  var_mean_83 = None
        add_2731: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_168, 1e-05)
        rsqrt_83: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2731);  add_2731 = None
        sub_629: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_39, getitem_169)
        mul_2095: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_629, rsqrt_83);  sub_629 = None
        squeeze_249: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
        squeeze_250: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
        mul_2096: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
        mul_2097: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_502, 0.9)
        add_2732: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2096, mul_2097);  mul_2096 = mul_2097 = None
        sym_numel_default_83: "Sym(188160 * s28)" = torch.ops.aten.sym_numel.default(cat_39)
        truediv_166: "Sym(IntTrueDiv(188160*s28, 960))" = sym_numel_default_83 / 960;  sym_numel_default_83 = None
        squeeze_251: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
        sub_630: "Sym(-1.00000000000000 + IntTrueDiv(188160*s28, 960))" = truediv_166 - 1.0
        truediv_167: "Sym(FloatTrueDiv(IntTrueDiv(188160*s28, 960), (IntTrueDiv(188160*s28, 960)) - 1.0))" = truediv_166 / sub_630;  truediv_166 = sub_630 = None
        mul_2098: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_251, truediv_167);  squeeze_251 = truediv_167 = None
        mul_2099: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2098, 0.1);  mul_2098 = None
        mul_2100: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_503, 0.9)
        add_2733: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2099, mul_2100);  mul_2099 = mul_2100 = None
        unsqueeze_332: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_504, -1)
        unsqueeze_333: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_2101: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2095, unsqueeze_333);  mul_2095 = unsqueeze_333 = None
        unsqueeze_334: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_505, -1);  primals_505 = None
        unsqueeze_335: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_2734: "f32[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2101, unsqueeze_335);  mul_2101 = unsqueeze_335 = None
        convert_element_type_251: "f16[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2734, torch.float16);  add_2734 = None
        relu_83: "f16[s28, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_251);  convert_element_type_251 = None
        convert_element_type_252: "f16[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_506, torch.float16);  primals_506 = None
        convolution_83: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_83, convert_element_type_252, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2760: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_507, 1)
        convert_element_type_253: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_83, torch.float32)
        var_mean_84 = torch.ops.aten.var_mean.correction(convert_element_type_253, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_253 = None
        getitem_170: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_84[0]
        getitem_171: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_84[1];  var_mean_84 = None
        add_2761: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_170, 1e-05)
        rsqrt_84: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2761);  add_2761 = None
        sub_636: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_83, getitem_171)
        mul_2119: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_636, rsqrt_84);  sub_636 = None
        squeeze_252: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
        squeeze_253: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
        mul_2120: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
        mul_2121: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_508, 0.9)
        add_2762: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2120, mul_2121);  mul_2120 = mul_2121 = None
        sym_numel_default_84: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_83)
        truediv_168: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_84 / 128;  sym_numel_default_84 = None
        squeeze_254: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
        sub_637: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_168 - 1.0
        truediv_169: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_168 / sub_637;  truediv_168 = sub_637 = None
        mul_2122: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_254, truediv_169);  squeeze_254 = truediv_169 = None
        mul_2123: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2122, 0.1);  mul_2122 = None
        mul_2124: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_509, 0.9)
        add_2763: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2123, mul_2124);  mul_2123 = mul_2124 = None
        unsqueeze_336: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_510, -1)
        unsqueeze_337: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        mul_2125: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2119, unsqueeze_337);  mul_2119 = unsqueeze_337 = None
        unsqueeze_338: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_511, -1);  primals_511 = None
        unsqueeze_339: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        add_2764: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2125, unsqueeze_339);  mul_2125 = unsqueeze_339 = None
        convert_element_type_254: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2764, torch.float16);  add_2764 = None
        relu_84: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_254);  convert_element_type_254 = None
        convert_element_type_255: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_512, torch.float16);  primals_512 = None
        convolution_84: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_84, convert_element_type_255, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_40: "f16[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80, convolution_82, convolution_84], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2795: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_513, 1)
        convert_element_type_256: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_40, torch.float32)
        var_mean_85 = torch.ops.aten.var_mean.correction(convert_element_type_256, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_256 = None
        getitem_172: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = var_mean_85[0]
        getitem_173: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = var_mean_85[1];  var_mean_85 = None
        add_2796: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_172, 1e-05)
        rsqrt_85: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2796);  add_2796 = None
        sub_644: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_40, getitem_173)
        mul_2145: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_644, rsqrt_85);  sub_644 = None
        squeeze_255: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
        squeeze_256: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
        mul_2146: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
        mul_2147: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_514, 0.9)
        add_2797: "f32[992][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2146, mul_2147);  mul_2146 = mul_2147 = None
        sym_numel_default_85: "Sym(194432 * s28)" = torch.ops.aten.sym_numel.default(cat_40)
        truediv_170: "Sym(IntTrueDiv(194432*s28, 992))" = sym_numel_default_85 / 992;  sym_numel_default_85 = None
        squeeze_257: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
        sub_645: "Sym(-1.00000000000000 + IntTrueDiv(194432*s28, 992))" = truediv_170 - 1.0
        truediv_171: "Sym(FloatTrueDiv(IntTrueDiv(194432*s28, 992), (IntTrueDiv(194432*s28, 992)) - 1.0))" = truediv_170 / sub_645;  truediv_170 = sub_645 = None
        mul_2148: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_257, truediv_171);  squeeze_257 = truediv_171 = None
        mul_2149: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2148, 0.1);  mul_2148 = None
        mul_2150: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_515, 0.9)
        add_2798: "f32[992][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2149, mul_2150);  mul_2149 = mul_2150 = None
        unsqueeze_340: "f32[992, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_516, -1)
        unsqueeze_341: "f32[992, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_2151: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2145, unsqueeze_341);  mul_2145 = unsqueeze_341 = None
        unsqueeze_342: "f32[992, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_517, -1);  primals_517 = None
        unsqueeze_343: "f32[992, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_2799: "f32[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2151, unsqueeze_343);  mul_2151 = unsqueeze_343 = None
        convert_element_type_257: "f16[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2799, torch.float16);  add_2799 = None
        relu_85: "f16[s28, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_257);  convert_element_type_257 = None
        convert_element_type_258: "f16[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_518, torch.float16);  primals_518 = None
        convolution_85: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_85, convert_element_type_258, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2825: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_519, 1)
        convert_element_type_259: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_85, torch.float32)
        var_mean_86 = torch.ops.aten.var_mean.correction(convert_element_type_259, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_259 = None
        getitem_174: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_86[0]
        getitem_175: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_86[1];  var_mean_86 = None
        add_2826: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_174, 1e-05)
        rsqrt_86: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2826);  add_2826 = None
        sub_651: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_85, getitem_175)
        mul_2169: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_651, rsqrt_86);  sub_651 = None
        squeeze_258: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
        squeeze_259: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
        mul_2170: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
        mul_2171: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_520, 0.9)
        add_2827: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2170, mul_2171);  mul_2170 = mul_2171 = None
        sym_numel_default_86: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(convolution_85)
        truediv_172: "Sym(IntTrueDiv(25088*s28, 128))" = sym_numel_default_86 / 128;  sym_numel_default_86 = None
        squeeze_260: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
        sub_652: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 128))" = truediv_172 - 1.0
        truediv_173: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 128), (IntTrueDiv(25088*s28, 128)) - 1.0))" = truediv_172 / sub_652;  truediv_172 = sub_652 = None
        mul_2172: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_260, truediv_173);  squeeze_260 = truediv_173 = None
        mul_2173: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2172, 0.1);  mul_2172 = None
        mul_2174: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_521, 0.9)
        add_2828: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2173, mul_2174);  mul_2173 = mul_2174 = None
        unsqueeze_344: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_522, -1)
        unsqueeze_345: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        mul_2175: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2169, unsqueeze_345);  mul_2169 = unsqueeze_345 = None
        unsqueeze_346: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_523, -1);  primals_523 = None
        unsqueeze_347: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        add_2829: "f32[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2175, unsqueeze_347);  mul_2175 = unsqueeze_347 = None
        convert_element_type_260: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2829, torch.float16);  add_2829 = None
        relu_86: "f16[s28, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_260);  convert_element_type_260 = None
        convert_element_type_261: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_524, torch.float16);  primals_524 = None
        convolution_86: "f16[s28, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_86, convert_element_type_261, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        cat_41: "f16[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80, convolution_82, convolution_84, convolution_86], 1);  convolution_40 = convolution_42 = convolution_44 = convolution_46 = convolution_48 = convolution_50 = convolution_52 = convolution_54 = convolution_56 = convolution_58 = convolution_60 = convolution_62 = convolution_64 = convolution_66 = convolution_68 = convolution_70 = convolution_72 = convolution_74 = convolution_76 = convolution_78 = convolution_80 = convolution_82 = convolution_84 = convolution_86 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        add_2860: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_525, 1)
        convert_element_type_262: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_41, torch.float32)
        var_mean_87 = torch.ops.aten.var_mean.correction(convert_element_type_262, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_262 = None
        getitem_176: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_87[0]
        getitem_177: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_87[1];  var_mean_87 = None
        add_2861: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_176, 1e-05)
        rsqrt_87: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2861);  add_2861 = None
        sub_659: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_41, getitem_177)
        mul_2195: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_659, rsqrt_87);  sub_659 = None
        squeeze_261: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
        squeeze_262: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
        mul_2196: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
        mul_2197: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
        add_2862: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2196, mul_2197);  mul_2196 = mul_2197 = None
        sym_numel_default_87: "Sym(200704 * s28)" = torch.ops.aten.sym_numel.default(cat_41)
        truediv_174: "Sym(IntTrueDiv(200704*s28, 1024))" = sym_numel_default_87 / 1024;  sym_numel_default_87 = None
        squeeze_263: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
        sub_660: "Sym(-1.00000000000000 + IntTrueDiv(200704*s28, 1024))" = truediv_174 - 1.0
        truediv_175: "Sym(FloatTrueDiv(IntTrueDiv(200704*s28, 1024), (IntTrueDiv(200704*s28, 1024)) - 1.0))" = truediv_174 / sub_660;  truediv_174 = sub_660 = None
        mul_2198: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_263, truediv_175);  squeeze_263 = truediv_175 = None
        mul_2199: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2198, 0.1);  mul_2198 = None
        mul_2200: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_527, 0.9)
        add_2863: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2199, mul_2200);  mul_2199 = mul_2200 = None
        unsqueeze_348: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_528, -1)
        unsqueeze_349: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_2201: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2195, unsqueeze_349);  mul_2195 = unsqueeze_349 = None
        unsqueeze_350: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_529, -1);  primals_529 = None
        unsqueeze_351: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_2864: "f32[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2201, unsqueeze_351);  mul_2201 = unsqueeze_351 = None
        convert_element_type_263: "f16[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2864, torch.float16);  add_2864 = None
        relu_87: "f16[s28, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_263);  convert_element_type_263 = None
        convert_element_type_264: "f16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_530, torch.float16);  primals_530 = None
        convolution_87: "f16[s28, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_87, convert_element_type_264, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        avg_pool2d_2: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.avg_pool2d.default(convolution_87, [2, 2], [2, 2])

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2900: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_531, 1)
        convert_element_type_265: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_2, torch.float32)
        var_mean_88 = torch.ops.aten.var_mean.correction(convert_element_type_265, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_265 = None
        getitem_178: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_88[0]
        getitem_179: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_88[1];  var_mean_88 = None
        add_2901: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_178, 1e-05)
        rsqrt_88: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2901);  add_2901 = None
        sub_668: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(avg_pool2d_2, getitem_179)
        mul_2223: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_668, rsqrt_88);  sub_668 = None
        squeeze_264: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
        squeeze_265: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
        mul_2224: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1)
        mul_2225: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
        add_2902: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2224, mul_2225);  mul_2224 = mul_2225 = None
        sym_numel_default_88: "Sym(25088 * s28)" = torch.ops.aten.sym_numel.default(avg_pool2d_2)
        truediv_176: "Sym(IntTrueDiv(25088*s28, 512))" = sym_numel_default_88 / 512;  sym_numel_default_88 = None
        squeeze_266: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
        sub_669: "Sym(-1.00000000000000 + IntTrueDiv(25088*s28, 512))" = truediv_176 - 1.0
        truediv_177: "Sym(FloatTrueDiv(IntTrueDiv(25088*s28, 512), (IntTrueDiv(25088*s28, 512)) - 1.0))" = truediv_176 / sub_669;  truediv_176 = sub_669 = None
        mul_2226: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_266, truediv_177);  squeeze_266 = truediv_177 = None
        mul_2227: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2226, 0.1);  mul_2226 = None
        mul_2228: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_533, 0.9)
        add_2903: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2227, mul_2228);  mul_2227 = mul_2228 = None
        unsqueeze_352: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_534, -1)
        unsqueeze_353: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        mul_2229: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2223, unsqueeze_353);  mul_2223 = unsqueeze_353 = None
        unsqueeze_354: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_535, -1);  primals_535 = None
        unsqueeze_355: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        add_2904: "f32[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2229, unsqueeze_355);  mul_2229 = unsqueeze_355 = None
        convert_element_type_266: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2904, torch.float16);  add_2904 = None
        relu_88: "f16[s28, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_266);  convert_element_type_266 = None
        convert_element_type_267: "f16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_536, torch.float16);  primals_536 = None
        convolution_88: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_88, convert_element_type_267, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2930: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_537, 1)
        convert_element_type_268: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_88, torch.float32)
        var_mean_89 = torch.ops.aten.var_mean.correction(convert_element_type_268, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_268 = None
        getitem_180: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_89[0]
        getitem_181: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_89[1];  var_mean_89 = None
        add_2931: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_180, 1e-05)
        rsqrt_89: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2931);  add_2931 = None
        sub_675: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_88, getitem_181)
        mul_2247: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_675, rsqrt_89);  sub_675 = None
        squeeze_267: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
        squeeze_268: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
        mul_2248: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
        mul_2249: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_538, 0.9)
        add_2932: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2248, mul_2249);  mul_2248 = mul_2249 = None
        sym_numel_default_89: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_88)
        truediv_178: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_89 / 128;  sym_numel_default_89 = None
        squeeze_269: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
        sub_676: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_178 - 1.0
        truediv_179: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_178 / sub_676;  truediv_178 = sub_676 = None
        mul_2250: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_269, truediv_179);  squeeze_269 = truediv_179 = None
        mul_2251: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2250, 0.1);  mul_2250 = None
        mul_2252: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_539, 0.9)
        add_2933: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2251, mul_2252);  mul_2251 = mul_2252 = None
        unsqueeze_356: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_540, -1)
        unsqueeze_357: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_2253: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2247, unsqueeze_357);  mul_2247 = unsqueeze_357 = None
        unsqueeze_358: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_541, -1);  primals_541 = None
        unsqueeze_359: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_2934: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2253, unsqueeze_359);  mul_2253 = unsqueeze_359 = None
        convert_element_type_269: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2934, torch.float16);  add_2934 = None
        relu_89: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_269);  convert_element_type_269 = None
        convert_element_type_270: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_542, torch.float16);  primals_542 = None
        convolution_89: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_89, convert_element_type_270, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_42: "f16[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_2965: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_543, 1)
        convert_element_type_271: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_42, torch.float32)
        var_mean_90 = torch.ops.aten.var_mean.correction(convert_element_type_271, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_271 = None
        getitem_182: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = var_mean_90[0]
        getitem_183: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = var_mean_90[1];  var_mean_90 = None
        add_2966: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_182, 1e-05)
        rsqrt_90: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2966);  add_2966 = None
        sub_683: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_42, getitem_183)
        mul_2273: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_683, rsqrt_90);  sub_683 = None
        squeeze_270: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
        squeeze_271: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
        mul_2274: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
        mul_2275: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_544, 0.9)
        add_2967: "f32[544][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2274, mul_2275);  mul_2274 = mul_2275 = None
        sym_numel_default_90: "Sym(26656 * s28)" = torch.ops.aten.sym_numel.default(cat_42)
        truediv_180: "Sym(IntTrueDiv(26656*s28, 544))" = sym_numel_default_90 / 544;  sym_numel_default_90 = None
        squeeze_272: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
        sub_684: "Sym(-1.00000000000000 + IntTrueDiv(26656*s28, 544))" = truediv_180 - 1.0
        truediv_181: "Sym(FloatTrueDiv(IntTrueDiv(26656*s28, 544), (IntTrueDiv(26656*s28, 544)) - 1.0))" = truediv_180 / sub_684;  truediv_180 = sub_684 = None
        mul_2276: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_272, truediv_181);  squeeze_272 = truediv_181 = None
        mul_2277: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2276, 0.1);  mul_2276 = None
        mul_2278: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_545, 0.9)
        add_2968: "f32[544][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2277, mul_2278);  mul_2277 = mul_2278 = None
        unsqueeze_360: "f32[544, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_546, -1)
        unsqueeze_361: "f32[544, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        mul_2279: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2273, unsqueeze_361);  mul_2273 = unsqueeze_361 = None
        unsqueeze_362: "f32[544, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_547, -1);  primals_547 = None
        unsqueeze_363: "f32[544, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        add_2969: "f32[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2279, unsqueeze_363);  mul_2279 = unsqueeze_363 = None
        convert_element_type_272: "f16[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2969, torch.float16);  add_2969 = None
        relu_90: "f16[s28, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_272);  convert_element_type_272 = None
        convert_element_type_273: "f16[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_548, torch.float16);  primals_548 = None
        convolution_90: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_90, convert_element_type_273, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_2995: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_549, 1)
        convert_element_type_274: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_90, torch.float32)
        var_mean_91 = torch.ops.aten.var_mean.correction(convert_element_type_274, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_274 = None
        getitem_184: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_91[0]
        getitem_185: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_91[1];  var_mean_91 = None
        add_2996: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_184, 1e-05)
        rsqrt_91: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_2996);  add_2996 = None
        sub_690: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_90, getitem_185)
        mul_2297: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_690, rsqrt_91);  sub_690 = None
        squeeze_273: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
        squeeze_274: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
        mul_2298: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
        mul_2299: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
        add_2997: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2298, mul_2299);  mul_2298 = mul_2299 = None
        sym_numel_default_91: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_90)
        truediv_182: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_91 / 128;  sym_numel_default_91 = None
        squeeze_275: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
        sub_691: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_182 - 1.0
        truediv_183: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_182 / sub_691;  truediv_182 = sub_691 = None
        mul_2300: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_275, truediv_183);  squeeze_275 = truediv_183 = None
        mul_2301: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2300, 0.1);  mul_2300 = None
        mul_2302: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_551, 0.9)
        add_2998: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2301, mul_2302);  mul_2301 = mul_2302 = None
        unsqueeze_364: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_552, -1)
        unsqueeze_365: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_2303: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2297, unsqueeze_365);  mul_2297 = unsqueeze_365 = None
        unsqueeze_366: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_553, -1);  primals_553 = None
        unsqueeze_367: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_2999: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2303, unsqueeze_367);  mul_2303 = unsqueeze_367 = None
        convert_element_type_275: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_2999, torch.float16);  add_2999 = None
        relu_91: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_275);  convert_element_type_275 = None
        convert_element_type_276: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_554, torch.float16);  primals_554 = None
        convolution_91: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_91, convert_element_type_276, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_43: "f16[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3030: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_555, 1)
        convert_element_type_277: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_43, torch.float32)
        var_mean_92 = torch.ops.aten.var_mean.correction(convert_element_type_277, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_277 = None
        getitem_186: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = var_mean_92[0]
        getitem_187: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = var_mean_92[1];  var_mean_92 = None
        add_3031: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_186, 1e-05)
        rsqrt_92: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3031);  add_3031 = None
        sub_698: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_43, getitem_187)
        mul_2323: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_698, rsqrt_92);  sub_698 = None
        squeeze_276: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
        squeeze_277: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
        mul_2324: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1)
        mul_2325: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
        add_3032: "f32[576][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2324, mul_2325);  mul_2324 = mul_2325 = None
        sym_numel_default_92: "Sym(28224 * s28)" = torch.ops.aten.sym_numel.default(cat_43)
        truediv_184: "Sym(IntTrueDiv(28224*s28, 576))" = sym_numel_default_92 / 576;  sym_numel_default_92 = None
        squeeze_278: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
        sub_699: "Sym(-1.00000000000000 + IntTrueDiv(28224*s28, 576))" = truediv_184 - 1.0
        truediv_185: "Sym(FloatTrueDiv(IntTrueDiv(28224*s28, 576), (IntTrueDiv(28224*s28, 576)) - 1.0))" = truediv_184 / sub_699;  truediv_184 = sub_699 = None
        mul_2326: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_278, truediv_185);  squeeze_278 = truediv_185 = None
        mul_2327: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2326, 0.1);  mul_2326 = None
        mul_2328: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_557, 0.9)
        add_3033: "f32[576][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2327, mul_2328);  mul_2327 = mul_2328 = None
        unsqueeze_368: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_558, -1)
        unsqueeze_369: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        mul_2329: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2323, unsqueeze_369);  mul_2323 = unsqueeze_369 = None
        unsqueeze_370: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_559, -1);  primals_559 = None
        unsqueeze_371: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        add_3034: "f32[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2329, unsqueeze_371);  mul_2329 = unsqueeze_371 = None
        convert_element_type_278: "f16[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3034, torch.float16);  add_3034 = None
        relu_92: "f16[s28, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_278);  convert_element_type_278 = None
        convert_element_type_279: "f16[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_560, torch.float16);  primals_560 = None
        convolution_92: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_92, convert_element_type_279, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3060: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_561, 1)
        convert_element_type_280: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_92, torch.float32)
        var_mean_93 = torch.ops.aten.var_mean.correction(convert_element_type_280, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_280 = None
        getitem_188: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_93[0]
        getitem_189: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_93[1];  var_mean_93 = None
        add_3061: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_188, 1e-05)
        rsqrt_93: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3061);  add_3061 = None
        sub_705: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_92, getitem_189)
        mul_2347: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_705, rsqrt_93);  sub_705 = None
        squeeze_279: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
        squeeze_280: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
        mul_2348: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
        mul_2349: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
        add_3062: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2348, mul_2349);  mul_2348 = mul_2349 = None
        sym_numel_default_93: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_92)
        truediv_186: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_93 / 128;  sym_numel_default_93 = None
        squeeze_281: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
        sub_706: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_186 - 1.0
        truediv_187: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_186 / sub_706;  truediv_186 = sub_706 = None
        mul_2350: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_281, truediv_187);  squeeze_281 = truediv_187 = None
        mul_2351: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2350, 0.1);  mul_2350 = None
        mul_2352: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_563, 0.9)
        add_3063: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2351, mul_2352);  mul_2351 = mul_2352 = None
        unsqueeze_372: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_564, -1)
        unsqueeze_373: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_2353: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2347, unsqueeze_373);  mul_2347 = unsqueeze_373 = None
        unsqueeze_374: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_565, -1);  primals_565 = None
        unsqueeze_375: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_3064: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2353, unsqueeze_375);  mul_2353 = unsqueeze_375 = None
        convert_element_type_281: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3064, torch.float16);  add_3064 = None
        relu_93: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_281);  convert_element_type_281 = None
        convert_element_type_282: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_566, torch.float16);  primals_566 = None
        convolution_93: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_93, convert_element_type_282, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_44: "f16[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3095: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_567, 1)
        convert_element_type_283: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_44, torch.float32)
        var_mean_94 = torch.ops.aten.var_mean.correction(convert_element_type_283, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_283 = None
        getitem_190: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = var_mean_94[0]
        getitem_191: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = var_mean_94[1];  var_mean_94 = None
        add_3096: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_190, 1e-05)
        rsqrt_94: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3096);  add_3096 = None
        sub_713: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_44, getitem_191)
        mul_2373: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_713, rsqrt_94);  sub_713 = None
        squeeze_282: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
        squeeze_283: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
        mul_2374: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_282, 0.1)
        mul_2375: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_568, 0.9)
        add_3097: "f32[608][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2374, mul_2375);  mul_2374 = mul_2375 = None
        sym_numel_default_94: "Sym(29792 * s28)" = torch.ops.aten.sym_numel.default(cat_44)
        truediv_188: "Sym(IntTrueDiv(29792*s28, 608))" = sym_numel_default_94 / 608;  sym_numel_default_94 = None
        squeeze_284: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
        sub_714: "Sym(-1.00000000000000 + IntTrueDiv(29792*s28, 608))" = truediv_188 - 1.0
        truediv_189: "Sym(FloatTrueDiv(IntTrueDiv(29792*s28, 608), (IntTrueDiv(29792*s28, 608)) - 1.0))" = truediv_188 / sub_714;  truediv_188 = sub_714 = None
        mul_2376: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_284, truediv_189);  squeeze_284 = truediv_189 = None
        mul_2377: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2376, 0.1);  mul_2376 = None
        mul_2378: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_569, 0.9)
        add_3098: "f32[608][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2377, mul_2378);  mul_2377 = mul_2378 = None
        unsqueeze_376: "f32[608, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_570, -1)
        unsqueeze_377: "f32[608, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        mul_2379: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2373, unsqueeze_377);  mul_2373 = unsqueeze_377 = None
        unsqueeze_378: "f32[608, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_571, -1);  primals_571 = None
        unsqueeze_379: "f32[608, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        add_3099: "f32[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2379, unsqueeze_379);  mul_2379 = unsqueeze_379 = None
        convert_element_type_284: "f16[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3099, torch.float16);  add_3099 = None
        relu_94: "f16[s28, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_284);  convert_element_type_284 = None
        convert_element_type_285: "f16[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_572, torch.float16);  primals_572 = None
        convolution_94: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_94, convert_element_type_285, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3125: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_573, 1)
        convert_element_type_286: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_94, torch.float32)
        var_mean_95 = torch.ops.aten.var_mean.correction(convert_element_type_286, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_286 = None
        getitem_192: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_95[0]
        getitem_193: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_95[1];  var_mean_95 = None
        add_3126: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_192, 1e-05)
        rsqrt_95: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3126);  add_3126 = None
        sub_720: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_94, getitem_193)
        mul_2397: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_720, rsqrt_95);  sub_720 = None
        squeeze_285: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
        squeeze_286: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
        mul_2398: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_285, 0.1)
        mul_2399: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_574, 0.9)
        add_3127: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2398, mul_2399);  mul_2398 = mul_2399 = None
        sym_numel_default_95: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_94)
        truediv_190: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_95 / 128;  sym_numel_default_95 = None
        squeeze_287: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
        sub_721: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_190 - 1.0
        truediv_191: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_190 / sub_721;  truediv_190 = sub_721 = None
        mul_2400: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_287, truediv_191);  squeeze_287 = truediv_191 = None
        mul_2401: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2400, 0.1);  mul_2400 = None
        mul_2402: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_575, 0.9)
        add_3128: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2401, mul_2402);  mul_2401 = mul_2402 = None
        unsqueeze_380: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_576, -1)
        unsqueeze_381: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_2403: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2397, unsqueeze_381);  mul_2397 = unsqueeze_381 = None
        unsqueeze_382: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_577, -1);  primals_577 = None
        unsqueeze_383: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_3129: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2403, unsqueeze_383);  mul_2403 = unsqueeze_383 = None
        convert_element_type_287: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3129, torch.float16);  add_3129 = None
        relu_95: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_287);  convert_element_type_287 = None
        convert_element_type_288: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_578, torch.float16);  primals_578 = None
        convolution_95: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_95, convert_element_type_288, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_45: "f16[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3160: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_579, 1)
        convert_element_type_289: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_45, torch.float32)
        var_mean_96 = torch.ops.aten.var_mean.correction(convert_element_type_289, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_289 = None
        getitem_194: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_96[0]
        getitem_195: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_96[1];  var_mean_96 = None
        add_3161: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_194, 1e-05)
        rsqrt_96: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3161);  add_3161 = None
        sub_728: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_45, getitem_195)
        mul_2423: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_728, rsqrt_96);  sub_728 = None
        squeeze_288: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
        squeeze_289: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_96, [0, 2, 3]);  rsqrt_96 = None
        mul_2424: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_288, 0.1)
        mul_2425: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_580, 0.9)
        add_3162: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2424, mul_2425);  mul_2424 = mul_2425 = None
        sym_numel_default_96: "Sym(31360 * s28)" = torch.ops.aten.sym_numel.default(cat_45)
        truediv_192: "Sym(IntTrueDiv(31360*s28, 640))" = sym_numel_default_96 / 640;  sym_numel_default_96 = None
        squeeze_290: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
        sub_729: "Sym(-1.00000000000000 + IntTrueDiv(31360*s28, 640))" = truediv_192 - 1.0
        truediv_193: "Sym(FloatTrueDiv(IntTrueDiv(31360*s28, 640), (IntTrueDiv(31360*s28, 640)) - 1.0))" = truediv_192 / sub_729;  truediv_192 = sub_729 = None
        mul_2426: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_290, truediv_193);  squeeze_290 = truediv_193 = None
        mul_2427: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2426, 0.1);  mul_2426 = None
        mul_2428: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_581, 0.9)
        add_3163: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2427, mul_2428);  mul_2427 = mul_2428 = None
        unsqueeze_384: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_582, -1)
        unsqueeze_385: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        mul_2429: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2423, unsqueeze_385);  mul_2423 = unsqueeze_385 = None
        unsqueeze_386: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_583, -1);  primals_583 = None
        unsqueeze_387: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        add_3164: "f32[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2429, unsqueeze_387);  mul_2429 = unsqueeze_387 = None
        convert_element_type_290: "f16[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3164, torch.float16);  add_3164 = None
        relu_96: "f16[s28, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_290);  convert_element_type_290 = None
        convert_element_type_291: "f16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_584, torch.float16);  primals_584 = None
        convolution_96: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_96, convert_element_type_291, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3190: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_585, 1)
        convert_element_type_292: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_96, torch.float32)
        var_mean_97 = torch.ops.aten.var_mean.correction(convert_element_type_292, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_292 = None
        getitem_196: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_97[0]
        getitem_197: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_97[1];  var_mean_97 = None
        add_3191: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_196, 1e-05)
        rsqrt_97: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3191);  add_3191 = None
        sub_735: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_96, getitem_197)
        mul_2447: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_735, rsqrt_97);  sub_735 = None
        squeeze_291: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
        squeeze_292: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
        mul_2448: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_291, 0.1)
        mul_2449: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_586, 0.9)
        add_3192: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2448, mul_2449);  mul_2448 = mul_2449 = None
        sym_numel_default_97: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_96)
        truediv_194: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_97 / 128;  sym_numel_default_97 = None
        squeeze_293: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
        sub_736: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_194 - 1.0
        truediv_195: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_194 / sub_736;  truediv_194 = sub_736 = None
        mul_2450: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_293, truediv_195);  squeeze_293 = truediv_195 = None
        mul_2451: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2450, 0.1);  mul_2450 = None
        mul_2452: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_587, 0.9)
        add_3193: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2451, mul_2452);  mul_2451 = mul_2452 = None
        unsqueeze_388: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_588, -1)
        unsqueeze_389: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_2453: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2447, unsqueeze_389);  mul_2447 = unsqueeze_389 = None
        unsqueeze_390: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_589, -1);  primals_589 = None
        unsqueeze_391: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_3194: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2453, unsqueeze_391);  mul_2453 = unsqueeze_391 = None
        convert_element_type_293: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3194, torch.float16);  add_3194 = None
        relu_97: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_293);  convert_element_type_293 = None
        convert_element_type_294: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_590, torch.float16);  primals_590 = None
        convolution_97: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_97, convert_element_type_294, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_46: "f16[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3225: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_591, 1)
        convert_element_type_295: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_46, torch.float32)
        var_mean_98 = torch.ops.aten.var_mean.correction(convert_element_type_295, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_295 = None
        getitem_198: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_98[0]
        getitem_199: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_98[1];  var_mean_98 = None
        add_3226: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_198, 1e-05)
        rsqrt_98: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3226);  add_3226 = None
        sub_743: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_46, getitem_199)
        mul_2473: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_743, rsqrt_98);  sub_743 = None
        squeeze_294: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
        squeeze_295: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_98, [0, 2, 3]);  rsqrt_98 = None
        mul_2474: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_294, 0.1)
        mul_2475: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_592, 0.9)
        add_3227: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2474, mul_2475);  mul_2474 = mul_2475 = None
        sym_numel_default_98: "Sym(32928 * s28)" = torch.ops.aten.sym_numel.default(cat_46)
        truediv_196: "Sym(IntTrueDiv(32928*s28, 672))" = sym_numel_default_98 / 672;  sym_numel_default_98 = None
        squeeze_296: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
        sub_744: "Sym(-1.00000000000000 + IntTrueDiv(32928*s28, 672))" = truediv_196 - 1.0
        truediv_197: "Sym(FloatTrueDiv(IntTrueDiv(32928*s28, 672), (IntTrueDiv(32928*s28, 672)) - 1.0))" = truediv_196 / sub_744;  truediv_196 = sub_744 = None
        mul_2476: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_296, truediv_197);  squeeze_296 = truediv_197 = None
        mul_2477: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2476, 0.1);  mul_2476 = None
        mul_2478: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_593, 0.9)
        add_3228: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2477, mul_2478);  mul_2477 = mul_2478 = None
        unsqueeze_392: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_594, -1)
        unsqueeze_393: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        mul_2479: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2473, unsqueeze_393);  mul_2473 = unsqueeze_393 = None
        unsqueeze_394: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_595, -1);  primals_595 = None
        unsqueeze_395: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        add_3229: "f32[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2479, unsqueeze_395);  mul_2479 = unsqueeze_395 = None
        convert_element_type_296: "f16[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3229, torch.float16);  add_3229 = None
        relu_98: "f16[s28, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_296);  convert_element_type_296 = None
        convert_element_type_297: "f16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_596, torch.float16);  primals_596 = None
        convolution_98: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_98, convert_element_type_297, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3255: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_597, 1)
        convert_element_type_298: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_98, torch.float32)
        var_mean_99 = torch.ops.aten.var_mean.correction(convert_element_type_298, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_298 = None
        getitem_200: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_99[0]
        getitem_201: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_99[1];  var_mean_99 = None
        add_3256: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_200, 1e-05)
        rsqrt_99: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3256);  add_3256 = None
        sub_750: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_98, getitem_201)
        mul_2497: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_750, rsqrt_99);  sub_750 = None
        squeeze_297: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
        squeeze_298: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_99, [0, 2, 3]);  rsqrt_99 = None
        mul_2498: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_297, 0.1)
        mul_2499: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_598, 0.9)
        add_3257: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2498, mul_2499);  mul_2498 = mul_2499 = None
        sym_numel_default_99: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_98)
        truediv_198: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_99 / 128;  sym_numel_default_99 = None
        squeeze_299: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
        sub_751: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_198 - 1.0
        truediv_199: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_198 / sub_751;  truediv_198 = sub_751 = None
        mul_2500: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_299, truediv_199);  squeeze_299 = truediv_199 = None
        mul_2501: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2500, 0.1);  mul_2500 = None
        mul_2502: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_599, 0.9)
        add_3258: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2501, mul_2502);  mul_2501 = mul_2502 = None
        unsqueeze_396: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_600, -1)
        unsqueeze_397: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_2503: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2497, unsqueeze_397);  mul_2497 = unsqueeze_397 = None
        unsqueeze_398: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_601, -1);  primals_601 = None
        unsqueeze_399: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_3259: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2503, unsqueeze_399);  mul_2503 = unsqueeze_399 = None
        convert_element_type_299: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3259, torch.float16);  add_3259 = None
        relu_99: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_299);  convert_element_type_299 = None
        convert_element_type_300: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_602, torch.float16);  primals_602 = None
        convolution_99: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_99, convert_element_type_300, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_47: "f16[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3290: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_603, 1)
        convert_element_type_301: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_47, torch.float32)
        var_mean_100 = torch.ops.aten.var_mean.correction(convert_element_type_301, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_301 = None
        getitem_202: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = var_mean_100[0]
        getitem_203: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = var_mean_100[1];  var_mean_100 = None
        add_3291: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_202, 1e-05)
        rsqrt_100: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3291);  add_3291 = None
        sub_758: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_47, getitem_203)
        mul_2523: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_758, rsqrt_100);  sub_758 = None
        squeeze_300: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_203, [0, 2, 3]);  getitem_203 = None
        squeeze_301: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_100, [0, 2, 3]);  rsqrt_100 = None
        mul_2524: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_300, 0.1)
        mul_2525: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_604, 0.9)
        add_3292: "f32[704][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2524, mul_2525);  mul_2524 = mul_2525 = None
        sym_numel_default_100: "Sym(34496 * s28)" = torch.ops.aten.sym_numel.default(cat_47)
        truediv_200: "Sym(IntTrueDiv(34496*s28, 704))" = sym_numel_default_100 / 704;  sym_numel_default_100 = None
        squeeze_302: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_202, [0, 2, 3]);  getitem_202 = None
        sub_759: "Sym(-1.00000000000000 + IntTrueDiv(34496*s28, 704))" = truediv_200 - 1.0
        truediv_201: "Sym(FloatTrueDiv(IntTrueDiv(34496*s28, 704), (IntTrueDiv(34496*s28, 704)) - 1.0))" = truediv_200 / sub_759;  truediv_200 = sub_759 = None
        mul_2526: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_302, truediv_201);  squeeze_302 = truediv_201 = None
        mul_2527: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2526, 0.1);  mul_2526 = None
        mul_2528: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_605, 0.9)
        add_3293: "f32[704][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2527, mul_2528);  mul_2527 = mul_2528 = None
        unsqueeze_400: "f32[704, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_606, -1)
        unsqueeze_401: "f32[704, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        mul_2529: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2523, unsqueeze_401);  mul_2523 = unsqueeze_401 = None
        unsqueeze_402: "f32[704, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_607, -1);  primals_607 = None
        unsqueeze_403: "f32[704, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        add_3294: "f32[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2529, unsqueeze_403);  mul_2529 = unsqueeze_403 = None
        convert_element_type_302: "f16[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3294, torch.float16);  add_3294 = None
        relu_100: "f16[s28, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_302);  convert_element_type_302 = None
        convert_element_type_303: "f16[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_608, torch.float16);  primals_608 = None
        convolution_100: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_100, convert_element_type_303, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3320: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_609, 1)
        convert_element_type_304: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_100, torch.float32)
        var_mean_101 = torch.ops.aten.var_mean.correction(convert_element_type_304, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_304 = None
        getitem_204: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_101[0]
        getitem_205: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_101[1];  var_mean_101 = None
        add_3321: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_204, 1e-05)
        rsqrt_101: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3321);  add_3321 = None
        sub_765: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_100, getitem_205)
        mul_2547: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_765, rsqrt_101);  sub_765 = None
        squeeze_303: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_205, [0, 2, 3]);  getitem_205 = None
        squeeze_304: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_101, [0, 2, 3]);  rsqrt_101 = None
        mul_2548: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_303, 0.1)
        mul_2549: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_610, 0.9)
        add_3322: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2548, mul_2549);  mul_2548 = mul_2549 = None
        sym_numel_default_101: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_100)
        truediv_202: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_101 / 128;  sym_numel_default_101 = None
        squeeze_305: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_204, [0, 2, 3]);  getitem_204 = None
        sub_766: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_202 - 1.0
        truediv_203: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_202 / sub_766;  truediv_202 = sub_766 = None
        mul_2550: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_305, truediv_203);  squeeze_305 = truediv_203 = None
        mul_2551: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2550, 0.1);  mul_2550 = None
        mul_2552: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_611, 0.9)
        add_3323: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2551, mul_2552);  mul_2551 = mul_2552 = None
        unsqueeze_404: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_612, -1)
        unsqueeze_405: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_2553: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2547, unsqueeze_405);  mul_2547 = unsqueeze_405 = None
        unsqueeze_406: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_613, -1);  primals_613 = None
        unsqueeze_407: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_3324: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2553, unsqueeze_407);  mul_2553 = unsqueeze_407 = None
        convert_element_type_305: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3324, torch.float16);  add_3324 = None
        relu_101: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_305);  convert_element_type_305 = None
        convert_element_type_306: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_614, torch.float16);  primals_614 = None
        convolution_101: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_101, convert_element_type_306, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_48: "f16[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3355: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_615, 1)
        convert_element_type_307: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_48, torch.float32)
        var_mean_102 = torch.ops.aten.var_mean.correction(convert_element_type_307, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_307 = None
        getitem_206: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = var_mean_102[0]
        getitem_207: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = var_mean_102[1];  var_mean_102 = None
        add_3356: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_206, 1e-05)
        rsqrt_102: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3356);  add_3356 = None
        sub_773: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_48, getitem_207)
        mul_2573: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_773, rsqrt_102);  sub_773 = None
        squeeze_306: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_207, [0, 2, 3]);  getitem_207 = None
        squeeze_307: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_102, [0, 2, 3]);  rsqrt_102 = None
        mul_2574: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_306, 0.1)
        mul_2575: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_616, 0.9)
        add_3357: "f32[736][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2574, mul_2575);  mul_2574 = mul_2575 = None
        sym_numel_default_102: "Sym(36064 * s28)" = torch.ops.aten.sym_numel.default(cat_48)
        truediv_204: "Sym(IntTrueDiv(36064*s28, 736))" = sym_numel_default_102 / 736;  sym_numel_default_102 = None
        squeeze_308: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_206, [0, 2, 3]);  getitem_206 = None
        sub_774: "Sym(-1.00000000000000 + IntTrueDiv(36064*s28, 736))" = truediv_204 - 1.0
        truediv_205: "Sym(FloatTrueDiv(IntTrueDiv(36064*s28, 736), (IntTrueDiv(36064*s28, 736)) - 1.0))" = truediv_204 / sub_774;  truediv_204 = sub_774 = None
        mul_2576: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_308, truediv_205);  squeeze_308 = truediv_205 = None
        mul_2577: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2576, 0.1);  mul_2576 = None
        mul_2578: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_617, 0.9)
        add_3358: "f32[736][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2577, mul_2578);  mul_2577 = mul_2578 = None
        unsqueeze_408: "f32[736, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_618, -1)
        unsqueeze_409: "f32[736, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        mul_2579: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2573, unsqueeze_409);  mul_2573 = unsqueeze_409 = None
        unsqueeze_410: "f32[736, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_619, -1);  primals_619 = None
        unsqueeze_411: "f32[736, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        add_3359: "f32[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2579, unsqueeze_411);  mul_2579 = unsqueeze_411 = None
        convert_element_type_308: "f16[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3359, torch.float16);  add_3359 = None
        relu_102: "f16[s28, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_308);  convert_element_type_308 = None
        convert_element_type_309: "f16[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_620, torch.float16);  primals_620 = None
        convolution_102: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_102, convert_element_type_309, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3385: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_621, 1)
        convert_element_type_310: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_102, torch.float32)
        var_mean_103 = torch.ops.aten.var_mean.correction(convert_element_type_310, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_310 = None
        getitem_208: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_103[0]
        getitem_209: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_103[1];  var_mean_103 = None
        add_3386: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_208, 1e-05)
        rsqrt_103: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3386);  add_3386 = None
        sub_780: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_102, getitem_209)
        mul_2597: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_780, rsqrt_103);  sub_780 = None
        squeeze_309: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
        squeeze_310: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_103, [0, 2, 3]);  rsqrt_103 = None
        mul_2598: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_309, 0.1)
        mul_2599: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_622, 0.9)
        add_3387: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2598, mul_2599);  mul_2598 = mul_2599 = None
        sym_numel_default_103: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_102)
        truediv_206: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_103 / 128;  sym_numel_default_103 = None
        squeeze_311: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
        sub_781: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_206 - 1.0
        truediv_207: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_206 / sub_781;  truediv_206 = sub_781 = None
        mul_2600: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_311, truediv_207);  squeeze_311 = truediv_207 = None
        mul_2601: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2600, 0.1);  mul_2600 = None
        mul_2602: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_623, 0.9)
        add_3388: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2601, mul_2602);  mul_2601 = mul_2602 = None
        unsqueeze_412: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_624, -1)
        unsqueeze_413: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_2603: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2597, unsqueeze_413);  mul_2597 = unsqueeze_413 = None
        unsqueeze_414: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_625, -1);  primals_625 = None
        unsqueeze_415: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_3389: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2603, unsqueeze_415);  mul_2603 = unsqueeze_415 = None
        convert_element_type_311: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3389, torch.float16);  add_3389 = None
        relu_103: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_311);  convert_element_type_311 = None
        convert_element_type_312: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_626, torch.float16);  primals_626 = None
        convolution_103: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_103, convert_element_type_312, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_49: "f16[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3420: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_627, 1)
        convert_element_type_313: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_49, torch.float32)
        var_mean_104 = torch.ops.aten.var_mean.correction(convert_element_type_313, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_313 = None
        getitem_210: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_104[0]
        getitem_211: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_104[1];  var_mean_104 = None
        add_3421: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_210, 1e-05)
        rsqrt_104: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3421);  add_3421 = None
        sub_788: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_49, getitem_211)
        mul_2623: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_788, rsqrt_104);  sub_788 = None
        squeeze_312: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
        squeeze_313: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_104, [0, 2, 3]);  rsqrt_104 = None
        mul_2624: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_312, 0.1)
        mul_2625: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_628, 0.9)
        add_3422: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2624, mul_2625);  mul_2624 = mul_2625 = None
        sym_numel_default_104: "Sym(37632 * s28)" = torch.ops.aten.sym_numel.default(cat_49)
        truediv_208: "Sym(IntTrueDiv(37632*s28, 768))" = sym_numel_default_104 / 768;  sym_numel_default_104 = None
        squeeze_314: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
        sub_789: "Sym(-1.00000000000000 + IntTrueDiv(37632*s28, 768))" = truediv_208 - 1.0
        truediv_209: "Sym(FloatTrueDiv(IntTrueDiv(37632*s28, 768), (IntTrueDiv(37632*s28, 768)) - 1.0))" = truediv_208 / sub_789;  truediv_208 = sub_789 = None
        mul_2626: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_314, truediv_209);  squeeze_314 = truediv_209 = None
        mul_2627: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2626, 0.1);  mul_2626 = None
        mul_2628: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_629, 0.9)
        add_3423: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2627, mul_2628);  mul_2627 = mul_2628 = None
        unsqueeze_416: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_630, -1)
        unsqueeze_417: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        mul_2629: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2623, unsqueeze_417);  mul_2623 = unsqueeze_417 = None
        unsqueeze_418: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_631, -1);  primals_631 = None
        unsqueeze_419: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        add_3424: "f32[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2629, unsqueeze_419);  mul_2629 = unsqueeze_419 = None
        convert_element_type_314: "f16[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3424, torch.float16);  add_3424 = None
        relu_104: "f16[s28, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_314);  convert_element_type_314 = None
        convert_element_type_315: "f16[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_632, torch.float16);  primals_632 = None
        convolution_104: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_104, convert_element_type_315, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3450: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_633, 1)
        convert_element_type_316: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_104, torch.float32)
        var_mean_105 = torch.ops.aten.var_mean.correction(convert_element_type_316, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_316 = None
        getitem_212: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_105[0]
        getitem_213: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_105[1];  var_mean_105 = None
        add_3451: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_212, 1e-05)
        rsqrt_105: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3451);  add_3451 = None
        sub_795: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_104, getitem_213)
        mul_2647: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_795, rsqrt_105);  sub_795 = None
        squeeze_315: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3]);  getitem_213 = None
        squeeze_316: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_105, [0, 2, 3]);  rsqrt_105 = None
        mul_2648: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_315, 0.1)
        mul_2649: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_634, 0.9)
        add_3452: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2648, mul_2649);  mul_2648 = mul_2649 = None
        sym_numel_default_105: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_104)
        truediv_210: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_105 / 128;  sym_numel_default_105 = None
        squeeze_317: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
        sub_796: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_210 - 1.0
        truediv_211: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_210 / sub_796;  truediv_210 = sub_796 = None
        mul_2650: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_317, truediv_211);  squeeze_317 = truediv_211 = None
        mul_2651: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2650, 0.1);  mul_2650 = None
        mul_2652: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_635, 0.9)
        add_3453: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2651, mul_2652);  mul_2651 = mul_2652 = None
        unsqueeze_420: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_636, -1)
        unsqueeze_421: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_2653: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2647, unsqueeze_421);  mul_2647 = unsqueeze_421 = None
        unsqueeze_422: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_637, -1);  primals_637 = None
        unsqueeze_423: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_3454: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2653, unsqueeze_423);  mul_2653 = unsqueeze_423 = None
        convert_element_type_317: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3454, torch.float16);  add_3454 = None
        relu_105: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_317);  convert_element_type_317 = None
        convert_element_type_318: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_638, torch.float16);  primals_638 = None
        convolution_105: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_105, convert_element_type_318, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_50: "f16[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3485: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_639, 1)
        convert_element_type_319: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_50, torch.float32)
        var_mean_106 = torch.ops.aten.var_mean.correction(convert_element_type_319, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_319 = None
        getitem_214: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = var_mean_106[0]
        getitem_215: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = var_mean_106[1];  var_mean_106 = None
        add_3486: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_214, 1e-05)
        rsqrt_106: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3486);  add_3486 = None
        sub_803: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_50, getitem_215)
        mul_2673: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_803, rsqrt_106);  sub_803 = None
        squeeze_318: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_215, [0, 2, 3]);  getitem_215 = None
        squeeze_319: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_106, [0, 2, 3]);  rsqrt_106 = None
        mul_2674: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_318, 0.1)
        mul_2675: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_640, 0.9)
        add_3487: "f32[800][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2674, mul_2675);  mul_2674 = mul_2675 = None
        sym_numel_default_106: "Sym(39200 * s28)" = torch.ops.aten.sym_numel.default(cat_50)
        truediv_212: "Sym(IntTrueDiv(39200*s28, 800))" = sym_numel_default_106 / 800;  sym_numel_default_106 = None
        squeeze_320: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_214, [0, 2, 3]);  getitem_214 = None
        sub_804: "Sym(-1.00000000000000 + IntTrueDiv(39200*s28, 800))" = truediv_212 - 1.0
        truediv_213: "Sym(FloatTrueDiv(IntTrueDiv(39200*s28, 800), (IntTrueDiv(39200*s28, 800)) - 1.0))" = truediv_212 / sub_804;  truediv_212 = sub_804 = None
        mul_2676: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_320, truediv_213);  squeeze_320 = truediv_213 = None
        mul_2677: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2676, 0.1);  mul_2676 = None
        mul_2678: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_641, 0.9)
        add_3488: "f32[800][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2677, mul_2678);  mul_2677 = mul_2678 = None
        unsqueeze_424: "f32[800, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_642, -1)
        unsqueeze_425: "f32[800, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        mul_2679: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2673, unsqueeze_425);  mul_2673 = unsqueeze_425 = None
        unsqueeze_426: "f32[800, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_643, -1);  primals_643 = None
        unsqueeze_427: "f32[800, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        add_3489: "f32[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2679, unsqueeze_427);  mul_2679 = unsqueeze_427 = None
        convert_element_type_320: "f16[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3489, torch.float16);  add_3489 = None
        relu_106: "f16[s28, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_320);  convert_element_type_320 = None
        convert_element_type_321: "f16[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_644, torch.float16);  primals_644 = None
        convolution_106: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_106, convert_element_type_321, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3515: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_645, 1)
        convert_element_type_322: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_106, torch.float32)
        var_mean_107 = torch.ops.aten.var_mean.correction(convert_element_type_322, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_322 = None
        getitem_216: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_107[0]
        getitem_217: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_107[1];  var_mean_107 = None
        add_3516: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_216, 1e-05)
        rsqrt_107: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3516);  add_3516 = None
        sub_810: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_106, getitem_217)
        mul_2697: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_810, rsqrt_107);  sub_810 = None
        squeeze_321: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_217, [0, 2, 3]);  getitem_217 = None
        squeeze_322: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_107, [0, 2, 3]);  rsqrt_107 = None
        mul_2698: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_321, 0.1)
        mul_2699: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_646, 0.9)
        add_3517: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2698, mul_2699);  mul_2698 = mul_2699 = None
        sym_numel_default_107: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_106)
        truediv_214: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_107 / 128;  sym_numel_default_107 = None
        squeeze_323: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_216, [0, 2, 3]);  getitem_216 = None
        sub_811: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_214 - 1.0
        truediv_215: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_214 / sub_811;  truediv_214 = sub_811 = None
        mul_2700: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_323, truediv_215);  squeeze_323 = truediv_215 = None
        mul_2701: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2700, 0.1);  mul_2700 = None
        mul_2702: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_647, 0.9)
        add_3518: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2701, mul_2702);  mul_2701 = mul_2702 = None
        unsqueeze_428: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_648, -1)
        unsqueeze_429: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_2703: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2697, unsqueeze_429);  mul_2697 = unsqueeze_429 = None
        unsqueeze_430: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_649, -1);  primals_649 = None
        unsqueeze_431: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_3519: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2703, unsqueeze_431);  mul_2703 = unsqueeze_431 = None
        convert_element_type_323: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3519, torch.float16);  add_3519 = None
        relu_107: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_323);  convert_element_type_323 = None
        convert_element_type_324: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_650, torch.float16);  primals_650 = None
        convolution_107: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_107, convert_element_type_324, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_51: "f16[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3550: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_651, 1)
        convert_element_type_325: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_51, torch.float32)
        var_mean_108 = torch.ops.aten.var_mean.correction(convert_element_type_325, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_325 = None
        getitem_218: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = var_mean_108[0]
        getitem_219: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = var_mean_108[1];  var_mean_108 = None
        add_3551: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_218, 1e-05)
        rsqrt_108: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3551);  add_3551 = None
        sub_818: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_51, getitem_219)
        mul_2723: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_818, rsqrt_108);  sub_818 = None
        squeeze_324: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_219, [0, 2, 3]);  getitem_219 = None
        squeeze_325: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_108, [0, 2, 3]);  rsqrt_108 = None
        mul_2724: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_324, 0.1)
        mul_2725: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_652, 0.9)
        add_3552: "f32[832][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2724, mul_2725);  mul_2724 = mul_2725 = None
        sym_numel_default_108: "Sym(40768 * s28)" = torch.ops.aten.sym_numel.default(cat_51)
        truediv_216: "Sym(IntTrueDiv(40768*s28, 832))" = sym_numel_default_108 / 832;  sym_numel_default_108 = None
        squeeze_326: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_218, [0, 2, 3]);  getitem_218 = None
        sub_819: "Sym(-1.00000000000000 + IntTrueDiv(40768*s28, 832))" = truediv_216 - 1.0
        truediv_217: "Sym(FloatTrueDiv(IntTrueDiv(40768*s28, 832), (IntTrueDiv(40768*s28, 832)) - 1.0))" = truediv_216 / sub_819;  truediv_216 = sub_819 = None
        mul_2726: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_326, truediv_217);  squeeze_326 = truediv_217 = None
        mul_2727: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2726, 0.1);  mul_2726 = None
        mul_2728: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_653, 0.9)
        add_3553: "f32[832][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2727, mul_2728);  mul_2727 = mul_2728 = None
        unsqueeze_432: "f32[832, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_654, -1)
        unsqueeze_433: "f32[832, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        mul_2729: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2723, unsqueeze_433);  mul_2723 = unsqueeze_433 = None
        unsqueeze_434: "f32[832, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_655, -1);  primals_655 = None
        unsqueeze_435: "f32[832, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        add_3554: "f32[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2729, unsqueeze_435);  mul_2729 = unsqueeze_435 = None
        convert_element_type_326: "f16[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3554, torch.float16);  add_3554 = None
        relu_108: "f16[s28, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_326);  convert_element_type_326 = None
        convert_element_type_327: "f16[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_656, torch.float16);  primals_656 = None
        convolution_108: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_108, convert_element_type_327, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3580: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_657, 1)
        convert_element_type_328: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_108, torch.float32)
        var_mean_109 = torch.ops.aten.var_mean.correction(convert_element_type_328, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_328 = None
        getitem_220: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_109[0]
        getitem_221: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_109[1];  var_mean_109 = None
        add_3581: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_220, 1e-05)
        rsqrt_109: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3581);  add_3581 = None
        sub_825: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_108, getitem_221)
        mul_2747: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_825, rsqrt_109);  sub_825 = None
        squeeze_327: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_221, [0, 2, 3]);  getitem_221 = None
        squeeze_328: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_109, [0, 2, 3]);  rsqrt_109 = None
        mul_2748: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_327, 0.1)
        mul_2749: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_658, 0.9)
        add_3582: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2748, mul_2749);  mul_2748 = mul_2749 = None
        sym_numel_default_109: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_108)
        truediv_218: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_109 / 128;  sym_numel_default_109 = None
        squeeze_329: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_220, [0, 2, 3]);  getitem_220 = None
        sub_826: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_218 - 1.0
        truediv_219: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_218 / sub_826;  truediv_218 = sub_826 = None
        mul_2750: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_329, truediv_219);  squeeze_329 = truediv_219 = None
        mul_2751: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2750, 0.1);  mul_2750 = None
        mul_2752: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_659, 0.9)
        add_3583: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2751, mul_2752);  mul_2751 = mul_2752 = None
        unsqueeze_436: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_660, -1)
        unsqueeze_437: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_2753: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2747, unsqueeze_437);  mul_2747 = unsqueeze_437 = None
        unsqueeze_438: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_661, -1);  primals_661 = None
        unsqueeze_439: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_3584: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2753, unsqueeze_439);  mul_2753 = unsqueeze_439 = None
        convert_element_type_329: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3584, torch.float16);  add_3584 = None
        relu_109: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_329);  convert_element_type_329 = None
        convert_element_type_330: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_662, torch.float16);  primals_662 = None
        convolution_109: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_109, convert_element_type_330, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_52: "f16[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3615: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_663, 1)
        convert_element_type_331: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_52, torch.float32)
        var_mean_110 = torch.ops.aten.var_mean.correction(convert_element_type_331, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_331 = None
        getitem_222: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = var_mean_110[0]
        getitem_223: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = var_mean_110[1];  var_mean_110 = None
        add_3616: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_222, 1e-05)
        rsqrt_110: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3616);  add_3616 = None
        sub_833: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_52, getitem_223)
        mul_2773: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_833, rsqrt_110);  sub_833 = None
        squeeze_330: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
        squeeze_331: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_110, [0, 2, 3]);  rsqrt_110 = None
        mul_2774: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_330, 0.1)
        mul_2775: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_664, 0.9)
        add_3617: "f32[864][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2774, mul_2775);  mul_2774 = mul_2775 = None
        sym_numel_default_110: "Sym(42336 * s28)" = torch.ops.aten.sym_numel.default(cat_52)
        truediv_220: "Sym(IntTrueDiv(42336*s28, 864))" = sym_numel_default_110 / 864;  sym_numel_default_110 = None
        squeeze_332: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
        sub_834: "Sym(-1.00000000000000 + IntTrueDiv(42336*s28, 864))" = truediv_220 - 1.0
        truediv_221: "Sym(FloatTrueDiv(IntTrueDiv(42336*s28, 864), (IntTrueDiv(42336*s28, 864)) - 1.0))" = truediv_220 / sub_834;  truediv_220 = sub_834 = None
        mul_2776: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_332, truediv_221);  squeeze_332 = truediv_221 = None
        mul_2777: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2776, 0.1);  mul_2776 = None
        mul_2778: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_665, 0.9)
        add_3618: "f32[864][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2777, mul_2778);  mul_2777 = mul_2778 = None
        unsqueeze_440: "f32[864, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_666, -1)
        unsqueeze_441: "f32[864, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        mul_2779: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2773, unsqueeze_441);  mul_2773 = unsqueeze_441 = None
        unsqueeze_442: "f32[864, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_667, -1);  primals_667 = None
        unsqueeze_443: "f32[864, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        add_3619: "f32[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2779, unsqueeze_443);  mul_2779 = unsqueeze_443 = None
        convert_element_type_332: "f16[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3619, torch.float16);  add_3619 = None
        relu_110: "f16[s28, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_332);  convert_element_type_332 = None
        convert_element_type_333: "f16[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_668, torch.float16);  primals_668 = None
        convolution_110: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_110, convert_element_type_333, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3645: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_669, 1)
        convert_element_type_334: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_110, torch.float32)
        var_mean_111 = torch.ops.aten.var_mean.correction(convert_element_type_334, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_334 = None
        getitem_224: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_111[0]
        getitem_225: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_111[1];  var_mean_111 = None
        add_3646: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_224, 1e-05)
        rsqrt_111: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3646);  add_3646 = None
        sub_840: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_110, getitem_225)
        mul_2797: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_840, rsqrt_111);  sub_840 = None
        squeeze_333: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_225, [0, 2, 3]);  getitem_225 = None
        squeeze_334: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_111, [0, 2, 3]);  rsqrt_111 = None
        mul_2798: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_333, 0.1)
        mul_2799: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_670, 0.9)
        add_3647: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2798, mul_2799);  mul_2798 = mul_2799 = None
        sym_numel_default_111: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_110)
        truediv_222: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_111 / 128;  sym_numel_default_111 = None
        squeeze_335: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_224, [0, 2, 3]);  getitem_224 = None
        sub_841: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_222 - 1.0
        truediv_223: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_222 / sub_841;  truediv_222 = sub_841 = None
        mul_2800: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_335, truediv_223);  squeeze_335 = truediv_223 = None
        mul_2801: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2800, 0.1);  mul_2800 = None
        mul_2802: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_671, 0.9)
        add_3648: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2801, mul_2802);  mul_2801 = mul_2802 = None
        unsqueeze_444: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_672, -1)
        unsqueeze_445: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_2803: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2797, unsqueeze_445);  mul_2797 = unsqueeze_445 = None
        unsqueeze_446: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_673, -1);  primals_673 = None
        unsqueeze_447: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_3649: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2803, unsqueeze_447);  mul_2803 = unsqueeze_447 = None
        convert_element_type_335: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3649, torch.float16);  add_3649 = None
        relu_111: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_335);  convert_element_type_335 = None
        convert_element_type_336: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_674, torch.float16);  primals_674 = None
        convolution_111: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_111, convert_element_type_336, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_53: "f16[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3680: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_675, 1)
        convert_element_type_337: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_53, torch.float32)
        var_mean_112 = torch.ops.aten.var_mean.correction(convert_element_type_337, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_337 = None
        getitem_226: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = var_mean_112[0]
        getitem_227: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = var_mean_112[1];  var_mean_112 = None
        add_3681: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_226, 1e-05)
        rsqrt_112: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3681);  add_3681 = None
        sub_848: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_53, getitem_227)
        mul_2823: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_848, rsqrt_112);  sub_848 = None
        squeeze_336: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
        squeeze_337: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_112, [0, 2, 3]);  rsqrt_112 = None
        mul_2824: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_336, 0.1)
        mul_2825: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_676, 0.9)
        add_3682: "f32[896][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2824, mul_2825);  mul_2824 = mul_2825 = None
        sym_numel_default_112: "Sym(43904 * s28)" = torch.ops.aten.sym_numel.default(cat_53)
        truediv_224: "Sym(IntTrueDiv(43904*s28, 896))" = sym_numel_default_112 / 896;  sym_numel_default_112 = None
        squeeze_338: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_226, [0, 2, 3]);  getitem_226 = None
        sub_849: "Sym(-1.00000000000000 + IntTrueDiv(43904*s28, 896))" = truediv_224 - 1.0
        truediv_225: "Sym(FloatTrueDiv(IntTrueDiv(43904*s28, 896), (IntTrueDiv(43904*s28, 896)) - 1.0))" = truediv_224 / sub_849;  truediv_224 = sub_849 = None
        mul_2826: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_338, truediv_225);  squeeze_338 = truediv_225 = None
        mul_2827: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2826, 0.1);  mul_2826 = None
        mul_2828: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_677, 0.9)
        add_3683: "f32[896][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2827, mul_2828);  mul_2827 = mul_2828 = None
        unsqueeze_448: "f32[896, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_678, -1)
        unsqueeze_449: "f32[896, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        mul_2829: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2823, unsqueeze_449);  mul_2823 = unsqueeze_449 = None
        unsqueeze_450: "f32[896, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_679, -1);  primals_679 = None
        unsqueeze_451: "f32[896, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        add_3684: "f32[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2829, unsqueeze_451);  mul_2829 = unsqueeze_451 = None
        convert_element_type_338: "f16[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3684, torch.float16);  add_3684 = None
        relu_112: "f16[s28, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_338);  convert_element_type_338 = None
        convert_element_type_339: "f16[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_680, torch.float16);  primals_680 = None
        convolution_112: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_112, convert_element_type_339, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3710: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_681, 1)
        convert_element_type_340: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_112, torch.float32)
        var_mean_113 = torch.ops.aten.var_mean.correction(convert_element_type_340, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_340 = None
        getitem_228: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_113[0]
        getitem_229: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_113[1];  var_mean_113 = None
        add_3711: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_228, 1e-05)
        rsqrt_113: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3711);  add_3711 = None
        sub_855: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_112, getitem_229)
        mul_2847: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_855, rsqrt_113);  sub_855 = None
        squeeze_339: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
        squeeze_340: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_113, [0, 2, 3]);  rsqrt_113 = None
        mul_2848: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_339, 0.1)
        mul_2849: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_682, 0.9)
        add_3712: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2848, mul_2849);  mul_2848 = mul_2849 = None
        sym_numel_default_113: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_112)
        truediv_226: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_113 / 128;  sym_numel_default_113 = None
        squeeze_341: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
        sub_856: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_226 - 1.0
        truediv_227: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_226 / sub_856;  truediv_226 = sub_856 = None
        mul_2850: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_341, truediv_227);  squeeze_341 = truediv_227 = None
        mul_2851: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2850, 0.1);  mul_2850 = None
        mul_2852: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_683, 0.9)
        add_3713: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2851, mul_2852);  mul_2851 = mul_2852 = None
        unsqueeze_452: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_684, -1)
        unsqueeze_453: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_2853: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2847, unsqueeze_453);  mul_2847 = unsqueeze_453 = None
        unsqueeze_454: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_685, -1);  primals_685 = None
        unsqueeze_455: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_3714: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2853, unsqueeze_455);  mul_2853 = unsqueeze_455 = None
        convert_element_type_341: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3714, torch.float16);  add_3714 = None
        relu_113: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_341);  convert_element_type_341 = None
        convert_element_type_342: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_686, torch.float16);  primals_686 = None
        convolution_113: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_113, convert_element_type_342, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_54: "f16[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3745: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_687, 1)
        convert_element_type_343: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_54, torch.float32)
        var_mean_114 = torch.ops.aten.var_mean.correction(convert_element_type_343, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_343 = None
        getitem_230: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = var_mean_114[0]
        getitem_231: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = var_mean_114[1];  var_mean_114 = None
        add_3746: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_230, 1e-05)
        rsqrt_114: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3746);  add_3746 = None
        sub_863: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_54, getitem_231)
        mul_2873: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_863, rsqrt_114);  sub_863 = None
        squeeze_342: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_231, [0, 2, 3]);  getitem_231 = None
        squeeze_343: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_114, [0, 2, 3]);  rsqrt_114 = None
        mul_2874: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_342, 0.1)
        mul_2875: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_688, 0.9)
        add_3747: "f32[928][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2874, mul_2875);  mul_2874 = mul_2875 = None
        sym_numel_default_114: "Sym(45472 * s28)" = torch.ops.aten.sym_numel.default(cat_54)
        truediv_228: "Sym(IntTrueDiv(45472*s28, 928))" = sym_numel_default_114 / 928;  sym_numel_default_114 = None
        squeeze_344: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_230, [0, 2, 3]);  getitem_230 = None
        sub_864: "Sym(-1.00000000000000 + IntTrueDiv(45472*s28, 928))" = truediv_228 - 1.0
        truediv_229: "Sym(FloatTrueDiv(IntTrueDiv(45472*s28, 928), (IntTrueDiv(45472*s28, 928)) - 1.0))" = truediv_228 / sub_864;  truediv_228 = sub_864 = None
        mul_2876: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_344, truediv_229);  squeeze_344 = truediv_229 = None
        mul_2877: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2876, 0.1);  mul_2876 = None
        mul_2878: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_689, 0.9)
        add_3748: "f32[928][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2877, mul_2878);  mul_2877 = mul_2878 = None
        unsqueeze_456: "f32[928, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_690, -1)
        unsqueeze_457: "f32[928, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        mul_2879: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2873, unsqueeze_457);  mul_2873 = unsqueeze_457 = None
        unsqueeze_458: "f32[928, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_691, -1);  primals_691 = None
        unsqueeze_459: "f32[928, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        add_3749: "f32[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2879, unsqueeze_459);  mul_2879 = unsqueeze_459 = None
        convert_element_type_344: "f16[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3749, torch.float16);  add_3749 = None
        relu_114: "f16[s28, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_344);  convert_element_type_344 = None
        convert_element_type_345: "f16[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_692, torch.float16);  primals_692 = None
        convolution_114: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_114, convert_element_type_345, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3775: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_693, 1)
        convert_element_type_346: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_114, torch.float32)
        var_mean_115 = torch.ops.aten.var_mean.correction(convert_element_type_346, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_346 = None
        getitem_232: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_115[0]
        getitem_233: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_115[1];  var_mean_115 = None
        add_3776: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_232, 1e-05)
        rsqrt_115: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3776);  add_3776 = None
        sub_870: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_114, getitem_233)
        mul_2897: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_870, rsqrt_115);  sub_870 = None
        squeeze_345: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_233, [0, 2, 3]);  getitem_233 = None
        squeeze_346: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_115, [0, 2, 3]);  rsqrt_115 = None
        mul_2898: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_345, 0.1)
        mul_2899: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_694, 0.9)
        add_3777: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2898, mul_2899);  mul_2898 = mul_2899 = None
        sym_numel_default_115: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_114)
        truediv_230: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_115 / 128;  sym_numel_default_115 = None
        squeeze_347: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_232, [0, 2, 3]);  getitem_232 = None
        sub_871: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_230 - 1.0
        truediv_231: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_230 / sub_871;  truediv_230 = sub_871 = None
        mul_2900: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_347, truediv_231);  squeeze_347 = truediv_231 = None
        mul_2901: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2900, 0.1);  mul_2900 = None
        mul_2902: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_695, 0.9)
        add_3778: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2901, mul_2902);  mul_2901 = mul_2902 = None
        unsqueeze_460: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_696, -1)
        unsqueeze_461: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_2903: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2897, unsqueeze_461);  mul_2897 = unsqueeze_461 = None
        unsqueeze_462: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_697, -1);  primals_697 = None
        unsqueeze_463: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_3779: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2903, unsqueeze_463);  mul_2903 = unsqueeze_463 = None
        convert_element_type_347: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3779, torch.float16);  add_3779 = None
        relu_115: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_347);  convert_element_type_347 = None
        convert_element_type_348: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_698, torch.float16);  primals_698 = None
        convolution_115: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_115, convert_element_type_348, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_55: "f16[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113, convolution_115], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3810: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_699, 1)
        convert_element_type_349: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_55, torch.float32)
        var_mean_116 = torch.ops.aten.var_mean.correction(convert_element_type_349, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_349 = None
        getitem_234: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_116[0]
        getitem_235: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_116[1];  var_mean_116 = None
        add_3811: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_234, 1e-05)
        rsqrt_116: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3811);  add_3811 = None
        sub_878: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_55, getitem_235)
        mul_2923: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_878, rsqrt_116);  sub_878 = None
        squeeze_348: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_235, [0, 2, 3]);  getitem_235 = None
        squeeze_349: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_116, [0, 2, 3]);  rsqrt_116 = None
        mul_2924: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_348, 0.1)
        mul_2925: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_700, 0.9)
        add_3812: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2924, mul_2925);  mul_2924 = mul_2925 = None
        sym_numel_default_116: "Sym(47040 * s28)" = torch.ops.aten.sym_numel.default(cat_55)
        truediv_232: "Sym(IntTrueDiv(47040*s28, 960))" = sym_numel_default_116 / 960;  sym_numel_default_116 = None
        squeeze_350: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_234, [0, 2, 3]);  getitem_234 = None
        sub_879: "Sym(-1.00000000000000 + IntTrueDiv(47040*s28, 960))" = truediv_232 - 1.0
        truediv_233: "Sym(FloatTrueDiv(IntTrueDiv(47040*s28, 960), (IntTrueDiv(47040*s28, 960)) - 1.0))" = truediv_232 / sub_879;  truediv_232 = sub_879 = None
        mul_2926: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_350, truediv_233);  squeeze_350 = truediv_233 = None
        mul_2927: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2926, 0.1);  mul_2926 = None
        mul_2928: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_701, 0.9)
        add_3813: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2927, mul_2928);  mul_2927 = mul_2928 = None
        unsqueeze_464: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_702, -1)
        unsqueeze_465: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        mul_2929: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2923, unsqueeze_465);  mul_2923 = unsqueeze_465 = None
        unsqueeze_466: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_703, -1);  primals_703 = None
        unsqueeze_467: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        add_3814: "f32[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2929, unsqueeze_467);  mul_2929 = unsqueeze_467 = None
        convert_element_type_350: "f16[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3814, torch.float16);  add_3814 = None
        relu_116: "f16[s28, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_350);  convert_element_type_350 = None
        convert_element_type_351: "f16[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_704, torch.float16);  primals_704 = None
        convolution_116: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_116, convert_element_type_351, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3840: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_705, 1)
        convert_element_type_352: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_116, torch.float32)
        var_mean_117 = torch.ops.aten.var_mean.correction(convert_element_type_352, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_352 = None
        getitem_236: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_117[0]
        getitem_237: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_117[1];  var_mean_117 = None
        add_3841: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_236, 1e-05)
        rsqrt_117: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3841);  add_3841 = None
        sub_885: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_116, getitem_237)
        mul_2947: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_885, rsqrt_117);  sub_885 = None
        squeeze_351: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_237, [0, 2, 3]);  getitem_237 = None
        squeeze_352: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_117, [0, 2, 3]);  rsqrt_117 = None
        mul_2948: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_351, 0.1)
        mul_2949: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_706, 0.9)
        add_3842: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2948, mul_2949);  mul_2948 = mul_2949 = None
        sym_numel_default_117: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_116)
        truediv_234: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_117 / 128;  sym_numel_default_117 = None
        squeeze_353: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_236, [0, 2, 3]);  getitem_236 = None
        sub_886: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_234 - 1.0
        truediv_235: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_234 / sub_886;  truediv_234 = sub_886 = None
        mul_2950: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_353, truediv_235);  squeeze_353 = truediv_235 = None
        mul_2951: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2950, 0.1);  mul_2950 = None
        mul_2952: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_707, 0.9)
        add_3843: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2951, mul_2952);  mul_2951 = mul_2952 = None
        unsqueeze_468: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_708, -1)
        unsqueeze_469: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_2953: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2947, unsqueeze_469);  mul_2947 = unsqueeze_469 = None
        unsqueeze_470: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_709, -1);  primals_709 = None
        unsqueeze_471: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_3844: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2953, unsqueeze_471);  mul_2953 = unsqueeze_471 = None
        convert_element_type_353: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3844, torch.float16);  add_3844 = None
        relu_117: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_353);  convert_element_type_353 = None
        convert_element_type_354: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_710, torch.float16);  primals_710 = None
        convolution_117: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_117, convert_element_type_354, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_56: "f16[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113, convolution_115, convolution_117], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_3875: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_711, 1)
        convert_element_type_355: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_56, torch.float32)
        var_mean_118 = torch.ops.aten.var_mean.correction(convert_element_type_355, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_355 = None
        getitem_238: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = var_mean_118[0]
        getitem_239: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = var_mean_118[1];  var_mean_118 = None
        add_3876: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_238, 1e-05)
        rsqrt_118: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3876);  add_3876 = None
        sub_893: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_56, getitem_239)
        mul_2973: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_893, rsqrt_118);  sub_893 = None
        squeeze_354: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_239, [0, 2, 3]);  getitem_239 = None
        squeeze_355: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_118, [0, 2, 3]);  rsqrt_118 = None
        mul_2974: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_354, 0.1)
        mul_2975: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_712, 0.9)
        add_3877: "f32[992][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2974, mul_2975);  mul_2974 = mul_2975 = None
        sym_numel_default_118: "Sym(48608 * s28)" = torch.ops.aten.sym_numel.default(cat_56)
        truediv_236: "Sym(IntTrueDiv(48608*s28, 992))" = sym_numel_default_118 / 992;  sym_numel_default_118 = None
        squeeze_356: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_238, [0, 2, 3]);  getitem_238 = None
        sub_894: "Sym(-1.00000000000000 + IntTrueDiv(48608*s28, 992))" = truediv_236 - 1.0
        truediv_237: "Sym(FloatTrueDiv(IntTrueDiv(48608*s28, 992), (IntTrueDiv(48608*s28, 992)) - 1.0))" = truediv_236 / sub_894;  truediv_236 = sub_894 = None
        mul_2976: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_356, truediv_237);  squeeze_356 = truediv_237 = None
        mul_2977: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2976, 0.1);  mul_2976 = None
        mul_2978: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_713, 0.9)
        add_3878: "f32[992][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2977, mul_2978);  mul_2977 = mul_2978 = None
        unsqueeze_472: "f32[992, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_714, -1)
        unsqueeze_473: "f32[992, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        mul_2979: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2973, unsqueeze_473);  mul_2973 = unsqueeze_473 = None
        unsqueeze_474: "f32[992, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_715, -1);  primals_715 = None
        unsqueeze_475: "f32[992, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        add_3879: "f32[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_2979, unsqueeze_475);  mul_2979 = unsqueeze_475 = None
        convert_element_type_356: "f16[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3879, torch.float16);  add_3879 = None
        relu_118: "f16[s28, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_356);  convert_element_type_356 = None
        convert_element_type_357: "f16[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_716, torch.float16);  primals_716 = None
        convolution_118: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_118, convert_element_type_357, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_3905: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_717, 1)
        convert_element_type_358: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_118, torch.float32)
        var_mean_119 = torch.ops.aten.var_mean.correction(convert_element_type_358, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_358 = None
        getitem_240: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_119[0]
        getitem_241: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_119[1];  var_mean_119 = None
        add_3906: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_240, 1e-05)
        rsqrt_119: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3906);  add_3906 = None
        sub_900: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_118, getitem_241)
        mul_2997: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_900, rsqrt_119);  sub_900 = None
        squeeze_357: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_241, [0, 2, 3]);  getitem_241 = None
        squeeze_358: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_119, [0, 2, 3]);  rsqrt_119 = None
        mul_2998: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_357, 0.1)
        mul_2999: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_718, 0.9)
        add_3907: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_2998, mul_2999);  mul_2998 = mul_2999 = None
        sym_numel_default_119: "Sym(6272 * s28)" = torch.ops.aten.sym_numel.default(convolution_118)
        truediv_238: "Sym(IntTrueDiv(6272*s28, 128))" = sym_numel_default_119 / 128;  sym_numel_default_119 = None
        squeeze_359: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_240, [0, 2, 3]);  getitem_240 = None
        sub_901: "Sym(-1.00000000000000 + IntTrueDiv(6272*s28, 128))" = truediv_238 - 1.0
        truediv_239: "Sym(FloatTrueDiv(IntTrueDiv(6272*s28, 128), (IntTrueDiv(6272*s28, 128)) - 1.0))" = truediv_238 / sub_901;  truediv_238 = sub_901 = None
        mul_3000: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_359, truediv_239);  squeeze_359 = truediv_239 = None
        mul_3001: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3000, 0.1);  mul_3000 = None
        mul_3002: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_719, 0.9)
        add_3908: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_3001, mul_3002);  mul_3001 = mul_3002 = None
        unsqueeze_476: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_720, -1)
        unsqueeze_477: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_3003: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2997, unsqueeze_477);  mul_2997 = unsqueeze_477 = None
        unsqueeze_478: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_721, -1);  primals_721 = None
        unsqueeze_479: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_3909: "f32[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3003, unsqueeze_479);  mul_3003 = unsqueeze_479 = None
        convert_element_type_359: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3909, torch.float16);  add_3909 = None
        relu_119: "f16[s28, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_359);  convert_element_type_359 = None
        convert_element_type_360: "f16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_722, torch.float16);  primals_722 = None
        convolution_119: "f16[s28, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_119, convert_element_type_360, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        cat_57: "f16[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113, convolution_115, convolution_117, convolution_119], 1);  convolution_89 = convolution_91 = convolution_93 = convolution_95 = convolution_97 = convolution_99 = convolution_101 = convolution_103 = convolution_105 = convolution_107 = convolution_109 = convolution_111 = convolution_113 = convolution_115 = convolution_117 = convolution_119 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        add_3940: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_723, 1)
        convert_element_type_361: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_57, torch.float32)
        var_mean_120 = torch.ops.aten.var_mean.correction(convert_element_type_361, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_361 = None
        getitem_242: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_120[0]
        getitem_243: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_120[1];  var_mean_120 = None
        add_3941: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_242, 1e-05)
        rsqrt_120: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_3941);  add_3941 = None
        sub_908: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_57, getitem_243)
        mul_3023: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_908, rsqrt_120);  sub_908 = None
        squeeze_360: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_243, [0, 2, 3])
        mul_3024: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_360, 0.1);  squeeze_360 = None
        mul_3025: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_724, 0.9)
        add_3942: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_3024, mul_3025);  mul_3024 = mul_3025 = None
        sym_numel_default_120: "Sym(50176 * s28)" = torch.ops.aten.sym_numel.default(cat_57)
        truediv_240: "Sym(IntTrueDiv(50176*s28, 1024))" = sym_numel_default_120 / 1024;  sym_numel_default_120 = None
        squeeze_362: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_242, [0, 2, 3]);  getitem_242 = None
        sub_909: "Sym(-1.00000000000000 + IntTrueDiv(50176*s28, 1024))" = truediv_240 - 1.0
        truediv_241: "Sym(FloatTrueDiv(IntTrueDiv(50176*s28, 1024), (IntTrueDiv(50176*s28, 1024)) - 1.0))" = truediv_240 / sub_909;  truediv_240 = sub_909 = None
        mul_3026: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_362, truediv_241);  squeeze_362 = truediv_241 = None
        mul_3027: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3026, 0.1);  mul_3026 = None
        mul_3028: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_725, 0.9)
        add_3943: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_3027, mul_3028);  mul_3027 = mul_3028 = None
        unsqueeze_480: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_726, -1)
        unsqueeze_481: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        mul_3029: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3023, unsqueeze_481);  mul_3023 = unsqueeze_481 = None
        unsqueeze_482: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_727, -1)
        unsqueeze_483: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        add_3944: "f32[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_3029, unsqueeze_483);  mul_3029 = unsqueeze_483 = None
        convert_element_type_362: "f16[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3944, torch.float16);  add_3944 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:214 in forward, code: out = F.relu(features, inplace=True)
        relu_120: "f16[s28, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_362);  convert_element_type_362 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:215 in forward, code: out = F.adaptive_avg_pool2d(out, (1, 1))
        mean: "f16[s28, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_120, [-1, -2], True);  relu_120 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:216 in forward, code: out = torch.flatten(out, 1)
        view: "f16[s28, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mean, [primals_2, 1024]);  mean = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:217 in forward, code: out = self.classifier(out)
        convert_element_type_363: "f16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_729, torch.float16);  primals_729 = None
        convert_element_type_364: "f16[1000, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_728, torch.float16);  primals_728 = None
        permute: "f16[1024, 1000][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_364, [1, 0]);  convert_element_type_364 = None
        addmm: "f16[s28, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_363, view, permute);  convert_element_type_363 = None
        permute_1: "f16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_496: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_357, 0);  squeeze_357 = None
        unsqueeze_497: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_508: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_354, 0);  squeeze_354 = None
        unsqueeze_509: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_520: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_351, 0);  squeeze_351 = None
        unsqueeze_521: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_532: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_348, 0);  squeeze_348 = None
        unsqueeze_533: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
        unsqueeze_534: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_544: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_345, 0);  squeeze_345 = None
        unsqueeze_545: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
        unsqueeze_546: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_556: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_342, 0);  squeeze_342 = None
        unsqueeze_557: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
        unsqueeze_558: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_568: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_339, 0);  squeeze_339 = None
        unsqueeze_569: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
        unsqueeze_570: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_580: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_336, 0);  squeeze_336 = None
        unsqueeze_581: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
        unsqueeze_582: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_592: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_333, 0);  squeeze_333 = None
        unsqueeze_593: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
        unsqueeze_594: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_604: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_330, 0);  squeeze_330 = None
        unsqueeze_605: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
        unsqueeze_606: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_616: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_327, 0);  squeeze_327 = None
        unsqueeze_617: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
        unsqueeze_618: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_628: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_324, 0);  squeeze_324 = None
        unsqueeze_629: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
        unsqueeze_630: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_640: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_321, 0);  squeeze_321 = None
        unsqueeze_641: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
        unsqueeze_642: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_652: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_318, 0);  squeeze_318 = None
        unsqueeze_653: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
        unsqueeze_654: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_664: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_315, 0);  squeeze_315 = None
        unsqueeze_665: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
        unsqueeze_666: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_676: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_312, 0);  squeeze_312 = None
        unsqueeze_677: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
        unsqueeze_678: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_688: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_309, 0);  squeeze_309 = None
        unsqueeze_689: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
        unsqueeze_690: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_700: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_306, 0);  squeeze_306 = None
        unsqueeze_701: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
        unsqueeze_702: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_712: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_303, 0);  squeeze_303 = None
        unsqueeze_713: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
        unsqueeze_714: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_724: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_300, 0);  squeeze_300 = None
        unsqueeze_725: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
        unsqueeze_726: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_736: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_297, 0);  squeeze_297 = None
        unsqueeze_737: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
        unsqueeze_738: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_748: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_294, 0);  squeeze_294 = None
        unsqueeze_749: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
        unsqueeze_750: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_760: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_291, 0);  squeeze_291 = None
        unsqueeze_761: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
        unsqueeze_762: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_772: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_288, 0);  squeeze_288 = None
        unsqueeze_773: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
        unsqueeze_774: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_784: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
        unsqueeze_785: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
        unsqueeze_786: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_796: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
        unsqueeze_797: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
        unsqueeze_798: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_808: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
        unsqueeze_809: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
        unsqueeze_810: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_820: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
        unsqueeze_821: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
        unsqueeze_822: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_832: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
        unsqueeze_833: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
        unsqueeze_834: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_844: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
        unsqueeze_845: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
        unsqueeze_846: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_856: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
        unsqueeze_857: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
        unsqueeze_858: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_868: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
        unsqueeze_869: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
        unsqueeze_870: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        unsqueeze_880: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
        unsqueeze_881: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
        unsqueeze_882: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_892: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
        unsqueeze_893: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
        unsqueeze_894: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_904: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
        unsqueeze_905: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
        unsqueeze_906: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_916: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
        unsqueeze_917: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
        unsqueeze_918: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_928: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
        unsqueeze_929: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
        unsqueeze_930: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_940: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
        unsqueeze_941: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
        unsqueeze_942: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_952: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
        unsqueeze_953: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
        unsqueeze_954: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_964: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
        unsqueeze_965: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
        unsqueeze_966: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_976: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
        unsqueeze_977: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
        unsqueeze_978: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_988: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
        unsqueeze_989: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
        unsqueeze_990: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1000: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
        unsqueeze_1001: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
        unsqueeze_1002: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1012: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
        unsqueeze_1013: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
        unsqueeze_1014: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1024: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
        unsqueeze_1025: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
        unsqueeze_1026: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1036: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
        unsqueeze_1037: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
        unsqueeze_1038: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1048: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
        unsqueeze_1049: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
        unsqueeze_1050: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1060: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
        unsqueeze_1061: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 2);  unsqueeze_1060 = None
        unsqueeze_1062: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 3);  unsqueeze_1061 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1072: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
        unsqueeze_1073: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 2);  unsqueeze_1072 = None
        unsqueeze_1074: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 3);  unsqueeze_1073 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1084: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
        unsqueeze_1085: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 2);  unsqueeze_1084 = None
        unsqueeze_1086: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 3);  unsqueeze_1085 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1096: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
        unsqueeze_1097: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 2);  unsqueeze_1096 = None
        unsqueeze_1098: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 3);  unsqueeze_1097 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1108: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
        unsqueeze_1109: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 2);  unsqueeze_1108 = None
        unsqueeze_1110: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 3);  unsqueeze_1109 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1120: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
        unsqueeze_1121: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 2);  unsqueeze_1120 = None
        unsqueeze_1122: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 3);  unsqueeze_1121 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1132: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
        unsqueeze_1133: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 2);  unsqueeze_1132 = None
        unsqueeze_1134: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 3);  unsqueeze_1133 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1144: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
        unsqueeze_1145: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 2);  unsqueeze_1144 = None
        unsqueeze_1146: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 3);  unsqueeze_1145 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1156: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
        unsqueeze_1157: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 2);  unsqueeze_1156 = None
        unsqueeze_1158: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 3);  unsqueeze_1157 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1168: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
        unsqueeze_1169: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 2);  unsqueeze_1168 = None
        unsqueeze_1170: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 3);  unsqueeze_1169 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1180: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
        unsqueeze_1181: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 2);  unsqueeze_1180 = None
        unsqueeze_1182: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 3);  unsqueeze_1181 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1192: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
        unsqueeze_1193: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 2);  unsqueeze_1192 = None
        unsqueeze_1194: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 3);  unsqueeze_1193 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1204: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
        unsqueeze_1205: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 2);  unsqueeze_1204 = None
        unsqueeze_1206: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 3);  unsqueeze_1205 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1216: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
        unsqueeze_1217: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 2);  unsqueeze_1216 = None
        unsqueeze_1218: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 3);  unsqueeze_1217 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1228: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
        unsqueeze_1229: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 2);  unsqueeze_1228 = None
        unsqueeze_1230: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 3);  unsqueeze_1229 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1240: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
        unsqueeze_1241: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 2);  unsqueeze_1240 = None
        unsqueeze_1242: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 3);  unsqueeze_1241 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1252: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
        unsqueeze_1253: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 2);  unsqueeze_1252 = None
        unsqueeze_1254: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 3);  unsqueeze_1253 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1264: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
        unsqueeze_1265: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 2);  unsqueeze_1264 = None
        unsqueeze_1266: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 3);  unsqueeze_1265 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1276: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
        unsqueeze_1277: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 2);  unsqueeze_1276 = None
        unsqueeze_1278: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 3);  unsqueeze_1277 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1288: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
        unsqueeze_1289: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 2);  unsqueeze_1288 = None
        unsqueeze_1290: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 3);  unsqueeze_1289 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1300: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_1301: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 2);  unsqueeze_1300 = None
        unsqueeze_1302: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 3);  unsqueeze_1301 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1312: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_1313: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 2);  unsqueeze_1312 = None
        unsqueeze_1314: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 3);  unsqueeze_1313 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1324: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_1325: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 2);  unsqueeze_1324 = None
        unsqueeze_1326: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 3);  unsqueeze_1325 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1336: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_1337: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 2);  unsqueeze_1336 = None
        unsqueeze_1338: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 3);  unsqueeze_1337 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1348: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_1349: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 2);  unsqueeze_1348 = None
        unsqueeze_1350: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 3);  unsqueeze_1349 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1360: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_1361: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 2);  unsqueeze_1360 = None
        unsqueeze_1362: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 3);  unsqueeze_1361 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1372: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_1373: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 2);  unsqueeze_1372 = None
        unsqueeze_1374: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 3);  unsqueeze_1373 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1384: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_1385: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 2);  unsqueeze_1384 = None
        unsqueeze_1386: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 3);  unsqueeze_1385 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1396: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_1397: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, 2);  unsqueeze_1396 = None
        unsqueeze_1398: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 3);  unsqueeze_1397 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1408: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_1409: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, 2);  unsqueeze_1408 = None
        unsqueeze_1410: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 3);  unsqueeze_1409 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1420: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_1421: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, 2);  unsqueeze_1420 = None
        unsqueeze_1422: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 3);  unsqueeze_1421 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1432: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_1433: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 2);  unsqueeze_1432 = None
        unsqueeze_1434: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 3);  unsqueeze_1433 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1444: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_1445: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 2);  unsqueeze_1444 = None
        unsqueeze_1446: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 3);  unsqueeze_1445 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1456: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_1457: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 2);  unsqueeze_1456 = None
        unsqueeze_1458: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 3);  unsqueeze_1457 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        unsqueeze_1468: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_1469: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 2);  unsqueeze_1468 = None
        unsqueeze_1470: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 3);  unsqueeze_1469 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1480: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_1481: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, 2);  unsqueeze_1480 = None
        unsqueeze_1482: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 3);  unsqueeze_1481 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1492: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_1493: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, 2);  unsqueeze_1492 = None
        unsqueeze_1494: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 3);  unsqueeze_1493 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1504: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_1505: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, 2);  unsqueeze_1504 = None
        unsqueeze_1506: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1505, 3);  unsqueeze_1505 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1516: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_1517: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, 2);  unsqueeze_1516 = None
        unsqueeze_1518: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1517, 3);  unsqueeze_1517 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1528: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_1529: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, 2);  unsqueeze_1528 = None
        unsqueeze_1530: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1529, 3);  unsqueeze_1529 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1540: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_1541: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, 2);  unsqueeze_1540 = None
        unsqueeze_1542: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1541, 3);  unsqueeze_1541 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1552: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_1553: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, 2);  unsqueeze_1552 = None
        unsqueeze_1554: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1553, 3);  unsqueeze_1553 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1564: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_1565: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1564, 2);  unsqueeze_1564 = None
        unsqueeze_1566: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1565, 3);  unsqueeze_1565 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1576: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_1577: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, 2);  unsqueeze_1576 = None
        unsqueeze_1578: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1577, 3);  unsqueeze_1577 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1588: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_1589: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, 2);  unsqueeze_1588 = None
        unsqueeze_1590: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1589, 3);  unsqueeze_1589 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1600: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_1601: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, 2);  unsqueeze_1600 = None
        unsqueeze_1602: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1601, 3);  unsqueeze_1601 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1612: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_1613: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, 2);  unsqueeze_1612 = None
        unsqueeze_1614: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1613, 3);  unsqueeze_1613 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1624: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_1625: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, 2);  unsqueeze_1624 = None
        unsqueeze_1626: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1625, 3);  unsqueeze_1625 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1636: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_1637: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1636, 2);  unsqueeze_1636 = None
        unsqueeze_1638: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1637, 3);  unsqueeze_1637 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1648: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_1649: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, 2);  unsqueeze_1648 = None
        unsqueeze_1650: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1649, 3);  unsqueeze_1649 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1660: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_1661: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, 2);  unsqueeze_1660 = None
        unsqueeze_1662: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1661, 3);  unsqueeze_1661 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1672: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_1673: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, 2);  unsqueeze_1672 = None
        unsqueeze_1674: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1673, 3);  unsqueeze_1673 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1684: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_1685: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1684, 2);  unsqueeze_1684 = None
        unsqueeze_1686: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1685, 3);  unsqueeze_1685 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1696: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_1697: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1696, 2);  unsqueeze_1696 = None
        unsqueeze_1698: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1697, 3);  unsqueeze_1697 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1708: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_1709: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1708, 2);  unsqueeze_1708 = None
        unsqueeze_1710: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1709, 3);  unsqueeze_1709 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1720: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_1721: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1720, 2);  unsqueeze_1720 = None
        unsqueeze_1722: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1721, 3);  unsqueeze_1721 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1732: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_1733: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1732, 2);  unsqueeze_1732 = None
        unsqueeze_1734: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1733, 3);  unsqueeze_1733 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1744: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_1745: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1744, 2);  unsqueeze_1744 = None
        unsqueeze_1746: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1745, 3);  unsqueeze_1745 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1756: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_1757: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1756, 2);  unsqueeze_1756 = None
        unsqueeze_1758: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1757, 3);  unsqueeze_1757 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        unsqueeze_1768: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_1769: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1768, 2);  unsqueeze_1768 = None
        unsqueeze_1770: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1769, 3);  unsqueeze_1769 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1780: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_1781: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1780, 2);  unsqueeze_1780 = None
        unsqueeze_1782: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1781, 3);  unsqueeze_1781 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1792: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_1793: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1792, 2);  unsqueeze_1792 = None
        unsqueeze_1794: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1793, 3);  unsqueeze_1793 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1804: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_1805: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1804, 2);  unsqueeze_1804 = None
        unsqueeze_1806: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1805, 3);  unsqueeze_1805 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1816: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_1817: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1816, 2);  unsqueeze_1816 = None
        unsqueeze_1818: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1817, 3);  unsqueeze_1817 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1828: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_1829: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1828, 2);  unsqueeze_1828 = None
        unsqueeze_1830: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1829, 3);  unsqueeze_1829 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1840: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_1841: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1840, 2);  unsqueeze_1840 = None
        unsqueeze_1842: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1841, 3);  unsqueeze_1841 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1852: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_1853: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1852, 2);  unsqueeze_1852 = None
        unsqueeze_1854: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1853, 3);  unsqueeze_1853 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1864: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_1865: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1864, 2);  unsqueeze_1864 = None
        unsqueeze_1866: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1865, 3);  unsqueeze_1865 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1876: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_1877: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1876, 2);  unsqueeze_1876 = None
        unsqueeze_1878: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1877, 3);  unsqueeze_1877 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1888: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_1889: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1888, 2);  unsqueeze_1888 = None
        unsqueeze_1890: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1889, 3);  unsqueeze_1889 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1900: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_1901: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1900, 2);  unsqueeze_1900 = None
        unsqueeze_1902: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1901, 3);  unsqueeze_1901 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1912: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_1913: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1912, 2);  unsqueeze_1912 = None
        unsqueeze_1914: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1913, 3);  unsqueeze_1913 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_10);  primals_4 = add_10 = copy_ = None
        copy__1: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_12);  primals_5 = add_12 = copy__1 = None
        copy__2: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_6, add_13);  primals_6 = add_13 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_9, add_50);  primals_9 = add_50 = copy__3 = None
        copy__4: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_52);  primals_10 = add_52 = copy__4 = None
        copy__5: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_11, add_53);  primals_11 = add_53 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_15, add_80);  primals_15 = add_80 = copy__6 = None
        copy__7: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_16, add_82);  primals_16 = add_82 = copy__7 = None
        copy__8: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_17, add_83);  primals_17 = add_83 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_115);  primals_21 = add_115 = copy__9 = None
        copy__10: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_22, add_117);  primals_22 = add_117 = copy__10 = None
        copy__11: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_23, add_118);  primals_23 = add_118 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_145);  primals_27 = add_145 = copy__12 = None
        copy__13: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_28, add_147);  primals_28 = add_147 = copy__13 = None
        copy__14: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_29, add_148);  primals_29 = add_148 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_180);  primals_33 = add_180 = copy__15 = None
        copy__16: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_182);  primals_34 = add_182 = copy__16 = None
        copy__17: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_35, add_183);  primals_35 = add_183 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_39, add_210);  primals_39 = add_210 = copy__18 = None
        copy__19: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_212);  primals_40 = add_212 = copy__19 = None
        copy__20: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_213);  primals_41 = add_213 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_45, add_245);  primals_45 = add_245 = copy__21 = None
        copy__22: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_46, add_247);  primals_46 = add_247 = copy__22 = None
        copy__23: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_47, add_248);  primals_47 = add_248 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_51, add_275);  primals_51 = add_275 = copy__24 = None
        copy__25: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_52, add_277);  primals_52 = add_277 = copy__25 = None
        copy__26: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_53, add_278);  primals_53 = add_278 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_310);  primals_57 = add_310 = copy__27 = None
        copy__28: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_312);  primals_58 = add_312 = copy__28 = None
        copy__29: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_59, add_313);  primals_59 = add_313 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_340);  primals_63 = add_340 = copy__30 = None
        copy__31: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_342);  primals_64 = add_342 = copy__31 = None
        copy__32: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_343);  primals_65 = add_343 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_69, add_375);  primals_69 = add_375 = copy__33 = None
        copy__34: "f32[224][1]cuda:0" = torch.ops.aten.copy_.default(primals_70, add_377);  primals_70 = add_377 = copy__34 = None
        copy__35: "f32[224][1]cuda:0" = torch.ops.aten.copy_.default(primals_71, add_378);  primals_71 = add_378 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_405);  primals_75 = add_405 = copy__36 = None
        copy__37: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_407);  primals_76 = add_407 = copy__37 = None
        copy__38: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_77, add_408);  primals_77 = add_408 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_440);  primals_81 = add_440 = copy__39 = None
        copy__40: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_82, add_442);  primals_82 = add_442 = copy__40 = None
        copy__41: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_83, add_443);  primals_83 = add_443 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_480);  primals_87 = add_480 = copy__42 = None
        copy__43: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_88, add_482);  primals_88 = add_482 = copy__43 = None
        copy__44: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_89, add_483);  primals_89 = add_483 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_93, add_510);  primals_93 = add_510 = copy__45 = None
        copy__46: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_94, add_512);  primals_94 = add_512 = copy__46 = None
        copy__47: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_95, add_513);  primals_95 = add_513 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_99, add_545);  primals_99 = add_545 = copy__48 = None
        copy__49: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_100, add_547);  primals_100 = add_547 = copy__49 = None
        copy__50: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_101, add_548);  primals_101 = add_548 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_105, add_575);  primals_105 = add_575 = copy__51 = None
        copy__52: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_106, add_577);  primals_106 = add_577 = copy__52 = None
        copy__53: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_107, add_578);  primals_107 = add_578 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_111, add_610);  primals_111 = add_610 = copy__54 = None
        copy__55: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_112, add_612);  primals_112 = add_612 = copy__55 = None
        copy__56: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_113, add_613);  primals_113 = add_613 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_117, add_640);  primals_117 = add_640 = copy__57 = None
        copy__58: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_118, add_642);  primals_118 = add_642 = copy__58 = None
        copy__59: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_119, add_643);  primals_119 = add_643 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_123, add_675);  primals_123 = add_675 = copy__60 = None
        copy__61: "f32[224][1]cuda:0" = torch.ops.aten.copy_.default(primals_124, add_677);  primals_124 = add_677 = copy__61 = None
        copy__62: "f32[224][1]cuda:0" = torch.ops.aten.copy_.default(primals_125, add_678);  primals_125 = add_678 = copy__62 = None
        copy__63: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_129, add_705);  primals_129 = add_705 = copy__63 = None
        copy__64: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_130, add_707);  primals_130 = add_707 = copy__64 = None
        copy__65: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_131, add_708);  primals_131 = add_708 = copy__65 = None
        copy__66: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_135, add_740);  primals_135 = add_740 = copy__66 = None
        copy__67: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_136, add_742);  primals_136 = add_742 = copy__67 = None
        copy__68: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_137, add_743);  primals_137 = add_743 = copy__68 = None
        copy__69: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_141, add_770);  primals_141 = add_770 = copy__69 = None
        copy__70: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_142, add_772);  primals_142 = add_772 = copy__70 = None
        copy__71: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_143, add_773);  primals_143 = add_773 = copy__71 = None
        copy__72: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_147, add_805);  primals_147 = add_805 = copy__72 = None
        copy__73: "f32[288][1]cuda:0" = torch.ops.aten.copy_.default(primals_148, add_807);  primals_148 = add_807 = copy__73 = None
        copy__74: "f32[288][1]cuda:0" = torch.ops.aten.copy_.default(primals_149, add_808);  primals_149 = add_808 = copy__74 = None
        copy__75: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_153, add_835);  primals_153 = add_835 = copy__75 = None
        copy__76: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_154, add_837);  primals_154 = add_837 = copy__76 = None
        copy__77: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_155, add_838);  primals_155 = add_838 = copy__77 = None
        copy__78: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_159, add_870);  primals_159 = add_870 = copy__78 = None
        copy__79: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_160, add_872);  primals_160 = add_872 = copy__79 = None
        copy__80: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_161, add_873);  primals_161 = add_873 = copy__80 = None
        copy__81: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_165, add_900);  primals_165 = add_900 = copy__81 = None
        copy__82: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_166, add_902);  primals_166 = add_902 = copy__82 = None
        copy__83: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_167, add_903);  primals_167 = add_903 = copy__83 = None
        copy__84: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_171, add_935);  primals_171 = add_935 = copy__84 = None
        copy__85: "f32[352][1]cuda:0" = torch.ops.aten.copy_.default(primals_172, add_937);  primals_172 = add_937 = copy__85 = None
        copy__86: "f32[352][1]cuda:0" = torch.ops.aten.copy_.default(primals_173, add_938);  primals_173 = add_938 = copy__86 = None
        copy__87: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_177, add_965);  primals_177 = add_965 = copy__87 = None
        copy__88: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_178, add_967);  primals_178 = add_967 = copy__88 = None
        copy__89: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_179, add_968);  primals_179 = add_968 = copy__89 = None
        copy__90: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_183, add_1000);  primals_183 = add_1000 = copy__90 = None
        copy__91: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_184, add_1002);  primals_184 = add_1002 = copy__91 = None
        copy__92: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_185, add_1003);  primals_185 = add_1003 = copy__92 = None
        copy__93: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_189, add_1030);  primals_189 = add_1030 = copy__93 = None
        copy__94: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_190, add_1032);  primals_190 = add_1032 = copy__94 = None
        copy__95: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_191, add_1033);  primals_191 = add_1033 = copy__95 = None
        copy__96: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_195, add_1065);  primals_195 = add_1065 = copy__96 = None
        copy__97: "f32[416][1]cuda:0" = torch.ops.aten.copy_.default(primals_196, add_1067);  primals_196 = add_1067 = copy__97 = None
        copy__98: "f32[416][1]cuda:0" = torch.ops.aten.copy_.default(primals_197, add_1068);  primals_197 = add_1068 = copy__98 = None
        copy__99: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_201, add_1095);  primals_201 = add_1095 = copy__99 = None
        copy__100: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_202, add_1097);  primals_202 = add_1097 = copy__100 = None
        copy__101: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_203, add_1098);  primals_203 = add_1098 = copy__101 = None
        copy__102: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_207, add_1130);  primals_207 = add_1130 = copy__102 = None
        copy__103: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_208, add_1132);  primals_208 = add_1132 = copy__103 = None
        copy__104: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_209, add_1133);  primals_209 = add_1133 = copy__104 = None
        copy__105: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_213, add_1160);  primals_213 = add_1160 = copy__105 = None
        copy__106: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_214, add_1162);  primals_214 = add_1162 = copy__106 = None
        copy__107: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_215, add_1163);  primals_215 = add_1163 = copy__107 = None
        copy__108: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_219, add_1195);  primals_219 = add_1195 = copy__108 = None
        copy__109: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_220, add_1197);  primals_220 = add_1197 = copy__109 = None
        copy__110: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_221, add_1198);  primals_221 = add_1198 = copy__110 = None
        copy__111: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_225, add_1225);  primals_225 = add_1225 = copy__111 = None
        copy__112: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_226, add_1227);  primals_226 = add_1227 = copy__112 = None
        copy__113: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_227, add_1228);  primals_227 = add_1228 = copy__113 = None
        copy__114: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_231, add_1260);  primals_231 = add_1260 = copy__114 = None
        copy__115: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_232, add_1262);  primals_232 = add_1262 = copy__115 = None
        copy__116: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_233, add_1263);  primals_233 = add_1263 = copy__116 = None
        copy__117: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_237, add_1300);  primals_237 = add_1300 = copy__117 = None
        copy__118: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_238, add_1302);  primals_238 = add_1302 = copy__118 = None
        copy__119: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_239, add_1303);  primals_239 = add_1303 = copy__119 = None
        copy__120: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_243, add_1330);  primals_243 = add_1330 = copy__120 = None
        copy__121: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_244, add_1332);  primals_244 = add_1332 = copy__121 = None
        copy__122: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_245, add_1333);  primals_245 = add_1333 = copy__122 = None
        copy__123: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_249, add_1365);  primals_249 = add_1365 = copy__123 = None
        copy__124: "f32[288][1]cuda:0" = torch.ops.aten.copy_.default(primals_250, add_1367);  primals_250 = add_1367 = copy__124 = None
        copy__125: "f32[288][1]cuda:0" = torch.ops.aten.copy_.default(primals_251, add_1368);  primals_251 = add_1368 = copy__125 = None
        copy__126: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_255, add_1395);  primals_255 = add_1395 = copy__126 = None
        copy__127: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_256, add_1397);  primals_256 = add_1397 = copy__127 = None
        copy__128: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_257, add_1398);  primals_257 = add_1398 = copy__128 = None
        copy__129: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_261, add_1430);  primals_261 = add_1430 = copy__129 = None
        copy__130: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_262, add_1432);  primals_262 = add_1432 = copy__130 = None
        copy__131: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_263, add_1433);  primals_263 = add_1433 = copy__131 = None
        copy__132: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_267, add_1460);  primals_267 = add_1460 = copy__132 = None
        copy__133: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_268, add_1462);  primals_268 = add_1462 = copy__133 = None
        copy__134: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_269, add_1463);  primals_269 = add_1463 = copy__134 = None
        copy__135: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_273, add_1495);  primals_273 = add_1495 = copy__135 = None
        copy__136: "f32[352][1]cuda:0" = torch.ops.aten.copy_.default(primals_274, add_1497);  primals_274 = add_1497 = copy__136 = None
        copy__137: "f32[352][1]cuda:0" = torch.ops.aten.copy_.default(primals_275, add_1498);  primals_275 = add_1498 = copy__137 = None
        copy__138: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_279, add_1525);  primals_279 = add_1525 = copy__138 = None
        copy__139: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_280, add_1527);  primals_280 = add_1527 = copy__139 = None
        copy__140: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_281, add_1528);  primals_281 = add_1528 = copy__140 = None
        copy__141: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_285, add_1560);  primals_285 = add_1560 = copy__141 = None
        copy__142: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_286, add_1562);  primals_286 = add_1562 = copy__142 = None
        copy__143: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_287, add_1563);  primals_287 = add_1563 = copy__143 = None
        copy__144: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_291, add_1590);  primals_291 = add_1590 = copy__144 = None
        copy__145: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_292, add_1592);  primals_292 = add_1592 = copy__145 = None
        copy__146: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_293, add_1593);  primals_293 = add_1593 = copy__146 = None
        copy__147: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_297, add_1625);  primals_297 = add_1625 = copy__147 = None
        copy__148: "f32[416][1]cuda:0" = torch.ops.aten.copy_.default(primals_298, add_1627);  primals_298 = add_1627 = copy__148 = None
        copy__149: "f32[416][1]cuda:0" = torch.ops.aten.copy_.default(primals_299, add_1628);  primals_299 = add_1628 = copy__149 = None
        copy__150: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_303, add_1655);  primals_303 = add_1655 = copy__150 = None
        copy__151: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_304, add_1657);  primals_304 = add_1657 = copy__151 = None
        copy__152: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_305, add_1658);  primals_305 = add_1658 = copy__152 = None
        copy__153: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_309, add_1690);  primals_309 = add_1690 = copy__153 = None
        copy__154: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_310, add_1692);  primals_310 = add_1692 = copy__154 = None
        copy__155: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_311, add_1693);  primals_311 = add_1693 = copy__155 = None
        copy__156: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_315, add_1720);  primals_315 = add_1720 = copy__156 = None
        copy__157: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_316, add_1722);  primals_316 = add_1722 = copy__157 = None
        copy__158: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_317, add_1723);  primals_317 = add_1723 = copy__158 = None
        copy__159: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_321, add_1755);  primals_321 = add_1755 = copy__159 = None
        copy__160: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_322, add_1757);  primals_322 = add_1757 = copy__160 = None
        copy__161: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_323, add_1758);  primals_323 = add_1758 = copy__161 = None
        copy__162: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_327, add_1785);  primals_327 = add_1785 = copy__162 = None
        copy__163: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_328, add_1787);  primals_328 = add_1787 = copy__163 = None
        copy__164: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_329, add_1788);  primals_329 = add_1788 = copy__164 = None
        copy__165: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_333, add_1820);  primals_333 = add_1820 = copy__165 = None
        copy__166: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_334, add_1822);  primals_334 = add_1822 = copy__166 = None
        copy__167: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_335, add_1823);  primals_335 = add_1823 = copy__167 = None
        copy__168: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_339, add_1850);  primals_339 = add_1850 = copy__168 = None
        copy__169: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_340, add_1852);  primals_340 = add_1852 = copy__169 = None
        copy__170: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_341, add_1853);  primals_341 = add_1853 = copy__170 = None
        copy__171: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_345, add_1885);  primals_345 = add_1885 = copy__171 = None
        copy__172: "f32[544][1]cuda:0" = torch.ops.aten.copy_.default(primals_346, add_1887);  primals_346 = add_1887 = copy__172 = None
        copy__173: "f32[544][1]cuda:0" = torch.ops.aten.copy_.default(primals_347, add_1888);  primals_347 = add_1888 = copy__173 = None
        copy__174: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_351, add_1915);  primals_351 = add_1915 = copy__174 = None
        copy__175: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_352, add_1917);  primals_352 = add_1917 = copy__175 = None
        copy__176: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_353, add_1918);  primals_353 = add_1918 = copy__176 = None
        copy__177: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_357, add_1950);  primals_357 = add_1950 = copy__177 = None
        copy__178: "f32[576][1]cuda:0" = torch.ops.aten.copy_.default(primals_358, add_1952);  primals_358 = add_1952 = copy__178 = None
        copy__179: "f32[576][1]cuda:0" = torch.ops.aten.copy_.default(primals_359, add_1953);  primals_359 = add_1953 = copy__179 = None
        copy__180: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_363, add_1980);  primals_363 = add_1980 = copy__180 = None
        copy__181: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_364, add_1982);  primals_364 = add_1982 = copy__181 = None
        copy__182: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_365, add_1983);  primals_365 = add_1983 = copy__182 = None
        copy__183: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_369, add_2015);  primals_369 = add_2015 = copy__183 = None
        copy__184: "f32[608][1]cuda:0" = torch.ops.aten.copy_.default(primals_370, add_2017);  primals_370 = add_2017 = copy__184 = None
        copy__185: "f32[608][1]cuda:0" = torch.ops.aten.copy_.default(primals_371, add_2018);  primals_371 = add_2018 = copy__185 = None
        copy__186: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_375, add_2045);  primals_375 = add_2045 = copy__186 = None
        copy__187: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_376, add_2047);  primals_376 = add_2047 = copy__187 = None
        copy__188: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_377, add_2048);  primals_377 = add_2048 = copy__188 = None
        copy__189: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_381, add_2080);  primals_381 = add_2080 = copy__189 = None
        copy__190: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_382, add_2082);  primals_382 = add_2082 = copy__190 = None
        copy__191: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_383, add_2083);  primals_383 = add_2083 = copy__191 = None
        copy__192: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_387, add_2110);  primals_387 = add_2110 = copy__192 = None
        copy__193: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_388, add_2112);  primals_388 = add_2112 = copy__193 = None
        copy__194: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_389, add_2113);  primals_389 = add_2113 = copy__194 = None
        copy__195: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_393, add_2145);  primals_393 = add_2145 = copy__195 = None
        copy__196: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_394, add_2147);  primals_394 = add_2147 = copy__196 = None
        copy__197: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_395, add_2148);  primals_395 = add_2148 = copy__197 = None
        copy__198: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_399, add_2175);  primals_399 = add_2175 = copy__198 = None
        copy__199: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_400, add_2177);  primals_400 = add_2177 = copy__199 = None
        copy__200: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_401, add_2178);  primals_401 = add_2178 = copy__200 = None
        copy__201: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_405, add_2210);  primals_405 = add_2210 = copy__201 = None
        copy__202: "f32[704][1]cuda:0" = torch.ops.aten.copy_.default(primals_406, add_2212);  primals_406 = add_2212 = copy__202 = None
        copy__203: "f32[704][1]cuda:0" = torch.ops.aten.copy_.default(primals_407, add_2213);  primals_407 = add_2213 = copy__203 = None
        copy__204: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_411, add_2240);  primals_411 = add_2240 = copy__204 = None
        copy__205: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_412, add_2242);  primals_412 = add_2242 = copy__205 = None
        copy__206: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_413, add_2243);  primals_413 = add_2243 = copy__206 = None
        copy__207: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_417, add_2275);  primals_417 = add_2275 = copy__207 = None
        copy__208: "f32[736][1]cuda:0" = torch.ops.aten.copy_.default(primals_418, add_2277);  primals_418 = add_2277 = copy__208 = None
        copy__209: "f32[736][1]cuda:0" = torch.ops.aten.copy_.default(primals_419, add_2278);  primals_419 = add_2278 = copy__209 = None
        copy__210: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_423, add_2305);  primals_423 = add_2305 = copy__210 = None
        copy__211: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_424, add_2307);  primals_424 = add_2307 = copy__211 = None
        copy__212: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_425, add_2308);  primals_425 = add_2308 = copy__212 = None
        copy__213: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_429, add_2340);  primals_429 = add_2340 = copy__213 = None
        copy__214: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_430, add_2342);  primals_430 = add_2342 = copy__214 = None
        copy__215: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_431, add_2343);  primals_431 = add_2343 = copy__215 = None
        copy__216: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_435, add_2370);  primals_435 = add_2370 = copy__216 = None
        copy__217: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_436, add_2372);  primals_436 = add_2372 = copy__217 = None
        copy__218: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_437, add_2373);  primals_437 = add_2373 = copy__218 = None
        copy__219: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_441, add_2405);  primals_441 = add_2405 = copy__219 = None
        copy__220: "f32[800][1]cuda:0" = torch.ops.aten.copy_.default(primals_442, add_2407);  primals_442 = add_2407 = copy__220 = None
        copy__221: "f32[800][1]cuda:0" = torch.ops.aten.copy_.default(primals_443, add_2408);  primals_443 = add_2408 = copy__221 = None
        copy__222: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_447, add_2435);  primals_447 = add_2435 = copy__222 = None
        copy__223: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_448, add_2437);  primals_448 = add_2437 = copy__223 = None
        copy__224: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_449, add_2438);  primals_449 = add_2438 = copy__224 = None
        copy__225: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_453, add_2470);  primals_453 = add_2470 = copy__225 = None
        copy__226: "f32[832][1]cuda:0" = torch.ops.aten.copy_.default(primals_454, add_2472);  primals_454 = add_2472 = copy__226 = None
        copy__227: "f32[832][1]cuda:0" = torch.ops.aten.copy_.default(primals_455, add_2473);  primals_455 = add_2473 = copy__227 = None
        copy__228: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_459, add_2500);  primals_459 = add_2500 = copy__228 = None
        copy__229: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_460, add_2502);  primals_460 = add_2502 = copy__229 = None
        copy__230: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_461, add_2503);  primals_461 = add_2503 = copy__230 = None
        copy__231: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_465, add_2535);  primals_465 = add_2535 = copy__231 = None
        copy__232: "f32[864][1]cuda:0" = torch.ops.aten.copy_.default(primals_466, add_2537);  primals_466 = add_2537 = copy__232 = None
        copy__233: "f32[864][1]cuda:0" = torch.ops.aten.copy_.default(primals_467, add_2538);  primals_467 = add_2538 = copy__233 = None
        copy__234: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_471, add_2565);  primals_471 = add_2565 = copy__234 = None
        copy__235: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_472, add_2567);  primals_472 = add_2567 = copy__235 = None
        copy__236: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_473, add_2568);  primals_473 = add_2568 = copy__236 = None
        copy__237: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_477, add_2600);  primals_477 = add_2600 = copy__237 = None
        copy__238: "f32[896][1]cuda:0" = torch.ops.aten.copy_.default(primals_478, add_2602);  primals_478 = add_2602 = copy__238 = None
        copy__239: "f32[896][1]cuda:0" = torch.ops.aten.copy_.default(primals_479, add_2603);  primals_479 = add_2603 = copy__239 = None
        copy__240: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_483, add_2630);  primals_483 = add_2630 = copy__240 = None
        copy__241: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_484, add_2632);  primals_484 = add_2632 = copy__241 = None
        copy__242: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_485, add_2633);  primals_485 = add_2633 = copy__242 = None
        copy__243: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_489, add_2665);  primals_489 = add_2665 = copy__243 = None
        copy__244: "f32[928][1]cuda:0" = torch.ops.aten.copy_.default(primals_490, add_2667);  primals_490 = add_2667 = copy__244 = None
        copy__245: "f32[928][1]cuda:0" = torch.ops.aten.copy_.default(primals_491, add_2668);  primals_491 = add_2668 = copy__245 = None
        copy__246: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_495, add_2695);  primals_495 = add_2695 = copy__246 = None
        copy__247: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_496, add_2697);  primals_496 = add_2697 = copy__247 = None
        copy__248: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_497, add_2698);  primals_497 = add_2698 = copy__248 = None
        copy__249: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_501, add_2730);  primals_501 = add_2730 = copy__249 = None
        copy__250: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_502, add_2732);  primals_502 = add_2732 = copy__250 = None
        copy__251: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_503, add_2733);  primals_503 = add_2733 = copy__251 = None
        copy__252: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_507, add_2760);  primals_507 = add_2760 = copy__252 = None
        copy__253: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_508, add_2762);  primals_508 = add_2762 = copy__253 = None
        copy__254: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_509, add_2763);  primals_509 = add_2763 = copy__254 = None
        copy__255: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_513, add_2795);  primals_513 = add_2795 = copy__255 = None
        copy__256: "f32[992][1]cuda:0" = torch.ops.aten.copy_.default(primals_514, add_2797);  primals_514 = add_2797 = copy__256 = None
        copy__257: "f32[992][1]cuda:0" = torch.ops.aten.copy_.default(primals_515, add_2798);  primals_515 = add_2798 = copy__257 = None
        copy__258: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_519, add_2825);  primals_519 = add_2825 = copy__258 = None
        copy__259: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_520, add_2827);  primals_520 = add_2827 = copy__259 = None
        copy__260: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_521, add_2828);  primals_521 = add_2828 = copy__260 = None
        copy__261: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_525, add_2860);  primals_525 = add_2860 = copy__261 = None
        copy__262: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_526, add_2862);  primals_526 = add_2862 = copy__262 = None
        copy__263: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_527, add_2863);  primals_527 = add_2863 = copy__263 = None
        copy__264: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_531, add_2900);  primals_531 = add_2900 = copy__264 = None
        copy__265: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_532, add_2902);  primals_532 = add_2902 = copy__265 = None
        copy__266: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_533, add_2903);  primals_533 = add_2903 = copy__266 = None
        copy__267: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_537, add_2930);  primals_537 = add_2930 = copy__267 = None
        copy__268: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_538, add_2932);  primals_538 = add_2932 = copy__268 = None
        copy__269: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_539, add_2933);  primals_539 = add_2933 = copy__269 = None
        copy__270: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_543, add_2965);  primals_543 = add_2965 = copy__270 = None
        copy__271: "f32[544][1]cuda:0" = torch.ops.aten.copy_.default(primals_544, add_2967);  primals_544 = add_2967 = copy__271 = None
        copy__272: "f32[544][1]cuda:0" = torch.ops.aten.copy_.default(primals_545, add_2968);  primals_545 = add_2968 = copy__272 = None
        copy__273: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_549, add_2995);  primals_549 = add_2995 = copy__273 = None
        copy__274: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_550, add_2997);  primals_550 = add_2997 = copy__274 = None
        copy__275: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_551, add_2998);  primals_551 = add_2998 = copy__275 = None
        copy__276: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_555, add_3030);  primals_555 = add_3030 = copy__276 = None
        copy__277: "f32[576][1]cuda:0" = torch.ops.aten.copy_.default(primals_556, add_3032);  primals_556 = add_3032 = copy__277 = None
        copy__278: "f32[576][1]cuda:0" = torch.ops.aten.copy_.default(primals_557, add_3033);  primals_557 = add_3033 = copy__278 = None
        copy__279: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_561, add_3060);  primals_561 = add_3060 = copy__279 = None
        copy__280: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_562, add_3062);  primals_562 = add_3062 = copy__280 = None
        copy__281: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_563, add_3063);  primals_563 = add_3063 = copy__281 = None
        copy__282: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_567, add_3095);  primals_567 = add_3095 = copy__282 = None
        copy__283: "f32[608][1]cuda:0" = torch.ops.aten.copy_.default(primals_568, add_3097);  primals_568 = add_3097 = copy__283 = None
        copy__284: "f32[608][1]cuda:0" = torch.ops.aten.copy_.default(primals_569, add_3098);  primals_569 = add_3098 = copy__284 = None
        copy__285: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_573, add_3125);  primals_573 = add_3125 = copy__285 = None
        copy__286: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_574, add_3127);  primals_574 = add_3127 = copy__286 = None
        copy__287: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_575, add_3128);  primals_575 = add_3128 = copy__287 = None
        copy__288: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_579, add_3160);  primals_579 = add_3160 = copy__288 = None
        copy__289: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_580, add_3162);  primals_580 = add_3162 = copy__289 = None
        copy__290: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_581, add_3163);  primals_581 = add_3163 = copy__290 = None
        copy__291: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_585, add_3190);  primals_585 = add_3190 = copy__291 = None
        copy__292: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_586, add_3192);  primals_586 = add_3192 = copy__292 = None
        copy__293: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_587, add_3193);  primals_587 = add_3193 = copy__293 = None
        copy__294: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_591, add_3225);  primals_591 = add_3225 = copy__294 = None
        copy__295: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_592, add_3227);  primals_592 = add_3227 = copy__295 = None
        copy__296: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_593, add_3228);  primals_593 = add_3228 = copy__296 = None
        copy__297: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_597, add_3255);  primals_597 = add_3255 = copy__297 = None
        copy__298: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_598, add_3257);  primals_598 = add_3257 = copy__298 = None
        copy__299: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_599, add_3258);  primals_599 = add_3258 = copy__299 = None
        copy__300: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_603, add_3290);  primals_603 = add_3290 = copy__300 = None
        copy__301: "f32[704][1]cuda:0" = torch.ops.aten.copy_.default(primals_604, add_3292);  primals_604 = add_3292 = copy__301 = None
        copy__302: "f32[704][1]cuda:0" = torch.ops.aten.copy_.default(primals_605, add_3293);  primals_605 = add_3293 = copy__302 = None
        copy__303: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_609, add_3320);  primals_609 = add_3320 = copy__303 = None
        copy__304: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_610, add_3322);  primals_610 = add_3322 = copy__304 = None
        copy__305: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_611, add_3323);  primals_611 = add_3323 = copy__305 = None
        copy__306: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_615, add_3355);  primals_615 = add_3355 = copy__306 = None
        copy__307: "f32[736][1]cuda:0" = torch.ops.aten.copy_.default(primals_616, add_3357);  primals_616 = add_3357 = copy__307 = None
        copy__308: "f32[736][1]cuda:0" = torch.ops.aten.copy_.default(primals_617, add_3358);  primals_617 = add_3358 = copy__308 = None
        copy__309: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_621, add_3385);  primals_621 = add_3385 = copy__309 = None
        copy__310: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_622, add_3387);  primals_622 = add_3387 = copy__310 = None
        copy__311: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_623, add_3388);  primals_623 = add_3388 = copy__311 = None
        copy__312: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_627, add_3420);  primals_627 = add_3420 = copy__312 = None
        copy__313: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_628, add_3422);  primals_628 = add_3422 = copy__313 = None
        copy__314: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_629, add_3423);  primals_629 = add_3423 = copy__314 = None
        copy__315: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_633, add_3450);  primals_633 = add_3450 = copy__315 = None
        copy__316: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_634, add_3452);  primals_634 = add_3452 = copy__316 = None
        copy__317: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_635, add_3453);  primals_635 = add_3453 = copy__317 = None
        copy__318: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_639, add_3485);  primals_639 = add_3485 = copy__318 = None
        copy__319: "f32[800][1]cuda:0" = torch.ops.aten.copy_.default(primals_640, add_3487);  primals_640 = add_3487 = copy__319 = None
        copy__320: "f32[800][1]cuda:0" = torch.ops.aten.copy_.default(primals_641, add_3488);  primals_641 = add_3488 = copy__320 = None
        copy__321: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_645, add_3515);  primals_645 = add_3515 = copy__321 = None
        copy__322: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_646, add_3517);  primals_646 = add_3517 = copy__322 = None
        copy__323: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_647, add_3518);  primals_647 = add_3518 = copy__323 = None
        copy__324: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_651, add_3550);  primals_651 = add_3550 = copy__324 = None
        copy__325: "f32[832][1]cuda:0" = torch.ops.aten.copy_.default(primals_652, add_3552);  primals_652 = add_3552 = copy__325 = None
        copy__326: "f32[832][1]cuda:0" = torch.ops.aten.copy_.default(primals_653, add_3553);  primals_653 = add_3553 = copy__326 = None
        copy__327: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_657, add_3580);  primals_657 = add_3580 = copy__327 = None
        copy__328: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_658, add_3582);  primals_658 = add_3582 = copy__328 = None
        copy__329: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_659, add_3583);  primals_659 = add_3583 = copy__329 = None
        copy__330: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_663, add_3615);  primals_663 = add_3615 = copy__330 = None
        copy__331: "f32[864][1]cuda:0" = torch.ops.aten.copy_.default(primals_664, add_3617);  primals_664 = add_3617 = copy__331 = None
        copy__332: "f32[864][1]cuda:0" = torch.ops.aten.copy_.default(primals_665, add_3618);  primals_665 = add_3618 = copy__332 = None
        copy__333: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_669, add_3645);  primals_669 = add_3645 = copy__333 = None
        copy__334: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_670, add_3647);  primals_670 = add_3647 = copy__334 = None
        copy__335: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_671, add_3648);  primals_671 = add_3648 = copy__335 = None
        copy__336: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_675, add_3680);  primals_675 = add_3680 = copy__336 = None
        copy__337: "f32[896][1]cuda:0" = torch.ops.aten.copy_.default(primals_676, add_3682);  primals_676 = add_3682 = copy__337 = None
        copy__338: "f32[896][1]cuda:0" = torch.ops.aten.copy_.default(primals_677, add_3683);  primals_677 = add_3683 = copy__338 = None
        copy__339: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_681, add_3710);  primals_681 = add_3710 = copy__339 = None
        copy__340: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_682, add_3712);  primals_682 = add_3712 = copy__340 = None
        copy__341: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_683, add_3713);  primals_683 = add_3713 = copy__341 = None
        copy__342: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_687, add_3745);  primals_687 = add_3745 = copy__342 = None
        copy__343: "f32[928][1]cuda:0" = torch.ops.aten.copy_.default(primals_688, add_3747);  primals_688 = add_3747 = copy__343 = None
        copy__344: "f32[928][1]cuda:0" = torch.ops.aten.copy_.default(primals_689, add_3748);  primals_689 = add_3748 = copy__344 = None
        copy__345: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_693, add_3775);  primals_693 = add_3775 = copy__345 = None
        copy__346: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_694, add_3777);  primals_694 = add_3777 = copy__346 = None
        copy__347: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_695, add_3778);  primals_695 = add_3778 = copy__347 = None
        copy__348: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_699, add_3810);  primals_699 = add_3810 = copy__348 = None
        copy__349: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_700, add_3812);  primals_700 = add_3812 = copy__349 = None
        copy__350: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_701, add_3813);  primals_701 = add_3813 = copy__350 = None
        copy__351: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_705, add_3840);  primals_705 = add_3840 = copy__351 = None
        copy__352: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_706, add_3842);  primals_706 = add_3842 = copy__352 = None
        copy__353: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_707, add_3843);  primals_707 = add_3843 = copy__353 = None
        copy__354: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_711, add_3875);  primals_711 = add_3875 = copy__354 = None
        copy__355: "f32[992][1]cuda:0" = torch.ops.aten.copy_.default(primals_712, add_3877);  primals_712 = add_3877 = copy__355 = None
        copy__356: "f32[992][1]cuda:0" = torch.ops.aten.copy_.default(primals_713, add_3878);  primals_713 = add_3878 = copy__356 = None
        copy__357: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_717, add_3905);  primals_717 = add_3905 = copy__357 = None
        copy__358: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_718, add_3907);  primals_718 = add_3907 = copy__358 = None
        copy__359: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_719, add_3908);  primals_719 = add_3908 = copy__359 = None
        copy__360: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_723, add_3940);  primals_723 = add_3940 = copy__360 = None
        copy__361: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_724, add_3942);  primals_724 = add_3942 = copy__361 = None
        copy__362: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_725, add_3943);  primals_725 = add_3943 = copy__362 = None
        return (addmm, primals_7, primals_8, primals_12, primals_18, primals_24, primals_30, primals_36, primals_42, primals_48, primals_54, primals_60, primals_66, primals_72, primals_78, primals_84, primals_90, primals_96, primals_102, primals_108, primals_114, primals_120, primals_126, primals_132, primals_138, primals_144, primals_150, primals_156, primals_162, primals_168, primals_174, primals_180, primals_186, primals_192, primals_198, primals_204, primals_210, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, primals_252, primals_258, primals_264, primals_270, primals_276, primals_282, primals_288, primals_294, primals_300, primals_306, primals_312, primals_318, primals_324, primals_330, primals_336, primals_342, primals_348, primals_354, primals_360, primals_366, primals_372, primals_378, primals_384, primals_390, primals_396, primals_402, primals_408, primals_414, primals_420, primals_426, primals_432, primals_438, primals_444, primals_450, primals_456, primals_462, primals_468, primals_474, primals_480, primals_486, primals_492, primals_498, primals_504, primals_510, primals_516, primals_522, primals_528, primals_534, primals_540, primals_546, primals_552, primals_558, primals_564, primals_570, primals_576, primals_582, primals_588, primals_594, primals_600, primals_606, primals_612, primals_618, primals_624, primals_630, primals_636, primals_642, primals_648, primals_654, primals_660, primals_666, primals_672, primals_678, primals_684, primals_690, primals_696, primals_702, primals_708, primals_714, primals_720, primals_726, primals_727, convert_element_type, convert_element_type_1, convolution, getitem_1, rsqrt, getitem_2, getitem_3, squeeze_4, relu_1, convert_element_type_6, convolution_1, squeeze_7, relu_2, convert_element_type_9, cat, squeeze_10, relu_3, convert_element_type_12, convolution_3, squeeze_13, relu_4, convert_element_type_15, cat_1, squeeze_16, relu_5, convert_element_type_18, convolution_5, squeeze_19, relu_6, convert_element_type_21, cat_2, squeeze_22, relu_7, convert_element_type_24, convolution_7, squeeze_25, relu_8, convert_element_type_27, cat_3, squeeze_28, relu_9, convert_element_type_30, convolution_9, squeeze_31, relu_10, convert_element_type_33, cat_4, squeeze_34, relu_11, convert_element_type_36, convolution_11, squeeze_37, relu_12, convert_element_type_39, cat_5, squeeze_40, relu_13, convert_element_type_42, convolution_13, avg_pool2d, squeeze_43, relu_14, convert_element_type_45, convolution_14, squeeze_46, relu_15, convert_element_type_48, cat_6, squeeze_49, relu_16, convert_element_type_51, convolution_16, squeeze_52, relu_17, convert_element_type_54, cat_7, squeeze_55, relu_18, convert_element_type_57, convolution_18, squeeze_58, relu_19, convert_element_type_60, cat_8, squeeze_61, relu_20, convert_element_type_63, convolution_20, squeeze_64, relu_21, convert_element_type_66, cat_9, squeeze_67, relu_22, convert_element_type_69, convolution_22, squeeze_70, relu_23, convert_element_type_72, cat_10, squeeze_73, relu_24, convert_element_type_75, convolution_24, squeeze_76, relu_25, convert_element_type_78, cat_11, squeeze_79, relu_26, convert_element_type_81, convolution_26, squeeze_82, relu_27, convert_element_type_84, cat_12, squeeze_85, relu_28, convert_element_type_87, convolution_28, squeeze_88, relu_29, convert_element_type_90, cat_13, squeeze_91, relu_30, convert_element_type_93, convolution_30, squeeze_94, relu_31, convert_element_type_96, cat_14, squeeze_97, relu_32, convert_element_type_99, convolution_32, squeeze_100, relu_33, convert_element_type_102, cat_15, squeeze_103, relu_34, convert_element_type_105, convolution_34, squeeze_106, relu_35, convert_element_type_108, cat_16, squeeze_109, relu_36, convert_element_type_111, convolution_36, squeeze_112, relu_37, convert_element_type_114, cat_17, squeeze_115, relu_38, convert_element_type_117, convolution_38, avg_pool2d_1, squeeze_118, relu_39, convert_element_type_120, convolution_39, squeeze_121, relu_40, convert_element_type_123, cat_18, squeeze_124, relu_41, convert_element_type_126, convolution_41, squeeze_127, relu_42, convert_element_type_129, cat_19, squeeze_130, relu_43, convert_element_type_132, convolution_43, squeeze_133, relu_44, convert_element_type_135, cat_20, squeeze_136, relu_45, convert_element_type_138, convolution_45, squeeze_139, relu_46, convert_element_type_141, cat_21, squeeze_142, relu_47, convert_element_type_144, convolution_47, squeeze_145, relu_48, convert_element_type_147, cat_22, squeeze_148, relu_49, convert_element_type_150, convolution_49, squeeze_151, relu_50, convert_element_type_153, cat_23, squeeze_154, relu_51, convert_element_type_156, convolution_51, squeeze_157, relu_52, convert_element_type_159, cat_24, squeeze_160, relu_53, convert_element_type_162, convolution_53, squeeze_163, relu_54, convert_element_type_165, cat_25, squeeze_166, relu_55, convert_element_type_168, convolution_55, squeeze_169, relu_56, convert_element_type_171, cat_26, squeeze_172, relu_57, convert_element_type_174, convolution_57, squeeze_175, relu_58, convert_element_type_177, cat_27, squeeze_178, relu_59, convert_element_type_180, convolution_59, squeeze_181, relu_60, convert_element_type_183, cat_28, squeeze_184, relu_61, convert_element_type_186, convolution_61, squeeze_187, relu_62, convert_element_type_189, cat_29, squeeze_190, relu_63, convert_element_type_192, convolution_63, squeeze_193, relu_64, convert_element_type_195, cat_30, squeeze_196, relu_65, convert_element_type_198, convolution_65, squeeze_199, relu_66, convert_element_type_201, cat_31, squeeze_202, relu_67, convert_element_type_204, convolution_67, squeeze_205, relu_68, convert_element_type_207, cat_32, squeeze_208, relu_69, convert_element_type_210, convolution_69, squeeze_211, relu_70, convert_element_type_213, cat_33, squeeze_214, relu_71, convert_element_type_216, convolution_71, squeeze_217, relu_72, convert_element_type_219, cat_34, squeeze_220, relu_73, convert_element_type_222, convolution_73, squeeze_223, relu_74, convert_element_type_225, cat_35, squeeze_226, relu_75, convert_element_type_228, convolution_75, squeeze_229, relu_76, convert_element_type_231, cat_36, squeeze_232, relu_77, convert_element_type_234, convolution_77, squeeze_235, relu_78, convert_element_type_237, cat_37, squeeze_238, relu_79, convert_element_type_240, convolution_79, squeeze_241, relu_80, convert_element_type_243, cat_38, squeeze_244, relu_81, convert_element_type_246, convolution_81, squeeze_247, relu_82, convert_element_type_249, cat_39, squeeze_250, relu_83, convert_element_type_252, convolution_83, squeeze_253, relu_84, convert_element_type_255, cat_40, squeeze_256, relu_85, convert_element_type_258, convolution_85, squeeze_259, relu_86, convert_element_type_261, cat_41, squeeze_262, relu_87, convert_element_type_264, convolution_87, avg_pool2d_2, squeeze_265, relu_88, convert_element_type_267, convolution_88, squeeze_268, relu_89, convert_element_type_270, cat_42, squeeze_271, relu_90, convert_element_type_273, convolution_90, squeeze_274, relu_91, convert_element_type_276, cat_43, squeeze_277, relu_92, convert_element_type_279, convolution_92, squeeze_280, relu_93, convert_element_type_282, cat_44, squeeze_283, relu_94, convert_element_type_285, convolution_94, squeeze_286, relu_95, convert_element_type_288, cat_45, squeeze_289, relu_96, convert_element_type_291, convolution_96, squeeze_292, relu_97, convert_element_type_294, cat_46, squeeze_295, relu_98, convert_element_type_297, convolution_98, squeeze_298, relu_99, convert_element_type_300, cat_47, squeeze_301, relu_100, convert_element_type_303, convolution_100, squeeze_304, relu_101, convert_element_type_306, cat_48, squeeze_307, relu_102, convert_element_type_309, convolution_102, squeeze_310, relu_103, convert_element_type_312, cat_49, squeeze_313, relu_104, convert_element_type_315, convolution_104, squeeze_316, relu_105, convert_element_type_318, cat_50, squeeze_319, relu_106, convert_element_type_321, convolution_106, squeeze_322, relu_107, convert_element_type_324, cat_51, squeeze_325, relu_108, convert_element_type_327, convolution_108, squeeze_328, relu_109, convert_element_type_330, cat_52, squeeze_331, relu_110, convert_element_type_333, convolution_110, squeeze_334, relu_111, convert_element_type_336, cat_53, squeeze_337, relu_112, convert_element_type_339, convolution_112, squeeze_340, relu_113, convert_element_type_342, cat_54, squeeze_343, relu_114, convert_element_type_345, convolution_114, squeeze_346, relu_115, convert_element_type_348, cat_55, squeeze_349, relu_116, convert_element_type_351, convolution_116, squeeze_352, relu_117, convert_element_type_354, cat_56, squeeze_355, relu_118, convert_element_type_357, convolution_118, squeeze_358, relu_119, convert_element_type_360, cat_57, getitem_243, rsqrt_120, view, permute_1, unsqueeze_498, unsqueeze_510, unsqueeze_522, unsqueeze_534, unsqueeze_546, unsqueeze_558, unsqueeze_570, unsqueeze_582, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, unsqueeze_738, unsqueeze_750, unsqueeze_762, unsqueeze_774, unsqueeze_786, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, unsqueeze_858, unsqueeze_870, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, unsqueeze_978, unsqueeze_990, unsqueeze_1002, unsqueeze_1014, unsqueeze_1026, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, unsqueeze_1074, unsqueeze_1086, unsqueeze_1098, unsqueeze_1110, unsqueeze_1122, unsqueeze_1134, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, unsqueeze_1182, unsqueeze_1194, unsqueeze_1206, unsqueeze_1218, unsqueeze_1230, unsqueeze_1242, unsqueeze_1254, unsqueeze_1266, unsqueeze_1278, unsqueeze_1290, unsqueeze_1302, unsqueeze_1314, unsqueeze_1326, unsqueeze_1338, unsqueeze_1350, unsqueeze_1362, unsqueeze_1374, unsqueeze_1386, unsqueeze_1398, unsqueeze_1410, unsqueeze_1422, unsqueeze_1434, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1482, unsqueeze_1494, unsqueeze_1506, unsqueeze_1518, unsqueeze_1530, unsqueeze_1542, unsqueeze_1554, unsqueeze_1566, unsqueeze_1578, unsqueeze_1590, unsqueeze_1602, unsqueeze_1614, unsqueeze_1626, unsqueeze_1638, unsqueeze_1650, unsqueeze_1662, unsqueeze_1674, unsqueeze_1686, unsqueeze_1698, unsqueeze_1710, unsqueeze_1722, unsqueeze_1734, unsqueeze_1746, unsqueeze_1758, unsqueeze_1770, unsqueeze_1782, unsqueeze_1794, unsqueeze_1806, unsqueeze_1818, unsqueeze_1830, unsqueeze_1842, unsqueeze_1854, unsqueeze_1866, unsqueeze_1878, unsqueeze_1890, unsqueeze_1902, unsqueeze_1914, primals_2)
