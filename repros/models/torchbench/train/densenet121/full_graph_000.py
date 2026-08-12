class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 7, 7][147, 49, 7, 1]cuda:0", primals_2: "f32[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[64][1]cuda:0", primals_5: "f32[64][1]cuda:0", primals_6: "f32[64][1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_8: "i64[][]cuda:0", primals_9: "f32[64][1]cuda:0", primals_10: "f32[64][1]cuda:0", primals_11: "f32[64][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_13: "f32[128, 64, 1, 1][64, 1, 1, 1]cuda:0", primals_14: "i64[][]cuda:0", primals_15: "f32[128][1]cuda:0", primals_16: "f32[128][1]cuda:0", primals_17: "f32[128][1]cuda:0", primals_18: "f32[128][1]cuda:0", primals_19: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_20: "i64[][]cuda:0", primals_21: "f32[96][1]cuda:0", primals_22: "f32[96][1]cuda:0", primals_23: "f32[96][1]cuda:0", primals_24: "f32[96][1]cuda:0", primals_25: "f32[128, 96, 1, 1][96, 1, 1, 1]cuda:0", primals_26: "i64[][]cuda:0", primals_27: "f32[128][1]cuda:0", primals_28: "f32[128][1]cuda:0", primals_29: "f32[128][1]cuda:0", primals_30: "f32[128][1]cuda:0", primals_31: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_32: "i64[][]cuda:0", primals_33: "f32[128][1]cuda:0", primals_34: "f32[128][1]cuda:0", primals_35: "f32[128][1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_37: "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_38: "i64[][]cuda:0", primals_39: "f32[128][1]cuda:0", primals_40: "f32[128][1]cuda:0", primals_41: "f32[128][1]cuda:0", primals_42: "f32[128][1]cuda:0", primals_43: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_44: "i64[][]cuda:0", primals_45: "f32[160][1]cuda:0", primals_46: "f32[160][1]cuda:0", primals_47: "f32[160][1]cuda:0", primals_48: "f32[160][1]cuda:0", primals_49: "f32[128, 160, 1, 1][160, 1, 1, 1]cuda:0", primals_50: "i64[][]cuda:0", primals_51: "f32[128][1]cuda:0", primals_52: "f32[128][1]cuda:0", primals_53: "f32[128][1]cuda:0", primals_54: "f32[128][1]cuda:0", primals_55: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_56: "i64[][]cuda:0", primals_57: "f32[192][1]cuda:0", primals_58: "f32[192][1]cuda:0", primals_59: "f32[192][1]cuda:0", primals_60: "f32[192][1]cuda:0", primals_61: "f32[128, 192, 1, 1][192, 1, 1, 1]cuda:0", primals_62: "i64[][]cuda:0", primals_63: "f32[128][1]cuda:0", primals_64: "f32[128][1]cuda:0", primals_65: "f32[128][1]cuda:0", primals_66: "f32[128][1]cuda:0", primals_67: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_68: "i64[][]cuda:0", primals_69: "f32[224][1]cuda:0", primals_70: "f32[224][1]cuda:0", primals_71: "f32[224][1]cuda:0", primals_72: "f32[224][1]cuda:0", primals_73: "f32[128, 224, 1, 1][224, 1, 1, 1]cuda:0", primals_74: "i64[][]cuda:0", primals_75: "f32[128][1]cuda:0", primals_76: "f32[128][1]cuda:0", primals_77: "f32[128][1]cuda:0", primals_78: "f32[128][1]cuda:0", primals_79: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_80: "i64[][]cuda:0", primals_81: "f32[256][1]cuda:0", primals_82: "f32[256][1]cuda:0", primals_83: "f32[256][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_85: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_86: "i64[][]cuda:0", primals_87: "f32[128][1]cuda:0", primals_88: "f32[128][1]cuda:0", primals_89: "f32[128][1]cuda:0", primals_90: "f32[128][1]cuda:0", primals_91: "f32[128, 128, 1, 1][128, 1, 1, 1]cuda:0", primals_92: "i64[][]cuda:0", primals_93: "f32[128][1]cuda:0", primals_94: "f32[128][1]cuda:0", primals_95: "f32[128][1]cuda:0", primals_96: "f32[128][1]cuda:0", primals_97: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_98: "i64[][]cuda:0", primals_99: "f32[160][1]cuda:0", primals_100: "f32[160][1]cuda:0", primals_101: "f32[160][1]cuda:0", primals_102: "f32[160][1]cuda:0", primals_103: "f32[128, 160, 1, 1][160, 1, 1, 1]cuda:0", primals_104: "i64[][]cuda:0", primals_105: "f32[128][1]cuda:0", primals_106: "f32[128][1]cuda:0", primals_107: "f32[128][1]cuda:0", primals_108: "f32[128][1]cuda:0", primals_109: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_110: "i64[][]cuda:0", primals_111: "f32[192][1]cuda:0", primals_112: "f32[192][1]cuda:0", primals_113: "f32[192][1]cuda:0", primals_114: "f32[192][1]cuda:0", primals_115: "f32[128, 192, 1, 1][192, 1, 1, 1]cuda:0", primals_116: "i64[][]cuda:0", primals_117: "f32[128][1]cuda:0", primals_118: "f32[128][1]cuda:0", primals_119: "f32[128][1]cuda:0", primals_120: "f32[128][1]cuda:0", primals_121: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_122: "i64[][]cuda:0", primals_123: "f32[224][1]cuda:0", primals_124: "f32[224][1]cuda:0", primals_125: "f32[224][1]cuda:0", primals_126: "f32[224][1]cuda:0", primals_127: "f32[128, 224, 1, 1][224, 1, 1, 1]cuda:0", primals_128: "i64[][]cuda:0", primals_129: "f32[128][1]cuda:0", primals_130: "f32[128][1]cuda:0", primals_131: "f32[128][1]cuda:0", primals_132: "f32[128][1]cuda:0", primals_133: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_134: "i64[][]cuda:0", primals_135: "f32[256][1]cuda:0", primals_136: "f32[256][1]cuda:0", primals_137: "f32[256][1]cuda:0", primals_138: "f32[256][1]cuda:0", primals_139: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_140: "i64[][]cuda:0", primals_141: "f32[128][1]cuda:0", primals_142: "f32[128][1]cuda:0", primals_143: "f32[128][1]cuda:0", primals_144: "f32[128][1]cuda:0", primals_145: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_146: "i64[][]cuda:0", primals_147: "f32[288][1]cuda:0", primals_148: "f32[288][1]cuda:0", primals_149: "f32[288][1]cuda:0", primals_150: "f32[288][1]cuda:0", primals_151: "f32[128, 288, 1, 1][288, 1, 1, 1]cuda:0", primals_152: "i64[][]cuda:0", primals_153: "f32[128][1]cuda:0", primals_154: "f32[128][1]cuda:0", primals_155: "f32[128][1]cuda:0", primals_156: "f32[128][1]cuda:0", primals_157: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_158: "i64[][]cuda:0", primals_159: "f32[320][1]cuda:0", primals_160: "f32[320][1]cuda:0", primals_161: "f32[320][1]cuda:0", primals_162: "f32[320][1]cuda:0", primals_163: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0", primals_164: "i64[][]cuda:0", primals_165: "f32[128][1]cuda:0", primals_166: "f32[128][1]cuda:0", primals_167: "f32[128][1]cuda:0", primals_168: "f32[128][1]cuda:0", primals_169: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_170: "i64[][]cuda:0", primals_171: "f32[352][1]cuda:0", primals_172: "f32[352][1]cuda:0", primals_173: "f32[352][1]cuda:0", primals_174: "f32[352][1]cuda:0", primals_175: "f32[128, 352, 1, 1][352, 1, 1, 1]cuda:0", primals_176: "i64[][]cuda:0", primals_177: "f32[128][1]cuda:0", primals_178: "f32[128][1]cuda:0", primals_179: "f32[128][1]cuda:0", primals_180: "f32[128][1]cuda:0", primals_181: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_182: "i64[][]cuda:0", primals_183: "f32[384][1]cuda:0", primals_184: "f32[384][1]cuda:0", primals_185: "f32[384][1]cuda:0", primals_186: "f32[384][1]cuda:0", primals_187: "f32[128, 384, 1, 1][384, 1, 1, 1]cuda:0", primals_188: "i64[][]cuda:0", primals_189: "f32[128][1]cuda:0", primals_190: "f32[128][1]cuda:0", primals_191: "f32[128][1]cuda:0", primals_192: "f32[128][1]cuda:0", primals_193: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_194: "i64[][]cuda:0", primals_195: "f32[416][1]cuda:0", primals_196: "f32[416][1]cuda:0", primals_197: "f32[416][1]cuda:0", primals_198: "f32[416][1]cuda:0", primals_199: "f32[128, 416, 1, 1][416, 1, 1, 1]cuda:0", primals_200: "i64[][]cuda:0", primals_201: "f32[128][1]cuda:0", primals_202: "f32[128][1]cuda:0", primals_203: "f32[128][1]cuda:0", primals_204: "f32[128][1]cuda:0", primals_205: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_206: "i64[][]cuda:0", primals_207: "f32[448][1]cuda:0", primals_208: "f32[448][1]cuda:0", primals_209: "f32[448][1]cuda:0", primals_210: "f32[448][1]cuda:0", primals_211: "f32[128, 448, 1, 1][448, 1, 1, 1]cuda:0", primals_212: "i64[][]cuda:0", primals_213: "f32[128][1]cuda:0", primals_214: "f32[128][1]cuda:0", primals_215: "f32[128][1]cuda:0", primals_216: "f32[128][1]cuda:0", primals_217: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_218: "i64[][]cuda:0", primals_219: "f32[480][1]cuda:0", primals_220: "f32[480][1]cuda:0", primals_221: "f32[480][1]cuda:0", primals_222: "f32[480][1]cuda:0", primals_223: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0", primals_224: "i64[][]cuda:0", primals_225: "f32[128][1]cuda:0", primals_226: "f32[128][1]cuda:0", primals_227: "f32[128][1]cuda:0", primals_228: "f32[128][1]cuda:0", primals_229: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_230: "i64[][]cuda:0", primals_231: "f32[512][1]cuda:0", primals_232: "f32[512][1]cuda:0", primals_233: "f32[512][1]cuda:0", primals_234: "f32[512][1]cuda:0", primals_235: "f32[256, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_236: "i64[][]cuda:0", primals_237: "f32[256][1]cuda:0", primals_238: "f32[256][1]cuda:0", primals_239: "f32[256][1]cuda:0", primals_240: "f32[256][1]cuda:0", primals_241: "f32[128, 256, 1, 1][256, 1, 1, 1]cuda:0", primals_242: "i64[][]cuda:0", primals_243: "f32[128][1]cuda:0", primals_244: "f32[128][1]cuda:0", primals_245: "f32[128][1]cuda:0", primals_246: "f32[128][1]cuda:0", primals_247: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_248: "i64[][]cuda:0", primals_249: "f32[288][1]cuda:0", primals_250: "f32[288][1]cuda:0", primals_251: "f32[288][1]cuda:0", primals_252: "f32[288][1]cuda:0", primals_253: "f32[128, 288, 1, 1][288, 1, 1, 1]cuda:0", primals_254: "i64[][]cuda:0", primals_255: "f32[128][1]cuda:0", primals_256: "f32[128][1]cuda:0", primals_257: "f32[128][1]cuda:0", primals_258: "f32[128][1]cuda:0", primals_259: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_260: "i64[][]cuda:0", primals_261: "f32[320][1]cuda:0", primals_262: "f32[320][1]cuda:0", primals_263: "f32[320][1]cuda:0", primals_264: "f32[320][1]cuda:0", primals_265: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0", primals_266: "i64[][]cuda:0", primals_267: "f32[128][1]cuda:0", primals_268: "f32[128][1]cuda:0", primals_269: "f32[128][1]cuda:0", primals_270: "f32[128][1]cuda:0", primals_271: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_272: "i64[][]cuda:0", primals_273: "f32[352][1]cuda:0", primals_274: "f32[352][1]cuda:0", primals_275: "f32[352][1]cuda:0", primals_276: "f32[352][1]cuda:0", primals_277: "f32[128, 352, 1, 1][352, 1, 1, 1]cuda:0", primals_278: "i64[][]cuda:0", primals_279: "f32[128][1]cuda:0", primals_280: "f32[128][1]cuda:0", primals_281: "f32[128][1]cuda:0", primals_282: "f32[128][1]cuda:0", primals_283: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_284: "i64[][]cuda:0", primals_285: "f32[384][1]cuda:0", primals_286: "f32[384][1]cuda:0", primals_287: "f32[384][1]cuda:0", primals_288: "f32[384][1]cuda:0", primals_289: "f32[128, 384, 1, 1][384, 1, 1, 1]cuda:0", primals_290: "i64[][]cuda:0", primals_291: "f32[128][1]cuda:0", primals_292: "f32[128][1]cuda:0", primals_293: "f32[128][1]cuda:0", primals_294: "f32[128][1]cuda:0", primals_295: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_296: "i64[][]cuda:0", primals_297: "f32[416][1]cuda:0", primals_298: "f32[416][1]cuda:0", primals_299: "f32[416][1]cuda:0", primals_300: "f32[416][1]cuda:0", primals_301: "f32[128, 416, 1, 1][416, 1, 1, 1]cuda:0", primals_302: "i64[][]cuda:0", primals_303: "f32[128][1]cuda:0", primals_304: "f32[128][1]cuda:0", primals_305: "f32[128][1]cuda:0", primals_306: "f32[128][1]cuda:0", primals_307: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_308: "i64[][]cuda:0", primals_309: "f32[448][1]cuda:0", primals_310: "f32[448][1]cuda:0", primals_311: "f32[448][1]cuda:0", primals_312: "f32[448][1]cuda:0", primals_313: "f32[128, 448, 1, 1][448, 1, 1, 1]cuda:0", primals_314: "i64[][]cuda:0", primals_315: "f32[128][1]cuda:0", primals_316: "f32[128][1]cuda:0", primals_317: "f32[128][1]cuda:0", primals_318: "f32[128][1]cuda:0", primals_319: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_320: "i64[][]cuda:0", primals_321: "f32[480][1]cuda:0", primals_322: "f32[480][1]cuda:0", primals_323: "f32[480][1]cuda:0", primals_324: "f32[480][1]cuda:0", primals_325: "f32[128, 480, 1, 1][480, 1, 1, 1]cuda:0", primals_326: "i64[][]cuda:0", primals_327: "f32[128][1]cuda:0", primals_328: "f32[128][1]cuda:0", primals_329: "f32[128][1]cuda:0", primals_330: "f32[128][1]cuda:0", primals_331: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_332: "i64[][]cuda:0", primals_333: "f32[512][1]cuda:0", primals_334: "f32[512][1]cuda:0", primals_335: "f32[512][1]cuda:0", primals_336: "f32[512][1]cuda:0", primals_337: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_338: "i64[][]cuda:0", primals_339: "f32[128][1]cuda:0", primals_340: "f32[128][1]cuda:0", primals_341: "f32[128][1]cuda:0", primals_342: "f32[128][1]cuda:0", primals_343: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_344: "i64[][]cuda:0", primals_345: "f32[544][1]cuda:0", primals_346: "f32[544][1]cuda:0", primals_347: "f32[544][1]cuda:0", primals_348: "f32[544][1]cuda:0", primals_349: "f32[128, 544, 1, 1][544, 1, 1, 1]cuda:0", primals_350: "i64[][]cuda:0", primals_351: "f32[128][1]cuda:0", primals_352: "f32[128][1]cuda:0", primals_353: "f32[128][1]cuda:0", primals_354: "f32[128][1]cuda:0", primals_355: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_356: "i64[][]cuda:0", primals_357: "f32[576][1]cuda:0", primals_358: "f32[576][1]cuda:0", primals_359: "f32[576][1]cuda:0", primals_360: "f32[576][1]cuda:0", primals_361: "f32[128, 576, 1, 1][576, 1, 1, 1]cuda:0", primals_362: "i64[][]cuda:0", primals_363: "f32[128][1]cuda:0", primals_364: "f32[128][1]cuda:0", primals_365: "f32[128][1]cuda:0", primals_366: "f32[128][1]cuda:0", primals_367: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_368: "i64[][]cuda:0", primals_369: "f32[608][1]cuda:0", primals_370: "f32[608][1]cuda:0", primals_371: "f32[608][1]cuda:0", primals_372: "f32[608][1]cuda:0", primals_373: "f32[128, 608, 1, 1][608, 1, 1, 1]cuda:0", primals_374: "i64[][]cuda:0", primals_375: "f32[128][1]cuda:0", primals_376: "f32[128][1]cuda:0", primals_377: "f32[128][1]cuda:0", primals_378: "f32[128][1]cuda:0", primals_379: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_380: "i64[][]cuda:0", primals_381: "f32[640][1]cuda:0", primals_382: "f32[640][1]cuda:0", primals_383: "f32[640][1]cuda:0", primals_384: "f32[640][1]cuda:0", primals_385: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0", primals_386: "i64[][]cuda:0", primals_387: "f32[128][1]cuda:0", primals_388: "f32[128][1]cuda:0", primals_389: "f32[128][1]cuda:0", primals_390: "f32[128][1]cuda:0", primals_391: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_392: "i64[][]cuda:0", primals_393: "f32[672][1]cuda:0", primals_394: "f32[672][1]cuda:0", primals_395: "f32[672][1]cuda:0", primals_396: "f32[672][1]cuda:0", primals_397: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0", primals_398: "i64[][]cuda:0", primals_399: "f32[128][1]cuda:0", primals_400: "f32[128][1]cuda:0", primals_401: "f32[128][1]cuda:0", primals_402: "f32[128][1]cuda:0", primals_403: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_404: "i64[][]cuda:0", primals_405: "f32[704][1]cuda:0", primals_406: "f32[704][1]cuda:0", primals_407: "f32[704][1]cuda:0", primals_408: "f32[704][1]cuda:0", primals_409: "f32[128, 704, 1, 1][704, 1, 1, 1]cuda:0", primals_410: "i64[][]cuda:0", primals_411: "f32[128][1]cuda:0", primals_412: "f32[128][1]cuda:0", primals_413: "f32[128][1]cuda:0", primals_414: "f32[128][1]cuda:0", primals_415: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_416: "i64[][]cuda:0", primals_417: "f32[736][1]cuda:0", primals_418: "f32[736][1]cuda:0", primals_419: "f32[736][1]cuda:0", primals_420: "f32[736][1]cuda:0", primals_421: "f32[128, 736, 1, 1][736, 1, 1, 1]cuda:0", primals_422: "i64[][]cuda:0", primals_423: "f32[128][1]cuda:0", primals_424: "f32[128][1]cuda:0", primals_425: "f32[128][1]cuda:0", primals_426: "f32[128][1]cuda:0", primals_427: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_428: "i64[][]cuda:0", primals_429: "f32[768][1]cuda:0", primals_430: "f32[768][1]cuda:0", primals_431: "f32[768][1]cuda:0", primals_432: "f32[768][1]cuda:0", primals_433: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0", primals_434: "i64[][]cuda:0", primals_435: "f32[128][1]cuda:0", primals_436: "f32[128][1]cuda:0", primals_437: "f32[128][1]cuda:0", primals_438: "f32[128][1]cuda:0", primals_439: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_440: "i64[][]cuda:0", primals_441: "f32[800][1]cuda:0", primals_442: "f32[800][1]cuda:0", primals_443: "f32[800][1]cuda:0", primals_444: "f32[800][1]cuda:0", primals_445: "f32[128, 800, 1, 1][800, 1, 1, 1]cuda:0", primals_446: "i64[][]cuda:0", primals_447: "f32[128][1]cuda:0", primals_448: "f32[128][1]cuda:0", primals_449: "f32[128][1]cuda:0", primals_450: "f32[128][1]cuda:0", primals_451: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_452: "i64[][]cuda:0", primals_453: "f32[832][1]cuda:0", primals_454: "f32[832][1]cuda:0", primals_455: "f32[832][1]cuda:0", primals_456: "f32[832][1]cuda:0", primals_457: "f32[128, 832, 1, 1][832, 1, 1, 1]cuda:0", primals_458: "i64[][]cuda:0", primals_459: "f32[128][1]cuda:0", primals_460: "f32[128][1]cuda:0", primals_461: "f32[128][1]cuda:0", primals_462: "f32[128][1]cuda:0", primals_463: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_464: "i64[][]cuda:0", primals_465: "f32[864][1]cuda:0", primals_466: "f32[864][1]cuda:0", primals_467: "f32[864][1]cuda:0", primals_468: "f32[864][1]cuda:0", primals_469: "f32[128, 864, 1, 1][864, 1, 1, 1]cuda:0", primals_470: "i64[][]cuda:0", primals_471: "f32[128][1]cuda:0", primals_472: "f32[128][1]cuda:0", primals_473: "f32[128][1]cuda:0", primals_474: "f32[128][1]cuda:0", primals_475: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_476: "i64[][]cuda:0", primals_477: "f32[896][1]cuda:0", primals_478: "f32[896][1]cuda:0", primals_479: "f32[896][1]cuda:0", primals_480: "f32[896][1]cuda:0", primals_481: "f32[128, 896, 1, 1][896, 1, 1, 1]cuda:0", primals_482: "i64[][]cuda:0", primals_483: "f32[128][1]cuda:0", primals_484: "f32[128][1]cuda:0", primals_485: "f32[128][1]cuda:0", primals_486: "f32[128][1]cuda:0", primals_487: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_488: "i64[][]cuda:0", primals_489: "f32[928][1]cuda:0", primals_490: "f32[928][1]cuda:0", primals_491: "f32[928][1]cuda:0", primals_492: "f32[928][1]cuda:0", primals_493: "f32[128, 928, 1, 1][928, 1, 1, 1]cuda:0", primals_494: "i64[][]cuda:0", primals_495: "f32[128][1]cuda:0", primals_496: "f32[128][1]cuda:0", primals_497: "f32[128][1]cuda:0", primals_498: "f32[128][1]cuda:0", primals_499: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_500: "i64[][]cuda:0", primals_501: "f32[960][1]cuda:0", primals_502: "f32[960][1]cuda:0", primals_503: "f32[960][1]cuda:0", primals_504: "f32[960][1]cuda:0", primals_505: "f32[128, 960, 1, 1][960, 1, 1, 1]cuda:0", primals_506: "i64[][]cuda:0", primals_507: "f32[128][1]cuda:0", primals_508: "f32[128][1]cuda:0", primals_509: "f32[128][1]cuda:0", primals_510: "f32[128][1]cuda:0", primals_511: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_512: "i64[][]cuda:0", primals_513: "f32[992][1]cuda:0", primals_514: "f32[992][1]cuda:0", primals_515: "f32[992][1]cuda:0", primals_516: "f32[992][1]cuda:0", primals_517: "f32[128, 992, 1, 1][992, 1, 1, 1]cuda:0", primals_518: "i64[][]cuda:0", primals_519: "f32[128][1]cuda:0", primals_520: "f32[128][1]cuda:0", primals_521: "f32[128][1]cuda:0", primals_522: "f32[128][1]cuda:0", primals_523: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_524: "i64[][]cuda:0", primals_525: "f32[1024][1]cuda:0", primals_526: "f32[1024][1]cuda:0", primals_527: "f32[1024][1]cuda:0", primals_528: "f32[1024][1]cuda:0", primals_529: "f32[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0", primals_530: "i64[][]cuda:0", primals_531: "f32[512][1]cuda:0", primals_532: "f32[512][1]cuda:0", primals_533: "f32[512][1]cuda:0", primals_534: "f32[512][1]cuda:0", primals_535: "f32[128, 512, 1, 1][512, 1, 1, 1]cuda:0", primals_536: "i64[][]cuda:0", primals_537: "f32[128][1]cuda:0", primals_538: "f32[128][1]cuda:0", primals_539: "f32[128][1]cuda:0", primals_540: "f32[128][1]cuda:0", primals_541: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_542: "i64[][]cuda:0", primals_543: "f32[544][1]cuda:0", primals_544: "f32[544][1]cuda:0", primals_545: "f32[544][1]cuda:0", primals_546: "f32[544][1]cuda:0", primals_547: "f32[128, 544, 1, 1][544, 1, 1, 1]cuda:0", primals_548: "i64[][]cuda:0", primals_549: "f32[128][1]cuda:0", primals_550: "f32[128][1]cuda:0", primals_551: "f32[128][1]cuda:0", primals_552: "f32[128][1]cuda:0", primals_553: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_554: "i64[][]cuda:0", primals_555: "f32[576][1]cuda:0", primals_556: "f32[576][1]cuda:0", primals_557: "f32[576][1]cuda:0", primals_558: "f32[576][1]cuda:0", primals_559: "f32[128, 576, 1, 1][576, 1, 1, 1]cuda:0", primals_560: "i64[][]cuda:0", primals_561: "f32[128][1]cuda:0", primals_562: "f32[128][1]cuda:0", primals_563: "f32[128][1]cuda:0", primals_564: "f32[128][1]cuda:0", primals_565: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_566: "i64[][]cuda:0", primals_567: "f32[608][1]cuda:0", primals_568: "f32[608][1]cuda:0", primals_569: "f32[608][1]cuda:0", primals_570: "f32[608][1]cuda:0", primals_571: "f32[128, 608, 1, 1][608, 1, 1, 1]cuda:0", primals_572: "i64[][]cuda:0", primals_573: "f32[128][1]cuda:0", primals_574: "f32[128][1]cuda:0", primals_575: "f32[128][1]cuda:0", primals_576: "f32[128][1]cuda:0", primals_577: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_578: "i64[][]cuda:0", primals_579: "f32[640][1]cuda:0", primals_580: "f32[640][1]cuda:0", primals_581: "f32[640][1]cuda:0", primals_582: "f32[640][1]cuda:0", primals_583: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0", primals_584: "i64[][]cuda:0", primals_585: "f32[128][1]cuda:0", primals_586: "f32[128][1]cuda:0", primals_587: "f32[128][1]cuda:0", primals_588: "f32[128][1]cuda:0", primals_589: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_590: "i64[][]cuda:0", primals_591: "f32[672][1]cuda:0", primals_592: "f32[672][1]cuda:0", primals_593: "f32[672][1]cuda:0", primals_594: "f32[672][1]cuda:0", primals_595: "f32[128, 672, 1, 1][672, 1, 1, 1]cuda:0", primals_596: "i64[][]cuda:0", primals_597: "f32[128][1]cuda:0", primals_598: "f32[128][1]cuda:0", primals_599: "f32[128][1]cuda:0", primals_600: "f32[128][1]cuda:0", primals_601: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_602: "i64[][]cuda:0", primals_603: "f32[704][1]cuda:0", primals_604: "f32[704][1]cuda:0", primals_605: "f32[704][1]cuda:0", primals_606: "f32[704][1]cuda:0", primals_607: "f32[128, 704, 1, 1][704, 1, 1, 1]cuda:0", primals_608: "i64[][]cuda:0", primals_609: "f32[128][1]cuda:0", primals_610: "f32[128][1]cuda:0", primals_611: "f32[128][1]cuda:0", primals_612: "f32[128][1]cuda:0", primals_613: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_614: "i64[][]cuda:0", primals_615: "f32[736][1]cuda:0", primals_616: "f32[736][1]cuda:0", primals_617: "f32[736][1]cuda:0", primals_618: "f32[736][1]cuda:0", primals_619: "f32[128, 736, 1, 1][736, 1, 1, 1]cuda:0", primals_620: "i64[][]cuda:0", primals_621: "f32[128][1]cuda:0", primals_622: "f32[128][1]cuda:0", primals_623: "f32[128][1]cuda:0", primals_624: "f32[128][1]cuda:0", primals_625: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_626: "i64[][]cuda:0", primals_627: "f32[768][1]cuda:0", primals_628: "f32[768][1]cuda:0", primals_629: "f32[768][1]cuda:0", primals_630: "f32[768][1]cuda:0", primals_631: "f32[128, 768, 1, 1][768, 1, 1, 1]cuda:0", primals_632: "i64[][]cuda:0", primals_633: "f32[128][1]cuda:0", primals_634: "f32[128][1]cuda:0", primals_635: "f32[128][1]cuda:0", primals_636: "f32[128][1]cuda:0", primals_637: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_638: "i64[][]cuda:0", primals_639: "f32[800][1]cuda:0", primals_640: "f32[800][1]cuda:0", primals_641: "f32[800][1]cuda:0", primals_642: "f32[800][1]cuda:0", primals_643: "f32[128, 800, 1, 1][800, 1, 1, 1]cuda:0", primals_644: "i64[][]cuda:0", primals_645: "f32[128][1]cuda:0", primals_646: "f32[128][1]cuda:0", primals_647: "f32[128][1]cuda:0", primals_648: "f32[128][1]cuda:0", primals_649: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_650: "i64[][]cuda:0", primals_651: "f32[832][1]cuda:0", primals_652: "f32[832][1]cuda:0", primals_653: "f32[832][1]cuda:0", primals_654: "f32[832][1]cuda:0", primals_655: "f32[128, 832, 1, 1][832, 1, 1, 1]cuda:0", primals_656: "i64[][]cuda:0", primals_657: "f32[128][1]cuda:0", primals_658: "f32[128][1]cuda:0", primals_659: "f32[128][1]cuda:0", primals_660: "f32[128][1]cuda:0", primals_661: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_662: "i64[][]cuda:0", primals_663: "f32[864][1]cuda:0", primals_664: "f32[864][1]cuda:0", primals_665: "f32[864][1]cuda:0", primals_666: "f32[864][1]cuda:0", primals_667: "f32[128, 864, 1, 1][864, 1, 1, 1]cuda:0", primals_668: "i64[][]cuda:0", primals_669: "f32[128][1]cuda:0", primals_670: "f32[128][1]cuda:0", primals_671: "f32[128][1]cuda:0", primals_672: "f32[128][1]cuda:0", primals_673: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_674: "i64[][]cuda:0", primals_675: "f32[896][1]cuda:0", primals_676: "f32[896][1]cuda:0", primals_677: "f32[896][1]cuda:0", primals_678: "f32[896][1]cuda:0", primals_679: "f32[128, 896, 1, 1][896, 1, 1, 1]cuda:0", primals_680: "i64[][]cuda:0", primals_681: "f32[128][1]cuda:0", primals_682: "f32[128][1]cuda:0", primals_683: "f32[128][1]cuda:0", primals_684: "f32[128][1]cuda:0", primals_685: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_686: "i64[][]cuda:0", primals_687: "f32[928][1]cuda:0", primals_688: "f32[928][1]cuda:0", primals_689: "f32[928][1]cuda:0", primals_690: "f32[928][1]cuda:0", primals_691: "f32[128, 928, 1, 1][928, 1, 1, 1]cuda:0", primals_692: "i64[][]cuda:0", primals_693: "f32[128][1]cuda:0", primals_694: "f32[128][1]cuda:0", primals_695: "f32[128][1]cuda:0", primals_696: "f32[128][1]cuda:0", primals_697: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_698: "i64[][]cuda:0", primals_699: "f32[960][1]cuda:0", primals_700: "f32[960][1]cuda:0", primals_701: "f32[960][1]cuda:0", primals_702: "f32[960][1]cuda:0", primals_703: "f32[128, 960, 1, 1][960, 1, 1, 1]cuda:0", primals_704: "i64[][]cuda:0", primals_705: "f32[128][1]cuda:0", primals_706: "f32[128][1]cuda:0", primals_707: "f32[128][1]cuda:0", primals_708: "f32[128][1]cuda:0", primals_709: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_710: "i64[][]cuda:0", primals_711: "f32[992][1]cuda:0", primals_712: "f32[992][1]cuda:0", primals_713: "f32[992][1]cuda:0", primals_714: "f32[992][1]cuda:0", primals_715: "f32[128, 992, 1, 1][992, 1, 1, 1]cuda:0", primals_716: "i64[][]cuda:0", primals_717: "f32[128][1]cuda:0", primals_718: "f32[128][1]cuda:0", primals_719: "f32[128][1]cuda:0", primals_720: "f32[128][1]cuda:0", primals_721: "f32[32, 128, 3, 3][1152, 9, 3, 1]cuda:0", primals_722: "i64[][]cuda:0", primals_723: "f32[1024][1]cuda:0", primals_724: "f32[1024][1]cuda:0", primals_725: "f32[1024][1]cuda:0", primals_726: "f32[1024][1]cuda:0", primals_727: "f32[1000, 1024][1024, 1]cuda:0", primals_728: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        convert_element_type: "bf16[64, 3, 7, 7][147, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[4, 3, 224, 224][150528, 50176, 224, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convolution: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1)
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)
        convert_element_type_2: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3])
        mul_1: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1);  squeeze = None
        mul_2: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000199302441455);  squeeze_2 = None
        mul_4: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1)
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        relu: "bf16[4, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem_2: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[0]
        getitem_3: "i8[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_5: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_8, 1)
        convert_element_type_4: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_2, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_4, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_4 = None
        getitem_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(getitem_2, getitem_5)
        mul_7: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        squeeze_4: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_9, 0.9)
        add_7: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_10: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000797257434426);  squeeze_5 = None
        mul_11: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_8: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_11, -1)
        unsqueeze_5: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1);  primals_12 = None
        unsqueeze_7: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_5: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None
        relu_1: "bf16[4, 64, 56, 56][200704, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_5);  convert_element_type_5 = None
        convert_element_type_6: "bf16[128, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convolution_1: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_6, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_10: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_14, 1)
        convert_element_type_7: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_7, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_7 = None
        getitem_6: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_7: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_2: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_7)
        mul_14: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_7: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_15: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1)
        mul_16: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_15, 0.9)
        add_12: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_17: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000797257434426);  squeeze_8 = None
        mul_18: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_13: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_17, -1)
        unsqueeze_9: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1);  primals_18 = None
        unsqueeze_11: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_8: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        relu_2: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_8);  convert_element_type_8 = None
        convert_element_type_9: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        convolution_2: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_2, convert_element_type_9, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat: "bf16[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_15: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_20, 1)
        convert_element_type_10: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_10, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_10 = None
        getitem_8: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_9: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_3: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat, getitem_9)
        mul_21: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_21, 0.9)
        add_17: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000797257434426);  squeeze_11 = None
        mul_25: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_18: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_23, -1)
        unsqueeze_13: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1);  primals_24 = None
        unsqueeze_15: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_11: "bf16[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None
        relu_3: "bf16[4, 96, 56, 56][301056, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_11);  convert_element_type_11 = None
        convert_element_type_12: "bf16[128, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_25, torch.bfloat16);  primals_25 = None
        convolution_3: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_12, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_20: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_26, 1)
        convert_element_type_13: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_13, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_13 = None
        getitem_10: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_11: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_4: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_11)
        mul_28: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_13: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_29: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_30: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_27, 0.9)
        add_22: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.0000797257434426);  squeeze_14 = None
        mul_32: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_23: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_29, -1)
        unsqueeze_17: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1);  primals_30 = None
        unsqueeze_19: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_14: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None
        relu_4: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_14);  convert_element_type_14 = None
        convert_element_type_15: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_31, torch.bfloat16);  primals_31 = None
        convolution_4: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_4, convert_element_type_15, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_1: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_25: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_32, 1)
        convert_element_type_16: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_1, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_16, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_16 = None
        getitem_12: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_13: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_26: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_5: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_5: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_1, getitem_13)
        mul_35: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        squeeze_16: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_36: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_37: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_27: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_38: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000797257434426);  squeeze_17 = None
        mul_39: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_28: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_35, -1)
        unsqueeze_21: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1);  primals_36 = None
        unsqueeze_23: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_17: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        relu_5: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_17);  convert_element_type_17 = None
        convert_element_type_18: "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_37, torch.bfloat16);  primals_37 = None
        convolution_5: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_5, convert_element_type_18, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_30: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_38, 1)
        convert_element_type_19: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_19, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_19 = None
        getitem_14: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_15: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_31: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_6: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_6: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
        mul_42: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_19: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_39, 0.9)
        add_32: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_45: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000797257434426);  squeeze_20 = None
        mul_46: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_33: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_41, -1)
        unsqueeze_25: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1);  primals_42 = None
        unsqueeze_27: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_34: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        convert_element_type_20: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        relu_6: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_20);  convert_element_type_20 = None
        convert_element_type_21: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        convolution_6: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_6, convert_element_type_21, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_2: "bf16[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_35: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_44, 1)
        convert_element_type_22: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_2, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_22, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_22 = None
        getitem_16: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_17: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_36: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_7: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_7: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_2, getitem_17)
        mul_49: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_22: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_50: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_51: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_45, 0.9)
        add_37: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_52: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000797257434426);  squeeze_23 = None
        mul_53: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_38: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_47, -1)
        unsqueeze_29: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1);  primals_48 = None
        unsqueeze_31: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_23: "bf16[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        relu_7: "bf16[4, 160, 56, 56][501760, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_23);  convert_element_type_23 = None
        convert_element_type_24: "bf16[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        convolution_7: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_7, convert_element_type_24, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_40: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_50, 1)
        convert_element_type_25: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_25, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_25 = None
        getitem_18: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_19: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_41: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_8: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_8: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_19)
        mul_56: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_25: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_51, 0.9)
        add_42: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_59: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000797257434426);  squeeze_26 = None
        mul_60: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_43: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_53, -1)
        unsqueeze_33: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1);  primals_54 = None
        unsqueeze_35: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_44: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        convert_element_type_26: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None
        relu_8: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_26);  convert_element_type_26 = None
        convert_element_type_27: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_55, torch.bfloat16);  primals_55 = None
        convolution_8: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_27, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_3: "bf16[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6, convolution_8], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_45: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_56, 1)
        convert_element_type_28: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_3, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_28, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_28 = None
        getitem_20: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_21: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_46: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_9: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_9: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_3, getitem_21)
        mul_63: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_28: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_57, 0.9)
        add_47: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_66: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000797257434426);  squeeze_29 = None
        mul_67: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_48: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_59, -1)
        unsqueeze_37: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1);  primals_60 = None
        unsqueeze_39: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_49: "f32[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        convert_element_type_29: "bf16[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None
        relu_9: "bf16[4, 192, 56, 56][602112, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_29);  convert_element_type_29 = None
        convert_element_type_30: "bf16[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_61, torch.bfloat16);  primals_61 = None
        convolution_9: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_9, convert_element_type_30, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_50: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_62, 1)
        convert_element_type_31: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_31, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_31 = None
        getitem_22: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_23: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_51: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_10: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_10: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_23)
        mul_70: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_31: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_71: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_72: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_63, 0.9)
        add_52: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_73: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000797257434426);  squeeze_32 = None
        mul_74: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_53: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_65, -1)
        unsqueeze_41: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1);  primals_66 = None
        unsqueeze_43: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_54: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_32: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None
        relu_10: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_32);  convert_element_type_32 = None
        convert_element_type_33: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_67, torch.bfloat16);  primals_67 = None
        convolution_10: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_10, convert_element_type_33, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_4: "bf16[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6, convolution_8, convolution_10], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_55: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_68, 1)
        convert_element_type_34: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_4, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_34, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_34 = None
        getitem_24: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_25: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_56: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_11: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_11: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_4, getitem_25)
        mul_77: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_34: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_78: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_79: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_69, 0.9)
        add_57: "f32[224][1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_80: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000797257434426);  squeeze_35 = None
        mul_81: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_58: "f32[224][1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[224, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_71, -1)
        unsqueeze_45: "f32[224, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[224, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1);  primals_72 = None
        unsqueeze_47: "f32[224, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_59: "f32[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_35: "bf16[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None
        relu_11: "bf16[4, 224, 56, 56][702464, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_35);  convert_element_type_35 = None
        convert_element_type_36: "bf16[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convolution_11: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_11, convert_element_type_36, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_60: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_74, 1)
        convert_element_type_37: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_37, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_37 = None
        getitem_26: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_27: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_61: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_12: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_12: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_27)
        mul_84: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        squeeze_37: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_85: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_86: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_75, 0.9)
        add_62: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_87: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000797257434426);  squeeze_38 = None
        mul_88: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_63: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_77, -1)
        unsqueeze_49: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1);  primals_78 = None
        unsqueeze_51: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_64: "f32[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        convert_element_type_38: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None
        relu_12: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_38);  convert_element_type_38 = None
        convert_element_type_39: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        convolution_12: "bf16[4, 32, 56, 56][100352, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_12, convert_element_type_39, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        cat_5: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.cat.default([getitem_2, convolution_2, convolution_4, convolution_6, convolution_8, convolution_10, convolution_12], 1);  convolution_2 = convolution_4 = convolution_6 = convolution_8 = convolution_10 = convolution_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        add_65: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_80, 1)
        convert_element_type_40: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_5, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_40, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_40 = None
        getitem_28: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_29: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_66: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_13: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_5, getitem_29)
        mul_91: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_40: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_81, 0.9)
        add_67: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_94: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000797257434426);  squeeze_41 = None
        mul_95: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_68: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_83, -1)
        unsqueeze_53: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1);  primals_84 = None
        unsqueeze_55: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_69: "f32[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convert_element_type_41: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None
        relu_13: "bf16[4, 256, 56, 56][802816, 3136, 56, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_41);  convert_element_type_41 = None
        convert_element_type_42: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_85, torch.bfloat16);  primals_85 = None
        convolution_13: "bf16[4, 128, 56, 56][401408, 3136, 56, 1]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_42, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        avg_pool2d: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.avg_pool2d.default(convolution_13, [2, 2], [2, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_70: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_86, 1)
        convert_element_type_43: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_43, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_43 = None
        getitem_30: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_31: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_71: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_14: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_14: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(avg_pool2d, getitem_31)
        mul_98: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_43: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_99: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_100: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_87, 0.9)
        add_72: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_101: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0003189792663476);  squeeze_44 = None
        mul_102: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_73: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_89, -1)
        unsqueeze_57: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1);  primals_90 = None
        unsqueeze_59: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_74: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_44: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None
        relu_14: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_44);  convert_element_type_44 = None
        convert_element_type_45: "bf16[128, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_91, torch.bfloat16);  primals_91 = None
        convolution_14: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_14, convert_element_type_45, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_75: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_92, 1)
        convert_element_type_46: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_46, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_46 = None
        getitem_32: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_33: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_76: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_15: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_15: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_33)
        mul_105: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_46: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_93, 0.9)
        add_77: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_108: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0003189792663476);  squeeze_47 = None
        mul_109: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_78: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_95, -1)
        unsqueeze_61: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_96, -1);  primals_96 = None
        unsqueeze_63: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_79: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        convert_element_type_47: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None
        relu_15: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_47);  convert_element_type_47 = None
        convert_element_type_48: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_97, torch.bfloat16);  primals_97 = None
        convolution_15: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_15, convert_element_type_48, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_6: "bf16[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_80: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_98, 1)
        convert_element_type_49: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_6, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_49, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_49 = None
        getitem_34: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_35: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_81: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_16: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_16: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_6, getitem_35)
        mul_112: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_49: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_99, 0.9)
        add_82: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_115: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0003189792663476);  squeeze_50 = None
        mul_116: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_83: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_101, -1)
        unsqueeze_65: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1);  primals_102 = None
        unsqueeze_67: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_84: "f32[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_50: "bf16[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None
        relu_16: "bf16[4, 160, 28, 28][125440, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_50);  convert_element_type_50 = None
        convert_element_type_51: "bf16[128, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        convolution_16: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_16, convert_element_type_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_85: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_104, 1)
        convert_element_type_52: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_52, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_52 = None
        getitem_36: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_37: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_86: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_17: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_17: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_37)
        mul_119: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_52: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_120: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_121: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_105, 0.9)
        add_87: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_122: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0003189792663476);  squeeze_53 = None
        mul_123: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_88: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_107, -1)
        unsqueeze_69: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_108, -1);  primals_108 = None
        unsqueeze_71: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_89: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        convert_element_type_53: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None
        relu_17: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_53);  convert_element_type_53 = None
        convert_element_type_54: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        convolution_17: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_17, convert_element_type_54, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_7: "bf16[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_90: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_110, 1)
        convert_element_type_55: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_7, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_55, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_55 = None
        getitem_38: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_39: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_91: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_18: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_18: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_7, getitem_39)
        mul_126: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_55: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_127: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_128: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_111, 0.9)
        add_92: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_129: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0003189792663476);  squeeze_56 = None
        mul_130: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_93: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_113, -1)
        unsqueeze_73: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1);  primals_114 = None
        unsqueeze_75: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_94: "f32[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        convert_element_type_56: "bf16[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None
        relu_18: "bf16[4, 192, 28, 28][150528, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_56);  convert_element_type_56 = None
        convert_element_type_57: "bf16[128, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_115, torch.bfloat16);  primals_115 = None
        convolution_18: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_18, convert_element_type_57, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_95: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_116, 1)
        convert_element_type_58: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_58, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_58 = None
        getitem_40: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_41: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_96: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_19: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_19: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_41)
        mul_133: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_58: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_134: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1)
        mul_135: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
        add_97: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_136: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0003189792663476);  squeeze_59 = None
        mul_137: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_98: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_119, -1)
        unsqueeze_77: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1);  primals_120 = None
        unsqueeze_79: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_99: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        convert_element_type_59: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        relu_19: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_59);  convert_element_type_59 = None
        convert_element_type_60: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_121, torch.bfloat16);  primals_121 = None
        convolution_19: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_19, convert_element_type_60, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_8: "bf16[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_100: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_122, 1)
        convert_element_type_61: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_8, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_61, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_61 = None
        getitem_42: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_43: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_101: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_20: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_20: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_8, getitem_43)
        mul_140: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_61: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_123, 0.9)
        add_102: "f32[224][1]cuda:0" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62: "f32[224][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_143: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0003189792663476);  squeeze_62 = None
        mul_144: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145: "f32[224][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_103: "f32[224][1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80: "f32[224, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_125, -1)
        unsqueeze_81: "f32[224, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_146: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        unsqueeze_82: "f32[224, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_126, -1);  primals_126 = None
        unsqueeze_83: "f32[224, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_104: "f32[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
        convert_element_type_62: "bf16[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None
        relu_20: "bf16[4, 224, 28, 28][175616, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_62);  convert_element_type_62 = None
        convert_element_type_63: "bf16[128, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_127, torch.bfloat16);  primals_127 = None
        convolution_20: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_20, convert_element_type_63, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_105: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_128, 1)
        convert_element_type_64: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_64, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_64 = None
        getitem_44: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_45: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_106: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_21: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_21: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_45)
        mul_147: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_64: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_148: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_149: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_129, 0.9)
        add_107: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_150: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0003189792663476);  squeeze_65 = None
        mul_151: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_108: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_131, -1)
        unsqueeze_85: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_153: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        unsqueeze_86: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1);  primals_132 = None
        unsqueeze_87: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_109: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        convert_element_type_65: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None
        relu_21: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_65);  convert_element_type_65 = None
        convert_element_type_66: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        convolution_21: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_21, convert_element_type_66, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_9: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_110: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_134, 1)
        convert_element_type_67: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_9, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_67, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_67 = None
        getitem_46: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_47: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_111: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_22: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_22: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_9, getitem_47)
        mul_154: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_67: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_155: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_156: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_135, 0.9)
        add_112: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_157: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0003189792663476);  squeeze_68 = None
        mul_158: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_113: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_137, -1)
        unsqueeze_89: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1);  primals_138 = None
        unsqueeze_91: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_114: "f32[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        convert_element_type_68: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None
        relu_22: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_68);  convert_element_type_68 = None
        convert_element_type_69: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        convolution_22: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_22, convert_element_type_69, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_115: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_140, 1)
        convert_element_type_70: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_70, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_70 = None
        getitem_48: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_23[0]
        getitem_49: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_116: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_23: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_23: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, getitem_49)
        mul_161: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_70: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_163: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_141, 0.9)
        add_117: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_164: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0003189792663476);  squeeze_71 = None
        mul_165: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_118: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_143, -1)
        unsqueeze_93: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_167: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        unsqueeze_94: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, -1);  primals_144 = None
        unsqueeze_95: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_119: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
        convert_element_type_71: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None
        relu_23: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_71);  convert_element_type_71 = None
        convert_element_type_72: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_145, torch.bfloat16);  primals_145 = None
        convolution_23: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_23, convert_element_type_72, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_10: "bf16[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_120: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_146, 1)
        convert_element_type_73: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_10, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_73, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_73 = None
        getitem_50: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = var_mean_24[0]
        getitem_51: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_121: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_24: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        sub_24: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_10, getitem_51)
        mul_168: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_73: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_169: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_170: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_147, 0.9)
        add_122: "f32[288][1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_171: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0003189792663476);  squeeze_74 = None
        mul_172: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_123: "f32[288][1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96: "f32[288, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_149, -1)
        unsqueeze_97: "f32[288, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_174: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        unsqueeze_98: "f32[288, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_150, -1);  primals_150 = None
        unsqueeze_99: "f32[288, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_124: "f32[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        convert_element_type_74: "bf16[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None
        relu_24: "bf16[4, 288, 28, 28][225792, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_74);  convert_element_type_74 = None
        convert_element_type_75: "bf16[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_151, torch.bfloat16);  primals_151 = None
        convolution_24: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_24, convert_element_type_75, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_125: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_152, 1)
        convert_element_type_76: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_76, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_76 = None
        getitem_52: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_25[0]
        getitem_53: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_126: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_25: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_25: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_53)
        mul_175: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_76: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_176: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1)
        mul_177: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
        add_127: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_178: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0003189792663476);  squeeze_77 = None
        mul_179: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_128: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_155, -1)
        unsqueeze_101: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_156, -1);  primals_156 = None
        unsqueeze_103: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_129: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        convert_element_type_77: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None
        relu_25: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_77);  convert_element_type_77 = None
        convert_element_type_78: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_157, torch.bfloat16);  primals_157 = None
        convolution_25: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_25, convert_element_type_78, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_11: "bf16[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_130: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_158, 1)
        convert_element_type_79: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_11, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_79, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_79 = None
        getitem_54: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_26[0]
        getitem_55: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_131: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_26: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        sub_26: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_11, getitem_55)
        mul_182: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_79: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_183: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_184: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_159, 0.9)
        add_132: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_185: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0003189792663476);  squeeze_80 = None
        mul_186: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_133: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_161, -1)
        unsqueeze_105: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_188: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        unsqueeze_106: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_162, -1);  primals_162 = None
        unsqueeze_107: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_134: "f32[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
        convert_element_type_80: "bf16[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None
        relu_26: "bf16[4, 320, 28, 28][250880, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_80);  convert_element_type_80 = None
        convert_element_type_81: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convolution_26: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_26, convert_element_type_81, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_135: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_164, 1)
        convert_element_type_82: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_82, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_82 = None
        getitem_56: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_27[0]
        getitem_57: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_136: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_27: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_27: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_57)
        mul_189: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_82: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_191: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_165, 0.9)
        add_137: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_192: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0003189792663476);  squeeze_83 = None
        mul_193: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_138: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_167, -1)
        unsqueeze_109: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_195: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        unsqueeze_110: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_168, -1);  primals_168 = None
        unsqueeze_111: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_139: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
        convert_element_type_83: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None
        relu_27: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_83);  convert_element_type_83 = None
        convert_element_type_84: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        convolution_27: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_27, convert_element_type_84, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_12: "bf16[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_140: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_170, 1)
        convert_element_type_85: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_12, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_85, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_85 = None
        getitem_58: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = var_mean_28[0]
        getitem_59: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_141: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_28: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_28: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_12, getitem_59)
        mul_196: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_85: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_197: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_198: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
        add_142: "f32[352][1]cuda:0" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_199: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0003189792663476);  squeeze_86 = None
        mul_200: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
        add_143: "f32[352][1]cuda:0" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112: "f32[352, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_173, -1)
        unsqueeze_113: "f32[352, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[352, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_174, -1);  primals_174 = None
        unsqueeze_115: "f32[352, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_144: "f32[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        convert_element_type_86: "bf16[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None
        relu_28: "bf16[4, 352, 28, 28][275968, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_86);  convert_element_type_86 = None
        convert_element_type_87: "bf16[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_175, torch.bfloat16);  primals_175 = None
        convolution_28: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_28, convert_element_type_87, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_145: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_176, 1)
        convert_element_type_88: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_88, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_88 = None
        getitem_60: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_29[0]
        getitem_61: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_146: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_29: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_29: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_61)
        mul_203: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_88: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_204: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1)
        mul_205: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
        add_147: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_206: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0003189792663476);  squeeze_89 = None
        mul_207: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
        add_148: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_179, -1)
        unsqueeze_117: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_209: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        unsqueeze_118: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_180, -1);  primals_180 = None
        unsqueeze_119: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_149: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
        convert_element_type_89: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None
        relu_29: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_89);  convert_element_type_89 = None
        convert_element_type_90: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_181, torch.bfloat16);  primals_181 = None
        convolution_29: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_29, convert_element_type_90, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_13: "bf16[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_150: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_182, 1)
        convert_element_type_91: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_13, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_91, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_91 = None
        getitem_62: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_30[0]
        getitem_63: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_151: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_30: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_30: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_13, getitem_63)
        mul_210: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_91: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_211: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_212: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
        add_152: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_213: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0003189792663476);  squeeze_92 = None
        mul_214: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
        add_153: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_185, -1)
        unsqueeze_121: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_216: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        unsqueeze_122: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_186, -1);  primals_186 = None
        unsqueeze_123: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_154: "f32[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        convert_element_type_92: "bf16[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None
        relu_30: "bf16[4, 384, 28, 28][301056, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_92);  convert_element_type_92 = None
        convert_element_type_93: "bf16[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_187, torch.bfloat16);  primals_187 = None
        convolution_30: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_30, convert_element_type_93, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_155: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_188, 1)
        convert_element_type_94: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_94, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_94 = None
        getitem_64: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_31[0]
        getitem_65: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_156: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_31: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_31: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_65)
        mul_217: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_94: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_218: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_219: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_189, 0.9)
        add_157: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_220: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0003189792663476);  squeeze_95 = None
        mul_221: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
        add_158: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_191, -1)
        unsqueeze_125: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_192, -1);  primals_192 = None
        unsqueeze_127: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_159: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        convert_element_type_95: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None
        relu_31: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_95);  convert_element_type_95 = None
        convert_element_type_96: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16);  primals_193 = None
        convolution_31: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_31, convert_element_type_96, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_14: "bf16[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_160: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_194, 1)
        convert_element_type_97: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_14, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_97, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_97 = None
        getitem_66: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = var_mean_32[0]
        getitem_67: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_161: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_32: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_32: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_14, getitem_67)
        mul_224: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_97: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_226: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_195, 0.9)
        add_162: "f32[416][1]cuda:0" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_227: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0003189792663476);  squeeze_98 = None
        mul_228: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_163: "f32[416][1]cuda:0" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128: "f32[416, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_197, -1)
        unsqueeze_129: "f32[416, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_230: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        unsqueeze_130: "f32[416, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_198, -1);  primals_198 = None
        unsqueeze_131: "f32[416, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_164: "f32[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
        convert_element_type_98: "bf16[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None
        relu_32: "bf16[4, 416, 28, 28][326144, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_98);  convert_element_type_98 = None
        convert_element_type_99: "bf16[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16);  primals_199 = None
        convolution_32: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_32, convert_element_type_99, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_165: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_200, 1)
        convert_element_type_100: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_100, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_100 = None
        getitem_68: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_33[0]
        getitem_69: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_166: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_33: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_33: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_69)
        mul_231: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_100: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_232: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_233: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_201, 0.9)
        add_167: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_234: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0003189792663476);  squeeze_101 = None
        mul_235: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_168: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_203, -1)
        unsqueeze_133: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_237: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        unsqueeze_134: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1);  primals_204 = None
        unsqueeze_135: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_169: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        convert_element_type_101: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16);  add_169 = None
        relu_33: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_101);  convert_element_type_101 = None
        convert_element_type_102: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_205, torch.bfloat16);  primals_205 = None
        convolution_33: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_33, convert_element_type_102, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_15: "bf16[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_170: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_206, 1)
        convert_element_type_103: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_15, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_103, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_103 = None
        getitem_70: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_34[0]
        getitem_71: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_171: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_34: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_34: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_15, getitem_71)
        mul_238: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_103: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_239: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_240: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_207, 0.9)
        add_172: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_241: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0003189792663476);  squeeze_104 = None
        mul_242: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_173: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_209, -1)
        unsqueeze_137: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_210, -1);  primals_210 = None
        unsqueeze_139: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_174: "f32[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        convert_element_type_104: "bf16[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.bfloat16);  add_174 = None
        relu_34: "bf16[4, 448, 28, 28][351232, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_104);  convert_element_type_104 = None
        convert_element_type_105: "bf16[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_211, torch.bfloat16);  primals_211 = None
        convolution_34: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_34, convert_element_type_105, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_175: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_212, 1)
        convert_element_type_106: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_106, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_106 = None
        getitem_72: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_35[0]
        getitem_73: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_176: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_35: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_35: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_73)
        mul_245: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_106: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_247: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_213, 0.9)
        add_177: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_248: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0003189792663476);  squeeze_107 = None
        mul_249: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
        add_178: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_215, -1)
        unsqueeze_141: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_251: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        unsqueeze_142: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_216, -1);  primals_216 = None
        unsqueeze_143: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_179: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
        convert_element_type_107: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None
        relu_35: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_107);  convert_element_type_107 = None
        convert_element_type_108: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_217, torch.bfloat16);  primals_217 = None
        convolution_35: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_35, convert_element_type_108, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_16: "bf16[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33, convolution_35], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_180: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_218, 1)
        convert_element_type_109: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_16, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_109, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_109 = None
        getitem_74: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_36[0]
        getitem_75: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_181: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_36: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_36: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_16, getitem_75)
        mul_252: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_109: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_254: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_219, 0.9)
        add_182: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_255: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0003189792663476);  squeeze_110 = None
        mul_256: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_183: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_221, -1)
        unsqueeze_145: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_258: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        unsqueeze_146: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_222, -1);  primals_222 = None
        unsqueeze_147: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_184: "f32[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
        convert_element_type_110: "bf16[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None
        relu_36: "bf16[4, 480, 28, 28][376320, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_110);  convert_element_type_110 = None
        convert_element_type_111: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_223, torch.bfloat16);  primals_223 = None
        convolution_36: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_36, convert_element_type_111, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_185: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_224, 1)
        convert_element_type_112: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_112, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_112 = None
        getitem_76: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_37[0]
        getitem_77: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_186: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_37: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_37: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_77)
        mul_259: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_112: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_260: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_261: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_225, 0.9)
        add_187: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_262: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0003189792663476);  squeeze_113 = None
        mul_263: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_188: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_227, -1)
        unsqueeze_149: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1);  primals_228 = None
        unsqueeze_151: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_189: "f32[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        convert_element_type_113: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None
        relu_37: "bf16[4, 128, 28, 28][100352, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_113);  convert_element_type_113 = None
        convert_element_type_114: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_229, torch.bfloat16);  primals_229 = None
        convolution_37: "bf16[4, 32, 28, 28][25088, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_37, convert_element_type_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        cat_17: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d, convolution_15, convolution_17, convolution_19, convolution_21, convolution_23, convolution_25, convolution_27, convolution_29, convolution_31, convolution_33, convolution_35, convolution_37], 1);  convolution_15 = convolution_17 = convolution_19 = convolution_21 = convolution_23 = convolution_25 = convolution_27 = convolution_29 = convolution_31 = convolution_33 = convolution_35 = convolution_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        add_190: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_230, 1)
        convert_element_type_115: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_17, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_115, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_115 = None
        getitem_78: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_38[0]
        getitem_79: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_191: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_38: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_38: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_17, getitem_79)
        mul_266: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_115: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_267: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1)
        mul_268: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_231, 0.9)
        add_192: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_269: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0003189792663476);  squeeze_116 = None
        mul_270: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_193: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_233, -1)
        unsqueeze_153: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_272: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        unsqueeze_154: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_234, -1);  primals_234 = None
        unsqueeze_155: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_194: "f32[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
        convert_element_type_116: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_194, torch.bfloat16);  add_194 = None
        relu_38: "bf16[4, 512, 28, 28][401408, 784, 28, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_116);  convert_element_type_116 = None
        convert_element_type_117: "bf16[256, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_235, torch.bfloat16);  primals_235 = None
        convolution_38: "bf16[4, 256, 28, 28][200704, 784, 28, 1]cuda:0" = torch.ops.aten.convolution.default(relu_38, convert_element_type_117, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        avg_pool2d_1: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.avg_pool2d.default(convolution_38, [2, 2], [2, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_195: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_236, 1)
        convert_element_type_118: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_1, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_118, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_118 = None
        getitem_80: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_39[0]
        getitem_81: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_196: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_39: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_39: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(avg_pool2d_1, getitem_81)
        mul_273: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_118: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_274: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_275: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_237, 0.9)
        add_197: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        squeeze_119: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_276: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0012771392081736);  squeeze_119 = None
        mul_277: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
        mul_278: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_198: "f32[256][1]cuda:0" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_156: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_239, -1)
        unsqueeze_157: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_279: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        unsqueeze_158: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1);  primals_240 = None
        unsqueeze_159: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_199: "f32[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        convert_element_type_119: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16);  add_199 = None
        relu_39: "bf16[4, 256, 14, 14][50176, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_119);  convert_element_type_119 = None
        convert_element_type_120: "bf16[128, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_241, torch.bfloat16);  primals_241 = None
        convolution_39: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_39, convert_element_type_120, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_200: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_242, 1)
        convert_element_type_121: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_121, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_121 = None
        getitem_82: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_40[0]
        getitem_83: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_201: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_40: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        sub_40: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, getitem_83)
        mul_280: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_121: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_281: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
        mul_282: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
        add_202: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        squeeze_122: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_283: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0012771392081736);  squeeze_122 = None
        mul_284: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
        mul_285: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_203: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_160: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_245, -1)
        unsqueeze_161: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_246, -1);  primals_246 = None
        unsqueeze_163: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_204: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        convert_element_type_122: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None
        relu_40: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_122);  convert_element_type_122 = None
        convert_element_type_123: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_247, torch.bfloat16);  primals_247 = None
        convolution_40: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_40, convert_element_type_123, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_18: "bf16[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_205: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_248, 1)
        convert_element_type_124: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_18, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_124, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_124 = None
        getitem_84: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = var_mean_41[0]
        getitem_85: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_206: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_41: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        sub_41: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_18, getitem_85)
        mul_287: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_124: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_288: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_289: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_249, 0.9)
        add_207: "f32[288][1]cuda:0" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_125: "f32[288][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_290: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0012771392081736);  squeeze_125 = None
        mul_291: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292: "f32[288][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_208: "f32[288][1]cuda:0" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_164: "f32[288, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_251, -1)
        unsqueeze_165: "f32[288, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_293: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
        unsqueeze_166: "f32[288, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_252, -1);  primals_252 = None
        unsqueeze_167: "f32[288, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_209: "f32[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
        convert_element_type_125: "bf16[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16);  add_209 = None
        relu_41: "bf16[4, 288, 14, 14][56448, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_125);  convert_element_type_125 = None
        convert_element_type_126: "bf16[128, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_253, torch.bfloat16);  primals_253 = None
        convolution_41: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_41, convert_element_type_126, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_210: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_254, 1)
        convert_element_type_127: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_127, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_127 = None
        getitem_86: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_42[0]
        getitem_87: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_211: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_42: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        sub_42: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, getitem_87)
        mul_294: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_127: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_295: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_296: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
        add_212: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_128: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_297: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0012771392081736);  squeeze_128 = None
        mul_298: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
        add_213: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_168: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_257, -1)
        unsqueeze_169: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_300: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        unsqueeze_170: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_258, -1);  primals_258 = None
        unsqueeze_171: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_214: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        convert_element_type_128: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None
        relu_42: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_128);  convert_element_type_128 = None
        convert_element_type_129: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16);  primals_259 = None
        convolution_42: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_42, convert_element_type_129, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_19: "bf16[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_215: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_260, 1)
        convert_element_type_130: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_19, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_130, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_130 = None
        getitem_88: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_43[0]
        getitem_89: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_216: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_43: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_43: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_19, getitem_89)
        mul_301: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_130: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_302: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_303: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_261, 0.9)
        add_217: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_131: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_304: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0012771392081736);  squeeze_131 = None
        mul_305: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_218: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_172: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_263, -1)
        unsqueeze_173: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_307: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        unsqueeze_174: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1);  primals_264 = None
        unsqueeze_175: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_219: "f32[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        convert_element_type_131: "bf16[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None
        relu_43: "bf16[4, 320, 14, 14][62720, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_131);  convert_element_type_131 = None
        convert_element_type_132: "bf16[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_265, torch.bfloat16);  primals_265 = None
        convolution_43: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_43, convert_element_type_132, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_220: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_266, 1)
        convert_element_type_133: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_133, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_133 = None
        getitem_90: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_44[0]
        getitem_91: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_221: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_44: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        sub_44: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, getitem_91)
        mul_308: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_133: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_309: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_310: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_267, 0.9)
        add_222: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_134: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_311: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0012771392081736);  squeeze_134 = None
        mul_312: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_223: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_176: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_269, -1)
        unsqueeze_177: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_314: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
        unsqueeze_178: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_270, -1);  primals_270 = None
        unsqueeze_179: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_224: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
        convert_element_type_134: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_224, torch.bfloat16);  add_224 = None
        relu_44: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_134);  convert_element_type_134 = None
        convert_element_type_135: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_271, torch.bfloat16);  primals_271 = None
        convolution_44: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_44, convert_element_type_135, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_20: "bf16[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_225: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_272, 1)
        convert_element_type_136: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_20, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_136, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_136 = None
        getitem_92: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = var_mean_45[0]
        getitem_93: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_226: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_45: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_45: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_20, getitem_93)
        mul_315: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_136: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_316: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
        mul_317: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_273, 0.9)
        add_227: "f32[352][1]cuda:0" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_137: "f32[352][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_318: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0012771392081736);  squeeze_137 = None
        mul_319: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320: "f32[352][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
        add_228: "f32[352][1]cuda:0" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_180: "f32[352, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_275, -1)
        unsqueeze_181: "f32[352, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_321: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        unsqueeze_182: "f32[352, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_276, -1);  primals_276 = None
        unsqueeze_183: "f32[352, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_229: "f32[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        convert_element_type_137: "bf16[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_229, torch.bfloat16);  add_229 = None
        relu_45: "bf16[4, 352, 14, 14][68992, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_137);  convert_element_type_137 = None
        convert_element_type_138: "bf16[128, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_277, torch.bfloat16);  primals_277 = None
        convolution_45: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_45, convert_element_type_138, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_230: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_278, 1)
        convert_element_type_139: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_139, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_139 = None
        getitem_94: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_46[0]
        getitem_95: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_231: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_46: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        sub_46: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_95)
        mul_322: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_139: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_323: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_324: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_279, 0.9)
        add_232: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        squeeze_140: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_325: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0012771392081736);  squeeze_140 = None
        mul_326: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
        mul_327: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
        add_233: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_184: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_281, -1)
        unsqueeze_185: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_328: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        unsqueeze_186: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_282, -1);  primals_282 = None
        unsqueeze_187: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_234: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
        convert_element_type_140: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16);  add_234 = None
        relu_46: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_140);  convert_element_type_140 = None
        convert_element_type_141: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_283, torch.bfloat16);  primals_283 = None
        convolution_46: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_46, convert_element_type_141, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_21: "bf16[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_235: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_284, 1)
        convert_element_type_142: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_21, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_142, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_142 = None
        getitem_96: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_47[0]
        getitem_97: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_236: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
        rsqrt_47: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_47: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_21, getitem_97)
        mul_329: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_142: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_330: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_331: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_285, 0.9)
        add_237: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_143: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_332: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0012771392081736);  squeeze_143 = None
        mul_333: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
        add_238: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_188: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_287, -1)
        unsqueeze_189: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_335: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
        unsqueeze_190: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_288, -1);  primals_288 = None
        unsqueeze_191: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_239: "f32[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
        convert_element_type_143: "bf16[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_239, torch.bfloat16);  add_239 = None
        relu_47: "bf16[4, 384, 14, 14][75264, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_143);  convert_element_type_143 = None
        convert_element_type_144: "bf16[128, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_289, torch.bfloat16);  primals_289 = None
        convolution_47: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_47, convert_element_type_144, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_240: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_290, 1)
        convert_element_type_145: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_145, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_145 = None
        getitem_98: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_48[0]
        getitem_99: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_241: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_48: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
        sub_48: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, getitem_99)
        mul_336: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_145: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_337: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
        mul_338: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_291, 0.9)
        add_242: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_146: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_339: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0012771392081736);  squeeze_146 = None
        mul_340: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
        add_243: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_192: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_293, -1)
        unsqueeze_193: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_294, -1);  primals_294 = None
        unsqueeze_195: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_244: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        convert_element_type_146: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16);  add_244 = None
        relu_48: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_146);  convert_element_type_146 = None
        convert_element_type_147: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_295, torch.bfloat16);  primals_295 = None
        convolution_48: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_48, convert_element_type_147, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_22: "bf16[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_245: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_296, 1)
        convert_element_type_148: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_22, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_148, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_148 = None
        getitem_100: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = var_mean_49[0]
        getitem_101: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_246: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_49: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        sub_49: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_22, getitem_101)
        mul_343: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_148: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_344: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
        mul_345: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_297, 0.9)
        add_247: "f32[416][1]cuda:0" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_149: "f32[416][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_346: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0012771392081736);  squeeze_149 = None
        mul_347: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348: "f32[416][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
        add_248: "f32[416][1]cuda:0" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_196: "f32[416, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_299, -1)
        unsqueeze_197: "f32[416, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_349: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        unsqueeze_198: "f32[416, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_300, -1);  primals_300 = None
        unsqueeze_199: "f32[416, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_249: "f32[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        convert_element_type_149: "bf16[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.bfloat16);  add_249 = None
        relu_49: "bf16[4, 416, 14, 14][81536, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_149);  convert_element_type_149 = None
        convert_element_type_150: "bf16[128, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_301, torch.bfloat16);  primals_301 = None
        convolution_49: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_49, convert_element_type_150, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_250: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_302, 1)
        convert_element_type_151: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_151, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_151 = None
        getitem_102: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_50[0]
        getitem_103: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        add_251: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_50: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_50: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, getitem_103)
        mul_350: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_151: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_351: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1)
        mul_352: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_303, 0.9)
        add_252: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        squeeze_152: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_353: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0012771392081736);  squeeze_152 = None
        mul_354: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
        mul_355: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
        add_253: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        unsqueeze_200: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_305, -1)
        unsqueeze_201: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_356: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        unsqueeze_202: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_306, -1);  primals_306 = None
        unsqueeze_203: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_254: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
        convert_element_type_152: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16);  add_254 = None
        relu_50: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_152);  convert_element_type_152 = None
        convert_element_type_153: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_307, torch.bfloat16);  primals_307 = None
        convolution_50: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_50, convert_element_type_153, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_23: "bf16[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_255: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_308, 1)
        convert_element_type_154: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_23, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_154, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_154 = None
        getitem_104: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_51[0]
        getitem_105: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        add_256: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_51: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
        sub_51: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_23, getitem_105)
        mul_357: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_154: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_358: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
        mul_359: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
        add_257: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        squeeze_155: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_360: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0012771392081736);  squeeze_155 = None
        mul_361: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
        mul_362: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
        add_258: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        unsqueeze_204: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_311, -1)
        unsqueeze_205: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_312, -1);  primals_312 = None
        unsqueeze_207: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_259: "f32[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        convert_element_type_155: "bf16[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None
        relu_51: "bf16[4, 448, 14, 14][87808, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_155);  convert_element_type_155 = None
        convert_element_type_156: "bf16[128, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_313, torch.bfloat16);  primals_313 = None
        convolution_51: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_51, convert_element_type_156, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_260: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_314, 1)
        convert_element_type_157: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_157, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_157 = None
        getitem_106: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_52[0]
        getitem_107: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        add_261: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_52: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_52: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_107)
        mul_364: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        squeeze_157: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_365: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_366: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
        add_262: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        squeeze_158: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_367: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0012771392081736);  squeeze_158 = None
        mul_368: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
        mul_369: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
        add_263: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_208: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_317, -1)
        unsqueeze_209: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        mul_370: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
        unsqueeze_210: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_318, -1);  primals_318 = None
        unsqueeze_211: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        add_264: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
        convert_element_type_158: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16);  add_264 = None
        relu_52: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_158);  convert_element_type_158 = None
        convert_element_type_159: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_319, torch.bfloat16);  primals_319 = None
        convolution_52: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_52, convert_element_type_159, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_24: "bf16[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_265: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_320, 1)
        convert_element_type_160: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_24, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_160, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_160 = None
        getitem_108: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_53[0]
        getitem_109: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = var_mean_53[1];  var_mean_53 = None
        add_266: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
        rsqrt_53: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_53: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_24, getitem_109)
        mul_371: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        squeeze_159: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_160: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_53, [0, 2, 3]);  rsqrt_53 = None
        mul_372: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1)
        mul_373: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
        add_267: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
        squeeze_161: "f32[480][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_374: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0012771392081736);  squeeze_161 = None
        mul_375: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
        mul_376: "f32[480][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
        add_268: "f32[480][1]cuda:0" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
        unsqueeze_212: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_323, -1)
        unsqueeze_213: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_377: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
        unsqueeze_214: "f32[480, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_324, -1);  primals_324 = None
        unsqueeze_215: "f32[480, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_269: "f32[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
        convert_element_type_161: "bf16[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_269, torch.bfloat16);  add_269 = None
        relu_53: "bf16[4, 480, 14, 14][94080, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_161);  convert_element_type_161 = None
        convert_element_type_162: "bf16[128, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_325, torch.bfloat16);  primals_325 = None
        convolution_53: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_53, convert_element_type_162, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_270: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_326, 1)
        convert_element_type_163: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_53, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_163, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_163 = None
        getitem_110: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_54[0]
        getitem_111: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_54[1];  var_mean_54 = None
        add_271: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
        rsqrt_54: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_54: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_53, getitem_111)
        mul_378: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        squeeze_162: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_163: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_379: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
        mul_380: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_327, 0.9)
        add_272: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        squeeze_164: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_381: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0012771392081736);  squeeze_164 = None
        mul_382: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
        mul_383: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
        add_273: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
        unsqueeze_216: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_329, -1)
        unsqueeze_217: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        mul_384: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
        unsqueeze_218: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_330, -1);  primals_330 = None
        unsqueeze_219: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        add_274: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
        convert_element_type_164: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_274, torch.bfloat16);  add_274 = None
        relu_54: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_164);  convert_element_type_164 = None
        convert_element_type_165: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_331, torch.bfloat16);  primals_331 = None
        convolution_54: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_54, convert_element_type_165, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_25: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_275: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_332, 1)
        convert_element_type_166: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_25, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_166, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_166 = None
        getitem_112: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_55[0]
        getitem_113: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_55[1];  var_mean_55 = None
        add_276: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
        rsqrt_55: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_55: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_25, getitem_113)
        mul_385: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        squeeze_165: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_166: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_386: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
        mul_387: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_333, 0.9)
        add_277: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
        squeeze_167: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_388: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0012771392081736);  squeeze_167 = None
        mul_389: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
        mul_390: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
        add_278: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
        unsqueeze_220: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_335, -1)
        unsqueeze_221: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_391: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
        unsqueeze_222: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_336, -1);  primals_336 = None
        unsqueeze_223: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_279: "f32[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
        convert_element_type_167: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16);  add_279 = None
        relu_55: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_167);  convert_element_type_167 = None
        convert_element_type_168: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_337, torch.bfloat16);  primals_337 = None
        convolution_55: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_55, convert_element_type_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_280: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_338, 1)
        convert_element_type_169: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32)
        var_mean_56 = torch.ops.aten.var_mean.correction(convert_element_type_169, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_169 = None
        getitem_114: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_56[0]
        getitem_115: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_56[1];  var_mean_56 = None
        add_281: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
        rsqrt_56: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_281);  add_281 = None
        sub_56: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, getitem_115)
        mul_392: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
        squeeze_168: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        squeeze_169: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_393: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
        mul_394: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_339, 0.9)
        add_282: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
        squeeze_170: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_395: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0012771392081736);  squeeze_170 = None
        mul_396: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
        mul_397: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
        add_283: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
        unsqueeze_224: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_341, -1)
        unsqueeze_225: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        mul_398: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
        unsqueeze_226: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_342, -1);  primals_342 = None
        unsqueeze_227: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        add_284: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
        convert_element_type_170: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_284, torch.bfloat16);  add_284 = None
        relu_56: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_170);  convert_element_type_170 = None
        convert_element_type_171: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_343, torch.bfloat16);  primals_343 = None
        convolution_56: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_56, convert_element_type_171, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_26: "bf16[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_285: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_344, 1)
        convert_element_type_172: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_26, torch.float32)
        var_mean_57 = torch.ops.aten.var_mean.correction(convert_element_type_172, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_172 = None
        getitem_116: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = var_mean_57[0]
        getitem_117: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = var_mean_57[1];  var_mean_57 = None
        add_286: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
        rsqrt_57: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_57: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_26, getitem_117)
        mul_399: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
        squeeze_171: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_172: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_400: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
        mul_401: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
        add_287: "f32[544][1]cuda:0" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
        squeeze_173: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_402: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0012771392081736);  squeeze_173 = None
        mul_403: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
        mul_404: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
        add_288: "f32[544][1]cuda:0" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
        unsqueeze_228: "f32[544, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_347, -1)
        unsqueeze_229: "f32[544, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_405: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
        unsqueeze_230: "f32[544, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_348, -1);  primals_348 = None
        unsqueeze_231: "f32[544, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_289: "f32[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
        convert_element_type_173: "bf16[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_289, torch.bfloat16);  add_289 = None
        relu_57: "bf16[4, 544, 14, 14][106624, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_173);  convert_element_type_173 = None
        convert_element_type_174: "bf16[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_349, torch.bfloat16);  primals_349 = None
        convolution_57: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_57, convert_element_type_174, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_290: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_350, 1)
        convert_element_type_175: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32)
        var_mean_58 = torch.ops.aten.var_mean.correction(convert_element_type_175, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_175 = None
        getitem_118: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_58[0]
        getitem_119: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_58[1];  var_mean_58 = None
        add_291: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
        rsqrt_58: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_291);  add_291 = None
        sub_58: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_57, getitem_119)
        mul_406: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
        squeeze_174: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        squeeze_175: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_407: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
        mul_408: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
        add_292: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
        squeeze_176: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_409: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0012771392081736);  squeeze_176 = None
        mul_410: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
        mul_411: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
        add_293: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
        unsqueeze_232: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_353, -1)
        unsqueeze_233: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        mul_412: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
        unsqueeze_234: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_354, -1);  primals_354 = None
        unsqueeze_235: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        add_294: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
        convert_element_type_176: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_294, torch.bfloat16);  add_294 = None
        relu_58: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_176);  convert_element_type_176 = None
        convert_element_type_177: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_355, torch.bfloat16);  primals_355 = None
        convolution_58: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_58, convert_element_type_177, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_27: "bf16[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_295: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_356, 1)
        convert_element_type_178: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_27, torch.float32)
        var_mean_59 = torch.ops.aten.var_mean.correction(convert_element_type_178, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_178 = None
        getitem_120: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = var_mean_59[0]
        getitem_121: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = var_mean_59[1];  var_mean_59 = None
        add_296: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
        rsqrt_59: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        sub_59: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_27, getitem_121)
        mul_413: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
        squeeze_177: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        squeeze_178: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_414: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1)
        mul_415: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
        add_297: "f32[576][1]cuda:0" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
        squeeze_179: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_416: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0012771392081736);  squeeze_179 = None
        mul_417: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
        mul_418: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
        add_298: "f32[576][1]cuda:0" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
        unsqueeze_236: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_359, -1)
        unsqueeze_237: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_419: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
        unsqueeze_238: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_360, -1);  primals_360 = None
        unsqueeze_239: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_299: "f32[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
        convert_element_type_179: "bf16[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_299, torch.bfloat16);  add_299 = None
        relu_59: "bf16[4, 576, 14, 14][112896, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_179);  convert_element_type_179 = None
        convert_element_type_180: "bf16[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_361, torch.bfloat16);  primals_361 = None
        convolution_59: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_59, convert_element_type_180, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_300: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_362, 1)
        convert_element_type_181: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32)
        var_mean_60 = torch.ops.aten.var_mean.correction(convert_element_type_181, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_181 = None
        getitem_122: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_60[0]
        getitem_123: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_60[1];  var_mean_60 = None
        add_301: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
        rsqrt_60: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_301);  add_301 = None
        sub_60: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_59, getitem_123)
        mul_420: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
        squeeze_180: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        squeeze_181: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_421: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1)
        mul_422: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
        add_302: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
        squeeze_182: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_423: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0012771392081736);  squeeze_182 = None
        mul_424: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
        mul_425: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
        add_303: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        unsqueeze_240: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_365, -1)
        unsqueeze_241: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        mul_426: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
        unsqueeze_242: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_366, -1);  primals_366 = None
        unsqueeze_243: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        add_304: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
        convert_element_type_182: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_304, torch.bfloat16);  add_304 = None
        relu_60: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_182);  convert_element_type_182 = None
        convert_element_type_183: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_367, torch.bfloat16);  primals_367 = None
        convolution_60: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_60, convert_element_type_183, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_28: "bf16[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_305: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_368, 1)
        convert_element_type_184: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_28, torch.float32)
        var_mean_61 = torch.ops.aten.var_mean.correction(convert_element_type_184, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_184 = None
        getitem_124: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = var_mean_61[0]
        getitem_125: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = var_mean_61[1];  var_mean_61 = None
        add_306: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_124, 1e-05)
        rsqrt_61: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_61: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_28, getitem_125)
        mul_427: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
        squeeze_183: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_184: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_428: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
        mul_429: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
        add_307: "f32[608][1]cuda:0" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
        squeeze_185: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_430: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0012771392081736);  squeeze_185 = None
        mul_431: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
        mul_432: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
        add_308: "f32[608][1]cuda:0" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
        unsqueeze_244: "f32[608, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_371, -1)
        unsqueeze_245: "f32[608, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_433: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
        unsqueeze_246: "f32[608, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_372, -1);  primals_372 = None
        unsqueeze_247: "f32[608, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_309: "f32[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
        convert_element_type_185: "bf16[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_309, torch.bfloat16);  add_309 = None
        relu_61: "bf16[4, 608, 14, 14][119168, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_185);  convert_element_type_185 = None
        convert_element_type_186: "bf16[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_373, torch.bfloat16);  primals_373 = None
        convolution_61: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_61, convert_element_type_186, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_310: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_374, 1)
        convert_element_type_187: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32)
        var_mean_62 = torch.ops.aten.var_mean.correction(convert_element_type_187, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_187 = None
        getitem_126: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_62[0]
        getitem_127: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_62[1];  var_mean_62 = None
        add_311: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_126, 1e-05)
        rsqrt_62: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_311);  add_311 = None
        sub_62: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, getitem_127)
        mul_434: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
        squeeze_186: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        squeeze_187: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
        mul_435: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
        mul_436: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
        add_312: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
        squeeze_188: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_437: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0012771392081736);  squeeze_188 = None
        mul_438: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
        mul_439: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
        add_313: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
        unsqueeze_248: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_377, -1)
        unsqueeze_249: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        mul_440: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
        unsqueeze_250: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_378, -1);  primals_378 = None
        unsqueeze_251: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        add_314: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
        convert_element_type_188: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_314, torch.bfloat16);  add_314 = None
        relu_62: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_188);  convert_element_type_188 = None
        convert_element_type_189: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_379, torch.bfloat16);  primals_379 = None
        convolution_62: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_62, convert_element_type_189, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_29: "bf16[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_315: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_380, 1)
        convert_element_type_190: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_29, torch.float32)
        var_mean_63 = torch.ops.aten.var_mean.correction(convert_element_type_190, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_190 = None
        getitem_128: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_63[0]
        getitem_129: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_63[1];  var_mean_63 = None
        add_316: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 1e-05)
        rsqrt_63: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
        sub_63: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_29, getitem_129)
        mul_441: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
        squeeze_189: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        squeeze_190: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
        mul_442: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1)
        mul_443: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
        add_317: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
        squeeze_191: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
        mul_444: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0012771392081736);  squeeze_191 = None
        mul_445: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
        mul_446: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
        add_318: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
        unsqueeze_252: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_383, -1)
        unsqueeze_253: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_447: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
        unsqueeze_254: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_384, -1);  primals_384 = None
        unsqueeze_255: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_319: "f32[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
        convert_element_type_191: "bf16[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_319, torch.bfloat16);  add_319 = None
        relu_63: "bf16[4, 640, 14, 14][125440, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_191);  convert_element_type_191 = None
        convert_element_type_192: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_385, torch.bfloat16);  primals_385 = None
        convolution_63: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_63, convert_element_type_192, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_320: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_386, 1)
        convert_element_type_193: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32)
        var_mean_64 = torch.ops.aten.var_mean.correction(convert_element_type_193, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_193 = None
        getitem_130: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_64[0]
        getitem_131: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_64[1];  var_mean_64 = None
        add_321: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 1e-05)
        rsqrt_64: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
        sub_64: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_63, getitem_131)
        mul_448: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
        squeeze_192: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
        squeeze_193: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
        mul_449: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
        mul_450: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
        add_322: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
        squeeze_194: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
        mul_451: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0012771392081736);  squeeze_194 = None
        mul_452: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
        mul_453: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
        add_323: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
        unsqueeze_256: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_389, -1)
        unsqueeze_257: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        mul_454: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
        unsqueeze_258: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_390, -1);  primals_390 = None
        unsqueeze_259: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        add_324: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
        convert_element_type_194: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_324, torch.bfloat16);  add_324 = None
        relu_64: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_194);  convert_element_type_194 = None
        convert_element_type_195: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_391, torch.bfloat16);  primals_391 = None
        convolution_64: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_64, convert_element_type_195, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_30: "bf16[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_325: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_392, 1)
        convert_element_type_196: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_30, torch.float32)
        var_mean_65 = torch.ops.aten.var_mean.correction(convert_element_type_196, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_196 = None
        getitem_132: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_65[0]
        getitem_133: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_65[1];  var_mean_65 = None
        add_326: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 1e-05)
        rsqrt_65: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
        sub_65: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_30, getitem_133)
        mul_455: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
        squeeze_195: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        squeeze_196: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
        mul_456: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
        mul_457: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_393, 0.9)
        add_327: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
        squeeze_197: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
        mul_458: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0012771392081736);  squeeze_197 = None
        mul_459: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
        mul_460: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
        add_328: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
        unsqueeze_260: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_395, -1)
        unsqueeze_261: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_461: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_261);  mul_455 = unsqueeze_261 = None
        unsqueeze_262: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_396, -1);  primals_396 = None
        unsqueeze_263: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_329: "f32[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_263);  mul_461 = unsqueeze_263 = None
        convert_element_type_197: "bf16[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_329, torch.bfloat16);  add_329 = None
        relu_65: "bf16[4, 672, 14, 14][131712, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_197);  convert_element_type_197 = None
        convert_element_type_198: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_397, torch.bfloat16);  primals_397 = None
        convolution_65: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_65, convert_element_type_198, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_330: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_398, 1)
        convert_element_type_199: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_65, torch.float32)
        var_mean_66 = torch.ops.aten.var_mean.correction(convert_element_type_199, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_199 = None
        getitem_134: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_66[0]
        getitem_135: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_66[1];  var_mean_66 = None
        add_331: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_134, 1e-05)
        rsqrt_66: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_66: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_65, getitem_135)
        mul_462: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
        squeeze_198: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
        squeeze_199: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
        mul_463: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
        mul_464: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_399, 0.9)
        add_332: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
        squeeze_200: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
        mul_465: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0012771392081736);  squeeze_200 = None
        mul_466: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
        mul_467: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
        add_333: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        unsqueeze_264: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_401, -1)
        unsqueeze_265: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        mul_468: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_265);  mul_462 = unsqueeze_265 = None
        unsqueeze_266: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_402, -1);  primals_402 = None
        unsqueeze_267: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        add_334: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_267);  mul_468 = unsqueeze_267 = None
        convert_element_type_200: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_334, torch.bfloat16);  add_334 = None
        relu_66: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_200);  convert_element_type_200 = None
        convert_element_type_201: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_403, torch.bfloat16);  primals_403 = None
        convolution_66: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_66, convert_element_type_201, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_31: "bf16[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_335: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_404, 1)
        convert_element_type_202: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_31, torch.float32)
        var_mean_67 = torch.ops.aten.var_mean.correction(convert_element_type_202, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_202 = None
        getitem_136: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = var_mean_67[0]
        getitem_137: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = var_mean_67[1];  var_mean_67 = None
        add_336: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_136, 1e-05)
        rsqrt_67: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
        sub_67: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_31, getitem_137)
        mul_469: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
        squeeze_201: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        squeeze_202: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
        mul_470: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
        mul_471: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_405, 0.9)
        add_337: "f32[704][1]cuda:0" = torch.ops.aten.add.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
        squeeze_203: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
        mul_472: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0012771392081736);  squeeze_203 = None
        mul_473: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_472, 0.1);  mul_472 = None
        mul_474: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
        add_338: "f32[704][1]cuda:0" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
        unsqueeze_268: "f32[704, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_407, -1)
        unsqueeze_269: "f32[704, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_475: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_269);  mul_469 = unsqueeze_269 = None
        unsqueeze_270: "f32[704, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_408, -1);  primals_408 = None
        unsqueeze_271: "f32[704, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_339: "f32[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_271);  mul_475 = unsqueeze_271 = None
        convert_element_type_203: "bf16[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_339, torch.bfloat16);  add_339 = None
        relu_67: "bf16[4, 704, 14, 14][137984, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_203);  convert_element_type_203 = None
        convert_element_type_204: "bf16[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_409, torch.bfloat16);  primals_409 = None
        convolution_67: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_67, convert_element_type_204, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_340: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_410, 1)
        convert_element_type_205: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32)
        var_mean_68 = torch.ops.aten.var_mean.correction(convert_element_type_205, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_205 = None
        getitem_138: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_68[0]
        getitem_139: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_68[1];  var_mean_68 = None
        add_341: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_138, 1e-05)
        rsqrt_68: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_68: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_67, getitem_139)
        mul_476: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
        squeeze_204: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        squeeze_205: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_68, [0, 2, 3]);  rsqrt_68 = None
        mul_477: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1)
        mul_478: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_411, 0.9)
        add_342: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
        squeeze_206: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
        mul_479: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0012771392081736);  squeeze_206 = None
        mul_480: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_479, 0.1);  mul_479 = None
        mul_481: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
        add_343: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
        unsqueeze_272: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_413, -1)
        unsqueeze_273: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        mul_482: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
        unsqueeze_274: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_414, -1);  primals_414 = None
        unsqueeze_275: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        add_344: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None
        convert_element_type_206: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_344, torch.bfloat16);  add_344 = None
        relu_68: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_206);  convert_element_type_206 = None
        convert_element_type_207: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_415, torch.bfloat16);  primals_415 = None
        convolution_68: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_68, convert_element_type_207, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_32: "bf16[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_345: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_416, 1)
        convert_element_type_208: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_32, torch.float32)
        var_mean_69 = torch.ops.aten.var_mean.correction(convert_element_type_208, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_208 = None
        getitem_140: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = var_mean_69[0]
        getitem_141: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = var_mean_69[1];  var_mean_69 = None
        add_346: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_140, 1e-05)
        rsqrt_69: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
        sub_69: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_32, getitem_141)
        mul_483: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
        squeeze_207: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
        squeeze_208: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
        mul_484: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1)
        mul_485: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_417, 0.9)
        add_347: "f32[736][1]cuda:0" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
        squeeze_209: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
        mul_486: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0012771392081736);  squeeze_209 = None
        mul_487: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, 0.1);  mul_486 = None
        mul_488: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
        add_348: "f32[736][1]cuda:0" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
        unsqueeze_276: "f32[736, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_419, -1)
        unsqueeze_277: "f32[736, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_489: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
        unsqueeze_278: "f32[736, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_420, -1);  primals_420 = None
        unsqueeze_279: "f32[736, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_349: "f32[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None
        convert_element_type_209: "bf16[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_349, torch.bfloat16);  add_349 = None
        relu_69: "bf16[4, 736, 14, 14][144256, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_209);  convert_element_type_209 = None
        convert_element_type_210: "bf16[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_421, torch.bfloat16);  primals_421 = None
        convolution_69: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_69, convert_element_type_210, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_350: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_422, 1)
        convert_element_type_211: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32)
        var_mean_70 = torch.ops.aten.var_mean.correction(convert_element_type_211, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_211 = None
        getitem_142: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_70[0]
        getitem_143: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_70[1];  var_mean_70 = None
        add_351: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_142, 1e-05)
        rsqrt_70: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        sub_70: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_69, getitem_143)
        mul_490: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
        squeeze_210: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
        squeeze_211: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
        mul_491: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
        mul_492: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_423, 0.9)
        add_352: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
        squeeze_212: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
        mul_493: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0012771392081736);  squeeze_212 = None
        mul_494: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_493, 0.1);  mul_493 = None
        mul_495: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
        add_353: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
        unsqueeze_280: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_425, -1)
        unsqueeze_281: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        mul_496: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_281);  mul_490 = unsqueeze_281 = None
        unsqueeze_282: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_426, -1);  primals_426 = None
        unsqueeze_283: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        add_354: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_283);  mul_496 = unsqueeze_283 = None
        convert_element_type_212: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_354, torch.bfloat16);  add_354 = None
        relu_70: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_212);  convert_element_type_212 = None
        convert_element_type_213: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_427, torch.bfloat16);  primals_427 = None
        convolution_70: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_70, convert_element_type_213, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_33: "bf16[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_355: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_428, 1)
        convert_element_type_214: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_33, torch.float32)
        var_mean_71 = torch.ops.aten.var_mean.correction(convert_element_type_214, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_214 = None
        getitem_144: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_71[0]
        getitem_145: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_71[1];  var_mean_71 = None
        add_356: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_144, 1e-05)
        rsqrt_71: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
        sub_71: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_33, getitem_145)
        mul_497: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
        squeeze_213: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        squeeze_214: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
        mul_498: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1)
        mul_499: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_429, 0.9)
        add_357: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
        squeeze_215: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
        mul_500: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0012771392081736);  squeeze_215 = None
        mul_501: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_500, 0.1);  mul_500 = None
        mul_502: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
        add_358: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
        unsqueeze_284: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_431, -1)
        unsqueeze_285: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_503: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
        unsqueeze_286: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_432, -1);  primals_432 = None
        unsqueeze_287: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_359: "f32[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None
        convert_element_type_215: "bf16[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_359, torch.bfloat16);  add_359 = None
        relu_71: "bf16[4, 768, 14, 14][150528, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_215);  convert_element_type_215 = None
        convert_element_type_216: "bf16[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_433, torch.bfloat16);  primals_433 = None
        convolution_71: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_71, convert_element_type_216, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_360: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_434, 1)
        convert_element_type_217: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_71, torch.float32)
        var_mean_72 = torch.ops.aten.var_mean.correction(convert_element_type_217, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_217 = None
        getitem_146: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_72[0]
        getitem_147: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_72[1];  var_mean_72 = None
        add_361: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_146, 1e-05)
        rsqrt_72: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
        sub_72: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_71, getitem_147)
        mul_504: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
        squeeze_216: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
        squeeze_217: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
        mul_505: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
        mul_506: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_435, 0.9)
        add_362: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
        squeeze_218: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
        mul_507: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0012771392081736);  squeeze_218 = None
        mul_508: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
        mul_509: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
        add_363: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
        unsqueeze_288: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_437, -1)
        unsqueeze_289: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        mul_510: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_289);  mul_504 = unsqueeze_289 = None
        unsqueeze_290: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_438, -1);  primals_438 = None
        unsqueeze_291: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        add_364: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_291);  mul_510 = unsqueeze_291 = None
        convert_element_type_218: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_364, torch.bfloat16);  add_364 = None
        relu_72: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_218);  convert_element_type_218 = None
        convert_element_type_219: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_439, torch.bfloat16);  primals_439 = None
        convolution_72: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_72, convert_element_type_219, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_34: "bf16[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_365: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_440, 1)
        convert_element_type_220: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_34, torch.float32)
        var_mean_73 = torch.ops.aten.var_mean.correction(convert_element_type_220, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_220 = None
        getitem_148: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = var_mean_73[0]
        getitem_149: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = var_mean_73[1];  var_mean_73 = None
        add_366: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_148, 1e-05)
        rsqrt_73: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_366);  add_366 = None
        sub_73: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_34, getitem_149)
        mul_511: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
        squeeze_219: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
        squeeze_220: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
        mul_512: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
        mul_513: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_441, 0.9)
        add_367: "f32[800][1]cuda:0" = torch.ops.aten.add.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
        squeeze_221: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
        mul_514: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0012771392081736);  squeeze_221 = None
        mul_515: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_514, 0.1);  mul_514 = None
        mul_516: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
        add_368: "f32[800][1]cuda:0" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
        unsqueeze_292: "f32[800, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_443, -1)
        unsqueeze_293: "f32[800, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_517: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_293);  mul_511 = unsqueeze_293 = None
        unsqueeze_294: "f32[800, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_444, -1);  primals_444 = None
        unsqueeze_295: "f32[800, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_369: "f32[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_517, unsqueeze_295);  mul_517 = unsqueeze_295 = None
        convert_element_type_221: "bf16[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_369, torch.bfloat16);  add_369 = None
        relu_73: "bf16[4, 800, 14, 14][156800, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_221);  convert_element_type_221 = None
        convert_element_type_222: "bf16[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_445, torch.bfloat16);  primals_445 = None
        convolution_73: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_73, convert_element_type_222, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_370: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_446, 1)
        convert_element_type_223: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_73, torch.float32)
        var_mean_74 = torch.ops.aten.var_mean.correction(convert_element_type_223, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_223 = None
        getitem_150: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_74[0]
        getitem_151: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_74[1];  var_mean_74 = None
        add_371: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_150, 1e-05)
        rsqrt_74: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_371);  add_371 = None
        sub_74: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_73, getitem_151)
        mul_518: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
        squeeze_222: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
        squeeze_223: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
        mul_519: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
        mul_520: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_447, 0.9)
        add_372: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
        squeeze_224: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
        mul_521: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0012771392081736);  squeeze_224 = None
        mul_522: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_521, 0.1);  mul_521 = None
        mul_523: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
        add_373: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
        unsqueeze_296: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_449, -1)
        unsqueeze_297: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        mul_524: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_297);  mul_518 = unsqueeze_297 = None
        unsqueeze_298: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_450, -1);  primals_450 = None
        unsqueeze_299: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        add_374: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_299);  mul_524 = unsqueeze_299 = None
        convert_element_type_224: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_374, torch.bfloat16);  add_374 = None
        relu_74: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_224);  convert_element_type_224 = None
        convert_element_type_225: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_451, torch.bfloat16);  primals_451 = None
        convolution_74: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_74, convert_element_type_225, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_35: "bf16[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_375: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_452, 1)
        convert_element_type_226: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_35, torch.float32)
        var_mean_75 = torch.ops.aten.var_mean.correction(convert_element_type_226, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_226 = None
        getitem_152: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = var_mean_75[0]
        getitem_153: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = var_mean_75[1];  var_mean_75 = None
        add_376: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_152, 1e-05)
        rsqrt_75: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
        sub_75: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_35, getitem_153)
        mul_525: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
        squeeze_225: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        squeeze_226: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
        mul_526: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1)
        mul_527: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_453, 0.9)
        add_377: "f32[832][1]cuda:0" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
        squeeze_227: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
        mul_528: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0012771392081736);  squeeze_227 = None
        mul_529: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
        mul_530: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
        add_378: "f32[832][1]cuda:0" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
        unsqueeze_300: "f32[832, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_455, -1)
        unsqueeze_301: "f32[832, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_531: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
        unsqueeze_302: "f32[832, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_456, -1);  primals_456 = None
        unsqueeze_303: "f32[832, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_379: "f32[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None
        convert_element_type_227: "bf16[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_379, torch.bfloat16);  add_379 = None
        relu_75: "bf16[4, 832, 14, 14][163072, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_227);  convert_element_type_227 = None
        convert_element_type_228: "bf16[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_457, torch.bfloat16);  primals_457 = None
        convolution_75: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_75, convert_element_type_228, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_380: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_458, 1)
        convert_element_type_229: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32)
        var_mean_76 = torch.ops.aten.var_mean.correction(convert_element_type_229, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_229 = None
        getitem_154: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_76[0]
        getitem_155: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_76[1];  var_mean_76 = None
        add_381: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_154, 1e-05)
        rsqrt_76: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_381);  add_381 = None
        sub_76: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_75, getitem_155)
        mul_532: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
        squeeze_228: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
        squeeze_229: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_76, [0, 2, 3]);  rsqrt_76 = None
        mul_533: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1)
        mul_534: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_459, 0.9)
        add_382: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
        squeeze_230: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
        mul_535: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0012771392081736);  squeeze_230 = None
        mul_536: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_535, 0.1);  mul_535 = None
        mul_537: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
        add_383: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
        unsqueeze_304: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_461, -1)
        unsqueeze_305: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        mul_538: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
        unsqueeze_306: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_462, -1);  primals_462 = None
        unsqueeze_307: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        add_384: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None
        convert_element_type_230: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_384, torch.bfloat16);  add_384 = None
        relu_76: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_230);  convert_element_type_230 = None
        convert_element_type_231: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_463, torch.bfloat16);  primals_463 = None
        convolution_76: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_76, convert_element_type_231, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_36: "bf16[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_385: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_464, 1)
        convert_element_type_232: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_36, torch.float32)
        var_mean_77 = torch.ops.aten.var_mean.correction(convert_element_type_232, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_232 = None
        getitem_156: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = var_mean_77[0]
        getitem_157: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = var_mean_77[1];  var_mean_77 = None
        add_386: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_156, 1e-05)
        rsqrt_77: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_386);  add_386 = None
        sub_77: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_36, getitem_157)
        mul_539: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
        squeeze_231: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        squeeze_232: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
        mul_540: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
        mul_541: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_465, 0.9)
        add_387: "f32[864][1]cuda:0" = torch.ops.aten.add.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
        squeeze_233: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
        mul_542: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0012771392081736);  squeeze_233 = None
        mul_543: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_542, 0.1);  mul_542 = None
        mul_544: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
        add_388: "f32[864][1]cuda:0" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
        unsqueeze_308: "f32[864, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_467, -1)
        unsqueeze_309: "f32[864, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_545: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_309);  mul_539 = unsqueeze_309 = None
        unsqueeze_310: "f32[864, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_468, -1);  primals_468 = None
        unsqueeze_311: "f32[864, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_389: "f32[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_311);  mul_545 = unsqueeze_311 = None
        convert_element_type_233: "bf16[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_389, torch.bfloat16);  add_389 = None
        relu_77: "bf16[4, 864, 14, 14][169344, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_233);  convert_element_type_233 = None
        convert_element_type_234: "bf16[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_469, torch.bfloat16);  primals_469 = None
        convolution_77: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_77, convert_element_type_234, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_390: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_470, 1)
        convert_element_type_235: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32)
        var_mean_78 = torch.ops.aten.var_mean.correction(convert_element_type_235, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_235 = None
        getitem_158: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_78[0]
        getitem_159: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_78[1];  var_mean_78 = None
        add_391: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_158, 1e-05)
        rsqrt_78: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_391);  add_391 = None
        sub_78: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_77, getitem_159)
        mul_546: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
        squeeze_234: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
        squeeze_235: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
        mul_547: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1)
        mul_548: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_471, 0.9)
        add_392: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
        squeeze_236: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
        mul_549: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0012771392081736);  squeeze_236 = None
        mul_550: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
        mul_551: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
        add_393: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
        unsqueeze_312: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_473, -1)
        unsqueeze_313: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        mul_552: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
        unsqueeze_314: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_474, -1);  primals_474 = None
        unsqueeze_315: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        add_394: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None
        convert_element_type_236: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_394, torch.bfloat16);  add_394 = None
        relu_78: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_236);  convert_element_type_236 = None
        convert_element_type_237: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_475, torch.bfloat16);  primals_475 = None
        convolution_78: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_78, convert_element_type_237, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_37: "bf16[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_395: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_476, 1)
        convert_element_type_238: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_37, torch.float32)
        var_mean_79 = torch.ops.aten.var_mean.correction(convert_element_type_238, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_238 = None
        getitem_160: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = var_mean_79[0]
        getitem_161: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = var_mean_79[1];  var_mean_79 = None
        add_396: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_160, 1e-05)
        rsqrt_79: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_396);  add_396 = None
        sub_79: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_37, getitem_161)
        mul_553: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
        squeeze_237: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3]);  getitem_161 = None
        squeeze_238: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_79, [0, 2, 3]);  rsqrt_79 = None
        mul_554: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1)
        mul_555: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_477, 0.9)
        add_397: "f32[896][1]cuda:0" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
        squeeze_239: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
        mul_556: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0012771392081736);  squeeze_239 = None
        mul_557: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
        mul_558: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
        add_398: "f32[896][1]cuda:0" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
        unsqueeze_316: "f32[896, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_479, -1)
        unsqueeze_317: "f32[896, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_559: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
        unsqueeze_318: "f32[896, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_480, -1);  primals_480 = None
        unsqueeze_319: "f32[896, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_399: "f32[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None
        convert_element_type_239: "bf16[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_399, torch.bfloat16);  add_399 = None
        relu_79: "bf16[4, 896, 14, 14][175616, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_239);  convert_element_type_239 = None
        convert_element_type_240: "bf16[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_481, torch.bfloat16);  primals_481 = None
        convolution_79: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_79, convert_element_type_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_400: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_482, 1)
        convert_element_type_241: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_79, torch.float32)
        var_mean_80 = torch.ops.aten.var_mean.correction(convert_element_type_241, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_241 = None
        getitem_162: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_80[0]
        getitem_163: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_80[1];  var_mean_80 = None
        add_401: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_162, 1e-05)
        rsqrt_80: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
        sub_80: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_79, getitem_163)
        mul_560: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
        squeeze_240: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
        squeeze_241: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
        mul_561: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
        mul_562: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_483, 0.9)
        add_402: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
        squeeze_242: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
        mul_563: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0012771392081736);  squeeze_242 = None
        mul_564: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
        mul_565: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
        add_403: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
        unsqueeze_320: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_485, -1)
        unsqueeze_321: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        mul_566: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_321);  mul_560 = unsqueeze_321 = None
        unsqueeze_322: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_486, -1);  primals_486 = None
        unsqueeze_323: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        add_404: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_323);  mul_566 = unsqueeze_323 = None
        convert_element_type_242: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_404, torch.bfloat16);  add_404 = None
        relu_80: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_242);  convert_element_type_242 = None
        convert_element_type_243: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_487, torch.bfloat16);  primals_487 = None
        convolution_80: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_80, convert_element_type_243, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_38: "bf16[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_405: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_488, 1)
        convert_element_type_244: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_38, torch.float32)
        var_mean_81 = torch.ops.aten.var_mean.correction(convert_element_type_244, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_244 = None
        getitem_164: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = var_mean_81[0]
        getitem_165: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = var_mean_81[1];  var_mean_81 = None
        add_406: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_164, 1e-05)
        rsqrt_81: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_406);  add_406 = None
        sub_81: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_38, getitem_165)
        mul_567: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
        squeeze_243: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3]);  getitem_165 = None
        squeeze_244: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
        mul_568: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
        mul_569: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_489, 0.9)
        add_407: "f32[928][1]cuda:0" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
        squeeze_245: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
        mul_570: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0012771392081736);  squeeze_245 = None
        mul_571: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
        mul_572: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_490, 0.9)
        add_408: "f32[928][1]cuda:0" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
        unsqueeze_324: "f32[928, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_491, -1)
        unsqueeze_325: "f32[928, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_573: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_325);  mul_567 = unsqueeze_325 = None
        unsqueeze_326: "f32[928, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_492, -1);  primals_492 = None
        unsqueeze_327: "f32[928, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_409: "f32[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_327);  mul_573 = unsqueeze_327 = None
        convert_element_type_245: "bf16[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_409, torch.bfloat16);  add_409 = None
        relu_81: "bf16[4, 928, 14, 14][181888, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_245);  convert_element_type_245 = None
        convert_element_type_246: "bf16[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_493, torch.bfloat16);  primals_493 = None
        convolution_81: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_81, convert_element_type_246, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_410: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_494, 1)
        convert_element_type_247: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_81, torch.float32)
        var_mean_82 = torch.ops.aten.var_mean.correction(convert_element_type_247, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_247 = None
        getitem_166: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_82[0]
        getitem_167: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_82[1];  var_mean_82 = None
        add_411: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_166, 1e-05)
        rsqrt_82: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_411);  add_411 = None
        sub_82: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_81, getitem_167)
        mul_574: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
        squeeze_246: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3]);  getitem_167 = None
        squeeze_247: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_82, [0, 2, 3]);  rsqrt_82 = None
        mul_575: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1)
        mul_576: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_495, 0.9)
        add_412: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
        squeeze_248: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
        mul_577: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0012771392081736);  squeeze_248 = None
        mul_578: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_577, 0.1);  mul_577 = None
        mul_579: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_496, 0.9)
        add_413: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
        unsqueeze_328: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_497, -1)
        unsqueeze_329: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        mul_580: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
        unsqueeze_330: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_498, -1);  primals_498 = None
        unsqueeze_331: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        add_414: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None
        convert_element_type_248: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_414, torch.bfloat16);  add_414 = None
        relu_82: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_248);  convert_element_type_248 = None
        convert_element_type_249: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_499, torch.bfloat16);  primals_499 = None
        convolution_82: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_82, convert_element_type_249, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_39: "bf16[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80, convolution_82], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_415: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_500, 1)
        convert_element_type_250: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_39, torch.float32)
        var_mean_83 = torch.ops.aten.var_mean.correction(convert_element_type_250, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_250 = None
        getitem_168: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_83[0]
        getitem_169: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_83[1];  var_mean_83 = None
        add_416: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_168, 1e-05)
        rsqrt_83: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_416);  add_416 = None
        sub_83: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_39, getitem_169)
        mul_581: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
        squeeze_249: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
        squeeze_250: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_83, [0, 2, 3]);  rsqrt_83 = None
        mul_582: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1)
        mul_583: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_501, 0.9)
        add_417: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
        squeeze_251: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
        mul_584: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0012771392081736);  squeeze_251 = None
        mul_585: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_584, 0.1);  mul_584 = None
        mul_586: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_502, 0.9)
        add_418: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
        unsqueeze_332: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_503, -1)
        unsqueeze_333: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_587: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
        unsqueeze_334: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_504, -1);  primals_504 = None
        unsqueeze_335: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_419: "f32[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None
        convert_element_type_251: "bf16[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_419, torch.bfloat16);  add_419 = None
        relu_83: "bf16[4, 960, 14, 14][188160, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_251);  convert_element_type_251 = None
        convert_element_type_252: "bf16[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_505, torch.bfloat16);  primals_505 = None
        convolution_83: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_83, convert_element_type_252, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_420: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_506, 1)
        convert_element_type_253: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_83, torch.float32)
        var_mean_84 = torch.ops.aten.var_mean.correction(convert_element_type_253, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_253 = None
        getitem_170: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_84[0]
        getitem_171: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_84[1];  var_mean_84 = None
        add_421: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_170, 1e-05)
        rsqrt_84: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_421);  add_421 = None
        sub_84: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_83, getitem_171)
        mul_588: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
        squeeze_252: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
        squeeze_253: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_84, [0, 2, 3]);  rsqrt_84 = None
        mul_589: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1)
        mul_590: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_507, 0.9)
        add_422: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
        squeeze_254: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
        mul_591: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0012771392081736);  squeeze_254 = None
        mul_592: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
        mul_593: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_508, 0.9)
        add_423: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        unsqueeze_336: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_509, -1)
        unsqueeze_337: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        mul_594: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
        unsqueeze_338: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_510, -1);  primals_510 = None
        unsqueeze_339: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        add_424: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None
        convert_element_type_254: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_424, torch.bfloat16);  add_424 = None
        relu_84: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_254);  convert_element_type_254 = None
        convert_element_type_255: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_511, torch.bfloat16);  primals_511 = None
        convolution_84: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_84, convert_element_type_255, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_40: "bf16[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80, convolution_82, convolution_84], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_425: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_512, 1)
        convert_element_type_256: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_40, torch.float32)
        var_mean_85 = torch.ops.aten.var_mean.correction(convert_element_type_256, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_256 = None
        getitem_172: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = var_mean_85[0]
        getitem_173: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = var_mean_85[1];  var_mean_85 = None
        add_426: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_172, 1e-05)
        rsqrt_85: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_426);  add_426 = None
        sub_85: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_40, getitem_173)
        mul_595: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
        squeeze_255: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3]);  getitem_173 = None
        squeeze_256: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_85, [0, 2, 3]);  rsqrt_85 = None
        mul_596: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1)
        mul_597: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_513, 0.9)
        add_427: "f32[992][1]cuda:0" = torch.ops.aten.add.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
        squeeze_257: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
        mul_598: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0012771392081736);  squeeze_257 = None
        mul_599: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_598, 0.1);  mul_598 = None
        mul_600: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_514, 0.9)
        add_428: "f32[992][1]cuda:0" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
        unsqueeze_340: "f32[992, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_515, -1)
        unsqueeze_341: "f32[992, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_601: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_341);  mul_595 = unsqueeze_341 = None
        unsqueeze_342: "f32[992, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_516, -1);  primals_516 = None
        unsqueeze_343: "f32[992, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_429: "f32[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_343);  mul_601 = unsqueeze_343 = None
        convert_element_type_257: "bf16[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_429, torch.bfloat16);  add_429 = None
        relu_85: "bf16[4, 992, 14, 14][194432, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_257);  convert_element_type_257 = None
        convert_element_type_258: "bf16[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_517, torch.bfloat16);  primals_517 = None
        convolution_85: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_85, convert_element_type_258, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_430: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_518, 1)
        convert_element_type_259: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_85, torch.float32)
        var_mean_86 = torch.ops.aten.var_mean.correction(convert_element_type_259, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_259 = None
        getitem_174: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_86[0]
        getitem_175: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_86[1];  var_mean_86 = None
        add_431: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_174, 1e-05)
        rsqrt_86: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
        sub_86: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_85, getitem_175)
        mul_602: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
        squeeze_258: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3]);  getitem_175 = None
        squeeze_259: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
        mul_603: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
        mul_604: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_519, 0.9)
        add_432: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
        squeeze_260: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
        mul_605: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0012771392081736);  squeeze_260 = None
        mul_606: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_605, 0.1);  mul_605 = None
        mul_607: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_520, 0.9)
        add_433: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
        unsqueeze_344: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_521, -1)
        unsqueeze_345: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        mul_608: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_345);  mul_602 = unsqueeze_345 = None
        unsqueeze_346: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_522, -1);  primals_522 = None
        unsqueeze_347: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        add_434: "f32[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_347);  mul_608 = unsqueeze_347 = None
        convert_element_type_260: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_434, torch.bfloat16);  add_434 = None
        relu_86: "bf16[4, 128, 14, 14][25088, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_260);  convert_element_type_260 = None
        convert_element_type_261: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_523, torch.bfloat16);  primals_523 = None
        convolution_86: "bf16[4, 32, 14, 14][6272, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_86, convert_element_type_261, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        cat_41: "bf16[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_1, convolution_40, convolution_42, convolution_44, convolution_46, convolution_48, convolution_50, convolution_52, convolution_54, convolution_56, convolution_58, convolution_60, convolution_62, convolution_64, convolution_66, convolution_68, convolution_70, convolution_72, convolution_74, convolution_76, convolution_78, convolution_80, convolution_82, convolution_84, convolution_86], 1);  convolution_40 = convolution_42 = convolution_44 = convolution_46 = convolution_48 = convolution_50 = convolution_52 = convolution_54 = convolution_56 = convolution_58 = convolution_60 = convolution_62 = convolution_64 = convolution_66 = convolution_68 = convolution_70 = convolution_72 = convolution_74 = convolution_76 = convolution_78 = convolution_80 = convolution_82 = convolution_84 = convolution_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        add_435: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_524, 1)
        convert_element_type_262: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_41, torch.float32)
        var_mean_87 = torch.ops.aten.var_mean.correction(convert_element_type_262, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_262 = None
        getitem_176: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_87[0]
        getitem_177: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_87[1];  var_mean_87 = None
        add_436: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_176, 1e-05)
        rsqrt_87: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_436);  add_436 = None
        sub_87: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_41, getitem_177)
        mul_609: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
        squeeze_261: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3]);  getitem_177 = None
        squeeze_262: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_87, [0, 2, 3]);  rsqrt_87 = None
        mul_610: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1)
        mul_611: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_525, 0.9)
        add_437: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
        squeeze_263: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
        mul_612: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0012771392081736);  squeeze_263 = None
        mul_613: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_612, 0.1);  mul_612 = None
        mul_614: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
        add_438: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
        unsqueeze_348: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_527, -1)
        unsqueeze_349: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_615: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_349);  mul_609 = unsqueeze_349 = None
        unsqueeze_350: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_528, -1);  primals_528 = None
        unsqueeze_351: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_439: "f32[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_351);  mul_615 = unsqueeze_351 = None
        convert_element_type_263: "bf16[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_439, torch.bfloat16);  add_439 = None
        relu_87: "bf16[4, 1024, 14, 14][200704, 196, 14, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_263);  convert_element_type_263 = None
        convert_element_type_264: "bf16[512, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_529, torch.bfloat16);  primals_529 = None
        convolution_87: "bf16[4, 512, 14, 14][100352, 196, 14, 1]cuda:0" = torch.ops.aten.convolution.default(relu_87, convert_element_type_264, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)
        avg_pool2d_2: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.avg_pool2d.default(convolution_87, [2, 2], [2, 2])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_440: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_530, 1)
        convert_element_type_265: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(avg_pool2d_2, torch.float32)
        var_mean_88 = torch.ops.aten.var_mean.correction(convert_element_type_265, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_265 = None
        getitem_178: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_88[0]
        getitem_179: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = var_mean_88[1];  var_mean_88 = None
        add_441: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_178, 1e-05)
        rsqrt_88: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_441);  add_441 = None
        sub_88: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(avg_pool2d_2, getitem_179)
        mul_616: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
        squeeze_264: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3]);  getitem_179 = None
        squeeze_265: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_88, [0, 2, 3]);  rsqrt_88 = None
        mul_617: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1)
        mul_618: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_531, 0.9)
        add_442: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
        squeeze_266: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
        mul_619: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_266, 1.005128205128205);  squeeze_266 = None
        mul_620: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_619, 0.1);  mul_619 = None
        mul_621: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
        add_443: "f32[512][1]cuda:0" = torch.ops.aten.add.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
        unsqueeze_352: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_533, -1)
        unsqueeze_353: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        mul_622: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_353);  mul_616 = unsqueeze_353 = None
        unsqueeze_354: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_534, -1);  primals_534 = None
        unsqueeze_355: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        add_444: "f32[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_355);  mul_622 = unsqueeze_355 = None
        convert_element_type_266: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_444, torch.bfloat16);  add_444 = None
        relu_88: "bf16[4, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_266);  convert_element_type_266 = None
        convert_element_type_267: "bf16[128, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_535, torch.bfloat16);  primals_535 = None
        convolution_88: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_88, convert_element_type_267, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_445: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_536, 1)
        convert_element_type_268: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_88, torch.float32)
        var_mean_89 = torch.ops.aten.var_mean.correction(convert_element_type_268, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_268 = None
        getitem_180: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_89[0]
        getitem_181: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_89[1];  var_mean_89 = None
        add_446: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_180, 1e-05)
        rsqrt_89: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_446);  add_446 = None
        sub_89: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_88, getitem_181)
        mul_623: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
        squeeze_267: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
        squeeze_268: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
        mul_624: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
        mul_625: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_537, 0.9)
        add_447: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
        squeeze_269: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
        mul_626: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_269, 1.005128205128205);  squeeze_269 = None
        mul_627: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_626, 0.1);  mul_626 = None
        mul_628: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_538, 0.9)
        add_448: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
        unsqueeze_356: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_539, -1)
        unsqueeze_357: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_629: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_623, unsqueeze_357);  mul_623 = unsqueeze_357 = None
        unsqueeze_358: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_540, -1);  primals_540 = None
        unsqueeze_359: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_449: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_359);  mul_629 = unsqueeze_359 = None
        convert_element_type_269: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_449, torch.bfloat16);  add_449 = None
        relu_89: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_269);  convert_element_type_269 = None
        convert_element_type_270: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_541, torch.bfloat16);  primals_541 = None
        convolution_89: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_89, convert_element_type_270, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_42: "bf16[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_450: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_542, 1)
        convert_element_type_271: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_42, torch.float32)
        var_mean_90 = torch.ops.aten.var_mean.correction(convert_element_type_271, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_271 = None
        getitem_182: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = var_mean_90[0]
        getitem_183: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = var_mean_90[1];  var_mean_90 = None
        add_451: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_182, 1e-05)
        rsqrt_90: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_451);  add_451 = None
        sub_90: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_42, getitem_183)
        mul_630: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
        squeeze_270: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3]);  getitem_183 = None
        squeeze_271: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
        mul_631: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
        mul_632: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_543, 0.9)
        add_452: "f32[544][1]cuda:0" = torch.ops.aten.add.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
        squeeze_272: "f32[544][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
        mul_633: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_272, 1.005128205128205);  squeeze_272 = None
        mul_634: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_633, 0.1);  mul_633 = None
        mul_635: "f32[544][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_544, 0.9)
        add_453: "f32[544][1]cuda:0" = torch.ops.aten.add.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
        unsqueeze_360: "f32[544, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_545, -1)
        unsqueeze_361: "f32[544, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        mul_636: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_361);  mul_630 = unsqueeze_361 = None
        unsqueeze_362: "f32[544, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_546, -1);  primals_546 = None
        unsqueeze_363: "f32[544, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        add_454: "f32[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_636, unsqueeze_363);  mul_636 = unsqueeze_363 = None
        convert_element_type_272: "bf16[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_454, torch.bfloat16);  add_454 = None
        relu_90: "bf16[4, 544, 7, 7][26656, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_272);  convert_element_type_272 = None
        convert_element_type_273: "bf16[128, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_547, torch.bfloat16);  primals_547 = None
        convolution_90: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_90, convert_element_type_273, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_455: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_548, 1)
        convert_element_type_274: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_90, torch.float32)
        var_mean_91 = torch.ops.aten.var_mean.correction(convert_element_type_274, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_274 = None
        getitem_184: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_91[0]
        getitem_185: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_91[1];  var_mean_91 = None
        add_456: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_184, 1e-05)
        rsqrt_91: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_456);  add_456 = None
        sub_91: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_90, getitem_185)
        mul_637: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
        squeeze_273: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3]);  getitem_185 = None
        squeeze_274: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_91, [0, 2, 3]);  rsqrt_91 = None
        mul_638: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1)
        mul_639: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_549, 0.9)
        add_457: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
        squeeze_275: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
        mul_640: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_275, 1.005128205128205);  squeeze_275 = None
        mul_641: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_640, 0.1);  mul_640 = None
        mul_642: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
        add_458: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
        unsqueeze_364: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_551, -1)
        unsqueeze_365: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_643: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_365);  mul_637 = unsqueeze_365 = None
        unsqueeze_366: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_552, -1);  primals_552 = None
        unsqueeze_367: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_459: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_643, unsqueeze_367);  mul_643 = unsqueeze_367 = None
        convert_element_type_275: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_459, torch.bfloat16);  add_459 = None
        relu_91: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_275);  convert_element_type_275 = None
        convert_element_type_276: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_553, torch.bfloat16);  primals_553 = None
        convolution_91: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_91, convert_element_type_276, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_43: "bf16[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_460: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_554, 1)
        convert_element_type_277: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_43, torch.float32)
        var_mean_92 = torch.ops.aten.var_mean.correction(convert_element_type_277, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_277 = None
        getitem_186: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = var_mean_92[0]
        getitem_187: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = var_mean_92[1];  var_mean_92 = None
        add_461: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_186, 1e-05)
        rsqrt_92: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_461);  add_461 = None
        sub_92: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_43, getitem_187)
        mul_644: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
        squeeze_276: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
        squeeze_277: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_92, [0, 2, 3]);  rsqrt_92 = None
        mul_645: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1)
        mul_646: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_555, 0.9)
        add_462: "f32[576][1]cuda:0" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
        squeeze_278: "f32[576][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
        mul_647: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_278, 1.005128205128205);  squeeze_278 = None
        mul_648: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_647, 0.1);  mul_647 = None
        mul_649: "f32[576][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
        add_463: "f32[576][1]cuda:0" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
        unsqueeze_368: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_557, -1)
        unsqueeze_369: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        mul_650: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_369);  mul_644 = unsqueeze_369 = None
        unsqueeze_370: "f32[576, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_558, -1);  primals_558 = None
        unsqueeze_371: "f32[576, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        add_464: "f32[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_371);  mul_650 = unsqueeze_371 = None
        convert_element_type_278: "bf16[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_464, torch.bfloat16);  add_464 = None
        relu_92: "bf16[4, 576, 7, 7][28224, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_278);  convert_element_type_278 = None
        convert_element_type_279: "bf16[128, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_559, torch.bfloat16);  primals_559 = None
        convolution_92: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_92, convert_element_type_279, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_465: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_560, 1)
        convert_element_type_280: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_92, torch.float32)
        var_mean_93 = torch.ops.aten.var_mean.correction(convert_element_type_280, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_280 = None
        getitem_188: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_93[0]
        getitem_189: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_93[1];  var_mean_93 = None
        add_466: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_188, 1e-05)
        rsqrt_93: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_466);  add_466 = None
        sub_93: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_92, getitem_189)
        mul_651: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
        squeeze_279: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
        squeeze_280: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_93, [0, 2, 3]);  rsqrt_93 = None
        mul_652: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1)
        mul_653: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_561, 0.9)
        add_467: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
        squeeze_281: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
        mul_654: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_281, 1.005128205128205);  squeeze_281 = None
        mul_655: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_654, 0.1);  mul_654 = None
        mul_656: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
        add_468: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
        unsqueeze_372: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_563, -1)
        unsqueeze_373: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_657: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_373);  mul_651 = unsqueeze_373 = None
        unsqueeze_374: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_564, -1);  primals_564 = None
        unsqueeze_375: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_469: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_375);  mul_657 = unsqueeze_375 = None
        convert_element_type_281: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_469, torch.bfloat16);  add_469 = None
        relu_93: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_281);  convert_element_type_281 = None
        convert_element_type_282: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_565, torch.bfloat16);  primals_565 = None
        convolution_93: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_93, convert_element_type_282, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_44: "bf16[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_470: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_566, 1)
        convert_element_type_283: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_44, torch.float32)
        var_mean_94 = torch.ops.aten.var_mean.correction(convert_element_type_283, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_283 = None
        getitem_190: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = var_mean_94[0]
        getitem_191: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = var_mean_94[1];  var_mean_94 = None
        add_471: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_190, 1e-05)
        rsqrt_94: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_471);  add_471 = None
        sub_94: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_44, getitem_191)
        mul_658: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_94);  sub_94 = None
        squeeze_282: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3]);  getitem_191 = None
        squeeze_283: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_94, [0, 2, 3]);  rsqrt_94 = None
        mul_659: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_282, 0.1)
        mul_660: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_567, 0.9)
        add_472: "f32[608][1]cuda:0" = torch.ops.aten.add.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
        squeeze_284: "f32[608][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
        mul_661: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_284, 1.005128205128205);  squeeze_284 = None
        mul_662: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_661, 0.1);  mul_661 = None
        mul_663: "f32[608][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_568, 0.9)
        add_473: "f32[608][1]cuda:0" = torch.ops.aten.add.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
        unsqueeze_376: "f32[608, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_569, -1)
        unsqueeze_377: "f32[608, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        mul_664: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_658, unsqueeze_377);  mul_658 = unsqueeze_377 = None
        unsqueeze_378: "f32[608, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_570, -1);  primals_570 = None
        unsqueeze_379: "f32[608, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        add_474: "f32[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_664, unsqueeze_379);  mul_664 = unsqueeze_379 = None
        convert_element_type_284: "bf16[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_474, torch.bfloat16);  add_474 = None
        relu_94: "bf16[4, 608, 7, 7][29792, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_284);  convert_element_type_284 = None
        convert_element_type_285: "bf16[128, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_571, torch.bfloat16);  primals_571 = None
        convolution_94: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_94, convert_element_type_285, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_475: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_572, 1)
        convert_element_type_286: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_94, torch.float32)
        var_mean_95 = torch.ops.aten.var_mean.correction(convert_element_type_286, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_286 = None
        getitem_192: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_95[0]
        getitem_193: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_95[1];  var_mean_95 = None
        add_476: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_192, 1e-05)
        rsqrt_95: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_476);  add_476 = None
        sub_95: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_94, getitem_193)
        mul_665: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_95);  sub_95 = None
        squeeze_285: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3]);  getitem_193 = None
        squeeze_286: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_95, [0, 2, 3]);  rsqrt_95 = None
        mul_666: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_285, 0.1)
        mul_667: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_573, 0.9)
        add_477: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
        squeeze_287: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
        mul_668: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_287, 1.005128205128205);  squeeze_287 = None
        mul_669: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_668, 0.1);  mul_668 = None
        mul_670: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_574, 0.9)
        add_478: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
        unsqueeze_380: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_575, -1)
        unsqueeze_381: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_671: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_665, unsqueeze_381);  mul_665 = unsqueeze_381 = None
        unsqueeze_382: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_576, -1);  primals_576 = None
        unsqueeze_383: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_479: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_671, unsqueeze_383);  mul_671 = unsqueeze_383 = None
        convert_element_type_287: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_479, torch.bfloat16);  add_479 = None
        relu_95: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_287);  convert_element_type_287 = None
        convert_element_type_288: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_577, torch.bfloat16);  primals_577 = None
        convolution_95: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_95, convert_element_type_288, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_45: "bf16[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_480: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_578, 1)
        convert_element_type_289: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_45, torch.float32)
        var_mean_96 = torch.ops.aten.var_mean.correction(convert_element_type_289, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_289 = None
        getitem_194: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_96[0]
        getitem_195: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = var_mean_96[1];  var_mean_96 = None
        add_481: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_194, 1e-05)
        rsqrt_96: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_481);  add_481 = None
        sub_96: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_45, getitem_195)
        mul_672: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_96);  sub_96 = None
        squeeze_288: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3]);  getitem_195 = None
        squeeze_289: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_96, [0, 2, 3]);  rsqrt_96 = None
        mul_673: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_288, 0.1)
        mul_674: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_579, 0.9)
        add_482: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
        squeeze_290: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
        mul_675: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_290, 1.005128205128205);  squeeze_290 = None
        mul_676: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_675, 0.1);  mul_675 = None
        mul_677: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_580, 0.9)
        add_483: "f32[640][1]cuda:0" = torch.ops.aten.add.Tensor(mul_676, mul_677);  mul_676 = mul_677 = None
        unsqueeze_384: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_581, -1)
        unsqueeze_385: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        mul_678: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_672, unsqueeze_385);  mul_672 = unsqueeze_385 = None
        unsqueeze_386: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_582, -1);  primals_582 = None
        unsqueeze_387: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        add_484: "f32[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_678, unsqueeze_387);  mul_678 = unsqueeze_387 = None
        convert_element_type_290: "bf16[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_484, torch.bfloat16);  add_484 = None
        relu_96: "bf16[4, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_290);  convert_element_type_290 = None
        convert_element_type_291: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_583, torch.bfloat16);  primals_583 = None
        convolution_96: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_96, convert_element_type_291, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_485: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_584, 1)
        convert_element_type_292: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_96, torch.float32)
        var_mean_97 = torch.ops.aten.var_mean.correction(convert_element_type_292, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_292 = None
        getitem_196: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_97[0]
        getitem_197: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_97[1];  var_mean_97 = None
        add_486: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_196, 1e-05)
        rsqrt_97: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_486);  add_486 = None
        sub_97: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_96, getitem_197)
        mul_679: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, rsqrt_97);  sub_97 = None
        squeeze_291: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_197, [0, 2, 3]);  getitem_197 = None
        squeeze_292: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_97, [0, 2, 3]);  rsqrt_97 = None
        mul_680: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_291, 0.1)
        mul_681: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_585, 0.9)
        add_487: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_680, mul_681);  mul_680 = mul_681 = None
        squeeze_293: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_196, [0, 2, 3]);  getitem_196 = None
        mul_682: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_293, 1.005128205128205);  squeeze_293 = None
        mul_683: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_682, 0.1);  mul_682 = None
        mul_684: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_586, 0.9)
        add_488: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
        unsqueeze_388: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_587, -1)
        unsqueeze_389: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_685: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_679, unsqueeze_389);  mul_679 = unsqueeze_389 = None
        unsqueeze_390: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_588, -1);  primals_588 = None
        unsqueeze_391: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_489: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_685, unsqueeze_391);  mul_685 = unsqueeze_391 = None
        convert_element_type_293: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_489, torch.bfloat16);  add_489 = None
        relu_97: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_293);  convert_element_type_293 = None
        convert_element_type_294: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_589, torch.bfloat16);  primals_589 = None
        convolution_97: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_97, convert_element_type_294, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_46: "bf16[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_490: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_590, 1)
        convert_element_type_295: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_46, torch.float32)
        var_mean_98 = torch.ops.aten.var_mean.correction(convert_element_type_295, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_295 = None
        getitem_198: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_98[0]
        getitem_199: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = var_mean_98[1];  var_mean_98 = None
        add_491: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_198, 1e-05)
        rsqrt_98: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_491);  add_491 = None
        sub_98: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_46, getitem_199)
        mul_686: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_98);  sub_98 = None
        squeeze_294: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_199, [0, 2, 3]);  getitem_199 = None
        squeeze_295: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_98, [0, 2, 3]);  rsqrt_98 = None
        mul_687: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_294, 0.1)
        mul_688: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_591, 0.9)
        add_492: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_687, mul_688);  mul_687 = mul_688 = None
        squeeze_296: "f32[672][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_198, [0, 2, 3]);  getitem_198 = None
        mul_689: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_296, 1.005128205128205);  squeeze_296 = None
        mul_690: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_689, 0.1);  mul_689 = None
        mul_691: "f32[672][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_592, 0.9)
        add_493: "f32[672][1]cuda:0" = torch.ops.aten.add.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
        unsqueeze_392: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_593, -1)
        unsqueeze_393: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        mul_692: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_686, unsqueeze_393);  mul_686 = unsqueeze_393 = None
        unsqueeze_394: "f32[672, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_594, -1);  primals_594 = None
        unsqueeze_395: "f32[672, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        add_494: "f32[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_692, unsqueeze_395);  mul_692 = unsqueeze_395 = None
        convert_element_type_296: "bf16[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_494, torch.bfloat16);  add_494 = None
        relu_98: "bf16[4, 672, 7, 7][32928, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_296);  convert_element_type_296 = None
        convert_element_type_297: "bf16[128, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_595, torch.bfloat16);  primals_595 = None
        convolution_98: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_98, convert_element_type_297, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_495: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_596, 1)
        convert_element_type_298: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_98, torch.float32)
        var_mean_99 = torch.ops.aten.var_mean.correction(convert_element_type_298, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_298 = None
        getitem_200: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_99[0]
        getitem_201: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_99[1];  var_mean_99 = None
        add_496: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_200, 1e-05)
        rsqrt_99: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_496);  add_496 = None
        sub_99: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_98, getitem_201)
        mul_693: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_99);  sub_99 = None
        squeeze_297: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_201, [0, 2, 3]);  getitem_201 = None
        squeeze_298: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_99, [0, 2, 3]);  rsqrt_99 = None
        mul_694: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_297, 0.1)
        mul_695: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_597, 0.9)
        add_497: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_694, mul_695);  mul_694 = mul_695 = None
        squeeze_299: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_200, [0, 2, 3]);  getitem_200 = None
        mul_696: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_299, 1.005128205128205);  squeeze_299 = None
        mul_697: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_696, 0.1);  mul_696 = None
        mul_698: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_598, 0.9)
        add_498: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_697, mul_698);  mul_697 = mul_698 = None
        unsqueeze_396: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_599, -1)
        unsqueeze_397: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_699: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_693, unsqueeze_397);  mul_693 = unsqueeze_397 = None
        unsqueeze_398: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_600, -1);  primals_600 = None
        unsqueeze_399: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_499: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_699, unsqueeze_399);  mul_699 = unsqueeze_399 = None
        convert_element_type_299: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_499, torch.bfloat16);  add_499 = None
        relu_99: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_299);  convert_element_type_299 = None
        convert_element_type_300: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_601, torch.bfloat16);  primals_601 = None
        convolution_99: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_99, convert_element_type_300, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_47: "bf16[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_500: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_602, 1)
        convert_element_type_301: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_47, torch.float32)
        var_mean_100 = torch.ops.aten.var_mean.correction(convert_element_type_301, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_301 = None
        getitem_202: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = var_mean_100[0]
        getitem_203: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = var_mean_100[1];  var_mean_100 = None
        add_501: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_202, 1e-05)
        rsqrt_100: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_501);  add_501 = None
        sub_100: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_47, getitem_203)
        mul_700: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_100);  sub_100 = None
        squeeze_300: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_203, [0, 2, 3]);  getitem_203 = None
        squeeze_301: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_100, [0, 2, 3]);  rsqrt_100 = None
        mul_701: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_300, 0.1)
        mul_702: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_603, 0.9)
        add_502: "f32[704][1]cuda:0" = torch.ops.aten.add.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
        squeeze_302: "f32[704][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_202, [0, 2, 3]);  getitem_202 = None
        mul_703: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_302, 1.005128205128205);  squeeze_302 = None
        mul_704: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_703, 0.1);  mul_703 = None
        mul_705: "f32[704][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_604, 0.9)
        add_503: "f32[704][1]cuda:0" = torch.ops.aten.add.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
        unsqueeze_400: "f32[704, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_605, -1)
        unsqueeze_401: "f32[704, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        mul_706: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_700, unsqueeze_401);  mul_700 = unsqueeze_401 = None
        unsqueeze_402: "f32[704, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_606, -1);  primals_606 = None
        unsqueeze_403: "f32[704, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        add_504: "f32[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_706, unsqueeze_403);  mul_706 = unsqueeze_403 = None
        convert_element_type_302: "bf16[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_504, torch.bfloat16);  add_504 = None
        relu_100: "bf16[4, 704, 7, 7][34496, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_302);  convert_element_type_302 = None
        convert_element_type_303: "bf16[128, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_607, torch.bfloat16);  primals_607 = None
        convolution_100: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_100, convert_element_type_303, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_505: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_608, 1)
        convert_element_type_304: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_100, torch.float32)
        var_mean_101 = torch.ops.aten.var_mean.correction(convert_element_type_304, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_304 = None
        getitem_204: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_101[0]
        getitem_205: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_101[1];  var_mean_101 = None
        add_506: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_204, 1e-05)
        rsqrt_101: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_506);  add_506 = None
        sub_101: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_100, getitem_205)
        mul_707: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_101, rsqrt_101);  sub_101 = None
        squeeze_303: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_205, [0, 2, 3]);  getitem_205 = None
        squeeze_304: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_101, [0, 2, 3]);  rsqrt_101 = None
        mul_708: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_303, 0.1)
        mul_709: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_609, 0.9)
        add_507: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
        squeeze_305: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_204, [0, 2, 3]);  getitem_204 = None
        mul_710: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_305, 1.005128205128205);  squeeze_305 = None
        mul_711: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_710, 0.1);  mul_710 = None
        mul_712: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_610, 0.9)
        add_508: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_711, mul_712);  mul_711 = mul_712 = None
        unsqueeze_404: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_611, -1)
        unsqueeze_405: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_713: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_707, unsqueeze_405);  mul_707 = unsqueeze_405 = None
        unsqueeze_406: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_612, -1);  primals_612 = None
        unsqueeze_407: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_509: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_713, unsqueeze_407);  mul_713 = unsqueeze_407 = None
        convert_element_type_305: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_509, torch.bfloat16);  add_509 = None
        relu_101: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_305);  convert_element_type_305 = None
        convert_element_type_306: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_613, torch.bfloat16);  primals_613 = None
        convolution_101: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_101, convert_element_type_306, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_48: "bf16[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_510: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_614, 1)
        convert_element_type_307: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_48, torch.float32)
        var_mean_102 = torch.ops.aten.var_mean.correction(convert_element_type_307, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_307 = None
        getitem_206: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = var_mean_102[0]
        getitem_207: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = var_mean_102[1];  var_mean_102 = None
        add_511: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_206, 1e-05)
        rsqrt_102: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_511);  add_511 = None
        sub_102: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_48, getitem_207)
        mul_714: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_102);  sub_102 = None
        squeeze_306: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_207, [0, 2, 3]);  getitem_207 = None
        squeeze_307: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_102, [0, 2, 3]);  rsqrt_102 = None
        mul_715: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_306, 0.1)
        mul_716: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_615, 0.9)
        add_512: "f32[736][1]cuda:0" = torch.ops.aten.add.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
        squeeze_308: "f32[736][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_206, [0, 2, 3]);  getitem_206 = None
        mul_717: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_308, 1.005128205128205);  squeeze_308 = None
        mul_718: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_717, 0.1);  mul_717 = None
        mul_719: "f32[736][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_616, 0.9)
        add_513: "f32[736][1]cuda:0" = torch.ops.aten.add.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
        unsqueeze_408: "f32[736, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_617, -1)
        unsqueeze_409: "f32[736, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        mul_720: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_714, unsqueeze_409);  mul_714 = unsqueeze_409 = None
        unsqueeze_410: "f32[736, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_618, -1);  primals_618 = None
        unsqueeze_411: "f32[736, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        add_514: "f32[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_720, unsqueeze_411);  mul_720 = unsqueeze_411 = None
        convert_element_type_308: "bf16[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_514, torch.bfloat16);  add_514 = None
        relu_102: "bf16[4, 736, 7, 7][36064, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_308);  convert_element_type_308 = None
        convert_element_type_309: "bf16[128, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_619, torch.bfloat16);  primals_619 = None
        convolution_102: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_102, convert_element_type_309, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_515: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_620, 1)
        convert_element_type_310: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_102, torch.float32)
        var_mean_103 = torch.ops.aten.var_mean.correction(convert_element_type_310, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_310 = None
        getitem_208: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_103[0]
        getitem_209: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_103[1];  var_mean_103 = None
        add_516: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_208, 1e-05)
        rsqrt_103: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_516);  add_516 = None
        sub_103: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_102, getitem_209)
        mul_721: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_103);  sub_103 = None
        squeeze_309: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_209, [0, 2, 3]);  getitem_209 = None
        squeeze_310: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_103, [0, 2, 3]);  rsqrt_103 = None
        mul_722: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_309, 0.1)
        mul_723: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_621, 0.9)
        add_517: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
        squeeze_311: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_208, [0, 2, 3]);  getitem_208 = None
        mul_724: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_311, 1.005128205128205);  squeeze_311 = None
        mul_725: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_724, 0.1);  mul_724 = None
        mul_726: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_622, 0.9)
        add_518: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
        unsqueeze_412: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_623, -1)
        unsqueeze_413: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_727: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_721, unsqueeze_413);  mul_721 = unsqueeze_413 = None
        unsqueeze_414: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_624, -1);  primals_624 = None
        unsqueeze_415: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_519: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_727, unsqueeze_415);  mul_727 = unsqueeze_415 = None
        convert_element_type_311: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_519, torch.bfloat16);  add_519 = None
        relu_103: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_311);  convert_element_type_311 = None
        convert_element_type_312: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_625, torch.bfloat16);  primals_625 = None
        convolution_103: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_103, convert_element_type_312, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_49: "bf16[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_520: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_626, 1)
        convert_element_type_313: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_49, torch.float32)
        var_mean_104 = torch.ops.aten.var_mean.correction(convert_element_type_313, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_313 = None
        getitem_210: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_104[0]
        getitem_211: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = var_mean_104[1];  var_mean_104 = None
        add_521: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_210, 1e-05)
        rsqrt_104: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_521);  add_521 = None
        sub_104: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_49, getitem_211)
        mul_728: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_104, rsqrt_104);  sub_104 = None
        squeeze_312: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_211, [0, 2, 3]);  getitem_211 = None
        squeeze_313: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_104, [0, 2, 3]);  rsqrt_104 = None
        mul_729: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_312, 0.1)
        mul_730: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_627, 0.9)
        add_522: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_729, mul_730);  mul_729 = mul_730 = None
        squeeze_314: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_210, [0, 2, 3]);  getitem_210 = None
        mul_731: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_314, 1.005128205128205);  squeeze_314 = None
        mul_732: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_731, 0.1);  mul_731 = None
        mul_733: "f32[768][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_628, 0.9)
        add_523: "f32[768][1]cuda:0" = torch.ops.aten.add.Tensor(mul_732, mul_733);  mul_732 = mul_733 = None
        unsqueeze_416: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_629, -1)
        unsqueeze_417: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        mul_734: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_728, unsqueeze_417);  mul_728 = unsqueeze_417 = None
        unsqueeze_418: "f32[768, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_630, -1);  primals_630 = None
        unsqueeze_419: "f32[768, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        add_524: "f32[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_734, unsqueeze_419);  mul_734 = unsqueeze_419 = None
        convert_element_type_314: "bf16[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_524, torch.bfloat16);  add_524 = None
        relu_104: "bf16[4, 768, 7, 7][37632, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_314);  convert_element_type_314 = None
        convert_element_type_315: "bf16[128, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_631, torch.bfloat16);  primals_631 = None
        convolution_104: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_104, convert_element_type_315, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_525: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_632, 1)
        convert_element_type_316: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_104, torch.float32)
        var_mean_105 = torch.ops.aten.var_mean.correction(convert_element_type_316, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_316 = None
        getitem_212: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_105[0]
        getitem_213: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_105[1];  var_mean_105 = None
        add_526: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_212, 1e-05)
        rsqrt_105: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_526);  add_526 = None
        sub_105: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_104, getitem_213)
        mul_735: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, rsqrt_105);  sub_105 = None
        squeeze_315: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_213, [0, 2, 3]);  getitem_213 = None
        squeeze_316: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_105, [0, 2, 3]);  rsqrt_105 = None
        mul_736: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_315, 0.1)
        mul_737: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_633, 0.9)
        add_527: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
        squeeze_317: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_212, [0, 2, 3]);  getitem_212 = None
        mul_738: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_317, 1.005128205128205);  squeeze_317 = None
        mul_739: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_738, 0.1);  mul_738 = None
        mul_740: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_634, 0.9)
        add_528: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
        unsqueeze_420: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_635, -1)
        unsqueeze_421: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_741: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_735, unsqueeze_421);  mul_735 = unsqueeze_421 = None
        unsqueeze_422: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_636, -1);  primals_636 = None
        unsqueeze_423: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_529: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_741, unsqueeze_423);  mul_741 = unsqueeze_423 = None
        convert_element_type_317: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_529, torch.bfloat16);  add_529 = None
        relu_105: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_317);  convert_element_type_317 = None
        convert_element_type_318: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_637, torch.bfloat16);  primals_637 = None
        convolution_105: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_105, convert_element_type_318, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_50: "bf16[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_530: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_638, 1)
        convert_element_type_319: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_50, torch.float32)
        var_mean_106 = torch.ops.aten.var_mean.correction(convert_element_type_319, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_319 = None
        getitem_214: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = var_mean_106[0]
        getitem_215: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = var_mean_106[1];  var_mean_106 = None
        add_531: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_214, 1e-05)
        rsqrt_106: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_531);  add_531 = None
        sub_106: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_50, getitem_215)
        mul_742: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_106);  sub_106 = None
        squeeze_318: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_215, [0, 2, 3]);  getitem_215 = None
        squeeze_319: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_106, [0, 2, 3]);  rsqrt_106 = None
        mul_743: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_318, 0.1)
        mul_744: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_639, 0.9)
        add_532: "f32[800][1]cuda:0" = torch.ops.aten.add.Tensor(mul_743, mul_744);  mul_743 = mul_744 = None
        squeeze_320: "f32[800][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_214, [0, 2, 3]);  getitem_214 = None
        mul_745: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_320, 1.005128205128205);  squeeze_320 = None
        mul_746: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_745, 0.1);  mul_745 = None
        mul_747: "f32[800][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_640, 0.9)
        add_533: "f32[800][1]cuda:0" = torch.ops.aten.add.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
        unsqueeze_424: "f32[800, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_641, -1)
        unsqueeze_425: "f32[800, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        mul_748: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_742, unsqueeze_425);  mul_742 = unsqueeze_425 = None
        unsqueeze_426: "f32[800, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_642, -1);  primals_642 = None
        unsqueeze_427: "f32[800, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        add_534: "f32[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_748, unsqueeze_427);  mul_748 = unsqueeze_427 = None
        convert_element_type_320: "bf16[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_534, torch.bfloat16);  add_534 = None
        relu_106: "bf16[4, 800, 7, 7][39200, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_320);  convert_element_type_320 = None
        convert_element_type_321: "bf16[128, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_643, torch.bfloat16);  primals_643 = None
        convolution_106: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_106, convert_element_type_321, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_535: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_644, 1)
        convert_element_type_322: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_106, torch.float32)
        var_mean_107 = torch.ops.aten.var_mean.correction(convert_element_type_322, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_322 = None
        getitem_216: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_107[0]
        getitem_217: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_107[1];  var_mean_107 = None
        add_536: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_216, 1e-05)
        rsqrt_107: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_536);  add_536 = None
        sub_107: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_106, getitem_217)
        mul_749: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_107);  sub_107 = None
        squeeze_321: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_217, [0, 2, 3]);  getitem_217 = None
        squeeze_322: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_107, [0, 2, 3]);  rsqrt_107 = None
        mul_750: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_321, 0.1)
        mul_751: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_645, 0.9)
        add_537: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_750, mul_751);  mul_750 = mul_751 = None
        squeeze_323: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_216, [0, 2, 3]);  getitem_216 = None
        mul_752: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_323, 1.005128205128205);  squeeze_323 = None
        mul_753: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_752, 0.1);  mul_752 = None
        mul_754: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_646, 0.9)
        add_538: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
        unsqueeze_428: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_647, -1)
        unsqueeze_429: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_755: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_749, unsqueeze_429);  mul_749 = unsqueeze_429 = None
        unsqueeze_430: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_648, -1);  primals_648 = None
        unsqueeze_431: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_539: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_755, unsqueeze_431);  mul_755 = unsqueeze_431 = None
        convert_element_type_323: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_539, torch.bfloat16);  add_539 = None
        relu_107: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_323);  convert_element_type_323 = None
        convert_element_type_324: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_649, torch.bfloat16);  primals_649 = None
        convolution_107: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_107, convert_element_type_324, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_51: "bf16[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_540: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_650, 1)
        convert_element_type_325: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_51, torch.float32)
        var_mean_108 = torch.ops.aten.var_mean.correction(convert_element_type_325, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_325 = None
        getitem_218: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = var_mean_108[0]
        getitem_219: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = var_mean_108[1];  var_mean_108 = None
        add_541: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_218, 1e-05)
        rsqrt_108: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_541);  add_541 = None
        sub_108: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_51, getitem_219)
        mul_756: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_108, rsqrt_108);  sub_108 = None
        squeeze_324: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_219, [0, 2, 3]);  getitem_219 = None
        squeeze_325: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_108, [0, 2, 3]);  rsqrt_108 = None
        mul_757: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_324, 0.1)
        mul_758: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_651, 0.9)
        add_542: "f32[832][1]cuda:0" = torch.ops.aten.add.Tensor(mul_757, mul_758);  mul_757 = mul_758 = None
        squeeze_326: "f32[832][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_218, [0, 2, 3]);  getitem_218 = None
        mul_759: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_326, 1.005128205128205);  squeeze_326 = None
        mul_760: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, 0.1);  mul_759 = None
        mul_761: "f32[832][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_652, 0.9)
        add_543: "f32[832][1]cuda:0" = torch.ops.aten.add.Tensor(mul_760, mul_761);  mul_760 = mul_761 = None
        unsqueeze_432: "f32[832, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_653, -1)
        unsqueeze_433: "f32[832, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        mul_762: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_756, unsqueeze_433);  mul_756 = unsqueeze_433 = None
        unsqueeze_434: "f32[832, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_654, -1);  primals_654 = None
        unsqueeze_435: "f32[832, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        add_544: "f32[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_762, unsqueeze_435);  mul_762 = unsqueeze_435 = None
        convert_element_type_326: "bf16[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_544, torch.bfloat16);  add_544 = None
        relu_108: "bf16[4, 832, 7, 7][40768, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_326);  convert_element_type_326 = None
        convert_element_type_327: "bf16[128, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_655, torch.bfloat16);  primals_655 = None
        convolution_108: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_108, convert_element_type_327, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_545: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_656, 1)
        convert_element_type_328: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_108, torch.float32)
        var_mean_109 = torch.ops.aten.var_mean.correction(convert_element_type_328, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_328 = None
        getitem_220: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_109[0]
        getitem_221: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_109[1];  var_mean_109 = None
        add_546: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_220, 1e-05)
        rsqrt_109: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_546);  add_546 = None
        sub_109: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_108, getitem_221)
        mul_763: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_109);  sub_109 = None
        squeeze_327: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_221, [0, 2, 3]);  getitem_221 = None
        squeeze_328: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_109, [0, 2, 3]);  rsqrt_109 = None
        mul_764: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_327, 0.1)
        mul_765: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_657, 0.9)
        add_547: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
        squeeze_329: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_220, [0, 2, 3]);  getitem_220 = None
        mul_766: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_329, 1.005128205128205);  squeeze_329 = None
        mul_767: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_766, 0.1);  mul_766 = None
        mul_768: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_658, 0.9)
        add_548: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_767, mul_768);  mul_767 = mul_768 = None
        unsqueeze_436: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_659, -1)
        unsqueeze_437: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_769: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_763, unsqueeze_437);  mul_763 = unsqueeze_437 = None
        unsqueeze_438: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_660, -1);  primals_660 = None
        unsqueeze_439: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_549: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_769, unsqueeze_439);  mul_769 = unsqueeze_439 = None
        convert_element_type_329: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_549, torch.bfloat16);  add_549 = None
        relu_109: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_329);  convert_element_type_329 = None
        convert_element_type_330: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_661, torch.bfloat16);  primals_661 = None
        convolution_109: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_109, convert_element_type_330, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_52: "bf16[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_550: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_662, 1)
        convert_element_type_331: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_52, torch.float32)
        var_mean_110 = torch.ops.aten.var_mean.correction(convert_element_type_331, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_331 = None
        getitem_222: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = var_mean_110[0]
        getitem_223: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = var_mean_110[1];  var_mean_110 = None
        add_551: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_222, 1e-05)
        rsqrt_110: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_551);  add_551 = None
        sub_110: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_52, getitem_223)
        mul_770: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_110);  sub_110 = None
        squeeze_330: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_223, [0, 2, 3]);  getitem_223 = None
        squeeze_331: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_110, [0, 2, 3]);  rsqrt_110 = None
        mul_771: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_330, 0.1)
        mul_772: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_663, 0.9)
        add_552: "f32[864][1]cuda:0" = torch.ops.aten.add.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
        squeeze_332: "f32[864][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_222, [0, 2, 3]);  getitem_222 = None
        mul_773: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_332, 1.005128205128205);  squeeze_332 = None
        mul_774: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_773, 0.1);  mul_773 = None
        mul_775: "f32[864][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_664, 0.9)
        add_553: "f32[864][1]cuda:0" = torch.ops.aten.add.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
        unsqueeze_440: "f32[864, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_665, -1)
        unsqueeze_441: "f32[864, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        mul_776: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_770, unsqueeze_441);  mul_770 = unsqueeze_441 = None
        unsqueeze_442: "f32[864, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_666, -1);  primals_666 = None
        unsqueeze_443: "f32[864, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        add_554: "f32[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_776, unsqueeze_443);  mul_776 = unsqueeze_443 = None
        convert_element_type_332: "bf16[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_554, torch.bfloat16);  add_554 = None
        relu_110: "bf16[4, 864, 7, 7][42336, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_332);  convert_element_type_332 = None
        convert_element_type_333: "bf16[128, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_667, torch.bfloat16);  primals_667 = None
        convolution_110: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_110, convert_element_type_333, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_555: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_668, 1)
        convert_element_type_334: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_110, torch.float32)
        var_mean_111 = torch.ops.aten.var_mean.correction(convert_element_type_334, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_334 = None
        getitem_224: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_111[0]
        getitem_225: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_111[1];  var_mean_111 = None
        add_556: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_224, 1e-05)
        rsqrt_111: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_556);  add_556 = None
        sub_111: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_110, getitem_225)
        mul_777: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_111, rsqrt_111);  sub_111 = None
        squeeze_333: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_225, [0, 2, 3]);  getitem_225 = None
        squeeze_334: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_111, [0, 2, 3]);  rsqrt_111 = None
        mul_778: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_333, 0.1)
        mul_779: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_669, 0.9)
        add_557: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_778, mul_779);  mul_778 = mul_779 = None
        squeeze_335: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_224, [0, 2, 3]);  getitem_224 = None
        mul_780: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_335, 1.005128205128205);  squeeze_335 = None
        mul_781: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_780, 0.1);  mul_780 = None
        mul_782: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_670, 0.9)
        add_558: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
        unsqueeze_444: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_671, -1)
        unsqueeze_445: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_783: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_777, unsqueeze_445);  mul_777 = unsqueeze_445 = None
        unsqueeze_446: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_672, -1);  primals_672 = None
        unsqueeze_447: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_559: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_783, unsqueeze_447);  mul_783 = unsqueeze_447 = None
        convert_element_type_335: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_559, torch.bfloat16);  add_559 = None
        relu_111: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_335);  convert_element_type_335 = None
        convert_element_type_336: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_673, torch.bfloat16);  primals_673 = None
        convolution_111: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_111, convert_element_type_336, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_53: "bf16[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_560: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_674, 1)
        convert_element_type_337: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_53, torch.float32)
        var_mean_112 = torch.ops.aten.var_mean.correction(convert_element_type_337, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_337 = None
        getitem_226: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = var_mean_112[0]
        getitem_227: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = var_mean_112[1];  var_mean_112 = None
        add_561: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_226, 1e-05)
        rsqrt_112: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_561);  add_561 = None
        sub_112: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_53, getitem_227)
        mul_784: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, rsqrt_112);  sub_112 = None
        squeeze_336: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_227, [0, 2, 3]);  getitem_227 = None
        squeeze_337: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_112, [0, 2, 3]);  rsqrt_112 = None
        mul_785: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_336, 0.1)
        mul_786: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_675, 0.9)
        add_562: "f32[896][1]cuda:0" = torch.ops.aten.add.Tensor(mul_785, mul_786);  mul_785 = mul_786 = None
        squeeze_338: "f32[896][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_226, [0, 2, 3]);  getitem_226 = None
        mul_787: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_338, 1.005128205128205);  squeeze_338 = None
        mul_788: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_787, 0.1);  mul_787 = None
        mul_789: "f32[896][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_676, 0.9)
        add_563: "f32[896][1]cuda:0" = torch.ops.aten.add.Tensor(mul_788, mul_789);  mul_788 = mul_789 = None
        unsqueeze_448: "f32[896, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_677, -1)
        unsqueeze_449: "f32[896, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        mul_790: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_784, unsqueeze_449);  mul_784 = unsqueeze_449 = None
        unsqueeze_450: "f32[896, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_678, -1);  primals_678 = None
        unsqueeze_451: "f32[896, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        add_564: "f32[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_790, unsqueeze_451);  mul_790 = unsqueeze_451 = None
        convert_element_type_338: "bf16[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_564, torch.bfloat16);  add_564 = None
        relu_112: "bf16[4, 896, 7, 7][43904, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_338);  convert_element_type_338 = None
        convert_element_type_339: "bf16[128, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_679, torch.bfloat16);  primals_679 = None
        convolution_112: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_112, convert_element_type_339, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_565: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_680, 1)
        convert_element_type_340: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_112, torch.float32)
        var_mean_113 = torch.ops.aten.var_mean.correction(convert_element_type_340, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_340 = None
        getitem_228: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_113[0]
        getitem_229: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_113[1];  var_mean_113 = None
        add_566: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_228, 1e-05)
        rsqrt_113: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_566);  add_566 = None
        sub_113: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_112, getitem_229)
        mul_791: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_113);  sub_113 = None
        squeeze_339: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_229, [0, 2, 3]);  getitem_229 = None
        squeeze_340: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_113, [0, 2, 3]);  rsqrt_113 = None
        mul_792: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_339, 0.1)
        mul_793: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_681, 0.9)
        add_567: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
        squeeze_341: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_228, [0, 2, 3]);  getitem_228 = None
        mul_794: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_341, 1.005128205128205);  squeeze_341 = None
        mul_795: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_794, 0.1);  mul_794 = None
        mul_796: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_682, 0.9)
        add_568: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
        unsqueeze_452: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_683, -1)
        unsqueeze_453: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_797: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_791, unsqueeze_453);  mul_791 = unsqueeze_453 = None
        unsqueeze_454: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_684, -1);  primals_684 = None
        unsqueeze_455: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_569: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_797, unsqueeze_455);  mul_797 = unsqueeze_455 = None
        convert_element_type_341: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_569, torch.bfloat16);  add_569 = None
        relu_113: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_341);  convert_element_type_341 = None
        convert_element_type_342: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_685, torch.bfloat16);  primals_685 = None
        convolution_113: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_113, convert_element_type_342, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_54: "bf16[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_570: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_686, 1)
        convert_element_type_343: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_54, torch.float32)
        var_mean_114 = torch.ops.aten.var_mean.correction(convert_element_type_343, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_343 = None
        getitem_230: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = var_mean_114[0]
        getitem_231: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = var_mean_114[1];  var_mean_114 = None
        add_571: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_230, 1e-05)
        rsqrt_114: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_571);  add_571 = None
        sub_114: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_54, getitem_231)
        mul_798: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_114);  sub_114 = None
        squeeze_342: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_231, [0, 2, 3]);  getitem_231 = None
        squeeze_343: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_114, [0, 2, 3]);  rsqrt_114 = None
        mul_799: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_342, 0.1)
        mul_800: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_687, 0.9)
        add_572: "f32[928][1]cuda:0" = torch.ops.aten.add.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
        squeeze_344: "f32[928][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_230, [0, 2, 3]);  getitem_230 = None
        mul_801: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_344, 1.005128205128205);  squeeze_344 = None
        mul_802: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_801, 0.1);  mul_801 = None
        mul_803: "f32[928][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_688, 0.9)
        add_573: "f32[928][1]cuda:0" = torch.ops.aten.add.Tensor(mul_802, mul_803);  mul_802 = mul_803 = None
        unsqueeze_456: "f32[928, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_689, -1)
        unsqueeze_457: "f32[928, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        mul_804: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_798, unsqueeze_457);  mul_798 = unsqueeze_457 = None
        unsqueeze_458: "f32[928, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_690, -1);  primals_690 = None
        unsqueeze_459: "f32[928, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        add_574: "f32[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_804, unsqueeze_459);  mul_804 = unsqueeze_459 = None
        convert_element_type_344: "bf16[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_574, torch.bfloat16);  add_574 = None
        relu_114: "bf16[4, 928, 7, 7][45472, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_344);  convert_element_type_344 = None
        convert_element_type_345: "bf16[128, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_691, torch.bfloat16);  primals_691 = None
        convolution_114: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_114, convert_element_type_345, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_575: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_692, 1)
        convert_element_type_346: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_114, torch.float32)
        var_mean_115 = torch.ops.aten.var_mean.correction(convert_element_type_346, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_346 = None
        getitem_232: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_115[0]
        getitem_233: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_115[1];  var_mean_115 = None
        add_576: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_232, 1e-05)
        rsqrt_115: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_576);  add_576 = None
        sub_115: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_114, getitem_233)
        mul_805: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, rsqrt_115);  sub_115 = None
        squeeze_345: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_233, [0, 2, 3]);  getitem_233 = None
        squeeze_346: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_115, [0, 2, 3]);  rsqrt_115 = None
        mul_806: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_345, 0.1)
        mul_807: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_693, 0.9)
        add_577: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
        squeeze_347: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_232, [0, 2, 3]);  getitem_232 = None
        mul_808: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_347, 1.005128205128205);  squeeze_347 = None
        mul_809: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_808, 0.1);  mul_808 = None
        mul_810: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_694, 0.9)
        add_578: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
        unsqueeze_460: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_695, -1)
        unsqueeze_461: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_811: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_805, unsqueeze_461);  mul_805 = unsqueeze_461 = None
        unsqueeze_462: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_696, -1);  primals_696 = None
        unsqueeze_463: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_579: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_811, unsqueeze_463);  mul_811 = unsqueeze_463 = None
        convert_element_type_347: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_579, torch.bfloat16);  add_579 = None
        relu_115: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_347);  convert_element_type_347 = None
        convert_element_type_348: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_697, torch.bfloat16);  primals_697 = None
        convolution_115: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_115, convert_element_type_348, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_55: "bf16[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113, convolution_115], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_580: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_698, 1)
        convert_element_type_349: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_55, torch.float32)
        var_mean_116 = torch.ops.aten.var_mean.correction(convert_element_type_349, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_349 = None
        getitem_234: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_116[0]
        getitem_235: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = var_mean_116[1];  var_mean_116 = None
        add_581: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_234, 1e-05)
        rsqrt_116: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_581);  add_581 = None
        sub_116: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_55, getitem_235)
        mul_812: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_116);  sub_116 = None
        squeeze_348: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_235, [0, 2, 3]);  getitem_235 = None
        squeeze_349: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_116, [0, 2, 3]);  rsqrt_116 = None
        mul_813: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_348, 0.1)
        mul_814: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_699, 0.9)
        add_582: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
        squeeze_350: "f32[960][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_234, [0, 2, 3]);  getitem_234 = None
        mul_815: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_350, 1.005128205128205);  squeeze_350 = None
        mul_816: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_815, 0.1);  mul_815 = None
        mul_817: "f32[960][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_700, 0.9)
        add_583: "f32[960][1]cuda:0" = torch.ops.aten.add.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
        unsqueeze_464: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_701, -1)
        unsqueeze_465: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        mul_818: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_812, unsqueeze_465);  mul_812 = unsqueeze_465 = None
        unsqueeze_466: "f32[960, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_702, -1);  primals_702 = None
        unsqueeze_467: "f32[960, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        add_584: "f32[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_818, unsqueeze_467);  mul_818 = unsqueeze_467 = None
        convert_element_type_350: "bf16[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_584, torch.bfloat16);  add_584 = None
        relu_116: "bf16[4, 960, 7, 7][47040, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_350);  convert_element_type_350 = None
        convert_element_type_351: "bf16[128, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_703, torch.bfloat16);  primals_703 = None
        convolution_116: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_116, convert_element_type_351, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_585: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_704, 1)
        convert_element_type_352: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_116, torch.float32)
        var_mean_117 = torch.ops.aten.var_mean.correction(convert_element_type_352, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_352 = None
        getitem_236: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_117[0]
        getitem_237: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_117[1];  var_mean_117 = None
        add_586: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_236, 1e-05)
        rsqrt_117: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_586);  add_586 = None
        sub_117: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_116, getitem_237)
        mul_819: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_117);  sub_117 = None
        squeeze_351: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_237, [0, 2, 3]);  getitem_237 = None
        squeeze_352: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_117, [0, 2, 3]);  rsqrt_117 = None
        mul_820: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_351, 0.1)
        mul_821: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_705, 0.9)
        add_587: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_820, mul_821);  mul_820 = mul_821 = None
        squeeze_353: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_236, [0, 2, 3]);  getitem_236 = None
        mul_822: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_353, 1.005128205128205);  squeeze_353 = None
        mul_823: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_822, 0.1);  mul_822 = None
        mul_824: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_706, 0.9)
        add_588: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_823, mul_824);  mul_823 = mul_824 = None
        unsqueeze_468: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_707, -1)
        unsqueeze_469: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_825: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_819, unsqueeze_469);  mul_819 = unsqueeze_469 = None
        unsqueeze_470: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_708, -1);  primals_708 = None
        unsqueeze_471: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_589: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_825, unsqueeze_471);  mul_825 = unsqueeze_471 = None
        convert_element_type_353: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_589, torch.bfloat16);  add_589 = None
        relu_117: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_353);  convert_element_type_353 = None
        convert_element_type_354: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_709, torch.bfloat16);  primals_709 = None
        convolution_117: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_117, convert_element_type_354, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:48 in bn_function, code: concated_features = torch.cat(inputs, 1)
        cat_56: "bf16[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113, convolution_115, convolution_117], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        add_590: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_710, 1)
        convert_element_type_355: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_56, torch.float32)
        var_mean_118 = torch.ops.aten.var_mean.correction(convert_element_type_355, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_355 = None
        getitem_238: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = var_mean_118[0]
        getitem_239: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = var_mean_118[1];  var_mean_118 = None
        add_591: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_238, 1e-05)
        rsqrt_118: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_591);  add_591 = None
        sub_118: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_56, getitem_239)
        mul_826: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_118, rsqrt_118);  sub_118 = None
        squeeze_354: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_239, [0, 2, 3]);  getitem_239 = None
        squeeze_355: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_118, [0, 2, 3]);  rsqrt_118 = None
        mul_827: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_354, 0.1)
        mul_828: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_711, 0.9)
        add_592: "f32[992][1]cuda:0" = torch.ops.aten.add.Tensor(mul_827, mul_828);  mul_827 = mul_828 = None
        squeeze_356: "f32[992][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_238, [0, 2, 3]);  getitem_238 = None
        mul_829: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_356, 1.005128205128205);  squeeze_356 = None
        mul_830: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_829, 0.1);  mul_829 = None
        mul_831: "f32[992][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_712, 0.9)
        add_593: "f32[992][1]cuda:0" = torch.ops.aten.add.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
        unsqueeze_472: "f32[992, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_713, -1)
        unsqueeze_473: "f32[992, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        mul_832: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_826, unsqueeze_473);  mul_826 = unsqueeze_473 = None
        unsqueeze_474: "f32[992, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_714, -1);  primals_714 = None
        unsqueeze_475: "f32[992, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        add_594: "f32[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_832, unsqueeze_475);  mul_832 = unsqueeze_475 = None
        convert_element_type_356: "bf16[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_594, torch.bfloat16);  add_594 = None
        relu_118: "bf16[4, 992, 7, 7][48608, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_356);  convert_element_type_356 = None
        convert_element_type_357: "bf16[128, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_715, torch.bfloat16);  primals_715 = None
        convolution_118: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_118, convert_element_type_357, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        add_595: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_716, 1)
        convert_element_type_358: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_118, torch.float32)
        var_mean_119 = torch.ops.aten.var_mean.correction(convert_element_type_358, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_358 = None
        getitem_240: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_119[0]
        getitem_241: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_119[1];  var_mean_119 = None
        add_596: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_240, 1e-05)
        rsqrt_119: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_596);  add_596 = None
        sub_119: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convolution_118, getitem_241)
        mul_833: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_119);  sub_119 = None
        squeeze_357: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_241, [0, 2, 3]);  getitem_241 = None
        squeeze_358: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_119, [0, 2, 3]);  rsqrt_119 = None
        mul_834: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_357, 0.1)
        mul_835: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_717, 0.9)
        add_597: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_834, mul_835);  mul_834 = mul_835 = None
        squeeze_359: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_240, [0, 2, 3]);  getitem_240 = None
        mul_836: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_359, 1.005128205128205);  squeeze_359 = None
        mul_837: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_836, 0.1);  mul_836 = None
        mul_838: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_718, 0.9)
        add_598: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_837, mul_838);  mul_837 = mul_838 = None
        unsqueeze_476: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_719, -1)
        unsqueeze_477: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_839: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_833, unsqueeze_477);  mul_833 = unsqueeze_477 = None
        unsqueeze_478: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_720, -1);  primals_720 = None
        unsqueeze_479: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_599: "f32[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_839, unsqueeze_479);  mul_839 = unsqueeze_479 = None
        convert_element_type_359: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_599, torch.bfloat16);  add_599 = None
        relu_119: "bf16[4, 128, 7, 7][6272, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_359);  convert_element_type_359 = None
        convert_element_type_360: "bf16[32, 128, 3, 3][1152, 9, 3, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_721, torch.bfloat16);  primals_721 = None
        convolution_119: "bf16[4, 32, 7, 7][1568, 49, 7, 1]cuda:0" = torch.ops.aten.convolution.default(relu_119, convert_element_type_360, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:124 in forward, code: return torch.cat(features, 1)
        cat_57: "bf16[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.cat.default([avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95, convolution_97, convolution_99, convolution_101, convolution_103, convolution_105, convolution_107, convolution_109, convolution_111, convolution_113, convolution_115, convolution_117, convolution_119], 1);  convolution_89 = convolution_91 = convolution_93 = convolution_95 = convolution_97 = convolution_99 = convolution_101 = convolution_103 = convolution_105 = convolution_107 = convolution_109 = convolution_111 = convolution_113 = convolution_115 = convolution_117 = convolution_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        add_600: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_722, 1)
        convert_element_type_361: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(cat_57, torch.float32)
        var_mean_120 = torch.ops.aten.var_mean.correction(convert_element_type_361, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_361 = None
        getitem_242: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_120[0]
        getitem_243: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = var_mean_120[1];  var_mean_120 = None
        add_601: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_242, 1e-05)
        rsqrt_120: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_601);  add_601 = None
        sub_120: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(cat_57, getitem_243)
        mul_840: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, rsqrt_120);  sub_120 = None
        squeeze_360: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_243, [0, 2, 3])
        mul_841: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_360, 0.1);  squeeze_360 = None
        mul_842: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_723, 0.9)
        add_602: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_841, mul_842);  mul_841 = mul_842 = None
        squeeze_362: "f32[1024][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_242, [0, 2, 3]);  getitem_242 = None
        mul_843: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_362, 1.005128205128205);  squeeze_362 = None
        mul_844: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_843, 0.1);  mul_843 = None
        mul_845: "f32[1024][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_724, 0.9)
        add_603: "f32[1024][1]cuda:0" = torch.ops.aten.add.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
        unsqueeze_480: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_725, -1)
        unsqueeze_481: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        mul_846: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_840, unsqueeze_481);  mul_840 = unsqueeze_481 = None
        unsqueeze_482: "f32[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_726, -1)
        unsqueeze_483: "f32[1024, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        add_604: "f32[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_846, unsqueeze_483);  mul_846 = unsqueeze_483 = None
        convert_element_type_362: "bf16[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_604, torch.bfloat16);  add_604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:214 in forward, code: out = F.relu(features, inplace=True)
        relu_120: "bf16[4, 1024, 7, 7][50176, 49, 7, 1]cuda:0" = torch.ops.aten.relu.default(convert_element_type_362);  convert_element_type_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:215 in forward, code: out = F.adaptive_avg_pool2d(out, (1, 1))
        mean: "bf16[4, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(relu_120, [-1, -2], True);  relu_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:216 in forward, code: out = torch.flatten(out, 1)
        view: "bf16[4, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mean, [4, 1024]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:217 in forward, code: out = self.classifier(out)
        convert_element_type_363: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_728, torch.bfloat16);  primals_728 = None
        convert_element_type_364: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_727, torch.bfloat16);  primals_727 = None
        permute: "bf16[1024, 1000][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_364, [1, 0]);  convert_element_type_364 = None
        addmm: "bf16[4, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_363, view, permute);  convert_element_type_363 = None
        permute_1: "bf16[1000, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_496: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_357, 0);  squeeze_357 = None
        unsqueeze_497: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_508: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_354, 0);  squeeze_354 = None
        unsqueeze_509: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_520: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_351, 0);  squeeze_351 = None
        unsqueeze_521: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_532: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_348, 0);  squeeze_348 = None
        unsqueeze_533: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
        unsqueeze_534: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_544: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_345, 0);  squeeze_345 = None
        unsqueeze_545: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
        unsqueeze_546: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_556: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_342, 0);  squeeze_342 = None
        unsqueeze_557: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
        unsqueeze_558: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_568: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_339, 0);  squeeze_339 = None
        unsqueeze_569: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
        unsqueeze_570: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_580: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_336, 0);  squeeze_336 = None
        unsqueeze_581: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
        unsqueeze_582: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_592: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_333, 0);  squeeze_333 = None
        unsqueeze_593: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
        unsqueeze_594: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_604: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_330, 0);  squeeze_330 = None
        unsqueeze_605: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
        unsqueeze_606: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_616: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_327, 0);  squeeze_327 = None
        unsqueeze_617: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
        unsqueeze_618: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_628: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_324, 0);  squeeze_324 = None
        unsqueeze_629: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
        unsqueeze_630: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_640: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_321, 0);  squeeze_321 = None
        unsqueeze_641: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
        unsqueeze_642: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_652: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_318, 0);  squeeze_318 = None
        unsqueeze_653: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
        unsqueeze_654: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_664: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_315, 0);  squeeze_315 = None
        unsqueeze_665: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
        unsqueeze_666: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_676: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_312, 0);  squeeze_312 = None
        unsqueeze_677: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
        unsqueeze_678: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_688: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_309, 0);  squeeze_309 = None
        unsqueeze_689: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
        unsqueeze_690: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_700: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_306, 0);  squeeze_306 = None
        unsqueeze_701: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
        unsqueeze_702: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_712: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_303, 0);  squeeze_303 = None
        unsqueeze_713: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
        unsqueeze_714: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_724: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_300, 0);  squeeze_300 = None
        unsqueeze_725: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
        unsqueeze_726: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_736: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_297, 0);  squeeze_297 = None
        unsqueeze_737: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
        unsqueeze_738: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_748: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_294, 0);  squeeze_294 = None
        unsqueeze_749: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
        unsqueeze_750: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_760: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_291, 0);  squeeze_291 = None
        unsqueeze_761: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
        unsqueeze_762: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_772: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_288, 0);  squeeze_288 = None
        unsqueeze_773: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
        unsqueeze_774: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_784: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_285, 0);  squeeze_285 = None
        unsqueeze_785: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
        unsqueeze_786: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_796: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_282, 0);  squeeze_282 = None
        unsqueeze_797: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
        unsqueeze_798: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_808: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_279, 0);  squeeze_279 = None
        unsqueeze_809: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
        unsqueeze_810: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_820: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_276, 0);  squeeze_276 = None
        unsqueeze_821: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
        unsqueeze_822: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_832: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_273, 0);  squeeze_273 = None
        unsqueeze_833: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
        unsqueeze_834: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_844: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
        unsqueeze_845: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
        unsqueeze_846: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_856: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
        unsqueeze_857: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
        unsqueeze_858: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_868: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_264, 0);  squeeze_264 = None
        unsqueeze_869: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
        unsqueeze_870: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        unsqueeze_880: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_261, 0);  squeeze_261 = None
        unsqueeze_881: "f32[1, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
        unsqueeze_882: "f32[1, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_892: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
        unsqueeze_893: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
        unsqueeze_894: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_904: "f32[1, 992][992, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_255, 0);  squeeze_255 = None
        unsqueeze_905: "f32[1, 992, 1][992, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 2);  unsqueeze_904 = None
        unsqueeze_906: "f32[1, 992, 1, 1][992, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 3);  unsqueeze_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_916: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_252, 0);  squeeze_252 = None
        unsqueeze_917: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_916, 2);  unsqueeze_916 = None
        unsqueeze_918: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 3);  unsqueeze_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_928: "f32[1, 960][960, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_249, 0);  squeeze_249 = None
        unsqueeze_929: "f32[1, 960, 1][960, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
        unsqueeze_930: "f32[1, 960, 1, 1][960, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_940: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_246, 0);  squeeze_246 = None
        unsqueeze_941: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
        unsqueeze_942: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_952: "f32[1, 928][928, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
        unsqueeze_953: "f32[1, 928, 1][928, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
        unsqueeze_954: "f32[1, 928, 1, 1][928, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_964: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
        unsqueeze_965: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
        unsqueeze_966: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_976: "f32[1, 896][896, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_237, 0);  squeeze_237 = None
        unsqueeze_977: "f32[1, 896, 1][896, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_976, 2);  unsqueeze_976 = None
        unsqueeze_978: "f32[1, 896, 1, 1][896, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 3);  unsqueeze_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_988: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_234, 0);  squeeze_234 = None
        unsqueeze_989: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
        unsqueeze_990: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1000: "f32[1, 864][864, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
        unsqueeze_1001: "f32[1, 864, 1][864, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
        unsqueeze_1002: "f32[1, 864, 1, 1][864, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1012: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_228, 0);  squeeze_228 = None
        unsqueeze_1013: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, 2);  unsqueeze_1012 = None
        unsqueeze_1014: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 3);  unsqueeze_1013 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1024: "f32[1, 832][832, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_225, 0);  squeeze_225 = None
        unsqueeze_1025: "f32[1, 832, 1][832, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, 2);  unsqueeze_1024 = None
        unsqueeze_1026: "f32[1, 832, 1, 1][832, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 3);  unsqueeze_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1036: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
        unsqueeze_1037: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, 2);  unsqueeze_1036 = None
        unsqueeze_1038: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 3);  unsqueeze_1037 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1048: "f32[1, 800][800, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
        unsqueeze_1049: "f32[1, 800, 1][800, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
        unsqueeze_1050: "f32[1, 800, 1, 1][800, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1060: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
        unsqueeze_1061: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 2);  unsqueeze_1060 = None
        unsqueeze_1062: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 3);  unsqueeze_1061 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1072: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_213, 0);  squeeze_213 = None
        unsqueeze_1073: "f32[1, 768, 1][768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 2);  unsqueeze_1072 = None
        unsqueeze_1074: "f32[1, 768, 1, 1][768, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 3);  unsqueeze_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1084: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
        unsqueeze_1085: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 2);  unsqueeze_1084 = None
        unsqueeze_1086: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 3);  unsqueeze_1085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1096: "f32[1, 736][736, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_207, 0);  squeeze_207 = None
        unsqueeze_1097: "f32[1, 736, 1][736, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, 2);  unsqueeze_1096 = None
        unsqueeze_1098: "f32[1, 736, 1, 1][736, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1097, 3);  unsqueeze_1097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1108: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_204, 0);  squeeze_204 = None
        unsqueeze_1109: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 2);  unsqueeze_1108 = None
        unsqueeze_1110: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 3);  unsqueeze_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1120: "f32[1, 704][704, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
        unsqueeze_1121: "f32[1, 704, 1][704, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 2);  unsqueeze_1120 = None
        unsqueeze_1122: "f32[1, 704, 1, 1][704, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 3);  unsqueeze_1121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1132: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
        unsqueeze_1133: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, 2);  unsqueeze_1132 = None
        unsqueeze_1134: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1133, 3);  unsqueeze_1133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1144: "f32[1, 672][672, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
        unsqueeze_1145: "f32[1, 672, 1][672, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, 2);  unsqueeze_1144 = None
        unsqueeze_1146: "f32[1, 672, 1, 1][672, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1145, 3);  unsqueeze_1145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1156: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
        unsqueeze_1157: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 2);  unsqueeze_1156 = None
        unsqueeze_1158: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 3);  unsqueeze_1157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1168: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_189, 0);  squeeze_189 = None
        unsqueeze_1169: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 2);  unsqueeze_1168 = None
        unsqueeze_1170: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 3);  unsqueeze_1169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1180: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
        unsqueeze_1181: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, 2);  unsqueeze_1180 = None
        unsqueeze_1182: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1181, 3);  unsqueeze_1181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1192: "f32[1, 608][608, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
        unsqueeze_1193: "f32[1, 608, 1][608, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, 2);  unsqueeze_1192 = None
        unsqueeze_1194: "f32[1, 608, 1, 1][608, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1193, 3);  unsqueeze_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1204: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_180, 0);  squeeze_180 = None
        unsqueeze_1205: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, 2);  unsqueeze_1204 = None
        unsqueeze_1206: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1205, 3);  unsqueeze_1205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1216: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
        unsqueeze_1217: "f32[1, 576, 1][576, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 2);  unsqueeze_1216 = None
        unsqueeze_1218: "f32[1, 576, 1, 1][576, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 3);  unsqueeze_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1228: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
        unsqueeze_1229: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 2);  unsqueeze_1228 = None
        unsqueeze_1230: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 3);  unsqueeze_1229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1240: "f32[1, 544][544, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
        unsqueeze_1241: "f32[1, 544, 1][544, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1240, 2);  unsqueeze_1240 = None
        unsqueeze_1242: "f32[1, 544, 1, 1][544, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1241, 3);  unsqueeze_1241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1252: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
        unsqueeze_1253: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 2);  unsqueeze_1252 = None
        unsqueeze_1254: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 3);  unsqueeze_1253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1264: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
        unsqueeze_1265: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1264, 2);  unsqueeze_1264 = None
        unsqueeze_1266: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1265, 3);  unsqueeze_1265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1276: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
        unsqueeze_1277: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1276, 2);  unsqueeze_1276 = None
        unsqueeze_1278: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1277, 3);  unsqueeze_1277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1288: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_159, 0);  squeeze_159 = None
        unsqueeze_1289: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1288, 2);  unsqueeze_1288 = None
        unsqueeze_1290: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1289, 3);  unsqueeze_1289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1300: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_1301: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 2);  unsqueeze_1300 = None
        unsqueeze_1302: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 3);  unsqueeze_1301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1312: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_1313: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 2);  unsqueeze_1312 = None
        unsqueeze_1314: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 3);  unsqueeze_1313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1324: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_150, 0);  squeeze_150 = None
        unsqueeze_1325: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1324, 2);  unsqueeze_1324 = None
        unsqueeze_1326: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1325, 3);  unsqueeze_1325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1336: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_1337: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 2);  unsqueeze_1336 = None
        unsqueeze_1338: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 3);  unsqueeze_1337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1348: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_1349: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1348, 2);  unsqueeze_1348 = None
        unsqueeze_1350: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1349, 3);  unsqueeze_1349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1360: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_1361: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1360, 2);  unsqueeze_1360 = None
        unsqueeze_1362: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1361, 3);  unsqueeze_1361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1372: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_1373: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1372, 2);  unsqueeze_1372 = None
        unsqueeze_1374: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1373, 3);  unsqueeze_1373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1384: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_1385: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 2);  unsqueeze_1384 = None
        unsqueeze_1386: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 3);  unsqueeze_1385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1396: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_1397: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, 2);  unsqueeze_1396 = None
        unsqueeze_1398: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 3);  unsqueeze_1397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1408: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_1409: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1408, 2);  unsqueeze_1408 = None
        unsqueeze_1410: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1409, 3);  unsqueeze_1409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1420: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_1421: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, 2);  unsqueeze_1420 = None
        unsqueeze_1422: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 3);  unsqueeze_1421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1432: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_1433: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1432, 2);  unsqueeze_1432 = None
        unsqueeze_1434: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1433, 3);  unsqueeze_1433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1444: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_1445: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1444, 2);  unsqueeze_1444 = None
        unsqueeze_1446: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1445, 3);  unsqueeze_1445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1456: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_1457: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 2);  unsqueeze_1456 = None
        unsqueeze_1458: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 3);  unsqueeze_1457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        unsqueeze_1468: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_1469: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1468, 2);  unsqueeze_1468 = None
        unsqueeze_1470: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1469, 3);  unsqueeze_1469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1480: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_1481: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, 2);  unsqueeze_1480 = None
        unsqueeze_1482: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 3);  unsqueeze_1481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1492: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_1493: "f32[1, 480, 1][480, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, 2);  unsqueeze_1492 = None
        unsqueeze_1494: "f32[1, 480, 1, 1][480, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 3);  unsqueeze_1493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1504: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_1505: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1504, 2);  unsqueeze_1504 = None
        unsqueeze_1506: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1505, 3);  unsqueeze_1505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1516: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_1517: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1516, 2);  unsqueeze_1516 = None
        unsqueeze_1518: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1517, 3);  unsqueeze_1517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1528: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_1529: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1528, 2);  unsqueeze_1528 = None
        unsqueeze_1530: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1529, 3);  unsqueeze_1529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1540: "f32[1, 416][416, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_1541: "f32[1, 416, 1][416, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1540, 2);  unsqueeze_1540 = None
        unsqueeze_1542: "f32[1, 416, 1, 1][416, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1541, 3);  unsqueeze_1541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1552: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_1553: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1552, 2);  unsqueeze_1552 = None
        unsqueeze_1554: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1553, 3);  unsqueeze_1553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1564: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_1565: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1564, 2);  unsqueeze_1564 = None
        unsqueeze_1566: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1565, 3);  unsqueeze_1565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1576: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_1577: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1576, 2);  unsqueeze_1576 = None
        unsqueeze_1578: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1577, 3);  unsqueeze_1577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1588: "f32[1, 352][352, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_1589: "f32[1, 352, 1][352, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1588, 2);  unsqueeze_1588 = None
        unsqueeze_1590: "f32[1, 352, 1, 1][352, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1589, 3);  unsqueeze_1589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1600: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_1601: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1600, 2);  unsqueeze_1600 = None
        unsqueeze_1602: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1601, 3);  unsqueeze_1601 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1612: "f32[1, 320][320, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_1613: "f32[1, 320, 1][320, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1612, 2);  unsqueeze_1612 = None
        unsqueeze_1614: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1613, 3);  unsqueeze_1613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1624: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_1625: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1624, 2);  unsqueeze_1624 = None
        unsqueeze_1626: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1625, 3);  unsqueeze_1625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1636: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_1637: "f32[1, 288, 1][288, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1636, 2);  unsqueeze_1636 = None
        unsqueeze_1638: "f32[1, 288, 1, 1][288, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1637, 3);  unsqueeze_1637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1648: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_1649: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1648, 2);  unsqueeze_1648 = None
        unsqueeze_1650: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1649, 3);  unsqueeze_1649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1660: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_1661: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1660, 2);  unsqueeze_1660 = None
        unsqueeze_1662: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1661, 3);  unsqueeze_1661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1672: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_1673: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1672, 2);  unsqueeze_1672 = None
        unsqueeze_1674: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1673, 3);  unsqueeze_1673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1684: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_1685: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1684, 2);  unsqueeze_1684 = None
        unsqueeze_1686: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1685, 3);  unsqueeze_1685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1696: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_1697: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1696, 2);  unsqueeze_1696 = None
        unsqueeze_1698: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1697, 3);  unsqueeze_1697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1708: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_1709: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1708, 2);  unsqueeze_1708 = None
        unsqueeze_1710: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1709, 3);  unsqueeze_1709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1720: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_1721: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1720, 2);  unsqueeze_1720 = None
        unsqueeze_1722: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1721, 3);  unsqueeze_1721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1732: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_1733: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1732, 2);  unsqueeze_1732 = None
        unsqueeze_1734: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1733, 3);  unsqueeze_1733 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1744: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_1745: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1744, 2);  unsqueeze_1744 = None
        unsqueeze_1746: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1745, 3);  unsqueeze_1745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1756: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_1757: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1756, 2);  unsqueeze_1756 = None
        unsqueeze_1758: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1757, 3);  unsqueeze_1757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:213 in forward, code: features = self.features(x)
        unsqueeze_1768: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_1769: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1768, 2);  unsqueeze_1768 = None
        unsqueeze_1770: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1769, 3);  unsqueeze_1769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1780: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_1781: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1780, 2);  unsqueeze_1780 = None
        unsqueeze_1782: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1781, 3);  unsqueeze_1781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1792: "f32[1, 224][224, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_1793: "f32[1, 224, 1][224, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1792, 2);  unsqueeze_1792 = None
        unsqueeze_1794: "f32[1, 224, 1, 1][224, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1793, 3);  unsqueeze_1793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1804: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_1805: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1804, 2);  unsqueeze_1804 = None
        unsqueeze_1806: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1805, 3);  unsqueeze_1805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1816: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_1817: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1816, 2);  unsqueeze_1816 = None
        unsqueeze_1818: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1817, 3);  unsqueeze_1817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1828: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_1829: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1828, 2);  unsqueeze_1828 = None
        unsqueeze_1830: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1829, 3);  unsqueeze_1829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1840: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_1841: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1840, 2);  unsqueeze_1840 = None
        unsqueeze_1842: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1841, 3);  unsqueeze_1841 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1852: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_1853: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1852, 2);  unsqueeze_1852 = None
        unsqueeze_1854: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1853, 3);  unsqueeze_1853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1864: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_1865: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1864, 2);  unsqueeze_1864 = None
        unsqueeze_1866: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1865, 3);  unsqueeze_1865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1876: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_1877: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1876, 2);  unsqueeze_1876 = None
        unsqueeze_1878: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1877, 3);  unsqueeze_1877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1888: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_1889: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1888, 2);  unsqueeze_1888 = None
        unsqueeze_1890: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1889, 3);  unsqueeze_1889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:90 in forward, code: new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        unsqueeze_1900: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_1901: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1900, 2);  unsqueeze_1900 = None
        unsqueeze_1902: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1901, 3);  unsqueeze_1901 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/densenet.py:49 in bn_function, code: bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        unsqueeze_1912: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_1913: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1912, 2);  unsqueeze_1912 = None
        unsqueeze_1914: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1913, 3);  unsqueeze_1913 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_8, add_5);  primals_8 = add_5 = copy__3 = None
        copy__4: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_9, add_7);  primals_9 = add_7 = copy__4 = None
        copy__5: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_8);  primals_10 = add_8 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_14, add_10);  primals_14 = add_10 = copy__6 = None
        copy__7: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_15, add_12);  primals_15 = add_12 = copy__7 = None
        copy__8: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_16, add_13);  primals_16 = add_13 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_20, add_15);  primals_20 = add_15 = copy__9 = None
        copy__10: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_17);  primals_21 = add_17 = copy__10 = None
        copy__11: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_22, add_18);  primals_22 = add_18 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_26, add_20);  primals_26 = add_20 = copy__12 = None
        copy__13: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_22);  primals_27 = add_22 = copy__13 = None
        copy__14: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_28, add_23);  primals_28 = add_23 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_32, add_25);  primals_32 = add_25 = copy__15 = None
        copy__16: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_27);  primals_33 = add_27 = copy__16 = None
        copy__17: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_28);  primals_34 = add_28 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_38, add_30);  primals_38 = add_30 = copy__18 = None
        copy__19: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_39, add_32);  primals_39 = add_32 = copy__19 = None
        copy__20: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_33);  primals_40 = add_33 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_44, add_35);  primals_44 = add_35 = copy__21 = None
        copy__22: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_45, add_37);  primals_45 = add_37 = copy__22 = None
        copy__23: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_46, add_38);  primals_46 = add_38 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_50, add_40);  primals_50 = add_40 = copy__24 = None
        copy__25: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_51, add_42);  primals_51 = add_42 = copy__25 = None
        copy__26: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_52, add_43);  primals_52 = add_43 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_56, add_45);  primals_56 = add_45 = copy__27 = None
        copy__28: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_47);  primals_57 = add_47 = copy__28 = None
        copy__29: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_48);  primals_58 = add_48 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_62, add_50);  primals_62 = add_50 = copy__30 = None
        copy__31: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_52);  primals_63 = add_52 = copy__31 = None
        copy__32: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_53);  primals_64 = add_53 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_68, add_55);  primals_68 = add_55 = copy__33 = None
        copy__34: "f32[224][1]cuda:0" = torch.ops.aten.copy_.default(primals_69, add_57);  primals_69 = add_57 = copy__34 = None
        copy__35: "f32[224][1]cuda:0" = torch.ops.aten.copy_.default(primals_70, add_58);  primals_70 = add_58 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_74, add_60);  primals_74 = add_60 = copy__36 = None
        copy__37: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_62);  primals_75 = add_62 = copy__37 = None
        copy__38: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_63);  primals_76 = add_63 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_80, add_65);  primals_80 = add_65 = copy__39 = None
        copy__40: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_67);  primals_81 = add_67 = copy__40 = None
        copy__41: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_82, add_68);  primals_82 = add_68 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_86, add_70);  primals_86 = add_70 = copy__42 = None
        copy__43: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_72);  primals_87 = add_72 = copy__43 = None
        copy__44: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_88, add_73);  primals_88 = add_73 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_92, add_75);  primals_92 = add_75 = copy__45 = None
        copy__46: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_93, add_77);  primals_93 = add_77 = copy__46 = None
        copy__47: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_94, add_78);  primals_94 = add_78 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_98, add_80);  primals_98 = add_80 = copy__48 = None
        copy__49: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_99, add_82);  primals_99 = add_82 = copy__49 = None
        copy__50: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_100, add_83);  primals_100 = add_83 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_104, add_85);  primals_104 = add_85 = copy__51 = None
        copy__52: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_105, add_87);  primals_105 = add_87 = copy__52 = None
        copy__53: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_106, add_88);  primals_106 = add_88 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_110, add_90);  primals_110 = add_90 = copy__54 = None
        copy__55: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_111, add_92);  primals_111 = add_92 = copy__55 = None
        copy__56: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_112, add_93);  primals_112 = add_93 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_116, add_95);  primals_116 = add_95 = copy__57 = None
        copy__58: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_117, add_97);  primals_117 = add_97 = copy__58 = None
        copy__59: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_118, add_98);  primals_118 = add_98 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_122, add_100);  primals_122 = add_100 = copy__60 = None
        copy__61: "f32[224][1]cuda:0" = torch.ops.aten.copy_.default(primals_123, add_102);  primals_123 = add_102 = copy__61 = None
        copy__62: "f32[224][1]cuda:0" = torch.ops.aten.copy_.default(primals_124, add_103);  primals_124 = add_103 = copy__62 = None
        copy__63: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_128, add_105);  primals_128 = add_105 = copy__63 = None
        copy__64: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_129, add_107);  primals_129 = add_107 = copy__64 = None
        copy__65: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_130, add_108);  primals_130 = add_108 = copy__65 = None
        copy__66: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_134, add_110);  primals_134 = add_110 = copy__66 = None
        copy__67: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_135, add_112);  primals_135 = add_112 = copy__67 = None
        copy__68: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_136, add_113);  primals_136 = add_113 = copy__68 = None
        copy__69: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_140, add_115);  primals_140 = add_115 = copy__69 = None
        copy__70: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_141, add_117);  primals_141 = add_117 = copy__70 = None
        copy__71: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_142, add_118);  primals_142 = add_118 = copy__71 = None
        copy__72: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_146, add_120);  primals_146 = add_120 = copy__72 = None
        copy__73: "f32[288][1]cuda:0" = torch.ops.aten.copy_.default(primals_147, add_122);  primals_147 = add_122 = copy__73 = None
        copy__74: "f32[288][1]cuda:0" = torch.ops.aten.copy_.default(primals_148, add_123);  primals_148 = add_123 = copy__74 = None
        copy__75: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_152, add_125);  primals_152 = add_125 = copy__75 = None
        copy__76: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_153, add_127);  primals_153 = add_127 = copy__76 = None
        copy__77: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_154, add_128);  primals_154 = add_128 = copy__77 = None
        copy__78: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_158, add_130);  primals_158 = add_130 = copy__78 = None
        copy__79: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_159, add_132);  primals_159 = add_132 = copy__79 = None
        copy__80: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_160, add_133);  primals_160 = add_133 = copy__80 = None
        copy__81: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_164, add_135);  primals_164 = add_135 = copy__81 = None
        copy__82: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_165, add_137);  primals_165 = add_137 = copy__82 = None
        copy__83: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_166, add_138);  primals_166 = add_138 = copy__83 = None
        copy__84: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_170, add_140);  primals_170 = add_140 = copy__84 = None
        copy__85: "f32[352][1]cuda:0" = torch.ops.aten.copy_.default(primals_171, add_142);  primals_171 = add_142 = copy__85 = None
        copy__86: "f32[352][1]cuda:0" = torch.ops.aten.copy_.default(primals_172, add_143);  primals_172 = add_143 = copy__86 = None
        copy__87: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_176, add_145);  primals_176 = add_145 = copy__87 = None
        copy__88: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_177, add_147);  primals_177 = add_147 = copy__88 = None
        copy__89: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_178, add_148);  primals_178 = add_148 = copy__89 = None
        copy__90: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_182, add_150);  primals_182 = add_150 = copy__90 = None
        copy__91: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_183, add_152);  primals_183 = add_152 = copy__91 = None
        copy__92: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_184, add_153);  primals_184 = add_153 = copy__92 = None
        copy__93: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_188, add_155);  primals_188 = add_155 = copy__93 = None
        copy__94: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_189, add_157);  primals_189 = add_157 = copy__94 = None
        copy__95: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_190, add_158);  primals_190 = add_158 = copy__95 = None
        copy__96: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_194, add_160);  primals_194 = add_160 = copy__96 = None
        copy__97: "f32[416][1]cuda:0" = torch.ops.aten.copy_.default(primals_195, add_162);  primals_195 = add_162 = copy__97 = None
        copy__98: "f32[416][1]cuda:0" = torch.ops.aten.copy_.default(primals_196, add_163);  primals_196 = add_163 = copy__98 = None
        copy__99: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_200, add_165);  primals_200 = add_165 = copy__99 = None
        copy__100: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_201, add_167);  primals_201 = add_167 = copy__100 = None
        copy__101: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_202, add_168);  primals_202 = add_168 = copy__101 = None
        copy__102: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_206, add_170);  primals_206 = add_170 = copy__102 = None
        copy__103: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_207, add_172);  primals_207 = add_172 = copy__103 = None
        copy__104: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_208, add_173);  primals_208 = add_173 = copy__104 = None
        copy__105: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_212, add_175);  primals_212 = add_175 = copy__105 = None
        copy__106: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_213, add_177);  primals_213 = add_177 = copy__106 = None
        copy__107: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_214, add_178);  primals_214 = add_178 = copy__107 = None
        copy__108: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_218, add_180);  primals_218 = add_180 = copy__108 = None
        copy__109: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_219, add_182);  primals_219 = add_182 = copy__109 = None
        copy__110: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_220, add_183);  primals_220 = add_183 = copy__110 = None
        copy__111: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_224, add_185);  primals_224 = add_185 = copy__111 = None
        copy__112: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_225, add_187);  primals_225 = add_187 = copy__112 = None
        copy__113: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_226, add_188);  primals_226 = add_188 = copy__113 = None
        copy__114: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_230, add_190);  primals_230 = add_190 = copy__114 = None
        copy__115: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_231, add_192);  primals_231 = add_192 = copy__115 = None
        copy__116: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_232, add_193);  primals_232 = add_193 = copy__116 = None
        copy__117: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_236, add_195);  primals_236 = add_195 = copy__117 = None
        copy__118: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_237, add_197);  primals_237 = add_197 = copy__118 = None
        copy__119: "f32[256][1]cuda:0" = torch.ops.aten.copy_.default(primals_238, add_198);  primals_238 = add_198 = copy__119 = None
        copy__120: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_242, add_200);  primals_242 = add_200 = copy__120 = None
        copy__121: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_243, add_202);  primals_243 = add_202 = copy__121 = None
        copy__122: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_244, add_203);  primals_244 = add_203 = copy__122 = None
        copy__123: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_248, add_205);  primals_248 = add_205 = copy__123 = None
        copy__124: "f32[288][1]cuda:0" = torch.ops.aten.copy_.default(primals_249, add_207);  primals_249 = add_207 = copy__124 = None
        copy__125: "f32[288][1]cuda:0" = torch.ops.aten.copy_.default(primals_250, add_208);  primals_250 = add_208 = copy__125 = None
        copy__126: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_254, add_210);  primals_254 = add_210 = copy__126 = None
        copy__127: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_255, add_212);  primals_255 = add_212 = copy__127 = None
        copy__128: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_256, add_213);  primals_256 = add_213 = copy__128 = None
        copy__129: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_260, add_215);  primals_260 = add_215 = copy__129 = None
        copy__130: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_261, add_217);  primals_261 = add_217 = copy__130 = None
        copy__131: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_262, add_218);  primals_262 = add_218 = copy__131 = None
        copy__132: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_266, add_220);  primals_266 = add_220 = copy__132 = None
        copy__133: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_267, add_222);  primals_267 = add_222 = copy__133 = None
        copy__134: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_268, add_223);  primals_268 = add_223 = copy__134 = None
        copy__135: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_272, add_225);  primals_272 = add_225 = copy__135 = None
        copy__136: "f32[352][1]cuda:0" = torch.ops.aten.copy_.default(primals_273, add_227);  primals_273 = add_227 = copy__136 = None
        copy__137: "f32[352][1]cuda:0" = torch.ops.aten.copy_.default(primals_274, add_228);  primals_274 = add_228 = copy__137 = None
        copy__138: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_278, add_230);  primals_278 = add_230 = copy__138 = None
        copy__139: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_279, add_232);  primals_279 = add_232 = copy__139 = None
        copy__140: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_280, add_233);  primals_280 = add_233 = copy__140 = None
        copy__141: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_284, add_235);  primals_284 = add_235 = copy__141 = None
        copy__142: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_285, add_237);  primals_285 = add_237 = copy__142 = None
        copy__143: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_286, add_238);  primals_286 = add_238 = copy__143 = None
        copy__144: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_290, add_240);  primals_290 = add_240 = copy__144 = None
        copy__145: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_291, add_242);  primals_291 = add_242 = copy__145 = None
        copy__146: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_292, add_243);  primals_292 = add_243 = copy__146 = None
        copy__147: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_296, add_245);  primals_296 = add_245 = copy__147 = None
        copy__148: "f32[416][1]cuda:0" = torch.ops.aten.copy_.default(primals_297, add_247);  primals_297 = add_247 = copy__148 = None
        copy__149: "f32[416][1]cuda:0" = torch.ops.aten.copy_.default(primals_298, add_248);  primals_298 = add_248 = copy__149 = None
        copy__150: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_302, add_250);  primals_302 = add_250 = copy__150 = None
        copy__151: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_303, add_252);  primals_303 = add_252 = copy__151 = None
        copy__152: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_304, add_253);  primals_304 = add_253 = copy__152 = None
        copy__153: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_308, add_255);  primals_308 = add_255 = copy__153 = None
        copy__154: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_309, add_257);  primals_309 = add_257 = copy__154 = None
        copy__155: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_310, add_258);  primals_310 = add_258 = copy__155 = None
        copy__156: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_314, add_260);  primals_314 = add_260 = copy__156 = None
        copy__157: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_315, add_262);  primals_315 = add_262 = copy__157 = None
        copy__158: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_316, add_263);  primals_316 = add_263 = copy__158 = None
        copy__159: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_320, add_265);  primals_320 = add_265 = copy__159 = None
        copy__160: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_321, add_267);  primals_321 = add_267 = copy__160 = None
        copy__161: "f32[480][1]cuda:0" = torch.ops.aten.copy_.default(primals_322, add_268);  primals_322 = add_268 = copy__161 = None
        copy__162: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_326, add_270);  primals_326 = add_270 = copy__162 = None
        copy__163: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_327, add_272);  primals_327 = add_272 = copy__163 = None
        copy__164: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_328, add_273);  primals_328 = add_273 = copy__164 = None
        copy__165: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_332, add_275);  primals_332 = add_275 = copy__165 = None
        copy__166: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_333, add_277);  primals_333 = add_277 = copy__166 = None
        copy__167: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_334, add_278);  primals_334 = add_278 = copy__167 = None
        copy__168: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_338, add_280);  primals_338 = add_280 = copy__168 = None
        copy__169: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_339, add_282);  primals_339 = add_282 = copy__169 = None
        copy__170: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_340, add_283);  primals_340 = add_283 = copy__170 = None
        copy__171: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_344, add_285);  primals_344 = add_285 = copy__171 = None
        copy__172: "f32[544][1]cuda:0" = torch.ops.aten.copy_.default(primals_345, add_287);  primals_345 = add_287 = copy__172 = None
        copy__173: "f32[544][1]cuda:0" = torch.ops.aten.copy_.default(primals_346, add_288);  primals_346 = add_288 = copy__173 = None
        copy__174: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_350, add_290);  primals_350 = add_290 = copy__174 = None
        copy__175: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_351, add_292);  primals_351 = add_292 = copy__175 = None
        copy__176: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_352, add_293);  primals_352 = add_293 = copy__176 = None
        copy__177: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_356, add_295);  primals_356 = add_295 = copy__177 = None
        copy__178: "f32[576][1]cuda:0" = torch.ops.aten.copy_.default(primals_357, add_297);  primals_357 = add_297 = copy__178 = None
        copy__179: "f32[576][1]cuda:0" = torch.ops.aten.copy_.default(primals_358, add_298);  primals_358 = add_298 = copy__179 = None
        copy__180: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_362, add_300);  primals_362 = add_300 = copy__180 = None
        copy__181: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_363, add_302);  primals_363 = add_302 = copy__181 = None
        copy__182: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_364, add_303);  primals_364 = add_303 = copy__182 = None
        copy__183: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_368, add_305);  primals_368 = add_305 = copy__183 = None
        copy__184: "f32[608][1]cuda:0" = torch.ops.aten.copy_.default(primals_369, add_307);  primals_369 = add_307 = copy__184 = None
        copy__185: "f32[608][1]cuda:0" = torch.ops.aten.copy_.default(primals_370, add_308);  primals_370 = add_308 = copy__185 = None
        copy__186: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_374, add_310);  primals_374 = add_310 = copy__186 = None
        copy__187: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_375, add_312);  primals_375 = add_312 = copy__187 = None
        copy__188: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_376, add_313);  primals_376 = add_313 = copy__188 = None
        copy__189: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_380, add_315);  primals_380 = add_315 = copy__189 = None
        copy__190: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_381, add_317);  primals_381 = add_317 = copy__190 = None
        copy__191: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_382, add_318);  primals_382 = add_318 = copy__191 = None
        copy__192: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_386, add_320);  primals_386 = add_320 = copy__192 = None
        copy__193: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_387, add_322);  primals_387 = add_322 = copy__193 = None
        copy__194: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_388, add_323);  primals_388 = add_323 = copy__194 = None
        copy__195: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_392, add_325);  primals_392 = add_325 = copy__195 = None
        copy__196: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_393, add_327);  primals_393 = add_327 = copy__196 = None
        copy__197: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_394, add_328);  primals_394 = add_328 = copy__197 = None
        copy__198: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_398, add_330);  primals_398 = add_330 = copy__198 = None
        copy__199: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_399, add_332);  primals_399 = add_332 = copy__199 = None
        copy__200: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_400, add_333);  primals_400 = add_333 = copy__200 = None
        copy__201: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_404, add_335);  primals_404 = add_335 = copy__201 = None
        copy__202: "f32[704][1]cuda:0" = torch.ops.aten.copy_.default(primals_405, add_337);  primals_405 = add_337 = copy__202 = None
        copy__203: "f32[704][1]cuda:0" = torch.ops.aten.copy_.default(primals_406, add_338);  primals_406 = add_338 = copy__203 = None
        copy__204: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_410, add_340);  primals_410 = add_340 = copy__204 = None
        copy__205: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_411, add_342);  primals_411 = add_342 = copy__205 = None
        copy__206: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_412, add_343);  primals_412 = add_343 = copy__206 = None
        copy__207: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_416, add_345);  primals_416 = add_345 = copy__207 = None
        copy__208: "f32[736][1]cuda:0" = torch.ops.aten.copy_.default(primals_417, add_347);  primals_417 = add_347 = copy__208 = None
        copy__209: "f32[736][1]cuda:0" = torch.ops.aten.copy_.default(primals_418, add_348);  primals_418 = add_348 = copy__209 = None
        copy__210: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_422, add_350);  primals_422 = add_350 = copy__210 = None
        copy__211: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_423, add_352);  primals_423 = add_352 = copy__211 = None
        copy__212: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_424, add_353);  primals_424 = add_353 = copy__212 = None
        copy__213: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_428, add_355);  primals_428 = add_355 = copy__213 = None
        copy__214: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_429, add_357);  primals_429 = add_357 = copy__214 = None
        copy__215: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_430, add_358);  primals_430 = add_358 = copy__215 = None
        copy__216: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_434, add_360);  primals_434 = add_360 = copy__216 = None
        copy__217: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_435, add_362);  primals_435 = add_362 = copy__217 = None
        copy__218: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_436, add_363);  primals_436 = add_363 = copy__218 = None
        copy__219: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_440, add_365);  primals_440 = add_365 = copy__219 = None
        copy__220: "f32[800][1]cuda:0" = torch.ops.aten.copy_.default(primals_441, add_367);  primals_441 = add_367 = copy__220 = None
        copy__221: "f32[800][1]cuda:0" = torch.ops.aten.copy_.default(primals_442, add_368);  primals_442 = add_368 = copy__221 = None
        copy__222: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_446, add_370);  primals_446 = add_370 = copy__222 = None
        copy__223: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_447, add_372);  primals_447 = add_372 = copy__223 = None
        copy__224: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_448, add_373);  primals_448 = add_373 = copy__224 = None
        copy__225: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_452, add_375);  primals_452 = add_375 = copy__225 = None
        copy__226: "f32[832][1]cuda:0" = torch.ops.aten.copy_.default(primals_453, add_377);  primals_453 = add_377 = copy__226 = None
        copy__227: "f32[832][1]cuda:0" = torch.ops.aten.copy_.default(primals_454, add_378);  primals_454 = add_378 = copy__227 = None
        copy__228: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_458, add_380);  primals_458 = add_380 = copy__228 = None
        copy__229: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_459, add_382);  primals_459 = add_382 = copy__229 = None
        copy__230: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_460, add_383);  primals_460 = add_383 = copy__230 = None
        copy__231: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_464, add_385);  primals_464 = add_385 = copy__231 = None
        copy__232: "f32[864][1]cuda:0" = torch.ops.aten.copy_.default(primals_465, add_387);  primals_465 = add_387 = copy__232 = None
        copy__233: "f32[864][1]cuda:0" = torch.ops.aten.copy_.default(primals_466, add_388);  primals_466 = add_388 = copy__233 = None
        copy__234: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_470, add_390);  primals_470 = add_390 = copy__234 = None
        copy__235: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_471, add_392);  primals_471 = add_392 = copy__235 = None
        copy__236: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_472, add_393);  primals_472 = add_393 = copy__236 = None
        copy__237: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_476, add_395);  primals_476 = add_395 = copy__237 = None
        copy__238: "f32[896][1]cuda:0" = torch.ops.aten.copy_.default(primals_477, add_397);  primals_477 = add_397 = copy__238 = None
        copy__239: "f32[896][1]cuda:0" = torch.ops.aten.copy_.default(primals_478, add_398);  primals_478 = add_398 = copy__239 = None
        copy__240: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_482, add_400);  primals_482 = add_400 = copy__240 = None
        copy__241: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_483, add_402);  primals_483 = add_402 = copy__241 = None
        copy__242: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_484, add_403);  primals_484 = add_403 = copy__242 = None
        copy__243: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_488, add_405);  primals_488 = add_405 = copy__243 = None
        copy__244: "f32[928][1]cuda:0" = torch.ops.aten.copy_.default(primals_489, add_407);  primals_489 = add_407 = copy__244 = None
        copy__245: "f32[928][1]cuda:0" = torch.ops.aten.copy_.default(primals_490, add_408);  primals_490 = add_408 = copy__245 = None
        copy__246: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_494, add_410);  primals_494 = add_410 = copy__246 = None
        copy__247: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_495, add_412);  primals_495 = add_412 = copy__247 = None
        copy__248: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_496, add_413);  primals_496 = add_413 = copy__248 = None
        copy__249: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_500, add_415);  primals_500 = add_415 = copy__249 = None
        copy__250: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_501, add_417);  primals_501 = add_417 = copy__250 = None
        copy__251: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_502, add_418);  primals_502 = add_418 = copy__251 = None
        copy__252: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_506, add_420);  primals_506 = add_420 = copy__252 = None
        copy__253: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_507, add_422);  primals_507 = add_422 = copy__253 = None
        copy__254: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_508, add_423);  primals_508 = add_423 = copy__254 = None
        copy__255: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_512, add_425);  primals_512 = add_425 = copy__255 = None
        copy__256: "f32[992][1]cuda:0" = torch.ops.aten.copy_.default(primals_513, add_427);  primals_513 = add_427 = copy__256 = None
        copy__257: "f32[992][1]cuda:0" = torch.ops.aten.copy_.default(primals_514, add_428);  primals_514 = add_428 = copy__257 = None
        copy__258: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_518, add_430);  primals_518 = add_430 = copy__258 = None
        copy__259: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_519, add_432);  primals_519 = add_432 = copy__259 = None
        copy__260: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_520, add_433);  primals_520 = add_433 = copy__260 = None
        copy__261: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_524, add_435);  primals_524 = add_435 = copy__261 = None
        copy__262: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_525, add_437);  primals_525 = add_437 = copy__262 = None
        copy__263: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_526, add_438);  primals_526 = add_438 = copy__263 = None
        copy__264: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_530, add_440);  primals_530 = add_440 = copy__264 = None
        copy__265: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_531, add_442);  primals_531 = add_442 = copy__265 = None
        copy__266: "f32[512][1]cuda:0" = torch.ops.aten.copy_.default(primals_532, add_443);  primals_532 = add_443 = copy__266 = None
        copy__267: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_536, add_445);  primals_536 = add_445 = copy__267 = None
        copy__268: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_537, add_447);  primals_537 = add_447 = copy__268 = None
        copy__269: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_538, add_448);  primals_538 = add_448 = copy__269 = None
        copy__270: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_542, add_450);  primals_542 = add_450 = copy__270 = None
        copy__271: "f32[544][1]cuda:0" = torch.ops.aten.copy_.default(primals_543, add_452);  primals_543 = add_452 = copy__271 = None
        copy__272: "f32[544][1]cuda:0" = torch.ops.aten.copy_.default(primals_544, add_453);  primals_544 = add_453 = copy__272 = None
        copy__273: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_548, add_455);  primals_548 = add_455 = copy__273 = None
        copy__274: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_549, add_457);  primals_549 = add_457 = copy__274 = None
        copy__275: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_550, add_458);  primals_550 = add_458 = copy__275 = None
        copy__276: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_554, add_460);  primals_554 = add_460 = copy__276 = None
        copy__277: "f32[576][1]cuda:0" = torch.ops.aten.copy_.default(primals_555, add_462);  primals_555 = add_462 = copy__277 = None
        copy__278: "f32[576][1]cuda:0" = torch.ops.aten.copy_.default(primals_556, add_463);  primals_556 = add_463 = copy__278 = None
        copy__279: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_560, add_465);  primals_560 = add_465 = copy__279 = None
        copy__280: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_561, add_467);  primals_561 = add_467 = copy__280 = None
        copy__281: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_562, add_468);  primals_562 = add_468 = copy__281 = None
        copy__282: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_566, add_470);  primals_566 = add_470 = copy__282 = None
        copy__283: "f32[608][1]cuda:0" = torch.ops.aten.copy_.default(primals_567, add_472);  primals_567 = add_472 = copy__283 = None
        copy__284: "f32[608][1]cuda:0" = torch.ops.aten.copy_.default(primals_568, add_473);  primals_568 = add_473 = copy__284 = None
        copy__285: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_572, add_475);  primals_572 = add_475 = copy__285 = None
        copy__286: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_573, add_477);  primals_573 = add_477 = copy__286 = None
        copy__287: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_574, add_478);  primals_574 = add_478 = copy__287 = None
        copy__288: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_578, add_480);  primals_578 = add_480 = copy__288 = None
        copy__289: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_579, add_482);  primals_579 = add_482 = copy__289 = None
        copy__290: "f32[640][1]cuda:0" = torch.ops.aten.copy_.default(primals_580, add_483);  primals_580 = add_483 = copy__290 = None
        copy__291: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_584, add_485);  primals_584 = add_485 = copy__291 = None
        copy__292: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_585, add_487);  primals_585 = add_487 = copy__292 = None
        copy__293: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_586, add_488);  primals_586 = add_488 = copy__293 = None
        copy__294: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_590, add_490);  primals_590 = add_490 = copy__294 = None
        copy__295: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_591, add_492);  primals_591 = add_492 = copy__295 = None
        copy__296: "f32[672][1]cuda:0" = torch.ops.aten.copy_.default(primals_592, add_493);  primals_592 = add_493 = copy__296 = None
        copy__297: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_596, add_495);  primals_596 = add_495 = copy__297 = None
        copy__298: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_597, add_497);  primals_597 = add_497 = copy__298 = None
        copy__299: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_598, add_498);  primals_598 = add_498 = copy__299 = None
        copy__300: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_602, add_500);  primals_602 = add_500 = copy__300 = None
        copy__301: "f32[704][1]cuda:0" = torch.ops.aten.copy_.default(primals_603, add_502);  primals_603 = add_502 = copy__301 = None
        copy__302: "f32[704][1]cuda:0" = torch.ops.aten.copy_.default(primals_604, add_503);  primals_604 = add_503 = copy__302 = None
        copy__303: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_608, add_505);  primals_608 = add_505 = copy__303 = None
        copy__304: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_609, add_507);  primals_609 = add_507 = copy__304 = None
        copy__305: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_610, add_508);  primals_610 = add_508 = copy__305 = None
        copy__306: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_614, add_510);  primals_614 = add_510 = copy__306 = None
        copy__307: "f32[736][1]cuda:0" = torch.ops.aten.copy_.default(primals_615, add_512);  primals_615 = add_512 = copy__307 = None
        copy__308: "f32[736][1]cuda:0" = torch.ops.aten.copy_.default(primals_616, add_513);  primals_616 = add_513 = copy__308 = None
        copy__309: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_620, add_515);  primals_620 = add_515 = copy__309 = None
        copy__310: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_621, add_517);  primals_621 = add_517 = copy__310 = None
        copy__311: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_622, add_518);  primals_622 = add_518 = copy__311 = None
        copy__312: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_626, add_520);  primals_626 = add_520 = copy__312 = None
        copy__313: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_627, add_522);  primals_627 = add_522 = copy__313 = None
        copy__314: "f32[768][1]cuda:0" = torch.ops.aten.copy_.default(primals_628, add_523);  primals_628 = add_523 = copy__314 = None
        copy__315: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_632, add_525);  primals_632 = add_525 = copy__315 = None
        copy__316: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_633, add_527);  primals_633 = add_527 = copy__316 = None
        copy__317: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_634, add_528);  primals_634 = add_528 = copy__317 = None
        copy__318: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_638, add_530);  primals_638 = add_530 = copy__318 = None
        copy__319: "f32[800][1]cuda:0" = torch.ops.aten.copy_.default(primals_639, add_532);  primals_639 = add_532 = copy__319 = None
        copy__320: "f32[800][1]cuda:0" = torch.ops.aten.copy_.default(primals_640, add_533);  primals_640 = add_533 = copy__320 = None
        copy__321: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_644, add_535);  primals_644 = add_535 = copy__321 = None
        copy__322: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_645, add_537);  primals_645 = add_537 = copy__322 = None
        copy__323: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_646, add_538);  primals_646 = add_538 = copy__323 = None
        copy__324: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_650, add_540);  primals_650 = add_540 = copy__324 = None
        copy__325: "f32[832][1]cuda:0" = torch.ops.aten.copy_.default(primals_651, add_542);  primals_651 = add_542 = copy__325 = None
        copy__326: "f32[832][1]cuda:0" = torch.ops.aten.copy_.default(primals_652, add_543);  primals_652 = add_543 = copy__326 = None
        copy__327: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_656, add_545);  primals_656 = add_545 = copy__327 = None
        copy__328: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_657, add_547);  primals_657 = add_547 = copy__328 = None
        copy__329: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_658, add_548);  primals_658 = add_548 = copy__329 = None
        copy__330: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_662, add_550);  primals_662 = add_550 = copy__330 = None
        copy__331: "f32[864][1]cuda:0" = torch.ops.aten.copy_.default(primals_663, add_552);  primals_663 = add_552 = copy__331 = None
        copy__332: "f32[864][1]cuda:0" = torch.ops.aten.copy_.default(primals_664, add_553);  primals_664 = add_553 = copy__332 = None
        copy__333: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_668, add_555);  primals_668 = add_555 = copy__333 = None
        copy__334: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_669, add_557);  primals_669 = add_557 = copy__334 = None
        copy__335: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_670, add_558);  primals_670 = add_558 = copy__335 = None
        copy__336: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_674, add_560);  primals_674 = add_560 = copy__336 = None
        copy__337: "f32[896][1]cuda:0" = torch.ops.aten.copy_.default(primals_675, add_562);  primals_675 = add_562 = copy__337 = None
        copy__338: "f32[896][1]cuda:0" = torch.ops.aten.copy_.default(primals_676, add_563);  primals_676 = add_563 = copy__338 = None
        copy__339: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_680, add_565);  primals_680 = add_565 = copy__339 = None
        copy__340: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_681, add_567);  primals_681 = add_567 = copy__340 = None
        copy__341: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_682, add_568);  primals_682 = add_568 = copy__341 = None
        copy__342: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_686, add_570);  primals_686 = add_570 = copy__342 = None
        copy__343: "f32[928][1]cuda:0" = torch.ops.aten.copy_.default(primals_687, add_572);  primals_687 = add_572 = copy__343 = None
        copy__344: "f32[928][1]cuda:0" = torch.ops.aten.copy_.default(primals_688, add_573);  primals_688 = add_573 = copy__344 = None
        copy__345: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_692, add_575);  primals_692 = add_575 = copy__345 = None
        copy__346: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_693, add_577);  primals_693 = add_577 = copy__346 = None
        copy__347: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_694, add_578);  primals_694 = add_578 = copy__347 = None
        copy__348: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_698, add_580);  primals_698 = add_580 = copy__348 = None
        copy__349: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_699, add_582);  primals_699 = add_582 = copy__349 = None
        copy__350: "f32[960][1]cuda:0" = torch.ops.aten.copy_.default(primals_700, add_583);  primals_700 = add_583 = copy__350 = None
        copy__351: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_704, add_585);  primals_704 = add_585 = copy__351 = None
        copy__352: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_705, add_587);  primals_705 = add_587 = copy__352 = None
        copy__353: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_706, add_588);  primals_706 = add_588 = copy__353 = None
        copy__354: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_710, add_590);  primals_710 = add_590 = copy__354 = None
        copy__355: "f32[992][1]cuda:0" = torch.ops.aten.copy_.default(primals_711, add_592);  primals_711 = add_592 = copy__355 = None
        copy__356: "f32[992][1]cuda:0" = torch.ops.aten.copy_.default(primals_712, add_593);  primals_712 = add_593 = copy__356 = None
        copy__357: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_716, add_595);  primals_716 = add_595 = copy__357 = None
        copy__358: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_717, add_597);  primals_717 = add_597 = copy__358 = None
        copy__359: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_718, add_598);  primals_718 = add_598 = copy__359 = None
        copy__360: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_722, add_600);  primals_722 = add_600 = copy__360 = None
        copy__361: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_723, add_602);  primals_723 = add_602 = copy__361 = None
        copy__362: "f32[1024][1]cuda:0" = torch.ops.aten.copy_.default(primals_724, add_603);  primals_724 = add_603 = copy__362 = None
        return (addmm, primals_6, primals_7, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_155, primals_161, primals_167, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_215, primals_221, primals_227, primals_233, primals_239, primals_245, primals_251, primals_257, primals_263, primals_269, primals_275, primals_281, primals_287, primals_293, primals_299, primals_305, primals_311, primals_317, primals_323, primals_329, primals_335, primals_341, primals_347, primals_353, primals_359, primals_365, primals_371, primals_377, primals_383, primals_389, primals_395, primals_401, primals_407, primals_413, primals_419, primals_425, primals_431, primals_437, primals_443, primals_449, primals_455, primals_461, primals_467, primals_473, primals_479, primals_485, primals_491, primals_497, primals_503, primals_509, primals_515, primals_521, primals_527, primals_533, primals_539, primals_545, primals_551, primals_557, primals_563, primals_569, primals_575, primals_581, primals_587, primals_593, primals_599, primals_605, primals_611, primals_617, primals_623, primals_629, primals_635, primals_641, primals_647, primals_653, primals_659, primals_665, primals_671, primals_677, primals_683, primals_689, primals_695, primals_701, primals_707, primals_713, primals_719, primals_725, primals_726, convert_element_type, convert_element_type_1, convolution, getitem_1, rsqrt, getitem_2, getitem_3, squeeze_4, relu_1, convert_element_type_6, convolution_1, squeeze_7, relu_2, convert_element_type_9, cat, squeeze_10, relu_3, convert_element_type_12, convolution_3, squeeze_13, relu_4, convert_element_type_15, cat_1, squeeze_16, relu_5, convert_element_type_18, convolution_5, squeeze_19, relu_6, convert_element_type_21, cat_2, squeeze_22, relu_7, convert_element_type_24, convolution_7, squeeze_25, relu_8, convert_element_type_27, cat_3, squeeze_28, relu_9, convert_element_type_30, convolution_9, squeeze_31, relu_10, convert_element_type_33, cat_4, squeeze_34, relu_11, convert_element_type_36, convolution_11, squeeze_37, relu_12, convert_element_type_39, cat_5, squeeze_40, relu_13, convert_element_type_42, convolution_13, avg_pool2d, squeeze_43, relu_14, convert_element_type_45, convolution_14, squeeze_46, relu_15, convert_element_type_48, cat_6, squeeze_49, relu_16, convert_element_type_51, convolution_16, squeeze_52, relu_17, convert_element_type_54, cat_7, squeeze_55, relu_18, convert_element_type_57, convolution_18, squeeze_58, relu_19, convert_element_type_60, cat_8, squeeze_61, relu_20, convert_element_type_63, convolution_20, squeeze_64, relu_21, convert_element_type_66, cat_9, squeeze_67, relu_22, convert_element_type_69, convolution_22, squeeze_70, relu_23, convert_element_type_72, cat_10, squeeze_73, relu_24, convert_element_type_75, convolution_24, squeeze_76, relu_25, convert_element_type_78, cat_11, squeeze_79, relu_26, convert_element_type_81, convolution_26, squeeze_82, relu_27, convert_element_type_84, cat_12, squeeze_85, relu_28, convert_element_type_87, convolution_28, squeeze_88, relu_29, convert_element_type_90, cat_13, squeeze_91, relu_30, convert_element_type_93, convolution_30, squeeze_94, relu_31, convert_element_type_96, cat_14, squeeze_97, relu_32, convert_element_type_99, convolution_32, squeeze_100, relu_33, convert_element_type_102, cat_15, squeeze_103, relu_34, convert_element_type_105, convolution_34, squeeze_106, relu_35, convert_element_type_108, cat_16, squeeze_109, relu_36, convert_element_type_111, convolution_36, squeeze_112, relu_37, convert_element_type_114, cat_17, squeeze_115, relu_38, convert_element_type_117, convolution_38, avg_pool2d_1, squeeze_118, relu_39, convert_element_type_120, convolution_39, squeeze_121, relu_40, convert_element_type_123, cat_18, squeeze_124, relu_41, convert_element_type_126, convolution_41, squeeze_127, relu_42, convert_element_type_129, cat_19, squeeze_130, relu_43, convert_element_type_132, convolution_43, squeeze_133, relu_44, convert_element_type_135, cat_20, squeeze_136, relu_45, convert_element_type_138, convolution_45, squeeze_139, relu_46, convert_element_type_141, cat_21, squeeze_142, relu_47, convert_element_type_144, convolution_47, squeeze_145, relu_48, convert_element_type_147, cat_22, squeeze_148, relu_49, convert_element_type_150, convolution_49, squeeze_151, relu_50, convert_element_type_153, cat_23, squeeze_154, relu_51, convert_element_type_156, convolution_51, squeeze_157, relu_52, convert_element_type_159, cat_24, squeeze_160, relu_53, convert_element_type_162, convolution_53, squeeze_163, relu_54, convert_element_type_165, cat_25, squeeze_166, relu_55, convert_element_type_168, convolution_55, squeeze_169, relu_56, convert_element_type_171, cat_26, squeeze_172, relu_57, convert_element_type_174, convolution_57, squeeze_175, relu_58, convert_element_type_177, cat_27, squeeze_178, relu_59, convert_element_type_180, convolution_59, squeeze_181, relu_60, convert_element_type_183, cat_28, squeeze_184, relu_61, convert_element_type_186, convolution_61, squeeze_187, relu_62, convert_element_type_189, cat_29, squeeze_190, relu_63, convert_element_type_192, convolution_63, squeeze_193, relu_64, convert_element_type_195, cat_30, squeeze_196, relu_65, convert_element_type_198, convolution_65, squeeze_199, relu_66, convert_element_type_201, cat_31, squeeze_202, relu_67, convert_element_type_204, convolution_67, squeeze_205, relu_68, convert_element_type_207, cat_32, squeeze_208, relu_69, convert_element_type_210, convolution_69, squeeze_211, relu_70, convert_element_type_213, cat_33, squeeze_214, relu_71, convert_element_type_216, convolution_71, squeeze_217, relu_72, convert_element_type_219, cat_34, squeeze_220, relu_73, convert_element_type_222, convolution_73, squeeze_223, relu_74, convert_element_type_225, cat_35, squeeze_226, relu_75, convert_element_type_228, convolution_75, squeeze_229, relu_76, convert_element_type_231, cat_36, squeeze_232, relu_77, convert_element_type_234, convolution_77, squeeze_235, relu_78, convert_element_type_237, cat_37, squeeze_238, relu_79, convert_element_type_240, convolution_79, squeeze_241, relu_80, convert_element_type_243, cat_38, squeeze_244, relu_81, convert_element_type_246, convolution_81, squeeze_247, relu_82, convert_element_type_249, cat_39, squeeze_250, relu_83, convert_element_type_252, convolution_83, squeeze_253, relu_84, convert_element_type_255, cat_40, squeeze_256, relu_85, convert_element_type_258, convolution_85, squeeze_259, relu_86, convert_element_type_261, cat_41, squeeze_262, relu_87, convert_element_type_264, convolution_87, avg_pool2d_2, squeeze_265, relu_88, convert_element_type_267, convolution_88, squeeze_268, relu_89, convert_element_type_270, cat_42, squeeze_271, relu_90, convert_element_type_273, convolution_90, squeeze_274, relu_91, convert_element_type_276, cat_43, squeeze_277, relu_92, convert_element_type_279, convolution_92, squeeze_280, relu_93, convert_element_type_282, cat_44, squeeze_283, relu_94, convert_element_type_285, convolution_94, squeeze_286, relu_95, convert_element_type_288, cat_45, squeeze_289, relu_96, convert_element_type_291, convolution_96, squeeze_292, relu_97, convert_element_type_294, cat_46, squeeze_295, relu_98, convert_element_type_297, convolution_98, squeeze_298, relu_99, convert_element_type_300, cat_47, squeeze_301, relu_100, convert_element_type_303, convolution_100, squeeze_304, relu_101, convert_element_type_306, cat_48, squeeze_307, relu_102, convert_element_type_309, convolution_102, squeeze_310, relu_103, convert_element_type_312, cat_49, squeeze_313, relu_104, convert_element_type_315, convolution_104, squeeze_316, relu_105, convert_element_type_318, cat_50, squeeze_319, relu_106, convert_element_type_321, convolution_106, squeeze_322, relu_107, convert_element_type_324, cat_51, squeeze_325, relu_108, convert_element_type_327, convolution_108, squeeze_328, relu_109, convert_element_type_330, cat_52, squeeze_331, relu_110, convert_element_type_333, convolution_110, squeeze_334, relu_111, convert_element_type_336, cat_53, squeeze_337, relu_112, convert_element_type_339, convolution_112, squeeze_340, relu_113, convert_element_type_342, cat_54, squeeze_343, relu_114, convert_element_type_345, convolution_114, squeeze_346, relu_115, convert_element_type_348, cat_55, squeeze_349, relu_116, convert_element_type_351, convolution_116, squeeze_352, relu_117, convert_element_type_354, cat_56, squeeze_355, relu_118, convert_element_type_357, convolution_118, squeeze_358, relu_119, convert_element_type_360, cat_57, getitem_243, rsqrt_120, view, permute_1, unsqueeze_498, unsqueeze_510, unsqueeze_522, unsqueeze_534, unsqueeze_546, unsqueeze_558, unsqueeze_570, unsqueeze_582, unsqueeze_594, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_642, unsqueeze_654, unsqueeze_666, unsqueeze_678, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, unsqueeze_738, unsqueeze_750, unsqueeze_762, unsqueeze_774, unsqueeze_786, unsqueeze_798, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, unsqueeze_858, unsqueeze_870, unsqueeze_882, unsqueeze_894, unsqueeze_906, unsqueeze_918, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, unsqueeze_978, unsqueeze_990, unsqueeze_1002, unsqueeze_1014, unsqueeze_1026, unsqueeze_1038, unsqueeze_1050, unsqueeze_1062, unsqueeze_1074, unsqueeze_1086, unsqueeze_1098, unsqueeze_1110, unsqueeze_1122, unsqueeze_1134, unsqueeze_1146, unsqueeze_1158, unsqueeze_1170, unsqueeze_1182, unsqueeze_1194, unsqueeze_1206, unsqueeze_1218, unsqueeze_1230, unsqueeze_1242, unsqueeze_1254, unsqueeze_1266, unsqueeze_1278, unsqueeze_1290, unsqueeze_1302, unsqueeze_1314, unsqueeze_1326, unsqueeze_1338, unsqueeze_1350, unsqueeze_1362, unsqueeze_1374, unsqueeze_1386, unsqueeze_1398, unsqueeze_1410, unsqueeze_1422, unsqueeze_1434, unsqueeze_1446, unsqueeze_1458, unsqueeze_1470, unsqueeze_1482, unsqueeze_1494, unsqueeze_1506, unsqueeze_1518, unsqueeze_1530, unsqueeze_1542, unsqueeze_1554, unsqueeze_1566, unsqueeze_1578, unsqueeze_1590, unsqueeze_1602, unsqueeze_1614, unsqueeze_1626, unsqueeze_1638, unsqueeze_1650, unsqueeze_1662, unsqueeze_1674, unsqueeze_1686, unsqueeze_1698, unsqueeze_1710, unsqueeze_1722, unsqueeze_1734, unsqueeze_1746, unsqueeze_1758, unsqueeze_1770, unsqueeze_1782, unsqueeze_1794, unsqueeze_1806, unsqueeze_1818, unsqueeze_1830, unsqueeze_1842, unsqueeze_1854, unsqueeze_1866, unsqueeze_1878, unsqueeze_1890, unsqueeze_1902, unsqueeze_1914)
