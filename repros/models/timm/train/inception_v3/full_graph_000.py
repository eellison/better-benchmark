class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3][27, 1, 9, 3]cuda:0", primals_2: "f32[128, 3, 299, 299][268203, 1, 897, 3]cuda:0", primals_3: "i64[][]cuda:0", primals_4: "f32[32][1]cuda:0", primals_5: "f32[32][1]cuda:0", primals_6: "f32[32][1]cuda:0", primals_7: "f32[32][1]cuda:0", primals_8: "f32[32, 32, 3, 3][288, 1, 96, 32]cuda:0", primals_9: "i64[][]cuda:0", primals_10: "f32[32][1]cuda:0", primals_11: "f32[32][1]cuda:0", primals_12: "f32[32][1]cuda:0", primals_13: "f32[32][1]cuda:0", primals_14: "f32[64, 32, 3, 3][288, 1, 96, 32]cuda:0", primals_15: "i64[][]cuda:0", primals_16: "f32[64][1]cuda:0", primals_17: "f32[64][1]cuda:0", primals_18: "f32[64][1]cuda:0", primals_19: "f32[64][1]cuda:0", primals_20: "f32[80, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_21: "i64[][]cuda:0", primals_22: "f32[80][1]cuda:0", primals_23: "f32[80][1]cuda:0", primals_24: "f32[80][1]cuda:0", primals_25: "f32[80][1]cuda:0", primals_26: "f32[192, 80, 3, 3][720, 1, 240, 80]cuda:0", primals_27: "i64[][]cuda:0", primals_28: "f32[192][1]cuda:0", primals_29: "f32[192][1]cuda:0", primals_30: "f32[192][1]cuda:0", primals_31: "f32[192][1]cuda:0", primals_32: "f32[64, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_33: "i64[][]cuda:0", primals_34: "f32[64][1]cuda:0", primals_35: "f32[64][1]cuda:0", primals_36: "f32[64][1]cuda:0", primals_37: "f32[64][1]cuda:0", primals_38: "f32[48, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_39: "i64[][]cuda:0", primals_40: "f32[48][1]cuda:0", primals_41: "f32[48][1]cuda:0", primals_42: "f32[48][1]cuda:0", primals_43: "f32[48][1]cuda:0", primals_44: "f32[64, 48, 5, 5][1200, 1, 240, 48]cuda:0", primals_45: "i64[][]cuda:0", primals_46: "f32[64][1]cuda:0", primals_47: "f32[64][1]cuda:0", primals_48: "f32[64][1]cuda:0", primals_49: "f32[64][1]cuda:0", primals_50: "f32[64, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_51: "i64[][]cuda:0", primals_52: "f32[64][1]cuda:0", primals_53: "f32[64][1]cuda:0", primals_54: "f32[64][1]cuda:0", primals_55: "f32[64][1]cuda:0", primals_56: "f32[96, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_57: "i64[][]cuda:0", primals_58: "f32[96][1]cuda:0", primals_59: "f32[96][1]cuda:0", primals_60: "f32[96][1]cuda:0", primals_61: "f32[96][1]cuda:0", primals_62: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0", primals_63: "i64[][]cuda:0", primals_64: "f32[96][1]cuda:0", primals_65: "f32[96][1]cuda:0", primals_66: "f32[96][1]cuda:0", primals_67: "f32[96][1]cuda:0", primals_68: "f32[32, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_69: "i64[][]cuda:0", primals_70: "f32[32][1]cuda:0", primals_71: "f32[32][1]cuda:0", primals_72: "f32[32][1]cuda:0", primals_73: "f32[32][1]cuda:0", primals_74: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_75: "i64[][]cuda:0", primals_76: "f32[64][1]cuda:0", primals_77: "f32[64][1]cuda:0", primals_78: "f32[64][1]cuda:0", primals_79: "f32[64][1]cuda:0", primals_80: "f32[48, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_81: "i64[][]cuda:0", primals_82: "f32[48][1]cuda:0", primals_83: "f32[48][1]cuda:0", primals_84: "f32[48][1]cuda:0", primals_85: "f32[48][1]cuda:0", primals_86: "f32[64, 48, 5, 5][1200, 1, 240, 48]cuda:0", primals_87: "i64[][]cuda:0", primals_88: "f32[64][1]cuda:0", primals_89: "f32[64][1]cuda:0", primals_90: "f32[64][1]cuda:0", primals_91: "f32[64][1]cuda:0", primals_92: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_93: "i64[][]cuda:0", primals_94: "f32[64][1]cuda:0", primals_95: "f32[64][1]cuda:0", primals_96: "f32[64][1]cuda:0", primals_97: "f32[64][1]cuda:0", primals_98: "f32[96, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_99: "i64[][]cuda:0", primals_100: "f32[96][1]cuda:0", primals_101: "f32[96][1]cuda:0", primals_102: "f32[96][1]cuda:0", primals_103: "f32[96][1]cuda:0", primals_104: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0", primals_105: "i64[][]cuda:0", primals_106: "f32[96][1]cuda:0", primals_107: "f32[96][1]cuda:0", primals_108: "f32[96][1]cuda:0", primals_109: "f32[96][1]cuda:0", primals_110: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_111: "i64[][]cuda:0", primals_112: "f32[64][1]cuda:0", primals_113: "f32[64][1]cuda:0", primals_114: "f32[64][1]cuda:0", primals_115: "f32[64][1]cuda:0", primals_116: "f32[64, 288, 1, 1][288, 1, 288, 288]cuda:0", primals_117: "i64[][]cuda:0", primals_118: "f32[64][1]cuda:0", primals_119: "f32[64][1]cuda:0", primals_120: "f32[64][1]cuda:0", primals_121: "f32[64][1]cuda:0", primals_122: "f32[48, 288, 1, 1][288, 1, 288, 288]cuda:0", primals_123: "i64[][]cuda:0", primals_124: "f32[48][1]cuda:0", primals_125: "f32[48][1]cuda:0", primals_126: "f32[48][1]cuda:0", primals_127: "f32[48][1]cuda:0", primals_128: "f32[64, 48, 5, 5][1200, 1, 240, 48]cuda:0", primals_129: "i64[][]cuda:0", primals_130: "f32[64][1]cuda:0", primals_131: "f32[64][1]cuda:0", primals_132: "f32[64][1]cuda:0", primals_133: "f32[64][1]cuda:0", primals_134: "f32[64, 288, 1, 1][288, 1, 288, 288]cuda:0", primals_135: "i64[][]cuda:0", primals_136: "f32[64][1]cuda:0", primals_137: "f32[64][1]cuda:0", primals_138: "f32[64][1]cuda:0", primals_139: "f32[64][1]cuda:0", primals_140: "f32[96, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_141: "i64[][]cuda:0", primals_142: "f32[96][1]cuda:0", primals_143: "f32[96][1]cuda:0", primals_144: "f32[96][1]cuda:0", primals_145: "f32[96][1]cuda:0", primals_146: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0", primals_147: "i64[][]cuda:0", primals_148: "f32[96][1]cuda:0", primals_149: "f32[96][1]cuda:0", primals_150: "f32[96][1]cuda:0", primals_151: "f32[96][1]cuda:0", primals_152: "f32[64, 288, 1, 1][288, 1, 288, 288]cuda:0", primals_153: "i64[][]cuda:0", primals_154: "f32[64][1]cuda:0", primals_155: "f32[64][1]cuda:0", primals_156: "f32[64][1]cuda:0", primals_157: "f32[64][1]cuda:0", primals_158: "f32[384, 288, 3, 3][2592, 1, 864, 288]cuda:0", primals_159: "i64[][]cuda:0", primals_160: "f32[384][1]cuda:0", primals_161: "f32[384][1]cuda:0", primals_162: "f32[384][1]cuda:0", primals_163: "f32[384][1]cuda:0", primals_164: "f32[64, 288, 1, 1][288, 1, 288, 288]cuda:0", primals_165: "i64[][]cuda:0", primals_166: "f32[64][1]cuda:0", primals_167: "f32[64][1]cuda:0", primals_168: "f32[64][1]cuda:0", primals_169: "f32[64][1]cuda:0", primals_170: "f32[96, 64, 3, 3][576, 1, 192, 64]cuda:0", primals_171: "i64[][]cuda:0", primals_172: "f32[96][1]cuda:0", primals_173: "f32[96][1]cuda:0", primals_174: "f32[96][1]cuda:0", primals_175: "f32[96][1]cuda:0", primals_176: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0", primals_177: "i64[][]cuda:0", primals_178: "f32[96][1]cuda:0", primals_179: "f32[96][1]cuda:0", primals_180: "f32[96][1]cuda:0", primals_181: "f32[96][1]cuda:0", primals_182: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_183: "i64[][]cuda:0", primals_184: "f32[192][1]cuda:0", primals_185: "f32[192][1]cuda:0", primals_186: "f32[192][1]cuda:0", primals_187: "f32[192][1]cuda:0", primals_188: "f32[128, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_189: "i64[][]cuda:0", primals_190: "f32[128][1]cuda:0", primals_191: "f32[128][1]cuda:0", primals_192: "f32[128][1]cuda:0", primals_193: "f32[128][1]cuda:0", primals_194: "f32[128, 128, 1, 7][896, 1, 896, 128]cuda:0", primals_195: "i64[][]cuda:0", primals_196: "f32[128][1]cuda:0", primals_197: "f32[128][1]cuda:0", primals_198: "f32[128][1]cuda:0", primals_199: "f32[128][1]cuda:0", primals_200: "f32[192, 128, 7, 1][896, 1, 128, 128]cuda:0", primals_201: "i64[][]cuda:0", primals_202: "f32[192][1]cuda:0", primals_203: "f32[192][1]cuda:0", primals_204: "f32[192][1]cuda:0", primals_205: "f32[192][1]cuda:0", primals_206: "f32[128, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_207: "i64[][]cuda:0", primals_208: "f32[128][1]cuda:0", primals_209: "f32[128][1]cuda:0", primals_210: "f32[128][1]cuda:0", primals_211: "f32[128][1]cuda:0", primals_212: "f32[128, 128, 7, 1][896, 1, 128, 128]cuda:0", primals_213: "i64[][]cuda:0", primals_214: "f32[128][1]cuda:0", primals_215: "f32[128][1]cuda:0", primals_216: "f32[128][1]cuda:0", primals_217: "f32[128][1]cuda:0", primals_218: "f32[128, 128, 1, 7][896, 1, 896, 128]cuda:0", primals_219: "i64[][]cuda:0", primals_220: "f32[128][1]cuda:0", primals_221: "f32[128][1]cuda:0", primals_222: "f32[128][1]cuda:0", primals_223: "f32[128][1]cuda:0", primals_224: "f32[128, 128, 7, 1][896, 1, 128, 128]cuda:0", primals_225: "i64[][]cuda:0", primals_226: "f32[128][1]cuda:0", primals_227: "f32[128][1]cuda:0", primals_228: "f32[128][1]cuda:0", primals_229: "f32[128][1]cuda:0", primals_230: "f32[192, 128, 1, 7][896, 1, 896, 128]cuda:0", primals_231: "i64[][]cuda:0", primals_232: "f32[192][1]cuda:0", primals_233: "f32[192][1]cuda:0", primals_234: "f32[192][1]cuda:0", primals_235: "f32[192][1]cuda:0", primals_236: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_237: "i64[][]cuda:0", primals_238: "f32[192][1]cuda:0", primals_239: "f32[192][1]cuda:0", primals_240: "f32[192][1]cuda:0", primals_241: "f32[192][1]cuda:0", primals_242: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_243: "i64[][]cuda:0", primals_244: "f32[192][1]cuda:0", primals_245: "f32[192][1]cuda:0", primals_246: "f32[192][1]cuda:0", primals_247: "f32[192][1]cuda:0", primals_248: "f32[160, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_249: "i64[][]cuda:0", primals_250: "f32[160][1]cuda:0", primals_251: "f32[160][1]cuda:0", primals_252: "f32[160][1]cuda:0", primals_253: "f32[160][1]cuda:0", primals_254: "f32[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0", primals_255: "i64[][]cuda:0", primals_256: "f32[160][1]cuda:0", primals_257: "f32[160][1]cuda:0", primals_258: "f32[160][1]cuda:0", primals_259: "f32[160][1]cuda:0", primals_260: "f32[192, 160, 7, 1][1120, 1, 160, 160]cuda:0", primals_261: "i64[][]cuda:0", primals_262: "f32[192][1]cuda:0", primals_263: "f32[192][1]cuda:0", primals_264: "f32[192][1]cuda:0", primals_265: "f32[192][1]cuda:0", primals_266: "f32[160, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_267: "i64[][]cuda:0", primals_268: "f32[160][1]cuda:0", primals_269: "f32[160][1]cuda:0", primals_270: "f32[160][1]cuda:0", primals_271: "f32[160][1]cuda:0", primals_272: "f32[160, 160, 7, 1][1120, 1, 160, 160]cuda:0", primals_273: "i64[][]cuda:0", primals_274: "f32[160][1]cuda:0", primals_275: "f32[160][1]cuda:0", primals_276: "f32[160][1]cuda:0", primals_277: "f32[160][1]cuda:0", primals_278: "f32[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0", primals_279: "i64[][]cuda:0", primals_280: "f32[160][1]cuda:0", primals_281: "f32[160][1]cuda:0", primals_282: "f32[160][1]cuda:0", primals_283: "f32[160][1]cuda:0", primals_284: "f32[160, 160, 7, 1][1120, 1, 160, 160]cuda:0", primals_285: "i64[][]cuda:0", primals_286: "f32[160][1]cuda:0", primals_287: "f32[160][1]cuda:0", primals_288: "f32[160][1]cuda:0", primals_289: "f32[160][1]cuda:0", primals_290: "f32[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0", primals_291: "i64[][]cuda:0", primals_292: "f32[192][1]cuda:0", primals_293: "f32[192][1]cuda:0", primals_294: "f32[192][1]cuda:0", primals_295: "f32[192][1]cuda:0", primals_296: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_297: "i64[][]cuda:0", primals_298: "f32[192][1]cuda:0", primals_299: "f32[192][1]cuda:0", primals_300: "f32[192][1]cuda:0", primals_301: "f32[192][1]cuda:0", primals_302: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_303: "i64[][]cuda:0", primals_304: "f32[192][1]cuda:0", primals_305: "f32[192][1]cuda:0", primals_306: "f32[192][1]cuda:0", primals_307: "f32[192][1]cuda:0", primals_308: "f32[160, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_309: "i64[][]cuda:0", primals_310: "f32[160][1]cuda:0", primals_311: "f32[160][1]cuda:0", primals_312: "f32[160][1]cuda:0", primals_313: "f32[160][1]cuda:0", primals_314: "f32[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0", primals_315: "i64[][]cuda:0", primals_316: "f32[160][1]cuda:0", primals_317: "f32[160][1]cuda:0", primals_318: "f32[160][1]cuda:0", primals_319: "f32[160][1]cuda:0", primals_320: "f32[192, 160, 7, 1][1120, 1, 160, 160]cuda:0", primals_321: "i64[][]cuda:0", primals_322: "f32[192][1]cuda:0", primals_323: "f32[192][1]cuda:0", primals_324: "f32[192][1]cuda:0", primals_325: "f32[192][1]cuda:0", primals_326: "f32[160, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_327: "i64[][]cuda:0", primals_328: "f32[160][1]cuda:0", primals_329: "f32[160][1]cuda:0", primals_330: "f32[160][1]cuda:0", primals_331: "f32[160][1]cuda:0", primals_332: "f32[160, 160, 7, 1][1120, 1, 160, 160]cuda:0", primals_333: "i64[][]cuda:0", primals_334: "f32[160][1]cuda:0", primals_335: "f32[160][1]cuda:0", primals_336: "f32[160][1]cuda:0", primals_337: "f32[160][1]cuda:0", primals_338: "f32[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0", primals_339: "i64[][]cuda:0", primals_340: "f32[160][1]cuda:0", primals_341: "f32[160][1]cuda:0", primals_342: "f32[160][1]cuda:0", primals_343: "f32[160][1]cuda:0", primals_344: "f32[160, 160, 7, 1][1120, 1, 160, 160]cuda:0", primals_345: "i64[][]cuda:0", primals_346: "f32[160][1]cuda:0", primals_347: "f32[160][1]cuda:0", primals_348: "f32[160][1]cuda:0", primals_349: "f32[160][1]cuda:0", primals_350: "f32[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0", primals_351: "i64[][]cuda:0", primals_352: "f32[192][1]cuda:0", primals_353: "f32[192][1]cuda:0", primals_354: "f32[192][1]cuda:0", primals_355: "f32[192][1]cuda:0", primals_356: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_357: "i64[][]cuda:0", primals_358: "f32[192][1]cuda:0", primals_359: "f32[192][1]cuda:0", primals_360: "f32[192][1]cuda:0", primals_361: "f32[192][1]cuda:0", primals_362: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_363: "i64[][]cuda:0", primals_364: "f32[192][1]cuda:0", primals_365: "f32[192][1]cuda:0", primals_366: "f32[192][1]cuda:0", primals_367: "f32[192][1]cuda:0", primals_368: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_369: "i64[][]cuda:0", primals_370: "f32[192][1]cuda:0", primals_371: "f32[192][1]cuda:0", primals_372: "f32[192][1]cuda:0", primals_373: "f32[192][1]cuda:0", primals_374: "f32[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0", primals_375: "i64[][]cuda:0", primals_376: "f32[192][1]cuda:0", primals_377: "f32[192][1]cuda:0", primals_378: "f32[192][1]cuda:0", primals_379: "f32[192][1]cuda:0", primals_380: "f32[192, 192, 7, 1][1344, 1, 192, 192]cuda:0", primals_381: "i64[][]cuda:0", primals_382: "f32[192][1]cuda:0", primals_383: "f32[192][1]cuda:0", primals_384: "f32[192][1]cuda:0", primals_385: "f32[192][1]cuda:0", primals_386: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_387: "i64[][]cuda:0", primals_388: "f32[192][1]cuda:0", primals_389: "f32[192][1]cuda:0", primals_390: "f32[192][1]cuda:0", primals_391: "f32[192][1]cuda:0", primals_392: "f32[192, 192, 7, 1][1344, 1, 192, 192]cuda:0", primals_393: "i64[][]cuda:0", primals_394: "f32[192][1]cuda:0", primals_395: "f32[192][1]cuda:0", primals_396: "f32[192][1]cuda:0", primals_397: "f32[192][1]cuda:0", primals_398: "f32[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0", primals_399: "i64[][]cuda:0", primals_400: "f32[192][1]cuda:0", primals_401: "f32[192][1]cuda:0", primals_402: "f32[192][1]cuda:0", primals_403: "f32[192][1]cuda:0", primals_404: "f32[192, 192, 7, 1][1344, 1, 192, 192]cuda:0", primals_405: "i64[][]cuda:0", primals_406: "f32[192][1]cuda:0", primals_407: "f32[192][1]cuda:0", primals_408: "f32[192][1]cuda:0", primals_409: "f32[192][1]cuda:0", primals_410: "f32[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0", primals_411: "i64[][]cuda:0", primals_412: "f32[192][1]cuda:0", primals_413: "f32[192][1]cuda:0", primals_414: "f32[192][1]cuda:0", primals_415: "f32[192][1]cuda:0", primals_416: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_417: "i64[][]cuda:0", primals_418: "f32[192][1]cuda:0", primals_419: "f32[192][1]cuda:0", primals_420: "f32[192][1]cuda:0", primals_421: "f32[192][1]cuda:0", primals_422: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_423: "i64[][]cuda:0", primals_424: "f32[192][1]cuda:0", primals_425: "f32[192][1]cuda:0", primals_426: "f32[192][1]cuda:0", primals_427: "f32[192][1]cuda:0", primals_428: "f32[320, 192, 3, 3][1728, 1, 576, 192]cuda:0", primals_429: "i64[][]cuda:0", primals_430: "f32[320][1]cuda:0", primals_431: "f32[320][1]cuda:0", primals_432: "f32[320][1]cuda:0", primals_433: "f32[320][1]cuda:0", primals_434: "f32[192, 768, 1, 1][768, 1, 768, 768]cuda:0", primals_435: "i64[][]cuda:0", primals_436: "f32[192][1]cuda:0", primals_437: "f32[192][1]cuda:0", primals_438: "f32[192][1]cuda:0", primals_439: "f32[192][1]cuda:0", primals_440: "f32[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0", primals_441: "i64[][]cuda:0", primals_442: "f32[192][1]cuda:0", primals_443: "f32[192][1]cuda:0", primals_444: "f32[192][1]cuda:0", primals_445: "f32[192][1]cuda:0", primals_446: "f32[192, 192, 7, 1][1344, 1, 192, 192]cuda:0", primals_447: "i64[][]cuda:0", primals_448: "f32[192][1]cuda:0", primals_449: "f32[192][1]cuda:0", primals_450: "f32[192][1]cuda:0", primals_451: "f32[192][1]cuda:0", primals_452: "f32[192, 192, 3, 3][1728, 1, 576, 192]cuda:0", primals_453: "i64[][]cuda:0", primals_454: "f32[192][1]cuda:0", primals_455: "f32[192][1]cuda:0", primals_456: "f32[192][1]cuda:0", primals_457: "f32[192][1]cuda:0", primals_458: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_459: "i64[][]cuda:0", primals_460: "f32[320][1]cuda:0", primals_461: "f32[320][1]cuda:0", primals_462: "f32[320][1]cuda:0", primals_463: "f32[320][1]cuda:0", primals_464: "f32[384, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_465: "i64[][]cuda:0", primals_466: "f32[384][1]cuda:0", primals_467: "f32[384][1]cuda:0", primals_468: "f32[384][1]cuda:0", primals_469: "f32[384][1]cuda:0", primals_470: "f32[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0", primals_471: "i64[][]cuda:0", primals_472: "f32[384][1]cuda:0", primals_473: "f32[384][1]cuda:0", primals_474: "f32[384][1]cuda:0", primals_475: "f32[384][1]cuda:0", primals_476: "f32[384, 384, 3, 1][1152, 1, 384, 384]cuda:0", primals_477: "i64[][]cuda:0", primals_478: "f32[384][1]cuda:0", primals_479: "f32[384][1]cuda:0", primals_480: "f32[384][1]cuda:0", primals_481: "f32[384][1]cuda:0", primals_482: "f32[448, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_483: "i64[][]cuda:0", primals_484: "f32[448][1]cuda:0", primals_485: "f32[448][1]cuda:0", primals_486: "f32[448][1]cuda:0", primals_487: "f32[448][1]cuda:0", primals_488: "f32[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0", primals_489: "i64[][]cuda:0", primals_490: "f32[384][1]cuda:0", primals_491: "f32[384][1]cuda:0", primals_492: "f32[384][1]cuda:0", primals_493: "f32[384][1]cuda:0", primals_494: "f32[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0", primals_495: "i64[][]cuda:0", primals_496: "f32[384][1]cuda:0", primals_497: "f32[384][1]cuda:0", primals_498: "f32[384][1]cuda:0", primals_499: "f32[384][1]cuda:0", primals_500: "f32[384, 384, 3, 1][1152, 1, 384, 384]cuda:0", primals_501: "i64[][]cuda:0", primals_502: "f32[384][1]cuda:0", primals_503: "f32[384][1]cuda:0", primals_504: "f32[384][1]cuda:0", primals_505: "f32[384][1]cuda:0", primals_506: "f32[192, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_507: "i64[][]cuda:0", primals_508: "f32[192][1]cuda:0", primals_509: "f32[192][1]cuda:0", primals_510: "f32[192][1]cuda:0", primals_511: "f32[192][1]cuda:0", primals_512: "f32[320, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", primals_513: "i64[][]cuda:0", primals_514: "f32[320][1]cuda:0", primals_515: "f32[320][1]cuda:0", primals_516: "f32[320][1]cuda:0", primals_517: "f32[320][1]cuda:0", primals_518: "f32[384, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", primals_519: "i64[][]cuda:0", primals_520: "f32[384][1]cuda:0", primals_521: "f32[384][1]cuda:0", primals_522: "f32[384][1]cuda:0", primals_523: "f32[384][1]cuda:0", primals_524: "f32[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0", primals_525: "i64[][]cuda:0", primals_526: "f32[384][1]cuda:0", primals_527: "f32[384][1]cuda:0", primals_528: "f32[384][1]cuda:0", primals_529: "f32[384][1]cuda:0", primals_530: "f32[384, 384, 3, 1][1152, 1, 384, 384]cuda:0", primals_531: "i64[][]cuda:0", primals_532: "f32[384][1]cuda:0", primals_533: "f32[384][1]cuda:0", primals_534: "f32[384][1]cuda:0", primals_535: "f32[384][1]cuda:0", primals_536: "f32[448, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", primals_537: "i64[][]cuda:0", primals_538: "f32[448][1]cuda:0", primals_539: "f32[448][1]cuda:0", primals_540: "f32[448][1]cuda:0", primals_541: "f32[448][1]cuda:0", primals_542: "f32[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0", primals_543: "i64[][]cuda:0", primals_544: "f32[384][1]cuda:0", primals_545: "f32[384][1]cuda:0", primals_546: "f32[384][1]cuda:0", primals_547: "f32[384][1]cuda:0", primals_548: "f32[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0", primals_549: "i64[][]cuda:0", primals_550: "f32[384][1]cuda:0", primals_551: "f32[384][1]cuda:0", primals_552: "f32[384][1]cuda:0", primals_553: "f32[384][1]cuda:0", primals_554: "f32[384, 384, 3, 1][1152, 1, 384, 384]cuda:0", primals_555: "i64[][]cuda:0", primals_556: "f32[384][1]cuda:0", primals_557: "f32[384][1]cuda:0", primals_558: "f32[384][1]cuda:0", primals_559: "f32[384][1]cuda:0", primals_560: "f32[192, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0", primals_561: "i64[][]cuda:0", primals_562: "f32[192][1]cuda:0", primals_563: "f32[192][1]cuda:0", primals_564: "f32[192][1]cuda:0", primals_565: "f32[192][1]cuda:0", primals_566: "f32[1000, 2048][2048, 1]cuda:0", primals_567: "f32[1000][1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type: "bf16[32, 3, 3, 3][27, 1, 9, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_1, torch.bfloat16);  primals_1 = None
        convert_element_type_1: "bf16[128, 3, 299, 299][268203, 1, 897, 3]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convolution: "bf16[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.convolution.default(convert_element_type_1, convert_element_type, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_2: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_2, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_2 = None
        getitem: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_1: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 0.001)
        rsqrt: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000003518986869);  squeeze_2 = None
        mul_4: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu: "bf16[128, 32, 149, 149][710432, 1, 4768, 32]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_4: "bf16[32, 32, 3, 3][288, 1, 96, 32]cuda:0" = torch.ops.prims.convert_element_type.default(primals_8, torch.bfloat16);  primals_8 = None
        convolution_1: "bf16[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.convolution.default(relu, convert_element_type_4, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_5: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_5: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_5, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_5 = None
        getitem_2: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 0.001)
        rsqrt_1: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000003615393043);  squeeze_5 = None
        mul_11: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None
        convert_element_type_6: "bf16[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_9, torch.bfloat16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_1: "bf16[128, 32, 147, 147][691488, 1, 4704, 32]cuda:0" = torch.ops.aten.relu.default(convert_element_type_6);  convert_element_type_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_7: "bf16[64, 32, 3, 3][288, 1, 96, 32]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        convolution_2: "bf16[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.convolution.default(relu_1, convert_element_type_7, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_10: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_15, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_8: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_8, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_8 = None
        getitem_4: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 0.001)
        rsqrt_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3])
        mul_15: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_8, 1.0000003615393043);  squeeze_8 = None
        mul_18: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1)
        unsqueeze_11: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        convert_element_type_9: "bf16[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_2: "bf16[128, 64, 147, 147][1382976, 1, 9408, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_9);  convert_element_type_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:396 in forward_preaux, code: x = self.Pool1(x)  # N x 64 x 73 x 73
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_2, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_2 = None
        getitem_6: "bf16[128, 64, 73, 73][341056, 1, 4672, 64]cuda:0" = _low_memory_max_pool_with_offsets[0]
        getitem_7: "i8[128, 64, 73, 73][341056, 1, 4672, 64]cuda:0" = _low_memory_max_pool_with_offsets[1];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_10: "bf16[80, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        convolution_3: "bf16[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.convolution.default(getitem_6, convert_element_type_10, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_15: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_21, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_11: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_11, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_11 = None
        getitem_8: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_3[0]
        getitem_9: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 0.001)
        rsqrt_3: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, getitem_9)
        mul_21: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_10: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_22: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_23: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_17: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
        squeeze_11: "f32[80][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_24: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_11, 1.0000014660370526);  squeeze_11 = None
        mul_25: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, 0.1);  mul_24 = None
        mul_26: "f32[80][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_18: "f32[80][1]cuda:0" = torch.ops.aten.add.Tensor(mul_25, mul_26);  mul_25 = mul_26 = None
        unsqueeze_12: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_24, -1)
        unsqueeze_13: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_27: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, unsqueeze_13);  mul_21 = unsqueeze_13 = None
        unsqueeze_14: "f32[80, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_25, -1);  primals_25 = None
        unsqueeze_15: "f32[80, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.add.Tensor(mul_27, unsqueeze_15);  mul_27 = unsqueeze_15 = None
        convert_element_type_12: "bf16[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.prims.convert_element_type.default(add_19, torch.bfloat16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_3: "bf16[128, 80, 73, 73][426320, 1, 5840, 80]cuda:0" = torch.ops.aten.relu.default(convert_element_type_12);  convert_element_type_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_13: "bf16[192, 80, 3, 3][720, 1, 240, 80]cuda:0" = torch.ops.prims.convert_element_type.default(primals_26, torch.bfloat16);  primals_26 = None
        convolution_4: "bf16[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.convolution.default(relu_3, convert_element_type_13, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_20: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_27, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_14: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_14, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_14 = None
        getitem_10: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_4[0]
        getitem_11: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 0.001)
        rsqrt_4: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_11)
        mul_28: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3])
        mul_29: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1);  squeeze_12 = None
        mul_30: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_22: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_29, mul_30);  mul_29 = mul_30 = None
        squeeze_14: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_31: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000154979411);  squeeze_14 = None
        mul_32: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, 0.1);  mul_31 = None
        mul_33: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_23: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
        unsqueeze_16: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1)
        unsqueeze_19: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None
        convert_element_type_15: "bf16[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_4: "bf16[128, 192, 71, 71][967872, 1, 13632, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_15);  convert_element_type_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:399 in forward_preaux, code: x = self.Pool2(x)  # N x 192 x 35 x 35
        _low_memory_max_pool_with_offsets_1 = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_4, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_4 = None
        getitem_12: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = _low_memory_max_pool_with_offsets_1[0]
        getitem_13: "i8[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = _low_memory_max_pool_with_offsets_1[1];  _low_memory_max_pool_with_offsets_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_16: "bf16[64, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_32, torch.bfloat16);  primals_32 = None
        convolution_5: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(getitem_12, convert_element_type_16, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_25: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_33, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_17: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_17, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_17 = None
        getitem_14: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_5[0]
        getitem_15: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_26: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 0.001)
        rsqrt_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
        sub_5: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_15)
        mul_35: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3])
        mul_36: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1);  squeeze_15 = None
        mul_37: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_27: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_36, mul_37);  mul_36 = mul_37 = None
        squeeze_17: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_38: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_17, 1.0000063775916939);  squeeze_17 = None
        mul_39: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, 0.1);  mul_38 = None
        mul_40: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_28: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        unsqueeze_20: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1)
        unsqueeze_23: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        convert_element_type_18: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_5: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_18);  convert_element_type_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_19: "bf16[48, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_38, torch.bfloat16);  primals_38 = None
        convolution_6: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.convolution.default(getitem_12, convert_element_type_19, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_30: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_39, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_20: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_20, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_20 = None
        getitem_16: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = var_mean_6[0]
        getitem_17: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_31: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 0.001)
        rsqrt_6: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_6: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, getitem_17)
        mul_42: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[48][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_19: "f32[48][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_43: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1)
        mul_44: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_32: "f32[48][1]cuda:0" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        squeeze_20: "f32[48][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_45: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_20, 1.0000063775916939);  squeeze_20 = None
        mul_46: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, 0.1);  mul_45 = None
        mul_47: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_33: "f32[48][1]cuda:0" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
        unsqueeze_24: "f32[48, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_42, -1)
        unsqueeze_25: "f32[48, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_48: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, unsqueeze_25);  mul_42 = unsqueeze_25 = None
        unsqueeze_26: "f32[48, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_43, -1);  primals_43 = None
        unsqueeze_27: "f32[48, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_34: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.add.Tensor(mul_48, unsqueeze_27);  mul_48 = unsqueeze_27 = None
        convert_element_type_21: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_6: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.relu.default(convert_element_type_21);  convert_element_type_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_22: "bf16[64, 48, 5, 5][1200, 1, 240, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        convolution_7: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(relu_6, convert_element_type_22, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_35: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_45, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_23: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_23, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_23 = None
        getitem_18: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_7[0]
        getitem_19: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_36: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 0.001)
        rsqrt_7: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        sub_7: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_19)
        mul_49: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3])
        mul_50: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1);  squeeze_21 = None
        mul_51: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_37: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_50, mul_51);  mul_50 = mul_51 = None
        squeeze_23: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_52: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_23, 1.0000063775916939);  squeeze_23 = None
        mul_53: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, 0.1);  mul_52 = None
        mul_54: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_38: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
        unsqueeze_28: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1)
        unsqueeze_31: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None
        convert_element_type_24: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_7: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_24);  convert_element_type_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_25: "bf16[64, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        convolution_8: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(getitem_12, convert_element_type_25, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_40: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_51, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_26: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_26, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_26 = None
        getitem_20: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_8[0]
        getitem_21: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_41: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 0.001)
        rsqrt_8: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        sub_8: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_21)
        mul_56: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_25: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_57: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_58: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_42: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_57, mul_58);  mul_57 = mul_58 = None
        squeeze_26: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_59: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_26, 1.0000063775916939);  squeeze_26 = None
        mul_60: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, 0.1);  mul_59 = None
        mul_61: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_43: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, mul_61);  mul_60 = mul_61 = None
        unsqueeze_32: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_44: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        convert_element_type_27: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_8: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_27);  convert_element_type_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_28: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_56, torch.bfloat16);  primals_56 = None
        convolution_9: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.convolution.default(relu_8, convert_element_type_28, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_45: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_57, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_29: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_29, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_29 = None
        getitem_22: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_9[0]
        getitem_23: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_46: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 0.001)
        rsqrt_9: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        sub_9: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, getitem_23)
        mul_63: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_28: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_64: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1)
        mul_65: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_47: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        squeeze_29: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_66: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000063775916939);  squeeze_29 = None
        mul_67: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 0.1);  mul_66 = None
        mul_68: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_48: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, mul_68);  mul_67 = mul_68 = None
        unsqueeze_36: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_60, -1)
        unsqueeze_37: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_69: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, unsqueeze_37);  mul_63 = unsqueeze_37 = None
        unsqueeze_38: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_61, -1);  primals_61 = None
        unsqueeze_39: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_49: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_69, unsqueeze_39);  mul_69 = unsqueeze_39 = None
        convert_element_type_30: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_49, torch.bfloat16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_9: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_30);  convert_element_type_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_31: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(primals_62, torch.bfloat16);  primals_62 = None
        convolution_10: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.convolution.default(relu_9, convert_element_type_31, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_50: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_63, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_32: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_32, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_32 = None
        getitem_24: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_10[0]
        getitem_25: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_51: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 0.001)
        rsqrt_10: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
        sub_10: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_25)
        mul_70: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3])
        mul_71: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1);  squeeze_30 = None
        mul_72: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_52: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_71, mul_72);  mul_71 = mul_72 = None
        squeeze_32: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_73: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000063775916939);  squeeze_32 = None
        mul_74: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 0.1);  mul_73 = None
        mul_75: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_53: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        unsqueeze_40: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1)
        unsqueeze_43: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_54: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None
        convert_element_type_33: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_10: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_33);  convert_element_type_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d: "bf16[128, 192, 35, 35][235200, 1, 6720, 192]cuda:0" = torch.ops.aten.avg_pool2d.default(getitem_12, [3, 3], [1, 1], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_34: "bf16[32, 192, 1, 1][192, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_68, torch.bfloat16);  primals_68 = None
        convolution_11: "bf16[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d, convert_element_type_34, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_55: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_69, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_35: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_35, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_35 = None
        getitem_26: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_11[0]
        getitem_27: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_56: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 0.001)
        rsqrt_11: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_11: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_27)
        mul_77: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3])
        mul_78: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1);  squeeze_33 = None
        mul_79: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_57: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
        squeeze_35: "f32[32][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_80: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000063775916939);  squeeze_35 = None
        mul_81: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 0.1);  mul_80 = None
        mul_82: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_58: "f32[32][1]cuda:0" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        unsqueeze_44: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1)
        unsqueeze_47: "f32[32, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_59: "f32[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        convert_element_type_36: "bf16[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16);  add_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_11: "bf16[128, 32, 35, 35][39200, 1, 1120, 32]cuda:0" = torch.ops.aten.relu.default(convert_element_type_36);  convert_element_type_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        cat: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.cat.default([relu_5, relu_7, relu_10, relu_11], 1);  relu_5 = relu_7 = relu_10 = relu_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_37: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        convolution_12: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(cat, convert_element_type_37, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_60: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_75, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_38: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_38, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_38 = None
        getitem_28: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_12[0]
        getitem_29: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_61: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 0.001)
        rsqrt_12: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
        sub_12: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, getitem_29)
        mul_84: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3])
        mul_85: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1);  squeeze_36 = None
        mul_86: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_62: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        squeeze_38: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_87: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000063775916939);  squeeze_38 = None
        mul_88: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 0.1);  mul_87 = None
        mul_89: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_63: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        unsqueeze_48: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_49: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_90: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, unsqueeze_49);  mul_84 = unsqueeze_49 = None
        unsqueeze_50: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_79, -1)
        unsqueeze_51: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_64: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_90, unsqueeze_51);  mul_90 = unsqueeze_51 = None
        convert_element_type_39: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_12: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_39);  convert_element_type_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_40: "bf16[48, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        convolution_13: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.convolution.default(cat, convert_element_type_40, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_65: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_81, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_41: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_41, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_41 = None
        getitem_30: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = var_mean_13[0]
        getitem_31: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_66: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 0.001)
        rsqrt_13: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_13: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_31)
        mul_91: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[48][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_40: "f32[48][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_92: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1)
        mul_93: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_67: "f32[48][1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
        squeeze_41: "f32[48][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_94: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000063775916939);  squeeze_41 = None
        mul_95: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, 0.1);  mul_94 = None
        mul_96: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_68: "f32[48][1]cuda:0" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        unsqueeze_52: "f32[48, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[48, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[48, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[48, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_69: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None
        convert_element_type_42: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16);  add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_13: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.relu.default(convert_element_type_42);  convert_element_type_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_43: "bf16[64, 48, 5, 5][1200, 1, 240, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_86, torch.bfloat16);  primals_86 = None
        convolution_14: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(relu_13, convert_element_type_43, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_70: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_87, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_44: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_44, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_44 = None
        getitem_32: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_14[0]
        getitem_33: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_71: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 0.001)
        rsqrt_14: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_14: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_33)
        mul_98: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3])
        mul_99: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1);  squeeze_42 = None
        mul_100: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_72: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        squeeze_44: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_101: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000063775916939);  squeeze_44 = None
        mul_102: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_101, 0.1);  mul_101 = None
        mul_103: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_73: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
        unsqueeze_56: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1)
        unsqueeze_59: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_74: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        convert_element_type_45: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_14: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_45);  convert_element_type_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_46: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(primals_92, torch.bfloat16);  primals_92 = None
        convolution_15: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(cat, convert_element_type_46, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_75: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_93, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_47: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_47, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_47 = None
        getitem_34: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_15[0]
        getitem_35: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_76: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 0.001)
        rsqrt_15: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
        sub_15: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, getitem_35)
        mul_105: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_46: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_106: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_107: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_77: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        squeeze_47: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_108: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000063775916939);  squeeze_47 = None
        mul_109: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_108, 0.1);  mul_108 = None
        mul_110: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_78: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_109, mul_110);  mul_109 = mul_110 = None
        unsqueeze_60: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_61: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_111: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, unsqueeze_61);  mul_105 = unsqueeze_61 = None
        unsqueeze_62: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_97, -1);  primals_97 = None
        unsqueeze_63: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_79: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_111, unsqueeze_63);  mul_111 = unsqueeze_63 = None
        convert_element_type_48: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_79, torch.bfloat16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_15: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_48);  convert_element_type_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_49: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_98, torch.bfloat16);  primals_98 = None
        convolution_16: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.convolution.default(relu_15, convert_element_type_49, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_80: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_99, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_50: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_50, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_50 = None
        getitem_36: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_16[0]
        getitem_37: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_81: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 0.001)
        rsqrt_16: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        sub_16: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_37)
        mul_112: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_49: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_113: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1)
        mul_114: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_82: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        squeeze_50: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_115: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000063775916939);  squeeze_50 = None
        mul_116: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_115, 0.1);  mul_115 = None
        mul_117: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_83: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        unsqueeze_64: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_84: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        convert_element_type_51: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_16: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_51);  convert_element_type_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_52: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        convolution_17: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.convolution.default(relu_16, convert_element_type_52, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_85: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_105, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_53: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_53, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_53 = None
        getitem_38: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_17[0]
        getitem_39: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_86: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 0.001)
        rsqrt_17: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_17: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_17, getitem_39)
        mul_119: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3])
        mul_120: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1);  squeeze_51 = None
        mul_121: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_87: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        squeeze_53: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_122: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000063775916939);  squeeze_53 = None
        mul_123: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_122, 0.1);  mul_122 = None
        mul_124: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_88: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
        unsqueeze_68: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_108, -1)
        unsqueeze_69: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_125: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, unsqueeze_69);  mul_119 = unsqueeze_69 = None
        unsqueeze_70: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_109, -1)
        unsqueeze_71: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_89: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_71);  mul_125 = unsqueeze_71 = None
        convert_element_type_54: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_17: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_54);  convert_element_type_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_1: "bf16[128, 256, 35, 35][313600, 1, 8960, 256]cuda:0" = torch.ops.aten.avg_pool2d.default(cat, [3, 3], [1, 1], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_55: "bf16[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        convolution_18: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_1, convert_element_type_55, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_90: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_111, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_56: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_56, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_56 = None
        getitem_40: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_18[0]
        getitem_41: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_91: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 0.001)
        rsqrt_18: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
        sub_18: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_41)
        mul_126: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3])
        mul_127: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1);  squeeze_54 = None
        mul_128: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_92: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
        squeeze_56: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_129: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000063775916939);  squeeze_56 = None
        mul_130: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, 0.1);  mul_129 = None
        mul_131: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_93: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_130, mul_131);  mul_130 = mul_131 = None
        unsqueeze_72: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_114, -1)
        unsqueeze_73: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_132: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, unsqueeze_73);  mul_126 = unsqueeze_73 = None
        unsqueeze_74: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_115, -1)
        unsqueeze_75: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_94: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_132, unsqueeze_75);  mul_132 = unsqueeze_75 = None
        convert_element_type_57: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_18: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_57);  convert_element_type_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        cat_1: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.cat.default([relu_12, relu_14, relu_17, relu_18], 1);  relu_12 = relu_14 = relu_17 = relu_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_58: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(primals_116, torch.bfloat16);  primals_116 = None
        convolution_19: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(cat_1, convert_element_type_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_95: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_117, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_59: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_59, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_59 = None
        getitem_42: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_19[0]
        getitem_43: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_96: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 0.001)
        rsqrt_19: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_19: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_43)
        mul_133: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3])
        mul_134: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_135: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_118, 0.9)
        add_97: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
        squeeze_59: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_136: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000063775916939);  squeeze_59 = None
        mul_137: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, 0.1);  mul_136 = None
        mul_138: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_119, 0.9)
        add_98: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_137, mul_138);  mul_137 = mul_138 = None
        unsqueeze_76: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_121, -1)
        unsqueeze_79: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_99: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        convert_element_type_60: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_19: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_60);  convert_element_type_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_61: "bf16[48, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(primals_122, torch.bfloat16);  primals_122 = None
        convolution_20: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.convolution.default(cat_1, convert_element_type_61, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_100: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_123, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_62: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_20, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_62, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_62 = None
        getitem_44: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = var_mean_20[0]
        getitem_45: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_101: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 0.001)
        rsqrt_20: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_20: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_45)
        mul_140: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[48][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_61: "f32[48][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_141: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_142: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_124, 0.9)
        add_102: "f32[48][1]cuda:0" = torch.ops.aten.add.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        squeeze_62: "f32[48][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_143: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0000063775916939);  squeeze_62 = None
        mul_144: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, 0.1);  mul_143 = None
        mul_145: "f32[48][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_125, 0.9)
        add_103: "f32[48][1]cuda:0" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
        unsqueeze_80: "f32[48, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_126, -1)
        unsqueeze_81: "f32[48, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_146: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.mul.Tensor(mul_140, unsqueeze_81);  mul_140 = unsqueeze_81 = None
        unsqueeze_82: "f32[48, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_127, -1);  primals_127 = None
        unsqueeze_83: "f32[48, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_104: "f32[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_83);  mul_146 = unsqueeze_83 = None
        convert_element_type_63: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_20: "bf16[128, 48, 35, 35][58800, 1, 1680, 48]cuda:0" = torch.ops.aten.relu.default(convert_element_type_63);  convert_element_type_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_64: "bf16[64, 48, 5, 5][1200, 1, 240, 48]cuda:0" = torch.ops.prims.convert_element_type.default(primals_128, torch.bfloat16);  primals_128 = None
        convolution_21: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(relu_20, convert_element_type_64, None, [1, 1], [2, 2], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_105: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_129, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_65: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_21, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_65, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_65 = None
        getitem_46: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_21[0]
        getitem_47: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_106: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 0.001)
        rsqrt_21: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_21: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_47)
        mul_147: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3])
        mul_148: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1);  squeeze_63 = None
        mul_149: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_130, 0.9)
        add_107: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_148, mul_149);  mul_148 = mul_149 = None
        squeeze_65: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_150: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0000063775916939);  squeeze_65 = None
        mul_151: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 0.1);  mul_150 = None
        mul_152: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_131, 0.9)
        add_108: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_84: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_132, -1)
        unsqueeze_85: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_153: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_147, unsqueeze_85);  mul_147 = unsqueeze_85 = None
        unsqueeze_86: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_133, -1)
        unsqueeze_87: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_109: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_153, unsqueeze_87);  mul_153 = unsqueeze_87 = None
        convert_element_type_66: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_21: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_66);  convert_element_type_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_67: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        convolution_22: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(cat_1, convert_element_type_67, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_110: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_135, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_68: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_68, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_68 = None
        getitem_48: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_22[0]
        getitem_49: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_111: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 0.001)
        rsqrt_22: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        sub_22: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, getitem_49)
        mul_154: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_67: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_155: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1)
        mul_156: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_136, 0.9)
        add_112: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
        squeeze_68: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_157: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0000063775916939);  squeeze_68 = None
        mul_158: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, 0.1);  mul_157 = None
        mul_159: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_137, 0.9)
        add_113: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_158, mul_159);  mul_158 = mul_159 = None
        unsqueeze_88: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_89: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_91: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_114: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        convert_element_type_69: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_22: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_69);  convert_element_type_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_70: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        convolution_23: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.convolution.default(relu_22, convert_element_type_70, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_115: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_141, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_71: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_23, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_71, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_71 = None
        getitem_50: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_23[0]
        getitem_51: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_116: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 0.001)
        rsqrt_23: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        sub_23: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, getitem_51)
        mul_161: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_70: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_162: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_163: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_142, 0.9)
        add_117: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
        squeeze_71: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_164: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0000063775916939);  squeeze_71 = None
        mul_165: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, 0.1);  mul_164 = None
        mul_166: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_143, 0.9)
        add_118: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
        unsqueeze_92: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_93: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_167: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_93);  mul_161 = unsqueeze_93 = None
        unsqueeze_94: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_95: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_119: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_95);  mul_167 = unsqueeze_95 = None
        convert_element_type_72: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_23: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_72);  convert_element_type_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_73: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(primals_146, torch.bfloat16);  primals_146 = None
        convolution_24: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.convolution.default(relu_23, convert_element_type_73, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_120: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_147, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_74: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_24, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_74, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_74 = None
        getitem_52: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_24[0]
        getitem_53: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_121: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 0.001)
        rsqrt_24: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
        sub_24: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_24, getitem_53)
        mul_168: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3])
        mul_169: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1);  squeeze_72 = None
        mul_170: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_122: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_74: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_171: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0000063775916939);  squeeze_74 = None
        mul_172: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_123: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_96: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_97: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_174: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_97);  mul_168 = unsqueeze_97 = None
        unsqueeze_98: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_151, -1)
        unsqueeze_99: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_124: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_99);  mul_174 = unsqueeze_99 = None
        convert_element_type_75: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_24: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_75);  convert_element_type_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:57 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_2: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.avg_pool2d.default(cat_1, [3, 3], [1, 1], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_76: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(primals_152, torch.bfloat16);  primals_152 = None
        convolution_25: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_2, convert_element_type_76, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_125: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_153, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_77: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_77, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_77 = None
        getitem_54: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_25[0]
        getitem_55: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_126: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 0.001)
        rsqrt_25: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_25: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_55)
        mul_175: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3])
        mul_176: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1);  squeeze_75 = None
        mul_177: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_127: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_77: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_178: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0000063775916939);  squeeze_77 = None
        mul_179: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
        add_128: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_100: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_157, -1)
        unsqueeze_103: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_129: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        convert_element_type_78: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16);  add_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_25: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_78);  convert_element_type_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:65 in forward, code: return torch.cat(outputs, 1)
        cat_2: "bf16[128, 288, 35, 35][352800, 1, 10080, 288]cuda:0" = torch.ops.aten.cat.default([relu_19, relu_21, relu_24, relu_25], 1);  relu_19 = relu_21 = relu_24 = relu_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_79: "bf16[384, 288, 3, 3][2592, 1, 864, 288]cuda:0" = torch.ops.prims.convert_element_type.default(primals_158, torch.bfloat16);  primals_158 = None
        convolution_26: "bf16[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.convolution.default(cat_2, convert_element_type_79, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_130: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_159, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_80: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_26, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_80, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_80 = None
        getitem_56: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_26[0]
        getitem_57: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_131: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 0.001)
        rsqrt_26: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
        sub_26: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_57)
        mul_182: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3])
        mul_183: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1);  squeeze_78 = None
        mul_184: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_132: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_80: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_185: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0000270336027683);  squeeze_80 = None
        mul_186: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
        add_133: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_104: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_105: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_188: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        unsqueeze_106: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_163, -1)
        unsqueeze_107: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_134: "f32[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None
        convert_element_type_81: "bf16[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16);  add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_26: "bf16[128, 384, 17, 17][110976, 1, 6528, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_81);  convert_element_type_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_82: "bf16[64, 288, 1, 1][288, 1, 288, 288]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        convolution_27: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.convolution.default(cat_2, convert_element_type_82, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_135: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_165, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_83: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_27, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_83, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_83 = None
        getitem_58: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_27[0]
        getitem_59: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_136: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 0.001)
        rsqrt_27: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
        sub_27: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, getitem_59)
        mul_189: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_82: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_190: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_191: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_166, 0.9)
        add_137: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_190, mul_191);  mul_190 = mul_191 = None
        squeeze_83: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_192: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0000063775916939);  squeeze_83 = None
        mul_193: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 0.1);  mul_192 = None
        mul_194: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_167, 0.9)
        add_138: "f32[64][1]cuda:0" = torch.ops.aten.add.Tensor(mul_193, mul_194);  mul_193 = mul_194 = None
        unsqueeze_108: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_168, -1)
        unsqueeze_109: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_195: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_109);  mul_189 = unsqueeze_109 = None
        unsqueeze_110: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_169, -1);  primals_169 = None
        unsqueeze_111: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_139: "f32[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_111);  mul_195 = unsqueeze_111 = None
        convert_element_type_84: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_27: "bf16[128, 64, 35, 35][78400, 1, 2240, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_84);  convert_element_type_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_85: "bf16[96, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        convolution_28: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.convolution.default(relu_27, convert_element_type_85, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_140: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_171, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_86: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_86, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_86 = None
        getitem_60: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_28[0]
        getitem_61: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_141: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 0.001)
        rsqrt_28: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_28: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_61)
        mul_196: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_85: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_197: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_198: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_172, 0.9)
        add_142: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_197, mul_198);  mul_197 = mul_198 = None
        squeeze_86: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_199: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0000063775916939);  squeeze_86 = None
        mul_200: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, 0.1);  mul_199 = None
        mul_201: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_173, 0.9)
        add_143: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        unsqueeze_112: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_113: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_115: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_144: "f32[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        convert_element_type_87: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_28: "bf16[128, 96, 35, 35][117600, 1, 3360, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_87);  convert_element_type_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_88: "bf16[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = torch.ops.prims.convert_element_type.default(primals_176, torch.bfloat16);  primals_176 = None
        convolution_29: "bf16[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.convolution.default(relu_28, convert_element_type_88, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_145: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_177, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_89: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_29, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_89, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_89 = None
        getitem_62: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_29[0]
        getitem_63: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_146: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 0.001)
        rsqrt_29: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
        sub_29: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, getitem_63)
        mul_203: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3])
        mul_204: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1);  squeeze_87 = None
        mul_205: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_178, 0.9)
        add_147: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
        squeeze_89: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_206: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0000270336027683);  squeeze_89 = None
        mul_207: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, 0.1);  mul_206 = None
        mul_208: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_179, 0.9)
        add_148: "f32[96][1]cuda:0" = torch.ops.aten.add.Tensor(mul_207, mul_208);  mul_207 = mul_208 = None
        unsqueeze_116: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_180, -1)
        unsqueeze_117: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_209: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_117);  mul_203 = unsqueeze_117 = None
        unsqueeze_118: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_181, -1)
        unsqueeze_119: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_149: "f32[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_119);  mul_209 = unsqueeze_119 = None
        convert_element_type_90: "bf16[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_29: "bf16[128, 96, 17, 17][27744, 1, 1632, 96]cuda:0" = torch.ops.aten.relu.default(convert_element_type_90);  convert_element_type_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:93 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        _low_memory_max_pool_with_offsets_2 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_2, [3, 3], [2, 2], [0, 0], [1, 1], False)
        getitem_64: "bf16[128, 288, 17, 17][83232, 1, 4896, 288]cuda:0" = _low_memory_max_pool_with_offsets_2[0]
        getitem_65: "i8[128, 288, 17, 17][83232, 1, 4896, 288]cuda:0" = _low_memory_max_pool_with_offsets_2[1];  _low_memory_max_pool_with_offsets_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:100 in forward, code: return torch.cat(outputs, 1)
        cat_3: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.cat.default([relu_26, relu_29, getitem_64], 1);  relu_26 = relu_29 = getitem_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_91: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_182, torch.bfloat16);  primals_182 = None
        convolution_30: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(cat_3, convert_element_type_91, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_150: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_183, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_92: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_30, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_92, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_92 = None
        getitem_66: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_30[0]
        getitem_67: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_151: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 0.001)
        rsqrt_30: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        sub_30: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_67)
        mul_210: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3])
        mul_211: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1);  squeeze_90 = None
        mul_212: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_184, 0.9)
        add_152: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_211, mul_212);  mul_211 = mul_212 = None
        squeeze_92: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_213: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0000270336027683);  squeeze_92 = None
        mul_214: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, 0.1);  mul_213 = None
        mul_215: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_185, 0.9)
        add_153: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_120: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_186, -1)
        unsqueeze_121: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_216: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_121);  mul_210 = unsqueeze_121 = None
        unsqueeze_122: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_187, -1)
        unsqueeze_123: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_154: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_123);  mul_216 = unsqueeze_123 = None
        convert_element_type_93: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_30: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_93);  convert_element_type_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_94: "bf16[128, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_188, torch.bfloat16);  primals_188 = None
        convolution_31: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.convolution.default(cat_3, convert_element_type_94, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_155: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_189, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_95: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_95, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_95 = None
        getitem_68: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_31[0]
        getitem_69: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_156: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 0.001)
        rsqrt_31: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
        sub_31: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_31, getitem_69)
        mul_217: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_94: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_218: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_219: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_190, 0.9)
        add_157: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, mul_219);  mul_218 = mul_219 = None
        squeeze_95: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_220: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0000270336027683);  squeeze_95 = None
        mul_221: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_220, 0.1);  mul_220 = None
        mul_222: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_191, 0.9)
        add_158: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_221, mul_222);  mul_221 = mul_222 = None
        unsqueeze_124: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
        unsqueeze_127: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_159: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        convert_element_type_96: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_159, torch.bfloat16);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_31: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.relu.default(convert_element_type_96);  convert_element_type_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_97: "bf16[128, 128, 1, 7][896, 1, 896, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        convolution_32: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.convolution.default(relu_31, convert_element_type_97, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_160: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_195, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_98: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_32, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_98, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_98 = None
        getitem_70: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_32[0]
        getitem_71: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_161: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 0.001)
        rsqrt_32: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
        sub_32: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_71)
        mul_224: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        squeeze_97: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_225: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1)
        mul_226: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_196, 0.9)
        add_162: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        squeeze_98: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_227: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0000270336027683);  squeeze_98 = None
        mul_228: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_227, 0.1);  mul_227 = None
        mul_229: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_197, 0.9)
        add_163: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_228, mul_229);  mul_228 = mul_229 = None
        unsqueeze_128: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_198, -1)
        unsqueeze_129: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_230: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_129);  mul_224 = unsqueeze_129 = None
        unsqueeze_130: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_199, -1);  primals_199 = None
        unsqueeze_131: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_164: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_131);  mul_230 = unsqueeze_131 = None
        convert_element_type_99: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16);  add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_32: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.relu.default(convert_element_type_99);  convert_element_type_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_100: "bf16[192, 128, 7, 1][896, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        convolution_33: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_32, convert_element_type_100, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_165: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_201, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_101: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_33, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_101, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_101 = None
        getitem_72: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_33[0]
        getitem_73: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_166: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 0.001)
        rsqrt_33: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        sub_33: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_73)
        mul_231: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3])
        mul_232: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1);  squeeze_99 = None
        mul_233: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_202, 0.9)
        add_167: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        squeeze_101: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_234: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0000270336027683);  squeeze_101 = None
        mul_235: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, 0.1);  mul_234 = None
        mul_236: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_203, 0.9)
        add_168: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_235, mul_236);  mul_235 = mul_236 = None
        unsqueeze_132: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_133: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_237: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, unsqueeze_133);  mul_231 = unsqueeze_133 = None
        unsqueeze_134: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_205, -1)
        unsqueeze_135: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_169: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_237, unsqueeze_135);  mul_237 = unsqueeze_135 = None
        convert_element_type_102: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16);  add_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_33: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_102);  convert_element_type_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_103: "bf16[128, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_206, torch.bfloat16);  primals_206 = None
        convolution_34: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.convolution.default(cat_3, convert_element_type_103, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_170: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_207, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_104: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_104, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_104 = None
        getitem_74: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_34[0]
        getitem_75: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_171: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 0.001)
        rsqrt_34: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
        sub_34: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_75)
        mul_238: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_103: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_239: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_240: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_208, 0.9)
        add_172: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
        squeeze_104: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_241: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0000270336027683);  squeeze_104 = None
        mul_242: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, 0.1);  mul_241 = None
        mul_243: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_209, 0.9)
        add_173: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        unsqueeze_136: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_137: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_139: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_174: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        convert_element_type_105: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.bfloat16);  add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_34: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.relu.default(convert_element_type_105);  convert_element_type_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_106: "bf16[128, 128, 7, 1][896, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_212, torch.bfloat16);  primals_212 = None
        convolution_35: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.convolution.default(relu_34, convert_element_type_106, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_175: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_213, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_107: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_35, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_107, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_107 = None
        getitem_76: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_35[0]
        getitem_77: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_176: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 0.001)
        rsqrt_35: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        sub_35: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_35, getitem_77)
        mul_245: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_106: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
        mul_246: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1)
        mul_247: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_214, 0.9)
        add_177: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        squeeze_107: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_248: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0000270336027683);  squeeze_107 = None
        mul_249: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_248, 0.1);  mul_248 = None
        mul_250: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_215, 0.9)
        add_178: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        unsqueeze_140: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_216, -1)
        unsqueeze_141: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_251: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_141);  mul_245 = unsqueeze_141 = None
        unsqueeze_142: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_217, -1);  primals_217 = None
        unsqueeze_143: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_179: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_143);  mul_251 = unsqueeze_143 = None
        convert_element_type_108: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_35: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.relu.default(convert_element_type_108);  convert_element_type_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_109: "bf16[128, 128, 1, 7][896, 1, 896, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_218, torch.bfloat16);  primals_218 = None
        convolution_36: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.convolution.default(relu_35, convert_element_type_109, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_180: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_219, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_110: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_36, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_110, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_110 = None
        getitem_78: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_36[0]
        getitem_79: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_181: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 0.001)
        rsqrt_36: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_36: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_36, getitem_79)
        mul_252: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_109: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_253: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_254: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_182: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_253, mul_254);  mul_253 = mul_254 = None
        squeeze_110: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_255: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0000270336027683);  squeeze_110 = None
        mul_256: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, 0.1);  mul_255 = None
        mul_257: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_183: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        unsqueeze_144: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_145: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_258: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_145);  mul_252 = unsqueeze_145 = None
        unsqueeze_146: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_147: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_184: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_147);  mul_258 = unsqueeze_147 = None
        convert_element_type_111: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_36: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.relu.default(convert_element_type_111);  convert_element_type_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_112: "bf16[128, 128, 7, 1][896, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        convolution_37: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.convolution.default(relu_36, convert_element_type_112, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_185: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_225, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_113: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_113, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_113 = None
        getitem_80: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_37[0]
        getitem_81: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_186: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 0.001)
        rsqrt_37: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_37: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_37, getitem_81)
        mul_259: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_112: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_260: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_261: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_187: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_260, mul_261);  mul_260 = mul_261 = None
        squeeze_113: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_262: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0000270336027683);  squeeze_113 = None
        mul_263: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, 0.1);  mul_262 = None
        mul_264: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_188: "f32[128][1]cuda:0" = torch.ops.aten.add.Tensor(mul_263, mul_264);  mul_263 = mul_264 = None
        unsqueeze_148: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_151: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_189: "f32[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        convert_element_type_114: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_37: "bf16[128, 128, 17, 17][36992, 1, 2176, 128]cuda:0" = torch.ops.aten.relu.default(convert_element_type_114);  convert_element_type_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_115: "bf16[192, 128, 1, 7][896, 1, 896, 128]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        convolution_38: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_37, convert_element_type_115, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_190: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_231, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_116: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_38, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_116, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_116 = None
        getitem_82: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_38[0]
        getitem_83: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_191: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 0.001)
        rsqrt_38: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_38: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_38, getitem_83)
        mul_266: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3])
        mul_267: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1);  squeeze_114 = None
        mul_268: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_192: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        squeeze_116: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_269: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0000270336027683);  squeeze_116 = None
        mul_270: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, 0.1);  mul_269 = None
        mul_271: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_193: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
        unsqueeze_152: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_153: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_272: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        unsqueeze_154: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_235, -1)
        unsqueeze_155: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_194: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None
        convert_element_type_117: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_194, torch.bfloat16);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_38: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_117);  convert_element_type_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_3: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.avg_pool2d.default(cat_3, [3, 3], [1, 1], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_118: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_236, torch.bfloat16);  primals_236 = None
        convolution_39: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_3, convert_element_type_118, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_195: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_237, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_119: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_39, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_119, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_119 = None
        getitem_84: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_39[0]
        getitem_85: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_196: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 0.001)
        rsqrt_39: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_39: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_39, getitem_85)
        mul_273: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3])
        mul_274: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1);  squeeze_117 = None
        mul_275: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_238, 0.9)
        add_197: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        squeeze_119: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_276: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0000270336027683);  squeeze_119 = None
        mul_277: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, 0.1);  mul_276 = None
        mul_278: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_239, 0.9)
        add_198: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_156: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_157: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_279: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_273, unsqueeze_157);  mul_273 = unsqueeze_157 = None
        unsqueeze_158: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_241, -1)
        unsqueeze_159: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_199: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_279, unsqueeze_159);  mul_279 = unsqueeze_159 = None
        convert_element_type_120: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16);  add_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_39: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_120);  convert_element_type_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        cat_4: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.cat.default([relu_30, relu_33, relu_38, relu_39], 1);  relu_30 = relu_33 = relu_38 = relu_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_121: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_242, torch.bfloat16);  primals_242 = None
        convolution_40: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(cat_4, convert_element_type_121, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_200: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_243, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_122: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_40, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_122, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_122 = None
        getitem_86: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_40[0]
        getitem_87: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_201: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 0.001)
        rsqrt_40: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_201);  add_201 = None
        sub_40: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_40, getitem_87)
        mul_280: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3])
        mul_281: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1);  squeeze_120 = None
        mul_282: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_244, 0.9)
        add_202: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        squeeze_122: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_283: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0000270336027683);  squeeze_122 = None
        mul_284: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, 0.1);  mul_283 = None
        mul_285: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_245, 0.9)
        add_203: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_160: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_161: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_247, -1)
        unsqueeze_163: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_204: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        convert_element_type_123: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16);  add_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_40: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_123);  convert_element_type_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_124: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_248, torch.bfloat16);  primals_248 = None
        convolution_41: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(cat_4, convert_element_type_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_205: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_249, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_125: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_125, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_125 = None
        getitem_88: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_41[0]
        getitem_89: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_206: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 0.001)
        rsqrt_41: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        sub_41: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_41, getitem_89)
        mul_287: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_124: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_288: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1)
        mul_289: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_250, 0.9)
        add_207: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
        squeeze_125: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_290: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0000270336027683);  squeeze_125 = None
        mul_291: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, 0.1);  mul_290 = None
        mul_292: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_251, 0.9)
        add_208: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        unsqueeze_164: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_252, -1)
        unsqueeze_165: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_293: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, unsqueeze_165);  mul_287 = unsqueeze_165 = None
        unsqueeze_166: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
        unsqueeze_167: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_209: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_167);  mul_293 = unsqueeze_167 = None
        convert_element_type_126: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16);  add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_41: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_126);  convert_element_type_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_127: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        convolution_42: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(relu_41, convert_element_type_127, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_210: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_255, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_128: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_42, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_128, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_128 = None
        getitem_90: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_42[0]
        getitem_91: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_211: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 0.001)
        rsqrt_42: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        sub_42: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_42, getitem_91)
        mul_294: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_127: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_295: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_296: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_256, 0.9)
        add_212: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        squeeze_128: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_297: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0000270336027683);  squeeze_128 = None
        mul_298: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, 0.1);  mul_297 = None
        mul_299: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_257, 0.9)
        add_213: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_168: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_258, -1)
        unsqueeze_169: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_300: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        unsqueeze_170: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
        unsqueeze_171: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_214: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        convert_element_type_129: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_42: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_129);  convert_element_type_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_130: "bf16[192, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        convolution_43: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_42, convert_element_type_130, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_215: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_261, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_131: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_43, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_131, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_131 = None
        getitem_92: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_43[0]
        getitem_93: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_216: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 0.001)
        rsqrt_43: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
        sub_43: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_43, getitem_93)
        mul_301: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3])
        mul_302: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1);  squeeze_129 = None
        mul_303: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_262, 0.9)
        add_217: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        squeeze_131: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_304: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0000270336027683);  squeeze_131 = None
        mul_305: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, 0.1);  mul_304 = None
        mul_306: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_263, 0.9)
        add_218: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
        unsqueeze_172: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_264, -1)
        unsqueeze_173: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_307: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_173);  mul_301 = unsqueeze_173 = None
        unsqueeze_174: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_265, -1)
        unsqueeze_175: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_219: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_307, unsqueeze_175);  mul_307 = unsqueeze_175 = None
        convert_element_type_132: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_43: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_132);  convert_element_type_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_133: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_266, torch.bfloat16);  primals_266 = None
        convolution_44: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(cat_4, convert_element_type_133, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_220: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_267, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_134: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_134, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_134 = None
        getitem_94: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_44[0]
        getitem_95: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_221: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 0.001)
        rsqrt_44: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        sub_44: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_44, getitem_95)
        mul_308: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        squeeze_133: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_309: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1)
        mul_310: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_268, 0.9)
        add_222: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
        squeeze_134: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_311: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0000270336027683);  squeeze_134 = None
        mul_312: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 0.1);  mul_311 = None
        mul_313: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_269, 0.9)
        add_223: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
        unsqueeze_176: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_177: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_314: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_177);  mul_308 = unsqueeze_177 = None
        unsqueeze_178: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_271, -1);  primals_271 = None
        unsqueeze_179: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_224: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_179);  mul_314 = unsqueeze_179 = None
        convert_element_type_135: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_224, torch.bfloat16);  add_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_44: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_135);  convert_element_type_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_136: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_272, torch.bfloat16);  primals_272 = None
        convolution_45: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(relu_44, convert_element_type_136, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_225: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_273, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_137: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_45, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_137, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_137 = None
        getitem_96: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_45[0]
        getitem_97: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_226: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_96, 0.001)
        rsqrt_45: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_226);  add_226 = None
        sub_45: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_45, getitem_97)
        mul_315: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_136: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_316: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
        mul_317: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_274, 0.9)
        add_227: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        squeeze_137: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_318: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0000270336027683);  squeeze_137 = None
        mul_319: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 0.1);  mul_318 = None
        mul_320: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_275, 0.9)
        add_228: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_319, mul_320);  mul_319 = mul_320 = None
        unsqueeze_180: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_181: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_321: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        unsqueeze_182: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_183: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_229: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        convert_element_type_138: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_229, torch.bfloat16);  add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_45: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_138);  convert_element_type_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_139: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_278, torch.bfloat16);  primals_278 = None
        convolution_46: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(relu_45, convert_element_type_139, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_230: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_279, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_140: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_46, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_140, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_140 = None
        getitem_98: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_46[0]
        getitem_99: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_231: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_98, 0.001)
        rsqrt_46: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_231);  add_231 = None
        sub_46: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_46, getitem_99)
        mul_322: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_139: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_323: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_324: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_280, 0.9)
        add_232: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
        squeeze_140: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_325: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0000270336027683);  squeeze_140 = None
        mul_326: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, 0.1);  mul_325 = None
        mul_327: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_281, 0.9)
        add_233: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
        unsqueeze_184: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_282, -1)
        unsqueeze_185: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_328: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_185);  mul_322 = unsqueeze_185 = None
        unsqueeze_186: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_283, -1);  primals_283 = None
        unsqueeze_187: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_234: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_328, unsqueeze_187);  mul_328 = unsqueeze_187 = None
        convert_element_type_141: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16);  add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_46: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_141);  convert_element_type_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_142: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        convolution_47: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(relu_46, convert_element_type_142, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_235: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_285, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_143: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_47, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_143, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_143 = None
        getitem_100: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_47[0]
        getitem_101: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_236: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_100, 0.001)
        rsqrt_47: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_47: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_47, getitem_101)
        mul_329: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_142: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
        mul_330: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1)
        mul_331: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
        add_237: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_143: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_332: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0000270336027683);  squeeze_143 = None
        mul_333: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
        add_238: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_188: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_288, -1)
        unsqueeze_189: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_335: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_189);  mul_329 = unsqueeze_189 = None
        unsqueeze_190: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_289, -1);  primals_289 = None
        unsqueeze_191: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_239: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_191);  mul_335 = unsqueeze_191 = None
        convert_element_type_144: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_239, torch.bfloat16);  add_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_47: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_144);  convert_element_type_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_145: "bf16[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16);  primals_290 = None
        convolution_48: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_47, convert_element_type_145, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_240: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_291, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_146: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_48, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_146, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_146 = None
        getitem_102: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_48[0]
        getitem_103: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_48[1];  var_mean_48 = None
        add_241: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_102, 0.001)
        rsqrt_48: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_241);  add_241 = None
        sub_48: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_48, getitem_103)
        mul_336: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3])
        mul_337: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1);  squeeze_144 = None
        mul_338: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
        add_242: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_146: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_339: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0000270336027683);  squeeze_146 = None
        mul_340: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
        add_243: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_192: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_193: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_295, -1)
        unsqueeze_195: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_244: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        convert_element_type_147: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16);  add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_48: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_147);  convert_element_type_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_4: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.avg_pool2d.default(cat_4, [3, 3], [1, 1], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_148: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_296, torch.bfloat16);  primals_296 = None
        convolution_49: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_4, convert_element_type_148, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_245: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_297, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_149: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_49, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_149, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_149 = None
        getitem_104: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_49[0]
        getitem_105: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_49[1];  var_mean_49 = None
        add_246: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, 0.001)
        rsqrt_49: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        sub_49: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_49, getitem_105)
        mul_343: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3])
        mul_344: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1);  squeeze_147 = None
        mul_345: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
        add_247: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_149: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_346: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0000270336027683);  squeeze_149 = None
        mul_347: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
        add_248: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_196: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_197: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_349: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_197);  mul_343 = unsqueeze_197 = None
        unsqueeze_198: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_301, -1)
        unsqueeze_199: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_249: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_199);  mul_349 = unsqueeze_199 = None
        convert_element_type_150: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.bfloat16);  add_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_49: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_150);  convert_element_type_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        cat_5: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.cat.default([relu_40, relu_43, relu_48, relu_49], 1);  relu_40 = relu_43 = relu_48 = relu_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_151: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_302, torch.bfloat16);  primals_302 = None
        convolution_50: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(cat_5, convert_element_type_151, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_250: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_303, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_152: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_50, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_152, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_152 = None
        getitem_106: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_50[0]
        getitem_107: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_50[1];  var_mean_50 = None
        add_251: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_106, 0.001)
        rsqrt_50: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_50: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_50, getitem_107)
        mul_350: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3])
        mul_351: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1);  squeeze_150 = None
        mul_352: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
        add_252: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
        squeeze_152: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_353: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0000270336027683);  squeeze_152 = None
        mul_354: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, 0.1);  mul_353 = None
        mul_355: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
        add_253: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
        unsqueeze_200: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_306, -1)
        unsqueeze_201: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_356: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_350, unsqueeze_201);  mul_350 = unsqueeze_201 = None
        unsqueeze_202: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_307, -1)
        unsqueeze_203: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_254: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_203);  mul_356 = unsqueeze_203 = None
        convert_element_type_153: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_50: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_153);  convert_element_type_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_154: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_308, torch.bfloat16);  primals_308 = None
        convolution_51: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(cat_5, convert_element_type_154, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_255: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_309, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_155: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_51, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_155, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_155 = None
        getitem_108: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_51[0]
        getitem_109: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_51[1];  var_mean_51 = None
        add_256: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_108, 0.001)
        rsqrt_51: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
        sub_51: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_51, getitem_109)
        mul_357: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_154: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_358: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
        mul_359: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
        add_257: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        squeeze_155: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_360: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0000270336027683);  squeeze_155 = None
        mul_361: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, 0.1);  mul_360 = None
        mul_362: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
        add_258: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        unsqueeze_204: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_205: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_207: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_259: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        convert_element_type_156: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_51: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_156);  convert_element_type_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_157: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_314, torch.bfloat16);  primals_314 = None
        convolution_52: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(relu_51, convert_element_type_157, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_260: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_315, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_158: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_52, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_158, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_158 = None
        getitem_110: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_52[0]
        getitem_111: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_52[1];  var_mean_52 = None
        add_261: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_110, 0.001)
        rsqrt_52: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_52: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_52, getitem_111)
        mul_364: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_157: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_365: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_366: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
        add_262: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        squeeze_158: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_367: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0000270336027683);  squeeze_158 = None
        mul_368: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, 0.1);  mul_367 = None
        mul_369: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
        add_263: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_208: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_318, -1)
        unsqueeze_209: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        mul_370: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_209);  mul_364 = unsqueeze_209 = None
        unsqueeze_210: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_319, -1);  primals_319 = None
        unsqueeze_211: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        add_264: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_370, unsqueeze_211);  mul_370 = unsqueeze_211 = None
        convert_element_type_159: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_52: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_159);  convert_element_type_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_160: "bf16[192, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_320, torch.bfloat16);  primals_320 = None
        convolution_53: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_52, convert_element_type_160, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_265: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_321, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_161: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_53, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_161, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_161 = None
        getitem_112: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_53[0]
        getitem_113: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_53[1];  var_mean_53 = None
        add_266: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_112, 0.001)
        rsqrt_53: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_53: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_53, getitem_113)
        mul_371: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        squeeze_159: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3])
        mul_372: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1);  squeeze_159 = None
        mul_373: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
        add_267: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
        squeeze_161: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_374: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0000270336027683);  squeeze_161 = None
        mul_375: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_374, 0.1);  mul_374 = None
        mul_376: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
        add_268: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
        unsqueeze_212: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_324, -1)
        unsqueeze_213: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_377: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_371, unsqueeze_213);  mul_371 = unsqueeze_213 = None
        unsqueeze_214: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_325, -1)
        unsqueeze_215: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_269: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_215);  mul_377 = unsqueeze_215 = None
        convert_element_type_162: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_269, torch.bfloat16);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_53: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_162);  convert_element_type_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_163: "bf16[160, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_326, torch.bfloat16);  primals_326 = None
        convolution_54: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(cat_5, convert_element_type_163, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_270: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_327, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_164: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_54, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_164, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_164 = None
        getitem_114: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_54[0]
        getitem_115: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_54[1];  var_mean_54 = None
        add_271: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_114, 0.001)
        rsqrt_54: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_54: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_54, getitem_115)
        mul_378: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        squeeze_162: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        squeeze_163: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_379: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
        mul_380: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
        add_272: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
        squeeze_164: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_381: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0000270336027683);  squeeze_164 = None
        mul_382: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, 0.1);  mul_381 = None
        mul_383: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
        add_273: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_382, mul_383);  mul_382 = mul_383 = None
        unsqueeze_216: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_330, -1)
        unsqueeze_217: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        mul_384: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
        unsqueeze_218: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_331, -1);  primals_331 = None
        unsqueeze_219: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        add_274: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
        convert_element_type_165: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_274, torch.bfloat16);  add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_54: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_165);  convert_element_type_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_166: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_332, torch.bfloat16);  primals_332 = None
        convolution_55: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(relu_54, convert_element_type_166, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_275: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_333, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_167: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_55, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_167, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_167 = None
        getitem_116: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_55[0]
        getitem_117: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_55[1];  var_mean_55 = None
        add_276: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_116, 0.001)
        rsqrt_55: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        sub_55: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_55, getitem_117)
        mul_385: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        squeeze_165: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_166: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_386: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
        mul_387: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
        add_277: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
        squeeze_167: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_388: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0000270336027683);  squeeze_167 = None
        mul_389: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_388, 0.1);  mul_388 = None
        mul_390: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
        add_278: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_389, mul_390);  mul_389 = mul_390 = None
        unsqueeze_220: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_221: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_391: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
        unsqueeze_222: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_223: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_279: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
        convert_element_type_168: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16);  add_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_55: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_168);  convert_element_type_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_169: "bf16[160, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_338, torch.bfloat16);  primals_338 = None
        convolution_56: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(relu_55, convert_element_type_169, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_280: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_339, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_170: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_56, torch.float32)
        var_mean_56 = torch.ops.aten.var_mean.correction(convert_element_type_170, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_170 = None
        getitem_118: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_56[0]
        getitem_119: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_56[1];  var_mean_56 = None
        add_281: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_118, 0.001)
        rsqrt_56: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_281);  add_281 = None
        sub_56: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_56, getitem_119)
        mul_392: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
        squeeze_168: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        squeeze_169: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_393: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1)
        mul_394: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_340, 0.9)
        add_282: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
        squeeze_170: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_395: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0000270336027683);  squeeze_170 = None
        mul_396: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, 0.1);  mul_395 = None
        mul_397: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_341, 0.9)
        add_283: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
        unsqueeze_224: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_342, -1)
        unsqueeze_225: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        mul_398: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_392, unsqueeze_225);  mul_392 = unsqueeze_225 = None
        unsqueeze_226: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_343, -1);  primals_343 = None
        unsqueeze_227: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        add_284: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_227);  mul_398 = unsqueeze_227 = None
        convert_element_type_171: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_284, torch.bfloat16);  add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_56: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_171);  convert_element_type_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_172: "bf16[160, 160, 7, 1][1120, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_344, torch.bfloat16);  primals_344 = None
        convolution_57: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.convolution.default(relu_56, convert_element_type_172, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_285: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_345, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_173: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_57, torch.float32)
        var_mean_57 = torch.ops.aten.var_mean.correction(convert_element_type_173, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_173 = None
        getitem_120: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_57[0]
        getitem_121: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = var_mean_57[1];  var_mean_57 = None
        add_286: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_120, 0.001)
        rsqrt_57: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_286);  add_286 = None
        sub_57: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_57, getitem_121)
        mul_399: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
        squeeze_171: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        squeeze_172: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_400: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
        mul_401: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_346, 0.9)
        add_287: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_400, mul_401);  mul_400 = mul_401 = None
        squeeze_173: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_402: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0000270336027683);  squeeze_173 = None
        mul_403: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 0.1);  mul_402 = None
        mul_404: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_347, 0.9)
        add_288: "f32[160][1]cuda:0" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
        unsqueeze_228: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_348, -1)
        unsqueeze_229: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_405: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_399, unsqueeze_229);  mul_399 = unsqueeze_229 = None
        unsqueeze_230: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_349, -1);  primals_349 = None
        unsqueeze_231: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_289: "f32[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_405, unsqueeze_231);  mul_405 = unsqueeze_231 = None
        convert_element_type_174: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.prims.convert_element_type.default(add_289, torch.bfloat16);  add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_57: "bf16[128, 160, 17, 17][46240, 1, 2720, 160]cuda:0" = torch.ops.aten.relu.default(convert_element_type_174);  convert_element_type_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_175: "bf16[192, 160, 1, 7][1120, 1, 1120, 160]cuda:0" = torch.ops.prims.convert_element_type.default(primals_350, torch.bfloat16);  primals_350 = None
        convolution_58: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_57, convert_element_type_175, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_290: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_351, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_176: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_58, torch.float32)
        var_mean_58 = torch.ops.aten.var_mean.correction(convert_element_type_176, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_176 = None
        getitem_122: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_58[0]
        getitem_123: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_58[1];  var_mean_58 = None
        add_291: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_122, 0.001)
        rsqrt_58: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_291);  add_291 = None
        sub_58: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_58, getitem_123)
        mul_406: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
        squeeze_174: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3])
        mul_407: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1);  squeeze_174 = None
        mul_408: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
        add_292: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
        squeeze_176: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_409: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0000270336027683);  squeeze_176 = None
        mul_410: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_409, 0.1);  mul_409 = None
        mul_411: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
        add_293: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
        unsqueeze_232: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_354, -1)
        unsqueeze_233: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        mul_412: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_233);  mul_406 = unsqueeze_233 = None
        unsqueeze_234: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_355, -1)
        unsqueeze_235: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        add_294: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_412, unsqueeze_235);  mul_412 = unsqueeze_235 = None
        convert_element_type_177: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_294, torch.bfloat16);  add_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_58: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_177);  convert_element_type_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_5: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.avg_pool2d.default(cat_5, [3, 3], [1, 1], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_178: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_356, torch.bfloat16);  primals_356 = None
        convolution_59: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_5, convert_element_type_178, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_295: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_357, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_179: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_59, torch.float32)
        var_mean_59 = torch.ops.aten.var_mean.correction(convert_element_type_179, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_179 = None
        getitem_124: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_59[0]
        getitem_125: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_59[1];  var_mean_59 = None
        add_296: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_124, 0.001)
        rsqrt_59: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_296);  add_296 = None
        sub_59: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_59, getitem_125)
        mul_413: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
        squeeze_177: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3])
        mul_414: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1);  squeeze_177 = None
        mul_415: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
        add_297: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_414, mul_415);  mul_414 = mul_415 = None
        squeeze_179: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_416: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0000270336027683);  squeeze_179 = None
        mul_417: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, 0.1);  mul_416 = None
        mul_418: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
        add_298: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
        unsqueeze_236: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_360, -1)
        unsqueeze_237: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_419: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_413, unsqueeze_237);  mul_413 = unsqueeze_237 = None
        unsqueeze_238: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_361, -1)
        unsqueeze_239: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_299: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_239);  mul_419 = unsqueeze_239 = None
        convert_element_type_180: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_299, torch.bfloat16);  add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_59: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_180);  convert_element_type_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        cat_6: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.cat.default([relu_50, relu_53, relu_58, relu_59], 1);  relu_50 = relu_53 = relu_58 = relu_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_181: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_362, torch.bfloat16);  primals_362 = None
        convolution_60: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(cat_6, convert_element_type_181, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_300: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_363, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_182: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_60, torch.float32)
        var_mean_60 = torch.ops.aten.var_mean.correction(convert_element_type_182, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_182 = None
        getitem_126: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_60[0]
        getitem_127: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_60[1];  var_mean_60 = None
        add_301: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_126, 0.001)
        rsqrt_60: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_301);  add_301 = None
        sub_60: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_60, getitem_127)
        mul_420: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
        squeeze_180: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3])
        mul_421: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1);  squeeze_180 = None
        mul_422: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
        add_302: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
        squeeze_182: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_423: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0000270336027683);  squeeze_182 = None
        mul_424: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_423, 0.1);  mul_423 = None
        mul_425: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
        add_303: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
        unsqueeze_240: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_366, -1)
        unsqueeze_241: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        mul_426: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, unsqueeze_241);  mul_420 = unsqueeze_241 = None
        unsqueeze_242: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_367, -1)
        unsqueeze_243: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        add_304: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_426, unsqueeze_243);  mul_426 = unsqueeze_243 = None
        convert_element_type_183: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_304, torch.bfloat16);  add_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_60: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_183);  convert_element_type_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_184: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_368, torch.bfloat16);  primals_368 = None
        convolution_61: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(cat_6, convert_element_type_184, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_305: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_369, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_185: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_61, torch.float32)
        var_mean_61 = torch.ops.aten.var_mean.correction(convert_element_type_185, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_185 = None
        getitem_128: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_61[0]
        getitem_129: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_61[1];  var_mean_61 = None
        add_306: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_128, 0.001)
        rsqrt_61: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
        sub_61: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_61, getitem_129)
        mul_427: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
        squeeze_183: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        squeeze_184: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_428: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1)
        mul_429: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_370, 0.9)
        add_307: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
        squeeze_185: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
        mul_430: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0000270336027683);  squeeze_185 = None
        mul_431: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, 0.1);  mul_430 = None
        mul_432: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_371, 0.9)
        add_308: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
        unsqueeze_244: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_372, -1)
        unsqueeze_245: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_433: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_245);  mul_427 = unsqueeze_245 = None
        unsqueeze_246: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_373, -1);  primals_373 = None
        unsqueeze_247: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_309: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_433, unsqueeze_247);  mul_433 = unsqueeze_247 = None
        convert_element_type_186: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_309, torch.bfloat16);  add_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_61: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_186);  convert_element_type_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_187: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_374, torch.bfloat16);  primals_374 = None
        convolution_62: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_61, convert_element_type_187, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_310: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_375, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_188: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_62, torch.float32)
        var_mean_62 = torch.ops.aten.var_mean.correction(convert_element_type_188, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_188 = None
        getitem_130: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_62[0]
        getitem_131: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_62[1];  var_mean_62 = None
        add_311: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_130, 0.001)
        rsqrt_62: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_311);  add_311 = None
        sub_62: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_62, getitem_131)
        mul_434: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = None
        squeeze_186: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
        squeeze_187: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
        mul_435: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_186, 0.1)
        mul_436: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_376, 0.9)
        add_312: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
        squeeze_188: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
        mul_437: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_188, 1.0000270336027683);  squeeze_188 = None
        mul_438: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_437, 0.1);  mul_437 = None
        mul_439: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_377, 0.9)
        add_313: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
        unsqueeze_248: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_378, -1)
        unsqueeze_249: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        mul_440: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_434, unsqueeze_249);  mul_434 = unsqueeze_249 = None
        unsqueeze_250: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_379, -1);  primals_379 = None
        unsqueeze_251: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        add_314: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_251);  mul_440 = unsqueeze_251 = None
        convert_element_type_189: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_314, torch.bfloat16);  add_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_62: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_189);  convert_element_type_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_190: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_380, torch.bfloat16);  primals_380 = None
        convolution_63: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_62, convert_element_type_190, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_315: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_381, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_191: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_63, torch.float32)
        var_mean_63 = torch.ops.aten.var_mean.correction(convert_element_type_191, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_191 = None
        getitem_132: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_63[0]
        getitem_133: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_63[1];  var_mean_63 = None
        add_316: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_132, 0.001)
        rsqrt_63: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_316);  add_316 = None
        sub_63: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_63, getitem_133)
        mul_441: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = None
        squeeze_189: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3])
        mul_442: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_189, 0.1);  squeeze_189 = None
        mul_443: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_382, 0.9)
        add_317: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
        squeeze_191: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
        mul_444: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_191, 1.0000270336027683);  squeeze_191 = None
        mul_445: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_444, 0.1);  mul_444 = None
        mul_446: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_383, 0.9)
        add_318: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_445, mul_446);  mul_445 = mul_446 = None
        unsqueeze_252: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_384, -1)
        unsqueeze_253: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_447: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, unsqueeze_253);  mul_441 = unsqueeze_253 = None
        unsqueeze_254: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_385, -1)
        unsqueeze_255: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_319: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_447, unsqueeze_255);  mul_447 = unsqueeze_255 = None
        convert_element_type_192: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_319, torch.bfloat16);  add_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_63: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_192);  convert_element_type_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_193: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_386, torch.bfloat16);  primals_386 = None
        convolution_64: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(cat_6, convert_element_type_193, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_320: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_387, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_194: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_64, torch.float32)
        var_mean_64 = torch.ops.aten.var_mean.correction(convert_element_type_194, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_194 = None
        getitem_134: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_64[0]
        getitem_135: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_64[1];  var_mean_64 = None
        add_321: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_134, 0.001)
        rsqrt_64: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_321);  add_321 = None
        sub_64: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_64, getitem_135)
        mul_448: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = None
        squeeze_192: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
        squeeze_193: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_64, [0, 2, 3]);  rsqrt_64 = None
        mul_449: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_192, 0.1)
        mul_450: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_388, 0.9)
        add_322: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
        squeeze_194: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
        mul_451: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_194, 1.0000270336027683);  squeeze_194 = None
        mul_452: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_451, 0.1);  mul_451 = None
        mul_453: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_389, 0.9)
        add_323: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_452, mul_453);  mul_452 = mul_453 = None
        unsqueeze_256: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_390, -1)
        unsqueeze_257: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        mul_454: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_257);  mul_448 = unsqueeze_257 = None
        unsqueeze_258: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_391, -1);  primals_391 = None
        unsqueeze_259: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        add_324: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_454, unsqueeze_259);  mul_454 = unsqueeze_259 = None
        convert_element_type_195: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_324, torch.bfloat16);  add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_64: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_195);  convert_element_type_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_196: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_392, torch.bfloat16);  primals_392 = None
        convolution_65: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_64, convert_element_type_196, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_325: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_393, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_197: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_65, torch.float32)
        var_mean_65 = torch.ops.aten.var_mean.correction(convert_element_type_197, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_197 = None
        getitem_136: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_65[0]
        getitem_137: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_65[1];  var_mean_65 = None
        add_326: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_136, 0.001)
        rsqrt_65: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_326);  add_326 = None
        sub_65: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_65, getitem_137)
        mul_455: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = None
        squeeze_195: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        squeeze_196: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
        mul_456: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_195, 0.1)
        mul_457: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_394, 0.9)
        add_327: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
        squeeze_197: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
        mul_458: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_197, 1.0000270336027683);  squeeze_197 = None
        mul_459: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_458, 0.1);  mul_458 = None
        mul_460: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_395, 0.9)
        add_328: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
        unsqueeze_260: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_396, -1)
        unsqueeze_261: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_461: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_455, unsqueeze_261);  mul_455 = unsqueeze_261 = None
        unsqueeze_262: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_397, -1);  primals_397 = None
        unsqueeze_263: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_329: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_263);  mul_461 = unsqueeze_263 = None
        convert_element_type_198: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_329, torch.bfloat16);  add_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_65: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_198);  convert_element_type_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_199: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_398, torch.bfloat16);  primals_398 = None
        convolution_66: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_65, convert_element_type_199, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_330: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_399, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_200: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_66, torch.float32)
        var_mean_66 = torch.ops.aten.var_mean.correction(convert_element_type_200, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_200 = None
        getitem_138: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_66[0]
        getitem_139: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_66[1];  var_mean_66 = None
        add_331: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_138, 0.001)
        rsqrt_66: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_331);  add_331 = None
        sub_66: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_66, getitem_139)
        mul_462: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = None
        squeeze_198: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        squeeze_199: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
        mul_463: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_198, 0.1)
        mul_464: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_400, 0.9)
        add_332: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_463, mul_464);  mul_463 = mul_464 = None
        squeeze_200: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
        mul_465: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_200, 1.0000270336027683);  squeeze_200 = None
        mul_466: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_465, 0.1);  mul_465 = None
        mul_467: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_401, 0.9)
        add_333: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        unsqueeze_264: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_402, -1)
        unsqueeze_265: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        mul_468: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_462, unsqueeze_265);  mul_462 = unsqueeze_265 = None
        unsqueeze_266: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_403, -1);  primals_403 = None
        unsqueeze_267: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        add_334: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_468, unsqueeze_267);  mul_468 = unsqueeze_267 = None
        convert_element_type_201: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_334, torch.bfloat16);  add_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_66: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_201);  convert_element_type_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_202: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_404, torch.bfloat16);  primals_404 = None
        convolution_67: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_66, convert_element_type_202, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_335: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_405, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_203: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_67, torch.float32)
        var_mean_67 = torch.ops.aten.var_mean.correction(convert_element_type_203, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_203 = None
        getitem_140: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_67[0]
        getitem_141: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_67[1];  var_mean_67 = None
        add_336: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_140, 0.001)
        rsqrt_67: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_336);  add_336 = None
        sub_67: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_67, getitem_141)
        mul_469: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = None
        squeeze_201: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
        squeeze_202: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
        mul_470: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_201, 0.1)
        mul_471: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_406, 0.9)
        add_337: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_470, mul_471);  mul_470 = mul_471 = None
        squeeze_203: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
        mul_472: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_203, 1.0000270336027683);  squeeze_203 = None
        mul_473: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_472, 0.1);  mul_472 = None
        mul_474: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_407, 0.9)
        add_338: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
        unsqueeze_268: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_408, -1)
        unsqueeze_269: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_475: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_469, unsqueeze_269);  mul_469 = unsqueeze_269 = None
        unsqueeze_270: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_409, -1);  primals_409 = None
        unsqueeze_271: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_339: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_475, unsqueeze_271);  mul_475 = unsqueeze_271 = None
        convert_element_type_204: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_339, torch.bfloat16);  add_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_67: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_204);  convert_element_type_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_205: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_410, torch.bfloat16);  primals_410 = None
        convolution_68: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_67, convert_element_type_205, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_340: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_411, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_206: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_68, torch.float32)
        var_mean_68 = torch.ops.aten.var_mean.correction(convert_element_type_206, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_206 = None
        getitem_142: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_68[0]
        getitem_143: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_68[1];  var_mean_68 = None
        add_341: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_142, 0.001)
        rsqrt_68: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_341);  add_341 = None
        sub_68: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_68, getitem_143)
        mul_476: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = None
        squeeze_204: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3])
        mul_477: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_204, 0.1);  squeeze_204 = None
        mul_478: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_412, 0.9)
        add_342: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
        squeeze_206: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
        mul_479: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_206, 1.0000270336027683);  squeeze_206 = None
        mul_480: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_479, 0.1);  mul_479 = None
        mul_481: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_413, 0.9)
        add_343: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_480, mul_481);  mul_480 = mul_481 = None
        unsqueeze_272: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_414, -1)
        unsqueeze_273: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        mul_482: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_476, unsqueeze_273);  mul_476 = unsqueeze_273 = None
        unsqueeze_274: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_415, -1)
        unsqueeze_275: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        add_344: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_482, unsqueeze_275);  mul_482 = unsqueeze_275 = None
        convert_element_type_207: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_344, torch.bfloat16);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_68: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_207);  convert_element_type_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:144 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_6: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.avg_pool2d.default(cat_6, [3, 3], [1, 1], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_208: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_416, torch.bfloat16);  primals_416 = None
        convolution_69: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_6, convert_element_type_208, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_345: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_417, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_209: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_69, torch.float32)
        var_mean_69 = torch.ops.aten.var_mean.correction(convert_element_type_209, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_209 = None
        getitem_144: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_69[0]
        getitem_145: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_69[1];  var_mean_69 = None
        add_346: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_144, 0.001)
        rsqrt_69: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_346);  add_346 = None
        sub_69: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_69, getitem_145)
        mul_483: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = None
        squeeze_207: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3])
        mul_484: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_207, 0.1);  squeeze_207 = None
        mul_485: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
        add_347: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
        squeeze_209: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
        mul_486: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_209, 1.0000270336027683);  squeeze_209 = None
        mul_487: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_486, 0.1);  mul_486 = None
        mul_488: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
        add_348: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
        unsqueeze_276: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_420, -1)
        unsqueeze_277: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_489: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_483, unsqueeze_277);  mul_483 = unsqueeze_277 = None
        unsqueeze_278: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_421, -1)
        unsqueeze_279: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_349: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_489, unsqueeze_279);  mul_489 = unsqueeze_279 = None
        convert_element_type_210: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_349, torch.bfloat16);  add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_69: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_210);  convert_element_type_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:152 in forward, code: return torch.cat(outputs, 1)
        cat_7: "bf16[128, 768, 17, 17][221952, 1, 13056, 768]cuda:0" = torch.ops.aten.cat.default([relu_60, relu_63, relu_68, relu_69], 1);  relu_60 = relu_63 = relu_68 = relu_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_211: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_422, torch.bfloat16);  primals_422 = None
        convolution_70: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(cat_7, convert_element_type_211, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_350: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_423, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_212: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_70, torch.float32)
        var_mean_70 = torch.ops.aten.var_mean.correction(convert_element_type_212, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_212 = None
        getitem_146: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_70[0]
        getitem_147: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_70[1];  var_mean_70 = None
        add_351: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_146, 0.001)
        rsqrt_70: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_351);  add_351 = None
        sub_70: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_70, getitem_147)
        mul_490: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = None
        squeeze_210: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
        squeeze_211: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
        mul_491: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_210, 0.1)
        mul_492: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
        add_352: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
        squeeze_212: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
        mul_493: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_212, 1.0000270336027683);  squeeze_212 = None
        mul_494: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_493, 0.1);  mul_493 = None
        mul_495: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
        add_353: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
        unsqueeze_280: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_426, -1)
        unsqueeze_281: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        mul_496: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_490, unsqueeze_281);  mul_490 = unsqueeze_281 = None
        unsqueeze_282: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_427, -1);  primals_427 = None
        unsqueeze_283: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        add_354: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_496, unsqueeze_283);  mul_496 = unsqueeze_283 = None
        convert_element_type_213: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_354, torch.bfloat16);  add_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_70: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_213);  convert_element_type_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_214: "bf16[320, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_428, torch.bfloat16);  primals_428 = None
        convolution_71: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.convolution.default(relu_70, convert_element_type_214, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_355: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_429, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_215: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_71, torch.float32)
        var_mean_71 = torch.ops.aten.var_mean.correction(convert_element_type_215, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_215 = None
        getitem_148: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_71[0]
        getitem_149: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_71[1];  var_mean_71 = None
        add_356: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_148, 0.001)
        rsqrt_71: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_356);  add_356 = None
        sub_71: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convolution_71, getitem_149)
        mul_497: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = None
        squeeze_213: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3])
        mul_498: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_213, 0.1);  squeeze_213 = None
        mul_499: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
        add_357: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_498, mul_499);  mul_498 = mul_499 = None
        squeeze_215: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
        mul_500: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_215, 1.0001220852154804);  squeeze_215 = None
        mul_501: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_500, 0.1);  mul_500 = None
        mul_502: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
        add_358: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
        unsqueeze_284: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_432, -1)
        unsqueeze_285: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_503: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_497, unsqueeze_285);  mul_497 = unsqueeze_285 = None
        unsqueeze_286: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_433, -1)
        unsqueeze_287: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_359: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_503, unsqueeze_287);  mul_503 = unsqueeze_287 = None
        convert_element_type_216: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_359, torch.bfloat16);  add_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_71: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.relu.default(convert_element_type_216);  convert_element_type_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_217: "bf16[192, 768, 1, 1][768, 1, 768, 768]cuda:0" = torch.ops.prims.convert_element_type.default(primals_434, torch.bfloat16);  primals_434 = None
        convolution_72: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(cat_7, convert_element_type_217, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_360: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_435, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_218: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_72, torch.float32)
        var_mean_72 = torch.ops.aten.var_mean.correction(convert_element_type_218, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_218 = None
        getitem_150: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_72[0]
        getitem_151: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_72[1];  var_mean_72 = None
        add_361: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_150, 0.001)
        rsqrt_72: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_361);  add_361 = None
        sub_72: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_72, getitem_151)
        mul_504: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = None
        squeeze_216: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
        squeeze_217: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_72, [0, 2, 3]);  rsqrt_72 = None
        mul_505: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_216, 0.1)
        mul_506: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_436, 0.9)
        add_362: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
        squeeze_218: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
        mul_507: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_218, 1.0000270336027683);  squeeze_218 = None
        mul_508: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_507, 0.1);  mul_507 = None
        mul_509: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_437, 0.9)
        add_363: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_508, mul_509);  mul_508 = mul_509 = None
        unsqueeze_288: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_438, -1)
        unsqueeze_289: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        mul_510: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_504, unsqueeze_289);  mul_504 = unsqueeze_289 = None
        unsqueeze_290: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_439, -1);  primals_439 = None
        unsqueeze_291: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        add_364: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_510, unsqueeze_291);  mul_510 = unsqueeze_291 = None
        convert_element_type_219: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_364, torch.bfloat16);  add_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_72: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_219);  convert_element_type_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_220: "bf16[192, 192, 1, 7][1344, 1, 1344, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_440, torch.bfloat16);  primals_440 = None
        convolution_73: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_72, convert_element_type_220, None, [1, 1], [0, 3], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_365: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_441, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_221: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_73, torch.float32)
        var_mean_73 = torch.ops.aten.var_mean.correction(convert_element_type_221, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_221 = None
        getitem_152: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_73[0]
        getitem_153: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_73[1];  var_mean_73 = None
        add_366: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_152, 0.001)
        rsqrt_73: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_366);  add_366 = None
        sub_73: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_73, getitem_153)
        mul_511: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = None
        squeeze_219: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        squeeze_220: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
        mul_512: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_219, 0.1)
        mul_513: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
        add_367: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
        squeeze_221: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
        mul_514: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_221, 1.0000270336027683);  squeeze_221 = None
        mul_515: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_514, 0.1);  mul_514 = None
        mul_516: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
        add_368: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_515, mul_516);  mul_515 = mul_516 = None
        unsqueeze_292: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_444, -1)
        unsqueeze_293: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_517: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_511, unsqueeze_293);  mul_511 = unsqueeze_293 = None
        unsqueeze_294: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_445, -1);  primals_445 = None
        unsqueeze_295: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_369: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_517, unsqueeze_295);  mul_517 = unsqueeze_295 = None
        convert_element_type_222: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_369, torch.bfloat16);  add_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_73: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_222);  convert_element_type_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_223: "bf16[192, 192, 7, 1][1344, 1, 192, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_446, torch.bfloat16);  primals_446 = None
        convolution_74: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.convolution.default(relu_73, convert_element_type_223, None, [1, 1], [3, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_370: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_447, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_224: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_74, torch.float32)
        var_mean_74 = torch.ops.aten.var_mean.correction(convert_element_type_224, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_224 = None
        getitem_154: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_74[0]
        getitem_155: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_74[1];  var_mean_74 = None
        add_371: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_154, 0.001)
        rsqrt_74: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_371);  add_371 = None
        sub_74: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_74, getitem_155)
        mul_518: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = None
        squeeze_222: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
        squeeze_223: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
        mul_519: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_222, 0.1)
        mul_520: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
        add_372: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
        squeeze_224: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
        mul_521: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_224, 1.0000270336027683);  squeeze_224 = None
        mul_522: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_521, 0.1);  mul_521 = None
        mul_523: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
        add_373: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
        unsqueeze_296: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_450, -1)
        unsqueeze_297: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        mul_524: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_518, unsqueeze_297);  mul_518 = unsqueeze_297 = None
        unsqueeze_298: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_451, -1);  primals_451 = None
        unsqueeze_299: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        add_374: "f32[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_524, unsqueeze_299);  mul_524 = unsqueeze_299 = None
        convert_element_type_225: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_374, torch.bfloat16);  add_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_74: "bf16[128, 192, 17, 17][55488, 1, 3264, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_225);  convert_element_type_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_226: "bf16[192, 192, 3, 3][1728, 1, 576, 192]cuda:0" = torch.ops.prims.convert_element_type.default(primals_452, torch.bfloat16);  primals_452 = None
        convolution_75: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.convolution.default(relu_74, convert_element_type_226, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_375: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_453, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_227: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_75, torch.float32)
        var_mean_75 = torch.ops.aten.var_mean.correction(convert_element_type_227, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_227 = None
        getitem_156: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_75[0]
        getitem_157: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_75[1];  var_mean_75 = None
        add_376: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_156, 0.001)
        rsqrt_75: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_376);  add_376 = None
        sub_75: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_75, getitem_157)
        mul_525: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = None
        squeeze_225: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3])
        mul_526: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_225, 0.1);  squeeze_225 = None
        mul_527: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
        add_377: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_526, mul_527);  mul_526 = mul_527 = None
        squeeze_227: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
        mul_528: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_227, 1.0001220852154804);  squeeze_227 = None
        mul_529: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_528, 0.1);  mul_528 = None
        mul_530: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
        add_378: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
        unsqueeze_300: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_456, -1)
        unsqueeze_301: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_531: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, unsqueeze_301);  mul_525 = unsqueeze_301 = None
        unsqueeze_302: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_457, -1)
        unsqueeze_303: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_379: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_531, unsqueeze_303);  mul_531 = unsqueeze_303 = None
        convert_element_type_228: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_379, torch.bfloat16);  add_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_75: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_228);  convert_element_type_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:184 in _forward, code: branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        _low_memory_max_pool_with_offsets_3 = torch.ops.prims._low_memory_max_pool_with_offsets.default(cat_7, [3, 3], [2, 2], [0, 0], [1, 1], False)
        getitem_158: "bf16[128, 768, 8, 8][49152, 1, 6144, 768]cuda:0" = _low_memory_max_pool_with_offsets_3[0]
        getitem_159: "i8[128, 768, 8, 8][49152, 1, 6144, 768]cuda:0" = _low_memory_max_pool_with_offsets_3[1];  _low_memory_max_pool_with_offsets_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:190 in forward, code: return torch.cat(outputs, 1)
        cat_8: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = torch.ops.aten.cat.default([relu_71, relu_75, getitem_158], 1);  relu_71 = relu_75 = getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_229: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_458, torch.bfloat16);  primals_458 = None
        convolution_76: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.convolution.default(cat_8, convert_element_type_229, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_380: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_459, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_230: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_76, torch.float32)
        var_mean_76 = torch.ops.aten.var_mean.correction(convert_element_type_230, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_230 = None
        getitem_160: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_76[0]
        getitem_161: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_76[1];  var_mean_76 = None
        add_381: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_160, 0.001)
        rsqrt_76: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_381);  add_381 = None
        sub_76: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convolution_76, getitem_161)
        mul_532: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = None
        squeeze_228: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_161, [0, 2, 3])
        mul_533: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_228, 0.1);  squeeze_228 = None
        mul_534: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
        add_382: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
        squeeze_230: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_160, [0, 2, 3]);  getitem_160 = None
        mul_535: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_230, 1.0001220852154804);  squeeze_230 = None
        mul_536: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_535, 0.1);  mul_535 = None
        mul_537: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_461, 0.9)
        add_383: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
        unsqueeze_304: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_462, -1)
        unsqueeze_305: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        mul_538: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_532, unsqueeze_305);  mul_532 = unsqueeze_305 = None
        unsqueeze_306: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_463, -1)
        unsqueeze_307: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        add_384: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_538, unsqueeze_307);  mul_538 = unsqueeze_307 = None
        convert_element_type_231: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_384, torch.bfloat16);  add_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_76: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.relu.default(convert_element_type_231);  convert_element_type_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_232: "bf16[384, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_464, torch.bfloat16);  primals_464 = None
        convolution_77: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(cat_8, convert_element_type_232, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_385: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_465, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_233: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_77, torch.float32)
        var_mean_77 = torch.ops.aten.var_mean.correction(convert_element_type_233, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_233 = None
        getitem_162: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_77[0]
        getitem_163: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_77[1];  var_mean_77 = None
        add_386: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_162, 0.001)
        rsqrt_77: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_386);  add_386 = None
        sub_77: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_77, getitem_163)
        mul_539: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = None
        squeeze_231: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_163, [0, 2, 3]);  getitem_163 = None
        squeeze_232: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
        mul_540: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_231, 0.1)
        mul_541: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
        add_387: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
        squeeze_233: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_162, [0, 2, 3]);  getitem_162 = None
        mul_542: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_233, 1.0001220852154804);  squeeze_233 = None
        mul_543: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_542, 0.1);  mul_542 = None
        mul_544: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_467, 0.9)
        add_388: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
        unsqueeze_308: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_468, -1)
        unsqueeze_309: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_545: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_539, unsqueeze_309);  mul_539 = unsqueeze_309 = None
        unsqueeze_310: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_469, -1);  primals_469 = None
        unsqueeze_311: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_389: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_545, unsqueeze_311);  mul_545 = unsqueeze_311 = None
        convert_element_type_234: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_389, torch.bfloat16);  add_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_77: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_234);  convert_element_type_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_235: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_470, torch.bfloat16);  primals_470 = None
        convolution_78: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_77, convert_element_type_235, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_390: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_471, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_236: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_78, torch.float32)
        var_mean_78 = torch.ops.aten.var_mean.correction(convert_element_type_236, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_236 = None
        getitem_164: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_78[0]
        getitem_165: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_78[1];  var_mean_78 = None
        add_391: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_164, 0.001)
        rsqrt_78: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_391);  add_391 = None
        sub_78: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_78, getitem_165)
        mul_546: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = None
        squeeze_234: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_165, [0, 2, 3])
        mul_547: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_234, 0.1);  squeeze_234 = None
        mul_548: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
        add_392: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
        squeeze_236: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
        mul_549: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_236, 1.0001220852154804);  squeeze_236 = None
        mul_550: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_549, 0.1);  mul_549 = None
        mul_551: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_473, 0.9)
        add_393: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
        unsqueeze_312: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_474, -1)
        unsqueeze_313: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        mul_552: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_546, unsqueeze_313);  mul_546 = unsqueeze_313 = None
        unsqueeze_314: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_475, -1)
        unsqueeze_315: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        add_394: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_552, unsqueeze_315);  mul_552 = unsqueeze_315 = None
        convert_element_type_237: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_394, torch.bfloat16);  add_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_78: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_237);  convert_element_type_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_238: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_476, torch.bfloat16);  primals_476 = None
        convolution_79: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_77, convert_element_type_238, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_395: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_477, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_239: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_79, torch.float32)
        var_mean_79 = torch.ops.aten.var_mean.correction(convert_element_type_239, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_239 = None
        getitem_166: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_79[0]
        getitem_167: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_79[1];  var_mean_79 = None
        add_396: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_166, 0.001)
        rsqrt_79: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_396);  add_396 = None
        sub_79: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_79, getitem_167)
        mul_553: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = None
        squeeze_237: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_167, [0, 2, 3])
        mul_554: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_237, 0.1);  squeeze_237 = None
        mul_555: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
        add_397: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
        squeeze_239: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
        mul_556: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_239, 1.0001220852154804);  squeeze_239 = None
        mul_557: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_556, 0.1);  mul_556 = None
        mul_558: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_479, 0.9)
        add_398: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
        unsqueeze_316: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_480, -1)
        unsqueeze_317: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_559: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_553, unsqueeze_317);  mul_553 = unsqueeze_317 = None
        unsqueeze_318: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_481, -1)
        unsqueeze_319: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_399: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_559, unsqueeze_319);  mul_559 = unsqueeze_319 = None
        convert_element_type_240: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_399, torch.bfloat16);  add_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_79: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_240);  convert_element_type_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:226 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        cat_9: "bf16[128, 768, 8, 8][49152, 1, 6144, 768]cuda:0" = torch.ops.aten.cat.default([relu_78, relu_79], 1);  relu_78 = relu_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_241: "bf16[448, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_482, torch.bfloat16);  primals_482 = None
        convolution_80: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.convolution.default(cat_8, convert_element_type_241, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_400: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_483, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_242: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_80, torch.float32)
        var_mean_80 = torch.ops.aten.var_mean.correction(convert_element_type_242, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_242 = None
        getitem_168: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_80[0]
        getitem_169: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_80[1];  var_mean_80 = None
        add_401: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_168, 0.001)
        rsqrt_80: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_401);  add_401 = None
        sub_80: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.sub.Tensor(convolution_80, getitem_169)
        mul_560: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = None
        squeeze_240: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_169, [0, 2, 3]);  getitem_169 = None
        squeeze_241: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_80, [0, 2, 3]);  rsqrt_80 = None
        mul_561: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_240, 0.1)
        mul_562: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
        add_402: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_561, mul_562);  mul_561 = mul_562 = None
        squeeze_242: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
        mul_563: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_242, 1.0001220852154804);  squeeze_242 = None
        mul_564: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_563, 0.1);  mul_563 = None
        mul_565: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_485, 0.9)
        add_403: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
        unsqueeze_320: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_486, -1)
        unsqueeze_321: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        mul_566: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(mul_560, unsqueeze_321);  mul_560 = unsqueeze_321 = None
        unsqueeze_322: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_487, -1);  primals_487 = None
        unsqueeze_323: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        add_404: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.add.Tensor(mul_566, unsqueeze_323);  mul_566 = unsqueeze_323 = None
        convert_element_type_243: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(add_404, torch.bfloat16);  add_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_80: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.relu.default(convert_element_type_243);  convert_element_type_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_244: "bf16[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0" = torch.ops.prims.convert_element_type.default(primals_488, torch.bfloat16);  primals_488 = None
        convolution_81: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_80, convert_element_type_244, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_405: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_489, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_245: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_81, torch.float32)
        var_mean_81 = torch.ops.aten.var_mean.correction(convert_element_type_245, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_245 = None
        getitem_170: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_81[0]
        getitem_171: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_81[1];  var_mean_81 = None
        add_406: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_170, 0.001)
        rsqrt_81: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_406);  add_406 = None
        sub_81: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_81, getitem_171)
        mul_567: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_81);  sub_81 = None
        squeeze_243: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_171, [0, 2, 3]);  getitem_171 = None
        squeeze_244: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_81, [0, 2, 3]);  rsqrt_81 = None
        mul_568: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_243, 0.1)
        mul_569: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_490, 0.9)
        add_407: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
        squeeze_245: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_170, [0, 2, 3]);  getitem_170 = None
        mul_570: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_245, 1.0001220852154804);  squeeze_245 = None
        mul_571: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_570, 0.1);  mul_570 = None
        mul_572: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_491, 0.9)
        add_408: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
        unsqueeze_324: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_492, -1)
        unsqueeze_325: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_573: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_567, unsqueeze_325);  mul_567 = unsqueeze_325 = None
        unsqueeze_326: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_493, -1);  primals_493 = None
        unsqueeze_327: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_409: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_573, unsqueeze_327);  mul_573 = unsqueeze_327 = None
        convert_element_type_246: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_409, torch.bfloat16);  add_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_81: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_246);  convert_element_type_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_247: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_494, torch.bfloat16);  primals_494 = None
        convolution_82: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_81, convert_element_type_247, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_410: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_495, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_248: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_82, torch.float32)
        var_mean_82 = torch.ops.aten.var_mean.correction(convert_element_type_248, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_248 = None
        getitem_172: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_82[0]
        getitem_173: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_82[1];  var_mean_82 = None
        add_411: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_172, 0.001)
        rsqrt_82: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_411);  add_411 = None
        sub_82: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_82, getitem_173)
        mul_574: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_82);  sub_82 = None
        squeeze_246: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_173, [0, 2, 3])
        mul_575: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_246, 0.1);  squeeze_246 = None
        mul_576: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_496, 0.9)
        add_412: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
        squeeze_248: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_172, [0, 2, 3]);  getitem_172 = None
        mul_577: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_248, 1.0001220852154804);  squeeze_248 = None
        mul_578: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_577, 0.1);  mul_577 = None
        mul_579: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_497, 0.9)
        add_413: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_578, mul_579);  mul_578 = mul_579 = None
        unsqueeze_328: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_498, -1)
        unsqueeze_329: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        mul_580: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_574, unsqueeze_329);  mul_574 = unsqueeze_329 = None
        unsqueeze_330: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_499, -1)
        unsqueeze_331: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        add_414: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_580, unsqueeze_331);  mul_580 = unsqueeze_331 = None
        convert_element_type_249: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_414, torch.bfloat16);  add_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_82: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_249);  convert_element_type_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_250: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_500, torch.bfloat16);  primals_500 = None
        convolution_83: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_81, convert_element_type_250, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_415: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_501, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_251: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_83, torch.float32)
        var_mean_83 = torch.ops.aten.var_mean.correction(convert_element_type_251, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_251 = None
        getitem_174: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_83[0]
        getitem_175: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_83[1];  var_mean_83 = None
        add_416: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_174, 0.001)
        rsqrt_83: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_416);  add_416 = None
        sub_83: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_83, getitem_175)
        mul_581: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_83);  sub_83 = None
        squeeze_249: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_175, [0, 2, 3])
        mul_582: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_249, 0.1);  squeeze_249 = None
        mul_583: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_502, 0.9)
        add_417: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
        squeeze_251: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_174, [0, 2, 3]);  getitem_174 = None
        mul_584: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_251, 1.0001220852154804);  squeeze_251 = None
        mul_585: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_584, 0.1);  mul_584 = None
        mul_586: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_503, 0.9)
        add_418: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
        unsqueeze_332: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_504, -1)
        unsqueeze_333: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_587: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, unsqueeze_333);  mul_581 = unsqueeze_333 = None
        unsqueeze_334: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_505, -1)
        unsqueeze_335: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_419: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_587, unsqueeze_335);  mul_587 = unsqueeze_335 = None
        convert_element_type_252: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_419, torch.bfloat16);  add_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_83: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_252);  convert_element_type_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:234 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        cat_10: "bf16[128, 768, 8, 8][49152, 1, 6144, 768]cuda:0" = torch.ops.aten.cat.default([relu_82, relu_83], 1);  relu_82 = relu_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:236 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_7: "bf16[128, 1280, 8, 8][81920, 1, 10240, 1280]cuda:0" = torch.ops.aten.avg_pool2d.default(cat_8, [3, 3], [1, 1], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_253: "bf16[192, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(primals_506, torch.bfloat16);  primals_506 = None
        convolution_84: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_7, convert_element_type_253, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_420: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_507, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_254: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_84, torch.float32)
        var_mean_84 = torch.ops.aten.var_mean.correction(convert_element_type_254, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_254 = None
        getitem_176: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_84[0]
        getitem_177: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_84[1];  var_mean_84 = None
        add_421: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_176, 0.001)
        rsqrt_84: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_421);  add_421 = None
        sub_84: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_84, getitem_177)
        mul_588: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_84);  sub_84 = None
        squeeze_252: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_177, [0, 2, 3])
        mul_589: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_252, 0.1);  squeeze_252 = None
        mul_590: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_508, 0.9)
        add_422: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
        squeeze_254: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_176, [0, 2, 3]);  getitem_176 = None
        mul_591: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_254, 1.0001220852154804);  squeeze_254 = None
        mul_592: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, 0.1);  mul_591 = None
        mul_593: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_509, 0.9)
        add_423: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        unsqueeze_336: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_510, -1)
        unsqueeze_337: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        mul_594: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_588, unsqueeze_337);  mul_588 = unsqueeze_337 = None
        unsqueeze_338: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_511, -1)
        unsqueeze_339: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        add_424: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_594, unsqueeze_339);  mul_594 = unsqueeze_339 = None
        convert_element_type_255: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_424, torch.bfloat16);  add_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_84: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_255);  convert_element_type_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        cat_11: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.cat.default([relu_76, cat_9, cat_10, relu_84], 1);  relu_76 = cat_9 = cat_10 = relu_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_256: "bf16[320, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(primals_512, torch.bfloat16);  primals_512 = None
        convolution_85: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.convolution.default(cat_11, convert_element_type_256, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_425: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_513, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_257: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_85, torch.float32)
        var_mean_85 = torch.ops.aten.var_mean.correction(convert_element_type_257, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_257 = None
        getitem_178: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_85[0]
        getitem_179: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = var_mean_85[1];  var_mean_85 = None
        add_426: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_178, 0.001)
        rsqrt_85: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_426);  add_426 = None
        sub_85: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.sub.Tensor(convolution_85, getitem_179)
        mul_595: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_85);  sub_85 = None
        squeeze_255: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_179, [0, 2, 3])
        mul_596: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_255, 0.1);  squeeze_255 = None
        mul_597: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_514, 0.9)
        add_427: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
        squeeze_257: "f32[320][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_178, [0, 2, 3]);  getitem_178 = None
        mul_598: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_257, 1.0001220852154804);  squeeze_257 = None
        mul_599: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_598, 0.1);  mul_598 = None
        mul_600: "f32[320][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_515, 0.9)
        add_428: "f32[320][1]cuda:0" = torch.ops.aten.add.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
        unsqueeze_340: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_516, -1)
        unsqueeze_341: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_601: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_595, unsqueeze_341);  mul_595 = unsqueeze_341 = None
        unsqueeze_342: "f32[320, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_517, -1)
        unsqueeze_343: "f32[320, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_429: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_601, unsqueeze_343);  mul_601 = unsqueeze_343 = None
        convert_element_type_258: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_429, torch.bfloat16);  add_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_85: "bf16[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.relu.default(convert_element_type_258);  convert_element_type_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_259: "bf16[384, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(primals_518, torch.bfloat16);  primals_518 = None
        convolution_86: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(cat_11, convert_element_type_259, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_430: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_519, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_260: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_86, torch.float32)
        var_mean_86 = torch.ops.aten.var_mean.correction(convert_element_type_260, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_260 = None
        getitem_180: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_86[0]
        getitem_181: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_86[1];  var_mean_86 = None
        add_431: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_180, 0.001)
        rsqrt_86: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_431);  add_431 = None
        sub_86: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_86, getitem_181)
        mul_602: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_86);  sub_86 = None
        squeeze_258: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_181, [0, 2, 3]);  getitem_181 = None
        squeeze_259: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_86, [0, 2, 3]);  rsqrt_86 = None
        mul_603: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_258, 0.1)
        mul_604: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_520, 0.9)
        add_432: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
        squeeze_260: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_180, [0, 2, 3]);  getitem_180 = None
        mul_605: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_260, 1.0001220852154804);  squeeze_260 = None
        mul_606: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_605, 0.1);  mul_605 = None
        mul_607: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_521, 0.9)
        add_433: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
        unsqueeze_344: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_522, -1)
        unsqueeze_345: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        mul_608: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_602, unsqueeze_345);  mul_602 = unsqueeze_345 = None
        unsqueeze_346: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_523, -1);  primals_523 = None
        unsqueeze_347: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        add_434: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_608, unsqueeze_347);  mul_608 = unsqueeze_347 = None
        convert_element_type_261: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_434, torch.bfloat16);  add_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_86: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_261);  convert_element_type_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_262: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_524, torch.bfloat16);  primals_524 = None
        convolution_87: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_86, convert_element_type_262, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_435: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_525, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_263: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_87, torch.float32)
        var_mean_87 = torch.ops.aten.var_mean.correction(convert_element_type_263, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_263 = None
        getitem_182: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_87[0]
        getitem_183: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_87[1];  var_mean_87 = None
        add_436: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_182, 0.001)
        rsqrt_87: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_436);  add_436 = None
        sub_87: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_87, getitem_183)
        mul_609: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_87);  sub_87 = None
        squeeze_261: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_183, [0, 2, 3])
        mul_610: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_261, 0.1);  squeeze_261 = None
        mul_611: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_526, 0.9)
        add_437: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
        squeeze_263: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_182, [0, 2, 3]);  getitem_182 = None
        mul_612: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_263, 1.0001220852154804);  squeeze_263 = None
        mul_613: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_612, 0.1);  mul_612 = None
        mul_614: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_527, 0.9)
        add_438: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
        unsqueeze_348: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_528, -1)
        unsqueeze_349: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_615: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_609, unsqueeze_349);  mul_609 = unsqueeze_349 = None
        unsqueeze_350: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_529, -1)
        unsqueeze_351: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_439: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_615, unsqueeze_351);  mul_615 = unsqueeze_351 = None
        convert_element_type_264: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_439, torch.bfloat16);  add_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_87: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_264);  convert_element_type_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_265: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_530, torch.bfloat16);  primals_530 = None
        convolution_88: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_86, convert_element_type_265, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_440: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_531, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_266: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_88, torch.float32)
        var_mean_88 = torch.ops.aten.var_mean.correction(convert_element_type_266, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_266 = None
        getitem_184: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_88[0]
        getitem_185: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_88[1];  var_mean_88 = None
        add_441: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_184, 0.001)
        rsqrt_88: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_441);  add_441 = None
        sub_88: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_88, getitem_185)
        mul_616: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_88);  sub_88 = None
        squeeze_264: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_185, [0, 2, 3])
        mul_617: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_264, 0.1);  squeeze_264 = None
        mul_618: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_532, 0.9)
        add_442: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
        squeeze_266: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_184, [0, 2, 3]);  getitem_184 = None
        mul_619: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_266, 1.0001220852154804);  squeeze_266 = None
        mul_620: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_619, 0.1);  mul_619 = None
        mul_621: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_533, 0.9)
        add_443: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
        unsqueeze_352: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_534, -1)
        unsqueeze_353: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        mul_622: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_616, unsqueeze_353);  mul_616 = unsqueeze_353 = None
        unsqueeze_354: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_535, -1)
        unsqueeze_355: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        add_444: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_622, unsqueeze_355);  mul_622 = unsqueeze_355 = None
        convert_element_type_267: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_444, torch.bfloat16);  add_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_88: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_267);  convert_element_type_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:226 in _forward, code: branch3x3 = torch.cat(branch3x3, 1)
        cat_12: "bf16[128, 768, 8, 8][49152, 1, 6144, 768]cuda:0" = torch.ops.aten.cat.default([relu_87, relu_88], 1);  relu_87 = relu_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_268: "bf16[448, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(primals_536, torch.bfloat16);  primals_536 = None
        convolution_89: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.convolution.default(cat_11, convert_element_type_268, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_445: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_537, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_269: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_89, torch.float32)
        var_mean_89 = torch.ops.aten.var_mean.correction(convert_element_type_269, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_269 = None
        getitem_186: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_89[0]
        getitem_187: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = var_mean_89[1];  var_mean_89 = None
        add_446: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_186, 0.001)
        rsqrt_89: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_446);  add_446 = None
        sub_89: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.sub.Tensor(convolution_89, getitem_187)
        mul_623: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_89);  sub_89 = None
        squeeze_267: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_187, [0, 2, 3]);  getitem_187 = None
        squeeze_268: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_89, [0, 2, 3]);  rsqrt_89 = None
        mul_624: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_267, 0.1)
        mul_625: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_538, 0.9)
        add_447: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
        squeeze_269: "f32[448][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_186, [0, 2, 3]);  getitem_186 = None
        mul_626: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_269, 1.0001220852154804);  squeeze_269 = None
        mul_627: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_626, 0.1);  mul_626 = None
        mul_628: "f32[448][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_539, 0.9)
        add_448: "f32[448][1]cuda:0" = torch.ops.aten.add.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
        unsqueeze_356: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_540, -1)
        unsqueeze_357: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_629: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.mul.Tensor(mul_623, unsqueeze_357);  mul_623 = unsqueeze_357 = None
        unsqueeze_358: "f32[448, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_541, -1);  primals_541 = None
        unsqueeze_359: "f32[448, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_449: "f32[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.add.Tensor(mul_629, unsqueeze_359);  mul_629 = unsqueeze_359 = None
        convert_element_type_270: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.prims.convert_element_type.default(add_449, torch.bfloat16);  add_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_89: "bf16[128, 448, 8, 8][28672, 1, 3584, 448]cuda:0" = torch.ops.aten.relu.default(convert_element_type_270);  convert_element_type_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_271: "bf16[384, 448, 3, 3][4032, 1, 1344, 448]cuda:0" = torch.ops.prims.convert_element_type.default(primals_542, torch.bfloat16);  primals_542 = None
        convolution_90: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_89, convert_element_type_271, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_450: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_543, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_272: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_90, torch.float32)
        var_mean_90 = torch.ops.aten.var_mean.correction(convert_element_type_272, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_272 = None
        getitem_188: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_90[0]
        getitem_189: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_90[1];  var_mean_90 = None
        add_451: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_188, 0.001)
        rsqrt_90: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_451);  add_451 = None
        sub_90: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_90, getitem_189)
        mul_630: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_90);  sub_90 = None
        squeeze_270: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_189, [0, 2, 3]);  getitem_189 = None
        squeeze_271: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_90, [0, 2, 3]);  rsqrt_90 = None
        mul_631: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_270, 0.1)
        mul_632: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_544, 0.9)
        add_452: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
        squeeze_272: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_188, [0, 2, 3]);  getitem_188 = None
        mul_633: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_272, 1.0001220852154804);  squeeze_272 = None
        mul_634: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_633, 0.1);  mul_633 = None
        mul_635: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_545, 0.9)
        add_453: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_634, mul_635);  mul_634 = mul_635 = None
        unsqueeze_360: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_546, -1)
        unsqueeze_361: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        mul_636: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, unsqueeze_361);  mul_630 = unsqueeze_361 = None
        unsqueeze_362: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_547, -1);  primals_547 = None
        unsqueeze_363: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        add_454: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_636, unsqueeze_363);  mul_636 = unsqueeze_363 = None
        convert_element_type_273: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_454, torch.bfloat16);  add_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_90: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_273);  convert_element_type_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_274: "bf16[384, 384, 1, 3][1152, 1, 1152, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_548, torch.bfloat16);  primals_548 = None
        convolution_91: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_90, convert_element_type_274, None, [1, 1], [0, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_455: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_549, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_275: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_91, torch.float32)
        var_mean_91 = torch.ops.aten.var_mean.correction(convert_element_type_275, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_275 = None
        getitem_190: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_91[0]
        getitem_191: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_91[1];  var_mean_91 = None
        add_456: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_190, 0.001)
        rsqrt_91: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_456);  add_456 = None
        sub_91: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_91, getitem_191)
        mul_637: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_91);  sub_91 = None
        squeeze_273: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_191, [0, 2, 3])
        mul_638: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_273, 0.1);  squeeze_273 = None
        mul_639: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_550, 0.9)
        add_457: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
        squeeze_275: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_190, [0, 2, 3]);  getitem_190 = None
        mul_640: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_275, 1.0001220852154804);  squeeze_275 = None
        mul_641: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_640, 0.1);  mul_640 = None
        mul_642: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_551, 0.9)
        add_458: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
        unsqueeze_364: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_552, -1)
        unsqueeze_365: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_643: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_637, unsqueeze_365);  mul_637 = unsqueeze_365 = None
        unsqueeze_366: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_553, -1)
        unsqueeze_367: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_459: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_643, unsqueeze_367);  mul_643 = unsqueeze_367 = None
        convert_element_type_276: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_459, torch.bfloat16);  add_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_91: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_276);  convert_element_type_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_277: "bf16[384, 384, 3, 1][1152, 1, 384, 384]cuda:0" = torch.ops.prims.convert_element_type.default(primals_554, torch.bfloat16);  primals_554 = None
        convolution_92: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.convolution.default(relu_90, convert_element_type_277, None, [1, 1], [1, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_460: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_555, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_278: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_92, torch.float32)
        var_mean_92 = torch.ops.aten.var_mean.correction(convert_element_type_278, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_278 = None
        getitem_192: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_92[0]
        getitem_193: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = var_mean_92[1];  var_mean_92 = None
        add_461: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_192, 0.001)
        rsqrt_92: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_461);  add_461 = None
        sub_92: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_92, getitem_193)
        mul_644: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_92);  sub_92 = None
        squeeze_276: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_193, [0, 2, 3])
        mul_645: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_276, 0.1);  squeeze_276 = None
        mul_646: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_556, 0.9)
        add_462: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
        squeeze_278: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_192, [0, 2, 3]);  getitem_192 = None
        mul_647: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_278, 1.0001220852154804);  squeeze_278 = None
        mul_648: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_647, 0.1);  mul_647 = None
        mul_649: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_557, 0.9)
        add_463: "f32[384][1]cuda:0" = torch.ops.aten.add.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
        unsqueeze_368: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_558, -1)
        unsqueeze_369: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        mul_650: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_644, unsqueeze_369);  mul_644 = unsqueeze_369 = None
        unsqueeze_370: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_559, -1)
        unsqueeze_371: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        add_464: "f32[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_650, unsqueeze_371);  mul_650 = unsqueeze_371 = None
        convert_element_type_279: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.prims.convert_element_type.default(add_464, torch.bfloat16);  add_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_92: "bf16[128, 384, 8, 8][24576, 1, 3072, 384]cuda:0" = torch.ops.aten.relu.default(convert_element_type_279);  convert_element_type_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:234 in _forward, code: branch3x3dbl = torch.cat(branch3x3dbl, 1)
        cat_13: "bf16[128, 768, 8, 8][49152, 1, 6144, 768]cuda:0" = torch.ops.aten.cat.default([relu_91, relu_92], 1);  relu_91 = relu_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:236 in _forward, code: branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool2d_8: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.avg_pool2d.default(cat_11, [3, 3], [1, 1], [1, 1])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convert_element_type_280: "bf16[192, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.prims.convert_element_type.default(primals_560, torch.bfloat16);  primals_560 = None
        convolution_93: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.convolution.default(avg_pool2d_8, convert_element_type_280, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_465: "i64[][]cuda:0" = torch.ops.aten.add.Tensor(primals_561, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        convert_element_type_281: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_93, torch.float32)
        var_mean_93 = torch.ops.aten.var_mean.correction(convert_element_type_281, [0, 2, 3], correction = 0, keepdim = True);  convert_element_type_281 = None
        getitem_194: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_93[0]
        getitem_195: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = var_mean_93[1];  var_mean_93 = None
        add_466: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_194, 0.001)
        rsqrt_93: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_466);  add_466 = None
        sub_93: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.sub.Tensor(convolution_93, getitem_195)
        mul_651: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_93);  sub_93 = None
        squeeze_279: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_195, [0, 2, 3])
        mul_652: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_279, 0.1);  squeeze_279 = None
        mul_653: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_562, 0.9)
        add_467: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
        squeeze_281: "f32[192][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_194, [0, 2, 3]);  getitem_194 = None
        mul_654: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_281, 1.0001220852154804);  squeeze_281 = None
        mul_655: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_654, 0.1);  mul_654 = None
        mul_656: "f32[192][1]cuda:0" = torch.ops.aten.mul.Tensor(primals_563, 0.9)
        add_468: "f32[192][1]cuda:0" = torch.ops.aten.add.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
        unsqueeze_372: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_564, -1)
        unsqueeze_373: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_657: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.mul.Tensor(mul_651, unsqueeze_373);  mul_651 = unsqueeze_373 = None
        unsqueeze_374: "f32[192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_565, -1)
        unsqueeze_375: "f32[192, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_469: "f32[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.add.Tensor(mul_657, unsqueeze_375);  mul_657 = unsqueeze_375 = None
        convert_element_type_282: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.prims.convert_element_type.default(add_469, torch.bfloat16);  add_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_93: "bf16[128, 192, 8, 8][12288, 1, 1536, 192]cuda:0" = torch.ops.aten.relu.default(convert_element_type_282);  convert_element_type_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:244 in forward, code: return torch.cat(outputs, 1)
        cat_14: "bf16[128, 2048, 8, 8][131072, 1, 16384, 2048]cuda:0" = torch.ops.aten.cat.default([relu_85, cat_12, cat_13, relu_93], 1);  relu_85 = cat_12 = cat_13 = relu_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean: "bf16[128, 2048, 1, 1][2048, 1, 1, 1]cuda:0" = torch.ops.aten.mean.dim(cat_14, [-1, -2], True);  cat_14 = None
        as_strided: "bf16[128, 2048, 1, 1][2048, 1, 2048, 2048]cuda:0" = torch.ops.aten.as_strided.default(mean, [128, 2048, 1, 1], [2048, 1, 2048, 2048]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view: "bf16[128, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided, [128, 2048]);  as_strided = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/inception_v3.py:430 in forward_head, code: x = self.fc(x)
        convert_element_type_283: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_567, torch.bfloat16);  primals_567 = None
        convert_element_type_284: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_566, torch.bfloat16);  primals_566 = None
        permute: "bf16[2048, 1000][1, 2048]cuda:0" = torch.ops.aten.permute.default(convert_element_type_284, [1, 0]);  convert_element_type_284 = None
        addmm: "bf16[128, 1000][1000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_283, view, permute);  convert_element_type_283 = None
        permute_1: "bf16[1000, 2048][2048, 1]cuda:0" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_412: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_270, 0);  squeeze_270 = None
        unsqueeze_413: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
        unsqueeze_424: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_267, 0);  squeeze_267 = None
        unsqueeze_425: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
        unsqueeze_426: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
        unsqueeze_460: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_258, 0);  squeeze_258 = None
        unsqueeze_461: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
        unsqueeze_462: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
        unsqueeze_520: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_243, 0);  squeeze_243 = None
        unsqueeze_521: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
        unsqueeze_532: "f32[1, 448][448, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_240, 0);  squeeze_240 = None
        unsqueeze_533: "f32[1, 448, 1][448, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
        unsqueeze_534: "f32[1, 448, 1, 1][448, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
        unsqueeze_568: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_231, 0);  squeeze_231 = None
        unsqueeze_569: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
        unsqueeze_570: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
        unsqueeze_604: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_222, 0);  squeeze_222 = None
        unsqueeze_605: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
        unsqueeze_606: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
        unsqueeze_616: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_219, 0);  squeeze_219 = None
        unsqueeze_617: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
        unsqueeze_618: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
        unsqueeze_628: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_216, 0);  squeeze_216 = None
        unsqueeze_629: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
        unsqueeze_630: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
        unsqueeze_652: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_210, 0);  squeeze_210 = None
        unsqueeze_653: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
        unsqueeze_654: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
        unsqueeze_688: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_201, 0);  squeeze_201 = None
        unsqueeze_689: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
        unsqueeze_690: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
        unsqueeze_700: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_198, 0);  squeeze_198 = None
        unsqueeze_701: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
        unsqueeze_702: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
        unsqueeze_712: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_195, 0);  squeeze_195 = None
        unsqueeze_713: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
        unsqueeze_714: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
        unsqueeze_724: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_192, 0);  squeeze_192 = None
        unsqueeze_725: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
        unsqueeze_726: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
        unsqueeze_748: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_186, 0);  squeeze_186 = None
        unsqueeze_749: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
        unsqueeze_750: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
        unsqueeze_760: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_183, 0);  squeeze_183 = None
        unsqueeze_761: "f32[1, 192, 1][192, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
        unsqueeze_762: "f32[1, 192, 1, 1][192, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
        unsqueeze_808: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
        unsqueeze_809: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
        unsqueeze_810: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
        unsqueeze_820: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_168, 0);  squeeze_168 = None
        unsqueeze_821: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
        unsqueeze_822: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
        unsqueeze_832: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
        unsqueeze_833: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
        unsqueeze_834: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
        unsqueeze_844: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
        unsqueeze_845: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
        unsqueeze_846: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
        unsqueeze_868: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_869: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
        unsqueeze_870: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
        unsqueeze_880: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_881: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
        unsqueeze_882: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
        unsqueeze_928: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
        unsqueeze_929: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_928, 2);  unsqueeze_928 = None
        unsqueeze_930: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 3);  unsqueeze_929 = None
        unsqueeze_940: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_941: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_940, 2);  unsqueeze_940 = None
        unsqueeze_942: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 3);  unsqueeze_941 = None
        unsqueeze_952: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_953: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_952, 2);  unsqueeze_952 = None
        unsqueeze_954: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 3);  unsqueeze_953 = None
        unsqueeze_964: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_132, 0);  squeeze_132 = None
        unsqueeze_965: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_964, 2);  unsqueeze_964 = None
        unsqueeze_966: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 3);  unsqueeze_965 = None
        unsqueeze_988: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_989: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_988, 2);  unsqueeze_988 = None
        unsqueeze_990: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 3);  unsqueeze_989 = None
        unsqueeze_1000: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_123, 0);  squeeze_123 = None
        unsqueeze_1001: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, 2);  unsqueeze_1000 = None
        unsqueeze_1002: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 3);  unsqueeze_1001 = None
        unsqueeze_1048: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_1049: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, 2);  unsqueeze_1048 = None
        unsqueeze_1050: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1049, 3);  unsqueeze_1049 = None
        unsqueeze_1060: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_1061: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, 2);  unsqueeze_1060 = None
        unsqueeze_1062: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1061, 3);  unsqueeze_1061 = None
        unsqueeze_1072: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
        unsqueeze_1073: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, 2);  unsqueeze_1072 = None
        unsqueeze_1074: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1073, 3);  unsqueeze_1073 = None
        unsqueeze_1084: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_1085: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, 2);  unsqueeze_1084 = None
        unsqueeze_1086: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1085, 3);  unsqueeze_1085 = None
        unsqueeze_1108: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_96, 0);  squeeze_96 = None
        unsqueeze_1109: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, 2);  unsqueeze_1108 = None
        unsqueeze_1110: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1109, 3);  unsqueeze_1109 = None
        unsqueeze_1120: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_1121: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, 2);  unsqueeze_1120 = None
        unsqueeze_1122: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1121, 3);  unsqueeze_1121 = None
        unsqueeze_1156: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_1157: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, 2);  unsqueeze_1156 = None
        unsqueeze_1158: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1157, 3);  unsqueeze_1157 = None
        unsqueeze_1168: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_1169: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, 2);  unsqueeze_1168 = None
        unsqueeze_1170: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1169, 3);  unsqueeze_1169 = None
        unsqueeze_1216: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_1217: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, 2);  unsqueeze_1216 = None
        unsqueeze_1218: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1217, 3);  unsqueeze_1217 = None
        unsqueeze_1228: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_1229: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, 2);  unsqueeze_1228 = None
        unsqueeze_1230: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1229, 3);  unsqueeze_1229 = None
        unsqueeze_1252: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_1253: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1252, 2);  unsqueeze_1252 = None
        unsqueeze_1254: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1253, 3);  unsqueeze_1253 = None
        unsqueeze_1300: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_1301: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1300, 2);  unsqueeze_1300 = None
        unsqueeze_1302: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1301, 3);  unsqueeze_1301 = None
        unsqueeze_1312: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_1313: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1312, 2);  unsqueeze_1312 = None
        unsqueeze_1314: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1313, 3);  unsqueeze_1313 = None
        unsqueeze_1336: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_1337: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1336, 2);  unsqueeze_1336 = None
        unsqueeze_1338: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1337, 3);  unsqueeze_1337 = None
        unsqueeze_1384: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_1385: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1384, 2);  unsqueeze_1384 = None
        unsqueeze_1386: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1385, 3);  unsqueeze_1385 = None
        unsqueeze_1396: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_1397: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1396, 2);  unsqueeze_1396 = None
        unsqueeze_1398: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1397, 3);  unsqueeze_1397 = None
        unsqueeze_1420: "f32[1, 48][48, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_1421: "f32[1, 48, 1][48, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1420, 2);  unsqueeze_1420 = None
        unsqueeze_1422: "f32[1, 48, 1, 1][48, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1421, 3);  unsqueeze_1421 = None
        unsqueeze_1456: "f32[1, 80][80, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_1457: "f32[1, 80, 1][80, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1456, 2);  unsqueeze_1456 = None
        unsqueeze_1458: "f32[1, 80, 1, 1][80, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1457, 3);  unsqueeze_1457 = None
        unsqueeze_1480: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_1481: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1480, 2);  unsqueeze_1480 = None
        unsqueeze_1482: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1481, 3);  unsqueeze_1481 = None
        unsqueeze_1492: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_1493: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1492, 2);  unsqueeze_1492 = None
        unsqueeze_1494: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_1493, 3);  unsqueeze_1493 = None

        # No stacktrace found for following nodes
        copy_: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_9, add_5);  primals_9 = add_5 = copy__3 = None
        copy__4: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_10, add_7);  primals_10 = add_7 = copy__4 = None
        copy__5: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_11, add_8);  primals_11 = add_8 = copy__5 = None
        copy__6: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_15, add_10);  primals_15 = add_10 = copy__6 = None
        copy__7: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_16, add_12);  primals_16 = add_12 = copy__7 = None
        copy__8: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__8 = None
        copy__9: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_21, add_15);  primals_21 = add_15 = copy__9 = None
        copy__10: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_22, add_17);  primals_22 = add_17 = copy__10 = None
        copy__11: "f32[80][1]cuda:0" = torch.ops.aten.copy_.default(primals_23, add_18);  primals_23 = add_18 = copy__11 = None
        copy__12: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_27, add_20);  primals_27 = add_20 = copy__12 = None
        copy__13: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_28, add_22);  primals_28 = add_22 = copy__13 = None
        copy__14: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_29, add_23);  primals_29 = add_23 = copy__14 = None
        copy__15: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_33, add_25);  primals_33 = add_25 = copy__15 = None
        copy__16: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_34, add_27);  primals_34 = add_27 = copy__16 = None
        copy__17: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_35, add_28);  primals_35 = add_28 = copy__17 = None
        copy__18: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_39, add_30);  primals_39 = add_30 = copy__18 = None
        copy__19: "f32[48][1]cuda:0" = torch.ops.aten.copy_.default(primals_40, add_32);  primals_40 = add_32 = copy__19 = None
        copy__20: "f32[48][1]cuda:0" = torch.ops.aten.copy_.default(primals_41, add_33);  primals_41 = add_33 = copy__20 = None
        copy__21: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_45, add_35);  primals_45 = add_35 = copy__21 = None
        copy__22: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_46, add_37);  primals_46 = add_37 = copy__22 = None
        copy__23: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_47, add_38);  primals_47 = add_38 = copy__23 = None
        copy__24: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_51, add_40);  primals_51 = add_40 = copy__24 = None
        copy__25: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_52, add_42);  primals_52 = add_42 = copy__25 = None
        copy__26: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_53, add_43);  primals_53 = add_43 = copy__26 = None
        copy__27: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_57, add_45);  primals_57 = add_45 = copy__27 = None
        copy__28: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_58, add_47);  primals_58 = add_47 = copy__28 = None
        copy__29: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_59, add_48);  primals_59 = add_48 = copy__29 = None
        copy__30: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_63, add_50);  primals_63 = add_50 = copy__30 = None
        copy__31: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_64, add_52);  primals_64 = add_52 = copy__31 = None
        copy__32: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_65, add_53);  primals_65 = add_53 = copy__32 = None
        copy__33: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_69, add_55);  primals_69 = add_55 = copy__33 = None
        copy__34: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_70, add_57);  primals_70 = add_57 = copy__34 = None
        copy__35: "f32[32][1]cuda:0" = torch.ops.aten.copy_.default(primals_71, add_58);  primals_71 = add_58 = copy__35 = None
        copy__36: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_75, add_60);  primals_75 = add_60 = copy__36 = None
        copy__37: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_76, add_62);  primals_76 = add_62 = copy__37 = None
        copy__38: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_77, add_63);  primals_77 = add_63 = copy__38 = None
        copy__39: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_81, add_65);  primals_81 = add_65 = copy__39 = None
        copy__40: "f32[48][1]cuda:0" = torch.ops.aten.copy_.default(primals_82, add_67);  primals_82 = add_67 = copy__40 = None
        copy__41: "f32[48][1]cuda:0" = torch.ops.aten.copy_.default(primals_83, add_68);  primals_83 = add_68 = copy__41 = None
        copy__42: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_87, add_70);  primals_87 = add_70 = copy__42 = None
        copy__43: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_88, add_72);  primals_88 = add_72 = copy__43 = None
        copy__44: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_89, add_73);  primals_89 = add_73 = copy__44 = None
        copy__45: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_93, add_75);  primals_93 = add_75 = copy__45 = None
        copy__46: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_94, add_77);  primals_94 = add_77 = copy__46 = None
        copy__47: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_95, add_78);  primals_95 = add_78 = copy__47 = None
        copy__48: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_99, add_80);  primals_99 = add_80 = copy__48 = None
        copy__49: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_100, add_82);  primals_100 = add_82 = copy__49 = None
        copy__50: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_101, add_83);  primals_101 = add_83 = copy__50 = None
        copy__51: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_105, add_85);  primals_105 = add_85 = copy__51 = None
        copy__52: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_106, add_87);  primals_106 = add_87 = copy__52 = None
        copy__53: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_107, add_88);  primals_107 = add_88 = copy__53 = None
        copy__54: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_111, add_90);  primals_111 = add_90 = copy__54 = None
        copy__55: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_112, add_92);  primals_112 = add_92 = copy__55 = None
        copy__56: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_113, add_93);  primals_113 = add_93 = copy__56 = None
        copy__57: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_117, add_95);  primals_117 = add_95 = copy__57 = None
        copy__58: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_118, add_97);  primals_118 = add_97 = copy__58 = None
        copy__59: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_119, add_98);  primals_119 = add_98 = copy__59 = None
        copy__60: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_123, add_100);  primals_123 = add_100 = copy__60 = None
        copy__61: "f32[48][1]cuda:0" = torch.ops.aten.copy_.default(primals_124, add_102);  primals_124 = add_102 = copy__61 = None
        copy__62: "f32[48][1]cuda:0" = torch.ops.aten.copy_.default(primals_125, add_103);  primals_125 = add_103 = copy__62 = None
        copy__63: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_129, add_105);  primals_129 = add_105 = copy__63 = None
        copy__64: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_130, add_107);  primals_130 = add_107 = copy__64 = None
        copy__65: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_131, add_108);  primals_131 = add_108 = copy__65 = None
        copy__66: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_135, add_110);  primals_135 = add_110 = copy__66 = None
        copy__67: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_136, add_112);  primals_136 = add_112 = copy__67 = None
        copy__68: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_137, add_113);  primals_137 = add_113 = copy__68 = None
        copy__69: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_141, add_115);  primals_141 = add_115 = copy__69 = None
        copy__70: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_142, add_117);  primals_142 = add_117 = copy__70 = None
        copy__71: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_143, add_118);  primals_143 = add_118 = copy__71 = None
        copy__72: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_147, add_120);  primals_147 = add_120 = copy__72 = None
        copy__73: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_148, add_122);  primals_148 = add_122 = copy__73 = None
        copy__74: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_149, add_123);  primals_149 = add_123 = copy__74 = None
        copy__75: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_153, add_125);  primals_153 = add_125 = copy__75 = None
        copy__76: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_154, add_127);  primals_154 = add_127 = copy__76 = None
        copy__77: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_155, add_128);  primals_155 = add_128 = copy__77 = None
        copy__78: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_159, add_130);  primals_159 = add_130 = copy__78 = None
        copy__79: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_160, add_132);  primals_160 = add_132 = copy__79 = None
        copy__80: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_161, add_133);  primals_161 = add_133 = copy__80 = None
        copy__81: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_165, add_135);  primals_165 = add_135 = copy__81 = None
        copy__82: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_166, add_137);  primals_166 = add_137 = copy__82 = None
        copy__83: "f32[64][1]cuda:0" = torch.ops.aten.copy_.default(primals_167, add_138);  primals_167 = add_138 = copy__83 = None
        copy__84: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_171, add_140);  primals_171 = add_140 = copy__84 = None
        copy__85: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_172, add_142);  primals_172 = add_142 = copy__85 = None
        copy__86: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_173, add_143);  primals_173 = add_143 = copy__86 = None
        copy__87: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_177, add_145);  primals_177 = add_145 = copy__87 = None
        copy__88: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_178, add_147);  primals_178 = add_147 = copy__88 = None
        copy__89: "f32[96][1]cuda:0" = torch.ops.aten.copy_.default(primals_179, add_148);  primals_179 = add_148 = copy__89 = None
        copy__90: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_183, add_150);  primals_183 = add_150 = copy__90 = None
        copy__91: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_184, add_152);  primals_184 = add_152 = copy__91 = None
        copy__92: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_185, add_153);  primals_185 = add_153 = copy__92 = None
        copy__93: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_189, add_155);  primals_189 = add_155 = copy__93 = None
        copy__94: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_190, add_157);  primals_190 = add_157 = copy__94 = None
        copy__95: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_191, add_158);  primals_191 = add_158 = copy__95 = None
        copy__96: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_195, add_160);  primals_195 = add_160 = copy__96 = None
        copy__97: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_196, add_162);  primals_196 = add_162 = copy__97 = None
        copy__98: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_197, add_163);  primals_197 = add_163 = copy__98 = None
        copy__99: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_201, add_165);  primals_201 = add_165 = copy__99 = None
        copy__100: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_202, add_167);  primals_202 = add_167 = copy__100 = None
        copy__101: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_203, add_168);  primals_203 = add_168 = copy__101 = None
        copy__102: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_207, add_170);  primals_207 = add_170 = copy__102 = None
        copy__103: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_208, add_172);  primals_208 = add_172 = copy__103 = None
        copy__104: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_209, add_173);  primals_209 = add_173 = copy__104 = None
        copy__105: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_213, add_175);  primals_213 = add_175 = copy__105 = None
        copy__106: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_214, add_177);  primals_214 = add_177 = copy__106 = None
        copy__107: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_215, add_178);  primals_215 = add_178 = copy__107 = None
        copy__108: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_219, add_180);  primals_219 = add_180 = copy__108 = None
        copy__109: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_220, add_182);  primals_220 = add_182 = copy__109 = None
        copy__110: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_221, add_183);  primals_221 = add_183 = copy__110 = None
        copy__111: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_225, add_185);  primals_225 = add_185 = copy__111 = None
        copy__112: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_226, add_187);  primals_226 = add_187 = copy__112 = None
        copy__113: "f32[128][1]cuda:0" = torch.ops.aten.copy_.default(primals_227, add_188);  primals_227 = add_188 = copy__113 = None
        copy__114: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_231, add_190);  primals_231 = add_190 = copy__114 = None
        copy__115: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_232, add_192);  primals_232 = add_192 = copy__115 = None
        copy__116: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_233, add_193);  primals_233 = add_193 = copy__116 = None
        copy__117: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_237, add_195);  primals_237 = add_195 = copy__117 = None
        copy__118: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_238, add_197);  primals_238 = add_197 = copy__118 = None
        copy__119: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_239, add_198);  primals_239 = add_198 = copy__119 = None
        copy__120: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_243, add_200);  primals_243 = add_200 = copy__120 = None
        copy__121: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_244, add_202);  primals_244 = add_202 = copy__121 = None
        copy__122: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_245, add_203);  primals_245 = add_203 = copy__122 = None
        copy__123: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_249, add_205);  primals_249 = add_205 = copy__123 = None
        copy__124: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_250, add_207);  primals_250 = add_207 = copy__124 = None
        copy__125: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_251, add_208);  primals_251 = add_208 = copy__125 = None
        copy__126: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_255, add_210);  primals_255 = add_210 = copy__126 = None
        copy__127: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_256, add_212);  primals_256 = add_212 = copy__127 = None
        copy__128: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_257, add_213);  primals_257 = add_213 = copy__128 = None
        copy__129: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_261, add_215);  primals_261 = add_215 = copy__129 = None
        copy__130: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_262, add_217);  primals_262 = add_217 = copy__130 = None
        copy__131: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_263, add_218);  primals_263 = add_218 = copy__131 = None
        copy__132: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_267, add_220);  primals_267 = add_220 = copy__132 = None
        copy__133: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_268, add_222);  primals_268 = add_222 = copy__133 = None
        copy__134: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_269, add_223);  primals_269 = add_223 = copy__134 = None
        copy__135: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_273, add_225);  primals_273 = add_225 = copy__135 = None
        copy__136: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_274, add_227);  primals_274 = add_227 = copy__136 = None
        copy__137: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_275, add_228);  primals_275 = add_228 = copy__137 = None
        copy__138: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_279, add_230);  primals_279 = add_230 = copy__138 = None
        copy__139: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_280, add_232);  primals_280 = add_232 = copy__139 = None
        copy__140: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_281, add_233);  primals_281 = add_233 = copy__140 = None
        copy__141: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_285, add_235);  primals_285 = add_235 = copy__141 = None
        copy__142: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_286, add_237);  primals_286 = add_237 = copy__142 = None
        copy__143: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_287, add_238);  primals_287 = add_238 = copy__143 = None
        copy__144: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_291, add_240);  primals_291 = add_240 = copy__144 = None
        copy__145: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_292, add_242);  primals_292 = add_242 = copy__145 = None
        copy__146: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_293, add_243);  primals_293 = add_243 = copy__146 = None
        copy__147: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_297, add_245);  primals_297 = add_245 = copy__147 = None
        copy__148: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_298, add_247);  primals_298 = add_247 = copy__148 = None
        copy__149: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_299, add_248);  primals_299 = add_248 = copy__149 = None
        copy__150: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_303, add_250);  primals_303 = add_250 = copy__150 = None
        copy__151: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_304, add_252);  primals_304 = add_252 = copy__151 = None
        copy__152: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_305, add_253);  primals_305 = add_253 = copy__152 = None
        copy__153: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_309, add_255);  primals_309 = add_255 = copy__153 = None
        copy__154: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_310, add_257);  primals_310 = add_257 = copy__154 = None
        copy__155: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_311, add_258);  primals_311 = add_258 = copy__155 = None
        copy__156: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_315, add_260);  primals_315 = add_260 = copy__156 = None
        copy__157: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_316, add_262);  primals_316 = add_262 = copy__157 = None
        copy__158: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_317, add_263);  primals_317 = add_263 = copy__158 = None
        copy__159: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_321, add_265);  primals_321 = add_265 = copy__159 = None
        copy__160: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_322, add_267);  primals_322 = add_267 = copy__160 = None
        copy__161: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_323, add_268);  primals_323 = add_268 = copy__161 = None
        copy__162: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_327, add_270);  primals_327 = add_270 = copy__162 = None
        copy__163: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_328, add_272);  primals_328 = add_272 = copy__163 = None
        copy__164: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_329, add_273);  primals_329 = add_273 = copy__164 = None
        copy__165: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_333, add_275);  primals_333 = add_275 = copy__165 = None
        copy__166: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_334, add_277);  primals_334 = add_277 = copy__166 = None
        copy__167: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_335, add_278);  primals_335 = add_278 = copy__167 = None
        copy__168: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_339, add_280);  primals_339 = add_280 = copy__168 = None
        copy__169: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_340, add_282);  primals_340 = add_282 = copy__169 = None
        copy__170: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_341, add_283);  primals_341 = add_283 = copy__170 = None
        copy__171: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_345, add_285);  primals_345 = add_285 = copy__171 = None
        copy__172: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_346, add_287);  primals_346 = add_287 = copy__172 = None
        copy__173: "f32[160][1]cuda:0" = torch.ops.aten.copy_.default(primals_347, add_288);  primals_347 = add_288 = copy__173 = None
        copy__174: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_351, add_290);  primals_351 = add_290 = copy__174 = None
        copy__175: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_352, add_292);  primals_352 = add_292 = copy__175 = None
        copy__176: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_353, add_293);  primals_353 = add_293 = copy__176 = None
        copy__177: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_357, add_295);  primals_357 = add_295 = copy__177 = None
        copy__178: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_358, add_297);  primals_358 = add_297 = copy__178 = None
        copy__179: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_359, add_298);  primals_359 = add_298 = copy__179 = None
        copy__180: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_363, add_300);  primals_363 = add_300 = copy__180 = None
        copy__181: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_364, add_302);  primals_364 = add_302 = copy__181 = None
        copy__182: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_365, add_303);  primals_365 = add_303 = copy__182 = None
        copy__183: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_369, add_305);  primals_369 = add_305 = copy__183 = None
        copy__184: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_370, add_307);  primals_370 = add_307 = copy__184 = None
        copy__185: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_371, add_308);  primals_371 = add_308 = copy__185 = None
        copy__186: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_375, add_310);  primals_375 = add_310 = copy__186 = None
        copy__187: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_376, add_312);  primals_376 = add_312 = copy__187 = None
        copy__188: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_377, add_313);  primals_377 = add_313 = copy__188 = None
        copy__189: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_381, add_315);  primals_381 = add_315 = copy__189 = None
        copy__190: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_382, add_317);  primals_382 = add_317 = copy__190 = None
        copy__191: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_383, add_318);  primals_383 = add_318 = copy__191 = None
        copy__192: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_387, add_320);  primals_387 = add_320 = copy__192 = None
        copy__193: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_388, add_322);  primals_388 = add_322 = copy__193 = None
        copy__194: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_389, add_323);  primals_389 = add_323 = copy__194 = None
        copy__195: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_393, add_325);  primals_393 = add_325 = copy__195 = None
        copy__196: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_394, add_327);  primals_394 = add_327 = copy__196 = None
        copy__197: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_395, add_328);  primals_395 = add_328 = copy__197 = None
        copy__198: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_399, add_330);  primals_399 = add_330 = copy__198 = None
        copy__199: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_400, add_332);  primals_400 = add_332 = copy__199 = None
        copy__200: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_401, add_333);  primals_401 = add_333 = copy__200 = None
        copy__201: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_405, add_335);  primals_405 = add_335 = copy__201 = None
        copy__202: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_406, add_337);  primals_406 = add_337 = copy__202 = None
        copy__203: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_407, add_338);  primals_407 = add_338 = copy__203 = None
        copy__204: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_411, add_340);  primals_411 = add_340 = copy__204 = None
        copy__205: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_412, add_342);  primals_412 = add_342 = copy__205 = None
        copy__206: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_413, add_343);  primals_413 = add_343 = copy__206 = None
        copy__207: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_417, add_345);  primals_417 = add_345 = copy__207 = None
        copy__208: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_418, add_347);  primals_418 = add_347 = copy__208 = None
        copy__209: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_419, add_348);  primals_419 = add_348 = copy__209 = None
        copy__210: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_423, add_350);  primals_423 = add_350 = copy__210 = None
        copy__211: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_424, add_352);  primals_424 = add_352 = copy__211 = None
        copy__212: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_425, add_353);  primals_425 = add_353 = copy__212 = None
        copy__213: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_429, add_355);  primals_429 = add_355 = copy__213 = None
        copy__214: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_430, add_357);  primals_430 = add_357 = copy__214 = None
        copy__215: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_431, add_358);  primals_431 = add_358 = copy__215 = None
        copy__216: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_435, add_360);  primals_435 = add_360 = copy__216 = None
        copy__217: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_436, add_362);  primals_436 = add_362 = copy__217 = None
        copy__218: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_437, add_363);  primals_437 = add_363 = copy__218 = None
        copy__219: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_441, add_365);  primals_441 = add_365 = copy__219 = None
        copy__220: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_442, add_367);  primals_442 = add_367 = copy__220 = None
        copy__221: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_443, add_368);  primals_443 = add_368 = copy__221 = None
        copy__222: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_447, add_370);  primals_447 = add_370 = copy__222 = None
        copy__223: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_448, add_372);  primals_448 = add_372 = copy__223 = None
        copy__224: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_449, add_373);  primals_449 = add_373 = copy__224 = None
        copy__225: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_453, add_375);  primals_453 = add_375 = copy__225 = None
        copy__226: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_454, add_377);  primals_454 = add_377 = copy__226 = None
        copy__227: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_455, add_378);  primals_455 = add_378 = copy__227 = None
        copy__228: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_459, add_380);  primals_459 = add_380 = copy__228 = None
        copy__229: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_460, add_382);  primals_460 = add_382 = copy__229 = None
        copy__230: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_461, add_383);  primals_461 = add_383 = copy__230 = None
        copy__231: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_465, add_385);  primals_465 = add_385 = copy__231 = None
        copy__232: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_466, add_387);  primals_466 = add_387 = copy__232 = None
        copy__233: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_467, add_388);  primals_467 = add_388 = copy__233 = None
        copy__234: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_471, add_390);  primals_471 = add_390 = copy__234 = None
        copy__235: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_472, add_392);  primals_472 = add_392 = copy__235 = None
        copy__236: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_473, add_393);  primals_473 = add_393 = copy__236 = None
        copy__237: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_477, add_395);  primals_477 = add_395 = copy__237 = None
        copy__238: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_478, add_397);  primals_478 = add_397 = copy__238 = None
        copy__239: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_479, add_398);  primals_479 = add_398 = copy__239 = None
        copy__240: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_483, add_400);  primals_483 = add_400 = copy__240 = None
        copy__241: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_484, add_402);  primals_484 = add_402 = copy__241 = None
        copy__242: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_485, add_403);  primals_485 = add_403 = copy__242 = None
        copy__243: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_489, add_405);  primals_489 = add_405 = copy__243 = None
        copy__244: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_490, add_407);  primals_490 = add_407 = copy__244 = None
        copy__245: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_491, add_408);  primals_491 = add_408 = copy__245 = None
        copy__246: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_495, add_410);  primals_495 = add_410 = copy__246 = None
        copy__247: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_496, add_412);  primals_496 = add_412 = copy__247 = None
        copy__248: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_497, add_413);  primals_497 = add_413 = copy__248 = None
        copy__249: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_501, add_415);  primals_501 = add_415 = copy__249 = None
        copy__250: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_502, add_417);  primals_502 = add_417 = copy__250 = None
        copy__251: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_503, add_418);  primals_503 = add_418 = copy__251 = None
        copy__252: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_507, add_420);  primals_507 = add_420 = copy__252 = None
        copy__253: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_508, add_422);  primals_508 = add_422 = copy__253 = None
        copy__254: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_509, add_423);  primals_509 = add_423 = copy__254 = None
        copy__255: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_513, add_425);  primals_513 = add_425 = copy__255 = None
        copy__256: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_514, add_427);  primals_514 = add_427 = copy__256 = None
        copy__257: "f32[320][1]cuda:0" = torch.ops.aten.copy_.default(primals_515, add_428);  primals_515 = add_428 = copy__257 = None
        copy__258: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_519, add_430);  primals_519 = add_430 = copy__258 = None
        copy__259: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_520, add_432);  primals_520 = add_432 = copy__259 = None
        copy__260: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_521, add_433);  primals_521 = add_433 = copy__260 = None
        copy__261: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_525, add_435);  primals_525 = add_435 = copy__261 = None
        copy__262: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_526, add_437);  primals_526 = add_437 = copy__262 = None
        copy__263: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_527, add_438);  primals_527 = add_438 = copy__263 = None
        copy__264: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_531, add_440);  primals_531 = add_440 = copy__264 = None
        copy__265: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_532, add_442);  primals_532 = add_442 = copy__265 = None
        copy__266: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_533, add_443);  primals_533 = add_443 = copy__266 = None
        copy__267: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_537, add_445);  primals_537 = add_445 = copy__267 = None
        copy__268: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_538, add_447);  primals_538 = add_447 = copy__268 = None
        copy__269: "f32[448][1]cuda:0" = torch.ops.aten.copy_.default(primals_539, add_448);  primals_539 = add_448 = copy__269 = None
        copy__270: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_543, add_450);  primals_543 = add_450 = copy__270 = None
        copy__271: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_544, add_452);  primals_544 = add_452 = copy__271 = None
        copy__272: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_545, add_453);  primals_545 = add_453 = copy__272 = None
        copy__273: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_549, add_455);  primals_549 = add_455 = copy__273 = None
        copy__274: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_550, add_457);  primals_550 = add_457 = copy__274 = None
        copy__275: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_551, add_458);  primals_551 = add_458 = copy__275 = None
        copy__276: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_555, add_460);  primals_555 = add_460 = copy__276 = None
        copy__277: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_556, add_462);  primals_556 = add_462 = copy__277 = None
        copy__278: "f32[384][1]cuda:0" = torch.ops.aten.copy_.default(primals_557, add_463);  primals_557 = add_463 = copy__278 = None
        copy__279: "i64[][]cuda:0" = torch.ops.aten.copy_.default(primals_561, add_465);  primals_561 = add_465 = copy__279 = None
        copy__280: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_562, add_467);  primals_562 = add_467 = copy__280 = None
        copy__281: "f32[192][1]cuda:0" = torch.ops.aten.copy_.default(primals_563, add_468);  primals_563 = add_468 = copy__281 = None
        return (addmm, primals_6, primals_12, primals_18, primals_19, primals_24, primals_30, primals_31, primals_36, primals_37, primals_42, primals_48, primals_49, primals_54, primals_60, primals_66, primals_67, primals_72, primals_73, primals_78, primals_79, primals_84, primals_90, primals_91, primals_96, primals_102, primals_108, primals_109, primals_114, primals_115, primals_120, primals_121, primals_126, primals_132, primals_133, primals_138, primals_144, primals_150, primals_151, primals_156, primals_157, primals_162, primals_163, primals_168, primals_174, primals_180, primals_181, primals_186, primals_187, primals_192, primals_198, primals_204, primals_205, primals_210, primals_216, primals_222, primals_228, primals_234, primals_235, primals_240, primals_241, primals_246, primals_247, primals_252, primals_258, primals_264, primals_265, primals_270, primals_276, primals_282, primals_288, primals_294, primals_295, primals_300, primals_301, primals_306, primals_307, primals_312, primals_318, primals_324, primals_325, primals_330, primals_336, primals_342, primals_348, primals_354, primals_355, primals_360, primals_361, primals_366, primals_367, primals_372, primals_378, primals_384, primals_385, primals_390, primals_396, primals_402, primals_408, primals_414, primals_415, primals_420, primals_421, primals_426, primals_432, primals_433, primals_438, primals_444, primals_450, primals_456, primals_457, primals_462, primals_463, primals_468, primals_474, primals_475, primals_480, primals_481, primals_486, primals_492, primals_498, primals_499, primals_504, primals_505, primals_510, primals_511, primals_516, primals_517, primals_522, primals_528, primals_529, primals_534, primals_535, primals_540, primals_546, primals_552, primals_553, primals_558, primals_559, primals_564, primals_565, convert_element_type, convert_element_type_1, convolution, squeeze_1, relu, convert_element_type_4, convolution_1, squeeze_4, relu_1, convert_element_type_7, convolution_2, getitem_5, rsqrt_2, getitem_6, getitem_7, convert_element_type_10, convolution_3, squeeze_10, relu_3, convert_element_type_13, convolution_4, getitem_11, rsqrt_4, getitem_12, getitem_13, convert_element_type_16, convolution_5, getitem_15, rsqrt_5, convert_element_type_19, convolution_6, squeeze_19, relu_6, convert_element_type_22, convolution_7, getitem_19, rsqrt_7, convert_element_type_25, convolution_8, squeeze_25, relu_8, convert_element_type_28, convolution_9, squeeze_28, relu_9, convert_element_type_31, convolution_10, getitem_25, rsqrt_10, avg_pool2d, convert_element_type_34, convolution_11, getitem_27, rsqrt_11, cat, convert_element_type_37, convolution_12, getitem_29, rsqrt_12, convert_element_type_40, convolution_13, squeeze_40, relu_13, convert_element_type_43, convolution_14, getitem_33, rsqrt_14, convert_element_type_46, convolution_15, squeeze_46, relu_15, convert_element_type_49, convolution_16, squeeze_49, relu_16, convert_element_type_52, convolution_17, getitem_39, rsqrt_17, avg_pool2d_1, convert_element_type_55, convolution_18, getitem_41, rsqrt_18, cat_1, convert_element_type_58, convolution_19, getitem_43, rsqrt_19, convert_element_type_61, convolution_20, squeeze_61, relu_20, convert_element_type_64, convolution_21, getitem_47, rsqrt_21, convert_element_type_67, convolution_22, squeeze_67, relu_22, convert_element_type_70, convolution_23, squeeze_70, relu_23, convert_element_type_73, convolution_24, getitem_53, rsqrt_24, avg_pool2d_2, convert_element_type_76, convolution_25, getitem_55, rsqrt_25, cat_2, convert_element_type_79, convolution_26, getitem_57, rsqrt_26, convert_element_type_82, convolution_27, squeeze_82, relu_27, convert_element_type_85, convolution_28, squeeze_85, relu_28, convert_element_type_88, convolution_29, getitem_63, rsqrt_29, getitem_65, cat_3, convert_element_type_91, convolution_30, getitem_67, rsqrt_30, convert_element_type_94, convolution_31, squeeze_94, relu_31, convert_element_type_97, convolution_32, squeeze_97, relu_32, convert_element_type_100, convolution_33, getitem_73, rsqrt_33, convert_element_type_103, convolution_34, squeeze_103, relu_34, convert_element_type_106, convolution_35, squeeze_106, relu_35, convert_element_type_109, convolution_36, squeeze_109, relu_36, convert_element_type_112, convolution_37, squeeze_112, relu_37, convert_element_type_115, convolution_38, getitem_83, rsqrt_38, avg_pool2d_3, convert_element_type_118, convolution_39, getitem_85, rsqrt_39, cat_4, convert_element_type_121, convolution_40, getitem_87, rsqrt_40, convert_element_type_124, convolution_41, squeeze_124, relu_41, convert_element_type_127, convolution_42, squeeze_127, relu_42, convert_element_type_130, convolution_43, getitem_93, rsqrt_43, convert_element_type_133, convolution_44, squeeze_133, relu_44, convert_element_type_136, convolution_45, squeeze_136, relu_45, convert_element_type_139, convolution_46, squeeze_139, relu_46, convert_element_type_142, convolution_47, squeeze_142, relu_47, convert_element_type_145, convolution_48, getitem_103, rsqrt_48, avg_pool2d_4, convert_element_type_148, convolution_49, getitem_105, rsqrt_49, cat_5, convert_element_type_151, convolution_50, getitem_107, rsqrt_50, convert_element_type_154, convolution_51, squeeze_154, relu_51, convert_element_type_157, convolution_52, squeeze_157, relu_52, convert_element_type_160, convolution_53, getitem_113, rsqrt_53, convert_element_type_163, convolution_54, squeeze_163, relu_54, convert_element_type_166, convolution_55, squeeze_166, relu_55, convert_element_type_169, convolution_56, squeeze_169, relu_56, convert_element_type_172, convolution_57, squeeze_172, relu_57, convert_element_type_175, convolution_58, getitem_123, rsqrt_58, avg_pool2d_5, convert_element_type_178, convolution_59, getitem_125, rsqrt_59, cat_6, convert_element_type_181, convolution_60, getitem_127, rsqrt_60, convert_element_type_184, convolution_61, squeeze_184, relu_61, convert_element_type_187, convolution_62, squeeze_187, relu_62, convert_element_type_190, convolution_63, getitem_133, rsqrt_63, convert_element_type_193, convolution_64, squeeze_193, relu_64, convert_element_type_196, convolution_65, squeeze_196, relu_65, convert_element_type_199, convolution_66, squeeze_199, relu_66, convert_element_type_202, convolution_67, squeeze_202, relu_67, convert_element_type_205, convolution_68, getitem_143, rsqrt_68, avg_pool2d_6, convert_element_type_208, convolution_69, getitem_145, rsqrt_69, cat_7, convert_element_type_211, convolution_70, squeeze_211, relu_70, convert_element_type_214, convolution_71, getitem_149, rsqrt_71, convert_element_type_217, convolution_72, squeeze_217, relu_72, convert_element_type_220, convolution_73, squeeze_220, relu_73, convert_element_type_223, convolution_74, squeeze_223, relu_74, convert_element_type_226, convolution_75, getitem_157, rsqrt_75, getitem_159, cat_8, convert_element_type_229, convolution_76, getitem_161, rsqrt_76, convert_element_type_232, convolution_77, squeeze_232, relu_77, convert_element_type_235, convolution_78, getitem_165, rsqrt_78, convert_element_type_238, convolution_79, getitem_167, rsqrt_79, convert_element_type_241, convolution_80, squeeze_241, relu_80, convert_element_type_244, convolution_81, squeeze_244, relu_81, convert_element_type_247, convolution_82, getitem_173, rsqrt_82, convert_element_type_250, convolution_83, getitem_175, rsqrt_83, avg_pool2d_7, convert_element_type_253, convolution_84, getitem_177, rsqrt_84, cat_11, convert_element_type_256, convolution_85, getitem_179, rsqrt_85, convert_element_type_259, convolution_86, squeeze_259, relu_86, convert_element_type_262, convolution_87, getitem_183, rsqrt_87, convert_element_type_265, convolution_88, getitem_185, rsqrt_88, convert_element_type_268, convolution_89, squeeze_268, relu_89, convert_element_type_271, convolution_90, squeeze_271, relu_90, convert_element_type_274, convolution_91, getitem_191, rsqrt_91, convert_element_type_277, convolution_92, getitem_193, rsqrt_92, avg_pool2d_8, convert_element_type_280, convolution_93, getitem_195, rsqrt_93, view, permute_1, unsqueeze_414, unsqueeze_426, unsqueeze_462, unsqueeze_522, unsqueeze_534, unsqueeze_570, unsqueeze_606, unsqueeze_618, unsqueeze_630, unsqueeze_654, unsqueeze_690, unsqueeze_702, unsqueeze_714, unsqueeze_726, unsqueeze_750, unsqueeze_762, unsqueeze_810, unsqueeze_822, unsqueeze_834, unsqueeze_846, unsqueeze_870, unsqueeze_882, unsqueeze_930, unsqueeze_942, unsqueeze_954, unsqueeze_966, unsqueeze_990, unsqueeze_1002, unsqueeze_1050, unsqueeze_1062, unsqueeze_1074, unsqueeze_1086, unsqueeze_1110, unsqueeze_1122, unsqueeze_1158, unsqueeze_1170, unsqueeze_1218, unsqueeze_1230, unsqueeze_1254, unsqueeze_1302, unsqueeze_1314, unsqueeze_1338, unsqueeze_1386, unsqueeze_1398, unsqueeze_1422, unsqueeze_1458, unsqueeze_1482, unsqueeze_1494)
